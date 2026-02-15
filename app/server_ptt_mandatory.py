import os
import json
import time
import base64
import audioop
import re
import threading
import queue
from typing import Optional, Tuple, List, Dict
import noisereduce as nr

import numpy as np
import webrtcvad
import requests
from flask import Flask, request, Response
from flask_sock import Sock
from piper.voice import PiperVoice
from faster_whisper import WhisperModel

# ===========================================
# Flask + WebSocket setup
# ===========================================
app = Flask(__name__)
sock = Sock(app)

# ===========================================
# Env config
# ===========================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
#OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
#OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Reply language policy:
# - "auto" => reply in detected language (ne/hi/en) and TTS in same language
# - "ne"/"hi"/"en" => force reply language (TTS will also follow forced lang)
REPLY_LANG = os.getenv("REPLY_LANG", "auto").strip().lower()

# Whisper config
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int16")

# Piper multi-voice config:
# Put all voices under one directory, each voice has its own subdir named by model id:
#   /root/.local/share/piper/voices/ne_NP-chitwan-medium/ne_NP-chitwan-medium.onnx(.json)
#   /root/.local/share/piper/voices/hi_IN-rohan-medium/hi_IN-rohan-medium.onnx(.json)
#   /root/.local/share/piper/voices/en_US-amy-low/en_US-amy-low.onnx(.json)
PIPER_VOICES_DIR = os.getenv("PIPER_VOICES_DIR", "/root/.local/share/piper/voices").strip()
PIPER_NE_MODEL = os.getenv("PIPER_NE_MODEL", "ne_NP-chitwan-medium").strip()
PIPER_HI_MODEL = os.getenv("PIPER_HI_MODEL", "hi_IN-rohan-medium").strip()
PIPER_EN_MODEL = os.getenv("PIPER_EN_MODEL", "en_US-amy-low").strip()

# Backward compatibility (if you still export single-voice vars)
PIPER_PATH = os.getenv("PIPER_PATH", "").strip()
PIPER_MODEL = os.getenv("PIPER_MODEL", "").strip()

# Twilio media stream audio details:
# - inbound audio: 8kHz mu-law payload, 20ms frames
TWILIO_SR = 8000
FRAME_MS = 20
SAMPLES_PER_FRAME_8K = int(TWILIO_SR * FRAME_MS / 1000)  # 160
PCM16_BYTES_PER_FRAME = SAMPLES_PER_FRAME_8K * 2         # 320 bytes (16-bit)

# ===========================================
# Initialize Whisper
# ===========================================
print(f"[INIT] Loading Whisper model: {WHISPER_MODEL_NAME} ({WHISPER_DEVICE}, {WHISPER_COMPUTE})")
whisper = WhisperModel(
    WHISPER_MODEL_NAME,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE,
)
print("[INIT] ✅ Whisper loaded.")

# ===========================================
# Initialize Piper voices
# ===========================================
def _piper_files_exist(onnx_path: str, json_path: str) -> bool:
    return os.path.exists(onnx_path) and os.path.exists(json_path)

def _load_piper_from_dir(model_dir: str, model_name: str) -> PiperVoice:
    onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
    json_path = os.path.join(model_dir, f"{model_name}.onnx.json")
    if not _piper_files_exist(onnx_path, json_path):
        raise RuntimeError(
            f"❌ Piper files missing for model '{model_name}'.\n"
            f"Expected:\n  {onnx_path}\n  {json_path}\n"
        )
    return PiperVoice.load(onnx_path, config_path=json_path)

def _load_piper_voice(model_name: str) -> PiperVoice:
    # Prefer multi-voice layout: {PIPER_VOICES_DIR}/{model_name}/{model_name}.onnx(.json)
    model_dir = os.path.join(PIPER_VOICES_DIR, model_name)
    if os.path.isdir(model_dir):
        return _load_piper_from_dir(model_dir, model_name)

    # Back-compat: single voice layout using PIPER_PATH + PIPER_MODEL
    if PIPER_PATH and PIPER_MODEL and model_name == PIPER_MODEL:
        return _load_piper_from_dir(PIPER_PATH, PIPER_MODEL)

    # Last resort: try PIPER_VOICES_DIR directly (flat) if someone put files there
    onnx_flat = os.path.join(PIPER_VOICES_DIR, f"{model_name}.onnx")
    json_flat = os.path.join(PIPER_VOICES_DIR, f"{model_name}.onnx.json")
    if _piper_files_exist(onnx_flat, json_flat):
        return PiperVoice.load(onnx_flat, config_path=json_flat)

    raise RuntimeError(
        f"❌ Could not locate Piper model '{model_name}'.\n"
        f"Tried:\n  {model_dir}/...\n  {PIPER_PATH} (if configured)\n  flat: {onnx_flat}\n"
    )

print("[INIT] Loading Piper voices (ne/hi/en)...")
PIPER_VOICES: Dict[str, PiperVoice] = {
    "ne": _load_piper_voice(PIPER_NE_MODEL),
    "hi": _load_piper_voice(PIPER_HI_MODEL),
    "en": _load_piper_voice(PIPER_EN_MODEL),
}
print("[INIT] ✅ Piper voices loaded.")

# ===========================================
# Helper: build Twilio-compliant outbound audio
# ===========================================
def speak_text_to_mulaw_8k(text: str, tts_voice: PiperVoice) -> bytes:
    """
    Piper -> PCM16 (native sr) -> resample to 8k -> mu-law bytes for Twilio.
    Robust against piper returning a generator of AudioChunk.
    """
    text = (text or "").strip()
    if not text:
        return b""

    try:
        result = tts_voice.synthesize(text)
        pcm_bytes = bytearray()
        sr = 22050

        # Piper commonly returns a generator of AudioChunk
        if hasattr(result, "__iter__") and not isinstance(result, (bytes, bytearray, np.ndarray, str)):
            for chunk in result:
                if hasattr(chunk, "audio_int16_bytes"):
                    pcm_bytes.extend(chunk.audio_int16_bytes)
                    sr = int(getattr(chunk, "sample_rate", sr))
                elif hasattr(chunk, "audio_int16_array"):
                    arr = chunk.audio_int16_array
                    pcm_bytes.extend(np.asarray(arr, dtype=np.int16).tobytes())
                    sr = int(getattr(chunk, "sample_rate", sr))
                elif hasattr(chunk, "audio_float_array"):
                    arr = np.asarray(chunk.audio_float_array, dtype=np.float32)
                    arr = np.clip(arr, -1.0, 1.0)
                    pcm16 = (arr * 32767.0).astype(np.int16)
                    pcm_bytes.extend(pcm16.tobytes())
                    sr = int(getattr(chunk, "sample_rate", sr))
                else:
                    arr = np.asarray(chunk, dtype=np.float32)
                    arr = np.clip(arr, -1.0, 1.0)
                    pcm16 = (arr * 32767.0).astype(np.int16)
                    pcm_bytes.extend(pcm16.tobytes())
        else:
            arr = np.asarray(result, dtype=np.float32)
            arr = np.clip(arr, -1.0, 1.0)
            pcm16 = (arr * 32767.0).astype(np.int16)
            pcm_bytes.extend(pcm16.tobytes())
            sr = int(getattr(tts_voice, "sample_rate", sr))

        if not pcm_bytes:
            return b""

        # resample to 8k for Twilio
        pcm_8k = audioop.ratecv(bytes(pcm_bytes), 2, 1, sr, TWILIO_SR, None)[0]
        mulaw = audioop.lin2ulaw(pcm_8k, 2)
        return mulaw

    except Exception as e:
        print(f"[ERROR] Piper synthesis failed: {e}")
        return b""

def send_audio_to_twilio(
    ws,
    stream_sid: str,
    mulaw_bytes: bytes,
    send_lock: Optional[threading.Lock] = None,
    cancel_event: Optional[threading.Event] = None,
):
    """
    Twilio expects JSON messages with:
      event=media, streamSid, media.payload (base64)
    and 20ms frames at 8kHz mu-law => 160 bytes per frame.

    If cancel_event is set, stop sending immediately (barge-in / interruption).
    """
    if not mulaw_bytes:
        return

    frame_size = 160
    for i in range(0, len(mulaw_bytes), frame_size):
        if cancel_event is not None and cancel_event.is_set():
            return

        frame = mulaw_bytes[i:i + frame_size]
        payload = base64.b64encode(frame).decode("ascii")
        message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }
        payload_json = json.dumps(message)
        if send_lock:
            with send_lock:
                ws.send(payload_json)
        else:
            ws.send(payload_json)

        time.sleep(0.02)  # pacing: 20ms

    if cancel_event is not None and cancel_event.is_set():
        return

    mark_json = json.dumps({
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": "done"},
    })
    if send_lock:
        with send_lock:
            ws.send(mark_json)
    else:
        ws.send(mark_json)


def send_clear_to_twilio(ws, stream_sid: str, send_lock: Optional[threading.Lock] = None):
    """Clear any buffered outbound audio in Twilio for this stream."""
    msg = {"event": "clear", "streamSid": stream_sid}
    payload_json = json.dumps(msg)
    if send_lock:
        with send_lock:
            ws.send(payload_json)
    else:
        ws.send(payload_json)


def beep_mulaw_8k(freq: int = 880, duration_ms: int = 200, sr: int = 8000, amp: float = 0.25):
    """Generate a short beep (mulaw/8k) to acknowledge push-to-talk start."""
    n = int(sr * (duration_ms / 1000.0))
    if n <= 0:
        return b""
    t = (np.arange(n, dtype=np.float32) / float(sr))
    x = amp * np.sin(2.0 * np.pi * float(freq) * t)
    pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    return audioop.lin2ulaw(pcm16, 2)

# ===========================================
# Whisper STT helpers
# ===========================================
def pcm8k_to_float16k(pcm16_8k: bytes) -> np.ndarray:
    """
    Convert 8k PCM16 -> 16k float32 in [-1, 1] for Whisper.
    """
    pcm16_16k = audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)[0]
    audio_i16 = np.frombuffer(pcm16_16k, dtype=np.int16)
    audio_f32 = (audio_i16.astype(np.float32) / 32768.0)
    return audio_f32

NEPALI_HINTS = [
    "तपाईं", "तिमी", "मेरो", "मलाई", "हामी", "हुन", "छ", "छु", "छैन", "गर", "गर्न", "के", "किन",
    "कसरी", "कहाँ", "भएको", "भए", "भन्छ", "हो", "होइन", "आज", "भोलि", "भएपछि"
]

def _devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = 0
    dev = 0
    for c in text:
        if c.isspace():
            continue
        total += 1
        if 0x0900 <= ord(c) <= 0x097F:
            dev += 1
    return dev / max(1, total)

def _looks_nepali(text: str) -> bool:
    # strong hint: danda + nepali common words
    if any(h in text for h in NEPALI_HINTS):
        return True
    # "छ/छु/छैन" occur a lot in Nepali; Hindi has "है/हूँ/नहीं" more
    if ("छ" in text) and ("है" not in text) and ("हूँ" not in text):
        return True
    return False

def correct_language(text: str, detected: Optional[str]) -> str:
    """
    Post-detection correction heuristic:
    - If Whisper says 'hi' but text looks Nepali => flip to 'ne'
    - If text is mostly Devanagari and has Nepali hints => 'ne'
    - Else keep detected if it is ne/hi/en, otherwise default to 'en'
    """
    d = (detected or "").lower().strip()
    if d not in ("ne", "hi", "en"):
        d = ""

    dev_ratio = _devanagari_ratio(text)
    if dev_ratio >= 0.60 and _looks_nepali(text):
        return "ne"

    if d == "hi" and dev_ratio >= 0.60 and _looks_nepali(text):
        return "ne"

    if d in ("ne", "hi", "en"):
        return d

    # If Devanagari but no strong Nepali hints, keep as Hindi (common) else English
    if dev_ratio >= 0.60:
        return "hi"
    return "en"

# def whisper_transcribe(pcm16_8k: bytes) -> Tuple[str, Optional[str]]:
#     """
#     Runs Whisper on an utterance (PCM16 @8k) and returns (text, detected_lang),
#     with post-detection correction for Nepali vs Hindi.
#     """
#     audio = pcm8k_to_float16k(pcm16_8k)

#     segments, info = whisper.transcribe(
#         audio,
#         beam_size=5,
#         vad_filter=False,
#         language=None,   # auto-detect
#         task="transcribe"
#     )

#     text_parts: List[str] = []
#     for seg in segments:
#         if seg.text:
#             text_parts.append(seg.text.strip())

#     text = " ".join([t for t in text_parts if t]).strip()
#     lang = getattr(info, "language", None)
#     lang = correct_language(text, lang)
#     return text, lang

def whisper_transcribe(pcm16_8k: bytes, noise_pcm16_8k: Optional[bytes] = None) -> Tuple[str, Optional[str]]:
    audio = pcm8k_to_float16k(pcm16_8k)

    noise = None
    if noise_pcm16_8k:
        noise = pcm8k_to_float16k(noise_pcm16_8k)

    audio = enhance_audio_for_whisper(audio, noise_f32_16k=noise)

    segments, info = whisper.transcribe(
        audio,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,  # faster-whisper built-in VAD {index=3}
        vad_parameters=dict(min_silence_duration_ms=250),
        language=None,
        task="transcribe",
        condition_on_previous_text=False,
    )

    text_parts: List[str] = []
    for seg in segments:
        if seg.text:
            text_parts.append(seg.text.strip())

    text = " ".join([t for t in text_parts if t]).strip()
    lang = getattr(info, "language", None)
    lang = correct_language(text, lang)
    return text, lang

# ===========================================
# LLM streaming (Ollama)
# ===========================================
def stream_llm_reply(user_text: str, user_lang: Optional[str]):
    # Decide reply language:
    # - If REPLY_LANG=auto => follow corrected detected language
    # - Else force ne/hi/en
    ul = (user_lang or "").strip().lower()
    if REPLY_LANG == "auto":
        reply_lang = ul if ul in ("ne", "hi", "en") else "en"
    else:
        reply_lang = REPLY_LANG if REPLY_LANG in ("ne", "hi", "en") else "en"

    if reply_lang == "ne":
        instruction = "Reply in Nepali (नेपाली) using Devanagari script."
    elif reply_lang == "hi":
        instruction = "Reply in Hindi (हिन्दी) using Devanagari script."
    else:
        instruction = "Reply in English."

    prompt = (
        f"{instruction}\n"
        f"User said: {user_text}\n"
        f"Assistant:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }

    with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                chunk = data.get("response", "")
                if chunk:
                    yield chunk, reply_lang
            except Exception:
                continue

# ===========================================
# Simple VAD segmenter (webrtcvad)
# ===========================================
# class VadSegmenter:
#     def __init__(self, mode: int = 2, end_silence_ms: int = 600, max_utt_ms: int = 12000):
#         self.vad = webrtcvad.Vad(mode)
#         self.end_silence_ms = end_silence_ms
#         self.max_utt_ms = max_utt_ms
#         self.reset()

#     def reset(self):
#         self.in_speech = False
#         self.frames: List[bytes] = []
#         self.silence_ms = 0
#         self.utt_ms = 0

#     def push_frame(self, pcm16_frame_8k: bytes) -> Optional[bytes]:
#         if len(pcm16_frame_8k) != PCM16_BYTES_PER_FRAME:
#             return None

#         is_speech = self.vad.is_speech(pcm16_frame_8k, TWILIO_SR)

#         if is_speech:
#             if not self.in_speech:
#                 self.in_speech = True
#                 self.frames = []
#                 self.silence_ms = 0
#                 self.utt_ms = 0

#             self.frames.append(pcm16_frame_8k)
#             self.utt_ms += FRAME_MS
#             self.silence_ms = 0

#             if self.utt_ms >= self.max_utt_ms:
#                 utt = b"".join(self.frames)
#                 self.reset()
#                 return utt

#         else:
#             if self.in_speech:
#                 self.silence_ms += FRAME_MS
#                 self.frames.append(pcm16_frame_8k)
#                 self.utt_ms += FRAME_MS

#                 if self.silence_ms >= self.end_silence_ms:
#                     utt = b"".join(self.frames)
#                     self.reset()
#                     return utt

#         return None

class VadSegmenter:
    def __init__(
        self,
        mode: int = 2,
        end_silence_ms: int = 600,
        max_utt_ms: int = 12000,
        preroll_ms: int = 300,
        noise_ms: int = 1000,
    ):
        self.vad = webrtcvad.Vad(mode)
        self.end_silence_ms = end_silence_ms
        self.max_utt_ms = max_utt_ms

        self.preroll_frames = int(preroll_ms / FRAME_MS)          # e.g., 300ms -> 15 frames
        self.noise_frames_max = int(noise_ms / FRAME_MS)          # e.g., 1000ms -> 50 frames

        self.reset()

    def reset(self):
        self.in_speech = False
        self.frames: List[bytes] = []
        self.silence_ms = 0
        self.utt_ms = 0

        # ring buffers
        self.preroll: List[bytes] = []
        self.noise_buf: List[bytes] = []

    def _push_ring(self, ring: List[bytes], frame: bytes, maxlen: int):
        ring.append(frame)
        if len(ring) > maxlen:
            ring.pop(0)

    def push_frame(self, pcm16_frame_8k: bytes) -> Optional[Tuple[bytes, bytes]]:
        """
        Returns (utterance_pcm16_8k, noise_pcm16_8k) when an utterance ends.
        noise_pcm16_8k is recent non-speech audio you can use as a noise profile.
        """
        if len(pcm16_frame_8k) != PCM16_BYTES_PER_FRAME:
            return None

        is_speech = self.vad.is_speech(pcm16_frame_8k, TWILIO_SR)

        if not self.in_speech:
            # maintain preroll + noise buffers during silence
            self._push_ring(self.preroll, pcm16_frame_8k, self.preroll_frames)
            if not is_speech:
                self._push_ring(self.noise_buf, pcm16_frame_8k, self.noise_frames_max)

        if is_speech:
            if not self.in_speech:
                self.in_speech = True
                self.frames = []
                self.silence_ms = 0
                self.utt_ms = 0

                # prepend preroll to avoid clipping initial phonemes
                self.frames.extend(self.preroll)

            self.frames.append(pcm16_frame_8k)
            self.utt_ms += FRAME_MS
            self.silence_ms = 0

            if self.utt_ms >= self.max_utt_ms:
                utt = b"".join(self.frames)
                noise = b"".join(self.noise_buf)
                self.reset()
                return (utt, noise)

        else:
            if self.in_speech:
                self.silence_ms += FRAME_MS
                self.frames.append(pcm16_frame_8k)
                self.utt_ms += FRAME_MS

                if self.silence_ms >= self.end_silence_ms:
                    utt = b"".join(self.frames)
                    noise = b"".join(self.noise_buf)
                    self.reset()
                    return (utt, noise)

        return None


def enhance_audio_for_whisper(audio_f32_16k: np.ndarray, noise_f32_16k: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Lightweight telephony-oriented enhancement.
    audio_f32_16k: float32 in [-1,1], 16kHz
    """
    if audio_f32_16k.size == 0:
        return audio_f32_16k

    x = audio_f32_16k.astype(np.float32)

    # 1) DC removal
    x = x - np.mean(x)

    # 2) RMS normalize (target ~ -20 dBFS => rms ~ 0.1)
    rms = float(np.sqrt(np.mean(x * x)) + 1e-8)
    target_rms = 0.10
    gain = target_rms / rms
    # limit gain to avoid amplifying pure noise too much
    gain = float(np.clip(gain, 0.25, 8.0))
    x = x * gain

    # 3) Pre-emphasis (helps intelligibility on narrowband speech)
    pre = 0.97
    x = np.append(x[0], x[1:] - pre * x[:-1])

    # 4) Optional noise reduction (CPU-costly; keep easy to disable)
    if noise_f32_16k is not None and noise_f32_16k.size > 0:
        try:
            x = nr.reduce_noise(y=x, sr=16000, y_noise=noise_f32_16k.astype(np.float32), stationary=True)
        except Exception as e:
            print(f"[WARN] noise reduction failed: {e}")

    # 5) Soft clip
    x = np.tanh(1.5 * x)

    return x.astype(np.float32)


# ===========================================
# WebSocket: Twilio Media Stream endpoint
# ===========================================
@sock.route("/ws")
def ws(ws):
    print(">> Stream connected")

    stream_sid: Optional[str] = None

    # Inbound audio buffering (Twilio sends mulaw/8k, we decode to PCM16/8k)
    audio_buf = bytearray()

    # Push-to-talk (DTMF "5") is MANDATORY now.
    PTT_DIGIT = "5"
    ptt_active = False
    ptt_buf = bytearray()

    # Noise profile ring buffer (used for optional noise reduction)
    noise_vad = webrtcvad.Vad(2)
    noise_ring: List[bytes] = []
    NOISE_FRAMES_MAX = int(1000 / FRAME_MS)  # ~1s

    def push_noise(frame: bytes):
        nonlocal noise_ring
        try:
            if not noise_vad.is_speech(frame, TWILIO_SR):
                noise_ring.append(frame)
                if len(noise_ring) > NOISE_FRAMES_MAX:
                    noise_ring.pop(0)
        except Exception:
            pass

    # Threading: keep receiving audio while STT/LLM/TTS runs
    send_lock = threading.Lock()

    # Use a small queue and keep only the latest utterance
    work_q: "queue.Queue[Tuple[bytes, bytes]]" = queue.Queue(maxsize=2)
    stop_evt = threading.Event()
    assistant_busy = threading.Event()  # set while generating/speaking

    # Cancellation token for "barge-in": pressing 5 while bot is talking cancels generation immediately.
    cancel_evt = threading.Event()
    cancel_lock = threading.Lock()

    def cancel_current_generation():
        with cancel_lock:
            cancel_evt.set()

    def reset_cancel_token():
        with cancel_lock:
            cancel_evt.clear()

    def put_latest(item: Tuple[bytes, bytes]):
        """If queue is full, drop the oldest and enqueue the latest."""
        try:
            work_q.put_nowait(item)
        except queue.Full:
            try:
                _ = work_q.get_nowait()
                work_q.task_done()
            except Exception:
                pass
            try:
                work_q.put_nowait(item)
            except queue.Full:
                print("[WARN] work queue full, dropping utterance")

    # LLM->TTS batching (avoid speaking tiny subwords)
    tts_buffer = ""
    MIN_CHARS = 30  # tune 20-40

    def should_flush(buf: str) -> bool:
        if len(buf) >= MIN_CHARS:
            return True
        return bool(re.search(r"[\.!\?,;:]|।", buf))

    def respond_to_utterance(utt_pcm16_8k: bytes, noise_pcm16_8k: bytes):
        nonlocal tts_buffer

        text, lang = whisper_transcribe(utt_pcm16_8k, noise_pcm16_8k)
        text = (text or "").strip()
        if not text:
            return

        print(f"[User] ({lang}) {text}")

        full_reply = ""
        current_tts_lang = lang if lang in ("ne", "hi", "en") else "hi"
        tts_voice = PIPER_VOICES.get(current_tts_lang, PIPER_VOICES["hi"])

        # Start a new generation token
        reset_cancel_token()

        for rchunk, reply_lang in stream_llm_reply(text, lang):
            # If user pressed 5 while we were speaking, abort ASAP.
            if cancel_evt.is_set():
                print("[INFO] Generation cancelled (barge-in).")
                break

            full_reply += rchunk

            if reply_lang in ("ne", "hi", "en"):
                current_tts_lang = reply_lang
                tts_voice = PIPER_VOICES.get(current_tts_lang, PIPER_VOICES["hi"])

            tts_buffer += rchunk
            if should_flush(tts_buffer):
                if cancel_evt.is_set():
                    break
                mulaw = speak_text_to_mulaw_8k(tts_buffer, tts_voice)
                if cancel_evt.is_set():
                    break
                if mulaw and stream_sid:
                    send_audio_to_twilio(ws, stream_sid, mulaw, send_lock=send_lock, cancel_event=cancel_evt)
                tts_buffer = ""

        # Flush remainder (unless cancelled)
        if not cancel_evt.is_set() and tts_buffer.strip():
            mulaw = speak_text_to_mulaw_8k(tts_buffer, tts_voice)
            if mulaw and stream_sid and not cancel_evt.is_set():
                send_audio_to_twilio(ws, stream_sid, mulaw, send_lock=send_lock, cancel_event=cancel_evt)
            tts_buffer = ""
        else:
            tts_buffer = ""

        if full_reply.strip():
            print("[LLM full]", full_reply.strip())

    def worker_loop():
        while not stop_evt.is_set():
            item = work_q.get()
            if item is None:
                break
            utt_pcm16_8k, noise_pcm16_8k = item
            assistant_busy.set()
            try:
                respond_to_utterance(utt_pcm16_8k, noise_pcm16_8k)
            except Exception as e:
                print(f"[ERROR] worker: {e}")
            finally:
                assistant_busy.clear()
                work_q.task_done()

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    try:
        while True:
            msg = ws.receive()
            if not msg:
                break

            data = json.loads(msg)
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                print(f">> Stream started (SID: {stream_sid})")

            elif event == "dtmf":
                # Twilio Media Streams sends DTMF keypress events over the same WS. 
                digit = None
                d = data.get("dtmf") or {}
                digit = d.get("digit") or d.get("digits") or data.get("digit")

                print(f">> DTMF: {digit}")

                if digit == PTT_DIGIT and stream_sid:
                    if not ptt_active:
                        # START recording
                        ptt_active = True
                        ptt_buf = bytearray()

                        # If assistant is talking, cancel immediately + clear Twilio buffer.
                        if assistant_busy.is_set():
                            cancel_current_generation()
                        send_clear_to_twilio(ws, stream_sid, send_lock=send_lock)

                        # Beep acknowledgement
                        b = beep_mulaw_8k()
                        if b:
                            send_audio_to_twilio(ws, stream_sid, b, send_lock=send_lock)

                        print(">> PTT ON (recording)")
                    else:
                        # STOP recording
                        ptt_active = False
                        print(">> PTT OFF (transcribing)")

                        # Minimum length guard (~300ms)
                        if len(ptt_buf) >= int(0.3 * TWILIO_SR) * 2:
                            noise_bytes = b"".join(noise_ring)
                            put_latest((bytes(ptt_buf), noise_bytes))
                        ptt_buf = bytearray()

            elif event == "media":
                if not stream_sid:
                    continue

                chunk = base64.b64decode(data["media"]["payload"])
                pcm16 = audioop.ulaw2lin(chunk, 2)  # 16-bit PCM @ 8k
                audio_buf.extend(pcm16)

                while len(audio_buf) >= PCM16_BYTES_PER_FRAME:
                    frame = bytes(audio_buf[:PCM16_BYTES_PER_FRAME])
                    del audio_buf[:PCM16_BYTES_PER_FRAME]

                    # Build a noise profile continuously during non-speech periods
                    push_noise(frame)

                    # Mandatory PTT: only buffer speech when PTT is ON.
                    if ptt_active:
                        ptt_buf.extend(frame)

            elif event == "stop":
                print(">> Stream stopped")
                break

            elif event == "mark":
                pass

    except Exception as e:
        print(f"[ERROR] WS Handler: {e}")

    stop_evt.set()
    try:
        work_q.put_nowait(None)
    except Exception:
        pass

    print(">> Stream closed")


# ===========================================
# Twilio webhook
# ===========================================
@app.route("/call", methods=["POST"])
def twilio_call():
    ngrok_domain = (os.getenv("NGROK_DOMAIN") or "").strip()

    if ngrok_domain:
        stream_url = f"wss://{ngrok_domain}/ws"
    else:
        stream_url = request.host_url.replace("http://", "wss://").replace("https://", "wss://").strip("/") + "/ws"

    print(f"[Twilio] Incoming call. Streaming audio to: {stream_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Hello! Press 5 to start talking, then press 5 again when you are done. You will hear a beep when recording starts.</Say>
  <Connect>
    <Stream url="{stream_url}"><Parameter name="pttDigit" value="5" /></Stream>
  </Connect>
</Response>"""
    return Response(twiml, mimetype="text/xml")

@app.route("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.route("/")
def index():
    return "<h3>✅ Voice LLM Server Running</h3>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
