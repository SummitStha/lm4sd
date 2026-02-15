import os
import json
import time
import base64
import audioop
import re
from typing import Optional, Tuple, List, Dict

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

def send_audio_to_twilio(ws, stream_sid: str, mulaw_bytes: bytes):
    """
    Twilio expects JSON messages with:
      event=media, streamSid, media.payload (base64)
    and 20ms frames at 8kHz mu-law => 160 bytes per frame.
    """
    if not mulaw_bytes:
        return

    frame_size = 160
    for i in range(0, len(mulaw_bytes), frame_size):
        frame = mulaw_bytes[i:i + frame_size]
        payload = base64.b64encode(frame).decode("ascii")
        message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }
        ws.send(json.dumps(message))
        time.sleep(0.02)  # pacing: 20ms

    ws.send(json.dumps({
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": "done"},
    }))

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

def whisper_transcribe(pcm16_8k: bytes) -> Tuple[str, Optional[str]]:
    """
    Runs Whisper on an utterance (PCM16 @8k) and returns (text, detected_lang),
    with post-detection correction for Nepali vs Hindi.
    """
    audio = pcm8k_to_float16k(pcm16_8k)

    segments, info = whisper.transcribe(
        audio,
        beam_size=5,
        vad_filter=False,
        language=None,   # auto-detect
        task="transcribe"
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
class VadSegmenter:
    def __init__(self, mode: int = 2, end_silence_ms: int = 600, max_utt_ms: int = 12000):
        self.vad = webrtcvad.Vad(mode)
        self.end_silence_ms = end_silence_ms
        self.max_utt_ms = max_utt_ms
        self.reset()

    def reset(self):
        self.in_speech = False
        self.frames: List[bytes] = []
        self.silence_ms = 0
        self.utt_ms = 0

    def push_frame(self, pcm16_frame_8k: bytes) -> Optional[bytes]:
        if len(pcm16_frame_8k) != PCM16_BYTES_PER_FRAME:
            return None

        is_speech = self.vad.is_speech(pcm16_frame_8k, TWILIO_SR)

        if is_speech:
            if not self.in_speech:
                self.in_speech = True
                self.frames = []
                self.silence_ms = 0
                self.utt_ms = 0

            self.frames.append(pcm16_frame_8k)
            self.utt_ms += FRAME_MS
            self.silence_ms = 0

            if self.utt_ms >= self.max_utt_ms:
                utt = b"".join(self.frames)
                self.reset()
                return utt

        else:
            if self.in_speech:
                self.silence_ms += FRAME_MS
                self.frames.append(pcm16_frame_8k)
                self.utt_ms += FRAME_MS

                if self.silence_ms >= self.end_silence_ms:
                    utt = b"".join(self.frames)
                    self.reset()
                    return utt

        return None

# ===========================================
# WebSocket: Twilio Media Stream endpoint
# ===========================================
@sock.route("/ws")
def ws(ws):
    print(">> Stream connected")

    stream_sid = None
    audio_buf = bytearray()
    segmenter = VadSegmenter(mode=2, end_silence_ms=600, max_utt_ms=12000)

    # LLM->TTS batching (avoid speaking tiny subwords)
    tts_buffer = ""
    MIN_CHARS = 60

    def should_flush(buf: str) -> bool:
        if len(buf) >= MIN_CHARS:
            return True
        return bool(re.search(r"[\.!\?\n]|।", buf))

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

            elif event == "media":
                if not stream_sid:
                    continue

                chunk = base64.b64decode(data["media"]["payload"])
                pcm16 = audioop.ulaw2lin(chunk, 2)  # 16-bit PCM @ 8k
                audio_buf.extend(pcm16)

                while len(audio_buf) >= PCM16_BYTES_PER_FRAME:
                    frame = bytes(audio_buf[:PCM16_BYTES_PER_FRAME])
                    del audio_buf[:PCM16_BYTES_PER_FRAME]

                    utt = segmenter.push_frame(frame)
                    if utt:
                        text, lang = whisper_transcribe(utt)
                        text = (text or "").strip()
                        if not text:
                            continue

                        # corrected language already applied
                        print(f"[User] ({lang}) {text}")

                        full_reply = ""
                        current_tts_lang = lang if lang in ("ne", "hi", "en") else "hi"
                        tts_voice = PIPER_VOICES.get(current_tts_lang, PIPER_VOICES["hi"])

                        for rchunk, reply_lang in stream_llm_reply(text, lang):
                            full_reply += rchunk
                            print("[LLM partial]", rchunk.strip())

                            # Keep TTS voice synced to reply language (auto mode)
                            if reply_lang in ("ne", "hi", "en"):
                                current_tts_lang = reply_lang
                                tts_voice = PIPER_VOICES.get(current_tts_lang, PIPER_VOICES["hi"])

                            tts_buffer += rchunk
                            if should_flush(tts_buffer):
                                mulaw = speak_text_to_mulaw_8k(tts_buffer, tts_voice)
                                if mulaw:
                                    send_audio_to_twilio(ws, stream_sid, mulaw)
                                tts_buffer = ""

                        if tts_buffer.strip():
                            mulaw = speak_text_to_mulaw_8k(tts_buffer, tts_voice)
                            if mulaw:
                                send_audio_to_twilio(ws, stream_sid, mulaw)
                            tts_buffer = ""

                        print("[LLM full]", full_reply.strip())

            elif event == "stop":
                print(">> Stream stopped")
                break

    except Exception as e:
        print(f"[ERROR] WS Handler: {e}")

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
  <Say>Hello! Connecting you to the multilingual voice assistant now.</Say>
  <Connect>
    <Stream url="{stream_url}" />
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
