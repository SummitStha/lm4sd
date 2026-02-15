import os, json, sys, base64, audioop
import numpy as np
from typing import Dict
from piper.voice import PiperVoice

TWILIO_SR = 8000

PIPER_VOICES_DIR = os.getenv("PIPER_VOICES_DIR", "/root/.local/share/piper/voices").strip()
PIPER_NE_MODEL = os.getenv("PIPER_NE_MODEL", "ne_NP-chitwan-medium").strip()
PIPER_HI_MODEL = os.getenv("PIPER_HI_MODEL", "hi_IN-rohan-medium").strip()
PIPER_EN_MODEL = os.getenv("PIPER_EN_MODEL", "en_US-amy-low").strip()
PIPER_PATH = os.getenv("PIPER_PATH", "").strip()
PIPER_MODEL = os.getenv("PIPER_MODEL", "").strip()

def _piper_files_exist(onnx_path: str, json_path: str) -> bool:
    return os.path.exists(onnx_path) and os.path.exists(json_path)

def _load_piper_from_dir(model_dir: str, model_name: str) -> PiperVoice:
    onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
    json_path = os.path.join(model_dir, f"{model_name}.onnx.json")
    if not _piper_files_exist(onnx_path, json_path):
        raise RuntimeError(f"Piper files missing for model '{model_name}' at {onnx_path} / {json_path}")
    return PiperVoice.load(onnx_path, config_path=json_path)

def _load_piper_voice(model_name: str) -> PiperVoice:
    model_dir = os.path.join(PIPER_VOICES_DIR, model_name)
    if os.path.isdir(model_dir):
        return _load_piper_from_dir(model_dir, model_name)

    if PIPER_PATH and PIPER_MODEL and model_name == PIPER_MODEL:
        return _load_piper_from_dir(PIPER_PATH, PIPER_MODEL)

    onnx_flat = os.path.join(PIPER_VOICES_DIR, f"{model_name}.onnx")
    json_flat = os.path.join(PIPER_VOICES_DIR, f"{model_name}.onnx.json")
    if _piper_files_exist(onnx_flat, json_flat):
        return PiperVoice.load(onnx_flat, config_path=json_flat)

    raise RuntimeError(f"Could not locate Piper model '{model_name}'")

print("[TTS_WORKER] Loading Piper voices (ne/hi/en)...", file=sys.stderr, flush=True)
VOICES: Dict[str, PiperVoice] = {
    "ne": _load_piper_voice(PIPER_NE_MODEL),
    "hi": _load_piper_voice(PIPER_HI_MODEL),
    "en": _load_piper_voice(PIPER_EN_MODEL),
}
print("[TTS_WORKER] Ready.", file=sys.stderr, flush=True)

def synth_to_mulaw_8k(text: str, voice: PiperVoice) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""
    result = voice.synthesize(text)

    pcm_bytes = bytearray()
    sr = 22050

    if hasattr(result, "__iter__") and not isinstance(result, (bytes, bytearray, np.ndarray, str)):
        for chunk in result:
            if hasattr(chunk, "audio_int16_bytes"):
                pcm_bytes.extend(chunk.audio_int16_bytes)
                sr = int(getattr(chunk, "sample_rate", sr))
            elif hasattr(chunk, "audio_int16_array"):
                arr = np.asarray(chunk.audio_int16_array, dtype=np.int16)
                pcm_bytes.extend(arr.tobytes())
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
        sr = int(getattr(voice, "sample_rate", sr))

    if not pcm_bytes:
        return b""

    pcm_8k = audioop.ratecv(bytes(pcm_bytes), 2, 1, sr, TWILIO_SR, None)[0]
    mulaw = audioop.lin2ulaw(pcm_8k, 2)
    return mulaw

def write_json(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except Exception:
        continue

    if req.get("cmd") != "synth":
        continue

    lang = (req.get("lang") or "hi").lower().strip()
    if lang not in ("ne", "hi", "en"):
        lang = "hi"
    text = req.get("text") or ""

    try:
        mulaw = synth_to_mulaw_8k(text, VOICES[lang])
        # stream in chunks (arbitrary, e.g. 8000 bytes)
        CHUNK = 8000
        for i in range(0, len(mulaw), CHUNK):
            b64 = base64.b64encode(mulaw[i:i+CHUNK]).decode("ascii")
            write_json({"type": "chunk", "b64": b64})
        write_json({"type": "done"})
    except Exception as e:
        write_json({"type": "err", "msg": str(e)})