import os
import io
import json
import time
import base64
import audioop
import asyncio
import inspect
import numpy as np
import soundfile as sf
import types, time
from flask import Flask, request, Response
from flask_sock import Sock
from vosk import Model, KaldiRecognizer
from piper.voice import PiperVoice
import requests

# ===========================================
# Flask + WebSocket setup
# ===========================================
app = Flask(__name__)
sock = Sock(app)

# ===========================================
# Paths and model initialization
# ===========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")

# ----- VOSK -----
vosk_path = os.path.join(BASE_DIR, "models", "vosk_model_8k")
if not os.path.exists(vosk_path):
    raise RuntimeError(f"❌ Vosk model not found at {vosk_path}")
print("[INIT] Loading Vosk model...")
vosk_model = Model(vosk_path)
print("[INIT] ✅ Vosk model loaded.")

recognizer = KaldiRecognizer(vosk_model, 8000)

# ----- PIPER -----
PIPER_PATH = "/root/.local/share/piper/voices/en_US-amy-low"
voice = PiperVoice.load(
    os.path.join(PIPER_PATH, "en_US-amy-low.onnx"),
    config_path=os.path.join(PIPER_PATH, "en_US-amy-low.onnx.json"),
)
print("[INIT] ✅ Piper voice loaded.")

# ===========================================
# Utilities
# ===========================================
def pcm16_to_mulaw(pcm16_bytes):
    """Convert 16-bit PCM bytes → 8-bit μ-law for Twilio."""
    return audioop.lin2ulaw(pcm16_bytes, 2)


def speak_text(text: str) -> bytes:
    if not text.strip():
        return b""
    result = voice.synthesize(text)
    pcm_bytes = bytearray()
    sr = getattr(voice, "sample_rate", 22050)
    if isinstance(result, types.GeneratorType):
        for chunk in result:
            if hasattr(chunk, "audio_int16_bytes"):
                pcm_bytes.extend(chunk.audio_int16_bytes)
                sr = chunk.sample_rate
            elif hasattr(chunk, "audio_float_array"):
                pcm16 = (np.clip(chunk.audio_float_array, -1, 1) * 32767).astype(np.int16)
                pcm_bytes.extend(pcm16.tobytes())
                sr = chunk.sample_rate
            else:
                pcm16 = (np.asarray(chunk, dtype=np.float32) * 32767).astype(np.int16)
                pcm_bytes.extend(pcm16.tobytes())
    elif isinstance(result, tuple):
        audio, sr = result
        pcm_bytes = (np.asarray(audio, dtype=np.float32) * 32767).astype(np.int16).tobytes()
    else:
        audio = result
        pcm_bytes = (np.asarray(audio, dtype=np.float32) * 32767).astype(np.int16).tobytes()

    pcm_8k = audioop.ratecv(pcm_bytes, 2, 1, sr, 8000, None)[0]
    return audioop.lin2ulaw(pcm_8k, 2)

# === Helper: send audio to Twilio correctly ===
def send_audio_to_twilio(ws, stream_sid: str, mulaw_bytes: bytes):
    frame_size = 160
    for i in range(0, len(mulaw_bytes), frame_size):
        frame = mulaw_bytes[i:i + frame_size]
        payload = base64.b64encode(frame).decode("ascii")
        message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        }
        ws.send(json.dumps(message))
        time.sleep(0.02)
    # mark end
    ws.send(json.dumps({
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": "done"}
    }))


def stream_llm_reply(prompt: str):
    """Stream response text chunks from local Ollama model."""
    print(f"[LLM] Processing prompt: {prompt}")
    payload = {"model": "llama3.2:3b", "prompt": prompt, "stream": True}

    with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
        if r.status_code != 200:
            print(f"[LLM ERROR] Ollama returned {r.status_code}")
            return
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
            except Exception as e:
                print(f"[LLM Parse Error] {e}")

# ===========================================
# Hybrid-safe WebSocket wrapper
# ===========================================
# @sock.route("/ws")
# def ws(ws):
#     """Hybrid sync/async handler for Twilio media streams."""
#     if inspect.iscoroutinefunction(handle_ws):
#         asyncio.run(handle_ws(ws))
#     else:
#         handle_ws(ws)


# async def handle_ws(ws):
#     """Main Twilio ↔ LLM ↔ TTS pipeline."""
#     print(">> Stream connected")
#     recognizer = KaldiRecognizer(vosk_model, 8000)
#     pcm_buffer = bytearray()
#     stream_sid = None

#     try:
#         while True:
#             message = ws.receive()
#             if message is None:
#                 print(">> Stream closed")
#                 break

#             msg = json.loads(message)
#             event = msg.get("event")

#             if event == "start":
#                 stream_sid = msg["start"]["streamSid"]
#                 print(f">> Stream started (SID: {stream_sid})")

#             elif event == "media":
#                 payload = base64.b64decode(msg["media"]["payload"])
#                 pcm_chunk = audioop.ulaw2lin(payload, 2)
#                 pcm_buffer.extend(pcm_chunk)

#                 if len(pcm_buffer) > 3200:
#                     if recognizer.AcceptWaveform(bytes(pcm_buffer)):
#                         result = json.loads(recognizer.Result())
#                         text = result.get("text", "").strip()
#                         if text:
#                             print(f"[User] {text}")

#                             # LLM streaming response
#                             for chunk in stream_llm_reply(text):
#                                 print(f"[LLM partial] {chunk}")
#                                 mulaw_reply = speak_text(chunk)
#                                 if mulaw_reply:
#                                     send_audio_to_twilio(ws, stream_sid, mulaw_reply)

#                             final_result = json.loads(recognizer.FinalResult())
#                             print(f"[LLM full] {final_result}")
#                     pcm_buffer.clear()

#             elif event == "stop":
#                 print(">> Stream stopped")
#                 break

#     except Exception as e:
#         print(f"[ERROR] WS Handler: {e}")


# # === Main WebSocket handler ===
@sock.route("/ws")
def ws(ws):
    print(">> Stream connected")

    stream_sid = None
    buffer = ""
    min_chars = 60
    silence_timeout = 1
    last_time = time.time()

    while True:
        msg = ws.receive()
        if not msg:
            break
        data = json.loads(msg)

        if data["event"] == "start":
            stream_sid = data["start"]["streamSid"]
            print(f">> Stream started (SID: {stream_sid})")

        elif data["event"] == "media":
            # convert caller audio → text
            chunk = base64.b64decode(data["media"]["payload"])
            pcm = audioop.ulaw2lin(chunk, 2)
            if recognizer.AcceptWaveform(pcm):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"[User] {text}")

                    reply_text = ""
                    for rchunk in stream_llm_reply(text):
                        reply_text += rchunk
                        print("[LLM partial]", rchunk.strip())
                        buffer += rchunk
                        # batch speech to avoid tiny words
                        if len(buffer) >= min_chars or (time.time() - last_time > silence_timeout):
                            mulaw = speak_text(buffer)
                            if mulaw:
                                send_audio_to_twilio(ws, stream_sid, mulaw)
                            buffer = ""
                        last_time = time.time()

                    # flush any remaining text
                    if buffer.strip():
                        mulaw = speak_text(buffer)
                        if mulaw:
                            send_audio_to_twilio(ws, stream_sid, mulaw)
                    print("[LLM full]", reply_text)
                    buffer = ""

        elif data["event"] == "stop":
            print(">> Stream stopped")
            break

    print(">> Stream closed")

# # === WebSocket route ===
# # @sock.route("/ws")
# # def ws(ws):
# #     print(">> Stream connected")
# #     buffer = ""
# #     min_chars = 60
# #     silence_timeout = 1.0
# #     last_time = time.time()

# #     while True:
# #         msg = ws.receive()
# #         if not msg:
# #             break
# #         data = json.loads(msg)

# #         # Handle Twilio events
# #         if data["event"] == "start":
# #             print(">> Stream started")
# #         elif data["event"] == "media":
# #             chunk = base64.b64decode(data["media"]["payload"])
# #             pcm = audioop.ulaw2lin(chunk, 2)
# #             if recognizer.AcceptWaveform(pcm):
# #                 result = json.loads(recognizer.Result())
# #                 text = result.get("text", "").strip()
# #                 if text:
# #                     print(f"[User] {text}")
# #                     reply_text = ""
# #                     for rchunk in stream_reply(text):
# #                         reply_text += rchunk
# #                         print("[LLM partial]", rchunk.strip())
# #                         buffer += rchunk
# #                         if len(buffer) >= min_chars or (time.time() - last_time > silence_timeout):
# #                             mulaw_reply = speak_text(buffer)
# #                             if mulaw_reply:
# #                                 print(f"[DEBUG] sending {len(mulaw_reply)} bytes audio")
# #                                 send_audio_to_twilio(ws, mulaw_reply)
# #                             buffer = ""
# #                         last_time = time.time()
# #                     if buffer.strip():
# #                         mulaw_reply = speak_text(buffer)
# #                         if mulaw_reply:
# #                             send_audio_to_twilio(ws, mulaw_reply)
# #                     print("[LLM full]", reply_text)
# #                     buffer = ""
# #         elif data["event"] == "stop":
# #             print(">> Stream stopped")
# #             break
# #     print(">> Stream closed")


# # @sock.route("/ws")
# # def ws(ws):
# #     """
# #     Minimal Twilio playback test.
# #     Generates "Hello from your voice server" and sends it correctly.
# #     """
# #     import base64, json, time

# #     print(">> Stream connected")

# #     stream_sid = None

# #     # Wait for Twilio start event
# #     while True:
# #         msg = ws.receive()
# #         if not msg:
# #             return
# #         data = json.loads(msg)
# #         if data["event"] == "start":
# #             stream_sid = data["start"]["streamSid"]
# #             print(f">> Stream started (SID: {stream_sid})")
# #             break

# #     # === Synthesize a simple test phrase ===
# #     test_text = "Hello from your voice server. This is a test response."
# #     mulaw = speak_text(test_text)
# #     print(f"[DEBUG] Generated {len(mulaw)} μ-law bytes")

# #     # === Send frames with correct Twilio JSON structure ===
# #     frame_size = 160  # 20 ms per frame @8 kHz μ-law
# #     for i in range(0, len(mulaw), frame_size):
# #         frame = mulaw[i:i+frame_size]
# #         payload = base64.b64encode(frame).decode("ascii")
# #         message = {
# #             "event": "media",
# #             "streamSid": stream_sid,
# #             "media": {"payload": payload}
# #         }
# #         ws.send(json.dumps(message))
# #         time.sleep(0.02)

# #     # Optional: tell Twilio playback is done
# #     ws.send(json.dumps({
# #         "event": "mark",
# #         "streamSid": stream_sid,
# #         "mark": {"name": "done"}
# #     }))

# #     print(">> Test audio sent, closing stream.")




# ===========================================
# Twilio webhook + utilities
# ===========================================
@app.route("/call", methods=["POST"])
def twilio_call():
    """TwiML webhook for incoming phone calls."""
    ngrok_host = request.host_url.replace("http://", "wss://").strip("/")
    stream_url = f"{ngrok_host}/ws"
    print(f"[Twilio] Incoming call. Streaming audio to: {stream_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">Hello! Connecting you to the AI assistant now.</Say>
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

# ===========================================
# Entrypoint
# ===========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)