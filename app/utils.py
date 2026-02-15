import base64, json, audioop

def decode_twilio_audio(payload_b64: str) -> bytes:
    mulaw_audio = base64.b64decode(payload_b64)
    return audioop.ulaw2lin(mulaw_audio, 2)
