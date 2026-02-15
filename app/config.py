# config.py
import os

# Twilio credentials (used only if needed for outbound calls or signature verification)
# TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
# TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

# Ollama endpoint
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")

# Ngrok public domain (used in TwiML)
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN", "anna-bushier-noncensoriously.ngrok-free.dev")
