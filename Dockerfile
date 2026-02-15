FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg sox libsox-fmt-all wget unzip curl && \
    pip install --no-cache-dir flask flask-sock vosk soundfile requests piper-tts twilio

# === Install Piper voice files ===
RUN mkdir -p /root/.local/share/piper/voices/en_US-amy-low && \
    cd /root/.local/share/piper/voices/en_US-amy-low && \
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx && \
    wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json

WORKDIR /app
COPY . /app

EXPOSE 5000
CMD ["bash", "run.sh"]
