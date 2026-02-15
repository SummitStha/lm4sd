from piper.download import download_voice_model

# Download the "en_US-amy-low" voice and keep it cached under
# /root/.local/share/piper/voices/en_US-amy-low
print("Downloading Piper voice model (en_US-amy-low)...")
download_voice_model("en_US-amy-low")
print("Piper voice model installed successfully.")
