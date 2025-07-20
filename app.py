import streamlit as st
from pathlib import Path
from speech_transcriber.transcript import transcribe

st.title("🗣️ Speech Transcriber")

audio = st.file_uploader("Upload an audio file", type=["wav","mp3","m4a","ogg","flac"])
if audio:
    ext = Path(audio.name).suffix           # get ".ogg", ".mp3", etc.
    temp_path = f"temp_audio{ext}"          # e.g. "temp_audio.ogg"
    with open(temp_path, "wb") as f:
        f.write(audio.read())

    text = transcribe(temp_path)
    st.text_area("Transcript", text, height=300)
    st.download_button("Copy transcript", text, file_name="transcript.txt")