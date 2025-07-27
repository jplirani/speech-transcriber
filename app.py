import streamlit as st
from pathlib import Path
from speech_transcriber.transcript import transcribe
import pyperclip

st.title("🗣️ Speech Transcriber")

audio = st.file_uploader("Upload an audio file", type=["wav","mp3","m4a","ogg","flac"])
if audio:
    ext = Path(audio.name).suffix           # get ".ogg", ".mp3", etc.
    temp_path = f"temp_audio{ext}"          # e.g. "temp_audio.ogg"
    audio_bytes = audio.read()
    st.write("🔑 Key present?  ", bool(st.secrets["openai"]["key"]))
    text = transcribe(audio_bytes, audio.name)
    st.text_area("Transcript", text, height=300)
    st.download_button("Download transcript", text, file_name="transcript.txt")
    # Native Streamlit copy button, using a hidden <textarea> and execCommand
    if st.button("📋 Copy transcript"):
        # Escape HTML so nothing breaks our hidden textarea
        pyperclip.copy(text)     
        st.success("✅ Transcript copied to clipboard!")
