import streamlit as st
from speech_transcriber.transcript import transcribe

st.title("🗣️ Speech Transcriber")
audio = st.file_uploader("Upload an audio file", type=["wav","mp3","m4a"])
if audio:
    # Save the uploaded file to a temp path
    with open("temp_audio", "wb") as f:
        f.write(audio.read())
    text = transcribe("temp_audio")
    st.text_area("Transcript", text, height=300)
    st.download_button("Copy transcript", text, file_name="transcript.txt")
    