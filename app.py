import streamlit as st
from pathlib import Path
from speech_transcriber.transcript import transcribe

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
    # 2️⃣ Copy button
    # We use backticks to wrap the JS template literal; if your transcript has backticks,
    # you may need to escape them or fall back to the JSON‑dump approach.
    st.markdown(f'''
        <button 
            onclick="navigator.clipboard.writeText(`{text}`)"
            style="padding:0.35em 0.75em; margin-top:0.5em; font-size:1em;"
        >📋 Copy transcript</button>
        ''',
        unsafe_allow_html=True
    )