from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import io

#load_dotenv()
#OPENAI_KEY = os.getenv("OPENAI_KEY")

def transcribe_from_file(filepath: str) -> str:
    """Return transcript of the audio file at audio_path."""
    OPENAI_KEY = st.secrets["openai"]["key"]
    if OPENAI_KEY is None:
        raise RuntimeError("Please set the OPENAI_KEY environment variable")
    client = OpenAI(api_key = OPENAI_KEY)
    audio_file = open(filepath, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
    )
    return transcription     

def transcribe(audio_bytes: bytes,filename: str) -> str:
    """Return transcript of the audio file at audio_path."""
    OPENAI_KEY = st.secrets["openai"]["key"]
    if OPENAI_KEY is None:
        raise RuntimeError("Please set the OPENAI_KEY environment variable")
    client = OpenAI(api_key = OPENAI_KEY)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
    )
    return transcription     