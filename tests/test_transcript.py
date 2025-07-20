# tests/test_transcript.py

import io
import pytest
import streamlit as st

# import your module once
import speech_transcriber.transcript as tx

def test_transcribe_success(monkeypatch):
    # 1) Stub out the secret so the key is "dummy-key"
    monkeypatch.setattr(st, "secrets", {"openai": {"key": "dummy-key"}})

    # 2) Create a DummyClient to replace tx.OpenAI
    class DummyTranscriptions:
        @staticmethod
        def create(model, file, response_format):
            # ensure we got a BytesIO with the right .name
            assert isinstance(file, io.BytesIO)
            assert file.name == "audio.wav"
            return "MOCK‑TRANSCRIPT"

    class DummyClient:
        def __init__(self, api_key):
            assert api_key == "dummy-key"
            self.audio = type("A", (), {"transcriptions": DummyTranscriptions()})()

    # 3) Patch the OpenAI constructor inside your transcript module
    monkeypatch.setattr(tx, "OpenAI", DummyClient)

    # 4) Call transcribe() and verify the result
    result = tx.transcribe(b"dummy audio bytes", "audio.wav")
    assert result == "MOCK‑TRANSCRIPT"


def test_transcribe_no_key(monkeypatch):
    # 1) Stub secrets with no "openai" key
    monkeypatch.setattr(st, "secrets", {})

    # 2) Expect a KeyError when trying to read st.secrets["openai"]
    with pytest.raises(KeyError):
        tx.transcribe(b"", "audio.wav")


def test_transcribe_api_error(monkeypatch):
    # 1) Stub secrets so the key exists
    monkeypatch.setattr(st, "secrets", {"openai": {"key": "dummy-key"}})

    # 2) DummyClient whose .create() raises an Exception
    class DummyClient:
        def __init__(self, api_key):
            self.audio = type(
                "A",
                (),
                {
                    "transcriptions": type(
                        "T",
                        (),
                        {
                            "create": staticmethod(
                                lambda *args, **kwargs: (_ for _ in ()).throw(Exception("API Error"))
                            )
                        },
                    )()
                },
            )()

    monkeypatch.setattr(tx, "OpenAI", DummyClient)

    # 3) The Exception("API Error") should propagate
    with pytest.raises(Exception) as excinfo:
        tx.transcribe(b"", "audio.wav")
    assert "API Error" in str(excinfo.value)
