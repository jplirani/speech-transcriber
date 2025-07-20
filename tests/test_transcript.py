# tests/test_transcript.py

import io
import importlib
import pytest
import streamlit as st
import openai

# helper to reload the module under a fresh secrets stub
def _load_module_with_secrets(monkeypatch, secrets_dict):
    monkeypatch.setattr(st, "secrets", secrets_dict)
    return importlib.reload(importlib.import_module("speech_transcriber.transcript"))

def test_transcribe_success(monkeypatch):
    # 1) Stub secrets so OPENAI_KEY = "dummy-key"
    tx = _load_module_with_secrets(monkeypatch, {"openai": {"key": "dummy-key"}})

    # 2) Patch openai.OpenAI to use a dummy client
    class DummyTranscriptions:
        @staticmethod
        def create(model, file, response_format):
            # should be BytesIO and have .name set to "audio.wav"
            assert isinstance(file, io.BytesIO)
            assert file.name == "audio.wav"
            return "MOCK‑TRANSCRIPT"

    class DummyClient:
        def __init__(self, api_key):
            assert api_key == "dummy-key"
            self.audio = type("A", (), {"transcriptions": DummyTranscriptions()})()

    monkeypatch.setattr(openai, "OpenAI", DummyClient)

    # 3) Call transcribe with dummy bytes + filename
    dummy = b"dummy audio data"
    result = tx.transcribe(dummy, "audio.wav")

    # 4) Verify
    assert result == "MOCK‑TRANSCRIPT"

def test_transcribe_no_key(monkeypatch):
    # No "openai" key in secrets → should raise RuntimeError
    tx = _load_module_with_secrets(monkeypatch, {})  # empty secrets
    with pytest.raises(RuntimeError) as excinfo:
        tx.transcribe(b"", "audio.wav")
    assert "Please set the OPENAI_KEY environment variable" in str(excinfo.value)

def test_transcribe_api_error(monkeypatch):
    # Stub secrets
    tx = _load_module_with_secrets(monkeypatch, {"openai": {"key": "dummy-key"}})

    # Dummy client whose .create() raises
    class DummyClient:
        def __init__(self, api_key):
            self.audio = type(
                "A",
                (),
                {"transcriptions": type(
                    "T",
                    (),
                    {"create": staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(Exception("API failure")))}
                )()}
            )()

    monkeypatch.setattr(openai, "OpenAI", DummyClient)

    # Should propagate the Exception("API failure")
    with pytest.raises(Exception) as excinfo:
        tx.transcribe(b"", "audio.wav")
    assert "API failure" in str(excinfo.value)
