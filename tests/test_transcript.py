import pytest
import streamlit
import importlib

def test_transcribe_empty(tmp_path, monkeypatch):
    # 1) Stub out secrets
    monkeypatch.setattr(streamlit, "secrets", {"openai": {"key": "dummy-key"}})

    # 2) Reload the module so it sees our fake secrets
    tx = importlib.reload(importlib.import_module("speech_transcriber.transcript"))

    # 3) Patch the OpenAI class inside that module
    class DummyTranscriptions:
        @staticmethod
        def create(model, file, response_format):
            return "MOCK‑TRANSCRIPT"

    class DummyClient:
        def __init__(self, api_key):
            self.audio = type("A", (), {"transcriptions": DummyTranscriptions})()

    monkeypatch.setattr(tx, "OpenAI", DummyClient)

    # 4) Create an empty audio file
    silent = tmp_path / "silence.wav"
    silent.write_bytes(b"")

    # 5) Call and assert
    result = tx.transcribe(str(silent))
    assert result == "MOCK‑TRANSCRIPT"
