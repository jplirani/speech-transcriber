import pytest
import io
from unittest.mock import MagicMock, patch, ANY
import streamlit
import importlib

def test_transcribe_success(monkeypatch):
    """Test successful transcription with bytes input"""
    # 1) Mock Streamlit secrets
    monkeypatch.setattr(streamlit, "secrets", {"openai": {"key": "sk-test123"}})
    
    # 2) Import and reload the module
    import speech_transcriber.transcript as tx
    importlib.reload(tx)
    
    # 3) Create a mock client
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = "Hello world"
    
    # 4) Patch the OpenAI client
    with patch.object(tx, "OpenAI", return_value=mock_client):
        # 5) Create test audio bytes (minimal MP3 header)
        test_audio = b'\xff\xf3P\x00\x00\x00\x00\x00\x00\x00\x00'
        
        # 6) Call the function
        result = tx.transcribe(test_audio, "test.mp3")
        
        # 7) Verify
        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once_with(
            model="whisper-1",
            file=ANY,
            response_format="text"
        )
        
        # Verify file object
        file_arg = mock_client.audio.transcriptions.create.call_args[1]['file']
        assert isinstance(file_arg, io.BytesIO)
        assert file_arg.name == "test.mp3"
        assert file_arg.read() == test_audio

def test_transcribe_handles_errors(monkeypatch):
    """Test error handling"""
    # 1) Setup
    monkeypatch.setattr(streamlit, "secrets", {"openai": {"key": "sk-test123"}})
    import speech_transcriber.transcript as tx
    importlib.reload(tx)
    
    # 2) Mock a failing API call
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.side_effect = Exception("API Error")
    
    # 3) Run test
    with patch.object(tx, "OpenAI", return_value=mock_client):
        result = tx.transcribe(b"test", "error.mp3")
        assert result == ""  # Or your expected error return