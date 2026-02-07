"""Tests for OpenAI TTS engine (mock API calls)."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from tts_pipeline.openai_tts import (
    OpenAITTSEngine,
    _split_for_openai,
    MAX_CHARS_PER_REQUEST,
)


# ---------------------------------------------------------------------------
# Text processing tests
# ---------------------------------------------------------------------------


def test_split_for_openai_short_text():
    """Test text under limit is not split."""
    text = "Short text that fits easily."
    chunks = _split_for_openai(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_for_openai_exact_limit():
    """Test text exactly at limit is not split."""
    text = "A" * MAX_CHARS_PER_REQUEST
    chunks = _split_for_openai(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_for_openai_long_text():
    """Test long text is split into multiple chunks."""
    # Create text that exceeds the limit
    long_text = ". ".join(["Sentence number {}".format(i) for i in range(1000)])
    chunks = _split_for_openai(long_text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= MAX_CHARS_PER_REQUEST
    # Verify all text is preserved
    assert " ".join(chunks).replace(" . ", ". ") == long_text


def test_split_for_openai_sentence_boundary():
    """Test splitting respects sentence boundaries."""
    # Multiple sentences that together exceed limit
    sentences = [f"Sentence {i}." for i in range(100)]
    text = " ".join(sentences)
    chunks = _split_for_openai(text, max_chars=200)

    assert len(chunks) > 1
    # Each chunk should end with a sentence
    for chunk in chunks[:-1]:  # All but last chunk
        assert chunk.strip().endswith(".")


def test_split_for_openai_oversized_sentence():
    """Test handling of single sentence exceeding limit."""
    # Single very long sentence
    long_sentence = " ".join(["word"] * 1000) + "."
    chunks = _split_for_openai(long_sentence, max_chars=200)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200


# ---------------------------------------------------------------------------
# Engine initialization tests
# ---------------------------------------------------------------------------


def test_engine_initialization_defaults():
    """Test engine can be constructed with default params."""
    engine = OpenAITTSEngine(api_key="sk-test")
    assert engine.model == "gpt-4o-mini-tts"
    assert engine.voice == "alloy"
    assert engine.api_key == "sk-test"
    assert engine.instructions is None
    assert engine.sample_rate == 24_000


def test_engine_initialization_custom():
    """Test engine accepts custom parameters."""
    engine = OpenAITTSEngine(
        model="tts-1",
        voice="nova",
        api_key="sk-custom",
        instructions="Speak slowly",
        device="cuda",  # should be ignored
    )
    assert engine.model == "tts-1"
    assert engine.voice == "nova"
    assert engine.api_key == "sk-custom"
    assert engine.instructions == "Speak slowly"
    assert engine.device == "cuda"  # stored but not used


def test_engine_sample_rate():
    """Test sample rate property returns correct value."""
    engine = OpenAITTSEngine(api_key="sk-test")
    assert engine.sample_rate == 24_000


# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------


def test_load_missing_api_key():
    """Test error when API key is missing."""
    engine = OpenAITTSEngine(api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        engine.load()


def test_load_missing_openai_library():
    """Test error when openai library not installed."""
    engine = OpenAITTSEngine(api_key="sk-test")
    with patch("tts_pipeline.openai_tts.OpenAI", side_effect=ImportError("No module named 'openai'")):
        with pytest.raises(ImportError, match="openai library is required"):
            engine.load()


@patch("tts_pipeline.openai_tts.OpenAI")
def test_load_success(mock_openai_class):
    """Test successful client initialization."""
    mock_client = Mock()
    mock_openai_class.return_value = mock_client

    engine = OpenAITTSEngine(api_key="sk-test")
    engine.load()

    assert engine._client is mock_client
    mock_openai_class.assert_called_once_with(api_key="sk-test")


# ---------------------------------------------------------------------------
# Synthesis tests
# ---------------------------------------------------------------------------


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_success(mock_openai_class):
    """Test successful synthesis returns numpy array."""
    # Mock API response
    mock_client = Mock()
    mock_response = Mock()
    # Create fake audio data: 1 second at 24kHz = 24000 samples
    fake_audio = np.zeros(24000, dtype=np.int16)
    mock_response.content = fake_audio.tobytes()
    mock_client.audio.speech.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    engine = OpenAITTSEngine(api_key="sk-test")
    result = engine.synthesize("Test text")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == 24000
    assert result.min() >= -1.0
    assert result.max() <= 1.0

    # Verify API was called with correct parameters
    mock_client.audio.speech.create.assert_called_once()
    call_kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini-tts"
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["input"] == "Test text"
    assert call_kwargs["response_format"] == "pcm"


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_empty_text(mock_openai_class):
    """Test error on empty text."""
    engine = OpenAITTSEngine(api_key="sk-test")
    with pytest.raises(ValueError, match="empty after normalization"):
        engine.synthesize("")


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_whitespace_only(mock_openai_class):
    """Test error on whitespace-only text."""
    engine = OpenAITTSEngine(api_key="sk-test")
    with pytest.raises(ValueError, match="empty after normalization"):
        engine.synthesize("   \n\t  ")


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_long_text_chunking(mock_openai_class):
    """Test long text is split and concatenated."""
    # Mock API response
    mock_client = Mock()

    def create_response(*args, **kwargs):
        """Create a mock response with 1000 samples per chunk."""
        mock_response = Mock()
        fake_audio = np.zeros(1000, dtype=np.int16)
        mock_response.content = fake_audio.tobytes()
        return mock_response

    mock_client.audio.speech.create.side_effect = create_response
    mock_openai_class.return_value = mock_client

    # Create text that will be split into chunks
    long_text = ". ".join([f"Sentence {i}" for i in range(500)])
    engine = OpenAITTSEngine(api_key="sk-test")
    result = engine.synthesize(long_text)

    # Should have made multiple API calls
    assert mock_client.audio.speech.create.call_count > 1

    # Result should be concatenated audio
    assert isinstance(result, np.ndarray)
    assert len(result) == 1000 * mock_client.audio.speech.create.call_count


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_rate_limit_error(mock_openai_class):
    """Test rate limit error is caught and re-raised with helpful message."""
    mock_client = Mock()
    # Create an exception that looks like OpenAI's RateLimitError
    rate_limit_error = Exception("Rate limit exceeded")
    rate_limit_error.__class__.__name__ = "RateLimitError"
    mock_client.audio.speech.create.side_effect = rate_limit_error
    mock_openai_class.return_value = mock_client

    engine = OpenAITTSEngine(api_key="sk-test")
    with pytest.raises(RuntimeError, match="rate limit"):
        engine.synthesize("Test text")


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_auth_error(mock_openai_class):
    """Test authentication error is caught with helpful message."""
    mock_client = Mock()
    # Create an exception that looks like OpenAI's AuthenticationError
    auth_error = Exception("Invalid API key")
    auth_error.__class__.__name__ = "AuthenticationError"
    mock_client.audio.speech.create.side_effect = auth_error
    mock_openai_class.return_value = mock_client

    engine = OpenAITTSEngine(api_key="sk-test")
    with pytest.raises(ValueError, match="authentication failed"):
        engine.synthesize("Test text")


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_api_error(mock_openai_class):
    """Test generic API error is caught."""
    mock_client = Mock()
    api_error = Exception("API error occurred")
    api_error.__class__.__name__ = "APIError"
    mock_client.audio.speech.create.side_effect = api_error
    mock_openai_class.return_value = mock_client

    engine = OpenAITTSEngine(api_key="sk-test")
    with pytest.raises(RuntimeError, match="API error"):
        engine.synthesize("Test text")


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_network_error(mock_openai_class):
    """Test network error is caught with helpful message."""
    mock_client = Mock()
    network_error = Exception("Connection timeout")
    network_error.__class__.__name__ = "ConnectionError"
    mock_client.audio.speech.create.side_effect = network_error
    mock_openai_class.return_value = mock_client

    engine = OpenAITTSEngine(api_key="sk-test")
    with pytest.raises(RuntimeError, match="Network error"):
        engine.synthesize("Test text")


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_lazy_loading(mock_openai_class):
    """Test client is loaded automatically on first synthesize call."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = np.zeros(1000, dtype=np.int16).tobytes()
    mock_client.audio.speech.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    engine = OpenAITTSEngine(api_key="sk-test")
    assert engine._client is None  # Not loaded yet

    engine.synthesize("Test")

    assert engine._client is not None  # Now loaded
    mock_openai_class.assert_called_once()


# ---------------------------------------------------------------------------
# Model and voice parameter tests
# ---------------------------------------------------------------------------


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_with_tts_1_model(mock_openai_class):
    """Test synthesis with tts-1 model."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = np.zeros(1000, dtype=np.int16).tobytes()
    mock_client.audio.speech.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    engine = OpenAITTSEngine(model="tts-1", api_key="sk-test")
    engine.synthesize("Test")

    call_kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert call_kwargs["model"] == "tts-1"


@patch("tts_pipeline.openai_tts.OpenAI")
def test_synthesize_with_different_voices(mock_openai_class):
    """Test synthesis with different voice selections."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = np.zeros(1000, dtype=np.int16).tobytes()
    mock_client.audio.speech.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    for voice in voices:
        mock_client.reset_mock()
        engine = OpenAITTSEngine(voice=voice, api_key="sk-test")
        engine.synthesize("Test")

        call_kwargs = mock_client.audio.speech.create.call_args.kwargs
        assert call_kwargs["voice"] == voice
