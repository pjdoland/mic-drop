"""OpenAI TTS engine implementation.

API-based text-to-speech synthesis using OpenAI's TTS models with
support for the gpt-4o-mini-tts model with instructions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from tts_pipeline.base import TTSEngine
from tts_pipeline.text_processing import normalize_text, split_by_char_limit

# Optional import — set to None when not installed, allows tests to mock
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

logger = logging.getLogger("mic-drop.openai-tts")

# OpenAI TTS outputs at this rate (matches Tortoise)
OPENAI_SAMPLE_RATE: int = 24_000

# OpenAI TTS character limit per API request
MAX_CHARS_PER_REQUEST: int = 4096


class OpenAITTSEngine(TTSEngine):
    """OpenAI TTS engine with API-based synthesis.

    Attributes:
        model: OpenAI TTS model (gpt-4o-mini-tts, tts-1, tts-1-hd)
        voice: One of: alloy, echo, fable, onyx, nova, shimmer
        api_key: OpenAI API key (from config)
        instructions: Optional voice characteristic instructions (gpt-4o-mini-tts only)
        device: Ignored (for interface compatibility)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        api_key: Optional[str] = None,
        instructions: Optional[str] = None,
        device: str = "auto",  # ignored but kept for interface compatibility
    ) -> None:
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.instructions = instructions
        self.device = device  # stored but not used (API-based, no local compute)
        self._client = None  # loaded on first use

    @property
    def sample_rate(self) -> int:
        """Return OpenAI's output sample rate (24 kHz)."""
        return OPENAI_SAMPLE_RATE

    def load(self) -> None:
        """Initialize the OpenAI client.

        Called automatically on first synthesize() call if not done explicitly.

        Raises:
            ImportError: If openai library is not installed
            ValueError: If API key is missing
        """
        if OpenAI is None:
            raise ImportError(
                "openai library is required but not installed.\n"
                "Install it with: pip install openai"
            )

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI TTS.\n"
                "Set it in .mic-drop.env or pass via environment variable."
            )

        logger.info(
            "Initializing OpenAI TTS client (model=%s, voice=%s)",
            self.model,
            self.voice,
        )
        self._client = OpenAI(api_key=self.api_key)
        logger.info("OpenAI TTS ready.")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio.

        Long texts are automatically chunked at 4096 character boundaries
        and concatenated.

        Args:
            text: Input text to synthesize

        Returns:
            1-D float32 numpy array at 24 kHz

        Raises:
            ValueError: If text is empty after normalization
            ImportError: If openai library not installed
            RuntimeError: If API call fails (rate limit, auth, network, etc.)
        """
        if self._client is None:
            self.load()

        text = normalize_text(text)
        if not text:
            raise ValueError("Input text is empty after normalization.")

        # Split into chunks if necessary
        chunks = split_by_char_limit(text, max_chars=MAX_CHARS_PER_REQUEST)
        char_count = len(text)

        # Calculate estimated cost
        if self.model == "gpt-4o-mini-tts":
            cost_per_1k = 0.010
        elif self.model == "tts-1-hd":
            cost_per_1k = 0.030
        else:  # tts-1
            cost_per_1k = 0.015
        estimated_cost = (char_count / 1000) * cost_per_1k

        logger.info(
            "Synthesizing %d chunk(s) (%d characters) with %s "
            "(estimated cost: $%.4f)",
            len(chunks),
            char_count,
            self.model,
            estimated_cost,
        )

        from tqdm import tqdm

        segments: list[np.ndarray] = []
        for i, chunk in enumerate(
            tqdm(chunks, desc="OpenAI TTS", unit="chunk", leave=True)
        ):
            logger.debug(
                "Chunk %d/%d (%d chars): %.60s …",
                i + 1,
                len(chunks),
                len(chunk),
                chunk,
            )
            audio_segment = self._synthesize_chunk(chunk)
            segments.append(audio_segment)

        audio = np.concatenate(segments, axis=0).astype(np.float32)
        duration = len(audio) / self.sample_rate
        logger.info("Synthesis complete: %.2f s at %d Hz.", duration, self.sample_rate)
        return audio

    def _synthesize_chunk(self, text: str) -> np.ndarray:
        """Make OpenAI API call for a single chunk.

        Args:
            text: Text chunk to synthesize (≤ 4096 characters)

        Returns:
            1-D float32 numpy array of audio samples

        Raises:
            RuntimeError: If API call fails
        """
        try:
            # Build request kwargs
            kwargs = {
                "model": self.model,
                "voice": self.voice,
                "input": text,
                "response_format": "pcm",  # raw 16-bit PCM at 24kHz
            }

            # gpt-4o-mini-tts supports instructions parameter
            if self.instructions:
                kwargs["instructions"] = self.instructions
                logger.debug("Using instructions: %s", self.instructions)

            response = self._client.audio.speech.create(**kwargs)

            # Convert response bytes to numpy array
            # PCM format: 16-bit signed integers, little-endian, mono, 24kHz
            audio_bytes = response.content if hasattr(response, "content") else response.read()
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # Normalize to float32 range [-1.0, 1.0]
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            return audio_float32

        except Exception as exc:
            # Catch OpenAI-specific errors
            exc_type = type(exc).__name__

            if "RateLimitError" in exc_type:
                raise RuntimeError(
                    f"OpenAI rate limit exceeded: {exc}\n"
                    "Wait a moment and try again, or use --tts-engine tortoise"
                ) from exc
            elif "AuthenticationError" in exc_type:
                raise ValueError(
                    f"OpenAI authentication failed: {exc}\n"
                    "Check your OPENAI_API_KEY in .mic-drop.env"
                ) from exc
            elif "APIError" in exc_type or "OpenAIError" in exc_type:
                raise RuntimeError(f"OpenAI API error: {exc}") from exc
            elif "ConnectionError" in exc_type or "Timeout" in exc_type:
                raise RuntimeError(
                    f"Network error while calling OpenAI API: {exc}\n"
                    "Check your internet connection and try again"
                ) from exc
            else:
                # Generic error
                raise RuntimeError(
                    f"OpenAI TTS synthesis failed: {exc}\n"
                    f"Error type: {exc_type}"
                ) from exc
