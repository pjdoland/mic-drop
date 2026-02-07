"""Abstract base class for TTS engines.

Defines the interface that all TTS backends must implement for use
in the mic-drop pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TTSEngine(ABC):
    """Abstract base class for text-to-speech engines.

    All TTS backends (Tortoise, OpenAI, etc.) must implement this interface
    to be compatible with the pipeline.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the output sample rate in Hz.

        Returns:
            Sample rate (e.g., 24000 for 24 kHz)
        """
        ...

    @abstractmethod
    def load(self) -> None:
        """Initialize the TTS engine (load models, connect to API, etc.).

        This is called automatically on first use if not explicitly called.
        Raises appropriate exceptions for missing dependencies or configuration.
        """
        ...

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to speech audio.

        Args:
            text: Input text to synthesize

        Returns:
            1-D float32 numpy array containing audio samples in range [-1.0, 1.0]
            at the engine's sample_rate

        Raises:
            ValueError: For empty or invalid text
            RuntimeError: For synthesis failures
        """
        ...
