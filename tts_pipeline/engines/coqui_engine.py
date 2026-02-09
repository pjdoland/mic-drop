"""Coqui XTTS-v2 engine implementation.

Local text-to-speech synthesis with voice cloning using the Coqui TTS library
and XTTS-v2 model. Supports 17 languages and requires a speaker audio file
for voice cloning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from tts_pipeline.base import TTSEngine
from tts_pipeline.text_processing import normalize_text, split_by_char_limit

# Optional import — set to None when not installed
try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None  # type: ignore

logger = logging.getLogger("mic-drop.coqui")

# XTTS-v2 outputs at this rate
COQUI_SAMPLE_RATE: int = 24_000

# Coqui can handle reasonably long texts, but we chunk for safety
MAX_CHARS_PER_CHUNK: int = 500


class CoquiEngine(TTSEngine):
    """Coqui XTTS-v2 engine with local voice cloning.

    Attributes:
        speaker_wav: Path to reference audio file for voice cloning (6+ seconds recommended)
        language: Language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi)
        device: Torch device (auto, cpu, cuda, mps)
    """

    def __init__(
        self,
        speaker_wav: Optional[Path] = None,
        language: str = "en",
        device: str = "auto",
    ) -> None:
        from tts_pipeline import resolve_device

        self.speaker_wav = speaker_wav
        self.language = language
        self.device: str = resolve_device(device)
        self._tts = None  # loaded on first use

    @property
    def sample_rate(self) -> int:
        """Return Coqui's output sample rate (24 kHz)."""
        return COQUI_SAMPLE_RATE

    def load(self) -> None:
        """Initialize the Coqui XTTS-v2 model.

        Called automatically on first synthesize() call if not done explicitly.

        Raises:
            ImportError: If TTS library is not installed
            ValueError: If speaker_wav is not provided or doesn't exist
        """
        if CoquiTTS is None:
            raise ImportError(
                "TTS library (Coqui) is required but not installed.\n"
                "Install it with: pip install TTS"
            )

        if not self.speaker_wav:
            raise ValueError(
                "speaker_wav is required for Coqui XTTS-v2.\n"
                "Provide a reference audio file (6+ seconds) for voice cloning."
            )

        if not self.speaker_wav.exists():
            raise FileNotFoundError(f"Speaker audio file not found: {self.speaker_wav}")

        logger.info(
            "Loading Coqui XTTS-v2 (device=%s, language=%s, speaker=%s) …",
            self.device,
            self.language,
            self.speaker_wav.name,
        )

        try:
            self._tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")

            # Move model to device
            if self.device == "cuda":
                self._tts.to("cuda")
            elif self.device == "mps":
                # Try MPS, fall back to CPU if not supported
                try:
                    self._tts.to("mps")
                except Exception as exc:
                    logger.warning(
                        "Coqui TTS failed on MPS (%s). Falling back to CPU.", exc
                    )
                    self.device = "cpu"
                    self._tts.to("cpu")
            else:
                self._tts.to("cpu")

            logger.info("Coqui XTTS-v2 ready (device=%s).", self.device)

        except Exception as exc:
            raise RuntimeError(f"Failed to load Coqui XTTS-v2: {exc}") from exc

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio with voice cloning.

        Long texts are automatically chunked at 500-character boundaries
        and concatenated.

        Args:
            text: Input text to synthesize

        Returns:
            1-D float32 numpy array at 24 kHz

        Raises:
            ValueError: If text is empty after normalization
            RuntimeError: If synthesis fails
        """
        if self._tts is None:
            self.load()

        text = normalize_text(text)
        if not text:
            raise ValueError("Input text is empty after normalization.")

        # Split into chunks if necessary
        chunks = split_by_char_limit(text, max_chars=MAX_CHARS_PER_CHUNK)
        logger.info("Synthesizing %d chunk(s) with Coqui XTTS-v2 …", len(chunks))

        from tqdm import tqdm

        segments: list[np.ndarray] = []
        for i, chunk in enumerate(
            tqdm(chunks, desc="Coqui TTS", unit="chunk", leave=True)
        ):
            logger.debug(
                "Chunk %d/%d (%d chars): %.60s …",
                i + 1,
                len(chunks),
                len(chunk),
                chunk,
            )

            try:
                # Synthesize with voice cloning
                wav = self._tts.tts(
                    text=chunk,
                    speaker_wav=str(self.speaker_wav),
                    language=self.language,
                )

                # Convert to numpy array and ensure float32
                audio_array = np.array(wav, dtype=np.float32)
                segments.append(audio_array)

            except Exception as exc:
                raise RuntimeError(
                    f"Coqui TTS synthesis failed on chunk {i+1}: {exc}"
                ) from exc

        audio = np.concatenate(segments, axis=0).astype(np.float32)
        duration = len(audio) / self.sample_rate
        logger.info("Synthesis complete: %.2f s at %d Hz.", duration, self.sample_rate)
        return audio
