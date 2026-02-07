"""Tortoise TTS engine implementation.

Local text-to-speech synthesis using the tortoise-tts library with
lazy model loading and voice conditioning support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from tts_pipeline.base import TTSEngine
from tts_pipeline.text_processing import normalize_text, split_into_chunks

logger = logging.getLogger("mic-drop.tortoise")

# Tortoise always outputs at this rate
TORTOISE_SAMPLE_RATE: int = 24_000

# Soft cap on words per synthesis call
# Exceeding this can cause OOM on modest GPUs and degrades prosody
MAX_WORDS_PER_CHUNK: int = 150


class TortoiseEngine(TTSEngine):
    """Tortoise TTS engine with lazy loading and voice conditioning.

    Attributes:
        preset: Quality preset (ultra_fast, fast, standard, high_quality)
        device: Torch device string (auto, cpu, cuda, mps)
        voice: Built-in voice name, path to WAV clip(s), or None for random
        cache_dir: Directory to store model weights (~2-4 GB)
    """

    def __init__(
        self,
        preset: str = "standard",
        device: str = "auto",
        voice: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        from tts_pipeline import resolve_device

        self.device: str = resolve_device(device)
        self.preset = preset
        self.voice = voice
        self.cache_dir = cache_dir
        self._tts = None  # loaded on first use

    @property
    def sample_rate(self) -> int:
        """Return Tortoise's output sample rate (24 kHz)."""
        return TORTOISE_SAMPLE_RATE

    def load(self) -> None:
        """Initialize the Tortoise model.

        Called automatically on first synthesize() call if not done explicitly.

        MPS fallback:
            Tortoise's diffusion sampler uses ops not fully supported on Apple
            Silicon's MPS backend. When device is 'mps' and loading raises
            RuntimeError or NotImplementedError, automatically falls back to CPU.
        """
        import os
        from tortoise.api import TextToSpeech  # heavy — deferred

        logger.info(
            "Loading Tortoise TTS (device=%s, preset=%s) …",
            self.device,
            self.preset,
        )
        tt_kwargs: dict = {"device": self.device}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tt_kwargs["models_dir"] = str(self.cache_dir)
            # Set HuggingFace environment variables to use this cache
            os.environ["HF_HOME"] = str(self.cache_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.cache_dir)
            logger.info("Tortoise model cache → %s", self.cache_dir)

        try:
            self._tts = TextToSpeech(**tt_kwargs)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device == "mps":
                logger.warning(
                    "Tortoise TTS failed on MPS (%s). "
                    "Falling back to CPU automatically.",
                    exc,
                )
                self.device = "cpu"
                tt_kwargs["device"] = "cpu"
                self._tts = TextToSpeech(**tt_kwargs)
            else:
                raise
        logger.info("Tortoise TTS ready (device=%s).", self.device)

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio.

        Long texts are automatically chunked and concatenated.

        Args:
            text: Input text to synthesize

        Returns:
            1-D float32 numpy array at 24 kHz

        Raises:
            ValueError: If text is empty after normalization
        """
        if self._tts is None:
            self.load()

        text = normalize_text(text)
        if not text:
            raise ValueError("Input text is empty after normalization.")

        voice_samples, conditioning_latents = self._resolve_voices()
        chunks = split_into_chunks(text, max_words=MAX_WORDS_PER_CHUNK)
        logger.info("Synthesizing %d chunk(s) …", len(chunks))

        from tqdm import tqdm

        segments: list[np.ndarray] = []
        for i, chunk in enumerate(
            tqdm(chunks, desc="Tortoise", unit="chunk", leave=True)
        ):
            logger.debug(
                "Chunk %d/%d (%d words): %.60s …",
                i + 1,
                len(chunks),
                len(chunk.split()),
                chunk,
            )

            audio_tensor = self._tts.tts_with_preset(
                chunk,
                preset=self.preset,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
            )
            # Tortoise returns shape (1, samples) or (samples,)
            segments.append(audio_tensor.squeeze().cpu().numpy())

        audio = np.concatenate(segments, axis=0).astype(np.float32)
        duration = len(audio) / self.sample_rate
        logger.info("Synthesis complete: %.2f s at %d Hz.", duration, self.sample_rate)
        return audio

    def _resolve_voices(self) -> tuple:
        """Return (voice_samples, conditioning_latents) for Tortoise.

        Resolution order:
            1. None → random conditioning latents
            2. Existing dir → load all *.wav clips from it
            3. Existing .wav → single clip
            4. String → attempt to use as a built-in voice name

        Returns:
            Tuple of (voice_samples, conditioning_latents) for Tortoise API
        """
        import torch
        from tortoise.utils.audio import load_audio, load_voice

        if self.voice is None:
            logger.info("No voice specified — using random conditioning latents.")
            return None, None

        voice_path = Path(self.voice)

        # Directory of WAV clips
        if voice_path.is_dir():
            wavs = sorted(voice_path.glob("*.wav"))
            if not wavs:
                raise FileNotFoundError(
                    f"No .wav files found in voice directory: {voice_path}"
                )
            logger.info("Loading %d voice clip(s) from %s", len(wavs), voice_path)
            clips = torch.stack([load_audio(str(w), TORTOISE_SAMPLE_RATE) for w in wavs])
            cond_latents = self._tts.get_conditioning_latents(clips)
            return clips, cond_latents

        # Single WAV file
        if voice_path.is_file() and voice_path.suffix.lower() == ".wav":
            logger.info("Loading single voice clip: %s", voice_path)
            clip = load_audio(str(voice_path), TORTOISE_SAMPLE_RATE).unsqueeze(0)
            cond_latents = self._tts.get_conditioning_latents(clip)
            return clip, cond_latents

        # Built-in voice name
        logger.info("Attempting built-in Tortoise voice: %s", self.voice)
        clips, latents = load_voice(self.voice)

        # If we got clips but no pre-computed latents, compute them now
        if clips is not None and latents is None:
            latents = self._tts.get_conditioning_latents(clips)

        return clips, latents
