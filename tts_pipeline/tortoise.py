"""Tortoise TTS integration for mic-drop.

Responsibilities
----------------
* Lazy-load the heavy ``tortoise-tts`` library and model weights.
* Resolve voice sources (built-in name, WAV directory, single clip, or random).
* Split long input text into synthesis-friendly chunks and concatenate results.
* Return raw audio as a 1-D float32 NumPy array at 24 kHz.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("mic-drop.tortoise")

# Tortoise always outputs at this rate.
TORTOISE_SAMPLE_RATE: int = 24_000

# Soft cap on words per synthesis call.  Exceeding this can cause OOM on
# modest GPUs and degrades prosody.
MAX_WORDS_PER_CHUNK: int = 150


# ---------------------------------------------------------------------------
# Text helpers (pure-Python, testable without torch)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Light normalisation: strip BOM, collapse whitespace, trim."""
    text = text.lstrip("\ufeff")          # BOM
    text = re.sub(r"\s+", " ", text)      # collapse
    return text.strip()


def _split_into_chunks(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    """Sentence-aware text chunking.

    Splits on sentence-ending punctuation (.!?) followed by whitespace.
    If a single sentence exceeds *max_words* it is hard-wrapped on word
    boundaries so nothing is silently dropped.
    """
    text = text.strip()
    if not text:
        return []

    # Split into sentences; regex keeps the punctuation with the sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    buffer: list[str] = []       # sentences accumulated for the current chunk
    buf_words: int = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # Flush buffer if appending this sentence would bust the limit.
        if buf_words + word_count > max_words and buffer:
            chunks.append(" ".join(buffer))
            buffer = []
            buf_words = 0

        if word_count > max_words:
            # Flush any remaining buffer first.
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buf_words = 0
            # Hard-wrap the oversized sentence.
            words = sentence.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
        else:
            buffer.append(sentence)
            buf_words += word_count

    if buffer:
        chunks.append(" ".join(buffer))

    return [c for c in chunks if c.strip()]


def _strip_markdown(text: str) -> str:
    """Remove common Markdown syntax, returning plain text.

    Stripped constructs (in processing order)
    ------------------------------------------
    * Fenced code blocks  — content kept, fences removed
    * Inline code
    * Images              — alt text kept
    * Links               — link text kept
    * Strikethrough
    * Bold / italic
    * ATX headers (``#`` … ``######``)
    * Horizontal rules
    * Blockquotes
    * Unordered and ordered list markers
    * HTML tags
    """
    # -- block-level (multi-line, must come first) --------------------------
    text = re.sub(r"```[^\n]*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)

    # -- inline --------------------------------------------------------------
    text = re.sub(r"`([^`]+)`", r"\1", text)                          # code
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)             # image
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)              # link
    text = re.sub(r"~~(.+?)~~", r"\1", text)                          # strike
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                      # bold **
    text = re.sub(r"__(.+?)__", r"\1", text)                          # bold __
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)  # italic *
    text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", text)    # italic _

    # -- line-level ----------------------------------------------------------
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)        # headers
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)    # hr
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)             # blockquote
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)         # ul marker
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)         # ol marker

    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    return text


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TortoiseEngine:
    """Thin, lazy wrapper around ``tortoise-tts``.

    Attributes:
        preset: One of ``ultra_fast``, ``fast``, ``standard``, ``high_quality``.
        device: Torch device string resolved at construction time.
        voice: Built-in voice name, path to WAV clip(s), or ``None`` for random.
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

    # -- public -------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return TORTOISE_SAMPLE_RATE

    def load(self) -> None:
        """Explicitly initialise the Tortoise model.  Called automatically on
        the first call to :meth:`synthesize` if skipped.

        MPS fallback
        ------------
        Tortoise's diffusion sampler uses ops that are not yet fully
        supported on Apple Silicon's MPS backend.  When ``device`` is
        ``"mps"`` and the model raises a ``RuntimeError`` or
        ``NotImplementedError`` during loading, this method logs a warning
        and transparently retries on CPU.
        """
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
        """Synthesize *text* → 1-D float32 NumPy array @ 24 kHz.

        Long texts are automatically chunked; the resulting audio segments
        are concatenated in order.
        """
        if self._tts is None:
            self.load()

        text = _normalize_text(text)
        if not text:
            raise ValueError("Input text is empty after normalization.")

        voice_samples, conditioning_latents = self._resolve_voices()
        chunks = _split_into_chunks(text)
        logger.info("Synthesizing %d chunk(s) …", len(chunks))

        from tqdm import tqdm

        segments: list[np.ndarray] = []
        for i, chunk in enumerate(
            tqdm(chunks, desc="Tortoise", unit="chunk", leave=True)
        ):
            logger.debug("Chunk %d/%d (%d words): %.60s …", i + 1, len(chunks), len(chunk.split()), chunk)

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

    # -- private ------------------------------------------------------------

    def _resolve_voices(self) -> tuple:
        """Return ``(voice_samples, conditioning_latents)`` for Tortoise.

        Resolution order
        ----------------
        1. ``None``        → random conditioning latents
        2. Existing dir    → load all ``*.wav`` clips from it
        3. Existing .wav   → single clip
        4. String          → attempt to use as a built-in voice name
        """
        import torch
        from tortoise.utils.audio import load_audio

        if self.voice is None:
            logger.info("No voice specified — using random conditioning latents.")
            return None, None

        voice_path = Path(self.voice)

        # --- directory of WAV clips ---
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

        # --- single WAV file ---
        if voice_path.is_file() and voice_path.suffix.lower() == ".wav":
            logger.info("Loading single voice clip: %s", voice_path)
            clip = load_audio(str(voice_path), TORTOISE_SAMPLE_RATE).unsqueeze(0)
            cond_latents = self._tts.get_conditioning_latents(clip)
            return clip, cond_latents

        # --- built-in voice name (falls back to tts.load_voice) ---
        logger.info("Attempting built-in Tortoise voice: %s", self.voice)
        return self._tts.load_voice(self.voice)
