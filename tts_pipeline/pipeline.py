"""Main processing pipeline for mic-drop.

Orchestrates the full text → speech → voice-conversion → WAV flow,
coordinating :mod:`~tts_pipeline.tortoise` and :mod:`~tts_pipeline.rvc`
and handling the final resample / normalise / export steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from tts_pipeline.audio import peak_normalize as _peak_normalize, resample as _resample

logger = logging.getLogger("mic-drop.pipeline")


class Pipeline:
    """End-to-end TTS + RVC pipeline.

    Parameters
    ----------
    voice_model_path:
        Path to the RVC ``.pth`` model.
    rvc_index_path:
        Optional companion ``.index`` file for the RVC model.
    tortoise_preset:
        Tortoise quality preset (``ultra_fast`` … ``high_quality``).
    tortoise_voice:
        Built-in voice name, WAV clip path/directory, or ``None`` (random).
    rvc_pitch:
        Semitone pitch shift for RVC.
    rvc_method:
        Pitch-extraction method (``rmvpe``, ``pm``, ``crepe``).
    output_sample_rate:
        Target Hz for the final WAV file.
    device:
        Torch device (``auto``, ``cpu``, ``cuda``, ``mps``).
    cache_dir:
        Directory where Tortoise downloads its model weights (~2–4 GB).
        Useful when running from a USB drive.  ``None`` uses the Tortoise
        default (``~/.cache/tortoise-tts``).
    """

    def __init__(
        self,
        voice_model_path: Path,
        rvc_index_path: Optional[Path] = None,
        tortoise_preset: str = "standard",
        tortoise_voice: Optional[str] = None,
        rvc_pitch: int = 0,
        rvc_method: str = "rmvpe",
        output_sample_rate: int = 44_100,
        device: str = "auto",
        cache_dir: Optional[Path] = None,
    ) -> None:
        from tts_pipeline.tortoise import TortoiseEngine
        from tts_pipeline.rvc import RVCEngine

        self.output_sample_rate = output_sample_rate

        self.tortoise = TortoiseEngine(
            preset=tortoise_preset,
            device=device,
            voice=tortoise_voice,
            cache_dir=cache_dir,
        )
        self.rvc = RVCEngine(
            model_path=voice_model_path,
            index_path=rvc_index_path,
            pitch=rvc_pitch,
            method=rvc_method,
            device=device,
        )

    # -- public -------------------------------------------------------------

    def run(self, text: str, output_path: Path) -> None:
        """Full pipeline: *text* → WAV file at *output_path*.

        Stages
        ------
        1. Tortoise TTS   – text → raw speech (24 kHz)
        2. RVC            – raw speech → cloned voice (16 kHz)
        3. Post-process   – resample to target rate, peak-normalise, write WAV
        """
        logger.info("=== mic-drop pipeline started ===")

        # Stage 1 – synthesis
        logger.info("Stage 1/3: Tortoise TTS synthesis")
        raw_audio: np.ndarray = self.tortoise.synthesize(text)
        raw_sr: int = self.tortoise.sample_rate

        # Stage 2 – voice conversion
        logger.info("Stage 2/3: RVC voice conversion")
        converted_audio, converted_sr = self.rvc.convert(raw_audio, raw_sr)

        # Stage 3 – post-processing & export
        logger.info("Stage 3/3: Resample → normalise → export")
        final = _resample(converted_audio, converted_sr, self.output_sample_rate)
        final = _peak_normalize(final)
        _write_wav(final, self.output_sample_rate, output_path)

        duration = len(final) / self.output_sample_rate
        logger.info(
            "=== Done — %.2f s, %d Hz → %s ===",
            duration,
            self.output_sample_rate,
            output_path,
        )


def _write_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Write 16-bit PCM WAV via soundfile."""
    import soundfile as sf

    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    logger.info("Wrote %s (%.2f s)", path, len(audio) / sample_rate)
