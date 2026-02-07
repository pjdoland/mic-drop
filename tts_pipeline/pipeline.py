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
    tts_engine:
        TTS backend to use: ``tortoise`` (default, local) or ``openai`` (API-based).
    tortoise_preset:
        Tortoise quality preset (``ultra_fast`` … ``high_quality``).
        Ignored when using OpenAI TTS.
    tortoise_voice:
        Built-in voice name, WAV clip path/directory, or ``None`` (random).
        Ignored when using OpenAI TTS.
    openai_model:
        OpenAI TTS model: ``gpt-4o-mini-tts`` (default, supports instructions),
        ``tts-1``, or ``tts-1-hd``. Ignored when using Tortoise.
    openai_voice:
        OpenAI voice selection: ``alloy``, ``echo``, ``fable``, ``onyx``,
        ``nova``, or ``shimmer``. Ignored when using Tortoise.
    openai_api_key:
        OpenAI API key. Required when using OpenAI TTS. Ignored when using Tortoise.
    openai_instructions:
        Optional instructions for OpenAI voice characteristics.
        Ignored when using Tortoise.
    rvc_pitch:
        Semitone pitch shift for RVC.
    rvc_method:
        Pitch-extraction method (``rmvpe``, ``pm``, ``crepe``).
    output_sample_rate:
        Target Hz for the final WAV file.
    device:
        Torch device (``auto``, ``cpu``, ``cuda``, ``mps``).
        Ignored for OpenAI TTS (API-based, no local compute).
    cache_dir:
        Directory where Tortoise downloads its model weights (~2–4 GB).
        Useful when running from a USB drive.  ``None`` uses the Tortoise
        default (``~/.cache/tortoise-tts``). Not applicable for OpenAI TTS.
    save_intermediate:
        If ``True``, save the pre-RVC TTS output alongside the
        final output file (with ``_pre_rvc`` suffix).
    """

    def __init__(
        self,
        voice_model_path: Path,
        rvc_index_path: Optional[Path] = None,
        tts_engine: str = "tortoise",
        # Tortoise-specific parameters
        tortoise_preset: str = "standard",
        tortoise_voice: Optional[str] = None,
        # OpenAI-specific parameters
        openai_model: str = "gpt-4o-mini-tts",
        openai_voice: str = "alloy",
        openai_api_key: Optional[str] = None,
        openai_instructions: Optional[str] = None,
        # Common parameters
        rvc_pitch: int = 0,
        rvc_method: str = "rmvpe",
        output_sample_rate: int = 44_100,
        device: str = "auto",
        cache_dir: Optional[Path] = None,
        save_intermediate: bool = False,
    ) -> None:
        from tts_pipeline.rvc import RVCEngine

        self.output_sample_rate = output_sample_rate
        self.save_intermediate = save_intermediate
        self.tts_engine_name = tts_engine

        # Initialize appropriate TTS engine
        if tts_engine == "tortoise":
            from tts_pipeline.tortoise import TortoiseEngine
            self.tts = TortoiseEngine(
                preset=tortoise_preset,
                device=device,
                voice=tortoise_voice,
                cache_dir=cache_dir,
            )
        elif tts_engine == "openai":
            from tts_pipeline.openai_tts import OpenAITTSEngine
            self.tts = OpenAITTSEngine(
                model=openai_model,
                voice=openai_voice,
                api_key=openai_api_key,
                instructions=openai_instructions,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown TTS engine: {tts_engine}\n"
                f"Supported engines: 'tortoise', 'openai'"
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
        1. TTS (Tortoise or OpenAI)   – text → raw speech (24 kHz)
        2. RVC                        – raw speech → cloned voice (16 kHz)
        3. Post-process               – resample to target rate, peak-normalise, write WAV
        """
        logger.info("=== mic-drop pipeline started ===")

        # Stage 1 – synthesis (engine-agnostic)
        logger.info("Stage 1/3: %s TTS synthesis", self.tts_engine_name.title())
        raw_audio: np.ndarray = self.tts.synthesize(text)
        raw_sr: int = self.tts.sample_rate

        # Optionally save pre-RVC intermediate audio
        if self.save_intermediate:
            intermediate_path = output_path.parent / f"{output_path.stem}_pre_rvc{output_path.suffix}"
            logger.info(
                "Saving intermediate %s output → %s",
                self.tts_engine_name.title(),
                intermediate_path,
            )
            intermediate = _peak_normalize(raw_audio)
            _write_wav(intermediate, raw_sr, intermediate_path)

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
