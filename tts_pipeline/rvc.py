"""RVC (Retrieval-based Voice Conversion) integration for mic-drop.

Responsibilities
----------------
* Load and validate an RVC ``.pth`` model (and optional ``.index``).
* Resample incoming audio to RVC's expected 16 kHz input rate.
* Run voice conversion, returning audio at 16 kHz.
* Abstract over the ``rvc-python`` package while documenting a clean
  fallback path if users need a different RVC backend.

Backend notes
-------------
The default backend is **rvc-python** (``pip install rvc-python``).  Its
``RVC`` class expects:

    RVC(model_path, index_path=None, device="cpu"|"cuda"|"mps")

with an ``.infer(audio, input_sample_rate, pitch_shift, method)`` call that
returns a 1-D NumPy array at 16 kHz.

If you use a different RVC fork, subclass :class:`RVCEngine` and override
:meth:`_convert_audio`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from tts_pipeline.audio import resample as _resample

logger = logging.getLogger("mic-drop.rvc")

# RVC models operate internally at 16 kHz.
RVC_SAMPLE_RATE: int = 16_000


class RVCEngine:
    """Lazy-loading RVC voice-conversion engine.

    Parameters
    ----------
    model_path:
        Path to the ``.pth`` model file.
    index_path:
        Optional path to the companion ``.index`` file.  Improves
        conversion quality when present.
    pitch:
        Pitch shift in semitones applied during conversion.
    method:
        Pitch-extraction algorithm: ``rmvpe`` (default), ``pm``, or ``crepe``.
    device:
        Torch device — ``"auto"`` resolves via cuda → mps → cpu.
    """

    def __init__(
        self,
        model_path: Path,
        index_path: Optional[Path] = None,
        pitch: int = 0,
        method: str = "rmvpe",
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.index_path = index_path
        self.pitch = pitch
        self.method = method

        from tts_pipeline import resolve_device

        self.device: str = resolve_device(device)
        self._vc = None  # loaded on demand

    @property
    def sample_rate(self) -> int:
        return RVC_SAMPLE_RATE

    # -- public -------------------------------------------------------------

    def load(self) -> None:
        """Initialise the RVC model.  Called automatically by :meth:`convert`."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"RVC model not found: {self.model_path}")

        logger.info("Loading RVC model: %s (device=%s) …", self.model_path, self.device)

        try:
            self._load_rvc_python()
        except ImportError as exc:
            raise ImportError(
                "rvc-python is required but not installed.\n"
                "Install it with:  pip install rvc-python"
            ) from exc

        logger.info("RVC model loaded.")

    def convert(self, audio: np.ndarray, input_sample_rate: int) -> tuple[np.ndarray, int]:
        """Voice-convert *audio* and return ``(converted, sample_rate)``.

        Parameters
        ----------
        audio:
            1-D float32 array at *input_sample_rate*.
        input_sample_rate:
            Hz of the incoming audio (e.g. 24 000 from Tortoise).

        Returns
        -------
        tuple[np.ndarray, int]
            Converted audio (float32, 1-D) and its sample rate (16 000).
        """
        if self._vc is None:
            self.load()

        # Resample → 16 kHz for RVC
        resampled = _resample(audio, input_sample_rate, RVC_SAMPLE_RATE)

        logger.info(
            "Running RVC conversion (pitch=%+d semitones, method=%s) …",
            self.pitch,
            self.method,
        )

        converted = self._convert_audio(resampled)
        logger.info("RVC conversion complete.")
        return converted, RVC_SAMPLE_RATE

    # -- private ------------------------------------------------------------

    def _load_rvc_python(self) -> None:
        """Attempt to load via the rvc-python package.

        MPS fallback
        ------------
        If ``device`` is ``"mps"`` and the backend raises a
        ``RuntimeError`` or ``NotImplementedError``, the engine logs a
        warning and retries on CPU automatically.
        """
        from rvc import RVC as _RVC  # type: ignore[import-untyped]

        kwargs: dict = {
            "model_path": str(self.model_path),
            "device": self.device,
        }
        if self.index_path is not None and self.index_path.exists():
            kwargs["index_path"] = str(self.index_path)
            logger.info("Using index file: %s", self.index_path)

        try:
            self._vc = _RVC(**kwargs)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device == "mps":
                logger.warning(
                    "RVC failed on MPS (%s). Falling back to CPU.",
                    exc,
                )
                self.device = "cpu"
                kwargs["device"] = "cpu"
                self._vc = _RVC(**kwargs)
            else:
                raise

    def _convert_audio(self, audio: np.ndarray) -> np.ndarray:
        """Run the actual inference call on the loaded backend."""
        result = self._vc.infer(
            input_audio=audio,
            input_sample_rate=RVC_SAMPLE_RATE,
            pitch_shift=self.pitch,
            method=self.method,
        )
        return np.asarray(result, dtype=np.float32)

