"""RVC (Retrieval-based Voice Conversion) integration for mic-drop.

Responsibilities
----------------
* Load and validate an RVC ``.pth`` model (and optional ``.index``).
* Run voice conversion via ``rvc-python``'s underlying ``vc_single`` method.
* Return converted audio as a 1-D float32 array together with the
  output sample rate (determined by the model, typically 40 kHz).

Backend notes
-------------
The backend is **rvc-python** (``pip install rvc-python``).  The
``RVCInference`` class wraps an internal ``VC`` module that exposes
``vc_single(input_audio_path, ...)`` — file input, numpy array output.

Device strings are plain: ``"cpu"``, ``"mps"``, ``"cuda"`` (no ``:0`` suffix).

Pre-import patches
------------------
Two environment/monkey-patches are applied before importing rvc-python:

1. ``OMP_NUM_THREADS=1`` — prevents faiss/PyTorch OpenMP conflicts on macOS.
2. ``torch.load`` monkey-patch — forces ``weights_only=False`` for fairseq
   model loading (PyTorch 2.6+ changed the default to True, breaking fairseq).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("mic-drop.rvc")

# ---------------------------------------------------------------------------
# Pre-import patches (applied once at module load time)
# ---------------------------------------------------------------------------

# Patch 1: Prevent faiss/PyTorch OpenMP conflict on macOS
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
    logger.debug("Set OMP_NUM_THREADS=1 to prevent faiss/PyTorch OpenMP conflict")

# Patch 2: Force torch.load weights_only=False for fairseq compatibility
# PyTorch 2.6+ changed default from False → True; fairseq models break.
# This must run BEFORE rvc-python is imported anywhere in the process.
try:
    import torch

    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    logger.debug("Applied torch.load patch for fairseq compatibility")
except Exception as exc:
    logger.warning("Failed to patch torch.load: %s", exc)


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
        self._rvc = None  # loaded on demand

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
            Converted audio (float32, 1-D) and its sample rate (set by the
            RVC model, typically 16 000).
        """
        if self._rvc is None:
            self.load()

        logger.info(
            "Running RVC conversion (pitch=%+d semitones, method=%s) …",
            self.pitch,
            self.method,
        )

        converted, out_sr = self._convert_audio(audio, input_sample_rate)
        logger.info("RVC conversion complete.")
        return converted, out_sr

    # -- private ------------------------------------------------------------

    def _load_rvc_python(self) -> None:
        """Initialise RVCInference + load the voice model.

        MPS fallback
        ------------
        If ``device`` is ``"mps"`` and the backend raises a
        ``RuntimeError`` or ``NotImplementedError``, the engine logs a
        warning and retries on CPU automatically.
        """
        from rvc_python.infer import RVCInference  # type: ignore[import-untyped]

        try:
            self._rvc = RVCInference(device=self.device)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device == "mps":
                logger.warning(
                    "RVC failed on MPS (%s). Falling back to CPU.",
                    exc,
                )
                self.device = "cpu"
                self._rvc = RVCInference(device="cpu")
            else:
                raise

        index = str(self.index_path) if self.index_path and self.index_path.exists() else ""
        if index:
            logger.info("Using index file: %s", self.index_path)

        self._rvc.load_model(str(self.model_path), version="v2", index_path=index)
        self._rvc.set_params(f0up_key=self.pitch, f0method=self.method)

    def _convert_audio(self, audio: np.ndarray, input_sample_rate: int) -> tuple[np.ndarray, int]:
        """Write *audio* to temp WAV, call vc_single, normalize output.

        vc_single takes a file path and returns a numpy array in int16
        value range (-32768…32767) as float32 dtype.  We normalize to
        proper float32 range (-1.0…1.0) and read the output sample rate
        from the VC module's tgt_sr attribute.
        """
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            input_wav = tmp.name

        try:
            # Write input (16-bit PCM, clipped to [-1, 1])
            sf.write(input_wav, np.clip(audio, -1.0, 1.0), input_sample_rate, subtype="PCM_16")

            # Call underlying vc_single (returns numpy array or tuple)
            index = str(self.index_path) if self.index_path and self.index_path.exists() else ""
            result = self._rvc.vc.vc_single(
                sid=0,
                input_audio_path=input_wav,
                f0_up_key=self.pitch,
                f0_file=None,
                f0_method=self.method,
                file_index=index,
                file_index2="",
                index_rate=0.5,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=0.25,
                protect=0.33,
            )

            # Parse result — can be direct array or tuple
            if result is None:
                raise RuntimeError("RVC conversion returned None (internal error)")
            elif isinstance(result, np.ndarray):
                converted = result
            elif isinstance(result, tuple) and len(result) == 2:
                _message, audio_result = result
                if audio_result is None:
                    raise RuntimeError(f"RVC conversion failed: {_message}")
                _sr, converted = audio_result
            else:
                raise RuntimeError(f"Unexpected vc_single result type: {type(result)}")

            # Normalize int16-range values to float32 [-1.0, 1.0]
            converted = converted.astype(np.float32) / 32768.0
            out_sr = self._rvc.vc.tgt_sr

            return converted, out_sr

        finally:
            Path(input_wav).unlink(missing_ok=True)

