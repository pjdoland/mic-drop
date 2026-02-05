"""Shared audio helpers for mic-drop.

These live here (rather than in ``pipeline`` or ``rvc``) so that both
modules can use them without a circular import.
"""

from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample a 1-D float32 array via torchaudio.

    Returns *audio* unchanged when *from_sr* == *to_sr*.
    """
    if from_sr == to_sr:
        return audio

    import torch
    import torchaudio

    tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    resampled = torchaudio.transforms.Resample(from_sr, to_sr)(tensor)
    return resampled.squeeze(0).numpy()


def peak_normalize(audio: np.ndarray, target: float = 0.9) -> np.ndarray:
    """Peak-normalise so the loudest sample sits at *target* (0–1 scale).

    Silent audio (all zeros) is returned unchanged to avoid division by zero.
    """
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (target / peak)
    return audio
