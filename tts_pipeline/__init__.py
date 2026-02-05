"""mic-drop — local voice-cloning TTS pipeline.

Chains Tortoise TTS (text → speech) with RVC (speech → cloned voice)
into a single command-line tool.  Everything runs locally.
"""

import os as _os

# Apple Silicon: many MPS (Metal) ops are not yet implemented.  This env var
# makes PyTorch silently fall back to CPU for those ops instead of crashing.
# Must be set before the first ``import torch``.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"


def resolve_device(requested: str = "auto") -> str:
    """Return the best available torch device string.

    Priority when *requested* is ``"auto"``::

        cuda  →  mps  →  cpu

    Apple Silicon Macs expose Metal via ``torch.backends.mps``.
    """
    if requested != "auto":
        return requested

    import torch  # gate the heavy import

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
