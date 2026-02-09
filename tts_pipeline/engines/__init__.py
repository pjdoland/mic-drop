"""TTS engine implementations for mic-drop.

This package contains all supported text-to-speech backends:
- TortoiseEngine: Local synthesis using tortoise-tts
- OpenAITTSEngine: API-based synthesis using OpenAI's TTS models
- CoquiEngine: Local synthesis with voice cloning using Coqui XTTS-v2
"""

from tts_pipeline.engines.tortoise_engine import TortoiseEngine
from tts_pipeline.engines.openai_engine import OpenAITTSEngine
from tts_pipeline.engines.coqui_engine import CoquiEngine

__all__ = ["TortoiseEngine", "OpenAITTSEngine", "CoquiEngine"]
