"""TTS engine implementations for mic-drop.

This package contains all supported text-to-speech backends:
- TortoiseEngine: Local synthesis using tortoise-tts
- OpenAITTSEngine: API-based synthesis using OpenAI's TTS models
"""

from tts_pipeline.engines.tortoise_engine import TortoiseEngine
from tts_pipeline.engines.openai_engine import OpenAITTSEngine

__all__ = ["TortoiseEngine", "OpenAITTSEngine"]
