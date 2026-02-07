"""Configuration file loading for mic-drop.

Handles reading the .mic-drop.env file for persistent settings like
cache directories and API keys.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("mic-drop.config")

# Primary location: repo root (works for editable installs via setup.sh)
# Fallback: current working directory
_REPO_ROOT_ENV = Path(__file__).resolve().parent.parent / ".mic-drop.env"


def load_config(env_path: Path | None = None) -> dict[str, str]:
    """Read key=value pairs from a mic-drop env file.

    Search order when env_path is not given:
        1. <package-root>/.mic-drop.env   (editable install)
        2. $CWD/.mic-drop.env             (fallback)

    Lines starting with # and blank lines are skipped.

    Args:
        env_path: Optional explicit path to config file

    Returns:
        Dictionary of configuration key-value pairs
    """
    if env_path is None:
        env_path = _REPO_ROOT_ENV if _REPO_ROOT_ENV.is_file() else Path.cwd() / ".mic-drop.env"

    config: dict[str, str] = {}
    if not env_path.is_file():
        logger.debug("Config file not found: %s", env_path)
        return config

    logger.debug("Reading config from %s", env_path)
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config


def get_tortoise_cache_dir(config: dict[str, str] | None = None) -> Path | None:
    """Get Tortoise cache directory from config.

    Args:
        config: Optional pre-loaded config dict (loads from file if None)

    Returns:
        Path to cache directory, or None if not configured
    """
    if config is None:
        config = load_config()

    raw = config.get("TORTOISE_CACHE_DIR")
    return Path(raw) if raw else None


def get_openai_api_key(config: dict[str, str] | None = None) -> str | None:
    """Get OpenAI API key from config.

    Args:
        config: Optional pre-loaded config dict (loads from file if None)

    Returns:
        API key string, or None if not configured
    """
    if config is None:
        config = load_config()

    return config.get("OPENAI_API_KEY")
