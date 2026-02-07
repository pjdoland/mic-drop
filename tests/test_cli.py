"""Unit tests for CLI helpers — config loading, validation, and parser flags.

Dependency-free: no torch, tortoise, or rvc required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tts_pipeline.cli import CliError, _apply_config_defaults, build_parser
from tts_pipeline.config import load_config as _load_config


# ---------------------------------------------------------------------------
# CliError
# ---------------------------------------------------------------------------


class TestCliError:
    def test_message_and_default_code(self):
        exc = CliError("something broke")
        assert str(exc) == "something broke"
        assert exc.code == 1

    def test_custom_exit_code(self):
        exc = CliError("not found", code=3)
        assert exc.code == 3

    def test_is_an_exception(self):
        with pytest.raises(CliError):
            raise CliError("boom")


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert _load_config(env_path=tmp_path / "absent.env") == {}

    def test_parses_single_key(self, tmp_path):
        env = tmp_path / ".mic-drop.env"
        env.write_text("TORTOISE_CACHE_DIR=/mnt/usb\n", encoding="utf-8")
        assert _load_config(env_path=env) == {"TORTOISE_CACHE_DIR": "/mnt/usb"}

    def test_parses_multiple_keys(self, tmp_path):
        env = tmp_path / ".mic-drop.env"
        env.write_text("A=1\nB=2\n", encoding="utf-8")
        assert _load_config(env_path=env) == {"A": "1", "B": "2"}

    def test_skips_comments(self, tmp_path):
        env = tmp_path / ".mic-drop.env"
        env.write_text("# comment\nKEY=val\n", encoding="utf-8")
        assert _load_config(env_path=env) == {"KEY": "val"}

    def test_skips_blank_lines(self, tmp_path):
        env = tmp_path / ".mic-drop.env"
        env.write_text("\n  \nKEY=val\n\n", encoding="utf-8")
        assert _load_config(env_path=env) == {"KEY": "val"}

    def test_value_containing_equals(self, tmp_path):
        """Only the first ``=`` is the key/value delimiter."""
        env = tmp_path / ".mic-drop.env"
        env.write_text("X=a=b=c\n", encoding="utf-8")
        assert _load_config(env_path=env) == {"X": "a=b=c"}

    def test_strips_whitespace_around_key_and_value(self, tmp_path):
        env = tmp_path / ".mic-drop.env"
        env.write_text("  KEY  =  value  \n", encoding="utf-8")
        assert _load_config(env_path=env) == {"KEY": "value"}

    def test_empty_value_is_empty_string(self, tmp_path):
        env = tmp_path / ".mic-drop.env"
        env.write_text("EMPTY=\n", encoding="utf-8")
        assert _load_config(env_path=env) == {"EMPTY": ""}


# ---------------------------------------------------------------------------
# _apply_config_defaults
# ---------------------------------------------------------------------------


class TestApplyConfigDefaults:
    def test_explicit_flag_wins(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("TORTOISE_CACHE_DIR=/env/path\n", encoding="utf-8")
        args = argparse.Namespace(cache_dir=Path("/cli/path"))
        _apply_config_defaults(args, env_path=env)
        assert args.cache_dir == Path("/cli/path")

    def test_fills_from_env_when_none(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("TORTOISE_CACHE_DIR=/env/path\n", encoding="utf-8")
        args = argparse.Namespace(cache_dir=None)
        _apply_config_defaults(args, env_path=env)
        assert args.cache_dir == Path("/env/path")

    def test_missing_env_file_leaves_none(self, tmp_path):
        args = argparse.Namespace(cache_dir=None)
        _apply_config_defaults(args, env_path=tmp_path / "nope")
        assert args.cache_dir is None

    def test_blank_value_in_env_leaves_none(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("TORTOISE_CACHE_DIR=\n", encoding="utf-8")
        args = argparse.Namespace(cache_dir=None)
        _apply_config_defaults(args, env_path=env)
        assert args.cache_dir is None


# ---------------------------------------------------------------------------
# Parser flags
# ---------------------------------------------------------------------------


class TestParserFlags:
    def test_version_exits_0(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(["--version"])
        assert exc_info.value.code == 0
        assert "mic-drop" in capsys.readouterr().out

    def test_quiet_and_verbose_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(
                ["-v", "-q", "-o", "out.wav", "-m", "model.pth"]
            )
        assert exc_info.value.code == 2  # argparse usage error

    def test_strip_markdown_absent_is_none(self):
        args = build_parser().parse_args(["-o", "out.wav", "-m", "m.pth"])
        assert args.strip_markdown is None

    def test_strip_markdown_present_is_true(self):
        args = build_parser().parse_args(
            ["--strip-markdown", "-o", "out.wav", "-m", "m.pth"]
        )
        assert args.strip_markdown is True

    def test_quiet_flag_parsed(self):
        args = build_parser().parse_args(["-q", "-o", "out.wav", "-m", "m.pth"])
        assert args.quiet is True
        assert args.verbose is False

    def test_verbose_flag_parsed(self):
        args = build_parser().parse_args(["-v", "-o", "out.wav", "-m", "m.pth"])
        assert args.verbose is True
        assert args.quiet is False

    def test_defaults_neither_verbose_nor_quiet(self):
        args = build_parser().parse_args(["-o", "out.wav", "-m", "m.pth"])
        assert args.verbose is False
        assert args.quiet is False
