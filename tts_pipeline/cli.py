"""Command-line interface for mic-drop.

Handles argument parsing, input reading (file or stdin), and dispatches
to the processing pipeline for single-file or batch operation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tts_pipeline import __version__

logger = logging.getLogger("mic-drop")


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mic-drop",
        description="Local voice-cloning TTS: Tortoise TTS + RVC in one pipeline.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mic-drop {__version__}"
    )

    # -- I/O ----------------------------------------------------------------
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input text file. Omit to read from stdin.",
    )
    io_group.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output WAV file (single mode) or directory (batch mode).",
    )
    io_group.add_argument(
        "--strip-markdown",
        action="store_true",
        default=None,
        help=(
            "Strip Markdown formatting before synthesis. "
            "Automatic for .md files; use this flag when piping "
            "Markdown via stdin."
        ),
    )

    # -- Voice model ---------------------------------------------------------
    model_group = parser.add_argument_group("Voice Model")
    model_group.add_argument(
        "-m",
        "--voice-model",
        type=Path,
        required=True,
        help="Path to the RVC voice model (.pth).",
    )
    model_group.add_argument(
        "--rvc-index",
        type=Path,
        default=None,
        help="Path to the companion RVC index file (.index).",
    )

    # -- Tortoise options ----------------------------------------------------
    tt_group = parser.add_argument_group("Tortoise TTS")
    tt_group.add_argument(
        "--tortoise-preset",
        choices=["ultra_fast", "fast", "standard", "high_quality"],
        default="standard",
        help="Tortoise quality preset (default: standard).",
    )
    tt_group.add_argument(
        "--tortoise-voice",
        type=str,
        default=None,
        help=(
            "Built-in Tortoise voice name (e.g. 'female'), "
            "or path to a directory of reference WAV clips."
        ),
    )
    tt_group.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Tortoise model-cache directory (~2–4 GB on first run). "
            "Point at a USB drive to keep large files off your main "
            "disk.  Default: ~/.cache/tortoise-tts"
        ),
    )

    # -- RVC options ---------------------------------------------------------
    rvc_group = parser.add_argument_group("RVC Conversion")
    rvc_group.add_argument(
        "--rvc-pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones (default: 0).",
    )
    rvc_group.add_argument(
        "--rvc-method",
        choices=["rmvpe", "pm", "crepe"],
        default="rmvpe",
        help="Pitch extraction method (default: rmvpe).",
    )

    # -- Audio options -------------------------------------------------------
    audio_group = parser.add_argument_group("Audio")
    audio_group.add_argument(
        "--sample-rate",
        type=int,
        choices=[16000, 22050, 44100, 48000],
        default=44100,
        help="Output sample rate in Hz (default: 44100).",
    )

    # -- Batch / misc --------------------------------------------------------
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: --input is a directory of .txt / .md files.",
    )
    misc_group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto — CUDA > MPS > CPU).",
    )
    misc_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def read_input(path: Path | None) -> str:
    """Read text from *path* or from stdin when *path* is ``None``."""
    if path is None:
        if sys.stdin.isatty():
            logger.error("No --input file and stdin is a terminal.")
            sys.exit(1)
        return sys.stdin.read()

    if not path.exists():
        logger.error("Input file not found: %s", path)
        sys.exit(1)

    return path.read_text(encoding="utf-8")


def _maybe_strip_md(text: str, path: Path | None, flag: bool | None) -> str:
    """Strip Markdown when *flag* is ``True`` or *path* ends in ``.md``."""
    should_strip = flag if flag is not None else (
        path is not None and path.suffix.lower() == ".md"
    )
    if should_strip:
        from tts_pipeline.tortoise import _strip_markdown

        logger.info("Stripping Markdown formatting from input.")
        text = _strip_markdown(text)
    return text


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Pre-flight validation ------------------------------------------
    if not args.voice_model.exists():
        logger.error("Voice model not found: %s", args.voice_model)
        sys.exit(1)

    if args.rvc_index is not None and not args.rvc_index.exists():
        logger.error("RVC index file not found: %s", args.rvc_index)
        sys.exit(1)

    # Ensure the parent directory of the output exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # -- Dispatch -------------------------------------------------------
    if args.batch:
        _run_batch(args)
    else:
        text = read_input(args.input)
        text = _maybe_strip_md(text, args.input, args.strip_markdown)
        _run_single(text, args)


# ---------------------------------------------------------------------------
# Single-file and batch runners
# ---------------------------------------------------------------------------


def _build_pipeline(args: argparse.Namespace):  # noqa: ANN202
    """Lazy import + construct the Pipeline from parsed CLI args."""
    from tts_pipeline.pipeline import Pipeline

    return Pipeline(
        voice_model_path=args.voice_model,
        rvc_index_path=args.rvc_index,
        tortoise_preset=args.tortoise_preset,
        tortoise_voice=args.tortoise_voice,
        rvc_pitch=args.rvc_pitch,
        rvc_method=args.rvc_method,
        output_sample_rate=args.sample_rate,
        device=args.device,
        cache_dir=args.cache_dir,
    )


def _run_single(text: str, args: argparse.Namespace) -> None:
    pipeline = _build_pipeline(args)
    pipeline.run(text=text, output_path=args.output)
    logger.info("Output written to %s", args.output)


def _run_batch(args: argparse.Namespace) -> None:
    if args.input is None or not args.input.is_dir():
        logger.error("--batch requires --input to be a directory.")
        sys.exit(1)

    input_files = sorted(
        f for ext in ("*.txt", "*.md") for f in args.input.glob(ext)
    )
    if not input_files:
        logger.warning("No .txt or .md files found in %s", args.input)
        return

    pipeline = _build_pipeline(args)
    args.output.mkdir(parents=True, exist_ok=True)

    for src in input_files:
        logger.info("Batch: processing %s …", src.name)
        text = src.read_text(encoding="utf-8")
        text = _maybe_strip_md(text, src, args.strip_markdown)
        out_path = args.output / src.with_suffix(".wav").name
        pipeline.run(text=text, output_path=out_path)
        logger.info("  → %s", out_path)

    logger.info("Batch complete: %d file(s) processed.", len(input_files))
