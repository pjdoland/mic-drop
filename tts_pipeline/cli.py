"""Command-line interface for mic-drop.

Handles argument parsing, input reading (file or stdin), and dispatches
to the processing pipeline for single-file or batch operation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

from tts_pipeline import __version__

logger = logging.getLogger("mic-drop")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CliError(Exception):
    """User-facing error.  *code* becomes the process exit status."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def _apply_config_defaults(args: argparse.Namespace, env_path: Path | None = None) -> None:
    """Back-fill CLI args from the env file when the user didn't pass them."""
    from tts_pipeline.config import load_config

    config = load_config(env_path=env_path)

    # Tortoise cache directory
    if args.cache_dir is None:
        raw = config.get("TORTOISE_CACHE_DIR")
        if raw:
            args.cache_dir = Path(raw)
            logger.debug("cache_dir from config: %s", args.cache_dir)

    # OpenAI API key
    if not hasattr(args, "openai_api_key"):
        args.openai_api_key = config.get("OPENAI_API_KEY")
        if args.openai_api_key:
            logger.debug("openai_api_key loaded from config")
        elif hasattr(args, "tts_engine") and args.tts_engine == "openai":
            logger.warning(
                "OPENAI_API_KEY not found in config. "
                "OpenAI TTS will fail unless the key is set."
            )


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

_EPILOG = textwrap.dedent("""\
    examples:
      synthesise with Tortoise (default)
        mic-drop -i scripts/example.txt -o output/speech.wav -m models/myvoice.pth

      synthesise with OpenAI TTS
        mic-drop -i scripts/example.txt -o output/speech.wav -m models/myvoice.pth \\
          --tts-engine openai --openai-voice nova

      synthesise with Coqui XTTS-v2 (voice cloning, no RVC)
        mic-drop -i scripts/example.txt -o output/speech.wav \\
          --tts-engine coqui --coqui-speaker audio/myspeaker.wav --coqui-language en

      synthesise with Coqui + RVC (additional voice refinement)
        mic-drop -i scripts/example.txt -o output/speech.wav -m models/myvoice.pth \\
          --tts-engine coqui --coqui-speaker audio/myspeaker.wav --coqui-language en

      OpenAI with custom instructions and speed
        mic-drop -i scripts/example.txt -o output/speech.wav -m models/myvoice.pth \\
          --tts-engine openai --openai-instructions "Speak dramatically" --openai-speed 1.2

      pipe text from stdin
        echo "Hello world." | mic-drop -o output/hello.wav -m models/myvoice.pth

      batch-convert a directory
        mic-drop --batch -i scripts/ -o output/batch/ -m models/myvoice.pth

    The cache directory (where Tortoise stores ~2–4 GB of model weights) is
    configured once by ./setup.sh and persisted in .mic-drop.env. OpenAI TTS
    requires OPENAI_API_KEY in .mic-drop.env instead.

    exit codes
      0    success
      1    runtime error (bad input, missing file, processing failure)
      2    usage error (unrecognised flag, missing required argument)
      130  interrupted (Ctrl+C)
""")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mic-drop",
        description="Multi-engine voice-cloning TTS: Tortoise, OpenAI, or Coqui + RVC in one pipeline.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Input .txt or .md file (single mode), or directory (--batch). Omit to read from stdin.",
    )
    io_group.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output WAV file (single mode) or directory (--batch).",
    )
    io_group.add_argument(
        "--strip-markdown",
        action="store_const",
        const=True,
        default=None,
        help=(
            "Strip Markdown formatting before synthesis. "
            "Automatic for .md files; required when piping "
            "Markdown via stdin."
        ),
    )
    io_group.add_argument(
        "--save-intermediate",
        action="store_true",
        help=(
            "Save the pre-RVC Tortoise TTS output alongside the final "
            "output file (with _pre_rvc suffix). Useful for debugging."
        ),
    )

    # -- Voice model ---------------------------------------------------------
    model_group = parser.add_argument_group("Voice Model (RVC)")
    model_group.add_argument(
        "-m",
        "--voice-model",
        type=Path,
        default=None,
        help="Path to the RVC voice model (.pth). Optional — if omitted, RVC conversion is skipped.",
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
        help="Quality preset (default: standard).",
    )
    tt_group.add_argument(
        "--tortoise-voice",
        type=str,
        default=None,
        help=(
            "Built-in voice name (e.g. 'female'), "
            "or path to a directory of reference WAV clips."
        ),
    )
    tt_group.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Tortoise model-cache directory (~2–4 GB on first run). "
            "Defaults to the value saved in .mic-drop.env by setup.sh, "
            "falling back to ~/.cache/tortoise-tts."
        ),
    )

    # -- TTS Engine Selection ------------------------------------------------
    engine_group = parser.add_argument_group("TTS Engine")
    engine_group.add_argument(
        "--tts-engine",
        choices=["tortoise", "openai", "coqui"],
        default="tortoise",
        help="TTS backend to use (default: tortoise).",
    )

    # -- OpenAI TTS options --------------------------------------------------
    openai_group = parser.add_argument_group("OpenAI TTS")
    openai_group.add_argument(
        "--openai-model",
        choices=["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
        default="gpt-4o-mini-tts",
        help="OpenAI TTS model (default: gpt-4o-mini-tts with instructions support).",
    )
    openai_group.add_argument(
        "--openai-voice",
        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        default="alloy",
        help="OpenAI voice selection (default: alloy).",
    )
    openai_group.add_argument(
        "--openai-instructions",
        type=str,
        default=None,
        help="Optional instructions to guide voice characteristics and style.",
    )
    openai_group.add_argument(
        "--openai-speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier: 0.25 to 4.0 (default: 1.0). Values >1.0 speed up, <1.0 slow down.",
    )

    # -- Coqui TTS options ---------------------------------------------------
    coqui_group = parser.add_argument_group("Coqui XTTS-v2")
    coqui_group.add_argument(
        "--coqui-speaker",
        type=Path,
        default=None,
        help="Path to speaker audio file for voice cloning (6+ seconds recommended).",
    )
    coqui_group.add_argument(
        "--coqui-language",
        type=str,
        default="en",
        help=(
            "Language code: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, "
            "zh-cn, ja, hu, ko, hi (default: en)."
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

    # -- Misc (verbosity is a mutually-exclusive pair) ----------------------
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process every .txt / .md in --input directory.",
    )
    misc_group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto — CUDA > MPS > CPU).",
    )
    verbosity = misc_group.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug-level logging.",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Show warnings and errors only.",
    )

    return parser


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _read_input(path: Path | None) -> str:
    """Read text from *path* or from stdin when *path* is ``None``."""
    if path is None:
        if sys.stdin.isatty():
            raise CliError(
                "No input file and stdin is a terminal.\n"
                '  Either pass -i <file> or pipe text:  echo "…" | mic-drop …'
            )
        return sys.stdin.read()

    if not path.exists():
        raise CliError(f"Input file not found: {path}")

    return path.read_text(encoding="utf-8")


def _maybe_strip_md(text: str, path: Path | None, flag: bool | None) -> str:
    """Strip Markdown when *flag* is ``True`` or *path* ends in ``.md``."""
    should_strip = flag if flag is not None else (
        path is not None and path.suffix.lower() == ".md"
    )
    if should_strip:
        from tts_pipeline.text_processing import strip_markdown

        logger.info("Stripping Markdown formatting from input.")
        text = strip_markdown(text)
    return text


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry-point.  Returns the integer exit code."""
    parser = build_parser()
    args = parser.parse_args()

    # -- Logging ------------------------------------------------------------
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Dispatch with unified error handling ------------------------------
    try:
        _run(args)
        return 0
    except CliError as exc:
        logger.error("%s", exc)
        return exc.code
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        logger.error("  Run ./setup.sh to install ML backends automatically.")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error: %s", exc)
        if args.verbose:
            raise
        return 1


def _run(args: argparse.Namespace) -> None:
    """Everything after logging is configured."""
    _apply_config_defaults(args)

    # -- Pre-flight validation ------------------------------------------
    # Validate OpenAI TTS requirements
    if args.tts_engine == "openai":
        if not getattr(args, "openai_api_key", None):
            raise CliError(
                "OPENAI_API_KEY is required for OpenAI TTS.\n"
                "  Set it in .mic-drop.env or pass via environment variable."
            )

    # RVC voice model validation (optional)
    if args.voice_model is not None:
        if not args.voice_model.exists():
            raise CliError(f"Voice model not found: {args.voice_model}")
        if args.rvc_index is not None and not args.rvc_index.exists():
            raise CliError(f"RVC index file not found: {args.rvc_index}")
    else:
        logger.info("No voice model specified — RVC conversion will be skipped.")

    # -- Dispatch -------------------------------------------------------
    if args.batch:
        _run_batch(args)
    else:
        _validate_single(args)
        text = _read_input(args.input)
        text = _maybe_strip_md(text, args.input, args.strip_markdown)
        if not text.strip():
            raise CliError("Input text is empty.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _run_single(text, args)


def _validate_single(args: argparse.Namespace) -> None:
    """Checks that only apply in single-file mode."""
    if args.input is not None:
        if args.input.is_dir():
            raise CliError(
                f"--input {args.input} is a directory. "
                "Use --batch to process all files in a directory."
            )
        if args.input.suffix.lower() not in (".txt", ".md"):
            raise CliError(
                f"Unsupported input extension: {args.input.suffix!r}. "
                "Supported: .txt, .md"
            )


# ---------------------------------------------------------------------------
# Single-file and batch runners
# ---------------------------------------------------------------------------


def _build_pipeline(args: argparse.Namespace):  # noqa: ANN202
    """Lazy import + construct the Pipeline from parsed CLI args."""
    from tts_pipeline.pipeline import Pipeline

    return Pipeline(
        voice_model_path=args.voice_model,
        rvc_index_path=args.rvc_index,
        tts_engine=args.tts_engine,
        # Tortoise params
        tortoise_preset=args.tortoise_preset,
        tortoise_voice=args.tortoise_voice,
        # OpenAI params
        openai_model=getattr(args, "openai_model", "gpt-4o-mini-tts"),
        openai_voice=getattr(args, "openai_voice", "alloy"),
        openai_api_key=getattr(args, "openai_api_key", None),
        openai_instructions=getattr(args, "openai_instructions", None),
        openai_speed=getattr(args, "openai_speed", 1.0),
        # Coqui params
        coqui_speaker=getattr(args, "coqui_speaker", None),
        coqui_language=getattr(args, "coqui_language", "en"),
        # Common params
        rvc_pitch=args.rvc_pitch,
        rvc_method=args.rvc_method,
        output_sample_rate=args.sample_rate,
        device=args.device,
        cache_dir=args.cache_dir,
        save_intermediate=args.save_intermediate,
    )


def _run_single(text: str, args: argparse.Namespace) -> None:
    pipeline = _build_pipeline(args)
    pipeline.run(text=text, output_path=args.output)
    logger.info("Output written to %s", args.output)


def _run_batch(args: argparse.Namespace) -> None:
    if args.input is None or not args.input.is_dir():
        raise CliError("--batch requires --input to be a directory.")

    if args.output.is_file():
        raise CliError(
            f"--output {args.output} is an existing file; "
            "batch mode writes one WAV per input file into a directory."
        )

    input_files = sorted(
        f for ext in ("*.txt", "*.md") for f in args.input.glob(ext)
    )
    if not input_files:
        raise CliError(f"No .txt or .md files found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = _build_pipeline(args)

    from tqdm import tqdm

    errors: list[tuple[str, str]] = []
    for src in tqdm(input_files, desc="Batch", unit="file", disable=args.quiet):
        try:
            text = src.read_text(encoding="utf-8")
            text = _maybe_strip_md(text, src, args.strip_markdown)
            if not text.strip():
                raise CliError("empty after processing")
            out_path = args.output / src.with_suffix(".wav").name
            pipeline.run(text=text, output_path=out_path)
            logger.debug("  → %s", out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipped %s: %s", src.name, exc)
            errors.append((src.name, str(exc)))

    succeeded = len(input_files) - len(errors)
    logger.info("Batch complete: %d/%d succeeded.", succeeded, len(input_files))
    if errors:
        logger.warning("%d file(s) failed:", len(errors))
        for name, msg in errors:
            logger.warning("  %s — %s", name, msg)
