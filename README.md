# mic-drop

**Local voice-cloning TTS — no mic required.**

mic-drop chains [Tortoise TTS](https://github.com/neonbjb/tortoise-tts) (text → speech) with [RVC](https://github.com/retrieval-based-voice-conversion) (speech → your voice) into a single command-line tool. Everything runs locally — no cloud APIs, no subscriptions.

```
text  ──▶  Tortoise TTS  ──▶  raw speech (24 kHz)
                                     │
                                     ▼
                             RVC conversion  ──▶  cloned voice (16 kHz)
                                                        │
                                                        ▼
                                                 Resample + Normalize
                                                        │
                                                        ▼
                                                   output.wav
```

---

## Quick start

**Requires Python 3.10.x** (`brew install python@3.10`)

```bash
./setup.sh                          # one-time bootstrap (venv, PyTorch, deps)
echo "Hello from mic-drop." | mic-drop -o output/hello.wav -m models/your_model.pth
```

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10.x | **Required** — RVC dependencies break on 3.11+ |
| NVIDIA GPU + CUDA | Optional but strongly recommended for speed |
| ~8 GB free RAM | Tortoise + RVC combined footprint |

### Steps — macOS (recommended)

```bash
git clone <repo-url>
cd mic-drop
chmod +x setup.sh
./setup.sh          # creates venv, installs PyTorch + deps, runs tests
```

`setup.sh` is idempotent — re-running it is safe and will pick up anything
that was missing the first time.

### Steps — manual / Linux

**Important:** Python 3.10.x is required. Python 3.11+ breaks RVC dependencies (fairseq, hydra, antlr4).

```bash
# 1. Clone
git clone <repo-url>
cd mic-drop

# 2. Virtual environment with Python 3.10
python3.10 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. PyTorch
#   macOS (Intel + Apple Silicon):
pip install torch torchaudio
#   Linux with NVIDIA GPU (adjust cu version as needed):
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. mic-drop core
pip install -e .

# 5. RVC (needs pip 24.0 for fairseq — see requirements-rvc.txt)
pip install pip==24.0
pip install -r requirements-rvc.txt
pip install --upgrade pip
```

---

## Obtaining an RVC voice model

mic-drop requires a pre-trained RVC `.pth` model — and optionally a companion `.index` file — for your target voice.

1. **Train your own** with [Applio](https://github.com/IAC/Applio) or the original RVC project using ~5-10 minutes of clean audio.
2. **Download** community models (ensure you have the rights to use them).
3. Place the files in the `models/` directory:
   ```
   models/
   ├── myvoice.pth
   └── myvoice.index        # optional — improves quality
   ```

---

## Running from a USB / thumb drive

Two things eat disk space: Tortoise downloads ~2–4 GB of model weights on
first run, and each RVC voice model is typically 200–500 MB.  Both can live
on a thumb drive so your main disk stays clean.

### What goes on the drive

```
USB drive/
├── tortoise_cache/     # created automatically on first run (~2–4 GB)
├── models/
│   ├── myvoice.pth
│   └── myvoice.index
├── scripts/            # optional — your .txt / .md input files
└── output/             # optional — generated WAVs
```

### Wiring it up

`--cache-dir` tells Tortoise where to download and cache its model weights.
`--voice-model` and `--rvc-index` already accept any path, so just point
everything at the drive:

```bash
mic-drop \
  --input           scripts/example.txt \
  --output          /Volumes/USB/output/speech.wav \
  --voice-model     /Volumes/USB/models/myvoice.pth \
  --rvc-index       /Volumes/USB/models/myvoice.index \
  --cache-dir       /Volumes/USB/tortoise_cache
```

The first run with a fresh `--cache-dir` will be slow while Tortoise
downloads its weights.  Every subsequent run reads from the cache and is
much faster.  The cache directory is fully portable — move the drive to
another machine or mount point and just update the paths.

**Tip:** `setup.sh` saves the cache directory you chose into
`.mic-drop.env` at the repo root.  On every subsequent run `mic-drop`
reads that file automatically, so you can omit `--cache-dir` entirely
once it has been set.  Edit `.mic-drop.env` any time you move the drive
to a new mount point.

---

## Usage

### Basic — file input

```bash
python -m tts_pipeline \
  --input  scripts/example.txt \
  --output output/speech.wav \
  --voice-model models/myvoice.pth
```

### Pipe from stdin

```bash
echo "Piped text works too." | python -m tts_pipeline \
  -o output/piped.wav \
  -m models/myvoice.pth
```

### Full options

```bash
python -m tts_pipeline \
  --input           scripts/dramatic.txt \
  --output          output/dramatic.wav \
  --voice-model     models/myvoice.pth \
  --rvc-index       models/myvoice.index \
  --tortoise-preset high_quality \
  --tortoise-voice  female \
  --rvc-pitch       -2 \
  --rvc-method      rmvpe \
  --sample-rate     48000 \
  --device          cuda \
  --verbose
```

### Markdown input

`.md` files are accepted anywhere a `.txt` file is.  Markdown syntax
(headers, bold, links, code blocks, lists …) is stripped automatically
before synthesis.  When piping Markdown via stdin, add `--strip-markdown`:

```bash
python -m tts_pipeline \
  --input           scripts/example.md \
  --output          output/from-md.wav \
  --voice-model     models/myvoice.pth

cat notes.md | python -m tts_pipeline \
  --strip-markdown \
  -o output/notes.wav \
  -m models/myvoice.pth
```

### Batch mode

Convert every `.txt` and `.md` file in a directory:

```bash
python -m tts_pipeline \
  --input       scripts/ \
  --output      output/batch/ \
  --voice-model models/myvoice.pth \
  --batch
```

### Installed entry-point

After `pip install -e .` the `mic-drop` command is available directly:

```bash
mic-drop -i scripts/example.txt -o output/example.wav -m models/myvoice.pth
```

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | stdin | Input `.txt` or `.md` file, or directory in `--batch` mode |
| `-o`, `--output` | _required_ | Output WAV path, or output directory in `--batch` mode |
| `--strip-markdown` | auto | Strip Markdown syntax before synthesis. Automatic for `.md` files; required when piping Markdown via stdin |
| `--save-intermediate` | — | Save pre-RVC Tortoise TTS output alongside final output (with `_pre_rvc` suffix). Useful for debugging |
| `-m`, `--voice-model` | _required_ | Path to RVC `.pth` model |
| `--rvc-index` | _none_ | Path to companion RVC `.index` file |
| `--tortoise-preset` | `standard` | Quality preset: `ultra_fast` / `fast` / `standard` / `high_quality` |
| `--tortoise-voice` | random | Built-in voice name or path to reference WAV clip(s) |
| `--cache-dir` | `.mic-drop.env`, then `~/.cache/tortoise-tts` | Tortoise model-cache directory (~2–4 GB on first run). Point at a USB drive to keep large files off your main disk |
| `--rvc-pitch` | `0` | Pitch shift in semitones |
| `--rvc-method` | `rmvpe` | Pitch extraction: `rmvpe` / `pm` / `crepe` |
| `--sample-rate` | `44100` | Output Hz: `16000` / `22050` / `44100` / `48000` |
| `--device` | `auto` | Torch device: `auto` / `cpu` / `cuda` / `mps` |
| `--batch` | — | Batch mode: process every `.txt` / `.md` in `--input` directory |
| `-v`, `--verbose` | — | Debug-level logging |
| `-q`, `--quiet` | — | Warnings and errors only (mutually exclusive with `-v`) |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Runtime error (bad input, missing file, processing failure) |
| `2` | Usage error (unrecognised flag, missing required argument) |
| `130` | Interrupted (Ctrl+C) |

---

## Project layout

```
mic-drop/
├── tts_pipeline/
│   ├── __init__.py          # package metadata, version & MPS env flag
│   ├── __main__.py          # python -m entry-point
│   ├── cli.py               # argument parsing, config loading & dispatch
│   ├── audio.py             # shared resample + peak-normalise helpers
│   ├── tortoise.py          # Tortoise TTS wrapper + text chunking
│   ├── rvc.py               # RVC voice-conversion (w/ torch.load patch)
│   └── pipeline.py          # end-to-end orchestration + WAV export
├── tests/
│   ├── test_text_processing.py   # chunking & normalisation tests
│   ├── test_audio_utils.py       # peak-normalisation tests
│   └── test_cli.py               # CLI parsing, config & error tests
├── models/                  # drop your .pth / .index files here
├── scripts/                 # example input texts
│   ├── example.txt
│   ├── example.md           # same example in Markdown
│   └── dramatic.txt
├── output/                  # generated WAVs land here
├── requirements.txt         # core runtime deps
├── requirements-rvc.txt     # RVC + fairseq (needs pip 24.0 — see below)
├── .mic-drop.env.example    # template for persistent config
├── setup.py
├── setup.sh                 # macOS bootstrap script
└── README.md
```

---

## Running the tests

The unit tests cover pure-Python utilities and require only `numpy` + `pytest`:

```bash
pip install pytest
pytest tests/
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: tortoise` | Run `./setup.sh` — it installs all ML backends in the correct order. |
| `ModuleNotFoundError: rvc` | Do **not** `pip install rvc-python` directly. Run `pip install pip==24.0`, then `pip install -r requirements-rvc.txt`, then `pip install --upgrade pip`. See `requirements-rvc.txt` for details. |
| RVC install fails / fairseq errors | Same as above — fairseq's dep tree breaks with pip ≥ 24.1. Pin to 24.0 first. |
| `setup.sh` fails with "Python 3.10.x is required" | Install Python 3.10 via Homebrew: `brew install python@3.10`. Python 3.11+ breaks RVC dependencies (fairseq → hydra → antlr4). Delete your existing `venv/` and re-run `./setup.sh`. |
| `ModuleNotFoundError: typing.io` or antlr4 errors | You're on Python 3.11+. Downgrade to Python 3.10 (see above). |
| Tortoise or RVC crash on Apple Silicon | MPS support is experimental. Both engines fall back to CPU automatically on failure. Force it explicitly with `--device cpu` if the auto-fallback doesn't trigger. |
| CUDA out of memory | Use `--tortoise-preset fast` or `ultra_fast`; fall back to `--device cpu` |
| Audio is very quiet | The pipeline peak-normalises to 0.9 by default; check source levels |
| Model file not found | Confirm the exact path passed to `--voice-model` |
| Very slow on long scripts | Lower `MAX_WORDS_PER_CHUNK` in `tortoise.py`; use `fast` preset |
| Pitch sounds wrong | Try `--rvc-pitch 0` first, then adjust ±1 semitone at a time |
| Tortoise re-downloads weights every run | `setup.sh` saves your chosen cache path to `.mic-drop.env`, which `mic-drop` reads automatically. If you skipped that step, pass `--cache-dir` explicitly to a persistent directory. Without either, Tortoise defaults to `~/.cache/tortoise-tts`. |

---

## License

Provided as-is for personal and research use.
