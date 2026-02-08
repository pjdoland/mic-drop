# mic-drop

**Multi-engine voice-cloning TTS — local or cloud, your choice.**

mic-drop is a flexible voice-cloning pipeline that lets you choose your TTS engine: run **Tortoise TTS locally** (no API costs, full privacy) or use **OpenAI TTS** (fast, cheap API). Both feed into RVC for voice cloning, giving you the best of both worlds.

```
                    ┌─────────────────┐
                    │  Choose Engine  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      Tortoise TTS                   OpenAI TTS
    (local, no API)                (fast, $0.01/1K)
              │                             │
              └──────────────┬──────────────┘
                             ▼
                      raw speech (24 kHz)
                             │
                             ▼
                      RVC conversion
                             │
                             ▼
                    cloned voice (16 kHz)
                             │
                             ▼
                  Resample + Normalize
                             │
                             ▼
                        output.wav
```

**Why mic-drop?**
- 🎭 **Multiple TTS engines**: Local Tortoise or cloud OpenAI TTS
- 🎤 **Voice cloning**: RVC transforms any TTS into your voice
- 💰 **Cost-effective**: Tortoise is free, OpenAI is $0.01/1K chars
- 🔒 **Privacy options**: Keep everything local with Tortoise
- ⚡ **Speed flexibility**: Ultra-fast API or quality local synthesis
- 📦 **Batteries included**: One command installs everything

---

## Quick start

**Requires Python 3.10.x** (`brew install python@3.10`)

```bash
./setup.sh                          # one-time bootstrap (venv, PyTorch, deps)

# With Tortoise TTS (default, local, free)
echo "Hello from mic-drop." | mic-drop -o output/hello.wav -m models/your_model.pth

# With OpenAI TTS (faster, requires API key, ~$0.01 per 1K chars)
export OPENAI_API_KEY=sk-...
echo "Hello from OpenAI." | mic-drop -o output/hello.wav -m models/your_model.pth \
  --tts-engine openai --openai-voice nova
```

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10.x | **Required** — RVC dependencies break on 3.11+ |
| NVIDIA GPU + CUDA | Optional but strongly recommended for Tortoise speed |
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

# 4. mic-drop core + OpenAI support
pip install -e .
pip install openai

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

## Choosing a TTS Engine

mic-drop supports two TTS backends: **Tortoise** (local) and **OpenAI TTS** (API-based).

### Quick Comparison

|  | **Tortoise TTS** | **OpenAI TTS** |
|---|---|---|
| **Cost** | Free | $0.010/1K chars (gpt-4o-mini-tts)<br>$0.015/1K (tts-1)<br>$0.030/1K (tts-1-hd) |
| **Speed** | Slow (30-60s per paragraph) | Fast (2-5s per paragraph) |
| **Privacy** | 100% local, no data leaves your machine | Text sent to OpenAI servers |
| **Internet** | Not required | Required |
| **Setup** | ~2-4 GB model download (one-time) | API key only |
| **Voices** | Many built-in + custom reference clips | 6 built-in voices |
| **Prosody Control** | Excellent (presets, voices, references) | Good (instructions with gpt-4o-mini-tts) |
| **Quality** | Very high (especially with high_quality preset) | Very high |
| **Limits** | None | API rate limits |

### Tortoise TTS (default)

**Best for:** Privacy-conscious users, no API costs, maximum prosody control, long-form content where you have time.

**Pros:**
- Runs entirely locally — no API costs, no data leaves your machine
- Highly customizable voice selection via reference clips
- Rich prosody control through different presets and voices
- No usage limits

**Cons:**
- Slow synthesis (even ultra_fast preset takes ~30-60s for a paragraph)
- Large model downloads (~2-4 GB on first run)
- Requires GPU for reasonable speed

**Example:**
```bash
mic-drop -i input.txt -o output.wav -m models/voice.pth \
  --tortoise-preset fast \
  --tortoise-voice tom
```

### OpenAI TTS

**Best for:** Quick iteration, testing, large batch jobs, when speed matters more than cost.

**Pros:**
- Very fast synthesis (typically 2-5s for a paragraph)
- No local model downloads
- 6 high-quality voices built-in (alloy, echo, fable, onyx, nova, shimmer)
- Consistent quality across different text lengths
- Instructions support with gpt-4o-mini-tts for tone and style control
- Adjustable speech speed (0.25x to 4.0x)

**Cons:**
- Costs money ($0.010/1K chars for gpt-4o-mini-tts)
- Requires API key and internet connection
- Your text is sent to OpenAI (privacy consideration)

**Example:**
```bash
mic-drop -i input.txt -o output.wav -m models/voice.pth \
  --tts-engine openai \
  --openai-voice nova \
  --openai-instructions "Speak in a calm, measured tone"
```

### Cost Examples (OpenAI TTS)

| Content Length | Model | Cost |
|---|---|---|
| Tweet (280 chars) | gpt-4o-mini-tts | $0.003 |
| Blog post (5,000 chars) | gpt-4o-mini-tts | $0.05 |
| Short story (20,000 chars) | gpt-4o-mini-tts | $0.20 |
| Novel chapter (100,000 chars) | gpt-4o-mini-tts | $1.00 |

---

## Running from a USB / thumb drive

Two things eat disk space: Tortoise downloads ~2–4 GB of model weights on
first run, and each RVC voice model is typically 200–500 MB. Both can live
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
downloads its weights (~2-4 GB). Every subsequent run reads from the cache and is
much faster. The cache directory is fully portable — move the drive to
another machine or mount point and just update the paths.

**Important:** mic-drop automatically configures HuggingFace's cache to use the same
directory as `--cache-dir`, ensuring ALL model downloads (both Tortoise and HuggingFace Hub)
go to your USB drive instead of filling up your system disk.

**Tip:** `setup.sh` saves the cache directory you chose into
`.mic-drop.env` at the repo root. On every subsequent run `mic-drop`
reads that file automatically, so you can omit `--cache-dir` entirely
once it has been set. Edit `.mic-drop.env` any time you move the drive
to a new mount point.

---

## Understanding voice selection and prosody

mic-drop uses a two-stage pipeline: **your chosen TTS engine generates the base audio with prosody**
(rhythm, intonation, pacing), then **RVC transforms the voice timbre** while preserving
that prosody.

### Tortoise Voice Selection

Even though RVC clones your target voice, the Tortoise voice you choose affects:
- **Speaking pace** - some voices are naturally faster or slower
- **Expressiveness** - some are more dynamic, others more monotone
- **Intonation patterns** - how pitch rises and falls
- **Pause placement** - natural rhythm and breathing

Built-in voices to try (pass via `--tortoise-voice`):

| Voice | Characteristics |
|-------|----------------|
| `tom`, `daniel`, `william` | Male, neutral prosody, clear articulation |
| `pat`, `pat2` | Different prosody styles |
| `deniro`, `freeman` | Actor-based, more expressive |
| `emma`, `angie`, `jlaw`, `halle` | Female voices |
| `geralt` | Character voice with unique prosody |

**Tip:** Use `--save-intermediate` to save the pre-RVC audio and compare:

```bash
mic-drop -i scripts/test.txt \
  --tortoise-voice tom \
  --save-intermediate \
  -m models/myvoice.pth \
  -o output/test.wav
# Creates: output/test.wav (final) and output/test_pre_rvc.wav (Tortoise only)
```

### OpenAI Voice Selection

OpenAI provides 6 built-in voices with different characteristics:

| Voice | Characteristics |
|-------|----------------|
| `alloy` | Neutral, balanced |
| `echo` | Male, clear |
| `fable` | Warm, expressive |
| `onyx` | Deep, authoritative |
| `nova` | Female, energetic |
| `shimmer` | Bright, friendly |

**OpenAI Speed vs. Instructions:**

OpenAI TTS provides two separate controls:
- **`--openai-speed`** (0.25-4.0): Controls playback tempo. Use this to make speech faster or slower.
  - `0.5` = half speed, `1.0` = normal (default), `1.5` = 50% faster, `2.0` = double speed
- **`--openai-instructions`** (gpt-4o-mini-tts only): Controls tone, emotion, style, and character.
  - Examples: "Speak dramatically", "Use a cheerful tone", "Sound like a professional narrator"

**Example:**
```bash
# Fast-paced dramatic narration
mic-drop -i script.txt -o output.wav -m models/voice.pth \
  --tts-engine openai \
  --openai-voice onyx \
  --openai-speed 1.3 \
  --openai-instructions "Speak in a dramatic movie trailer voice"
```

### Speed vs. Quality Presets (Tortoise only)

| Preset | Speed | Quality | Use case |
|--------|-------|---------|----------|
| `ultra_fast` | ⚡⚡⚡ | ★★☆☆ | Testing, iteration, drafts |
| `fast` | ⚡⚡ | ★★★☆ | Good balance for most use cases |
| `standard` | ⚡ | ★★★★ | Default, slower but better quality |
| `high_quality` | 🐌 | ★★★★★ | Final output, very slow |

**Recommendation:** Start with `ultra_fast` for iteration, then re-run with `fast` or
`standard` for final output. RVC does most of the voice work, so ultra_fast is often good enough.

---

## Usage

### Basic — file input

```bash
# Tortoise (default)
mic-drop -i scripts/example.txt -o output/speech.wav -m models/myvoice.pth

# OpenAI
mic-drop -i scripts/example.txt -o output/speech.wav -m models/myvoice.pth \
  --tts-engine openai --openai-voice nova
```

### Pipe from stdin

```bash
echo "Piped text works too." | mic-drop -o output/piped.wav -m models/myvoice.pth
```

### Full options example (Tortoise)

```bash
mic-drop \
  --input           scripts/dramatic.txt \
  --output          output/dramatic.wav \
  --voice-model     models/myvoice.pth \
  --rvc-index       models/myvoice.index \
  --tortoise-preset fast \
  --tortoise-voice  tom \
  --save-intermediate \
  --rvc-pitch       -2 \
  --rvc-method      rmvpe \
  --sample-rate     48000 \
  --device          cuda \
  --verbose
```

### Full options example (OpenAI)

```bash
mic-drop \
  --input           scripts/dramatic.txt \
  --output          output/dramatic.wav \
  --voice-model     models/myvoice.pth \
  --rvc-index       models/myvoice.index \
  --tts-engine      openai \
  --openai-model    gpt-4o-mini-tts \
  --openai-voice    echo \
  --openai-speed    1.2 \
  --openai-instructions "Speak dramatically like a movie trailer narrator" \
  --save-intermediate \
  --rvc-pitch       -2 \
  --rvc-method      rmvpe \
  --sample-rate     48000 \
  --verbose
```

### Markdown input

`.md` files are accepted anywhere a `.txt` file is. Markdown syntax
(headers, bold, links, code blocks, lists …) is stripped automatically
before synthesis. When piping Markdown via stdin, add `--strip-markdown`:

```bash
mic-drop -i scripts/example.md -o output/from-md.wav -m models/myvoice.pth

cat notes.md | mic-drop --strip-markdown -o output/notes.wav -m models/myvoice.pth
```

### Batch mode

Convert every `.txt` and `.md` file in a directory:

```bash
mic-drop \
  --input       scripts/ \
  --output      output/batch/ \
  --voice-model models/myvoice.pth \
  --batch
```

---

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | stdin | Input `.txt` or `.md` file, or directory in `--batch` mode |
| `-o`, `--output` | _required_ | Output WAV path, or output directory in `--batch` mode |
| `--strip-markdown` | auto | Strip Markdown syntax before synthesis. Automatic for `.md` files; required when piping Markdown via stdin |
| `--save-intermediate` | — | Save pre-RVC TTS output alongside final output (with `_pre_rvc` suffix). Useful for debugging and comparing TTS engines |
| `-m`, `--voice-model` | _required_ | Path to RVC `.pth` model |
| `--rvc-index` | _none_ | Path to companion RVC `.index` file |
| **TTS Engine Selection** | | |
| `--tts-engine` | `tortoise` | TTS backend: `tortoise` (local, free) or `openai` (API, fast) |
| **Tortoise TTS Options** | | |
| `--tortoise-preset` | `standard` | Quality preset: `ultra_fast` / `fast` / `standard` / `high_quality`. Recommend `ultra_fast` for iteration, `fast` for final output. |
| `--tortoise-voice` | random | Built-in voice name (tom, daniel, william, pat, emma, etc.) or path to reference WAV clip(s). Affects prosody (rhythm, intonation, pacing) even when using RVC. |
| `--cache-dir` | `.mic-drop.env`, then `~/.cache/tortoise-tts` | Tortoise model-cache directory (~2–4 GB on first run). Point at a USB drive to keep large files off your main disk. Not applicable for OpenAI TTS. |
| **OpenAI TTS Options** | | |
| `--openai-model` | `gpt-4o-mini-tts` | OpenAI model: `gpt-4o-mini-tts` (supports instructions, cheapest), `tts-1`, or `tts-1-hd` (highest quality) |
| `--openai-voice` | `alloy` | Voice selection: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `--openai-instructions` | _none_ | Optional instructions for voice tone, style, and character (works with gpt-4o-mini-tts): "Use a dramatic tone", "Sound cheerful", etc. Note: For speed control, use `--openai-speed` instead. |
| `--openai-speed` | `1.0` | Speech speed multiplier (0.25 to 4.0). Values >1.0 speed up, <1.0 slow down. Example: `1.5` = 50% faster, `0.75` = 25% slower |
| **RVC & Audio Options** | | |
| `--rvc-pitch` | `0` | Pitch shift in semitones |
| `--rvc-method` | `rmvpe` | Pitch extraction: `rmvpe` / `pm` / `crepe` |
| `--sample-rate` | `44100` | Output Hz: `16000` / `22050` / `44100` / `48000` |
| `--device` | `auto` | Torch device: `auto` / `cpu` / `cuda` / `mps` (affects Tortoise & RVC, not OpenAI) |
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
│   ├── __init__.py          # package metadata, version & device resolution
│   ├── __main__.py          # python -m entry-point
│   ├── base.py              # TTSEngine ABC (interface for all TTS engines)
│   ├── cli.py               # argument parsing & dispatch
│   ├── config.py            # .mic-drop.env loading
│   ├── text_processing.py  # shared text normalization & chunking
│   ├── audio.py             # resample + peak-normalize helpers
│   ├── engines/
│   │   ├── __init__.py      # engine exports
│   │   ├── tortoise_engine.py  # Tortoise TTS wrapper
│   │   └── openai_engine.py    # OpenAI TTS wrapper
│   ├── rvc.py               # RVC voice-conversion
│   └── pipeline.py          # end-to-end orchestration + WAV export
├── tests/
│   ├── test_text_processing.py   # chunking & normalization tests
│   ├── test_audio_utils.py       # peak-normalization tests
│   ├── test_openai_tts.py        # OpenAI engine tests
│   └── test_cli.py               # CLI parsing, config & error tests
├── models/                  # drop your .pth / .index files here
├── scripts/                 # example input texts
│   ├── example.txt
│   ├── example.md           # same example in Markdown
│   └── dramatic.txt
├── output/                  # generated WAVs land here
├── requirements.txt         # core runtime deps (includes openai)
├── requirements-rvc.txt     # RVC + fairseq (needs pip 24.0)
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

All 97 tests should pass.

---

## Troubleshooting

### General Issues

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: tortoise` | Run `./setup.sh` — it installs all ML backends in the correct order. |
| `ModuleNotFoundError: rvc` | Do **not** `pip install rvc-python` directly. Run `pip install pip==24.0`, then `pip install -r requirements-rvc.txt`, then `pip install --upgrade pip`. See `requirements-rvc.txt` for details. |
| RVC install fails / fairseq errors | Same as above — fairseq's dep tree breaks with pip ≥ 24.1. Pin to 24.0 first. |
| `setup.sh` fails with "Python 3.10.x is required" | Install Python 3.10 via Homebrew: `brew install python@3.10`. Python 3.11+ breaks RVC dependencies (fairseq → hydra → antlr4). Delete your existing `venv/` and re-run `./setup.sh`. |
| `ModuleNotFoundError: typing.io` or antlr4 errors | You're on Python 3.11+. Downgrade to Python 3.10 (see above). |

### Tortoise TTS Issues

| Symptom | Fix |
|---------|-----|
| `'TextToSpeech' object has no attribute 'load_voice'` | Outdated tortoise-tts installation. Pull latest changes and run `pip install -e .`. |
| `stack expects each tensor to be equal size` error with `--tortoise-voice` | Fixed in latest version. Update with `pip install -e .`. |
| `No space left on device` while downloading models | Your system disk is full. Use `--cache-dir` to point to a USB drive or external storage with at least 4GB free space. |
| Tortoise re-downloads weights every run | `setup.sh` saves your chosen cache path to `.mic-drop.env`, which `mic-drop` reads automatically. If you skipped that step, pass `--cache-dir` explicitly. |
| Very slow synthesis | Use `--tortoise-preset ultra_fast` for faster generation. Tortoise is inherently slow; even `ultra_fast` provides decent results since RVC does the heavy lifting. |
| Different voices have poor prosody | Try various built-in voices with `--tortoise-voice` (tom, daniel, william, pat, etc.). Use `--save-intermediate` to hear the Tortoise output before RVC transformation. |

### OpenAI TTS Issues

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: openai` | Install OpenAI library: `pip install openai` or re-run `./setup.sh` |
| `OPENAI_API_KEY is required` | Set your API key in `.mic-drop.env` or export it: `export OPENAI_API_KEY=sk-...` |
| `OpenAI authentication failed` | Check OPENAI_API_KEY in `.mic-drop.env`. Get a key from https://platform.openai.com/api-keys |
| `OpenAI rate limit exceeded` | Wait a moment and retry, or switch to `--tts-engine tortoise` |
| OpenAI TTS costs too much | Use `--openai-model gpt-4o-mini-tts` (cheapest at $0.01/1K chars) or switch to Tortoise (free) |
| Text is too long for OpenAI | mic-drop automatically splits long text into 4096-char chunks. Check logs for chunk count and estimated cost. |
| Instructions don't affect speech speed | Use `--openai-speed` to control tempo (e.g., `1.5` for faster). Instructions only control tone/style/character. |

### Device & Performance Issues

| Symptom | Fix |
|---------|-----|
| Tortoise or RVC crash on Apple Silicon | MPS support is experimental. Both engines fall back to CPU automatically on failure. Force it explicitly with `--device cpu` if the auto-fallback doesn't trigger. |
| CUDA out of memory | Use `--tortoise-preset fast` or `ultra_fast`; fall back to `--device cpu` |
| Audio is very quiet | The pipeline peak-normalises to 0.9 by default; check source levels |
| Pitch sounds wrong | Try `--rvc-pitch 0` first, then adjust ±1 semitone at a time |

---

## License

Provided as-is for personal and research use.
