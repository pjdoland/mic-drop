# mic-drop

**Local voice-cloning TTS — no mic required.**

mic-drop chains [Tortoise TTS](https://github.com/betteroi/tortoise-tts) (text → speech) with [RVC](https://github.com/retrieval-based-voice-conversion) (speech → your voice) into a single command-line tool. Everything runs locally — no cloud APIs, no subscriptions.

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

```bash
pip install -e .
echo "Hello from mic-drop." | python -m tts_pipeline -o output/hello.wav -m models/your_model.pth
```

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.9 | 3.10+ recommended |
| NVIDIA GPU + CUDA | Optional but strongly recommended for speed |
| ~8 GB free RAM | Tortoise + RVC combined footprint |

### Steps

```bash
# 1. Clone
git clone <repo-url>
cd mic-drop

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install CUDA-enabled PyTorch first (adjust URL for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install mic-drop and remaining deps
pip install -e .
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

### Batch mode

Convert every `.txt` file in a directory:

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
| `-i`, `--input` | stdin | Input `.txt` file, or directory in `--batch` mode |
| `-o`, `--output` | _required_ | Output WAV path, or output directory in `--batch` mode |
| `-m`, `--voice-model` | _required_ | Path to RVC `.pth` model |
| `--rvc-index` | _none_ | Path to companion RVC `.index` file |
| `--tortoise-preset` | `standard` | Quality preset: `ultra_fast` / `fast` / `standard` / `high_quality` |
| `--tortoise-voice` | random | Built-in voice name or path to reference WAV clip(s) |
| `--rvc-pitch` | `0` | Pitch shift in semitones |
| `--rvc-method` | `rmvpe` | Pitch extraction: `rmvpe` / `pm` / `crepe` |
| `--sample-rate` | `44100` | Output Hz: `16000` / `22050` / `44100` / `48000` |
| `--device` | `auto` | Torch device: `auto` / `cpu` / `cuda` |
| `--batch` | — | Enable batch mode |
| `-v`, `--verbose` | — | Debug-level logging |

---

## Project layout

```
mic-drop/
├── tts_pipeline/
│   ├── __init__.py          # package metadata & version
│   ├── __main__.py          # python -m entry-point
│   ├── cli.py               # argument parsing & dispatch
│   ├── tortoise.py          # Tortoise TTS wrapper + text chunking
│   ├── rvc.py               # RVC voice-conversion wrapper
│   └── pipeline.py          # end-to-end orchestration + audio helpers
├── tests/
│   ├── test_text_processing.py   # chunking & normalisation tests
│   └── test_audio_utils.py       # peak-normalisation tests
├── models/                  # drop your .pth / .index files here
├── scripts/                 # example input texts
│   ├── example.txt
│   └── dramatic.txt
├── output/                  # generated WAVs land here
├── requirements.txt
├── setup.py
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
| `ModuleNotFoundError: tortoise` | `pip install tortoise-tts` |
| `ModuleNotFoundError: rvc` | `pip install rvc-python` |
| CUDA out of memory | Use `--tortoise-preset fast` or `ultra_fast`; fall back to `--device cpu` |
| Audio is very quiet | The pipeline peak-normalises to 0.9 by default; check source levels |
| Model file not found | Confirm the exact path passed to `--voice-model` |
| Very slow on long scripts | Lower `MAX_WORDS_PER_CHUNK` in `tortoise.py`; use `fast` preset |
| Pitch sounds wrong | Try `--rvc-pitch 0` first, then adjust ±1 semitone at a time |

---

## License

Provided as-is for personal and research use.
