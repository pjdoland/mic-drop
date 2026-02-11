# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mic-drop** is a multi-engine voice-cloning TTS pipeline that supports three TTS backends:
- **Tortoise TTS** (local, high-quality, slow)
- **Coqui XTTS-v2** (local, multilingual, built-in voice cloning)
- **OpenAI TTS** (API-based, fast, costs money)

All engines can optionally feed into **RVC** (Retrieval-based Voice Conversion) for additional voice cloning/refinement.

## Development Commands

### Environment Setup
```bash
# One-time setup (macOS recommended)
./setup.sh

# Manual setup (Linux or troubleshooting)
python3.10 -m venv venv
source venv/bin/activate
pip install 'torch==2.5.1' 'torchaudio==2.5.1'
pip install -e .
pip install openai 'TTS==0.21.3' 'torchcodec==0.1.0' 'setuptools<81'
pip install pip==24.0
pip install -r requirements-rvc.txt
pip install --upgrade pip
```

**Critical:** Python 3.10.x is required. Python 3.11+ breaks RVC dependencies (fairseq → hydra → antlr4).

### Running the Pipeline
```bash
# Basic usage (Tortoise + RVC)
mic-drop -i input.txt -o output/speech.wav -m models/voice.pth

# Coqui without RVC (Apple Silicon: always use --device cpu)
mic-drop -i input.txt -o output/speech.wav \
  --tts-engine coqui --coqui-speaker audio/speaker.wav --device cpu

# OpenAI with RVC
mic-drop -i input.txt -o output/speech.wav -m models/voice.pth \
  --tts-engine openai --openai-voice nova
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_text_processing.py

# Run with verbose output
pytest tests/ -v
```

All 97 tests should pass. Tests cover text processing, audio utilities, OpenAI integration, and CLI argument parsing.

## Architecture

### Pipeline Flow
```
Text Input
    ↓
Text Processing (normalize, strip markdown, chunk)
    ↓
TTS Engine (Tortoise/Coqui/OpenAI) → raw speech (24 kHz)
    ↓
RVC Conversion (optional) → cloned voice (16 kHz)
    ↓
Post-processing (resample, peak-normalize)
    ↓
Output WAV (16-48 kHz configurable)
```

### Core Module Structure

**`tts_pipeline/`** — Main package
- **`cli.py`** — Argument parsing, config loading, and dispatch logic
  - `build_parser()` — Creates argparse parser with all CLI options
  - `main()` — Entry-point with unified error handling
  - `_apply_config_defaults()` — Loads values from `.mic-drop.env`
  - `_run_single()` / `_run_batch()` — Single-file and batch processing

- **`pipeline.py`** — End-to-end orchestration
  - `Pipeline` class coordinates TTS → RVC → post-processing
  - Lazy-loads appropriate TTS engine based on `--tts-engine` flag
  - Conditionally loads RVC only if `voice_model_path` is provided
  - Handles intermediate audio saving with `--save-intermediate`

- **`base.py`** — Abstract base class (`TTSEngine`)
  - Defines interface that all TTS engines must implement
  - `sample_rate` property, `load()`, and `synthesize(text)` methods

- **`engines/`** — TTS engine implementations
  - **`tortoise_engine.py`** — Tortoise TTS wrapper (local, quality-focused)
  - **`openai_engine.py`** — OpenAI TTS wrapper (API-based, fast)
  - **`coqui_engine.py`** — Coqui XTTS-v2 wrapper (local, multilingual, voice cloning)
  - All engines implement the `TTSEngine` ABC

- **`rvc.py`** — RVC voice conversion
  - `RVCEngine` class wraps `rvc-python` library
  - Applies two pre-import patches:
    1. `OMP_NUM_THREADS=1` — prevents faiss/PyTorch OpenMP conflicts on macOS
    2. Monkey-patches `torch.load` to force `weights_only=False` for fairseq compatibility (PyTorch 2.6+ breaks fairseq models)
  - Handles MPS fallback to CPU on Apple Silicon
  - Input: numpy array + sample rate → Output: converted audio + sample rate

- **`text_processing.py`** — Shared text utilities
  - `normalize_text()` — BOM removal, whitespace collapse
  - `strip_markdown()` — Removes MD syntax, preserves plain text
  - `split_into_chunks()` — Sentence-aware chunking (word-based)
  - `split_by_char_limit()` — Character-based chunking (for API limits)

- **`audio.py`** — Audio utilities
  - `resample()` — Sample rate conversion
  - `peak_normalize()` — Peak normalization to 0.9 amplitude

- **`config.py`** — `.mic-drop.env` file loading
  - Reads `TORTOISE_CACHE_DIR` and `OPENAI_API_KEY`

- **`__init__.py`** — Package metadata and device resolution
  - `resolve_device()` — Converts `"auto"` to cuda → mps → cpu priority

### Key Design Patterns

1. **Engine abstraction**: All TTS engines implement `TTSEngine` ABC, making it trivial to add new backends
2. **Lazy loading**: Models are loaded on first use, not at initialization
3. **Optional RVC**: RVC is completely optional — if no voice model is provided, raw TTS output is used
4. **Fallback logic**: Automatic CPU fallback for MPS failures (especially important for Coqui on Apple Silicon)
5. **Unified error handling**: `CliError` exception with exit codes (0=success, 1=runtime, 2=usage, 130=interrupted)

## Important Constraints & Gotchas

### Python Version
- **Python 3.10.x is mandatory**
- Python 3.11+ breaks fairseq (RVC dependency) due to `typing.io` removal
- `setup.py` enforces this with `python_requires="==3.10.*"`

### Apple Silicon (M1/M2/M3) Limitations
- **Coqui XTTS-v2**: Always use `--device cpu` due to MPS channel limitations
  - Error: "Output channels > 65536 not supported at the MPS device"
  - CPU mode is still fast (~3-4x real-time)
- **RVC**: May fail on MPS, automatic CPU fallback is implemented
- **Tortoise**: MPS support is experimental, auto-fallback to CPU on failure

### Dependency Installation Order
1. PyTorch 2.5.1 must be installed first (Coqui compatibility)
2. RVC dependencies require pip 24.0 during install, then upgrade back
3. Specific version pins needed to avoid conflicts:
   - `transformers==4.31.0`
   - `tokenizers==0.13.3`
   - `numpy==1.23.5`
   - `faiss-cpu==1.7.3`
   - `setuptools<81`

### Configuration File (`.mic-drop.env`)
- Created by `setup.sh`, read automatically by `cli.py`
- Stores `TORTOISE_CACHE_DIR` (for model weights on USB drives)
- Stores `OPENAI_API_KEY` (required for OpenAI TTS)
- **Never commit** (contains machine-specific paths and API keys)

### RVC Pre-Import Patches (in `rvc.py`)
Two patches are applied at module import time:
1. **OMP_NUM_THREADS=1** — Prevents faiss/PyTorch OpenMP conflicts on macOS
2. **torch.load monkey-patch** — Forces `weights_only=False` for fairseq compatibility
   - PyTorch 2.6+ changed default from `False` → `True`, breaking fairseq models
   - This is intentional and required for RVC to work

### Text Processing
- Markdown files (`.md`) have syntax automatically stripped before synthesis
- Long texts are chunked intelligently:
  - Tortoise/Coqui: Word-based chunking (default 150 words/chunk)
  - OpenAI: Character-based chunking (4096 chars/chunk for API limits)
- Chunks split on sentence boundaries when possible

### Audio Pipeline
- TTS engines output at their native sample rates (usually 24 kHz)
- RVC typically outputs at 16 kHz (model-dependent)
- Final output is resampled to user-specified rate (default 44.1 kHz)
- All audio is peak-normalized to 0.9 amplitude before export
- Output format is always 16-bit PCM WAV

## Testing Strategy

Tests are in `tests/` and use pytest. Coverage:
- **`test_text_processing.py`** — Markdown stripping, chunking, normalization
- **`test_audio_utils.py`** — Peak normalization logic
- **`test_openai_tts.py`** — OpenAI engine with mocked API calls
- **`test_cli.py`** — CLI argument parsing, config loading, validation

Tests are designed to run without ML dependencies (Tortoise/RVC) — they test pure-Python utilities and OpenAI integration with mocked HTTP.

## Adding a New TTS Engine

To add a new TTS backend:

1. Create `tts_pipeline/engines/new_engine.py`
2. Implement `TTSEngine` ABC from `tts_pipeline.base`:
   - `sample_rate` property (int)
   - `load()` method (lazy model initialization)
   - `synthesize(text: str) -> np.ndarray` (returns float32 audio in [-1.0, 1.0])
3. Add imports to `tts_pipeline/engines/__init__.py`
4. Update `cli.py`:
   - Add choice to `--tts-engine` argument
   - Add engine-specific arguments in `build_parser()`
   - Add engine initialization case in `pipeline.py` `__init__()`
5. Update `README.md` with engine comparison table
6. Add tests in `tests/test_<engine>_tts.py`

Example structure:
```python
from tts_pipeline.base import TTSEngine
import numpy as np

class NewEngine(TTSEngine):
    def __init__(self, device: str = "auto", **kwargs):
        self.device = device
        self._model = None

    @property
    def sample_rate(self) -> int:
        return 24000

    def load(self) -> None:
        # Lazy load model
        if self._model is None:
            self._model = load_model()

    def synthesize(self, text: str) -> np.ndarray:
        if self._model is None:
            self.load()
        # Return float32 array in [-1.0, 1.0]
        return self._model.generate(text)
```

## Common Development Scenarios

### Working with Voice Models
- RVC models (`.pth`) go in `models/` directory
- Companion `.index` files (optional) improve quality
- No model = no RVC conversion, raw TTS output is used

### Testing Without API Keys
- Tortoise and Coqui work completely offline
- OpenAI tests use mocked HTTP responses (see `test_openai_tts.py`)

### Debugging TTS Output
- Use `--save-intermediate` to save pre-RVC audio for comparison
- Use `-v`/`--verbose` for debug-level logging throughout pipeline
- Check `output/` for generated files

### Batch Processing
- `--batch` mode processes all `.txt`/`.md` files in input directory
- Failed files are logged but don't stop the batch
- Use `-q`/`--quiet` to suppress per-file progress

### Cache Management
- Tortoise downloads ~2-4 GB of models on first run
- Use `--cache-dir` to point to USB drive (keeps main disk clean)
- Coqui downloads ~1.5 GB from Hugging Face on first use
- HuggingFace cache is automatically configured to match `--cache-dir`
