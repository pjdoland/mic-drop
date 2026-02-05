#!/usr/bin/env bash
# =============================================================================
#  mic-drop — macOS setup & bootstrap
#
#  Requirements:
#    • Python 3.10.x (brew install python@3.10)
#    • macOS with Homebrew
#
#  Idempotent: safe to re-run at any time.  Every step checks whether its
#  work is already done before acting.
#
#  Usage:
#      chmod +x setup.sh       # first time only
#      ./setup.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours & helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'   GREEN='\033[0;32m'   YELLOW='\033[1;33m'
BLUE='\033[0;34m'  CYAN='\033[0;36m'    BOLD='\033[1m'
NC='\033[0m'       # reset

info() { printf "  ${BLUE}ℹ${NC}  $*\n"; }
ok()   { printf "  ${GREEN}✓${NC}  $*\n"; }
warn() { printf "  ${YELLOW}⚠${NC}  $*\n"; }
fail() { printf "  ${RED}✗${NC}  $*\n" >&2; exit 1; }
step() { printf "\n${BOLD}${CYAN}── $* ──${NC}\n"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SCRIPT_DIR}/venv"
ENV_FILE="${SCRIPT_DIR}/.mic-drop.env"
BUILD_DIR="${SCRIPT_DIR}/.build"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
printf "\n${CYAN}${BOLD}"
printf "  ╔══════════════════════════════════════╗\n"
printf "  ║   mic-drop  —  local voice TTS       ║\n"
printf "  ║           macOS setup script          ║\n"
printf "  ╚══════════════════════════════════════╝\n"
printf "${NC}\n"

# =============================================================================
# 1. Platform guard — script is tuned for macOS
# =============================================================================
step "Platform"

[[ "$(uname -s)" == "Darwin" ]] || \
    fail "This script is written for macOS.  Linux users: follow README.md manually."

ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
    info "Apple Silicon (arm64) — MPS Metal acceleration will be available."
else
    info "Intel Mac (x86_64) — Metal is not available; will fall back to CPU."
fi

# =============================================================================
# 2. Python 3.10 — required for RVC compatibility
#    RVC dependencies (fairseq → hydra → antlr4) break on Python 3.11+
# =============================================================================
step "Python"

PYTHON=""
# Prefer python3.10 explicitly (RVC requirement)
if   [[ -x /opt/homebrew/bin/python3.10 ]]; then
    PYTHON=/opt/homebrew/bin/python3.10        # Homebrew — Apple Silicon
elif [[ -x /usr/local/bin/python3.10   ]]; then
    PYTHON=/usr/local/bin/python3.10          # Homebrew — Intel
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON="$(command -v python3.10)"         # system
elif [[ -x /opt/homebrew/bin/python3 ]]; then
    PYTHON=/opt/homebrew/bin/python3          # Homebrew — Apple Silicon (fallback)
elif [[ -x /usr/local/bin/python3   ]]; then
    PYTHON=/usr/local/bin/python3            # Homebrew — Intel (fallback)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"           # system / Xcode CLI (fallback)
fi

[[ -z "$PYTHON" ]] && \
    fail "Python not found.  Install Python 3.10 via Homebrew:  brew install python@3.10"

# Version gate — RVC dependencies require exactly Python 3.10
read -r PY_MAJ PY_MIN < <(
    "$PYTHON" -c "import sys; print(sys.version_info.major, sys.version_info.minor)"
)

if [[ $PY_MAJ -ne 3 ]] || [[ $PY_MIN -ne 10 ]]; then
    fail "Python 3.10.x is required for RVC compatibility (found ${PY_MAJ}.${PY_MIN}).\n  Python 3.11+ breaks fairseq/hydra/antlr4 dependencies.\n  Install Python 3.10 via Homebrew:  brew install python@3.10"
fi

ok "Python ${PY_MAJ}.${PY_MIN}  (${PYTHON})"

# =============================================================================
# 3. Virtual environment
# =============================================================================
step "Virtual environment"

if [[ -d "$VENV" ]]; then
    warn "venv already exists — reusing."
else
    "$PYTHON" -m venv "$VENV"
    ok "Created ${VENV}"
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

pip install --upgrade pip --quiet 2>/dev/null || true

# =============================================================================
# 4. Cache directory — ask where Tortoise should store its model weights
#    (~2–4 GB).  The choice is persisted in .mic-drop.env so subsequent runs
#    and the CLI itself can read it without the user having to re-type it.
# =============================================================================
step "Cache directory"

# Helper: read a line from the terminal even when stdin is a pipe.
prompt() {
    # $1 = prompt text
    if [[ -t 0 ]]; then
        read -r -p "$1" REPLY
    else
        # Non-interactive (piped stdin) — print the prompt, default to empty
        printf "%s" "$1" >&2
        REPLY=""
    fi
}

_read_existing_cache() {
    # Returns the value of TORTOISE_CACHE_DIR from .mic-drop.env, or empty.
    if [[ -f "$ENV_FILE" ]]; then
        grep -m1 '^TORTOISE_CACHE_DIR=' "$ENV_FILE" | cut -d= -f2-
    fi
}

EXISTING_CACHE="$(_read_existing_cache)"

if [[ -n "$EXISTING_CACHE" ]]; then
    # Already configured — show current value and let user keep or change it.
    info "Current cache directory: ${EXISTING_CACHE}"
    prompt "  Keep this path? [Y/n] "
    if [[ "$REPLY" =~ ^[nN] ]]; then
        EXISTING_CACHE=""   # fall through to the chooser below
    else
        CACHE_DIR="$EXISTING_CACHE"
        ok "Using existing cache directory."
    fi
fi

if [[ -z "$EXISTING_CACHE" ]]; then
    printf "\n"
    printf "  Tortoise downloads ~2–4 GB of model weights on first run.\n"
    printf "  Where should they be stored?\n\n"
    printf "    ${BOLD}1${NC}  Inside this project    (${SCRIPT_DIR}/tortoise_cache)\n"
    printf "    ${BOLD}2${NC}  On a USB / thumb drive (you'll enter the path)\n"
    printf "    ${BOLD}3${NC}  macOS default          (~/.cache/tortoise-tts)\n\n"

    prompt "  Your choice [1/2/3] (default 1): "
    CHOICE="${REPLY:-1}"

    case "$CHOICE" in
        1)
            CACHE_DIR="${SCRIPT_DIR}/tortoise_cache"
            ;;
        2)
            printf "\n"
            prompt "  Enter the path to the cache directory on your USB drive: "
            CACHE_DIR="$REPLY"
            if [[ -z "$CACHE_DIR" ]]; then
                warn "No path entered — falling back to project-local cache."
                CACHE_DIR="${SCRIPT_DIR}/tortoise_cache"
            fi
            ;;
        3)
            CACHE_DIR="${HOME}/.cache/tortoise-tts"
            ;;
        *)
            warn "Unrecognised choice '${CHOICE}' — falling back to project-local cache."
            CACHE_DIR="${SCRIPT_DIR}/tortoise_cache"
            ;;
    esac

    # Persist the choice
    printf "TORTOISE_CACHE_DIR=%s\n" "$CACHE_DIR" > "$ENV_FILE"
    ok "Cache directory set to: ${CACHE_DIR}"
    info "Saved to .mic-drop.env (loaded automatically by the CLI)."
fi

# Create the directory now so the user gets immediate feedback
mkdir -p "$CACHE_DIR"

# =============================================================================
# 5. PyTorch + torchaudio
#    Apple Silicon (arm64):
#        Tortoise needs ops that the stable macOS wheel doesn't always expose.
#        The nightly CPU build covers all MPS-fallback paths.
#        PYTORCH_ENABLE_MPS_FALLBACK=1 lets Metal ops silently fall back to CPU
#        when the kernel isn't implemented yet.
#    Intel Mac:
#        The default PyPI (stable) wheel is fine.
# =============================================================================
step "PyTorch"

export PYTORCH_ENABLE_MPS_FALLBACK=1

if python -c "import torch" 2>/dev/null; then
    TORCH_VER="$(python -c "import torch; print(torch.__version__)")"

    # On arm64 the nightly build is required.  If the user has a stable build
    # we need to upgrade — check for the '+' that nightlies carry in their
    # version string (e.g. "2.3.0.dev20240101+cpu").
    if [[ "$ARCH" == "arm64" ]]; then
        if [[ "$TORCH_VER" != *"+"* ]] && [[ "$TORCH_VER" != *"dev"* ]]; then
            warn "Stable PyTorch ${TORCH_VER} detected on Apple Silicon."
            warn "  Upgrading to nightly build for full MPS-fallback support …"
            pip install --pre torch torchaudio \
                --index-url https://download.pytorch.org/whl/nightly/cpu \
                --quiet 2>/dev/null || true
            TORCH_VER="$(python -c "import torch; print(torch.__version__)")"
            ok "PyTorch upgraded to ${TORCH_VER}"
        else
            ok "PyTorch ${TORCH_VER} (nightly) — already installed."
        fi
    else
        ok "PyTorch ${TORCH_VER} — already installed."
    fi
else
    if [[ "$ARCH" == "arm64" ]]; then
        info "Downloading nightly PyTorch for Apple Silicon (MPS fallback) …"
        pip install --pre torch torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cpu \
            --quiet
    else
        info "Downloading stable PyTorch for Intel Mac …"
        pip install torch torchaudio --quiet
    fi
    TORCH_VER="$(python -c "import torch; print(torch.__version__)")"
    ok "Installed PyTorch ${TORCH_VER}"
fi

# =============================================================================
# 6. mic-drop package  (editable link + runtime deps)
# =============================================================================
step "mic-drop core"

# Editable link without pulling heavy optional deps (tortoise / rvc).
# Those are installed individually in steps 7-8 so a failure in one doesn't
# poison the other.
pip install --no-deps -e "${SCRIPT_DIR}" --quiet
ok "mic-drop linked in editable mode"

# Remaining runtime deps (torch already present from step 5)
pip install numpy soundfile tqdm torchaudio --quiet
ok "Runtime dependencies installed"

# =============================================================================
# 7. Tortoise TTS  —  installed from GitHub source for best Apple Silicon
#    compatibility.  Pre-dependencies are installed first so that
#    tortoise-tts's setup.py doesn't choke on missing build tools.
# =============================================================================
step "Tortoise TTS"

install_tortoise() {
    if python -c "import tortoise" 2>/dev/null; then
        ok "tortoise-tts — already installed."
        return 0
    fi

    # Install all runtime deps with current wheels BEFORE the editable
    # install.  tortoise-tts hard-pins old versions of transformers /
    # tokenizers that have no Python 3.14 wheels; --no-deps below
    # prevents those pins from overriding what we install here.
    info "Installing tortoise-tts dependencies …"
    pip install numba inflect psutil transformers tokenizers \
        rotary_embedding_torch progressbar einops unidecode scipy \
        --quiet 2>/dev/null || true

    local CLONE_DIR="${BUILD_DIR}/tortoise-tts"

    if [[ -d "$CLONE_DIR" ]]; then
        info "Updating tortoise-tts source …"
        git -C "$CLONE_DIR" pull --quiet 2>/dev/null || true
    else
        info "Cloning tortoise-tts from GitHub …"
        mkdir -p "$BUILD_DIR"
        if git clone --depth 1 https://github.com/neonbjb/tortoise-tts.git \
                "$CLONE_DIR" --quiet 2>/dev/null; then
            ok "Cloned tortoise-tts source."
        else
            warn "git clone failed — trying PyPI wheel as fallback …"
            if pip install tortoise-tts --quiet 2>/dev/null; then
                ok "tortoise-tts installed from PyPI."
            else
                warn "tortoise-tts could not be installed."
                warn "  Try manually:"
                warn "      pip install numba inflect psutil transformers"
                warn "      git clone --depth 1 https://github.com/neonbjb/tortoise-tts.git"
                warn "      pip install -e tortoise-tts"
            fi
            return 0
        fi
    fi

    info "Installing tortoise-tts from local clone …"
    if pip install -e "$CLONE_DIR" --no-deps --quiet 2>/dev/null; then
        ok "tortoise-tts installed from source."
    else
        warn "pip install from clone failed — trying PyPI wheel as fallback …"
        if pip install tortoise-tts --quiet 2>/dev/null; then
            ok "tortoise-tts installed from PyPI (fallback)."
        else
            warn "tortoise-tts could not be installed."
            warn "  Try manually:  pip install -e ${CLONE_DIR}"
        fi
    fi
    return 0
}

install_tortoise

# =============================================================================
# 8. RVC  —  fairseq requires pip==24.0; pin, install, restore.
# =============================================================================
step "RVC"

install_rvc() {
    if python -c "import rvc_python" 2>/dev/null; then
        ok "rvc-python — already installed."
        return 0
    fi

    # rvc-python hard-pins faiss-cpu==1.7.3 (gone from PyPI) and
    # numpy<=1.23.5 (incompatible with torch 2.x).  Install every dep
    # with a current wheel first, then --no-deps the package itself so
    # those pins are never consulted.  fairseq is installed separately
    # because it requires pip==24.0 (omegaconf<2.1 non-standard spec).

    info "Installing RVC dependencies …"
    pip install faiss-cpu av fastapi ffmpeg-python loguru \
        praat-parselmouth pydantic python-multipart pyworld \
        requests torchcrepe uvicorn librosa resampy \
        --quiet 2>/dev/null || true
    ok "RVC dependencies installed."

    info "Installing fairseq (temporary pip 24.0 pin …)"
    pip install "pip==24.0" --quiet 2>/dev/null || true

    if pip install fairseq --quiet 2>/dev/null; then
        ok "fairseq installed."
    else
        warn "fairseq install failed."
        warn "  Try manually:  pip install pip==24.0 && pip install fairseq"
    fi

    info "  Restoring pip …"
    pip install --upgrade pip --quiet 2>/dev/null || true

    info "Installing rvc-python …"
    if pip install rvc-python --no-deps --quiet 2>/dev/null; then
        ok "rvc-python installed."
    else
        warn "rvc-python could not be installed."
        warn "  Try manually:  pip install rvc-python --no-deps"
    fi
}

install_rvc

# =============================================================================
# 9. Test tooling
# =============================================================================
step "Test dependencies"

pip install pytest --quiet
ok "pytest ready"

# =============================================================================
# 10. Verification — import every package and print a device summary
# =============================================================================
step "Verification"

CORE_PASS=0; CORE_TOTAL=0

check_core() {
    CORE_TOTAL=$((CORE_TOTAL + 1))
    if python -c "import $1" 2>/dev/null; then
        ok "$2";  CORE_PASS=$((CORE_PASS + 1))
    else
        warn "$2 — MISSING (required)"
    fi
}

check_core "torch"         "torch"
check_core "torchaudio"    "torchaudio"
check_core "numpy"         "numpy"
check_core "soundfile"     "soundfile"
check_core "tqdm"          "tqdm"
check_core "tts_pipeline"  "mic-drop"
check_core "tortoise"      "tortoise-tts"
check_core "rvc_python"    "rvc-python"
check_core "fairseq"       "fairseq"
check_core "librosa"       "librosa"

# Device summary (heredoc keeps the Python quoting painless)
printf "\n"
python <<'PYEOF'
import os, platform, torch

cuda = torch.cuda.is_available()
mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
best = "cuda" if cuda else ("mps" if mps else "cpu")
fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")

print("  Device summary:")
print(f"    Architecture          :  {platform.machine()}")
print(f"    CUDA                  :  {cuda}")
print(f"    MPS (Metal)           :  {mps}")
print(f"    MPS fallback enabled  :  {fallback == '1'}")
print(f"    auto  →                  {best}")
if mps:
    print("                             Apple Silicon Metal acceleration is available.")
PYEOF

# Show where the cache ended up
printf "\n"
info "Tortoise cache directory: ${CACHE_DIR}"

# =============================================================================
# 11. Unit tests
# =============================================================================
step "Unit tests"

python -m pytest "${SCRIPT_DIR}/tests" -v --tb=short

# =============================================================================
# 12. Summary
# =============================================================================
printf "\n${CYAN}${BOLD}"
printf "  ╔══════════════════════════════════════╗\n"

if [[ $CORE_PASS -eq $CORE_TOTAL ]]; then
    printf "  ║  ${GREEN}${BOLD}All checks passed${NC}${CYAN}${BOLD}                  ║\n"
else
    printf "  ║  ${RED}${BOLD}Dependencies missing!${NC}${CYAN}${BOLD}               ║\n"
fi

printf "  ╚══════════════════════════════════════╝${NC}\n"

printf "\n  Quick start:\n"
printf "      source venv/bin/activate\n"
printf "      echo \"Hello from mic-drop.\" \\\\\n"
printf "          | mic-drop \\\\\n"
printf "              -o output/test.wav \\\\\n"
printf "              -m models/your_model.pth\n\n"

# Exit non-zero when any required package is missing.
exit $(( CORE_TOTAL - CORE_PASS ))
