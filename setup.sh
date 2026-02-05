#!/usr/bin/env bash
# =============================================================================
#  mic-drop — macOS setup & bootstrap
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
# 2. Python — prefer Homebrew, fall back to system
# =============================================================================
step "Python"

PYTHON=""
if   [[ -x /opt/homebrew/bin/python3 ]]; then
    PYTHON=/opt/homebrew/bin/python3            # Homebrew — Apple Silicon
elif [[ -x /usr/local/bin/python3   ]]; then
    PYTHON=/usr/local/bin/python3              # Homebrew — Intel
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"             # system / Xcode CLI
fi

[[ -z "$PYTHON" ]] && \
    fail "Python 3.9+ not found.  Install via Homebrew:  brew install python"

# Version gate
read -r PY_MAJ PY_MIN < <(
    "$PYTHON" -c "import sys; print(sys.version_info.major, sys.version_info.minor)"
)
( [[ $PY_MAJ -gt 3 ]] || ( [[ $PY_MAJ -eq 3 ]] && [[ $PY_MIN -ge 9 ]] ) ) \
    || fail "Python 3.9+ required — found ${PY_MAJ}.${PY_MIN}."

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
# 4. PyTorch + torchaudio
#    macOS: the default PyPI wheel already includes MPS Metal support.
#    Do NOT pass --index-url here.
# =============================================================================
step "PyTorch"

if python -c "import torch" 2>/dev/null; then
    TORCH_VER="$(python -c "import torch; print(torch.__version__)")"
    ok "PyTorch ${TORCH_VER} — already installed."
else
    info "Downloading PyTorch for macOS (MPS Metal included) …"
    pip install torch torchaudio --quiet
    TORCH_VER="$(python -c "import torch; print(torch.__version__)")"
    ok "Installed PyTorch ${TORCH_VER}"
fi

# =============================================================================
# 5. mic-drop package  (editable link + runtime deps)
# =============================================================================
step "mic-drop core"

# Editable link without pulling heavy optional deps (tortoise / rvc).
# Those are installed individually in step 6 so a failure in one doesn't
# poison the other.
pip install --no-deps -e "${SCRIPT_DIR}" --quiet
ok "mic-drop linked in editable mode"

# Remaining runtime deps (torch already present from step 4)
pip install numpy soundfile tqdm torchaudio --quiet
ok "Runtime dependencies installed"

# =============================================================================
# 6. ML backends  —  tortoise-tts & rvc-python  (best-effort)
# =============================================================================
step "ML backends"

# Generic single-package installer.  Always returns 0 so set -e doesn't abort.
install_backend() {
    local pip_name="$1"
    local import_name="$2"
    local label="$3"

    if python -c "import ${import_name}" 2>/dev/null; then
        ok "${label} — already installed."
        return 0
    fi

    info "Installing ${label} …"
    if pip install "${pip_name}" --quiet 2>/dev/null; then
        ok "${label} installed."
    else
        warn "${label} could not be installed automatically."
        warn "  Retry manually later:  pip install ${pip_name}"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# RVC + fairseq need pip==24.0.  pip 24.1+ resolves fairseq's dependency
# tree incorrectly and the install fails.  We pin, install from the
# dedicated requirements file, then restore pip.
# ---------------------------------------------------------------------------
install_rvc() {
    if python -c "import rvc" 2>/dev/null; then
        ok "rvc-python — already installed."
        return 0
    fi

    info "Installing RVC (temporary pip 24.0 pin for fairseq …)"

    # Remember what's installed now so we can restore afterwards
    local ORIG_PIP
    ORIG_PIP="$(pip --version | awk '{print $2}')"
    info "  Current pip: ${ORIG_PIP}  →  pinning to 24.0"

    pip install "pip==24.0" --quiet 2>/dev/null || true

    if pip install -r "${SCRIPT_DIR}/requirements-rvc.txt" --quiet 2>/dev/null; then
        ok "rvc-python + fairseq + librosa installed."
    else
        warn "RVC install failed even at pip 24.0."
        warn "  Try the manual steps in requirements-rvc.txt:"
        warn "      pip install pip==24.0"
        warn "      pip install -r requirements-rvc.txt"
        warn "      pip install --upgrade pip"
    fi

    # Restore pip to latest regardless of success/failure
    info "  Restoring pip …"
    pip install --upgrade pip --quiet 2>/dev/null || true
    return 0   # always succeed — RVC is optional
}

install_backend "tortoise-tts"  "tortoise"  "tortoise-tts"
install_rvc

# =============================================================================
# 7. Test tooling
# =============================================================================
step "Test dependencies"

pip install pytest --quiet
ok "pytest ready"

# =============================================================================
# 8. Verification — import every package and print a device summary
# =============================================================================
step "Verification"

CORE_PASS=0; CORE_TOTAL=0
OPT_PASS=0;  OPT_TOTAL=0

check_core() {
    CORE_TOTAL=$((CORE_TOTAL + 1))
    if python -c "import $1" 2>/dev/null; then
        ok "$2";  CORE_PASS=$((CORE_PASS + 1))
    else
        warn "$2 — MISSING (required)"
    fi
}

check_opt() {
    OPT_TOTAL=$((OPT_TOTAL + 1))
    if python -c "import $1" 2>/dev/null; then
        ok "$2";  OPT_PASS=$((OPT_PASS + 1))
    else
        warn "$2 — not installed (optional)"
    fi
}

# Core
check_core "torch"         "torch"
check_core "torchaudio"    "torchaudio"
check_core "numpy"         "numpy"
check_core "soundfile"     "soundfile"
check_core "tqdm"          "tqdm"
check_core "tts_pipeline"  "mic-drop"

# Optional ML backends
check_opt "tortoise"  "tortoise-tts"
check_opt "rvc"       "rvc-python"
check_opt "fairseq"   "fairseq"
check_opt "librosa"   "librosa"

# Device summary (heredoc keeps the Python quoting painless)
printf "\n"
python <<'PYEOF'
import platform, torch

cuda = torch.cuda.is_available()
mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
best = "cuda" if cuda else ("mps" if mps else "cpu")

print("  Device summary:")
print(f"    Architecture :  {platform.machine()}")
print(f"    CUDA         :  {cuda}")
print(f"    MPS (Metal)  :  {mps}")
print(f"    auto  →         {best}")
if mps:
    print("                    Apple Silicon Metal acceleration is available.")
PYEOF

# =============================================================================
# 9. Unit tests
# =============================================================================
step "Unit tests"

python -m pytest "${SCRIPT_DIR}/tests" -v --tb=short

# =============================================================================
# 10. Summary
# =============================================================================
printf "\n${CYAN}${BOLD}"
printf "  ╔══════════════════════════════════════╗\n"

if [[ $CORE_PASS -eq $CORE_TOTAL ]]; then
    if [[ $OPT_PASS -eq $OPT_TOTAL ]]; then
        printf "  ║  ${GREEN}${BOLD}All checks passed${NC}${CYAN}${BOLD}                  ║\n"
    else
        printf "  ║  ${GREEN}${BOLD}Core OK${NC}${CYAN}${BOLD} — some optional pkgs missing  ║\n"
    fi
else
    printf "  ║  ${RED}${BOLD}Core dependencies missing!${NC}${CYAN}${BOLD}           ║\n"
fi

printf "  ╚══════════════════════════════════════╝${NC}\n"

printf "\n  Quick start:\n"
printf "      source venv/bin/activate\n"
printf "      echo \"Hello from mic-drop.\" \\\\\n"
printf "          | python -m tts_pipeline \\\\\n"
printf "              -o output/test.wav \\\\\n"
printf "              -m models/your_model.pth\n\n"

if [[ $OPT_PASS -lt $OPT_TOTAL ]]; then
    printf "  ${YELLOW}⚠  One or more optional ML backends are missing.\n"
    printf "     Install them manually and re-run ./setup.sh to verify.${NC}\n\n"
fi

# Exit non-zero only when a *core* package is missing.
exit $(( CORE_TOTAL - CORE_PASS ))
