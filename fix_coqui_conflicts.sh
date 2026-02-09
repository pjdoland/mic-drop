#!/usr/bin/env bash
# Fix dependency conflicts after installing Coqui TTS

set -euo pipefail

echo "Fixing dependency conflicts..."

# Reinstall critical packages with correct versions
# Order matters: PyTorch first (stable version for torchcodec compatibility),
# then other deps
pip install 'torch==2.5.1' 'torchaudio==2.5.1' --upgrade
pip install 'torchcodec==0.1.0'
pip install 'scipy>=1.11.2' --upgrade
pip install 'transformers==4.31.0' --force-reinstall
pip install 'tokenizers==0.13.3' --force-reinstall
pip install 'numpy==1.23.5' --force-reinstall
pip install 'faiss-cpu==1.7.3' --force-reinstall
pip install 'TTS==0.21.3' --force-reinstall
pip install 'setuptools<81' --force-reinstall

echo "✓ Dependencies fixed"
echo ""
echo "Testing imports..."
python -c "import torch; import numpy; import scipy; print('✓ Core packages OK')"
python -c "import tortoise; print('✓ Tortoise OK')"
python -c "import rvc_python; print('✓ RVC OK')"
python -c "from TTS.api import TTS; print('✓ Coqui TTS OK')"
python -c "import tts_pipeline; print('✓ mic-drop OK')"

echo ""
echo "All packages should now work together!"
