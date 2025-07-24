#!/bin/bash
set -e

# --- (MANDATORY PREREQUISITE) ---
# This script requires the NVIDIA CUDA Toolkit version 12.4 to be installed
# in your WSL2 environment to successfully compile the 'flash-attn' package.
# Your PyTorch version was built against CUDA 12.4, and a mismatch
# can cause a compiler crash (Segmentation Fault).
# Please see the Troubleshooting section in the README.md for installation instructions.
# ---

# --- Configuration for WSL2 (Linux) ---
PYTHON_VERSION="3.11"
TORCH_VERSION="2.6.0"
CUDA_SUFFIX="cu124" # Corresponds to your system's CUDA 12.4
XFORMERS_VERSION="0.0.29.post2"
NUMPY_VERSION="1.26.4"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_SUFFIX}"

echo "--- Building the High-Performance HRM Language Environment for WSL2 ---"

# Navigate to the script's directory to ensure paths are correct
# This allows running the script from anywhere, e.g., bash hrm-language/setup_wsl.sh
cd "$(dirname "$0")"

# 1. Clean Slate: Remove the old Windows venv
if [ -d ".venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
fi

# 2. Create a fresh virtual environment
echo "Creating a new Python $PYTHON_VERSION virtual environment..."
uv venv --python $PYTHON_VERSION

# 3. Step 1: Install PyTorch family from the special index URL
# The index URL ensures we get the correct CUDA-enabled build.
echo "Step 1: Installing PyTorch family from the dedicated PyTorch repository..."
uv pip install --python .venv \
    torch==$TORCH_VERSION \
    torchvision \
    torchaudio \
    --index-url $PYTORCH_INDEX_URL

# 4. Step 2: Install high-performance libraries and other dependencies from PyPI
echo "Step 2: Installing Triton, xFormers, and all other dependencies..."
uv pip install --python .venv \
    "triton" \
    "setuptools" \
    "tiktoken" \
    "flash-attn" \
    "xformers==$XFORMERS_VERSION" \
    "numpy==$NUMPY_VERSION" \
    "datasets==$DATASETS_VERSION" \
    "transformers" "datasets" "tokenizers" \
    "adam-atan2" "einops" "tqdm" "wandb" \
    "omegaconf" "hydra-core" "pydantic" "argdantic" \
    "coolname" "huggingface-hub" "matplotlib" "numba" "psutil"

# 5. Final, Rigorous Verification
echo "Verifying the final optimized environment..."
VERIFY_CODE=$(cat <<EOF
import torch
import numpy
import os

print('--- WSL2 Environment Verification ---')
print(f'PyTorch Version: {torch.__version__}')
print(f'NumPy Version: {numpy.__version__}')

cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')

if not cuda_available:
    print('>>> ERROR: CUDA is not available. Check NVIDIA driver and WSL2 CUDA setup.')
else:
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'CUDA Version (from PyTorch): {torch.version.cuda}')

    print('\n--- JIT Compiler Backend ---')
    try:
        import triton
        print(f'>>> SUCCESS: Triton version {triton.__version__} is installed.')
        print('    Performance Level: OPTIMAL (Inductor backend enabled)')
    except ImportError:
        print('>>> WARNING: Triton not found. Performance will be suboptimal.')

    print('\n--- Attention Optimizations ---')
    try:
        import xformers
        print(f'>>> SUCCESS: xFormers version {xformers.__version__} is installed and ready.')
    except ImportError:
        print('>>> WARNING: xFormers not found.')

print('\n--- Final Status ---')
if cuda_available:
    print('>>> Environment is correctly configured for high-performance GPU training on WSL2!')
else:
    print('>>> Environment is NOT correctly configured. Please review errors.')
EOF
)

uv run --python .venv python -c "$VERIFY_CODE"

echo ""
echo "--- WSL2 Setup complete. To activate, run: source .venv/bin/activate ---" 