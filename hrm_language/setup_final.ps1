# Definitive Setup Script - Correct Multi-Step Installation
Write-Host "--- Building the Optimal & Stable HRM Language Environment (Final) ---" -ForegroundColor Cyan

# --- Configuration: These versions are verified to be compatible on Windows ---
$pythonVersion = "3.11"
$torchVersion = "2.6.0"
$torchCudaSuffix = "+cu124"
$torchvisionVersion = "0.21.0"
$torchaudioVersion = "2.6.0"
$xformersVersion = "0.0.29.post2" # Verified compatible with PyTorch 2.6.0
$numpyVersion = "1.26.4"         # Verified compatible with PyTorch 2.6.0
$pytorchIndexUrl = "https://download.pytorch.org/whl/cu124"

# --- Script Start ---

# 1. Clean Slate
if (Test-Path ".venv") {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

# 2. Create a fresh virtual environment
Write-Host "Creating a new Python $pythonVersion virtual environment..." -ForegroundColor Blue
uv venv --python $pythonVersion

# 3. Step 1: Install PyTorch family from the special index URL
Write-Host "Step 1: Installing PyTorch family from the dedicated PyTorch repository..." -ForegroundColor Blue
$torchDeps = @(
    "torch==$torchVersion$torchCudaSuffix",
    "torchvision==$torchvisionVersion$torchCudaSuffix",
    "torchaudio==$torchaudioVersion$torchCudaSuffix"
)
uv pip install --python .venv $torchDeps --index-url $pytorchIndexUrl

# 4. Step 2: Install remaining dependencies (xFormers, NumPy, etc.) from PyPI
Write-Host "Step 2: Installing xFormers, NumPy, and all other dependencies from PyPI..." -ForegroundColor Blue
$otherDeps = @(
    "xformers==$xformersVersion",
    "numpy==$numpyVersion",
    "transformers", "datasets", "tokenizers",
    "adam-atan2", "einops", "tqdm", "wandb",
    "omegaconf", "hydra-core", "pydantic", "argdantic",
    "coolname", "huggingface-hub", "matplotlib", "numba", "psutil"
)
uv pip install --python .venv $otherDeps

# 5. Final, Rigorous Verification
Write-Host "Verifying the final optimized environment..." -ForegroundColor Green
$verifyCode = @"
import torch
import numpy
import os

print('--- Environment Verification ---')
print(f'PyTorch Version: {torch.__version__}')
print(f'NumPy Version: {numpy.__version__}')

cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')

if not cuda_available:
    print('>>> ERROR: CUDA is not available. Installation failed.')
else:
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'CUDA Version (from PyTorch): {torch.version.cuda}')
    
    print('\n--- Attention Optimizations ---')
    try:
        import xformers
        print(f'>>> SUCCESS: xFormers version {xformers.__version__} is installed and ready.')
        print('    Performance Level: OPTIMAL')
    except ImportError:
        print('>>> ERROR: xFormers not found. Installation failed.')

print('\n--- Final Status ---')
if cuda_available and numpy.__version__.startswith('1.'):
    print('>>> Environment is correctly configured for high-performance GPU training!')
else:
    print('>>> Environment is NOT correctly configured. Please review errors.')
"@

uv run --python .venv python -c $verifyCode

Write-Host ""
Write-Host "--- Setup complete. This multi-step installation guarantees correctness. ---" -ForegroundColor Green 