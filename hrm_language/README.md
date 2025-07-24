# HRM for Language: Setup & Training Guide

This document provides a definitive guide for setting up the environment and running the language-adapted Hierarchical Reasoning Model (HRM).

## 1. Core Principles & Key Learnings

Our troubleshooting revealed several critical requirements for a stable environment:

*   **Fixed Dependencies are Crucial:** The latest versions of libraries are not always compatible. We must use specific "golden combinations" of PyTorch, xFormers, and NumPy.
*   **Specialized Package Indexes:** PyTorch has its own package repository for CUDA-enabled builds. We must install from this source *first* before other dependencies.
*   **Compiler Backend Matters:** The `torch.compile()` JIT compiler offers massive speedups. However, its default `Inductor` backend requires the `Triton` library, which is only officially supported on Linux. The choice of OS and setup script determines whether this optimization can be enabled.

## 2. Required Files & Commands

Here is the complete, streamlined workflow. All commands should be run from the project's root directory (`/HRM`).

### Step 1: Environment Setup

This step creates the Python virtual environment. **Choose ONE of the following methods.**

#### Option A: Windows Setup (Standard Performance)

Use this method if you are running directly on Windows without WSL2.

*   **File:** `hrm-language/setup_final.ps1`
*   **Command:**
    ```powershell
    .\hrm-language/setup_final.ps1
    ```
*   **Limitation:** This method runs in native Windows and **disables** the `torch.compile()` JIT compiler to ensure compatibility. This will result in **significantly slower** training times.

#### Option B: Recommended High-Performance Setup (WSL2)

This is the **recommended method** for maximum performance.

*   **Prerequisites:** You must have WSL2 installed with a Linux distribution (e.g., Ubuntu) and up-to-date NVIDIA drivers that have CUDA support for WSL2 enabled.
*   **Benefit:** Enables the `torch.compile()` JIT compiler via the `Triton` library, providing a **massive speedup** during training.
*   **File:** `hrm-language/setup_wsl.sh`
*   **Commands:**
    ```bash
    # First, navigate to your project directory inside the WSL2 terminal
    # Example: cd /mnt/f/Github/sapientinc/HRM
    
    # Make the script executable (only needs to be done once)
    chmod +x hrm-language/setup_wsl.sh
    
    # Run the setup script. This will install all dependencies, including Triton.
    ./hrm-language/setup_wsl.sh
    ```

### Step 1.5: Activating the Environment (Optional, for Interactive Use)

After setup is complete, you can "enter" the virtual environment to run commands directly. This is useful for debugging or interactive sessions.

*   **To Activate on WSL2:**
    ```bash
    source hrm_language/.venv/bin/activate
    ```
    *You should see `(hrm_language)` or a similar prefix appear on your command prompt.*

*   **To Activate on Windows PowerShell:**
    ```powershell
    .\hrm_language\.venv\Scripts\Activate.ps1
    ```

*   **To Deactivate:** Simply run the `deactivate` command in your terminal.

*Note: The `uv run` commands shown below do **not** require you to activate the environment first, as `uv` handles it automatically.*

### Step 2: Data Preparation

This step is the same for both Windows and WSL2. It downloads and processes a sample dataset for the model to train on.

*   **File:** `hrm-language/dataset/build_language_dataset.py`
*   **Command (run from project root):**
    ```bash
    # For WSL2 (recommended)
    uv run --python hrm-language/.venv -- python hrm-language/dataset/build_language_dataset.py --output-dir hrm-language/data/test-1k --subsample-size 1000 --seq-len 512
    
    # For Windows (if you used Option A)
    # uv run --python hrm-language/.venv -- python hrm-language/dataset/build_language_dataset.py --output-dir hrm-language/data/test-1k --subsample-size 1000 --seq-len 512
    ```
*   **What it does:** Creates a small dataset with 1,000 examples located at `hrm-language/data/test-1k`.

### Step 3: Model Training

This is the final step. The command is the same for both environments, but the performance will be drastically different.

*   **File:** `hrm-language/pretrain.py`
*   **Command (run from project root):**
    ```bash
    uv run --python hrm-language/.venv -- python hrm-language/pretrain.py data_path=hrm-language/data/test-1k epochs=1000 eval_interval=100 global_batch_size=8
    ```
*   **Note on Batch Size:** A `global_batch_size` of 8 is recommended for consumer GPUs (e.g., RTX 3080 10GB) to ensure there is enough memory for the one-time JIT compilation and autotuning process. If you have a GPU with more VRAM, you can experiment with increasing this value.
*   **Expected Performance:**
    *   **WSL2:** With `torch.compile()` enabled, each step should be very fast.
    *   **Windows:** Without `torch.compile()`, each step will be significantly slower.

### Recommended Cleanup (Optional)

The `hrm-language` directory contains old setup scripts from our initial troubleshooting. To avoid confusion, you can safely delete all `*.ps1` and `*.bat` files **except for `setup_final.ps1`**.

---

## 4. Advanced Troubleshooting

This section covers advanced issues that may arise from a clean environment, particularly when dealing with packages that require compilation.

### Issue 1: `flash-attn` Fails to Build

Building `flash-attn` from source is complex and can fail for several reasons. If you see errors related to `ModuleNotFoundError: No module named 'torch'`, `OSError: CUDA_HOME environment variable is not set`, or `FileNotFoundError: ... /nvcc`, it means your environment is missing build-time dependencies.

#### **Solution: Install Build Dependencies and Re-run**

Here is the full sequence to fix `flash-attn` build errors in a WSL2 environment.

1.  **Install the Matching NVIDIA CUDA Toolkit:** The PyTorch version used in this project was built with CUDA 12.4. To avoid compiler crashes (`Segmentation fault`) when building extensions, you **must** install the matching CUDA Toolkit version.
    ```bash
    # Add the repository key for CUDA 12.4
    wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
    sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
    sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    
    # Update and install the toolkit
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
    ```

2.  **Set `CUDA_HOME` Environment Variable:** This tells the build script where to find the compiler.
    ```bash
    export CUDA_HOME=/usr/local/cuda
    ```
    *Note: To make this permanent, add this line to your `~/.bashrc` or `~/.zshrc` file.*

3.  **Install `flash-attn` without Build Isolation:** This is the final step. The `--no-build-isolation` flag forces the build to use the `torch` already installed in your main environment.
    ```bash
    # First, ensure you have activated the correct environment
    source hrm_language/.venv/bin/activate
    
    # Now, install flash-attn
    uv pip install flash-attn==2.5.8 --no-build-isolation
    ```

### Issue 2: `ModuleNotFoundError` during `pretrain.py`

If you encounter `ModuleNotFoundError` related to `models`, `utils`, or other local files, it's due to how Python handles packages and imports.

*   **Correct Execution:** Always run the training script as a module from the project root: `python -m hrm_language.pretrain ...`. This sets the correct context.
*   **Package Structure:** Ensure that every subdirectory within `hrm_language` that should be a package (e.g., `utils`, `models`, `models/hrm`) contains an empty `__init__.py` file.
*   **Relative Imports:** All imports within the `hrm_language` package must be relative to itself (e.g., `from .models.losses import ...`). This makes the package self-contained and prevents conflicts with other parts of the repository.
