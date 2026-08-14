# SemiRestoreNet: Joint Noise Removal & 2x Super-Resolution

---

## Table of Contents

- [Submission Checklist & Repository Map](#submission-checklist--repository-map)
- [Repository Structure](#repository-structure)
- [Quick Start for Reviewers (3 Steps)](#quick-start-for-reviewers-3-steps)
- [Full Environment Setup](#full-environment-setup)
  - [Prerequisites](#prerequisites)
  - [Option A - Virtual Environment (Recommended)](#option-a---virtual-environment-recommended)
  - [Option B - Conda Environment](#option-b---conda-environment)
  - [Option C - Direct Install (No Virtual Env)](#option-c---direct-install-no-virtual-env)
- [Environment Verification](#environment-verification)
- [Evaluation / Inference](#evaluation--inference)
  - [Test Set Inference (No Ground Truth)](#test-set-inference-no-ground-truth)
  - [Quantitative Evaluation (With Ground Truth)](#quantitative-evaluation-with-ground-truth)
  - [Custom Model Weights](#custom-model-weights)
  - [Quick Benchmark (Subset of Images)](#quick-benchmark-subset-of-images)
  - [Skip PNG Thumbnails (Faster I/O)](#skip-png-thumbnails-faster-io)
  - [All Evaluation Arguments](#all-evaluation-arguments)
- [Training from Scratch](#training-from-scratch)
  - [Train with Default Config](#train-with-default-config)
  - [Train with Extended Config (3200-step schedule)](#train-with-extended-config-3200-step-schedule)
  - [Resume Training from a Checkpoint](#resume-training-from-a-checkpoint)
  - [Training Output](#training-output)
- [Visualization](#visualization)
- [Model Architecture](#model-architecture)
- [Troubleshooting](#troubleshooting)
  - [1. ModuleNotFoundError: No module named 'torch'](#1-modulenotfounderror-no-module-named-torch)
  - [2. ModuleNotFoundError: No module named 'lpips'](#2-modulenotfounderror-no-module-named-lpips)
  - [3. CUDA out of memory](#3-cuda-out-of-memory--runtimeerror-cuda-error-out-of-memory)
  - [4. FileNotFoundError: Trained_Model_Weights.pt not found](#4-filenotfounderror-trained_model_weightspt-not-found)
  - [5. No .npy files found in input directory](#5-no-npy-files-found-in-input-directory)
  - [6. ValueError: could not broadcast input array](#6-valueerror-could-not-broadcast-input-array-from-shape-)
  - [7. pip install fails (version conflict)](#7-pip-install--r-requirementstxt-fails-version-conflict)
  - [8. Slow inference on CPU](#8-slow-inference-on-cpu)
  - [9. PermissionError when writing outputs (Windows)](#9-permissionerror-when-writing-outputs-windows)
  - [10. verify_setup.py fails at Forward Pass](#10-verify_setuppy-fails-at-forward-pass)
  - [11. Git LFS - Large File Warning](#11-git-lfs---large-file-warning)
  - [12. ImportError on Windows with scikit-image](#12-importerror-on-windows-with-scikit-image)
  - [13. PowerShell cannot activate venv — scripts disabled (Windows)](#13-powershell-cannot-activate-venv--scripts-disabled-windows)
  - [14. evaluate.py: error: unrecognized arguments: \ \ (Windows PowerShell)](#14-evaluatepy-error-unrecognized-arguments---windows-powershell)
  - [Full Environment Reset (Nuclear Option)](#full-environment-reset-nuclear-option)
- [Reproducing Benchmark Results](#reproducing-benchmark-results)
- [License](#license)

---

## Submission Checklist & Repository Map

This repository satisfies all **6 required deliverables** for the KLA benchmark evaluation (Component 2):

| # | Required Item | File / Folder | Status |
|---|---|---|---|
| 1 | README.md | `README.md` | Complete setup & inference instructions |
| 2 | Evaluation Script (standalone `.py`) | `evaluate.py` | Accepts `--input_dir` & `--output_dir`, no edits required |
| 3 | Training Script | `train.py` | Reproduces full training from scratch |
| 4 | Trained Model Weights | `Trained_Model_Weights.pt` | 14.27M params, 57.1 MB PyTorch `.pt` |
| 5 | Restored Test Outputs | `Restored_Test_Outputs/` | 400 float32 `.npy` arrays (256x256) |
| 6 | requirements.txt | `requirements.txt` | Complete pip dependencies |

> **CRITICAL**: The evaluation script runs **AS-IS** with no manual edits. Tested on a clean environment before submission.

---

## Repository Structure

```
.
├── README.md                    # This file - full setup, eval & troubleshooting guide
├── evaluate.py                  # Main evaluation/inference script (standalone)
├── train.py                     # Training reproduction script
├── verify_setup.py              # Environment & model sanity checker
├── visualize_comparison.py      # Side-by-side input vs. restored visualization
├── Trained_Model_Weights.pt     # Final trained model checkpoint (57.1 MB)
├── requirements.txt             # Python package dependencies
├── Restored_Test_Outputs/       # Pre-computed restored test outputs (.npy + .png)
│   ├── 000000.npy
│   ├── 000000.png
│   └── ... (400 images total)
├── configs/
│   ├── train_config.yaml        # Primary training configuration
│   └── train_config_3200.yaml   # Extended training config (3200-step schedule)
├── model/
│   ├── __init__.py
│   ├── network.py               # SemiRestoreNet architecture definition
│   ├── blocks.py                # NAFBlock & sub-pixel upsample modules
│   └── losses.py                # Combined loss functions (L1 + perceptual + SSIM)
├── data/
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset for NoisyLR/GT .npy pairs
│   └── transforms.py            # Augmentation & normalization transforms
└── utils/
    ├── __init__.py
    ├── metrics.py               # PSNR, SSIM, LPIPS metric computation
    └── visualization.py         # Image utility & thumbnail generation
```

---

## Quick Start for Reviewers (3 Steps)

A reviewer must be able to clone and run inference without contacting the author.

```bash
# Step 1 - Clone the repository
git clone https://github.com/hariprasanna26coder/AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection.git
cd AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspections

# Step 2 - Install dependencies
pip install -r requirements.txt

# Step 3 - Run inference (AS-IS, no edits needed)
python evaluate.py --input_dir /path/to/Test_NoisyLR/NoisyLR --output_dir outputs/restored
```

Restored `.npy` files and `.png` thumbnails will appear in `outputs/restored/`.

---

## Full Environment Setup

### Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.8 | 3.10+ recommended |
| PyTorch | 2.0.0 | CUDA build strongly preferred |
| CUDA | 11.7 | Optional - CPU fallback supported |
| RAM | 8 GB | 16 GB recommended for training |
| Disk | 2 GB free | For model + outputs |

### Option A - Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate - Linux / macOS
source .venv/bin/activate

# Activate - Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate - Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# Install all dependencies
pip install -r requirements.txt
```

### Option B - Conda Environment

```bash
# Create conda environment
conda create -n semirestore python=3.10 -y
conda activate semirestore

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -r requirements.txt
```

### Option C - Direct Install (No Virtual Env)

```bash
pip install -r requirements.txt
```

---

## Environment Verification

Before running evaluation or training, verify your setup is correct:

```bash
python verify_setup.py
```

**Expected output:**

```
✓ PyTorch import OK
✓ Model import OK (14.27M params)
✓ Forward pass OK (output shape: [2, 1, 256, 256])
✓ Backward pass OK
✓ Metrics OK
ALL CHECKS PASSED - ready to run!
```

To also verify CUDA availability:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Evaluation / Inference

> **💡 Quick Tip for PowerShell vs. Bash**:
> - **Single-line command** (recommended across all platforms):
>   `python evaluate.py --input_dir Test_NoisyLR/NoisyLR --output_dir outputs/restored`
> - On **Linux / macOS (Bash)**, multi-line commands use backslash: `\`
> - On **Windows PowerShell**, multi-line commands use backtick: `` ` `` (or just run on a single line)

### Test Set Inference (No Ground Truth)

Runs inference on all `.npy` files in `--input_dir` and writes restored 256x256 float32 `.npy` arrays (plus `.png` thumbnails) to `--output_dir`:

```bash
# Single line (Universal - Recommended)
python evaluate.py --input_dir Test_NoisyLR/NoisyLR --output_dir outputs/restored

# Multi-line (Windows PowerShell)
python evaluate.py `
    --input_dir Test_NoisyLR/NoisyLR `
    --output_dir outputs/restored

# Multi-line (Linux / macOS Bash)
python evaluate.py \
    --input_dir Test_NoisyLR/NoisyLR \
    --output_dir outputs/restored
```

### Quantitative Evaluation (With Ground Truth)

When ground truth `.npy` files are available, pass `--gt_dir` to compute **PSNR**, **SSIM**, and **LPIPS**:

```bash
# Single line
python evaluate.py --input_dir train/train/NoisyLR --output_dir outputs/val_restored --gt_dir train/train/GT
```

Per-image metrics and summary statistics are saved to `outputs/val_restored/metrics.json`.

### Custom Model Weights

If you wish to load an alternative checkpoint:

```bash
python evaluate.py --input_dir Test_NoisyLR/NoisyLR --output_dir outputs/custom_restored --weights path/to/your_checkpoint.pt
```

*(Note: If `--weights` is not specified, `evaluate.py` automatically detects and loads `Trained_Model_Weights.pt` from the repository root).*

### Quick Benchmark (Subset of Images)

Process only the first N images for a quick speed test:

```bash
python evaluate.py --input_dir Test_NoisyLR/NoisyLR --output_dir outputs/quick_test --max_samples 10
```

### Skip PNG Thumbnails (Faster I/O)

```bash
python evaluate.py --input_dir Test_NoisyLR/NoisyLR --output_dir outputs/restored --no_png
```

### All Evaluation Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input_dir` | `str` | **Required** | Path to directory with input noisy `.npy` files |
| `--output_dir` | `str` | **Required** | Destination directory for restored `.npy` outputs |
| `--weights` | `str` | `Trained_Model_Weights.pt` | Path to PyTorch model checkpoint |
| `--gt_dir` | `str` | `None` | Path to ground truth `.npy` directory (optional) |
| `--no_png` | flag | `False` | Suppress `.png` thumbnail generation |
| `--max_samples` | `int` | `None` | Limit inference to first N samples |

---

## Training from Scratch

### Train with Default Config

```bash
python train.py --config configs/train_config.yaml
```

### Train with Extended Config (3200-step schedule)

```bash
python train.py --config configs/train_config_3200.yaml
```

### Resume Training from a Checkpoint

To resume training from an intermediate epoch checkpoint or fine-tune from existing weights:

```bash
# Single line (Universal - Recommended)
python train.py --config configs/train_config.yaml --resume saved_models/checkpoint_epoch_0030.pt

# Windows PowerShell (Multi-line)
python train.py `
    --config configs/train_config.yaml `
    --resume saved_models/checkpoint_epoch_0030.pt

# Linux / macOS Bash (Multi-line)
python train.py \
    --config configs/train_config.yaml \
    --resume saved_models/checkpoint_epoch_0030.pt
```

> **Note**: 
> - Intermediate epoch checkpoints (e.g., `saved_models/checkpoint_epoch_XXXX.pt`) are automatically generated in `saved_models/` during training every 10 epochs.
> - You can also pass `--resume Trained_Model_Weights.pt` to fine-tune directly from the final released model weights.
> - Training requires the training dataset folders (`train/train/NoisyLR` and `train/train/GT`).

### Training Output

| Output | Location | Description |
|---|---|---|
| Best model | `saved_models/best_model.pt` | Lowest validation loss checkpoint |
| Epoch checkpoints | `saved_models/checkpoint_epoch_XXXX.pt` | Saved every N epochs per config |
| Training logs | stdout / console | Loss, PSNR, SSIM per epoch |

---

## Visualization

Generate side-by-side comparisons of noisy input vs. restored output:

```bash
# Single line (Recommended)
python visualize_comparison.py --input_dir Test_NoisyLR/NoisyLR --restored_dir outputs/restored --output_dir outputs/comparisons

# Windows PowerShell (Multi-line)
python visualize_comparison.py `
    --input_dir Test_NoisyLR/NoisyLR `
    --restored_dir outputs/restored `
    --output_dir outputs/comparisons

# Linux / macOS Bash (Multi-line)
python visualize_comparison.py \
    --input_dir Test_NoisyLR/NoisyLR \
    --restored_dir outputs/restored \
    --output_dir outputs/comparisons
```

---

## Model Architecture

SemiRestoreNet is a UNet-style encoder-decoder with NAFBlocks and PixelShuffle upsampling:

| Component | Details |
|---|---|
| **Input** | 128x128 noisy low-resolution `.npy` (float32, 1-channel) |
| **Output** | 256x256 clean high-resolution `.npy` (float32, 1-channel) |
| **Parameters** | 14.27M |
| **Normalization** | Per-image sigma-clipping (clip_sigma=3.0) |
| **Encoder** | Strided conv channels: 64 -> 128 -> 256 -> 512 |
| **Bottleneck** | 6 stacked NAFBlocks |
| **Decoder** | Transposed conv + skip connections |
| **Head** | PixelShuffle x2 sub-pixel convolution |
| **Loss** | L1 + Perceptual (VGG) + SSIM |

---

## Troubleshooting

### 1. `ModuleNotFoundError: No module named 'torch'`

**Cause**: PyTorch not installed, or wrong environment is active.

```bash
# Check which Python is active
python --version
which python        # Linux/macOS
where python        # Windows

# Reinstall inside the correct environment
pip install torch>=2.0.0 torchvision>=0.15.0

# For a specific CUDA version (e.g., CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### 2. `ModuleNotFoundError: No module named 'lpips'`

```bash
pip install lpips>=0.1.4

# If lpips install fails due to build errors:
pip install lpips --no-build-isolation
```

---

### 3. `CUDA out of memory` / `RuntimeError: CUDA error: out of memory`

**Cause**: GPU VRAM insufficient.

```bash
# Check current VRAM usage
nvidia-smi

# Force CPU inference
python evaluate.py --input_dir ... --output_dir ... --device cpu

# Kill other GPU processes, free VRAM, then retry
```

---

### 4. `FileNotFoundError: Trained_Model_Weights.pt not found`

**Cause**: Model weights file missing or wrong path.

```bash
# Check if the file exists
ls Trained_Model_Weights.pt        # Linux/macOS
dir Trained_Model_Weights.pt       # Windows

# Explicitly point to the weights file
python evaluate.py \
    --input_dir Test_NoisyLR/NoisyLR \
    --output_dir outputs/restored \
    --weights Trained_Model_Weights.pt
```

---

### 5. `No .npy files found in input directory`

**Cause**: Wrong `--input_dir` path or directory is empty.

```bash
# Verify input directory structure
ls Test_NoisyLR/NoisyLR/        # Linux/macOS
dir Test_NoisyLR\NoisyLR\       # Windows

# Count .npy files
ls Test_NoisyLR/NoisyLR/*.npy | wc -l                              # Linux/macOS
(Get-ChildItem "Test_NoisyLR\NoisyLR" -Filter "*.npy").Count       # Windows PowerShell
```

---

### 6. `ValueError: could not broadcast input array from shape ...`

**Cause**: Input `.npy` files have unexpected shape or dtype.

```bash
# Inspect your .npy files
python -c "
import numpy as np
f = 'Test_NoisyLR/NoisyLR/000000.npy'
arr = np.load(f)
print('Shape:', arr.shape, 'Dtype:', arr.dtype, 'Min:', arr.min(), 'Max:', arr.max())
"
```

Expected shape: `(128, 128)` or `(1, 128, 128)` with `float32` dtype.

---

### 7. `pip install -r requirements.txt` fails (version conflict)

```bash
# Install with relaxed version constraints, one by one
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install numpy>=1.24.0
pip install Pillow>=9.5.0
pip install scikit-image>=0.20.0
pip install pyyaml>=6.0
pip install tqdm>=4.65.0
pip install matplotlib>=3.7.0
pip install lpips>=0.1.4
```

---

### 8. Slow inference on CPU

**Cause**: No GPU available or CUDA not configured correctly.

```bash
# Verify CUDA status
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Install CUDA-enabled PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### 9. `PermissionError` when writing outputs (Windows)

```bash
# Run PowerShell as Administrator, or use a writable path
python evaluate.py \
    --input_dir Test_NoisyLR\NoisyLR \
    --output_dir C:\Users\YourName\Desktop\restored
```

---

### 10. verify_setup.py fails at Forward Pass

```bash
# Run with detailed traceback
python -u verify_setup.py 2>&1

# Ensure all model files are present
ls model/network.py model/blocks.py model/losses.py    # Linux/macOS
dir model\                                              # Windows
```

---

### 11. Git LFS - Large File Warning

The model weights file (`Trained_Model_Weights.pt`, 57.1 MB) may require Git LFS if cloning fails:

```bash
# Install Git LFS
git lfs install

# Pull LFS files after cloning
git lfs pull

# Or download weights directly
# See the Releases section of this repository for a direct download link
```

---

### 12. `ImportError` on Windows with scikit-image

```bash
pip uninstall scikit-image -y
pip install scikit-image --only-binary :all:
```

---

### 13. PowerShell cannot activate venv — scripts disabled (Windows)

**Error message:**
```
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is
disabled on this system.
```

**Cause**: Windows PowerShell blocks all `.ps1` scripts by default (`Restricted` execution policy).

**Fix** — run this once in PowerShell (no admin required):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

**Verify the policy was applied:**

```powershell
Get-ExecutionPolicy -List
# CurrentUser should now show: RemoteSigned
```

Then activate normally:

```powershell
.\.venv\Scripts\Activate.ps1
```

> **Why `RemoteSigned`?** It allows locally created scripts (like the venv activator) to run freely, while still blocking unsigned scripts downloaded from the internet — a safe, minimal change.

**Alternative** — if you don't want to change the policy permanently, activate for the current session only:

```powershell
powershell -ExecutionPolicy Bypass -File .venv\Scripts\Activate.ps1
```

**Alternative** — use Command Prompt instead of PowerShell (no policy issue):

```cmd
.venv\Scripts\activate.bat
```

---

### 14. `evaluate.py: error: unrecognized arguments: \ \` (Windows PowerShell)

**Error message:**
```text
evaluate.py: error: unrecognized arguments: \ \
```

**Cause**: In Linux / Bash shells, the backslash `\` is used for multi-line command continuation. In **Windows PowerShell**, `\` is treated as a literal text argument rather than a line break. When you copy-paste bash multi-line commands with `\`, PowerShell passes `\` to Python as an argument.

**Fix Option 1 (Recommended)** — Run the command on a single line:

```powershell
python evaluate.py --input_dir D:\Test_NoisyLR --output_dir outputs/restored
```

**Fix Option 2** — Use PowerShell's line continuation character, which is the backtick `` ` `` (not `\`):

```powershell
python evaluate.py `
    --input_dir D:\Test_NoisyLR `
    --output_dir outputs/restored
```

---

### Full Environment Reset (Nuclear Option)

If everything fails, start from a completely clean environment:

```bash
# Deactivate and delete old environment
deactivate
rm -rf .venv                       # Linux/macOS
Remove-Item -Recurse -Force .venv  # Windows PowerShell

# Recreate from scratch
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.\.venv\Scripts\Activate.ps1       # Windows PowerShell

# Fresh install
pip install --upgrade pip
pip install -r requirements.txt

# Verify everything works
python verify_setup.py
```

---

## Reproducing Benchmark Results

To exactly reproduce the submitted benchmark results:

```bash
# 1. Clone repo
git clone https://github.com/hariprasanna26coder/AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspections.git
cd AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspections

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.\.venv\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt

# 3. Verify setup
python verify_setup.py

# 4. Run evaluation on full test set
python evaluate.py \
    --input_dir Test_NoisyLR/NoisyLR \
    --output_dir outputs/restored

# 5. (Optional) Validate outputs match pre-computed Restored_Test_Outputs
python -c "
import numpy as np
ref = np.load('Restored_Test_Outputs/000000.npy')
new = np.load('outputs/restored/000000.npy')
print('Max diff:', np.abs(ref - new).max())
print('Match:', np.allclose(ref, new, atol=1e-5))
"
```

---

## License

This project is licensed under the [MIT License](LICENSE).
