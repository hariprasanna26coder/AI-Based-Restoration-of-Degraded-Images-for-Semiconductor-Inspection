"""
verify_setup.py
---------------
Quick sanity check — run this FIRST to verify your environment.
Does NOT require a GPU. Runs a complete forward + backward pass on CPU.

Usage:
    python verify_setup.py

Expected output (all lines with ✓):
    ✓ PyTorch import OK
    ✓ Model import OK
    ✓ Dataset import OK
    ✓ Loss import OK
    ✓ Forward pass OK  (B, 1, 256, 256)
    ✓ Backward pass OK
    ✓ Data loading OK  (noisy: [B, 1, 64, 64]  gt: [B, 1, 128, 128])
    ✓ ALL CHECKS PASSED
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

errors = []

# ── 1. PyTorch ─────────────────────────────────────────────────────────────
try:
    import torch
    import numpy as np
    print(f"✓ PyTorch import OK  (version {torch.__version__})")
except ImportError as e:
    errors.append(f"✗ PyTorch import FAILED: {e}")
    print(errors[-1])

# ── 2. Model ───────────────────────────────────────────────────────────────
try:
    from model.network import SemiRestoreNet, count_params
    net = SemiRestoreNet(in_ch=1, out_ch=1, base_ch=64, bottleneck_blocks=6, scale=2)
    n_params = count_params(net)
    print(f"✓ Model import OK  ({n_params/1e6:.2f}M params)")
except Exception as e:
    errors.append(f"✗ Model import FAILED: {e}")
    print(errors[-1])

# ── 3. Dataset + Transforms ────────────────────────────────────────────────
try:
    from data.transforms import TrainTransform, ValTransform
    from data.dataset import NpyPairDataset
    print("✓ Dataset import OK")
except Exception as e:
    errors.append(f"✗ Dataset import FAILED: {e}")
    print(errors[-1])

# ── 4. Loss ────────────────────────────────────────────────────────────────
try:
    from model.losses import RestorationLoss
    criterion = RestorationLoss()
    print("✓ Loss import OK")
except Exception as e:
    errors.append(f"✗ Loss import FAILED: {e}")
    print(errors[-1])

# ── 5. Forward pass ────────────────────────────────────────────────────────
try:
    dummy_in = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        out = net(dummy_in)
    assert out.shape == (2, 1, 256, 256), f"Wrong output shape: {out.shape}"
    print(f"✓ Forward pass OK  (output shape: {list(out.shape)})")
except Exception as e:
    errors.append(f"✗ Forward pass FAILED: {e}")
    print(errors[-1])

# ── 6. Backward pass (loss + grad) ────────────────────────────────────────
try:
    dummy_in  = torch.randn(2, 1, 128, 128)
    dummy_gt  = torch.rand(2, 1, 256, 256)
    out  = net(dummy_in)
    pred = torch.clamp(out, 0.0, 1.0)
    total, _ = criterion(pred, dummy_gt)
    total.backward()
    # Check that gradients flowed
    grad_ok = all(p.grad is not None for p in net.parameters() if p.requires_grad)
    assert grad_ok, "Some parameters have no gradient"
    net.zero_grad()
    print(f"✓ Backward pass OK  (loss={total.item():.4f})")
except Exception as e:
    errors.append(f"✗ Backward pass FAILED: {e}")
    print(errors[-1])

# ── 7. Data loading (if data exists) ──────────────────────────────────────
noisy_dir = os.path.join(os.path.dirname(__file__), 'train', 'train', 'NoisyLR')
gt_dir    = os.path.join(os.path.dirname(__file__), 'train', 'train', 'GT')

if os.path.exists(noisy_dir) and os.path.exists(gt_dir):
    try:
        from data.dataset import get_dataloaders
        tl, vl, _, _ = get_dataloaders(noisy_dir, gt_dir, val_split=0.1, batch_size=2, seed=42)
        batch = next(iter(tl))
        n, g = batch['noisy'], batch['gt']
        assert n.ndim == 4 and g.ndim == 4, "Tensors must be 4D"
        assert n.dtype == torch.float32
        assert n.min() >= 0.0 and n.max() <= 1.01  # allow tiny fp error
        print(f"✓ Data loading OK  (noisy:{list(n.shape)}  gt:{list(g.shape)})")
    except Exception as e:
        errors.append(f"✗ Data loading FAILED: {e}")
        print(errors[-1])
else:
    print(f"⚠ Data not found at {noisy_dir}  — skipping data loading check")

# ── 8. Metrics ────────────────────────────────────────────────────────────
try:
    from utils.metrics import compute_metrics
    p = torch.rand(1, 1, 256, 256)
    g = torch.rand(1, 1, 256, 256)
    m = compute_metrics(p, g)
    print(f"✓ Metrics OK  (PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.4f})")
except Exception as e:
    errors.append(f"✗ Metrics FAILED: {e}")
    print(errors[-1])

# ── Summary ───────────────────────────────────────────────────────────────
print()
if not errors:
    print("✅ ALL CHECKS PASSED — ready to train!")
    print()
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU detected: {gpu}  ({mem:.1f} GB VRAM)")
        print(f"   Run:  python train.py --config configs/train_config.yaml")
    else:
        print("   ⚠  No GPU detected. Training will be very slow on CPU.")
        print("   → Use Google Colab (free T4) or Kaggle (free P100) for training.")
        print("   → Upload the entire '1 final' folder to Colab and run train.py there.")
else:
    print(f"❌ {len(errors)} check(s) FAILED. Fix the above errors before training.")
    print("   Usually caused by missing pip packages. Run:")
    print("   pip install -r requirements.txt")
