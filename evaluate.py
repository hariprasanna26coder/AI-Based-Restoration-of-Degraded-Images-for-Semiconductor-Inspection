"""
evaluate.py
-----------
KLA i4C Hackathon — Standalone Evaluation Script

USAGE:
    python evaluate.py --input_dir /path/to/Test_NoisyLR/NoisyLR --output_dir /path/to/outputs

WHAT IT DOES:
    1. Loads the trained model from  saved_models/best_model.pt  (auto-detected)
    2. Runs inference on every .npy file in --input_dir
    3. Saves restored images as .npy (float32) to --output_dir
    4. Optionally saves .png thumbnails alongside .npy files
    5. Reports inference time per image

REQUIREMENTS:
    - Python 3.8+
    - torch, numpy, Pillow, scikit-image, pyyaml
    - (optional) lpips  for LPIPS metric

NOTES:
    - This script is self-contained. No manual edits needed to run.
    - Model weights path can be overridden with --weights flag.
    - If ground-truth is available (--gt_dir), PSNR/SSIM/LPIPS are reported.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
from torch.amp import autocast

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from model.network       import SemiRestoreNet
from data.transforms     import ValTransform
from utils.visualization import save_restored_image


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='KLA Image Restoration — Evaluation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference (test set, no GT)
  python evaluate.py --input_dir Test_NoisyLR/NoisyLR --output_dir outputs/restored

  # With ground truth for metric computation
  python evaluate.py \\
      --input_dir Test_NoisyLR/NoisyLR \\
      --output_dir outputs/restored \\
      --gt_dir train/train/GT

  # Custom weights path
    python evaluate.py --input_dir ... --output_dir ... --weights saved_models/checkpoint_epoch_100.pt
        """
    )
    p.add_argument('--input_dir',  type=str, required=True,
                   help='Directory containing NoisyLR .npy files (input)')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Directory to write restored .npy files')
    p.add_argument('--weights',    type=str, default=None,
                   help='Path to model .pt weights (default: saved_models/best_model.pt)')
    p.add_argument('--gt_dir',     type=str, default=None,
                   help='(Optional) Ground truth dir for PSNR/SSIM computation')
    p.add_argument('--save_png',   action='store_true', default=True,
                   help='Also save .png thumbnails alongside .npy outputs')
    p.add_argument('--no_png',     action='store_true',
                   help='Disable .png saving (only .npy)')
    p.add_argument('--batch_size', type=int, default=1,
                   help='Inference batch size (default: 1)')
    p.add_argument('--clip_sigma', type=float, default=3.0,
                   help='Sigma for robust normalization (default: 3.0)')
    p.add_argument('--device',     type=str, default=None,
                   help='Force device: "cuda" or "cpu" (auto-detected by default)')
    p.add_argument('--max_samples', type=int, default=None,
                   help='Limit evaluation to first N samples (e.g. --max_samples 20)')
    # Model architecture (must match trained weights)
    p.add_argument('--base_ch',    type=int, default=64)
    p.add_argument('--bottleneck', type=int, default=6)
    p.add_argument('--scale',      type=int, default=2)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(weights_path: str, device: torch.device, args) -> SemiRestoreNet:
    model = SemiRestoreNet(
        in_ch             = 1,
        out_ch            = 1,
        base_ch           = args.base_ch,
        bottleneck_blocks = args.bottleneck,
        scale             = args.scale,
    )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at: {weights_path}\n"
            f"Please download or train the model first (see README.md)."
        )
    state = torch.load(weights_path, map_location=device, weights_only=False)
    # Handle both raw state_dict and checkpoint dicts
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-image robust normalization (same as training)
# ---------------------------------------------------------------------------
def normalize_image(arr: np.ndarray, clip_sigma: float = 3.0) -> np.ndarray:
    arr = arr.astype(np.float32).squeeze()
    mu  = arr.mean()
    sig = arr.std() + 1e-8
    arr = np.clip(arr, mu - clip_sigma * sig, mu + clip_sigma * sig)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return arr


# ---------------------------------------------------------------------------
# Metrics (only when GT available)
# ---------------------------------------------------------------------------
def compute_image_metrics(pred_np: np.ndarray, gt_np: np.ndarray):
    """Returns dict of PSNR, SSIM, (LPIPS if available)."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    pred = np.nan_to_num(np.clip(pred_np.squeeze(), 0.0, 1.0), nan=0.0)
    gt   = gt_np.squeeze().astype(np.float32)
    # Normalize GT the same way
    mu   = gt.mean(); sig = gt.std() + 1e-8
    gt   = np.clip(gt, mu - 3*sig, mu + 3*sig)
    lo, hi = gt.min(), gt.max()
    if hi > lo:
        gt = (gt - lo) / (hi - lo)
    else:
        gt = np.zeros_like(gt)

    try:
        psnr_val = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
        psnr = psnr_val if not np.isnan(psnr_val) and not np.isinf(psnr_val) else None
    except Exception:
        psnr = None

    try:
        ssim_val = float(structural_similarity(gt, pred, data_range=1.0))
        ssim = ssim_val if not np.isnan(ssim_val) and not np.isinf(ssim_val) else None
    except Exception:
        ssim = None

    result = {'psnr': psnr, 'ssim': ssim}
    try:
        import lpips as lpips_lib
        import torch as _torch
        if not hasattr(compute_image_metrics, '_lpips_net'):
            compute_image_metrics._lpips_net = lpips_lib.LPIPS(net='alex', verbose=False)
        net = compute_image_metrics._lpips_net
        p_t = _torch.from_numpy(pred).unsqueeze(0).repeat(3,1,1).unsqueeze(0)*2-1
        g_t = _torch.from_numpy(gt).unsqueeze(0).repeat(3,1,1).unsqueeze(0)*2-1
        with _torch.no_grad():
            lp = float(net(p_t, g_t).item())
            result['lpips'] = lp if not np.isnan(lp) and not np.isinf(lp) else None
    except Exception:
        pass
    return result


def predict_tta(model, inp_t):
    """8-pass Test-Time Augmentation (rotations + flips) for maximum test quality."""
    preds = []
    for rot in range(4):
        x = torch.rot90(inp_t, rot, [2, 3])
        out = model(x)
        out = torch.rot90(out, -rot, [2, 3])
        preds.append(out)

        x_flip = torch.flip(x, [3])
        out_flip = model(x_flip)
        out_flip = torch.flip(out_flip, [3])
        out_flip = torch.rot90(out_flip, -rot, [2, 3])
        preds.append(out_flip)

    return torch.stack(preds, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    save_png = args.save_png and not args.no_png

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # AMP — only for CUDA
    amp_ok = device.type == 'cuda'

    # Weights path
    weights_path = args.weights or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'best_model.pt'
    )
    print(f"[Weights] Loading from: {weights_path}")
    model = load_model(weights_path, device, args)
    print("[Model] Loaded OK — running inference...")

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect input files
    input_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith('.npy')
    ])
    if not input_files:
        print(f"[ERROR] No .npy files found in: {args.input_dir}")
        sys.exit(1)
    if args.max_samples:
        input_files = input_files[:args.max_samples]
    print(f"[Input] {len(input_files)} .npy files found")

    # Metrics accumulator
    all_metrics = []
    times       = []

    with torch.no_grad():
        for i, fname in enumerate(input_files):
            noisy_path = os.path.join(args.input_dir, fname)

            # Load & normalize
            raw    = np.load(noisy_path)
            normed = normalize_image(raw, clip_sigma=args.clip_sigma)
            inp_t  = torch.from_numpy(normed).unsqueeze(0).unsqueeze(0).float().to(device)

            # Inference + time measurement
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with autocast(device_type='cuda', enabled=amp_ok):
                pred = predict_tta(model, inp_t)
                pred = torch.clamp(pred, 0.0, 1.0)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            # Convert to numpy
            pred_np = pred[0, 0].cpu().float().numpy()

            # Save outputs
            base_name = os.path.splitext(fname)[0]
            npy_out   = os.path.join(args.output_dir, fname)                  # same filename
            np.save(npy_out, pred_np.astype(np.float32))

            if save_png:
                png_out = os.path.join(args.output_dir, base_name + '.png')
                from PIL import Image
                uint8 = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(uint8).save(png_out)

            # Metrics (if GT available)
            if args.gt_dir:
                gt_path = os.path.join(args.gt_dir, fname)
                if os.path.exists(gt_path):
                    gt_raw = np.load(gt_path)
                    m = compute_image_metrics(pred_np, gt_raw)
                    m['filename'] = fname
                    all_metrics.append(m)

            # Progress
            if (i + 1) % 50 == 0 or (i + 1) == len(input_files):
                print(f"  [{i+1:4d}/{len(input_files)}]  {fname}  "
                      f"time={elapsed*1000:.1f}ms")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  INFERENCE COMPLETE")
    print(f"{'='*55}")
    print(f"  Images processed : {len(input_files)}")
    print(f"  Output directory : {args.output_dir}")
    avg_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    print(f"  Avg inference time : {avg_ms:.1f} ms/image")
    print(f"  P95 inference time : {p95_ms:.1f} ms/image")

    if all_metrics:
        psnrs  = [m['psnr']  for m in all_metrics]
        ssims  = [m['ssim']  for m in all_metrics]
        lpips_ = [m['lpips'] for m in all_metrics if 'lpips' in m]

        print(f"\n  Metrics on {len(all_metrics)} samples with GT:")
        print(f"  PSNR  : {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
        print(f"  SSIM  : {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
        if lpips_:
            print(f"  LPIPS : {np.mean(lpips_):.4f} ± {np.std(lpips_):.4f}")

        # Save metrics JSON
        summary = {
            'n_images'         : len(all_metrics),
            'psnr_mean'        : float(np.mean(psnrs)),
            'psnr_std'         : float(np.std(psnrs)),
            'ssim_mean'        : float(np.mean(ssims)),
            'ssim_std'         : float(np.std(ssims)),
            'avg_time_ms'      : float(avg_ms),
            'per_image'        : all_metrics,
        }
        if lpips_:
            summary['lpips_mean'] = float(np.mean(lpips_))
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Metrics saved → {metrics_path}")

    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
