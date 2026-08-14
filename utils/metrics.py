"""
utils/metrics.py
----------------
Evaluation metrics for image restoration:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity) — optional
"""

import numpy as np
import torch
from skimage.metrics import (
    peak_signal_noise_ratio as skimage_psnr,
    structural_similarity   as skimage_ssim,
)
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------
def compute_psnr(
    pred: np.ndarray,
    gt:   np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Compute PSNR between two numpy arrays (values in [0, data_range]).
    Both arrays should be float32 with shape (H, W) or (1, H, W) or (H, W, 1).
    """
    pred = np.nan_to_num(pred.squeeze(), nan=0.0, posinf=1.0, neginf=0.0)
    gt   = np.nan_to_num(gt.squeeze(),   nan=0.0, posinf=1.0, neginf=0.0)
    try:
        val = skimage_psnr(gt, pred, data_range=data_range)
        return float(val) if not np.isnan(val) and not np.isinf(val) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def compute_ssim(
    pred: np.ndarray,
    gt:   np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Compute SSIM between two numpy arrays.
    Both should be float32 with shape (H, W).
    """
    pred = np.nan_to_num(pred.squeeze(), nan=0.0, posinf=1.0, neginf=0.0)
    gt   = np.nan_to_num(gt.squeeze(),   nan=0.0, posinf=1.0, neginf=0.0)
    try:
        val = skimage_ssim(gt, pred, data_range=data_range)
        return float(val) if not np.isnan(val) and not np.isinf(val) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# LPIPS  (optional — requires lpips package)
# ---------------------------------------------------------------------------
_lpips_net = None  # lazy-load

def _get_lpips():
    global _lpips_net
    if _lpips_net is None:
        try:
            import lpips
            _lpips_net = lpips.LPIPS(net='alex', verbose=False)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _lpips_net = _lpips_net.to(device)
        except ImportError:
            _lpips_net = None
    return _lpips_net


def compute_lpips(
    pred: np.ndarray,
    gt:   np.ndarray,
) -> Optional[float]:
    """
    Compute LPIPS (lower = better).
    Returns None if lpips package is not installed.
    """
    net = _get_lpips()
    if net is None:
        return None

    device = next(net.parameters()).device

    # Convert grayscale to 3-channel (LPIPS expects RGB)
    pred_t = torch.from_numpy(pred.squeeze()).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
    gt_t   = torch.from_numpy(gt.squeeze()).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()

    # LPIPS expects values in [-1, 1]
    pred_t = pred_t * 2.0 - 1.0
    gt_t   = gt_t   * 2.0 - 1.0

    pred_t = pred_t.to(device)
    gt_t   = gt_t.to(device)

    with torch.no_grad():
        score = net(pred_t, gt_t).item()
    return float(score)


# ---------------------------------------------------------------------------
# Aggregate metrics over a batch of tensors
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics(
    pred_batch: torch.Tensor,
    gt_batch:   torch.Tensor,
    compute_lpips_flag: bool = False,
) -> Dict[str, float]:
    """
    Compute PSNR, SSIM (and optionally LPIPS) for a batch of predictions.

    Args:
        pred_batch: (B, 1, H, W) float tensor, values in [0, 1]
        gt_batch  : (B, 1, H, W) float tensor, values in [0, 1]

    Returns:
        dict with keys 'psnr', 'ssim', and optionally 'lpips'
    """
    pred_np = pred_batch.cpu().float().numpy()
    gt_np   = gt_batch.cpu().float().numpy()

    psnrs, ssims, lpips_scores = [], [], []

    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i, 0], 0.0, 1.0)
        g = gt_np[i, 0]

        psnrs.append(compute_psnr(p, g))
        ssims.append(compute_ssim(p, g))
        if compute_lpips_flag:
            lp = compute_lpips(p, g)
            if lp is not None:
                lpips_scores.append(lp)

    results = {
        'psnr': float(np.nanmean(psnrs)),
        'ssim': float(np.nanmean(ssims)),
    }
    if lpips_scores:
        results['lpips'] = float(np.nanmean(lpips_scores))

    return results


# ---------------------------------------------------------------------------
# Running averages for training loop
# ---------------------------------------------------------------------------
class AverageMeter:
    """Tracks running mean and count of a scalar metric."""
    def __init__(self, name: str = ""):
        self.name  = name
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        if not np.isnan(val) and not np.isinf(val):
            self.sum   += val * n
            self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"
