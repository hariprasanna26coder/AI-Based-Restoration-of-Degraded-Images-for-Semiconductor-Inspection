"""
model/losses.py
---------------
Multi-component restoration loss for SemiRestoreNet:

    L_total = 1.0 × L_Charbonnier  +  0.2 × L_SSIM  +  0.1 × L_Edge

Components:
  - Charbonnier Loss : Smooth L1 variant; robust to speckle outliers
  - SSIM Loss        : Penalizes structural dissimilarity directly
  - Edge Loss        : Sobel-based; preserves critical semiconductor edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Charbonnier Loss  (smooth L1 / pseudo-Huber)
# ---------------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    """
    L_char(pred, gt) = mean( sqrt( (pred - gt)^2 + eps^2 ) )

    Properties vs MSE:
      - Less sensitive to outlier pixels (e.g., extreme speckle values)
      - Does NOT produce blurring like MSE
      - Differentiable everywhere (unlike plain L1)
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = pred - gt
        loss = torch.sqrt(diff * diff + self.eps ** 2)
        return loss.mean()


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------
class SSIMLoss(nn.Module):
    """
    Structural Similarity loss:  L_ssim = 1 - SSIM(pred, gt)

    Uses a Gaussian window for local statistics.
    Works on single-channel (grayscale) images.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._create_window(window_size, sigma))

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g

    def _create_window(self, size: int, sigma: float) -> torch.Tensor:
        k1d = self._gaussian_kernel(size, sigma)
        k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)           # outer product
        return k2d.unsqueeze(0).unsqueeze(0)                 # (1, 1, size, size)

    def _ssim_map(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2
        win = self.window.expand(img1.size(1), 1, -1, -1).to(device=img1.device, dtype=img1.dtype)

        mu1    = F.conv2d(img1, win, padding=pad, groups=img1.size(1))
        mu2    = F.conv2d(img2, win, padding=pad, groups=img2.size(1))
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        s1   = torch.clamp(F.conv2d(img1 * img1, win, padding=pad, groups=img1.size(1)) - mu1_sq, min=0.0)
        s2   = torch.clamp(F.conv2d(img2 * img2, win, padding=pad, groups=img2.size(1)) - mu2_sq, min=0.0)
        s12  = F.conv2d(img1 * img2, win, padding=pad, groups=img1.size(1)) - mu1_mu2

        num  = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
        den  = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
        return num / den

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        ssim_map = self._ssim_map(pred, gt)
        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Edge Loss  (Sobel-based)
# ---------------------------------------------------------------------------
class EdgeLoss(nn.Module):
    """
    L_edge(pred, gt) = L1( Sobel(pred), Sobel(gt) )

    Critical for semiconductor images: ensures edges are sharp, not blurry.
    Uses fixed Sobel kernels (no learnable params).
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel filters."""
        B, C, H, W = x.shape
        # Reshape to apply per channel
        x_flat = x.view(B * C, 1, H, W)
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x_flat, sx, padding=1)
        gy = F.conv2d(x_flat, sy, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return mag.view(B, C, H, W)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edge_map(pred), self._edge_map(gt))


# ---------------------------------------------------------------------------
# Combined Restoration Loss
# ---------------------------------------------------------------------------
class RestorationLoss(nn.Module):
    """
    Combined loss for joint denoising + super-resolution:

        L = w_char × L_Charbonnier  +  w_ssim × L_SSIM  +  w_edge × L_Edge

    Default weights are tuned for semiconductor grayscale images.
    """
    def __init__(
        self,
        w_char: float = 1.0,
        w_ssim: float = 0.2,
        w_edge: float = 0.1,
        char_eps: float = 1e-3,
    ):
        super().__init__()
        self.w_char = w_char
        self.w_ssim = w_ssim
        self.w_edge = w_edge

        self.charbonnier = CharbonnierLoss(eps=char_eps)
        self.ssim        = SSIMLoss(window_size=11, sigma=1.5)
        self.edge        = EdgeLoss()

    def forward(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
    ) -> tuple:
        """
        Args:
            pred : (B, 1, H, W) model output, clamped to [0,1]
            gt   : (B, 1, H, W) ground truth, normalised [0,1]
        Returns:
            total_loss, (char_loss, ssim_loss, edge_loss)
        """
        pred = pred.float()
        gt   = gt.float()

        l_char = self.charbonnier(pred, gt)
        l_ssim = self.ssim(pred, gt)
        l_edge = self.edge(pred, gt)

        total = (
            self.w_char * l_char
            + self.w_ssim * l_ssim
            + self.w_edge * l_edge
        )
        return total, (l_char, l_ssim, l_edge)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pred = torch.rand(2, 1, 256, 256)
    gt   = torch.rand(2, 1, 256, 256)

    criterion = RestorationLoss()
    total, (lc, ls, le) = criterion(pred, gt)
    print(f"Total loss   : {total.item():.4f}")
    print(f"  Charbonnier: {lc.item():.4f}")
    print(f"  SSIM       : {ls.item():.4f}")
    print(f"  Edge       : {le.item():.4f}")
