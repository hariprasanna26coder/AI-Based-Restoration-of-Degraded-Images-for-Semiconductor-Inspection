"""
data/transforms.py
------------------
Augmentation transforms for semiconductor image restoration.

Key design decisions:
  - All spatial transforms (flip, rotate) applied IDENTICALLY to both
    NoisyLR and GT arrays to maintain pixel-to-pixel correspondence.
  - Intensity jitter applied ONLY to NoisyLR to simulate varying
    noise/exposure levels and improve OOD generalization.
  - RobustNormalize clips speckle outlier values using per-image
    statistics (μ ± k·σ), then scales to [0,1].
"""

import numpy as np
import random
from typing import Tuple


# ---------------------------------------------------------------------------
# Robust Normalization  (handles speckle value overflow)
# ---------------------------------------------------------------------------
class RobustNormalize:
    """
    Per-image normalization that handles speckle-noise outliers.

    Steps:
      1. Compute per-image mean (μ) and std (σ)
      2. Clip values to [μ - k·σ, μ + k·σ]  (removes extreme speckle spikes)
      3. Min-max scale to [0, 1]

    This is critical: standard /255 or /max normalization would fail because
    speckle noise can push pixel values far outside the true image range.
    """
    def __init__(self, clip_sigma: float = 3.0):
        self.clip_sigma = clip_sigma

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        mu  = arr.mean()
        sig = arr.std() + 1e-8
        lo  = mu - self.clip_sigma * sig
        hi  = mu + self.clip_sigma * sig
        arr = np.clip(arr, lo, hi)
        # Min-max to [0, 1]
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        return arr


# ---------------------------------------------------------------------------
# Geometry-consistent Augmentation  (applied to both NoisyLR and GT)
# ---------------------------------------------------------------------------
class RandomFlipAndRotate:
    """
    Random horizontal flip, vertical flip, and 90° rotation.
    Applied identically to both images in a pair.
    """
    def __init__(self, hflip_p: float = 0.5, vflip_p: float = 0.5, rot90_p: float = 0.25):
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rot90_p = rot90_p

    def __call__(
        self,
        noisy: np.ndarray,
        gt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontal flip
        if random.random() < self.hflip_p:
            noisy = np.flip(noisy, axis=-1)
            gt    = np.flip(gt,    axis=-1)

        # Vertical flip
        if random.random() < self.vflip_p:
            noisy = np.flip(noisy, axis=-2)
            gt    = np.flip(gt,    axis=-2)

        # 90° rotation (random number of times: 0-3)
        if random.random() < self.rot90_p:
            k = random.choice([1, 2, 3])
            noisy = np.rot90(noisy, k=k, axes=(-2, -1))
            gt    = np.rot90(gt,    k=k, axes=(-2, -1))

        # Make contiguous copies after flips/rots
        noisy = np.ascontiguousarray(noisy)
        gt    = np.ascontiguousarray(gt)
        return noisy, gt


# ---------------------------------------------------------------------------
# Intensity Jitter  (NoisyLR only — improves noise-level generalization)
# ---------------------------------------------------------------------------
class IntensityJitter:
    """
    Apply random gamma correction to NoisyLR ONLY.
    Models varying noise/exposure conditions for OOD robustness.
    GT remains unchanged.

    gamma ∈ [lo, hi], default [0.8, 1.2]
    """
    def __init__(self, gamma_lo: float = 0.8, gamma_hi: float = 1.2):
        self.gamma_lo = gamma_lo
        self.gamma_hi = gamma_hi

    def __call__(self, noisy: np.ndarray) -> np.ndarray:
        gamma = random.uniform(self.gamma_lo, self.gamma_hi)
        # Clip to valid range before power, then re-clip result
        noisy = np.clip(noisy, 0.0, 1.0)
        noisy = np.power(noisy, gamma)
        return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# Random Patch Crop  (applied identically at correct scale)
# ---------------------------------------------------------------------------
class RandomPairCrop:
    """
    Randomly crop a patch from NoisyLR and the corresponding HR patch from GT.

    Since GT is 2× the size of NoisyLR:
      - Crop noisy_patch_size from NoisyLR
      - Crop (noisy_patch_size × 2) from GT at the scaled location

    Args:
        noisy_patch_size: Size of the patch taken from the LR input.
                          GT patch will be 2× this size.
    """
    def __init__(self, noisy_patch_size: int = 64):
        self.np_size = noisy_patch_size
        self.gt_size = noisy_patch_size * 2

    def __call__(
        self,
        noisy: np.ndarray,
        gt:    np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # noisy: (H, W) or (1, H, W)
        # gt   : (2H, 2W) or (1, 2H, 2W)
        if noisy.ndim == 2:
            nH, nW = noisy.shape
        else:
            nH, nW = noisy.shape[-2], noisy.shape[-1]

        # Random top-left corner in NoisyLR space
        max_y = nH - self.np_size
        max_x = nW - self.np_size
        if max_y <= 0 or max_x <= 0:
            # Image smaller than patch — just return as-is
            return noisy, gt

        y0_n = random.randint(0, max_y)
        x0_n = random.randint(0, max_x)

        # Corresponding GT top-left (scale up by 2)
        y0_g = y0_n * 2
        x0_g = x0_n * 2

        # Slice
        if noisy.ndim == 2:
            noisy_crop = noisy[y0_n: y0_n + self.np_size,
                               x0_n: x0_n + self.np_size]
            gt_crop    = gt   [y0_g: y0_g + self.gt_size,
                               x0_g: x0_g + self.gt_size]
        else:
            noisy_crop = noisy[..., y0_n: y0_n + self.np_size,
                                    x0_n: x0_n + self.np_size]
            gt_crop    = gt   [..., y0_g: y0_g + self.gt_size,
                                    x0_g: x0_g + self.gt_size]

        return noisy_crop, gt_crop


# ---------------------------------------------------------------------------
# Composed Training Transform
# ---------------------------------------------------------------------------
class TrainTransform:
    """
    Full training augmentation pipeline:
      1. RobustNormalize (noisy)
      2. RobustNormalize (gt)
      3. Random patch crop (pair)
      4. Random flip + rotate (pair)
      5. Intensity jitter (noisy only)
    """
    def __init__(
        self,
        noisy_patch_size: int  = 64,
        clip_sigma: float      = 3.0,
        hflip_p: float         = 0.5,
        vflip_p: float         = 0.5,
        rot90_p: float         = 0.25,
        gamma_lo: float        = 0.8,
        gamma_hi: float        = 1.2,
        use_crops: bool        = True,
    ):
        self.normalize = RobustNormalize(clip_sigma)
        self.crop      = RandomPairCrop(noisy_patch_size) if use_crops else None
        self.flip_rot  = RandomFlipAndRotate(hflip_p, vflip_p, rot90_p)
        self.jitter    = IntensityJitter(gamma_lo, gamma_hi)

    def __call__(
        self,
        noisy: np.ndarray,
        gt:    np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Normalize
        noisy = self.normalize(noisy)
        gt    = self.normalize(gt)

        # 2. Crop
        if self.crop is not None:
            noisy, gt = self.crop(noisy, gt)

        # 3. Spatial augmentation
        noisy, gt = self.flip_rot(noisy, gt)

        # 4. Intensity jitter (noisy only)
        noisy = self.jitter(noisy)

        return noisy, gt


# ---------------------------------------------------------------------------
# Validation / Test Transform  (no augmentation, just normalization)
# ---------------------------------------------------------------------------
class ValTransform:
    """Validation/test transform: robust normalization only."""
    def __init__(self, clip_sigma: float = 3.0):
        self.normalize = RobustNormalize(clip_sigma)

    def __call__(
        self,
        noisy: np.ndarray,
        gt:    np.ndarray = None,
    ):
        noisy = self.normalize(noisy)
        if gt is not None:
            gt = self.normalize(gt)
            return noisy, gt
        return noisy
