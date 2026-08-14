"""
data/dataset.py
---------------
PyTorch Dataset classes for the KLA semiconductor image restoration task.

NpyPairDataset  — Paired NoisyLR + GT for training/validation.
NpyTestDataset  — NoisyLR-only for test inference.

Key design features:
  - Robust normalization that handles speckle value overflow
  - Optional patch cropping for data augmentation (4× more virtual samples)
  - Train/val split via index lists
  - Float32 tensors, single-channel (1, H, W)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from typing import Optional, List

from .transforms import TrainTransform, ValTransform


# ---------------------------------------------------------------------------
# Paired Dataset  (NoisyLR + GT)
# ---------------------------------------------------------------------------
class NpyPairDataset(Dataset):
    """
    Paired dataset for training and validation.

    Args:
        noisy_dir  : Directory containing NoisyLR .npy files (128×128)
        gt_dir     : Directory containing GT .npy files (256×256)
        transform  : TrainTransform or ValTransform instance
        file_list  : Optional list of filenames (subset for train/val split)
    """
    def __init__(
        self,
        noisy_dir: str,
        gt_dir: str,
        transform=None,
        file_list: Optional[List[str]] = None,
    ):
        self.noisy_dir = noisy_dir
        self.gt_dir    = gt_dir
        self.transform = transform

        # Discover paired files
        if file_list is not None:
            self.file_names = sorted(file_list)
        else:
            noisy_set = set(os.listdir(noisy_dir))
            gt_set    = set(os.listdir(gt_dir))
            common    = noisy_set.intersection(gt_set)
            self.file_names = sorted([f for f in common if f.endswith('.npy')])

        print(f"[NpyPairDataset] {len(self.file_names)} paired files loaded.")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict:
        fname = self.file_names[idx]

        # Load raw arrays
        noisy = np.load(os.path.join(self.noisy_dir, fname))
        gt    = np.load(os.path.join(self.gt_dir,    fname))

        # Ensure 2D (squeeze single trailing channel if needed)
        noisy = noisy.squeeze()
        gt    = gt.squeeze()

        # Apply transform (normalize + optional augment)
        if self.transform is not None:
            noisy, gt = self.transform(noisy, gt)
        else:
            # Minimal normalization fallback
            from .transforms import RobustNormalize
            norm  = RobustNormalize()
            noisy = norm(noisy)
            gt    = norm(gt)

        # Add channel dimension: (H, W) → (1, H, W)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()   # (1, 128, 128) or (1, 64, 64)
        gt_t    = torch.from_numpy(gt).unsqueeze(0).float()       # (1, 256, 256) or (1, 128, 128)

        return {
            'noisy'   : noisy_t,
            'gt'      : gt_t,
            'filename': fname,
        }


# ---------------------------------------------------------------------------
# Test Dataset  (NoisyLR only — no GT)
# ---------------------------------------------------------------------------
class NpyTestDataset(Dataset):
    """
    Inference-only dataset. Loads NoisyLR files and applies normalization.
    No GT is available for test set.
    """
    def __init__(self, noisy_dir: str, clip_sigma: float = 3.0):
        self.noisy_dir = noisy_dir
        self.file_names = sorted([
            f for f in os.listdir(noisy_dir) if f.endswith('.npy')
        ])
        self.transform = ValTransform(clip_sigma=clip_sigma)
        print(f"[NpyTestDataset] {len(self.file_names)} test files loaded.")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict:
        fname = self.file_names[idx]
        noisy = np.load(os.path.join(self.noisy_dir, fname)).squeeze()

        # Normalize
        noisy = self.transform(noisy)

        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        return {
            'noisy'   : noisy_t,
            'filename': fname,
        }


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    noisy_dir: str,
    gt_dir: str,
    val_split: float   = 0.1,
    batch_size: int    = 8,
    num_workers: int   = 0,
    noisy_patch_size: int = 64,
    clip_sigma: float  = 3.0,
    seed: int          = 42,
):
    """
    Create train and validation DataLoaders with stratified split.

    Args:
        noisy_dir       : Path to NoisyLR directory
        gt_dir          : Path to GT directory
        val_split       : Fraction reserved for validation (default 0.10)
        batch_size      : Batch size for training
        num_workers     : DataLoader worker processes (0 = main process)
        noisy_patch_size: Random crop size from NoisyLR (GT crop = 2×)
        clip_sigma      : σ multiplier for robust normalization
        seed            : Random seed for reproducible split

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Discover all file names
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.npy')])
    gt_files    = sorted([f for f in os.listdir(gt_dir)    if f.endswith('.npy')])
    all_files   = sorted(list(set(noisy_files) & set(gt_files)))
    n = len(all_files)

    # Deterministic split
    rng      = np.random.RandomState(seed)
    shuffled = rng.permutation(n)
    n_val    = max(1, int(n * val_split))
    val_idx  = shuffled[:n_val].tolist()
    trn_idx  = shuffled[n_val:].tolist()

    train_files = [all_files[i] for i in trn_idx]
    val_files   = [all_files[i] for i in val_idx]

    print(f"[DataLoader] Split: {len(train_files)} train / {len(val_files)} val")

    train_transform = TrainTransform(
        noisy_patch_size=noisy_patch_size,
        clip_sigma=clip_sigma,
        use_crops=True,
    )
    val_transform = ValTransform(clip_sigma=clip_sigma)

    train_ds = NpyPairDataset(noisy_dir, gt_dir, transform=train_transform, file_list=train_files)
    val_ds   = NpyPairDataset(noisy_dir, gt_dir, transform=val_transform,   file_list=val_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,          # one at a time for reliable SSIM/PSNR per image
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds, val_ds


def get_test_loader(
    noisy_dir: str,
    batch_size: int = 1,
    num_workers: int = 0,
    clip_sigma: float = 3.0,
):
    """Create a DataLoader for the test set (no GT)."""
    test_ds = NpyTestDataset(noisy_dir=noisy_dir, clip_sigma=clip_sigma)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader, test_ds


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    noisy_dir = r"c:\Users\Lenovo\OneDrive\Desktop\1 final\train\train\NoisyLR"
    gt_dir    = r"c:\Users\Lenovo\OneDrive\Desktop\1 final\train\train\GT"

    tl, vl, _, _ = get_dataloaders(noisy_dir, gt_dir, val_split=0.1, batch_size=4)

    for batch in tl:
        n, g = batch['noisy'], batch['gt']
        print(f"Noisy : {n.shape}  range [{n.min():.3f}, {n.max():.3f}]")
        print(f"GT    : {g.shape}  range [{g.min():.3f}, {g.max():.3f}]")
        break
