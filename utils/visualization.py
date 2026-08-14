"""
utils/visualization.py
-----------------------
Save before/after comparison grids for visual quality inspection.
Outputs a PNG grid: [NoisyLR (bicubic up) | Restored | Ground Truth]
"""

import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float [0,1] array to uint8 [0,255]."""
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) or (H, W) tensor to (H, W) float numpy."""
    arr = t.detach().cpu().float().numpy()
    return arr.squeeze()


def save_comparison_grid(
    noisy_batch:    torch.Tensor,
    pred_batch:     torch.Tensor,
    gt_batch:       torch.Tensor,
    save_path:      str,
    n_samples:      int = 6,
    titles:         List[str] = None,
    noisy_is_lr:    bool = True,  # True: bicubic-up noisy for display
):
    """
    Save a side-by-side grid:
        Column 1 : NoisyLR (bicubic upsampled to match GT size)
        Column 2 : Restored (model output)
        Column 3 : Ground Truth

    Args:
        noisy_batch : (B, 1, H/2, W/2) LR noisy input tensors
        pred_batch  : (B, 1, H,   W  ) model predictions
        gt_batch    : (B, 1, H,   W  ) ground truth targets
        save_path   : Output path for the PNG file
        n_samples   : Number of rows (samples) to include
        titles      : Optional column titles
        noisy_is_lr : If True, bicubic-upsample noisy to GT size for fair display
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    n = min(n_samples, noisy_batch.size(0))
    B_size   = pred_batch.shape[-1]   # output width
    H_size   = pred_batch.shape[-2]   # output height

    col_titles = titles or ["NoisyLR (bicubic)", "Restored (Ours)", "Ground Truth"]
    header_h   = 40  # pixels for column header
    pad        = 4   # pixels between cells
    n_cols     = 3

    canvas_w = n_cols * B_size + (n_cols + 1) * pad
    canvas_h = n * H_size + (n + 1) * pad + header_h

    canvas = Image.new('L', (canvas_w, canvas_h), color=240)
    draw   = ImageDraw.Draw(canvas)

    # Column headers
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for col_idx, title in enumerate(col_titles):
        x = pad + col_idx * (B_size + pad) + B_size // 2 - len(title) * 4
        draw.text((x, 10), title, fill=50, font=font)

    # Fill rows
    for row in range(n):
        # Convert tensors to numpy
        noisy_np = _tensor_to_np(noisy_batch[row])   # (H/2, W/2)
        pred_np  = _tensor_to_np(pred_batch[row])    # (H, W)
        gt_np    = _tensor_to_np(gt_batch[row])      # (H, W)

        # Upsample noisy to display size
        if noisy_is_lr:
            noisy_pil = Image.fromarray(_to_uint8(noisy_np)).resize(
                (B_size, H_size), resample=Image.BICUBIC
            )
        else:
            noisy_pil = Image.fromarray(_to_uint8(noisy_np))

        pred_pil = Image.fromarray(_to_uint8(pred_np))
        gt_pil   = Image.fromarray(_to_uint8(gt_np))

        y = header_h + pad + row * (H_size + pad)

        for col_idx, img in enumerate([noisy_pil, pred_pil, gt_pil]):
            x = pad + col_idx * (B_size + pad)
            canvas.paste(img, (x, y))

    canvas.save(save_path)
    print(f"[Visualization] Saved comparison grid → {save_path}")


def save_restored_image(
    pred_tensor: torch.Tensor,
    save_path: str,
    as_npy: bool = True,
    as_png: bool = True,
):
    """
    Save a single restored image tensor as .npy and/or .png.

    Args:
        pred_tensor : (1, 1, H, W) or (1, H, W) tensor
        save_path   : Path without extension (extensions added automatically)
        as_npy      : Save float32 .npy (required for KLA submission)
        as_png      : Also save visualizable .png
    """
    arr = _tensor_to_np(pred_tensor)   # (H, W) float

    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)

    if as_npy:
        npy_path = base + '.npy'
        np.save(npy_path, arr.astype(np.float32))

    if as_png:
        png_path = base + '.png'
        Image.fromarray(_to_uint8(arr)).save(png_path)
