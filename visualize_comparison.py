"""
visualize_comparison.py
-----------------------
Generates side-by-side visual comparison images:
[ Input (Noisy LR)  |  Output (Restored HR)  |  (Optional GT) ]

Usage:
  python visualize_comparison.py --input_dir Test_NoisyLR/NoisyLR --output_dir outputs/test_restored --save_dir outputs/comparisons

Optional:
  --gt_dir path/to/GT       Include ground truth column if available
  --num_samples 20          Limit number of images to generate (default: all)
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def normalize_for_display(arr: np.ndarray, clip_sigma: float = 3.0) -> np.ndarray:
    """Normalize float array to uint8 [0, 255] for visual display."""
    arr = arr.astype(np.float32).squeeze()
    mu = arr.mean()
    sig = arr.std() + 1e-8
    arr = np.clip(arr, mu - clip_sigma * sig, mu + clip_sigma * sig)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255.0).astype(np.uint8)


def create_comparison_image(noisy_arr: np.ndarray, pred_arr: np.ndarray, gt_arr: np.ndarray = None) -> Image.Image:
    """Stitches NoisyLR, Restored (and GT if present) side-by-side with labels."""
    h, w = pred_arr.shape[-2:]
    header_h = 35
    pad = 4

    # Convert to uint8
    noisy_uint8 = normalize_for_display(noisy_arr)
    pred_uint8 = (np.clip(pred_arr.squeeze(), 0.0, 1.0) * 255.0).astype(np.uint8)

    # Resize noisy to match target output resolution
    noisy_pil = Image.fromarray(noisy_uint8).resize((w, h), resample=Image.BICUBIC)
    pred_pil = Image.fromarray(pred_uint8)

    columns = [("Input (Noisy LR)", noisy_pil), ("Output (Restored)", pred_pil)]

    if gt_arr is not None:
        gt_uint8 = normalize_for_display(gt_arr)
        gt_pil = Image.fromarray(gt_uint8).resize((w, h), resample=Image.BICUBIC)
        columns.append(("Ground Truth", gt_pil))

    n_cols = len(columns)
    canvas_w = n_cols * w + (n_cols + 1) * pad
    canvas_h = h + pad * 2 + header_h

    # Create dark canvas for contrast
    canvas = Image.new('L', (canvas_w, canvas_h), color=30)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for idx, (title, img) in enumerate(columns):
        x = pad + idx * (w + pad)
        y = header_h + pad
        canvas.paste(img, (x, y))

        # Title label centered above column
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (w - tw) // 2
        draw.text((tx, 8), title, fill=240, font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side comparison images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input NoisyLR directory or file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output Restored directory")
    parser.add_argument("--gt_dir", type=str, default=None, help="Optional path to Ground Truth directory")
    parser.add_argument("--save_dir", type=str, default="outputs/comparisons", help="Directory to save comparison PNGs")
    parser.add_argument("--num_samples", type=int, default=None, help="Max number of comparison images to generate")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Handle if single file vs directory
    if os.path.isfile(args.input_dir):
        input_files = [os.path.basename(args.input_dir)]
        input_base_dir = os.path.dirname(args.input_dir)
    else:
        input_base_dir = args.input_dir
        input_files = sorted([f for f in os.listdir(input_base_dir) if f.endswith('.npy')])

    if args.num_samples:
        input_files = input_files[:args.num_samples]

    print(f"[Comparison Generator] Processing {len(input_files)} images...")
    count = 0

    for fname in input_files:
        noisy_path = os.path.join(input_base_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        if not os.path.exists(out_path):
            # Check if png version exists
            png_alt = os.path.join(args.output_dir, os.path.splitext(fname)[0] + ".png")
            if not os.path.exists(png_alt):
                print(f"Skipping {fname}: Restored output file not found in {args.output_dir}")
                continue

        noisy_arr = np.load(noisy_path)

        if os.path.exists(out_path) and out_path.endswith('.npy'):
            pred_arr = np.load(out_path)
        else:
            png_alt = os.path.join(args.output_dir, os.path.splitext(fname)[0] + ".png")
            pred_arr = np.array(Image.open(png_alt)).astype(np.float32) / 255.0

        gt_arr = None
        if args.gt_dir:
            gt_path = os.path.join(args.gt_dir, fname)
            if os.path.exists(gt_path):
                gt_arr = np.load(gt_path)

        comp_img = create_comparison_image(noisy_arr, pred_arr, gt_arr)
        out_png_path = os.path.join(args.save_dir, os.path.splitext(fname)[0] + "_comparison.png")
        comp_img.save(out_png_path)
        count += 1

    print(f"[Success] Generated {count} comparison images in: {args.save_dir}")


if __name__ == "__main__":
    main()
