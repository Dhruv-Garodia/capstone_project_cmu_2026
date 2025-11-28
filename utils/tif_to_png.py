#!/usr/bin/env python3
"""
Convert a 3D TIFF stack OR a .npy volume into individual PNG slices.

Usage:
  python stack_to_png_slices.py --input volume.tif --out-dir slices_png
  python stack_to_png_slices.py --input volume.npy --out-dir slices_png
"""

import os
import argparse
import numpy as np
import tifffile
import imageio.v2 as imageio


def load_stack(input_path):
    """Load TIFF or NPY into a 3D numpy array [Z, H, W]."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        arr = tifffile.imread(input_path)
    elif ext == ".npy":
        arr = np.load(input_path)
    else:
        raise ValueError(f"Unsupported input type: {ext}. Use .tif/.tiff or .npy")

    # Ensure 3D
    if arr.ndim == 2:
        arr = arr[None, ...]        # (H, W) → (1, H, W)
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    return arr


def normalize_to_uint8(img):
    """Normalize ndarray (any dtype) to uint8 [0–255]."""
    if img.dtype == np.uint8:
        return img

    arr = img.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()

    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)

    return (arr * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Convert 3D volume (.tif or .npy) into PNG slices.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .tif/.tiff/.npy file")
    parser.add_argument("--out-dir", type=str, default="png_slices", help="Folder to save output PNG slices")
    parser.add_argument("--prefix", type=str, default="slice", help="Output filename prefix")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading: {args.input}")
    stack = load_stack(args.input)

    Z, H, W = stack.shape
    print(f"[INFO] Loaded volume shape: Z={Z}, H={H}, W={W}")

    saved = 0
    for z in range(Z):
        img = normalize_to_uint8(stack[z])
        fname = f"{args.prefix}_{z:04d}.png"
        out_path = os.path.join(args.out_dir, fname)
        imageio.imwrite(out_path, img)
        saved += 1

    print(f"[DONE] Saved {saved} PNG slices to: {args.out_dir}")


if __name__ == "__main__":
    main()
