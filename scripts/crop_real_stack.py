#!/usr/bin/env python3
"""
Crop a 3D real pFIB stack into 300x300 connected slices, without GUI selection.

Workflow:
  1) First run without --cx/--cy:
       - script saves a reference slice PNG
       - you open it in any viewer, decide a center (cx, cy) in pixel coords
  2) Run again with --cx and --cy:
       - script crops all slices to a fixed crop_size x crop_size window
"""

import os
import argparse
import numpy as np
import tifffile
import imageio.v2 as imageio


def load_stack_3d(path):
    """Load a 3D stack (tif or npy), return array [Z, Y, X]."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".npy"):
        arr = np.load(path)
    else:
        arr = tifffile.imread(path)

    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D stack (Z,Y,X), got shape {arr.shape}")
    return arr


def normalize_to_uint8(img2d: np.ndarray) -> np.ndarray:
    """Normalize 2D image to [0,255] uint8 for saving."""
    if img2d.dtype == np.uint8:
        return img2d
    arr = img2d.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255).astype(np.uint8)


def compute_crop_box_from_center(cy, cx, H, W, crop_size):
    """
    Given center (cy, cx) in pixels and image size H,W,
    return a valid crop box [y0:y1, x0:x1] of size crop_size x crop_size.
    """
    half = crop_size // 2
    y0 = int(round(cy)) - half
    x0 = int(round(cx)) - half

    y0 = max(0, min(y0, H - crop_size))
    x0 = max(0, min(x0, W - crop_size))

    y1 = y0 + crop_size
    x1 = x0 + crop_size
    return y0, y1, x0, x1


def main():
    parser = argparse.ArgumentParser(
        description="Crop a 3D real stack into 300x300 connected slices (no GUI)."
    )
    parser.add_argument(
        "--input-tif",
        type=str,
        required=True,
        help="Path to lightened real stack (.tif or .npy), shape [Z,Y,X].",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="cropped_real_300",
        help="Output directory for cropped slices.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=300,
        help="Crop size (default 300 -> 300x300).",
    )
    parser.add_argument(
        "--ref-slice",
        type=str,
        default="middle",
        help="Reference slice index or 'middle' (used for preview).",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Center x (column) of crop window in pixels.",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Center y (row) of crop window in pixels.",
    )
    parser.add_argument(
        "--save-stack-tif",
        action="store_true",
        help="Also save the cropped volume as a multi-page TIF stack.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load stack ----
    print(f"[INFO] Loading stack from {args.input_tif} ...")
    stack = load_stack_3d(args.input_tif)
    Z, H, W = stack.shape
    print(f"[INFO] Stack shape: Z={Z}, H={H}, W={W}")

    crop = args.crop_size
    if H < crop or W < crop:
        raise ValueError(f"Image size ({H},{W}) is smaller than crop size {crop}x{crop}.")

    # ---- Determine reference slice for preview ----
    if args.ref_slice == "middle":
        z_ref = Z // 2
    else:
        z_ref = int(args.ref_slice)
        if z_ref < 0 or z_ref >= Z:
            raise ValueError(f"ref-slice index {z_ref} out of range [0, {Z-1}]")

    ref_img = stack[z_ref]
    ref_png_path = os.path.join(args.out_dir, f"ref_slice_z{z_ref:04d}.png")
    imageio.imwrite(ref_png_path, normalize_to_uint8(ref_img))
    print(f"[INFO] Saved reference slice for inspection: {ref_png_path}")

    # ---- Decide center (cx, cy) ----
    if args.cx is None or args.cy is None:
        # If user didn't give coordinates, suggest center of the image and exit after preview
        cx_suggest = W / 2.0
        cy_suggest = H / 2.0
        print()
        print("===================================================")
        print("[INFO] No --cx/--cy provided.")
        print("1) Open the reference slice PNG in any viewer,")
        print("   check pixel coordinates of the region you want.")
        print("2) Then rerun this script with, e.g.:")
        print(f"   --cx {cx_suggest:.1f} --cy {cy_suggest:.1f}")
        print("   (or other center based on what you see).")
        print("===================================================")
        return

    cx = args.cx
    cy = args.cy
    print(f"[INFO] Using center (cx={cx}, cy={cy}) for crop.")

    # Compute crop window once, reuse for all slices
    y0, y1, x0, x1 = compute_crop_box_from_center(cy, cx, H, W, crop)
    print(f"[INFO] Final crop box: y=[{y0}:{y1}), x=[{x0}:{x1})")

    # ---- Apply crop to all slices ----
    cropped_slices = []
    saved_count = 0

    for z in range(Z):
        patch = stack[z, y0:y1, x0:x1]
        if patch.shape != (crop, crop):
            print(f"[WARN] slice {z} cropped shape {patch.shape} != ({crop},{crop}), skipping.")
            continue

        patch_u8 = normalize_to_uint8(patch)
        fname = f"real_z{z:04d}_y{y0:04d}_x{x0:04d}.png"
        out_path = os.path.join(args.out_dir, fname)
        imageio.imwrite(out_path, patch_u8)
        cropped_slices.append(patch_u8)
        saved_count += 1

    print(f"[DONE] Saved {saved_count} cropped {crop}x{crop} slices to {args.out_dir}")

    # Optional: save as stacked TIF
    if args.save_stack_tif and cropped_slices:
        stack_out = np.stack(cropped_slices, axis=0)
        stack_tif_path = os.path.join(args.out_dir, f"cropped_stack_{crop}x{crop}.tif")
        tifffile.imwrite(stack_tif_path, stack_out)
        print(f"[DONE] Also saved stacked TIF: {stack_tif_path}")


if __name__ == "__main__":
    main()
