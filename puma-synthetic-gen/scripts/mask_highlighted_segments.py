#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import numpy as np
import tifffile as tiff
from skimage.transform import resize
from skimage import io as skio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mask a segmented 3D TIFF volume using a highlighted 3D TIFF: keep voxels where highlighted > 0, else set to 0."
        )
    )
    parser.add_argument(
        "--highlighted_tif",
        required=True,
        help="Path to highlighted_stack.tif (nonzero values indicate regions to keep)",
    )
    parser.add_argument(
        "--segmented_tif",
        required=True,
        help="Path to segmented_stack.tif (labels/segmentation to be masked)",
    )
    parser.add_argument(
        "--out_tif",
        default=None,
        help="Output path for masked segmented TIFF (default: alongside input, name masked_segmented_stack.tif)",
    )
    parser.add_argument(
        "--out_png_dir",
        default=None,
        help="Optional directory to write per-slice PNGs (e.g., data/.../masked_png_slices)",
    )
    parser.add_argument(
        "--out_npy",
        default=None,
        help="Optional output .npy file for the masked volume (written via memmap, streaming)",
    )
    parser.add_argument(
        "--highlight_threshold",
        type=float,
        default=0.0,
        help="Threshold for considering a voxel highlighted (default: > 0)",
    )
    parser.add_argument(
        "--big_tiff",
        action="store_true",
        help="Force BigTIFF for output (recommended for large volumes)",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=[150, 150],
        metavar=("HEIGHT", "WIDTH"),
        help="Output size as HEIGHT WIDTH (default: 150 150)",
    )
    parser.add_argument(
        "--max_slices",
        type=int,
        default=None,
        help="Maximum number of consecutive slices to preserve (default: all slices)",
    )
    parser.add_argument(
        "--start_slice",
        type=int,
        default=0,
        help="Starting slice index (default: 0; note: slice 0 is always skipped)",
    )
    return parser.parse_args()


def get_series_shape_dtype(path: str) -> Tuple[Tuple[int, ...], np.dtype, int]:
    with tiff.TiffFile(path) as tf:
        series = tf.series[0]
        shape = series.shape  # Expect (Z, Y, X) or (Z, Y, X, C)
        dtype = series.dtype
        num_pages = len(series.pages)
    return shape, dtype, num_pages


def main() -> None:
    args = parse_args()

    highlighted_path = args.highlighted_tif
    segmented_path = args.segmented_tif

    if args.out_tif is None:
        base_dir = os.path.dirname(segmented_path)
        out_tif = os.path.join(base_dir, "masked_segmented_stack.tif")
    else:
        out_tif = args.out_tif

    out_png_dir = args.out_png_dir
    if out_png_dir:
        os.makedirs(out_png_dir, exist_ok=True)

    # Inspect shapes/dtypes without loading full data
    h_shape, h_dtype, h_pages = get_series_shape_dtype(highlighted_path)
    s_shape, s_dtype, s_pages = get_series_shape_dtype(segmented_path)

    if h_pages != s_pages:
        raise ValueError(f"Z-slices mismatch: highlighted has {h_pages}, segmented has {s_pages}")
    if h_shape[-2:] != s_shape[-2:]:
        raise ValueError(f"XY size mismatch: highlighted {h_shape[-2:]}, segmented {s_shape[-2:]}")
    
    # Determine slice range (always skip the first slice index 0)
    start_slice = max(1, args.start_slice)
    end_slice = h_pages
    if args.max_slices is not None:
        end_slice = min(h_pages, start_slice + args.max_slices)
    
    actual_slices = max(0, end_slice - start_slice)
    
    print(f"Input shapes: highlighted {h_shape}, segmented {s_shape}")
    print(f"Total slices: {h_pages}")
    print(f"Processing slices: {start_slice} to {end_slice-1} ({actual_slices} slices)")
    print(f"Output size: {args.output_size}")
    print(f"Expected output shape: ({actual_slices}, {args.output_size[0]}, {args.output_size[1]})")

    # Prepare optional NPY memmap output
    npy_mm = None
    if args.out_npy is not None:
        out_height, out_width = args.output_size
        npy_mm = np.memmap(args.out_npy, dtype=s_dtype, mode="w+", shape=(actual_slices, out_height, out_width))

    # Remove existing out_tif to avoid appending to old file
    if os.path.exists(out_tif):
        os.remove(out_tif)

    bigtiff_flag = args.big_tiff or True  # default to True for large data safety

    # Process all slices and collect them in a list
    processed_slices = []
    
    with tiff.TiffFile(highlighted_path) as ht, tiff.TiffFile(segmented_path) as st:
        h_series = ht.series[0]
        s_series = st.series[0]
        for slice_idx, idx in enumerate(range(start_slice, end_slice)):
            h_slice = h_series.pages[idx].asarray()
            s_slice = s_series.pages[idx].asarray()

            # Build mask for highlighted regions
            mask = h_slice > args.highlight_threshold
            
            # Find the bounding box of highlighted regions
            highlighted_rows, highlighted_cols = np.where(mask)
            
            if len(highlighted_rows) == 0:
                # No highlighted regions in this slice, create empty slice
                resized_slice = np.zeros(args.output_size, dtype=s_dtype)
            else:
                # Get bounding box of highlighted regions
                min_row, max_row = highlighted_rows.min(), highlighted_rows.max()
                min_col, max_col = highlighted_cols.min(), highlighted_cols.max()
                
                # Calculate the size of the highlighted region
                region_height = max_row - min_row + 1
                region_width = max_col - min_col + 1
                
                # Target output size
                out_height, out_width = args.output_size
                
                # If highlighted region is smaller than target, pad it
                if region_height < out_height or region_width < out_width:
                    # Create a padded version of the segmented slice
                    padded_slice = np.zeros((max(out_height, region_height), max(out_width, region_width)), dtype=s_dtype)
                    
                    # Place the segmented data in the center
                    start_row = (padded_slice.shape[0] - region_height) // 2
                    start_col = (padded_slice.shape[1] - region_width) // 2
                    padded_slice[start_row:start_row + region_height, start_col:start_col + region_width] = s_slice[min_row:max_row+1, min_col:max_col+1]
                    
                    # Crop to target size from center
                    center_row = padded_slice.shape[0] // 2
                    center_col = padded_slice.shape[1] // 2
                    resized_slice = padded_slice[
                        center_row - out_height//2:center_row + out_height//2,
                        center_col - out_width//2:center_col + out_width//2
                    ]
                else:
                    # Highlighted region is larger than target, crop from center
                    center_row = (min_row + max_row) // 2
                    center_col = (min_col + max_col) // 2
                    
                    # Ensure the crop is entirely within the highlighted region
                    crop_start_row = max(min_row, center_row - out_height//2)
                    crop_end_row = min(max_row + 1, crop_start_row + out_height)
                    crop_start_col = max(min_col, center_col - out_width//2)
                    crop_end_col = min(max_col + 1, crop_start_col + out_width)
                    
                    # Adjust if we hit boundaries
                    if crop_end_row - crop_start_row < out_height:
                        crop_start_row = max(0, crop_end_row - out_height)
                    if crop_end_col - crop_start_col < out_width:
                        crop_start_col = max(0, crop_end_col - out_width)
                    
                    # Extract the crop
                    cropped_slice = s_slice[crop_start_row:crop_end_row, crop_start_col:crop_end_col]
                    
                    # Resize to exact target size if needed
                    if cropped_slice.shape != (out_height, out_width):
                        resized_slice = resize(
                            cropped_slice, 
                            (out_height, out_width), 
                            order=0,  # nearest neighbor
                            preserve_range=True,
                            anti_aliasing=False
                        ).astype(s_dtype)
                    else:
                        resized_slice = cropped_slice
            
            # Ensure we have a 2D slice (remove any extra dimensions)
            if resized_slice.ndim > 2:
                resized_slice = resized_slice.squeeze()
            if resized_slice.ndim == 1:
                # If somehow we got a 1D array, reshape it
                resized_slice = resized_slice.reshape(out_height, out_width)

            processed_slices.append(resized_slice)
            
            # Write NPY slice if requested
            if npy_mm is not None:
                npy_mm[slice_idx] = resized_slice

            # Optionally write PNG slice
            if out_png_dir:
                skio.imsave(
                    os.path.join(out_png_dir, f"slice_{slice_idx:04d}.png"),
                    resized_slice.astype(s_dtype, copy=False),
                    check_contrast=False,
                )

    # Write the entire 3D volume as a single TIFF stack
    volume_3d = np.stack(processed_slices, axis=0)
    print(f"Writing 3D volume with shape: {volume_3d.shape}")
    tiff.imwrite(out_tif, volume_3d, bigtiff=bigtiff_flag)

    if npy_mm is not None:
        # Ensure data is flushed to disk
        npy_mm.flush()

    print("Done.")
    print(f"Masked TIFF written to: {out_tif}")
    if out_png_dir:
        print(f"PNG slices written to: {out_png_dir}")
    if args.out_npy:
        print(f"Masked NPY written to: {args.out_npy}")


if __name__ == "__main__":
    main()
