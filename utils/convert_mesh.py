#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an image stack (folder of 2D slices) into a 3D STL mesh.

Dependencies:
    pip install numpy scikit-image trimesh
"""

import os
import glob
from pathlib import Path

import numpy as np
from skimage import io, filters
from skimage.measure import marching_cubes
import trimesh

# =================== CONFIG ===================

# Path to the folder containing slice images
INPUT_FOLDER = "data/pFIB-segmented_resized"
# INPUT_FOLDER = "test_out_single_300_nopadding"

# Supported image extensions
IMAGE_EXTENSIONS = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp"]

# Output STL file
OUTPUT_STL = "real_groundtruth.stl"

# Thresholding options
USE_OTSU_THRESHOLD = True
MANUAL_THRESHOLD = None   # If set, overrides Otsu

# Interpretation of pore color
PORE_IS_DARK = True       # True: pores are dark → solid is bright

# Voxel scaling factors
VOXEL_SIZE_Z = 1.0
VOXEL_SIZE_Y = 1.0
VOXEL_SIZE_X = 1.0

# Marching cubes step size (1 = highest resolution)
MARCHING_STEP_SIZE = 1

# =================================================


def load_image_stack(folder, exts):
    """
    Load a folder of image slices into a 3D volume with shape (Z, Y, X).
    Images are sorted by filename.
    """
    folder = Path(folder)
    files = []
    for ext in exts:
        files.extend(glob.glob(str(folder / ext)))
        files.extend(glob.glob(str(folder / ext.upper())))

    if not files:
        raise FileNotFoundError(f"No images found in {folder}")

    files = sorted(files)

    print(f"Found {len(files)} slices.")
    imgs = []
    for fp in files:
        img = io.imread(fp)
        if img.ndim == 3:
            # RGB → convert to grayscale
            img = np.mean(img, axis=2)
        imgs.append(img)

    # Output shape: (Z, Y, X)
    vol = np.stack(imgs, axis=0)
    return vol


def binarize_volume(volume, use_otsu=True, manual_threshold=None, pore_is_dark=True):
    """
    Convert a grayscale volume into a binary solid/pore mask.
    """
    vol = volume.astype(np.float32)
    max_val = vol.max()
    if max_val > 0:
        vol = vol / max_val

    # Determine threshold
    if manual_threshold is not None:
        thresh = manual_threshold
    elif use_otsu:
        thresh = filters.threshold_otsu(vol)
        print(f"Otsu threshold = {thresh:.4f}")
    else:
        thresh = 0.5
        print(f"Using default threshold = {thresh:.4f}")

    # Determine what counts as solid
    if pore_is_dark:
        solid = vol >= thresh  # dark = pore → bright = solid
    else:
        solid = vol <= thresh  # bright = pore → dark = solid

    return solid


def volume_to_stl(volume_bool, out_path, voxel_size_xyz=(1.0, 1.0, 1.0),
                  step_size=1):
    """
    Run marching cubes on a 3D boolean volume and export to STL.
    """
    print("Running marching_cubes...")
    vol_float = volume_bool.astype(np.float32)

    verts, faces, normals, values = marching_cubes(
        volume=vol_float,
        level=0.5,
        spacing=voxel_size_xyz,  # voxel scaling in (Z, Y, X)
        step_size=step_size
    )

    print(f"Mesh vertices: {verts.shape[0]}, faces: {faces.shape[0]}")

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=True
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    print(f"STL saved to: {out_path}")


def main():
    # 1. Load slice stack
    volume = load_image_stack(INPUT_FOLDER, IMAGE_EXTENSIONS)
    print(f"Volume shape (Z, Y, X): {volume.shape}")

    # 2. Convert to binary solid/pore mask
    solid_vol = binarize_volume(
        volume,
        use_otsu=USE_OTSU_THRESHOLD,
        manual_threshold=MANUAL_THRESHOLD,
        pore_is_dark=PORE_IS_DARK
    )
    print(f"Solid volume dtype: {solid_vol.dtype}, solid fraction = {solid_vol.mean():.3f}")

    # 3. Convert to STL using marching cubes
    voxel_size_xyz = (VOXEL_SIZE_Z, VOXEL_SIZE_Y, VOXEL_SIZE_X)
    volume_to_stl(
        solid_vol,
        out_path=OUTPUT_STL,
        voxel_size_xyz=voxel_size_xyz,
        step_size=MARCHING_STEP_SIZE
    )


if __name__ == "__main__":
    main()
