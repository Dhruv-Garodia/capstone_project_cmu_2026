#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an image stack (folder of slices) to a 3D STL mesh.

依赖：
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

INPUT_FOLDER = "data/pFIB-segmented_resized"
# INPUT_FOLDER = "test_out_single_300_nopadding"

IMAGE_EXTENSIONS = ["*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp"]

OUTPUT_STL = "real_groundtruth.stl"

USE_OTSU_THRESHOLD = True
MANUAL_THRESHOLD = None

PORE_IS_DARK = True

VOXEL_SIZE_Z = 1.0
VOXEL_SIZE_Y = 1.0
VOXEL_SIZE_X = 1.0

MARCHING_STEP_SIZE = 1

# =================================================


def load_image_stack(folder, exts):
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
            # RGB -> gray
            img = np.mean(img, axis=2)
        imgs.append(img)

    # shape: (Z, Y, X)
    vol = np.stack(imgs, axis=0)
    return vol


def binarize_volume(volume, use_otsu=True, manual_threshold=None, pore_is_dark=True):
    vol = volume.astype(np.float32)
    max_val = vol.max()
    if max_val > 0:
        vol = vol / max_val

    if manual_threshold is not None:
        thresh = manual_threshold
    elif use_otsu:
        thresh = filters.threshold_otsu(vol)
        print(f"Otsu threshold = {thresh:.4f}")
    else:
        thresh = 0.5
        print(f"Using default threshold = {thresh:.4f}")

    if pore_is_dark:
        solid = vol >= thresh
    else:
        solid = vol <= thresh

    return solid


def volume_to_stl(volume_bool, out_path, voxel_size_xyz=(1.0, 1.0, 1.0),
                  step_size=1):
    print("Running marching_cubes...")
    vol_float = volume_bool.astype(np.float32)
    verts, faces, normals, values = marching_cubes(
        volume=vol_float,
        level=0.5,                 
        spacing=voxel_size_xyz,    # (dz, dy, dx)
        step_size=step_size
    )

    print(f"Mesh vertices: {verts.shape[0]}, faces: {faces.shape[0]}")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           process=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    print(f"STL saved to: {out_path}")


def main():
    volume = load_image_stack(INPUT_FOLDER, IMAGE_EXTENSIONS)
    print(f"Volume shape (Z, Y, X): {volume.shape}")

    solid_vol = binarize_volume(
        volume,
        use_otsu=USE_OTSU_THRESHOLD,
        manual_threshold=MANUAL_THRESHOLD,
        pore_is_dark=PORE_IS_DARK
    )
    print(f"Solid volume: {solid_vol.dtype}, fraction solid = {solid_vol.mean():.3f}")

    # 3. marching cubes -> STL
    voxel_size_xyz = (VOXEL_SIZE_Z, VOXEL_SIZE_Y, VOXEL_SIZE_X)
    volume_to_stl(
        solid_vol,
        out_path=OUTPUT_STL,
        voxel_size_xyz=voxel_size_xyz,
        step_size=MARCHING_STEP_SIZE
    )


if __name__ == "__main__":
    main()
