"""Reading stacks, single images and annotation masks.

Label conventions
-----------------
Two encodings are used throughout the project:

* **Export / PNG encoding** (what napari annotation exports and the web app use):
  ``0 = pore``, ``255 = solid``, ``128 = outside ROI``.
* **Class-index encoding** (what the network predicts):
  ``0 = solid``, ``1 = pore``, ``2 = outside ROI``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

SOLID, PORE, OUTSIDE = 0, 1, 2
CLASS_NAMES = ("solid", "pore", "outside_roi")
EXPORT_VALUES = np.array([255, 0, 128], dtype=np.uint8)  # indexed by class id


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert any grayscale/RGB image to uint8 grayscale using a robust 0.1-99.9 percentile stretch."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [0.1, 99.9])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    """Load a single 2D grayscale image (PNG/JPG/TIF) as uint8."""
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        data = tifffile.imread(path)
        if data.ndim == 3 and data.shape[-1] not in (3, 4):
            data = data[0]
        return to_uint8(data)
    return to_uint8(np.asarray(Image.open(path)))


def load_stack(path: str | Path, frames: range | list[int] | None = None) -> np.ndarray:
    """Load a (Z, H, W) uint8 stack from a multi-page TIFF or a folder of numbered images."""
    path = Path(path)
    if path.is_dir():
        files = sorted_slice_files(path)
        if frames is not None:
            files = [files[i] for i in frames]
        return np.stack([load_image(f) for f in files])
    key = list(frames) if frames is not None else None
    data = tifffile.imread(path, key=key) if key is not None else tifffile.imread(path)
    if data.ndim == 2:
        data = data[None]
    return np.stack([to_uint8(s) for s in data]) if data.dtype != np.uint8 else data


def slice_index(path: str | Path) -> int:
    """Extract the frame index from names like ``slice_0030.png`` or the malformed ``slice_030.png``."""
    match = re.findall(r"(\d+)", Path(path).stem)
    if not match:
        raise ValueError(f"No frame index in file name: {path}")
    return int(match[-1])


def sorted_slice_files(folder: str | Path, pattern: str = "*.png") -> list[Path]:
    files = [p for p in Path(folder).glob(pattern) if p.is_file() and re.fullmatch(r"[A-Za-z]*[_-]?\d+", p.stem)]
    if not files:
        files = [p for ext in ("*.tif", "*.tiff", "*.jpg") for p in Path(folder).glob(ext)]
    return sorted(files, key=slice_index)


def load_masks(folder: str | Path) -> dict[int, np.ndarray]:
    """Load export-encoded masks (0/128/255) keyed by frame index."""
    masks: dict[int, np.ndarray] = {}
    for f in sorted_slice_files(folder):
        z = slice_index(f)
        if z in masks:
            raise ValueError(f"Duplicate mask for frame {z}: {f}")
        masks[z] = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)
    return masks


def export_to_classes(mask: np.ndarray) -> np.ndarray:
    """0/128/255 PNG encoding -> class indices (0 solid, 1 pore, 2 outside)."""
    unexpected = set(np.unique(mask).tolist()) - {0, 128, 255}
    if unexpected:
        raise ValueError(f"Unexpected mask values {sorted(unexpected)}")
    out = np.full(mask.shape, OUTSIDE, dtype=np.uint8)
    out[mask == 255] = SOLID
    out[mask == 0] = PORE
    return out


def classes_to_export(classes: np.ndarray) -> np.ndarray:
    return EXPORT_VALUES[classes]


def save_mask(path: str | Path, classes: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(classes_to_export(classes), mode="L").save(path)
