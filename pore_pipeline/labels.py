"""Label values, mask file conventions and overlays shared by every stage.

Every mask in this project is an 8-bit PNG named slice_XXXX.png (zero-based TIFF page):
    0 pore    64 uncertain    128 excluded / ignore    255 solid
"""
import re
from pathlib import Path

import numpy as np
from PIL import Image

PORE, UNCERTAIN, EXCLUDED, SOLID = 0, 64, 128, 255

_NAME = re.compile(r"slice_(\d+)\.png$", re.IGNORECASE)
_OVERLAY = ((PORE, (255, 0, 0), .45), (UNCERTAIN, (255, 210, 0), .45), (EXCLUDED, (0, 90, 255), .35))


def mask_name(z):
    return f"slice_{z:04d}.png"


def read_mask(folder, z):
    return np.array(Image.open(Path(folder) / mask_name(z)).convert("L"))


def write_mask(folder, z, mask):
    Image.fromarray(mask.astype(np.uint8)).save(Path(folder) / mask_name(z))


def slice_files(folder):
    """{slice index: path} for every slice_*.png in a folder, whatever the zero padding."""
    return {int(_NAME.search(p.name).group(1)): p for p in sorted(Path(folder).glob("slice_*.png"))}


def read_stack(folder, zs=None, cols=None):
    """Masks as a (Z, H, W) uint8 array; `cols` keeps a slice of columns to save memory."""
    files = slice_files(folder)
    zs = sorted(files) if zs is None else list(zs)
    cols = cols or slice(None)
    return np.stack([np.array(Image.open(files[z]).convert("L"))[:, cols] for z in zs])


def porosity(mask):
    """Pore fraction of the labelled (pore + solid) pixels."""
    pore, solid = (mask == PORE).sum(), (mask == SOLID).sum()
    return pore / max(int(pore + solid), 1)


def overlay(image, mask):
    """RGB image with pore red, uncertain yellow, excluded blue."""
    rgb = np.stack([np.clip(image, 0, 255)] * 3, -1).astype(np.float32)
    for value, colour, alpha in _OVERLAY:
        m = mask == value
        rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(colour)
    return rgb.astype(np.uint8)
