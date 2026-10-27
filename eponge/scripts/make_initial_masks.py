"""Initial 0/128/255 masks for a stack that has no manual annotation (uncomp_full, pristine_full).

Stands in for the napari step used on comp_full: the catalyst-layer region (inside vs outside) comes from
the MillNet 2D layer prediction, with dark void pockets that open onto the outside removed
(drop_void_pockets) and cleaned to one band per frame; inside it pixels darker than a fixed
grey level (140, the lowest threshold the comp_full annotators used) are marked pore. The result is then
treated exactly like the comp_full masks: boundary correction (auto_correct_real_masks.py, v7), then the
slice-window phase correction (correct_phase_window.py).

    python eponge/scripts/make_initial_masks.py --stack uncomp_full.tif --frames 0-117 --out ../runs/uncomp/initial_masks
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi

from eponge.inference import segment_stack
from eponge.models import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def clean_roi(layer: np.ndarray, min_frac: float = 0.02) -> np.ndarray:
    """Largest connected layer region, small outside specks inside it filled, small layer specks dropped."""
    lab, n = ndi.label(layer)
    if n == 0:
        return layer
    sizes = np.bincount(lab.ravel())[1:]
    roi = lab == (int(np.argmax(sizes)) + 1)
    holes = ndi.binary_fill_holes(roi) & ~roi
    hl, hn = ndi.label(holes)
    if hn:
        hs = np.bincount(hl.ravel())[1:]
        small = np.isin(hl, np.nonzero(hs < min_frac * roi.sum())[0] + 1)
        roi |= small
    return ndi.binary_opening(roi, iterations=2) | (roi & ndi.binary_erosion(roi, iterations=3))


def drop_void_pockets(roi: np.ndarray, img: np.ndarray, dark: float = 55.0, min_px: int = 600,
                      smooth_sd: float = 8.0, smooth_max_grey: float = 130.0) -> np.ndarray:
    """Remove very dark pockets that open onto the outside (the PTL void seen under the layer).

    The layer model was trained on comp_full, where the layer sits on fibres, and on uncomp_full it keeps
    the black shadowed pockets of the PTL void under the layer as layer. A void-like region (black, or
    smooth and grey) inside the layer region that is large and touches the outside is part of the void;
    with smooth_max_grey=256 the same rule removes the smooth protective coating, which on pristine_full
    is as bright as the layer (local sd ~5 vs > 14 for the layer); pores enclosed by material are not
    touched, whatever their size."""
    sm = ndi.gaussian_filter(img.astype(np.float32), 2.0)
    g = ndi.gaussian_filter(img.astype(np.float32), 1.5)
    local_sd = np.sqrt(np.maximum(ndi.uniform_filter(g * g, 15) - ndi.uniform_filter(g, 15) ** 2, 0))
    # void: black, or smooth and grey (deeper void seen at an angle; local sd 2-5 vs 12-55 for pores)
    void_like = (sm < dark) | ((local_sd < smooth_sd) & (sm < smooth_max_grey))
    for _ in range(2):
        lab, n = ndi.label(void_like & roi)
        if n == 0:
            break
        sizes = np.bincount(lab.ravel())
        touching = np.unique(lab[ndi.binary_dilation(~roi, iterations=2) & (lab > 0)])
        drop = [l for l in touching if sizes[l] >= min_px]
        if not drop:
            break
        roi = roi & ~np.isin(lab, drop)
    return roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--frames", default="0-117")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pore-below", type=float, default=140.0)
    ap.add_argument("--smooth-max-grey", type=float, default=130.0,
                    help="smooth regions darker than this that touch the outside are removed (256: any brightness)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-tta", action="store_true", help="single pass instead of 8 flips/rotations (faster on CPU)")
    a = ap.parse_args()
    z0, z1 = (int(v) for v in a.frames.split("-"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stack = tifffile.imread(a.stack, key=range(z0, z1 + 1))
    model, _ = load_checkpoint(ROOT / "runs/millnet_k0/best.pt", a.device)
    torch.set_num_threads(os.cpu_count() or 4)
    cls, _ = segment_stack(model, stack, 0, a.device, tta=not a.no_tta, progress=True)
    fracs = []
    for i, z in enumerate(range(z0, z1 + 1)):
        roi = clean_roi(drop_void_pockets(cls[i] != 2, stack[i], smooth_max_grey=a.smooth_max_grey))
        sm = ndi.gaussian_filter(stack[i].astype(np.float32), 1.0)
        m = np.full(roi.shape, 128, np.uint8)
        m[roi] = np.where(sm[roi] < a.pore_below, 0, 255)
        Image.fromarray(m).save(out / f"slice_{z:04d}.png")
        fracs.append(roi.mean())
    print(f"wrote {z1 - z0 + 1} masks to {out}; layer fraction {np.min(fracs):.3f}-{np.max(fracs):.3f}")


if __name__ == "__main__":
    main()
