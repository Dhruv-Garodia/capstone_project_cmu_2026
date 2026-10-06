"""Audit and correction of the napari annotations (v1 -> v2).

Findings on the v1 masks of comp_full.tif (frames 0-117, 116 masks):

1. Inside the ROI, pore/solid labels reproduce a *raw, un-denoised global threshold*
   with 100.0 % pixel agreement: ``pore = I < 142`` (frames 0-29), ``I < 146..150``
   (frames 30-60) and ``I < 140`` (frames 61-117). The labels therefore encode shot noise
   (salt-and-pepper boundaries) and change definition between annotation sessions.
2. The thresholds sit far below the cut-face/pore separation: mid-grey shine-through
   (pore back-walls seen through the opening) is labelled solid, giving ROI porosity
   0.14-0.22 versus 0.383 (Otsu) and 0.417 (Ferner et al. 2024) for this region.
3. ``slice_030.png`` is mis-named (should be ``slice_0030.png``); frames 91-92 are missing.
4. The ROI (catalyst-layer band) is the genuinely manual part and is good; it only
   needs light cleanup (holes, specks).

5. The hand-drawn ROI line is loose: it includes a rim of Pt cap (top) and
   membrane / delamination gap (bottom) that is neither CL pore nor CL solid.

The v2 correction keeps the manual ROI, trims the background rim with the envelope
of the solid network (closing, r = 10 px = 60 nm), and replaces the pore/solid split
with one documented rule applied to every frame: per-frame Otsu threshold on the
Gaussian-denoised (sigma = 1) ROI intensities, then removal of sub-noise specks.
The rule is parameter-free and consistent across frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from .io import OUTSIDE, PORE, SOLID, export_to_classes, load_masks, save_mask
from .preprocess import denoise


def implied_threshold(image: np.ndarray, mask_export: np.ndarray, candidates=range(60, 240)) -> tuple[int, float]:
    """Raw-intensity threshold t maximising agreement of (I < t) with the labelled pores inside the ROI."""
    roi = mask_export != 128
    f = image[roi]
    p = (mask_export == 0)[roi]
    hist_p = np.bincount(f[p], minlength=256)
    hist_s = np.bincount(f[~p], minlength=256)
    cp, cs = np.cumsum(hist_p), np.cumsum(hist_s)
    best_t, best_a = 0, -1.0
    for t in candidates:
        # pixels < t predicted pore: correct = pores below t + solids at/above t
        agree = (cp[t - 1] + (cs[-1] - cs[t - 1])) / f.size
        if agree > best_a:
            best_t, best_a = t, agree
    return int(best_t), float(best_a)


def clean_roi(roi: np.ndarray, min_size: int = 2000, max_hole: int = 2000) -> np.ndarray:
    roi = remove_small_objects(roi.astype(bool), max_size=min_size - 1) if _new_skimage() else remove_small_objects(roi.astype(bool), min_size)
    roi = remove_small_holes(roi, max_size=max_hole) if _new_skimage() else remove_small_holes(roi, max_hole)
    return roi


def _new_skimage() -> bool:
    import inspect
    return "max_size" in inspect.signature(remove_small_objects).parameters


def relabel_pores(image: np.ndarray, roi: np.ndarray, sigma: float = 1.0, min_speck: int = 6,
                  threshold: float | None = None) -> tuple[np.ndarray, float]:
    """Pore mask inside ``roi`` from denoised intensity < Otsu(ROI). Returns (pore, threshold)."""
    smooth = denoise(image, sigma)
    t = float(threshold_otsu(smooth[roi])) if threshold is None else float(threshold)
    pore = (smooth < t) & roi
    if _new_skimage():
        pore = remove_small_objects(pore, max_size=min_speck - 1)
        solid = remove_small_objects(roi & ~pore, max_size=min_speck - 1)
    else:
        pore = remove_small_objects(pore, min_speck)
        solid = remove_small_objects(roi & ~pore, min_speck)
    pore = roi & ~solid
    return pore, t


def solid_envelope(solid: np.ndarray, radius: int = 10) -> np.ndarray:
    """Region enclosed by the solid network: closing with a disc of ``radius`` px, holes filled.

    The hand-drawn ROI line is smooth and loose, so it includes a rim of Pt cap (top)
    and membrane/gap (bottom). Those grey pixels are neither CL pore nor CL solid.
    Intersecting the ROI with the solid envelope removes that rim (60 nm at 6 nm/px).
    """
    from skimage.morphology import disk
    p = radius + 2
    closed = ndi.binary_closing(np.pad(solid, p), structure=disk(radius))[p:-p, p:-p]
    return ndi.binary_fill_holes(closed)


def correct_mask(image: np.ndarray, mask_export: np.ndarray, envelope_radius: int = 10,
                 **kw) -> tuple[np.ndarray, dict]:
    """v1 export mask -> v2 class map (0 solid, 1 pore, 2 outside) plus a per-frame record."""
    roi_v1 = mask_export != 128
    roi = clean_roi(roi_v1)
    pore, t = relabel_pores(image, roi, **kw)
    if envelope_radius:
        roi = clean_roi(roi & solid_envelope(roi & ~pore, envelope_radius))
        pore, t = relabel_pores(image, roi, **kw)  # Otsu again without the background rim
    classes = np.full(image.shape, OUTSIDE, np.uint8)
    classes[roi] = SOLID
    classes[pore] = PORE
    t_v1, agree_v1 = implied_threshold(image, mask_export)
    rec = {
        "v1_implied_raw_threshold": t_v1,
        "v1_threshold_agreement": round(agree_v1, 5),
        "v1_porosity": float((mask_export[roi_v1] == 0).mean()),
        "v2_otsu_threshold_denoised": round(t, 2),
        "v2_porosity": float(pore[roi].mean()),
        "roi_pixels_changed": int((roi != roi_v1).sum()),
        "pixels_relabelled_in_roi": int(((mask_export == 0) != pore)[roi & roi_v1].sum()),
    }
    return classes, rec


def correct_folder(stack: np.ndarray, mask_dir: str | Path, out_dir: str | Path, **kw) -> dict:
    """Correct every mask in ``mask_dir`` (keyed by frame index into ``stack``) and write v2 PNGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks = load_masks(mask_dir)
    records = {}
    for z, m in sorted(masks.items()):
        export_to_classes(m)  # validates values
        classes, rec = correct_mask(stack[z], m, **kw)
        save_mask(out_dir / f"slice_{z:04d}.png", classes)
        records[z] = rec
    missing = [z for z in range(min(masks), max(masks) + 1) if z not in masks]
    summary = {
        "n_masks": len(masks),
        "missing_frames": missing,
        "rule": ("ROI = manual ROI (cleaned) intersected with the solid envelope (closing r=10 px); "
                 "pore = GaussianBlur(I, sigma=1) < Otsu(ROI) per frame; specks < 6 px removed"),
        "frames": records,
    }
    (out_dir / "correction_log.json").write_text(json.dumps(summary, indent=2))
    return summary


def roi_z_consistency(masks: dict[int, np.ndarray]) -> dict[int, float]:
    """Dice of each frame's ROI with the previous available frame (flags session jumps)."""
    out, prev = {}, None
    for z in sorted(masks):
        roi = masks[z] != 128
        if prev is not None:
            out[z] = float(2 * (roi & prev).sum() / (roi.sum() + prev.sum()))
        prev = roi
    return out
