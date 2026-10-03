"""Pore/solid segmentations and their consensus.

otsu          -- one global threshold per stack, inside the catalyst-layer ROI.
ferner_style  -- approximation of Ferner et al. 2024 (Sec. 2.4): gaussian,
                 forward gradient along the streak direction, low-grey removal,
                 high-grey addition, 3D open/close, 3D gaussian. The original
                 MATLAB code and supplementary were not available; every
                 parameter here is set from the data instead of by eye.
consensus     -- pore / solid where both agree, uncertain where they differ.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_multiotsu

from .labels import EXCLUDED, PORE, SOLID, UNCERTAIN

SIGMA_2D = 1.0        # px, step 1
SIGMA_3D = 1.0        # vox, step 6
GRAD_K = 3.0          # sharp transition = GRAD_K x robust noise sigma of the gradient
BALL_R = 1            # vox, step 5 structuring element radius


def smooth(img):
    return ndi.gaussian_filter(img.astype(np.float32), SIGMA_2D)


def calibrate(samples, rois):
    """Per-stack thresholds from smoothed sample slices restricted to their ROI."""
    vals = np.concatenate([smooth(s)[r][::3] for s, r in zip(samples, rois) if r.any()])
    lo, hi = threshold_multiotsu(vals, classes=3)
    return {"otsu": float(threshold_otsu(vals)), "low": float(lo), "high": float(hi)}


def estimate_streak_shift(stack_getter, zs, rois, gap=5, block=24, search=(-4, 13),
                          min_corr=0.4, default=1.0):
    """Rows per slice by which sub-surface content slides (median block match).

    Falls back to `default` (the value measured on the comp/uncomp stacks, and
    close to the 52-degree geometry) when too few blocks match confidently.
    """
    dys = np.arange(*search)
    best = []
    for z in zs:
        a, b, roi = smooth(stack_getter(z)), smooth(stack_getter(z + gap)), rois[z]
        ys, xs = np.where(roi)
        if ys.size == 0:
            continue
        for y in range(ys.min() - search[0], ys.max() - block - search[1], block):
            for x in range(xs.min(), xs.max() - block, block * 2):
                if roi[y:y + block, x:x + block].mean() < 0.9:
                    continue
                pa = a[y:y + block, x:x + block]
                pa = pa - pa.mean()
                na = np.sqrt((pa ** 2).sum())
                if na < 6 * block:
                    continue
                cc = []
                for dy in dys:
                    pb = b[y + dy:y + dy + block, x:x + block]
                    pb = pb - pb.mean()
                    cc.append((pa * pb).sum() / (na * np.sqrt((pb ** 2).sum()) + 1e-9))
                if max(cc) > min_corr:
                    best.append(dys[int(np.argmax(cc))])
    if len(best) < 50:
        print(f"  streak estimate: only {len(best)} confident blocks, using default {default}")
        return default, len(best)
    return float(np.median(best)) / gap, len(best)


def streak_gradient(cur, nxt, shift):
    """Forward difference along the streak: next slice sampled `shift` rows lower."""
    return ndi.shift(nxt, (-shift, 0), order=1, mode="nearest") - cur


def _ball(r):
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (g ** 2).sum(0) <= r * r


def ferner_style(chunk, cal, shift, grad_thr):
    """chunk: (Z, H, W) smoothed float32. Returns boolean solid volume."""
    grad = np.zeros_like(chunk)
    for z in range(chunk.shape[0] - 1):
        grad[z + 1] = streak_gradient(chunk[z], chunk[z + 1], shift)   # rise arriving at z+1
    solid = grad > grad_thr                      # step 2: sharp dark-to-light transition
    solid &= chunk >= cal["low"]                 # step 3: low grey -> pore
    solid |= chunk > cal["high"]                 # step 4: high grey -> solid
    solid[0] = solid[1]                          # first slice has no incoming gradient
    ball, pad = _ball(BALL_R), 2 * BALL_R + 2
    solid = np.pad(solid, ((pad, pad), (0, 0), (0, 0)), mode="edge")       # no erosion at stack ends
    solid = ndi.binary_closing(ndi.binary_opening(solid, ball), ball)      # step 5
    solid = ndi.gaussian_filter(solid.astype(np.float32), SIGMA_3D, mode="nearest") > 0.5   # step 6
    return solid[pad:-pad]


def gradient_noise(samples_pairs, rois, shift):
    """Robust sigma of the streak gradient inside the ROI."""
    vals = np.concatenate([streak_gradient(smooth(a), smooth(b), shift)[r][::5]
                           for (a, b), r in zip(samples_pairs, rois) if r.any()])
    return float(1.4826 * np.median(np.abs(vals - np.median(vals))))


def to_label(solid, roi):
    out = np.full(solid.shape, EXCLUDED, np.uint8)
    out[roi & solid] = SOLID
    out[roi & ~solid] = PORE
    return out


def consensus(solid_a, solid_b, roi):
    out = np.full(solid_a.shape, EXCLUDED, np.uint8)
    out[roi] = UNCERTAIN
    out[roi & solid_a & solid_b] = SOLID
    out[roi & ~solid_a & ~solid_b] = PORE
    return out
