"""Reimplementation of the pFIB-SEM segmentation of Ferner et al. (2024), Int. J. Hydrogen Energy 59:176.

From Sec. 2.4 of the paper (accepted manuscript, NREL/JA-5900-88803). The original MATLAB/ImageJ code
and its parameter values (supplementary) are not public; the six steps are:

  0. register the stack and match every slice's histogram to a reference slice;
     reslice the stack along the 52 deg streak direction (ImageJ) so the streaks lie in-plane;
  1. 2D Gaussian smoothing                                               -> sigma_2d
  2. forward gradient along the streak; sharp dark-to-light transitions -> solid      (grad_thr)
  3. grey values below a low threshold -> pore                           (low)
  4. grey values above a high threshold -> solid                         (high)
  5. 3D opening and closing with a spherical element                     (ball_r)
  6. 3D Gaussian smoothing, then binarise                                (sigma_3d)

Reslicing along the streak and taking the in-plane forward gradient is the same as differencing each
voxel with the voxel one slice deeper, displaced by the streak rate s along y; that is how step 2 is
computed here (``streak`` in px/slice; measured 1.0 on comp_full). With unknown original parameters,
``tune`` searches all six on data with exact truth (the digital twin) to give the method its best case.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import ndimage as ndi


def _ball(r: int) -> np.ndarray:
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (g ** 2).sum(0) <= r * r


def match_histograms(stack: np.ndarray, ref: int | None = None) -> np.ndarray:
    """Step 0: map every slice's grey-level histogram onto a reference slice (median-brightness slice)."""
    stack = stack.astype(np.float32)
    ref = int(np.argsort(stack.mean((1, 2)))[len(stack) // 2]) if ref is None else ref
    ref_sorted = np.sort(stack[ref].ravel())
    out = np.empty_like(stack)
    q = np.linspace(0, 1, 257)
    ref_q = np.quantile(ref_sorted, q)
    for i, f in enumerate(stack):
        out[i] = np.interp(f, np.quantile(f, q), ref_q)
    return out


def segment(stack: np.ndarray, streak: float = 1.0, sigma_2d: float = 1.0, grad_thr: float = 20.0,
            low: float = 110.0, high: float = 200.0, ball_r: int = 1, sigma_3d: float = 1.0,
            hist_match: bool = True) -> np.ndarray:
    """(Z, Y, X) grey stack -> boolean pore volume, following the six steps above."""
    v = match_histograms(stack) if hist_match else stack.astype(np.float32)
    v = np.stack([ndi.gaussian_filter(f, sigma_2d) for f in v])                       # step 1
    grad = np.zeros_like(v)                                                             # step 2
    for z in range(len(v) - 1):
        nxt = ndi.shift(v[z + 1], (-streak, 0), order=1, mode="nearest")
        grad[z + 1] = nxt - v[z]
    grad[0] = grad[1]
    solid = grad > grad_thr
    solid &= v >= low                                                                   # step 3
    solid |= v > high                                                                   # step 4
    if ball_r > 0:                                                                      # step 5
        b, pad = _ball(ball_r), 2 * ball_r + 2
        s = np.pad(solid, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        solid = ndi.binary_closing(ndi.binary_opening(s, b), b)[pad:-pad]
    if sigma_3d > 0:                                                                    # step 6
        solid = ndi.gaussian_filter(solid.astype(np.float32), sigma_3d, mode="nearest") > 0.5
    return ~solid


def tune(stacks: list[np.ndarray], truths: list[np.ndarray], streak: float = 1.0, n_random: int = 60,
         seed: int = 0, margin: int = 3) -> tuple[dict, float, list]:
    """Random search over the six parameters, maximising pore IoU against exact truth.

    Thresholds are searched relative to the data (percentiles of grey values and of the gradient), so the
    search covers every setting a user could reasonably pick.
    """
    rng = np.random.default_rng(seed)
    allv = np.concatenate([s[::2].ravel() for s in stacks])
    pct = lambda q: float(np.percentile(allv, q))
    best, best_iou, log = None, -1.0, []
    candidates = []
    for _ in range(n_random):
        candidates.append({
            "sigma_2d": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            "grad_thr": float(rng.uniform(0, 60)),
            "low": pct(rng.uniform(2, 45)),
            "high": pct(rng.uniform(55, 99)),
            "ball_r": int(rng.choice([0, 1, 2])),
            "sigma_3d": float(rng.choice([0.0, 0.7, 1.0, 1.5])),
        })
    for c in candidates:
        inter = union = 0
        for st, tr in zip(stacks, truths):
            p = segment(st, streak=streak, **c)[margin:len(st) - margin]
            t = tr[margin:len(st) - margin]
            inter += (p & t).sum(); union += (p | t).sum()
        iou = inter / union
        log.append({**c, "pore_iou": float(iou)})
        if iou > best_iou:
            best, best_iou = c, float(iou)
    # local refinement around the best setting
    for d in itertools.product([0.8, 1.0, 1.2], [0.97, 1.0, 1.03], [0.99, 1.0, 1.01]):
        c = {**best, "grad_thr": best["grad_thr"] * d[0], "low": best["low"] * d[1], "high": best["high"] * d[2]}
        inter = union = 0
        for st, tr in zip(stacks, truths):
            p = segment(st, streak=streak, **c)[margin:len(st) - margin]
            t = tr[margin:len(st) - margin]
            inter += (p & t).sum(); union += (p | t).sum()
        if inter / union > best_iou:
            best, best_iou = c, float(inter / union)
    return best, best_iou, log
