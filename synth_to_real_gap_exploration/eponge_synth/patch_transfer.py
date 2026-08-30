"""
Closing the LAST part of the appearance gap non-parametrically, from the real image itself.

Rationale (CV): a procedural renderer can match statistics, but "indistinguishable" is a statement
about the full patch distribution.  With ONE real image the right tools are single-image,
non-parametric ones — no network to train, nothing to overfit beyond the image you give it:

  * `gpnn_harmonize`   Generative Patch Nearest Neighbours (Granot et al. 2022, "Drop the GAN"):
                       coarse-to-fine, every patch of the simulated image is replaced by its nearest
                       real patch (with a bidirectional-similarity re-ranking so the output stays
                       diverse).  Structure comes from the simulation (and therefore the LABEL);
                       texture, tone, noise and edge rendering come from the real micrograph.
  * `pyramid_match`    Heeger-Bergen-style pyramid histogram matching: match the histogram of every
                       Laplacian band.  Lighter, label-safe, transfers contrast/crispness per scale.
  * `c2st`             Classifier two-sample test — the honest definition of "indistinguishable":
                       a classifier tries to tell real patches from synthetic ones on HELD-OUT real
                       data; accuracy/AUC 0.5 = indistinguishable, 1.0 = trivially separable.
                       Reported at two patch scales (texture, structure) with a real-vs-real control.

Label safety: GPNN moves content by at most ~half a patch at the finest scale (patch/2 px).
Use the harmonised images as an augmentation branch with a boundary-uncertainty band of that width,
or (recommended when many real slices exist) train a CUT translator instead — see cut_train.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.transform import resize


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _to_float(a):
    return np.asarray(a, np.float32) / 255.0


def _patches(img: np.ndarray, p: int):
    w = sliding_window_view(img, (p, p))
    return w.reshape(-1, p * p), w.shape[:2]


def _fold(vals: np.ndarray, grid: Tuple[int, int], p: int, H: int, W: int, sharp: float = 3.5) -> np.ndarray:
    """Centre-weighted aggregation of overlapping patches (Gaussian weights) — much sharper than a plain mean."""
    out = np.zeros((H, W), np.float64); cnt = np.zeros((H, W), np.float64)
    v = vals.reshape(grid[0], grid[1], p, p)
    c = (p - 1) / 2.0
    for i in range(p):
        for j in range(p):
            w = np.exp(-((i - c) ** 2 + (j - c) ** 2) / (2 * (p / sharp) ** 2))
            out[i:i + grid[0], j:j + grid[1]] += w * v[:, :, i, j]
            cnt[i:i + grid[0], j:j + grid[1]] += w
    return (out / cnt).astype(np.float32)


def _pca(keys: np.ndarray, d: int, rng):
    sub = keys[rng.choice(len(keys), size=min(20000, len(keys)), replace=False)]
    mu = sub.mean(0)
    _, _, vt = np.linalg.svd(sub - mu, full_matrices=False)
    return mu, vt[:d].T


def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s_vals, s_idx, s_cnt = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(ref.ravel(), return_counts=True)
    s_q = np.cumsum(s_cnt) / src.size; r_q = np.cumsum(r_cnt) / ref.size
    return np.interp(s_q, r_q, r_vals)[s_idx].reshape(src.shape).astype(np.float32)


# --------------------------------------------------------------------------------------
# 1. Laplacian-pyramid histogram matching (Heeger & Bergen 1995, Laplacian variant)
# --------------------------------------------------------------------------------------
def _lap_pyramid(img, levels):
    pyr, cur = [], img
    for _ in range(levels):
        low = ndi.gaussian_filter(cur, 1.0)
        pyr.append(cur - low)
        cur = low[::2, ::2]
    pyr.append(cur)
    return pyr


def _lap_collapse(pyr):
    cur = pyr[-1]
    for band in reversed(pyr[:-1]):
        up = resize(cur, band.shape, order=1, anti_aliasing=False)
        cur = band + ndi.gaussian_filter(up, 0.5)
    return cur


def pyramid_match(src_u8: np.ndarray, ref_u8: np.ndarray, levels: int = 4, iters: int = 3) -> np.ndarray:
    src, ref = _to_float(src_u8), _to_float(ref_u8)
    out = histogram_match(src, ref)
    rp = _lap_pyramid(ref, levels)
    for _ in range(iters):
        sp = _lap_pyramid(out, levels)
        sp = [histogram_match(b, r) for b, r in zip(sp, rp)]
        out = _lap_collapse(sp)
        out = histogram_match(out, ref)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# 2. GPNN-style multi-scale patch nearest-neighbour harmonisation
# --------------------------------------------------------------------------------------
def gpnn_harmonize(src_u8: np.ndarray, ref_u8: np.ndarray, patch=7, n_scales: int = 5, scale_factor: float = 0.72,
                   iters=2, pca_dim: int = 24, alpha: float = 0.03, k=6, structure_weight: float = 0.25,
                   seed: int = 0, log=None, sharp: float = 3.5) -> np.ndarray:
    """`sharp`: centre-weighting of the patch aggregation (3.5 = soft Gaussian, 10 = centre pixel only).
    `patch`, `iters`, `k` may be ints or per-scale lists (coarse -> fine): use a smaller patch, more
    iterations and k=1 at the finest scales for coherence; larger patch and k>1 at coarse scales for diversity."""
    """
    src_u8: simulated image (structure/labels live here).  ref_u8: real image (appearance source).
    Coarse-to-fine: at each scale the current output is rebuilt from the nearest real patches.
    `structure_weight` re-injects the simulated image at every scale so edges stay where the label says.
    `alpha`/`k`: bidirectional re-ranking (a real patch already 'used' by many queries is penalised).
    """
    rng = np.random.default_rng(seed)
    src, ref = _to_float(src_u8), _to_float(ref_u8)
    H, W = src.shape
    scales = [scale_factor ** i for i in range(n_scales)][::-1]      # coarse -> fine
    out = None
    per = lambda v, i: (v[i] if isinstance(v, (list, tuple)) else v)
    for si, sc in enumerate(scales):
        patch_s, iters_s, k_s = per(patch, si), per(iters, si), per(k, si)
        hs, ws = max(int(round(H * sc)), patch_s + 2), max(int(round(W * sc)), patch_s + 2)
        hr, wr = max(int(round(ref.shape[0] * sc)), patch_s + 2), max(int(round(ref.shape[1] * sc)), patch_s + 2)
        src_s = resize(src, (hs, ws), order=1, anti_aliasing=True).astype(np.float32)
        ref_s = resize(ref, (hr, wr), order=1, anti_aliasing=True).astype(np.float32)
        if out is None:
            out = src_s.copy()
            out = (out - out.mean()) / (out.std() + 1e-6) * ref_s.std() + ref_s.mean()   # coarse tone alignment
        else:
            out = resize(out, (hs, ws), order=1, anti_aliasing=False).astype(np.float32)
            out = (1 - structure_weight) * out + structure_weight * (src_s - src_s.mean() + out.mean())
        keys, kgrid = _patches(ref_s, patch_s)
        mu, P = _pca(keys, pca_dim, rng)
        kz = (keys - mu) @ P
        tree = cKDTree(kz)
        for it in range(iters_s):
            q, qgrid = _patches(out, patch_s)
            qz = (q - mu) @ P
            D, I = tree.query(qz, k=k_s)
            if k_s > 1:   # bidirectional similarity: penalise keys that are far from every query (keeps diversity)
                qtree = cKDTree(qz[rng.choice(len(qz), size=min(len(qz), 30000), replace=False)])
                dmin_k, _ = qtree.query(kz, k=1)
                score = D / (alpha + dmin_k[I])
                best = I[np.arange(len(I)), score.argmin(1)]
            else:
                D, I = D[:, None], I[:, None]; best = I[:, 0]
            out = _fold(keys[best], qgrid, patch_s, hs, ws, sharp=sharp)
            if log:
                log(f"scale {si+1}/{n_scales} ({hs}x{ws}) iter {it+1}: mean NN dist {D[:,0].mean():.4f}")
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# 3. Classifier two-sample test (C2ST): the operational meaning of "indistinguishable"
# --------------------------------------------------------------------------------------
def _sample_patches(img: np.ndarray, p: int, n: int, rng, downsample: int = 1):
    a = img.astype(np.float32)
    if downsample > 1:
        a = resize(a, (a.shape[0] // downsample, a.shape[1] // downsample), order=1, anti_aliasing=True)
    H, W = a.shape
    ys = rng.integers(0, H - p, n); xs = rng.integers(0, W - p, n)
    return np.stack([a[y:y + p, x:x + p].ravel() for y, x in zip(ys, xs)])


def c2st(real_holdout: np.ndarray, synth: np.ndarray, patch: int = 11, n: int = 3000, downsample: int = 1,
         k: int = 5, seed: int = 0) -> Dict[str, float]:
    """
    Returns knn_acc (fraction of a patch's k nearest neighbours in the pooled set that share its label;
    0.5 = indistinguishable) and rf_auc (random-forest 5-fold cross-validated AUC, 0.5 = indistinguishable).
    real_holdout must NOT be the region used as the GPNN reference.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    A = _sample_patches(real_holdout, patch, n, rng, downsample)
    B = _sample_patches(synth, patch, n, rng, downsample)
    X = np.concatenate([A, B]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    mu, sd = A.mean(), A.std() + 1e-6
    Xn = (X - mu) / sd
    tree = cKDTree(Xn)
    _, I = tree.query(Xn, k=k + 1)
    knn_acc = float((y[I[:, 1:]] == y[:, None]).mean())
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, n_jobs=1, random_state=seed)
    prob = cross_val_predict(rf, Xn, y, cv=5, method="predict_proba")[:, 1]
    return dict(knn_acc=knn_acc, rf_auc=float(roc_auc_score(y, prob)))


def indistinguishability_report(real_ref: np.ndarray, real_holdout: np.ndarray, candidates: Dict[str, np.ndarray],
                                seed: int = 0) -> Dict[str, Dict[str, float]]:
    """C2ST at two patch scales for every candidate, plus the real-vs-real control (test noise floor)."""
    out = {"real_vs_real (control)": {}}
    for tag, ds in [("texture_11px", 1), ("structure_33px", 3)]:
        r = c2st(real_holdout, real_ref, patch=11, downsample=ds, seed=seed)
        out["real_vs_real (control)"][tag + "_knn"] = r["knn_acc"]; out["real_vs_real (control)"][tag + "_rf_auc"] = r["rf_auc"]
    for name, img in candidates.items():
        out[name] = {}
        for tag, ds in [("texture_11px", 1), ("structure_33px", 3)]:
            r = c2st(real_holdout, img, patch=11, downsample=ds, seed=seed)
            out[name][tag + "_knn"] = r["knn_acc"]; out[name][tag + "_rf_auc"] = r["rf_auc"]
    return out
