"""
Closing the appearance gap with UNLABELLED real images.

Three complementary tools, cheapest first:
1. `calibrate_appearance`  — fit the SEM simulator parameters to real image statistics
                             (random search, no gradients).  Fixes the gap at the SOURCE so
                             every generated image is already close to the real distribution.
2. `histogram_match`       — per-image intensity transfer to a real reference (classic, robust).
3. `fda`                   — Fourier Domain Adaptation (Yang & Soatto, CVPR 2020): swap the low-frequency
                             amplitude spectrum of a synthetic image with a real one.  Keeps the
                             label-carrying phase/high frequencies, transfers illumination/contrast.
Use 1 at generation time; use 2/3 as on-the-fly training augmentation with random real references.
A learned translator (CUT / CycleGAN) is the fourth rung and only worth it if the three above
leave a measurable gap on the descriptor table.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .descriptors import image_stats
from .sem_render import SEMParams, SEMSimulator

STAT_KEYS = ("mean", "std", "entropy_bits", "extreme_fraction", "grad_mag_mean", "spectrum_slope", "local_contrast")
# scale of each statistic (so the distance is dimensionless)
STAT_SCALE = dict(mean=40.0, std=20.0, entropy_bits=0.6, extreme_fraction=0.08, grad_mag_mean=5.0,
                  spectrum_slope=0.4, local_contrast=6.0)


def stat_distance(a: Dict[str, float], b: Dict[str, float], keys: Sequence[str] = STAT_KEYS) -> float:
    return float(np.sqrt(np.mean([((a[k] - b[k]) / STAT_SCALE[k]) ** 2 for k in keys])))


def calibrate_appearance(material: np.ndarray, voxel_nm: float, real_images: List[np.ndarray],
                         base: Optional[SEMParams] = None, n_iter: int = 60, slices: Sequence[int] = (40, 75, 110),
                         downsample: int = 1, seed: int = 0, log=None) -> Tuple[SEMParams, float, Dict]:
    """
    Random search over SEMParams minimising the texture-statistic distance between simulated slices
    of `material` and the real images.  `downsample` lets you match a real image whose pixel size is
    coarser than the simulation voxel (e.g. 2 when comparing 6 nm voxels with a ~12 nm/px screenshot).
    Returns (best_params, best_distance, best_stats).
    """
    rng = np.random.default_rng(seed)
    real_stats = {k: float(np.mean([image_stats(im)[k] for im in real_images])) for k in STAT_KEYS}
    base = base or SEMParams()

    def run(p):
        sim = SEMSimulator(material, voxel_nm, p, seed=seed)
        ims = []
        for z in slices:
            im, _ = sim.image(min(z, material.shape[0] - 1))
            if downsample > 1:
                h, w = [(s // downsample) * downsample for s in im.shape]
                im = im[:h, :w].reshape(h // downsample, downsample, w // downsample, downsample).mean((1, 3)).astype(np.uint8)
            ims.append(im)
        st = image_stats(np.stack(ims))
        return stat_distance(st, real_stats), st

    best_p, best_d, best_s = base, *run(base)
    if log:
        log(f"init d={best_d:.3f}")
    # coarse random search, then a local refinement around the incumbent
    for i in range(n_iter):
        if i < n_iter * 0.6:
            cand = SEMParams.random(rng, base)
        else:  # local jitter of the incumbent
            u = rng.uniform
            cand = replace(best_p,
                           gain=best_p.gain * u(0.9, 1.1), offset=float(np.clip(best_p.offset + u(-0.02, 0.02), 0, 0.3)),
                           psf_sigma_px=best_p.psf_sigma_px * u(0.85, 1.15), grain_amp=best_p.grain_amp * u(0.8, 1.2),
                           poisson_scale=best_p.poisson_scale * u(0.7, 1.4), edge_enh=best_p.edge_enh * u(0.8, 1.2),
                           atten_nm=best_p.atten_nm * u(0.85, 1.15), charge_amp=best_p.charge_amp * u(0.8, 1.2))
        d, st = run(cand)
        if d < best_d:
            best_p, best_d, best_s = cand, d, st
            if log:
                log(f"iter {i:3d} d={d:.3f} " + " ".join(f"{k}={st[k]:.2f}" for k in STAT_KEYS))
    return best_p, best_d, dict(best=best_s, real=real_stats)


# --------------------------------------------------------------------------------------
# Per-image transfer operators (use as training-time augmentation with random real references)
# --------------------------------------------------------------------------------------
def histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match the intensity histogram of `src` (uint8) to `ref` (uint8)."""
    s_vals, s_idx, s_cnt = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(ref.ravel(), return_counts=True)
    s_q = np.cumsum(s_cnt) / src.size
    r_q = np.cumsum(r_cnt) / ref.size
    mapped = np.interp(s_q, r_q, r_vals)
    return mapped[s_idx].reshape(src.shape).astype(np.uint8)


def fda(src: np.ndarray, ref: np.ndarray, beta: float = 0.05) -> np.ndarray:
    """Fourier Domain Adaptation: replace the centred low-frequency amplitude (window of relative
    half-size `beta`) of `src` with that of `ref`.  Both uint8 2-D; ref is resized by cropping/tiling."""
    src = src.astype(np.float32)
    ref = ref.astype(np.float32)
    H, W = src.shape
    # tile/crop the reference to the source size
    reps = (int(np.ceil(H / ref.shape[0])), int(np.ceil(W / ref.shape[1])))
    ref = np.tile(ref, reps)[:H, :W]
    Fs, Fr = np.fft.fftshift(np.fft.fft2(src)), np.fft.fftshift(np.fft.fft2(ref))
    As, Ps, Ar = np.abs(Fs), np.angle(Fs), np.abs(Fr)
    b = int(np.floor(min(H, W) * beta))
    cy, cx = H // 2, W // 2
    As[cy - b:cy + b + 1, cx - b:cx + b + 1] = Ar[cy - b:cy + b + 1, cx - b:cx + b + 1]
    out = np.fft.ifft2(np.fft.ifftshift(As * np.exp(1j * Ps))).real
    return np.clip(out, 0, 255).astype(np.uint8)


def random_real_style(img: np.ndarray, refs: List[np.ndarray], rng: np.random.Generator,
                      p_hist: float = 0.5, p_fda: float = 0.5, beta_range=(0.01, 0.08)) -> np.ndarray:
    """Training-time augmentation: stochastic histogram matching and/or FDA toward a random real image."""
    out = img
    ref = refs[int(rng.integers(len(refs)))]
    if rng.uniform() < p_fda:
        out = fda(out, ref, beta=float(rng.uniform(*beta_range)))
    if rng.uniform() < p_hist:
        out = histogram_match(out, ref)
    return out
