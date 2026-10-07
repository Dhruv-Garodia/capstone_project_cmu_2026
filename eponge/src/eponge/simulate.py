"""Digital twin of pFIB-SEM serial sectioning with shine-through, for evaluation with exact truth.

Real FIB-SEM stacks of porous electrodes have no ground truth: the defining artefact,
shine-through, makes material *behind* the cut plane visible through pore openings. This module
renders synthetic stacks from 3D structures whose cut-plane phase is known exactly.

Imaging model (per cut plane z, per image pixel (y, x)):

* the viewing ray enters the sample at the cut plane and, at depth t slices behind it, reaches
  voxel (z + t, y + s*t, x). ``s`` is the streak rate: sub-surface material slides ~1 px/slice in
  y as the mill approaches (measured on comp_full and by pore_pipeline);
* solid at the cut plane (t = 0) -> cut face: bright, with edge brightening at pore rims;
* pore at the cut plane -> the first solid along the ray at depth t* is seen with brightness
  floor + (back - floor) * exp(-(t* - 1) / depth_decay); nothing within ``t_max`` -> floor (black);
* detector: Gaussian PSF, signal-dependent noise, vertical curtaining stripes, 8-bit saturation.

Truth for slice z is the phase of voxel layer z itself. Default parameters are calibrated to the
grey-level distribution of the comp_full catalyst layer (see ``calibrate``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Imaging:
    streak: float = 1.0          # rows per slice that sub-surface material slides (sign: moves down with milling)
    t_max: int = 24              # deepest slice that still contributes signal
    cut: float = 222.0           # cut-face brightness
    edge: float = 30.0           # extra brightness at cut-face pixels bordering a pore
    back: float = 196.0          # brightness of a back wall right behind the plane
    floor: float = 48.0          # signal from deep pores
    depth_decay: float = 5.0     # slices; shine-through attenuation length
    texture: float = 12.0        # sd of smooth brightness texture on solid
    psf: float = 0.8             # px, Gaussian blur
    noise: float = 9.0           # sd of additive noise (grey levels), plus shot-like component
    shot: float = 0.35           # sd multiplier of sqrt(signal)
    curtain: float = 6.0         # sd of vertical stripe amplitude
    grain: float = 10.0          # sd of fine (sub-particle) grain on the cut face
    streaks: float = 8.0         # sd of vertically elongated streak texture (shine-through/curtain look)


def random_structure(shape, porosity: float, corr_px: float, rng, macro_voids: int = 0,
                     anisotropy=(1.0, 1.0, 1.0), fine_weight: float = 0.0, fine_px: float = 1.6) -> np.ndarray:
    """Boolean solid volume (Z, Y, X) from a thresholded Gaussian random field.

    ``fine_weight`` > 0 mixes in a second, finer field: agglomerates of fine particles with
    coarse inter-agglomerate pores, as in IrO2 catalyst layers.
    """
    def grf(sig):
        f = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), [sig * a for a in anisotropy], mode="wrap")
        return (f - f.mean()) / (f.std() + 1e-9)
    field = grf(corr_px)
    if fine_weight > 0:
        field = (1 - fine_weight) * field + fine_weight * grf(fine_px)
    solid = field > np.quantile(field, porosity)
    for _ in range(macro_voids):  # occasional large voids, as seen in uncompressed layers
        c = [rng.integers(0, s) for s in shape]
        r = rng.uniform(8, 18)
        zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
        solid &= ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) > r ** 2
    return solid


def render_slice(solid: np.ndarray, z: int, p: Imaging, rng) -> tuple[np.ndarray, np.ndarray]:
    """Image (uint8) and truth (bool pore) of cut plane z of a solid volume."""
    Z, Y, X = solid.shape
    depth = np.full((Y, X), np.inf, np.float32)
    rows = np.arange(Y)
    for t in range(0, p.t_max + 1):
        if z + t >= Z:
            break
        r = np.clip(np.round(rows + p.streak * t).astype(int), 0, Y - 1)
        hit = solid[z + t][r] & np.isinf(depth)
        depth[hit] = t
    pore_plane = ~solid[z]
    img = np.full((Y, X), p.floor, np.float32)
    seen = np.isfinite(depth) & (depth > 0)
    img[seen] = p.floor + (p.back - p.floor) * np.exp(-(depth[seen] - 1) / p.depth_decay)
    face = depth == 0
    tex = ndi.gaussian_filter(rng.standard_normal((Y, X)).astype(np.float32), 3) * (p.texture / 0.09)
    rim = face & ndi.binary_dilation(pore_plane, iterations=1)
    grain = ndi.gaussian_filter(rng.standard_normal((Y, X)).astype(np.float32), 0.8) * (p.grain / 0.33)
    img[face] = p.cut + tex[face] + grain[face]
    img[rim] += p.edge
    streak = ndi.gaussian_filter(rng.standard_normal((Y, X)).astype(np.float32), (10.0, 0.7)) * (p.streaks / 0.11)
    img += streak
    img = ndi.gaussian_filter(img, p.psf)
    img += rng.standard_normal((Y, X)).astype(np.float32) * (p.noise + p.shot * np.sqrt(np.clip(img, 0, None)))
    img += ndi.gaussian_filter1d(rng.standard_normal(X).astype(np.float32), 1.5)[None, :] * p.curtain
    return np.clip(np.round(img), 0, 255).astype(np.uint8), pore_plane


def render_stack(solid: np.ndarray, p: Imaging, rng, z_range=None):
    """Render every cut plane that has ``t_max`` slices of material behind it."""
    zs = range(0, solid.shape[0] - p.t_max) if z_range is None else z_range
    imgs, truth = zip(*(render_slice(solid, z, p, rng) for z in zs))
    return np.stack(imgs), np.stack(truth)


def real_geometry(classes: np.ndarray, z_range, n: int, depth: int = 96, width: int = 256, seed: int = 0):
    """3D solid volumes cut from a real segmentation (classes: 0 solid, 1 pore, 2 outside).

    Each volume is a ``depth`` x H x ``width`` block inside the catalyst layer for every slice of
    the block, so the twin's structures carry real IrO2 morphology. ``z_range`` keeps train and
    test geometry apart (e.g. slices 0-219 for training, 240-353 for testing).
    """
    from .metrics import largest_inner_rectangle
    rng = np.random.default_rng(seed)
    z0s, z1 = list(z_range)[0], list(z_range)[-1] + 1
    out = []
    for _ in range(n):
        z0 = int(rng.integers(z0s, max(z0s + 1, z1 - depth)))
        block = classes[z0:z0 + depth]
        rs, cs = largest_inner_rectangle((block != 2).all(0))
        if cs.stop - cs.start < width or rs.stop - rs.start < 48:
            continue
        x0 = int(rng.integers(cs.start, cs.stop - width + 1))
        solid = block[:, rs, x0:x0 + width] == 0
        if rng.random() < 0.5:
            solid = solid[:, :, ::-1]
        out.append(np.ascontiguousarray(solid))
    return out


def make_dataset(n_volumes: int, shape=(96, 256, 256), seed: int = 0, p: Imaging | None = None,
                 geometries: list | None = None):
    """Varied structures (porosity 0.30-0.55, correlation length 2.5-5 px, some macro-voids) and
    lightly randomised imaging, so a model cannot memorise one setting."""
    rng = np.random.default_rng(seed)
    base = p or Imaging()
    out = []
    for i in range(n_volumes if geometries is None else len(geometries)):
        if geometries is not None:          # real-derived structure
            solid, phi, corr = geometries[i], float(1 - geometries[i].mean()), float("nan")
        else:
            phi = rng.uniform(0.30, 0.55)
            corr = rng.uniform(2.5, 5.0)
            solid = random_structure(shape, phi, corr, rng, macro_voids=int(rng.integers(0, 3)),
                                     anisotropy=(1.0, rng.uniform(1.0, 1.6), 1.0),
                                     fine_weight=rng.uniform(0.0, 0.45), fine_px=rng.uniform(1.2, 2.0))
        q = Imaging(**{**asdict(base),
                       "depth_decay": base.depth_decay * rng.uniform(0.7, 1.4),
                       "back": base.back * rng.uniform(0.92, 1.05),
                       "streak": base.streak * rng.uniform(0.85, 1.15)})
        imgs, truth = render_stack(solid, q, rng)
        out.append({"images": imgs, "truth": truth, "porosity_target": phi, "corr_px": corr, "imaging": asdict(q)})
    return out


def calibrate(real_roi_values: np.ndarray, grid=None, seed: int = 0, solid: np.ndarray | None = None,
              min_contrast: float = 0.0) -> tuple[Imaging, dict]:
    """Fit cut / back / floor / depth_decay so the simulated grey-level percentiles match the real layer's.

    Percentiles alone cannot tell a bright cut face from bright material seen through a pore, so only
    physical settings are searched: material behind the plane is at least `min_contrast` darker than the
    cut face (back <= cut - min_contrast)."""
    pct = [1, 5, 25, 50, 75, 95]
    target = np.percentile(real_roi_values, pct)
    rng = np.random.default_rng(seed)
    solid = random_structure((56, 160, 160), 0.40, 3.5, rng) if solid is None else solid
    best, best_err = None, np.inf
    grid = grid or {"depth_decay": [6, 10, 14, 18, 24, 32], "back": [175, 185, 200, 215], "cut": [212, 222, 232],
                    "floor": [45, 60, 75, 90, 105]}
    from itertools import product
    keys = list(grid)
    for vals in product(*grid.values()):
        p = Imaging(**dict(zip(keys, vals)))
        if p.back > p.cut - min_contrast:
            continue
        imgs, _ = render_stack(solid, p, rng, range(0, 24, 3))
        err = float(np.abs(np.percentile(imgs, pct) - target).mean())
        if err < best_err:
            best, best_err = p, err
    imgs, _ = render_stack(solid, best, rng, range(0, 24, 3))
    return best, {"target_percentiles": target.tolist(), "sim_percentiles": np.percentile(imgs, pct).tolist(),
                  "mean_abs_error": best_err, "params": asdict(best)}
