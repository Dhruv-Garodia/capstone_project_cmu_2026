"""
Microstructure + image descriptors: the quantitative ruler for the synthetic-real gap.

Everything here works on plain numpy arrays so it runs anywhere (no torch, no PuMA).

Conventions
-----------
* Binary volumes are (Z, Y, X) arrays with 1 = SOLID, 0 = PORE (same as the repo's masks:
  "Binary mask directory (1=solid, 0=pore)").
* Lengths are reported in nanometres given `voxel_nm`.
* The pore-size distribution (PSD) uses the local-thickness (maximum inscribed sphere)
  definition used by porespy and by Ferner et al. 2024 (IJHE 59:176), so the lognormal
  (mu, sigma) fitted here is directly comparable to their Table 1:
      pristine            mu=4.657 sigma=0.6352
      PTL-pore-area       mu=4.926 sigma=0.7396   (Drive: uncomp_full.tif)
      PTL-fiber-compressed mu=4.572 sigma=0.6127  (Drive: comp_full.tif)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional

import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------------------
# Published real-data targets (Ferner et al. 2024, blade-coated IrO2 anode CLs, 6 nm voxels)
# --------------------------------------------------------------------------------------
FERNER_2024 = {
    "pristine": dict(porosity=0.515, mean_pore_nm=123.0, std_pore_nm=None, max_pore_nm=600.0,
                     lognormal_mu=4.657, lognormal_sigma=0.6352),
    "uncomp":   dict(porosity=0.558, mean_pore_nm=166.0, std_pore_nm=114.0, max_pore_nm=600.0,
                     lognormal_mu=4.926, lognormal_sigma=0.7396),
    "comp":     dict(porosity=0.417, mean_pore_nm=104.0, std_pore_nm=55.8, max_pore_nm=300.0,
                     lognormal_mu=4.572, lognormal_sigma=0.6127),
}


@dataclass
class VolumeDescriptors:
    porosity: float
    slice_porosity_mean: float
    slice_porosity_std: float
    pore_mean_nm: float           # volume-weighted mean local thickness (pore diameter)
    pore_std_nm: float
    pore_max_nm: float
    lognormal_mu: float           # fit to volume-weighted pore diameters (nm)
    lognormal_sigma: float
    specific_surface_um_inv: float  # solid/pore interface area per total volume, 1/um
    pore_connected_fraction: float  # fraction of pore voxels in the largest connected pore cluster
    solid_connected_fraction: float
    chord_mean_pore_nm: float     # mean pore chord length (x-direction)
    chord_mean_solid_nm: float
    s2_corr_length_nm: float      # distance where normalised S2 autocovariance drops to 1/e
    shape: tuple
    voxel_nm: float

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# --------------------------------------------------------------------------------------
# Core morphology
# --------------------------------------------------------------------------------------
def porosity(solid: np.ndarray) -> float:
    return float(1.0 - solid.mean())


def slice_porosity(solid: np.ndarray, axis: int = 0) -> np.ndarray:
    s = np.moveaxis(solid, axis, 0)
    return 1.0 - s.reshape(s.shape[0], -1).mean(axis=1)


def local_thickness(pore: np.ndarray, radii: Optional[Iterable[int]] = None) -> np.ndarray:
    """
    Local thickness (maximum inscribed sphere DIAMETER, in voxels) of the `pore` phase (bool).
    Same definition as porespy.filters.local_thickness, implemented with two EDTs per radius
    so it runs on any machine.  Returns float array, 0 outside the pore phase.
    """
    pore = pore.astype(bool)
    dt = ndi.distance_transform_edt(pore)
    if radii is None:
        rmax = int(np.ceil(dt.max()))
        # geometric-ish spacing: fine at small radii, coarser at large ones
        radii = sorted(set(list(range(1, min(rmax, 12) + 1)) + [int(r) for r in np.geomspace(12, max(rmax, 12), 10)]))
    lt = np.zeros(pore.shape, dtype=np.float32)
    for r in sorted(radii, reverse=True):
        centres = dt >= r
        if not centres.any():
            continue
        # dilate sphere centres by r  ==  distance to nearest centre <= r
        reach = ndi.distance_transform_edt(~centres) <= r
        sel = reach & pore & (lt == 0)
        lt[sel] = 2.0 * r
    # voxels of the pore phase never reached (thin slivers) get diameter 1
    lt[pore & (lt == 0)] = 1.0
    return lt


def pore_size_distribution(solid: np.ndarray, voxel_nm: float, bins_nm: Optional[np.ndarray] = None):
    """Volume-weighted PSD from local thickness. Returns (bin_centres_nm, volume_fraction, lt_nm)."""
    lt = local_thickness(solid == 0) * voxel_nm
    d = lt[lt > 0]
    if bins_nm is None:
        bins_nm = np.linspace(0, max(d.max(), 1.0), 30)
    hist, edges = np.histogram(d, bins=bins_nm)
    frac = hist / max(hist.sum(), 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, frac, lt


def lognormal_fit(diameters_nm: np.ndarray):
    """MLE lognormal fit (mu, sigma) on diameters > 0 (same functional form as Ferner Eq. 1)."""
    x = np.log(diameters_nm[diameters_nm > 0])
    return float(x.mean()), float(x.std())


def specific_surface(solid: np.ndarray, voxel_nm: float) -> float:
    """Interface area per total volume, in 1/um (face-counting; consistent between datasets)."""
    s = solid.astype(np.int8)
    faces = 0
    for ax in range(3):
        faces += np.abs(np.diff(s, axis=ax)).sum()
    area_nm2 = faces * voxel_nm ** 2
    vol_nm3 = s.size * voxel_nm ** 3
    return float(area_nm2 / vol_nm3 * 1000.0)  # 1/nm -> 1/um


def connected_fraction(phase: np.ndarray) -> float:
    lab, n = ndi.label(phase)
    if n == 0:
        return 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max() / sizes.sum())


def chord_lengths(phase: np.ndarray, axis: int = 2) -> np.ndarray:
    """All chord lengths (in voxels) of `phase` (bool) along `axis`."""
    p = np.moveaxis(phase.astype(np.int8), axis, -1)
    flat = p.reshape(-1, p.shape[-1])
    padded = np.pad(flat, ((0, 0), (1, 1)))
    d = np.diff(padded, axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    # rows align because every run has exactly one start and one end
    return (ends[:, 1] - starts[:, 1]).astype(np.int32)


def two_point_correlation(solid: np.ndarray, max_lag: int = 60):
    """Radially averaged two-point (auto)covariance of the solid phase via FFT.
    Returns (lags_vox, normalised covariance) with value 1 at lag 0."""
    f = solid.astype(np.float32) - solid.mean()
    F = np.fft.rfftn(f)
    ac = np.fft.irfftn(np.abs(F) ** 2, s=f.shape) / f.size
    ac /= ac.flat[0]
    # radial average over the 3 axes (cheap proxy, avoids a full 3-D radial binning)
    out = np.zeros(max_lag)
    for lag in range(max_lag):
        out[lag] = np.mean([ac[lag, 0, 0], ac[0, lag, 0], ac[0, 0, lag]])
    return np.arange(max_lag), out


def ball(r: int) -> np.ndarray:
    zz, yy, xx = np.indices((2 * r + 1,) * 3)
    return (zz - r) ** 2 + (yy - r) ** 2 + (xx - r) ** 2 <= r * r


def paper_cleanup(solid: np.ndarray, r: int = 2) -> np.ndarray:
    """Steps 5-6 of the Ferner 2024 segmentation: 3-D opening + closing with a spherical element.
    Apply this BEFORE measuring when comparing against their published PSD/porosity numbers,
    because those numbers are properties of that measurement operator, not of the raw structure."""
    s = ndi.binary_opening(solid.astype(bool), ball(r))
    return ndi.binary_closing(s, ball(r))


def describe_volume(solid: np.ndarray, voxel_nm: float, max_lag: int = 60, clean_radius: int = 0) -> VolumeDescriptors:
    solid = (np.asarray(solid) > 0)
    if clean_radius > 0:
        solid = paper_cleanup(solid, clean_radius)
    pore = ~solid
    sp = slice_porosity(solid, axis=0)
    centres, frac, lt = pore_size_distribution(solid, voxel_nm)
    d = lt[lt > 0]
    mu, sig = lognormal_fit(d)
    lags, s2 = two_point_correlation(solid, max_lag=min(max_lag, min(solid.shape) - 1))
    below = np.where(s2 < np.exp(-1))[0]
    corr_len = float(lags[below[0]] * voxel_nm) if len(below) else float(lags[-1] * voxel_nm)
    cp = chord_lengths(pore, axis=2)
    cs = chord_lengths(solid, axis=2)
    return VolumeDescriptors(
        porosity=porosity(solid),
        slice_porosity_mean=float(sp.mean()),
        slice_porosity_std=float(sp.std()),
        pore_mean_nm=float(d.mean()),
        pore_std_nm=float(d.std()),
        pore_max_nm=float(d.max()),
        lognormal_mu=mu,
        lognormal_sigma=sig,
        specific_surface_um_inv=specific_surface(solid, voxel_nm),
        pore_connected_fraction=connected_fraction(pore),
        solid_connected_fraction=connected_fraction(solid),
        chord_mean_pore_nm=float(cp.mean() * voxel_nm) if len(cp) else 0.0,
        chord_mean_solid_nm=float(cs.mean() * voxel_nm) if len(cs) else 0.0,
        s2_corr_length_nm=corr_len,
        shape=tuple(int(s) for s in solid.shape),
        voxel_nm=float(voxel_nm),
    )


# --------------------------------------------------------------------------------------
# Grayscale image statistics (appearance gap)
# --------------------------------------------------------------------------------------
def image_stats(img: np.ndarray) -> Dict[str, float]:
    """Appearance descriptors of an 8-bit grayscale image (or stack: stats are pooled)."""
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    hist = np.histogram(x, bins=32, range=(0, 255))[0].astype(np.float64)
    hist /= hist.sum()
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    # fraction of pixels at the extremes: a near-binary image scores ~1, real SEM ~0.05
    extreme = float(((x < 16) | (x > 239)).mean())
    gy, gx = np.gradient(x, axis=(1, 2))
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    lap = np.stack([ndi.laplace(s) for s in x])
    # radial power-spectrum slope (log-log) of the mean 2-D spectrum: texture "roughness"
    P = np.mean([np.abs(np.fft.fftshift(np.fft.fft2(s - s.mean()))) ** 2 for s in x], axis=0)
    h, w = P.shape
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    radial = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    kk = np.arange(2, min(len(radial), min(h, w) // 2))
    slope = float(np.polyfit(np.log(kk), np.log(radial[kk] + 1e-9), 1)[0]) if len(kk) > 5 else 0.0
    # local contrast (std in 7x7 windows), a proxy for granular texture
    loc = np.mean([np.sqrt(np.maximum(ndi.uniform_filter(s ** 2, 7) - ndi.uniform_filter(s, 7) ** 2, 0)).mean() for s in x])
    return dict(mean=float(x.mean()), std=float(x.std()), entropy_bits=entropy,
                extreme_fraction=extreme, grad_mag_mean=float(gmag.mean()),
                laplacian_var=float(lap.var()), spectrum_slope=slope, local_contrast=float(loc))


def gap_table(rows: Dict[str, Dict[str, float]], keys: Optional[Iterable[str]] = None) -> str:
    """Render a markdown table from {name: {descriptor: value}}."""
    names = list(rows)
    if keys is None:
        keys = sorted({k for r in rows.values() for k in r})
    out = ["| descriptor | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in keys:
        cells = []
        for n in names:
            v = rows[n].get(k, None)
            cells.append("–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(out)
