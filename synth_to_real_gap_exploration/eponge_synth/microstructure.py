"""
Digital-twin microstructure generators for pFIB-SEM porous electrodes (numpy/scipy only).

PRIMARY  `catalyst_layer(...)`
    IrO2 primary-particle / ionomer agglomerate network — the material actually imaged in the
    Drive stacks (Ferner et al. 2024: blade-coated IrO2 anode CLs, 6 nm voxels, porosity 42-56 %,
    pores 50-600 nm, lognormal PSD).  Built as:
        1. correlated random field  -> agglomerate skeleton (controls the *secondary* pore PSD)
        2. skeleton decorated with lognormal-radius spheres -> granular, irregular edges
        3. partial ionomer films/necks (second solid sub-phase, darker in SE images)
        4. optional large dense low-porosity inclusions (seen in every real sample)
        5. optional through-plane compression (PTL-fiber-compressed regions -> comp_full.tif)
        6. bisection on the skeleton threshold so the FINAL porosity hits the target exactly
SECONDARY `fiber_network(...)`
    Random straight/curved fibre network (PTL / GDL-like, or any genuinely fibrous target).

Both return `Microstructure(solid, material, meta)` with solid (Z,Y,X) uint8 1=solid,
material 0=pore 1=catalyst/fibre 2=ionomer/binder 3=dense inclusion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Microstructure:
    solid: np.ndarray          # uint8 (Z,Y,X), 1 = solid
    material: np.ndarray       # uint8 (Z,Y,X), 0 pore / 1 catalyst / 2 ionomer / 3 dense inclusion
    meta: Dict = field(default_factory=dict)

    @property
    def porosity(self) -> float:
        return float(1.0 - self.solid.mean())


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _correlated_field(shape, sigma_vox: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    n = rng.standard_normal(shape).astype(np.float32)
    f = ndi.gaussian_filter(n, sigma=sigma_vox, mode="wrap")
    return (f - f.mean()) / (f.std() + 1e-8)


def _union_of_spheres(centres_by_r: Dict[int, np.ndarray]) -> np.ndarray:
    """centres_by_r: {radius_vox: bool array of centre positions} -> bool union of spheres."""
    out = None
    for r, c in centres_by_r.items():
        if not c.any():
            continue
        ball = ndi.distance_transform_edt(~c) <= r
        out = ball if out is None else (out | ball)
    return out if out is not None else np.zeros(next(iter(centres_by_r.values())).shape, bool)


def _sample_centres(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) == 0 or n <= 0:
        return np.zeros(mask.shape, bool)
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    c = np.zeros(mask.size, bool)
    c[pick] = True
    return c.reshape(mask.shape)


# --------------------------------------------------------------------------------------
# PRIMARY: IrO2 / ionomer catalyst layer
# --------------------------------------------------------------------------------------
def _voronoi_cracks(shape, domain_vox, crack_vox, roughness, rng):
    """Mud-crack network: gap of width `crack_vox` along the Voronoi boundaries of Poisson seeds,
    meandering via a smooth coordinate warp.  Returns bool array (True = crack)."""
    from scipy.spatial import cKDTree
    vol = float(np.prod(shape))
    n_seeds = max(2, int(vol / domain_vox ** 3))
    seeds = rng.uniform(0, 1, (n_seeds, 3)) * np.array(shape)
    # periodic-ish: tile seeds so cells wrap at the box faces
    tiles = [seeds + np.array(o) * np.array(shape) for o in
             [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    tree = cKDTree(np.concatenate(tiles))
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    warp_amp = roughness * crack_vox * 3.0
    pts = np.stack([zz + warp_amp * _correlated_field(shape, (8, 8, 8), rng),
                    yy + warp_amp * _correlated_field(shape, (8, 8, 8), rng),
                    xx + warp_amp * _correlated_field(shape, (8, 8, 8), rng)], axis=-1).reshape(-1, 3)
    d, _ = tree.query(pts, k=2)
    width = crack_vox * (1.0 + 0.5 * roughness * _correlated_field(shape, (6, 6, 6), rng).ravel())
    return ((d[:, 1] - d[:, 0]) < np.maximum(width, 0.5)).reshape(shape)


def _polyhedra(shape, n, radius_range_vox, squash_z, rng, faces=(8, 14)):
    """Union of n random convex polyhedra (faceted crystallites)."""
    out = np.zeros(shape, bool)
    for _ in range(n):
        r = rng.uniform(*radius_range_vox)
        c = rng.uniform(0, 1, 3) * np.array(shape)
        K = int(rng.integers(*faces))
        nrm = rng.standard_normal((K, 3)); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        off = r * rng.uniform(0.65, 1.0, K)
        rad = int(np.ceil(r)) + 1
        lo = np.maximum(np.floor(c - rad).astype(int), 0); hi = np.minimum(np.ceil(c + rad).astype(int) + 1, np.array(shape))
        if np.any(hi <= lo):
            continue
        zz, yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = np.stack([(zz - c[0]) / squash_z, yy - c[1], xx - c[2]], axis=-1).astype(np.float32)
        inside = np.all((d @ nrm.T) <= off[None, None, None, :], axis=-1)
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside
    return out


def catalyst_layer(
    shape: Tuple[int, int, int] = (150, 150, 150),
    voxel_nm: float = 6.0,
    porosity: float = 0.515,
    agglomerate_nm: float = 95.0,         # correlation length of the intra-domain skeleton field (secondary pores)
    fine_scale_weight: float = 0.1,       # weight of a finer field (roughens agglomerate surfaces)
    fine_scale_nm: float = 40.0,
    macro_weight: float = 0.25,           # weight of a coarse field: occasional large inter-agglomerate voids
    macro_scale_nm: float = 250.0,
    particle_radius_nm: Tuple[float, float] = (26.0, 0.35),   # lognormal (median, sigma) of primary particles
    particle_overlap: float = 2.5,        # expected sphere-volume coverage factor of the skeleton
    ionomer_nm: float = 6.0,              # film thickness
    ionomer_coverage: float = 0.55,       # fraction of the solid surface carrying a film
    domain_nm: Optional[float] = None,    # mud-crack hierarchy: Voronoi domain size (None = off)
    crack_nm: float = 80.0,               # crack (inter-domain gap) width
    crack_roughness: float = 0.6,         # meandering / width variation of the cracks
    crack_ionomer_fill: float = 0.3,      # fraction of crack volume filled with ionomer (dark solid)
    dense_inclusions: int = 1,            # number of large low-porosity particles (0 to disable)
    inclusion_radius_nm: Tuple[float, float] = (150.0, 320.0),
    inclusion_shape: str = "ellipsoid",   # 'ellipsoid' | 'polyhedron' (faceted crystallites)
    compression_z: float = 1.0,           # <1 squeezes structures through-plane (PTL-fibre-compressed regions)
    seed: int = 0,
    porosity_tol: float = 0.004,
) -> Microstructure:
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)

    # 1. skeleton field (anisotropic sigma implements compression: thinner pores along z)
    sa = agglomerate_nm / voxel_nm / 2.0
    sb = fine_scale_nm / voxel_nm / 2.0
    sc = macro_scale_nm / voxel_nm / 2.0
    fa = _correlated_field(shape, (sa * compression_z, sa, sa), rng)
    fb = _correlated_field(shape, (sb * compression_z, sb, sb), rng)
    fc = _correlated_field(shape, (sc * compression_z, sc, sc), rng)
    fld = (1 - fine_scale_weight - macro_weight) * fa + fine_scale_weight * fb + macro_weight * fc
    fld /= fld.std()

    # particle radius classes (voxels), lognormal; 4 classes for speed
    med, sg = particle_radius_nm
    r_vox = np.clip(np.round(np.exp(rng.normal(np.log(med), sg, 4000)) / voxel_nm), 1, None).astype(int)
    qs = np.quantile(r_vox, [0.125, 0.375, 0.625, 0.875])
    classes = np.unique(np.clip(np.round(qs), 1, None).astype(int))
    near = np.argmin(np.abs(r_vox[:, None] - classes[None]), axis=1)
    class_p = np.bincount(near, minlength=len(classes)) / len(r_vox)
    mean_sphere_vol = float(np.sum(class_p * (4 / 3) * np.pi * classes ** 3))

    # mud-crack domain boundaries (pore, partly ionomer-filled)
    crack = np.zeros(shape, bool)
    if domain_nm is not None:
        crack = _voronoi_cracks(shape, domain_nm / voxel_nm, crack_nm / voxel_nm, crack_roughness, rng)

    # dense inclusions
    inclusion_mask = np.zeros(shape, bool)
    if dense_inclusions > 0:
        rr = (inclusion_radius_nm[0] / voxel_nm, inclusion_radius_nm[1] / voxel_nm)
        if inclusion_shape == "polyhedron":
            inclusion_mask = _polyhedra(shape, dense_inclusions, rr, compression_z, rng)
        else:
            zz, yy, xx = np.indices(shape)
            for _ in range(dense_inclusions):
                rad = rng.uniform(*rr); ax = rad * rng.uniform(0.6, 1.4, 3); c = rng.uniform(0, 1, 3) * np.array(shape)
                e = ((zz - c[0]) / (ax[0] * compression_z)) ** 2 + ((yy - c[1]) / ax[1]) ** 2 + ((xx - c[2]) / ax[2]) ** 2
                inclusion_mask |= e <= 1.0
        micro = _correlated_field(shape, (1.5, 1.5, 1.5), rng) > 1.65   # ~5 % micro-porosity
        inclusion_mask &= ~micro

    cov_field = _correlated_field(shape, (2.0, 2.0, 2.0), np.random.default_rng(seed + 7))
    cov_thr = np.quantile(cov_field, 1 - ionomer_coverage)
    fill_field = _correlated_field(shape, (3.0, 3.0, 3.0), np.random.default_rng(seed + 11))
    fill_thr = np.quantile(fill_field, 1 - crack_ionomer_fill) if crack_ionomer_fill > 0 else np.inf

    def build(q: float):
        thr = np.quantile(fld, q)               # q = pore quantile of the skeleton field
        skel = (fld > thr) & ~crack
        n_centres = int(particle_overlap * skel.sum() / mean_sphere_vol)
        idx = np.flatnonzero(skel)
        pick = rng.choice(idx, size=min(n_centres, len(idx)), replace=False)
        cls_choice = rng.choice(len(classes), size=len(pick), p=class_p)
        centres_by_r = {}
        for k, r in enumerate(classes):
            c = np.zeros(skel.size, bool); c[pick[cls_choice == k]] = True
            centres_by_r[int(r)] = c.reshape(shape)
        cat = _union_of_spheres(centres_by_r) & ~crack
        cat |= inclusion_mask
        film = (ndi.distance_transform_edt(~cat) <= max(ionomer_nm / voxel_nm, 1.0)) & ~cat
        ion = film & (cov_field > cov_thr)
        ion |= crack & (fill_field > fill_thr) & ~cat
        mat = np.zeros(shape, np.uint8)
        mat[cat] = 1
        mat[ion] = 2
        mat[inclusion_mask & cat] = 3
        return mat

    lo, hi = 0.05, 0.97
    mat = None
    for _ in range(12):
        q = 0.5 * (lo + hi)
        mat = build(q)
        por = 1.0 - (mat > 0).mean()
        if abs(por - porosity) < porosity_tol:
            break
        if por < porosity:
            lo = q
        else:
            hi = q
    solid = (mat > 0).astype(np.uint8)
    meta = dict(generator="catalyst_layer", voxel_nm=voxel_nm, target_porosity=porosity,
                realized_porosity=float(1 - solid.mean()), agglomerate_nm=agglomerate_nm,
                fine_scale_weight=fine_scale_weight, fine_scale_nm=fine_scale_nm,
                macro_weight=macro_weight, macro_scale_nm=macro_scale_nm,
                particle_radius_nm=list(particle_radius_nm), particle_overlap=particle_overlap,
                ionomer_nm=ionomer_nm, ionomer_coverage=ionomer_coverage, domain_nm=domain_nm, crack_nm=crack_nm,
                crack_roughness=crack_roughness, crack_ionomer_fill=crack_ionomer_fill,
                dense_inclusions=dense_inclusions, inclusion_radius_nm=list(inclusion_radius_nm),
                inclusion_shape=inclusion_shape, compression_z=compression_z, seed=seed,
                skeleton_quantile=float(q), shape=list(shape), crack_fraction=float(crack.mean()))
    return Microstructure(solid=solid, material=mat, meta=meta)


# --------------------------------------------------------------------------------------
# SECONDARY: fibre network
# --------------------------------------------------------------------------------------
def fiber_network(
    shape: Tuple[int, int, int] = (150, 150, 150),
    voxel_nm: float = 50.0,
    porosity: float = 0.75,
    fiber_diameter_nm: Tuple[float, float] = (400.0, 0.15),   # lognormal (median, sigma)
    orientation: str = "inplane",         # 'inplane' (paper/PTL-like), 'random', or 'z'
    inplane_spread_deg: float = 15.0,     # out-of-plane tilt spread for 'inplane'
    curvature: float = 0.03,              # random-walk turning per step (0 = straight)
    length_nm: float = 4000.0,
    binder_nm: float = 0.0,               # optional binder film thickness at contacts
    seed: int = 0,
) -> Microstructure:
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)
    med, sg = fiber_diameter_nm
    centre = {}
    solid = np.zeros(shape, bool)
    L = length_nm / voxel_nm
    target_solid = 1.0 - porosity
    n_added = 0
    while solid.mean() < target_solid and n_added < 5000:
        r = max(1, int(round(np.exp(rng.normal(np.log(med), sg)) / 2 / voxel_nm)))
        p = rng.uniform(0, 1, 3) * np.array(shape)
        if orientation == "inplane":
            phi = rng.uniform(0, 2 * np.pi)
            tilt = np.deg2rad(rng.normal(0, inplane_spread_deg))
            d = np.array([np.sin(tilt), np.cos(tilt) * np.sin(phi), np.cos(tilt) * np.cos(phi)])
        elif orientation == "z":
            d = np.array([1.0, 0, 0]) + 0.1 * rng.standard_normal(3)
        else:
            d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        pts = []
        for _ in range(int(L)):
            pts.append(p.copy())
            d = d + curvature * rng.standard_normal(3)
            d /= np.linalg.norm(d)
            p = p + d
        pts = np.array(pts)
        ok = np.all((pts >= 0) & (pts < np.array(shape) - 1), axis=1)
        pts = np.round(pts[ok]).astype(int)
        if len(pts) == 0:
            continue
        c = centre.setdefault(r, np.zeros(shape, bool))
        c[pts[:, 0], pts[:, 1], pts[:, 2]] = True
        n_added += 1
        if n_added % 25 == 0:  # rasterise in batches (EDT is the expensive part)
            solid = _union_of_spheres(centre)
    solid = _union_of_spheres(centre) if centre else solid
    mat = np.zeros(shape, np.uint8)
    mat[solid] = 1
    if binder_nm > 0:
        film = (ndi.distance_transform_edt(~solid) <= binder_nm / voxel_nm) & ~solid
        # binder concentrates at contacts: keep film only where >=2 fibres are within reach
        nb = ndi.uniform_filter(solid.astype(np.float32), size=int(2 * binder_nm / voxel_nm) + 3)
        mat[film & (nb > 0.35)] = 2
    solid_u8 = (mat > 0).astype(np.uint8)
    meta = dict(generator="fiber_network", voxel_nm=voxel_nm, target_porosity=porosity,
                realized_porosity=float(1 - solid_u8.mean()), fiber_diameter_nm=list(fiber_diameter_nm),
                orientation=orientation, inplane_spread_deg=inplane_spread_deg, curvature=curvature,
                length_nm=length_nm, binder_nm=binder_nm, seed=seed, n_fibers=n_added, shape=list(shape))
    return Microstructure(solid=solid_u8, material=mat, meta=meta)


# --------------------------------------------------------------------------------------
# Presets calibrated against Ferner et al. 2024 (see calibrate.py for how these were found)
# --------------------------------------------------------------------------------------
PRESETS = {
    "pristine": dict(porosity=0.515, compression_z=1.0),
    "uncomp":   dict(porosity=0.558, compression_z=1.0),
    "comp":     dict(porosity=0.417, compression_z=0.7),
}


def make_preset(name: str, shape=(150, 150, 150), voxel_nm=6.0, seed=0, **overrides) -> Microstructure:
    kw = dict(PRESETS[name])
    kw.update(overrides)
    return catalyst_layer(shape=shape, voxel_nm=voxel_nm, seed=seed, **kw)


# --------------------------------------------------------------------------------------
# SECONDARY: flake / platelet felt (curled thin platelets, e.g. the 5 kV x1000 felt image)
# --------------------------------------------------------------------------------------
def flake_network(
    shape: Tuple[int, int, int] = (150, 150, 150),
    voxel_nm: float = 100.0,
    porosity: float = 0.72,
    flake_length_nm: Tuple[float, float] = (8000.0, 0.35),   # lognormal (median, sigma) of the long axis
    aspect_min: float = 0.35,             # short in-plane axis / long axis
    thickness_nm: Tuple[float, float] = (600.0, 0.3),
    curl: float = 0.35,                   # 0 = flat platelets, ~0.5 = strongly curled
    orientation: str = "random",          # 'random' or 'inplane' (platelets lying flat)
    inplane_spread_deg: float = 25.0,
    seed: int = 0,
) -> Microstructure:
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)
    solid = np.zeros(shape, bool)
    target = 1.0 - porosity
    n = 0
    while solid.mean() < target and n < 4000:
        L = np.exp(rng.normal(np.log(flake_length_nm[0]), flake_length_nm[1])) / voxel_nm
        a = L / 2
        b = a * rng.uniform(aspect_min, 1.0)
        c = max(0.5, np.exp(rng.normal(np.log(thickness_nm[0]), thickness_nm[1])) / voxel_nm / 2)
        # orientation: rotation matrix from a random (or near-in-plane) normal
        if orientation == "inplane":
            tilt = np.deg2rad(rng.normal(0, inplane_spread_deg))
            phi = rng.uniform(0, 2 * np.pi)
            nrm = np.array([np.cos(tilt), np.sin(tilt) * np.cos(phi), np.sin(tilt) * np.sin(phi)])
        else:
            nrm = rng.standard_normal(3)
        nrm /= np.linalg.norm(nrm)
        u = np.cross(nrm, rng.standard_normal(3)); u /= np.linalg.norm(u)
        v = np.cross(nrm, u)
        R = np.stack([u, v, nrm])            # rows: local axes (u, v, w) in world coords
        ctr = rng.uniform(0, 1, 3) * np.array(shape)
        rad = int(np.ceil(max(a, b) + c + curl * a + 1))
        lo = np.maximum(np.floor(ctr - rad).astype(int), 0)
        hi = np.minimum(np.ceil(ctr + rad).astype(int) + 1, np.array(shape))
        if np.any(hi <= lo):
            continue
        zz, yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = np.stack([zz - ctr[0], yy - ctr[1], xx - ctr[2]], axis=-1).astype(np.float32)
        loc = d @ R.T                         # local coords (u, v, w)
        uu, vv, ww = loc[..., 0], loc[..., 1], loc[..., 2]
        # curled platelet: mid-surface w = curl * (u^2 / a) ; irregular outline via noisy radius
        mid = curl * (uu ** 2) / max(a, 1e-3)
        rr = (uu / a) ** 2 + (vv / b) ** 2
        outline = 1.0 + 0.25 * np.sin(3 * np.arctan2(vv, uu) + rng.uniform(0, 6.28)) * rng.uniform(0, 1)
        inside = (np.abs(ww - mid) <= c) & (rr <= outline)
        solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside
        n += 1
    mat = np.zeros(shape, np.uint8)
    mat[solid] = 1
    meta = dict(generator="flake_network", voxel_nm=voxel_nm, target_porosity=porosity,
                realized_porosity=float(1 - solid.mean()), flake_length_nm=list(flake_length_nm),
                aspect_min=aspect_min, thickness_nm=list(thickness_nm), curl=curl, orientation=orientation,
                inplane_spread_deg=inplane_spread_deg, seed=seed, n_flakes=n, shape=list(shape))
    return Microstructure(solid=mat, material=mat, meta=meta)


# calibrated presets (see calibration_result.json / REEVALUATION.md)
PRESETS.update({
    "pristine": dict(porosity=0.515 - 0.03, agglomerate_nm=95, macro_weight=0.25, fine_scale_weight=0.1,
                     macro_scale_nm=250, particle_radius_nm=(26, 0.35), compression_z=1.0),
    "uncomp":   dict(porosity=0.558 - 0.03, agglomerate_nm=130, macro_weight=0.30, fine_scale_weight=0.1,
                     macro_scale_nm=250, particle_radius_nm=(26, 0.35), compression_z=1.0),
    "comp":     dict(porosity=0.417 - 0.03, agglomerate_nm=120, macro_weight=0.08, fine_scale_weight=0.1,
                     macro_scale_nm=250, particle_radius_nm=(26, 0.35), compression_z=0.85),
})
PRESETS.update({
    # image-1 left panel: dense speckled domains separated by meandering cracks (mud-crack CL)
    "mudcrack":  dict(porosity=0.40, agglomerate_nm=90, macro_weight=0.0, fine_scale_weight=0.25, fine_scale_nm=35,
                      macro_scale_nm=250, particle_radius_nm=(24, 0.35), domain_nm=720, crack_nm=40,
                      crack_roughness=0.7, crack_ionomer_fill=0.4, dense_inclusions=0),
    # image-1 right panel: faceted IrO2 crystallites in an ionomer/agglomerate matrix
    "faceted":   dict(porosity=0.45, agglomerate_nm=100, macro_weight=0.2, fine_scale_weight=0.15, macro_scale_nm=250,
                      particle_radius_nm=(22, 0.35), ionomer_coverage=0.85, ionomer_nm=10, dense_inclusions=7,
                      inclusion_radius_nm=(90, 240), inclusion_shape="polyhedron"),
})
FAMILIES = {"catalyst_layer": catalyst_layer, "fiber_network": fiber_network, "flake_network": flake_network}

try:
    from .felt import sheet_felt
    FAMILIES["sheet_felt"] = sheet_felt
except ImportError:
    pass
