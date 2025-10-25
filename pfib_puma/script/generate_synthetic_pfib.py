#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic PFIB-SEM data from targets using PuMA/pumapy.

- Reads a key:value txt spec (see example_specs.txt).
- Generates a 3D porous microstructure with target overall and per-slice porosity stats.
- Approximates pore size distribution (PSD) by layering multiple random-sphere generations.
- Exports per-slice SEM-like PNGs and/or a 3D TIFF, and optionally a 3D mesh directly from PuMA.

Refs:
- PuMA/pumapy docs: https://puma-nasa.readthedocs.io/
"""
import os, re, math, json, random
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np

# PuMA / pumapy
import pumapy as puma
from pumapy.generation.random_spheres import generate_random_spheres
from pumapy.generation.random_fibers import generate_random_fibers

# Export / SEM-like filters
from scipy.ndimage import gaussian_filter
import tifffile
try:
    import imageio.v3 as iio  # for PNGs
except Exception:
    iio = None

# --------------------------- seeding helpers ---------------------------

def _set_all_seeds(seed: Optional[int]) -> None:
    """Set Python, NumPy seeds (and hash seed) for reproducibility."""
    if seed is None or seed < 0:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# --------------------------- parsing ---------------------------

def parse_specs(path: str) -> dict:
    """Parse 'key: value' (or 'key=value') spec file robustly."""
    specs = {}
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    def clean_key(s: str) -> str:
        return (s.replace("\ufeff", "")
                 .replace("\u00a0", " ")
                 .strip())

    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "#" in s:
            s = s.split("#", 1)[0].strip()
            if not s:
                continue
        if ":" in s:
            k, v = s.split(":", 1)
        elif "=" in s:
            k, v = s.split("=", 1)
        else:
            continue
        k = clean_key(k)
        v = v.strip()
        if k:
            specs[k] = v
    return specs

def get_list_float(specs: Dict[str, str], key: str) -> List[float]:
    if key not in specs: return []
    return [float(x.strip()) for x in specs[key].split(",") if x.strip()]

def get_axis_index(axis_char: str) -> int:
    axis_char = axis_char.lower()
    return {"x": 0, "y": 1, "z": 2}.get(axis_char, 2)

# --------------------------- stats ---------------------------

def volume_porosity(bin_vol: np.ndarray) -> float:
    return float(np.mean(bin_vol == 0))

def slice_porosity_stats(bin_vol: np.ndarray, axis: int) -> Dict[str, float]:
    slices = np.moveaxis(bin_vol, axis, 0)
    per = [(sl == 0).mean() for sl in slices]
    return {
        "mean": float(np.mean(per)),
        "std": float(np.std(per, ddof=0)),
        "min": float(np.min(per)),
        "max": float(np.max(per)),
        "series": per,
    }

def union_match_porosity(weights, target):
    w = np.asarray(weights, float); w /= w.sum()
    lo, hi = 0.0, 0.999
    for _ in range(60):  # bisection
        mid = 0.5*(lo+hi)
        tot = 1.0 - np.prod(1.0 - mid*w)
        if tot < target: lo = mid
        else: hi = mid
    return hi*w  # per-bin porosities

# --------------------------- generation helpers ---------------------------

def nm_to_vox(nm: float, voxel_size_nm: float) -> int:
    return max(1, int(round(nm / voxel_size_nm)))

def build_psd_bins(specs: Dict[str, str], voxel_size_nm: float) -> List[Tuple[int, float]]:
    """Return list of (diameter_vox, weight). Normalized weights."""
    bins_nm = get_list_float(specs, "psd_bins_nm")
    weights = get_list_float(specs, "psd_weights")
    if bins_nm and weights and len(bins_nm) == len(weights):
        arr = np.asarray(weights, float)
        arr = arr / arr.sum()
        return [(nm_to_vox(d, voxel_size_nm), float(w)) for d, w in zip(bins_nm, arr)]
    # Fallback: 3 bins from mean/std
    mean_nm = float(specs.get("psd_mean_nm", 150))
    std_nm = float(specs.get("psd_std_nm", 40))
    bins = [mean_nm - std_nm, mean_nm, mean_nm + std_nm]
    bins = [max(10.0, b) for b in bins]
    weights = np.array([0.25, 0.5, 0.25], float)
    return [(nm_to_vox(d, voxel_size_nm), float(w)) for d, w in zip(bins, weights)]

def composite_random_spheres(shape: Tuple[int,int,int],
                             psd_bins: List[Tuple[int,float]],
                             target_porosity: float,
                             allow_intersect: bool = True,
                             seed: Optional[int] = None) -> puma.Workspace:
    """
    Composite by layering random-sphere generations with weighted porosity shares.
    """
    _set_all_seeds(seed)
    weights = [w for _, w in psd_bins]
    porosities = union_match_porosity(weights, target_porosity)
    print("per-bin por targets:", porosities)

    ws = puma.Workspace.from_array(np.ones(shape, dtype=np.uint8))
    for (diam_vox, w), por in zip(psd_bins, porosities):
        tmp = generate_random_spheres(shape=shape, diameter=int(diam_vox),
                                      porosity=float(por), allow_intersect=allow_intersect,
                                      segmented=True)
        print(f"Placing bin diam={diam_vox} vox, target_por={por:.3f}")
        ws.matrix = np.minimum(ws.matrix, tmp.matrix)  # union voids
    return ws

def generate_volume(specs: Dict[str, str], seed: Optional[int] = None) -> puma.Workspace:
    nx, ny, nz = int(specs.get("nx", 512)), int(specs.get("ny", 512)), int(specs.get("nz", 256))
    voxel_size_nm = float(specs.get("voxel_size_nm", 10.0))
    target_phi = float(specs.get("overall_porosity", 0.5))
    generator = specs.get("generator", "spheres").lower()
    allow_intersect = True
    shape = (nx, ny, nz)

    # Make this generation attempt deterministic
    _set_all_seeds(seed)

    if generator == "fibers":
        psd = build_psd_bins(specs, voxel_size_nm)
        print("DEBUG psd_bins_vox,weights:", psd)
        radius_vox = max(1, psd[len(psd)//2][0] // 2)
        ws = generate_random_fibers(shape=shape, radius=int(radius_vox),
                                    porosity=float(target_phi), phi=90, theta=90,
                                    allow_intersect=True)  # deterministic due to _set_all_seeds
    else:
        psd = build_psd_bins(specs, voxel_size_nm)
        print("DEBUG psd_bins_vox,weights:", psd)
        ws = composite_random_spheres(shape, psd, target_phi,
                                      allow_intersect=allow_intersect, seed=seed)

    ws.set_voxel_length(voxel_size_nm * 1e-9)  # meters
    return ws

# --------------------------- matching loop ---------------------------

def meets_slice_targets(stats: Dict[str, float], specs: Dict[str, str], tol=0.02) -> bool:
    targets = {
        "mean": float(specs.get("mean_slice_porosity", stats["mean"])),
        "std": float(specs.get("std_slice_porosity", stats["std"])),
        "min": float(specs.get("min_slice_porosity", stats["min"])),
        "max": float(specs.get("max_slice_porosity", stats["max"])),
    }
    ok = (abs(stats["mean"] - targets["mean"]) <= tol and
          abs(stats["std"] - targets["std"]) <= tol and
          stats["min"] <= targets["min"] + tol and
          stats["max"] >= targets["max"] - tol)
    return ok

def adjust_porosity(ws: puma.Workspace, target_phi: float) -> None:
    """Light-touch adjust by random thinning/thickening via binary erosions/dilations."""
    from scipy.ndimage import binary_dilation, binary_erosion
    vol = (ws.matrix == 0)  # void mask
    phi = vol.mean()
    if abs(phi - target_phi) < 0.002:
        return
    if phi < target_phi:
        vol = binary_dilation(vol, iterations=1)
    else:
        vol = binary_erosion(vol, iterations=1)
    ws.matrix[:] = np.where(vol, 0, 1).astype(np.uint8)

def generate_until_match(specs: Dict[str, str]) -> Tuple[puma.Workspace, Dict[str, float], Dict[str, float]]:
    max_iters = int(specs.get("max_iters", 30))
    slice_axis = get_axis_index(specs.get("slice_axis", "z"))
    target_phi = float(specs.get("overall_porosity", 0.5))

    # base seed from spec; <0 or missing => nondeterministic
    base_seed = int(specs.get("seed", "-1"))

    best_ws, best_loss, best_stats, best_overall = None, float("inf"), None, None

    for it in range(max_iters):
        it_seed = None if base_seed < 0 else (base_seed + it)
        ws = generate_volume(specs, seed=it_seed)

        for _ in range(3):
            adjust_porosity(ws, target_phi)

        vol = ws.matrix
        over_phi = volume_porosity(vol)
        sstats = slice_porosity_stats(vol, slice_axis)

        loss = (abs(over_phi - target_phi) +
                abs(sstats["mean"] - float(specs.get("mean_slice_porosity", sstats["mean"]))) +
                abs(sstats["std"] - float(specs.get("std_slice_porosity", sstats["std"]))) * 0.7)

        if loss < best_loss:
            best_ws, best_loss, best_stats, best_overall = ws.copy(), loss, sstats, {"overall": over_phi}

        if meets_slice_targets(sstats, specs, tol=0.02) and abs(over_phi - target_phi) <= 0.005:
            return ws, sstats, {"overall": over_phi}

    return best_ws, best_stats, best_overall

# --------------------------- exports ---------------------------

def to_sem_like(img_u8: np.ndarray, blur_sigma: float, poisson_scale: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    # blur
    if blur_sigma and blur_sigma > 0:
        img = gaussian_filter(img_u8.astype(np.float32), sigma=blur_sigma)
    else:
        img = img_u8.astype(np.float32)
    # Poisson-ish noise (deterministic if rng is provided)
    if poisson_scale and poisson_scale > 0:
        if rng is None:
            rng = np.random.default_rng()
        lam = np.clip(img / 255.0 * poisson_scale, 0, None)
        noise = rng.poisson(lam).astype(np.float32)
        img = img + (noise * (255.0 / max(poisson_scale, 1e-6)))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def export_outputs(ws: puma.Workspace, specs: Dict[str, str], sstats: Dict[str, float], over: Dict[str, float]):
    out_dir = Path(specs.get("out_dir", "./synthetic_pfib"))
    out_dir.mkdir(parents=True, exist_ok=True)

    export_png = specs.get("export_png", "true").lower() == "true"
    export_tiff3d = specs.get("export_tiff3d", "true").lower() == "true"
    export_mesh = specs.get("export_mesh", "true").lower() == "true"
    sem_like = specs.get("sem_like", "true").lower() == "true"
    blur_sigma = float(specs.get("sem_blur_sigma", 0.8))
    pois_scale = float(specs.get("sem_poisson_scale", 10.0))
    slice_axis = get_axis_index(specs.get("slice_axis", "z"))

    # deterministic SEM noise, if requested
    sem_noise_seed = specs.get("sem_noise_seed", specs.get("seed", None))
    rng = None
    if sem_noise_seed is not None:
        try:
            rng = np.random.default_rng(int(sem_noise_seed))
        except Exception:
            rng = None

    vol = np.moveaxis(ws.matrix, slice_axis, 0)  # slices-first
    vol_u8 = (vol.astype(np.uint8) * 255)        # void=0 black, solid=255 white

    # Save PNGs (SEM-like)
    if export_png and iio is not None:
        png_dir = out_dir / "png_slices"
        png_dir.mkdir(exist_ok=True)
        for i, sl in enumerate(vol_u8):
            img = to_sem_like(sl, blur_sigma, pois_scale, rng=rng) if sem_like else sl
            iio.imwrite(png_dir / f"slice_{i:04d}.png", img)

    # Save 3D TIFF stack (unfiltered binary/grayscale)
    if export_tiff3d:
        tiff_path = out_dir / "volume_stack.tif"
        tifffile.imwrite(str(tiff_path), vol_u8, imagej=True)

    # Direct 3D mesh export using PuMA (no PNGs required)
    if export_mesh:
        try:
            # cutoff=(1,1) extracts the solid phase surface; change to (0,0) for pores if needed.
            ws_to_export=ws.copy()
            ws_to_export.set_voxel_length(1e-2)
            r = puma.render_volume(ws_to_export, cutoff=(1, 1), plot_directly=False)
            mesh_name = specs.get("mesh_name", "synthetic_volume")
            mesh_fmt = specs.get("mesh_format", "obj").lower()  # PuMA provides OBJ; convert later if needed
            out_mesh = out_dir / f"{mesh_name}.obj"
            r.export_obj(str(out_mesh))
            print(f"3D mesh exported to {out_mesh}")
        except Exception as e:
            print(f"[WARN] Mesh export failed: {e}")

    # Dump realized stats
    realized = {
        "overall_porosity": round(over["overall"], 6),
        "mean_slice_porosity": round(sstats["mean"], 6),
        "std_slice_porosity": round(sstats["std"], 6),
        "min_slice_porosity": round(sstats["min"], 6),
        "max_slice_porosity": round(sstats["max"], 6),
        "nslices": int(vol_u8.shape[0]),
    }
    with open(out_dir / "realized_stats.json", "w", encoding="utf-8") as f:
        json.dump(realized, f, indent=2)
    print("Realized stats:", realized)
    print(f"Saved outputs to: {out_dir.resolve()}")

# --------------------------- main ---------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Synthetic PFIB-SEM generator using PuMA/pumapy.")
    ap.add_argument("--spec", required=True, help="Path to txt spec (key: value).")
    args = ap.parse_args()

    specs = parse_specs(args.spec)
    print("DEBUG path:", os.path.abspath(args.spec))
    print("DEBUG first keys:", sorted(list(specs.keys()))[:12])

    required = ["nx","ny","nz","overall_porosity"]
    missing = [k for k in required if k not in specs]
    if missing:
        raise SystemExit(f"Spec missing required keys: {missing}. "
                        f"Got {len(specs)} keys total. "
                        f"Double-check the file content/encoding.")
    print("DEBUG dims:",
        specs.get("nx"), specs.get("ny"), specs.get("nz"),
        "slice_axis:", specs.get("slice_axis"))

    # Set a global/base seed once (optional)
    base_seed = int(specs.get("seed", "-1")) if "seed" in specs else -1
    if base_seed >= 0:
        _set_all_seeds(base_seed)
        print(f"[seed] Using base seed = {base_seed}")

    ws, sstats, over = generate_until_match(specs)
    export_outputs(ws, specs, sstats, over)

if __name__ == "__main__":
    main()
