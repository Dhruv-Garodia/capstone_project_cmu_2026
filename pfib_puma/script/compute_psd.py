#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute pore size distribution (PSD) inside a highlighted region (mask) for PFIB-SEM data.

Inputs:
  --volume  : 3D binary stack (TIFF or .npy). Expected shape (Z, Y, X). Default pores=0, solid=1.
  --mask    : 3D mask stack (TIFF or .npy). Nonzero voxels are ROI; only these voxels are used to sample PSD.

Outputs:
  - Printed config-ready lines:
      psd_bins_nm: <d1>, <d2>, <d3>
      psd_weights: <w1>, <w2>, <w3>
  - Out dir (default psd_out_roi):
      psd_hist.csv  : histogram of pore diameters (nm) within ROI
      psd.png       : quick PSD plot with bin centers/weights overlaid

Notes:
  * We compute EDT on the FULL pore mask, then sample diameters only where ROI mask==True.
    This avoids shrinking diameters near ROI borders (ROI is used for sampling, not as a hard boundary).
  * If your stack is not binary, either threshold it beforehand or pass --mode threshold with --fixed_threshold.
"""

import argparse, os, csv, sys
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# --------------------------- I/O ---------------------------

def load_stack(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        arr = np.load(path)
    else:
        arr = tifffile.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {arr.shape}")
    return arr

def save_hist_plot(d_nm: np.ndarray, centers: np.ndarray, weights: np.ndarray, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Histogram (nm), 10 nm bins up to 99.9th percentile
    upper = max(10.0, float(np.percentile(d_nm, 99.9)))
    bins = np.arange(0, upper + 10.0, 10.0)
    hist, edges = np.histogram(d_nm, bins=bins, density=True)

    with open(outdir / "psd_hist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left_nm", "bin_right_nm", "pdf"])
        for i in range(hist.size):
            w.writerow([edges[i], edges[i+1], hist[i]])

    plt.figure()
    x = 0.5 * (edges[:-1] + edges[1:])
    plt.plot(x, hist, label="PDF (ROI)")
    ymax = hist.max() if hist.size else 1.0
    for c, wgt in zip(centers, weights):
        plt.axvline(c, linestyle="--")
        plt.text(float(c), ymax * 0.9, f"{c:.0f} nm\n{wgt:.2f}", ha="center", va="top")
    plt.xlabel("Pore diameter (nm)")
    plt.ylabel("PDF")
    plt.title("Pore Size Distribution (ROI)")
    plt.tight_layout()
    plt.savefig(outdir / "psd.png", dpi=150)
    plt.close()


# --------------------------- PSD core ---------------------------

def compute_diameters_nm_from_binary(volume: np.ndarray,
                                     voxel_sizes_nm: tuple[float, float, float],
                                     mode: str,
                                     pore_label: int | float,
                                     fixed_threshold: float | None,
                                     mask: np.ndarray,
                                     sample_frac: float = 0.2,
                                     seed: int = 0) -> np.ndarray:
    """
    volume: 3D array (Z, Y, X)
    mask  : 3D array (Z, Y, X); nonzero voxels are within ROI (where we sample diameters)
    mode  : 'binary' or 'threshold'
    pore_label: when mode='binary', which value is 'pore' (0 or 1 or 255)
    fixed_threshold: when mode='threshold', pores = (volume <= fixed_threshold) unless overridden
    """
    vz, vy, vx = voxel_sizes_nm  # careful: we'll pass as (vz, vy, vx) to EDT sampling

    # Build pore mask from volume
    if mode == "threshold":
        if fixed_threshold is None:
            # Otsu if available, else fallback to mean
            try:
                from skimage.filters import threshold_otsu
                thr = threshold_otsu(volume.astype(np.float32))
            except Exception:
                thr = float(volume.mean())
        else:
            thr = fixed_threshold
        pore_full = (volume.astype(np.float32) <= thr)
    else:
        # binary mode: map to pore by equality (support 0/1/255 etc.)
        if volume.dtype == np.bool_:
            pore_full = volume if pore_label else ~volume
        else:
            pore_full = (volume == pore_label)

    # Convert mask to boolean ROI
    roi = (mask != 0)

    # EDT on FULL pore mask (so ROI border does not artificially shrink diameters)
    # sampling order for EDT must match array axes order (Z, Y, X)
    edt_nm = distance_transform_edt(pore_full, sampling=(vz, vy, vx))

    # Sample diameters only inside ROI AND pore voxels
    sel = pore_full & roi
    if not np.any(sel):
        raise ValueError("No pore voxels found inside ROI (mask). Check mask/labels.")

    r_nm = edt_nm[sel]
    d_nm = 2.0 * r_nm
    # remove sub-voxel diameters
    base_vox_nm = min(vx, vy, vz)
    d_nm = d_nm[d_nm > 0.5 * base_vox_nm]

    # optional random subsampling
    if 0 < sample_frac < 1.0 and d_nm.size > 0:
        rng = np.random.default_rng(seed)
        m = int(max(1, round(d_nm.size * sample_frac)))
        idx = rng.choice(d_nm.size, size=m, replace=False)
        d_nm = d_nm[idx]

    return d_nm


def cluster_into_bins(d_nm: np.ndarray, K: int = 3, use_log: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (centers_nm[K], weights[K]), sorted by center.
    """
    if d_nm.size == 0:
        raise ValueError("Empty diameter sample; cannot fit bins.")

    # Try KMeans on log-space for stability; fallback to quantiles
    try:
        from sklearn.cluster import KMeans
        x = np.log(d_nm) if use_log else d_nm
        x = x.reshape(-1, 1)
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(x)
        centers = np.exp(km.cluster_centers_.ravel()) if use_log else km.cluster_centers_.ravel()
        labels = km.labels_
        weights = np.array([(labels == i).mean() for i in range(K)], dtype=float)
    except Exception:
        edges = np.quantile(d_nm, np.linspace(0, 1, K+1))
        centers, weights = [], []
        for i in range(K):
            sel = (d_nm >= edges[i]) & (d_nm < edges[i+1] if i < K-1 else d_nm <= edges[i+1])
            if sel.any():
                centers.append(float(np.median(d_nm[sel])))
                weights.append(float(sel.mean()))
        centers, weights = np.array(centers, float), np.array(weights, float)

    # sort ascending by center
    order = np.argsort(centers)
    centers, weights = centers[order], weights[order]
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return centers, weights


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute PSD within ROI mask for PFIB-SEM volume")
    ap.add_argument("--volume", required=True, help="Path to binary volume (3D TIFF or .npy), shape (Z,Y,X)")
    ap.add_argument("--mask",   required=True, help="Path to ROI mask (3D TIFF or .npy), same shape (Z,Y,X); nonzero=in-ROI")
    ap.add_argument("--voxel_size_nm", type=float, default=None, help="Isotropic voxel size (nm). If given, overrides XY/Z below.")
    ap.add_argument("--voxel_size_xy_nm", type=float, default=10.0, help="Pixel size in XY (nm)")
    ap.add_argument("--voxel_size_z_nm",  type=float, default=10.0, help="Slice thickness Z (nm)")
    ap.add_argument("--bins", type=int, default=3, help="Number of discrete bins for PSD (default 3)")
    ap.add_argument("--sample_frac", type=float, default=0.2, help="Random subsample fraction for speed (0<sf<=1)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    ap.add_argument("--mode", choices=["binary", "threshold"], default="binary", help="How to interpret volume")
    ap.add_argument("--pore_label", type=float, default=0.0, help="When mode=binary, which value is pore (0/1/255 etc.)")
    ap.add_argument("--fixed_threshold", type=float, default=None, help="When mode=threshold, fixed threshold value")
    ap.add_argument("--out_dir", default="psd_out_roi", help="Output directory")
    args = ap.parse_args()

    vol = load_stack(args.volume)
    msk = load_stack(args.mask)

    if vol.shape != msk.shape:
        raise SystemExit(f"Shape mismatch: volume {vol.shape} vs mask {msk.shape}")

    # Decide spacings
    if args.voxel_size_nm is not None:
        vx = vy = vz = float(args.voxel_size_nm)
    else:
        vx = vy = float(args.voxel_size_xy_nm)
        vz = float(args.voxel_size_z_nm)

    print(f"Loaded volume: {vol.shape}, dtype={vol.dtype}")
    print(f"Loaded mask:   {msk.shape}, dtype={msk.dtype}")
    print(f"Voxel sizes (nm): XY={vx}, Z={vz}")
    print(f"Mode={args.mode}, pore_label={args.pore_label}, sample_frac={args.sample_frac}")

    # Compute diameter samples within ROI
    d_nm = compute_diameters_nm_from_binary(
        volume=vol,
        voxel_sizes_nm=(vz, vy, vx),  # (Z,Y,X) pass-through; vy=vx here
        mode=args.mode,
        pore_label=args.pore_label,
        fixed_threshold=args.fixed_threshold,
        mask=msk,
        sample_frac=args.sample_frac,
        seed=args.seed
    )

    print(f"Pore voxels sampled in ROI: {d_nm.size}")
    if d_nm.size == 0:
        raise SystemExit("No pore voxels in ROI; adjust mask or pore_label/threshold.")

    # Fit discrete bins
    centers, weights = cluster_into_bins(d_nm, K=args.bins, use_log=True)

    # Print config-ready lines
    print("\nSuggested config lines (ROI-based):")
    print("psd_bins_nm:", ", ".join(f"{c:.0f}" for c in centers))
    print("psd_weights:", ", ".join(f"{w:.2f}" for w in weights))
    
if __name__ == "__main__":
    main()
