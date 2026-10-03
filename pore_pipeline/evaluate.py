"""Label-free checks on masks.

    python -m pore_pipeline.evaluate outputs/comp
    python -m pore_pipeline.evaluate outputs/comp --masks unet=outputs/comp/unet_k3/masks --tag unet --z0 118

1. z-profile (run folders only): porosity per slice for each classical method, with raw
   and normalized ROI brightness. A correct segmentation of a homogeneous layer is flat.
2. In-plane isotropy: two-point correlation of the pore phase along x (image row
   direction) and along z (slicing direction). Both are in-plane directions of the
   electrode, so any x-vs-z difference is an imaging or segmentation artifact, not
   material anisotropy (Sardhara et al. 2022 use the same test). y (through-plane) is
   reported for completeness but is physically distinct.
Writes zprofile.png, isotropy[_tag].png and evaluation[_tag].json into the run folder.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import labels as L

R_MAX = 40          # voxels (240 nm at 6 nm/voxel)
X_WIDTH = 800       # central columns used for the 3D statistics
VOXEL_NM = 6
DEFAULT_MASKS = ("otsu", "ferner_style", "consensus")


def two_point(pore, valid, axis, r_max=R_MAX):
    """S2(r) of the pore phase along one axis, counting only valid-valid pairs."""
    r_max = min(r_max, pore.shape[axis] - 1)
    p, v = pore.astype(np.float32), valid.astype(np.float32)
    s2 = np.empty(r_max + 1)
    for r in range(r_max + 1):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(0, pore.shape[axis] - r), slice(r, None)
        s2[r] = (p[tuple(a)] * p[tuple(b)]).sum() / max((v[tuple(a)] * v[tuple(b)]).sum(), 1)
    return s2


def isotropy(stack, r_max=R_MAX):
    """Two-point curves along z, y, x and summary statistics for a (Z, H, W) label stack."""
    valid = (stack == L.PORE) | (stack == L.SOLID)        # uncertain and excluded voxels drop out
    pore = stack == L.PORE
    s2 = {name: two_point(pore, valid, axis, r_max) for name, axis in (("z", 0), ("y", 1), ("x", 2))}
    phi = s2["x"][0]
    n = min(len(v) for v in s2.values())
    norm = {k: (v[:n] - phi ** 2) / max(phi - phi ** 2, 1e-9) for k, v in s2.items()}   # 1 at r=0, 0 far away
    area = lambda s: float(np.clip(s, 0, None).sum() - 0.5 * (s[0] + max(s[-1], 0)))    # trapezoid, voxels
    stats = {"porosity_3d": float(phi),
             "xz_anisotropy": float(np.linalg.norm(norm["x"] - norm["z"]) / np.linalg.norm(norm["x"])),
             "corr_len_x": area(norm["x"]), "corr_len_z": area(norm["z"]), "corr_len_y": area(norm["y"])}
    return s2, stats


def central_columns(folder, zs=None, width=X_WIDTH):
    files = L.slice_files(folder)
    w = Image.open(next(iter(files.values()))).width
    c0 = max(0, w // 2 - width // 2)
    return L.read_stack(folder, zs, slice(c0, c0 + width))


def zprofile(run, result):
    rows = list(csv.DictReader(open(run / "stats.csv")))
    z = np.array([int(r["slice"]) for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for k, lab in (("porosity_otsu", "Otsu"), ("porosity_ferner_style", "Ferner-style"),
                   ("confident_pore", "consensus: confident pore")):
        y = col(k)
        slope = np.polyfit(z, y, 1)[0] * 100
        ax[0].plot(z, y, lw=1, label=f"{lab}  (mean {y.mean():.3f}, sd {y.std():.3f}, {slope:+.3f}/100 slices)")
        result[k] = {"mean": float(y.mean()), "std": float(y.std()), "slope_per_100_slices": float(slope)}
    ax[0].set_ylabel("porosity in ROI")
    ax[0].legend(fontsize=8)
    ax[0].set_title(run.name)
    for k, lab in (("mean_intensity_roi_raw", "raw"), ("mean_intensity_roi_norm", "normalized")):
        if k in rows[0]:
            ax[1].plot(z, col(k), lw=1, label=lab)
    ax[1].set_ylabel("mean ROI intensity")
    ax[1].set_xlabel("slice")
    ax[1].legend(fontsize=8)
    fig.savefig(run / "zprofile.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--masks", nargs="*", default=[], metavar="NAME=FOLDER",
                    help="mask folders to test; default is the run's otsu, ferner_style and consensus")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--z0", type=int, default=None)
    ap.add_argument("--z1", type=int, default=None)
    args = ap.parse_args()

    run = Path(args.run)
    result = {}
    if not args.masks:
        zprofile(run, result)
    targets = dict(m.split("=", 1) for m in args.masks) or {m: run / "masks" / m for m in DEFAULT_MASKS}

    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 3.6), constrained_layout=True, squeeze=False)
    for ax, (name, folder) in zip(axes[0], targets.items()):
        files = L.slice_files(folder)
        zs = [z for z in sorted(files) if (args.z0 is None or z >= args.z0) and (args.z1 is None or z < args.z1)]
        s2, stats = isotropy(central_columns(folder, zs))
        stats["slices"] = [zs[0], zs[-1]]
        result[f"isotropy_{name}"] = stats
        for k, ls in (("x", "-"), ("z", "--"), ("y", ":")):
            ax.plot(np.arange(len(s2[k])) * VOXEL_NM, s2[k], ls,
                    label="y  (through-plane)" if k == "y" else f"{k}  (in-plane)")
        ax.set_title(f"{name}: x–z anisotropy {stats['xz_anisotropy']:.3f}", fontsize=10)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("S2 pore")
        ax.legend(fontsize=8)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})
    suffix = f"_{args.tag}" if args.tag else ""
    fig.savefig(run / f"isotropy{suffix}.png", dpi=130)
    (run / f"evaluation{suffix}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
