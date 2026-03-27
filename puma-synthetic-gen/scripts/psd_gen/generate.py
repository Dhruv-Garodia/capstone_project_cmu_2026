#!/usr/bin/env python3
"""
Synthetic pFIB-SEM volume generator parameterized by pore size distribution.

Subcommands
-----------
  lognormal  — unimodal log-normal PSD. Pass multiple --sigma values for a batch.
  bimodal    — two-peak log-normal sum PSD.

Both subcommands write a temporary config and delegate to generate_synthetic_pfib.py,
which is left completely untouched.

Examples
--------
  # Single lognormal dataset
  python psd_gen/generate.py lognormal --sigma 0.5 --out-dir output/psd_gen

  # Batch sweep across 5 sigma values at 150-cube resolution
  python psd_gen/generate.py lognormal \\
      --sigma 0.25 0.50 0.75 1.00 1.25 \\
      --nx 150 --ny 150 --nz 150 --out-dir output/psd_gen

  # Bimodal: equal mix of 150 nm and 500 nm pore populations
  python psd_gen/generate.py bimodal \\
      --mu1 150 --mu2 500 --w1 0.5 --w2 0.5 \\
      --nx 150 --ny 150 --nz 150 --out-dir output/psd_gen
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import lognorm

# Path to the unmodified generation engine (two levels up from this file)
GEN_SCRIPT = Path(__file__).parent.parent / "generate_synthetic_pfib.py"

# Default output root — relative to repo root (puma-synthetic-gen/)
DEFAULT_OUT = Path(__file__).parent.parent.parent / "output" / "psd_gen"

# Bins whose normalized weight falls below this threshold are pruned to avoid
# requesting near-zero porosity from the sphere packer (causes slow/failed runs)
MIN_BIN_WEIGHT = 0.05


# ── volume / shared argument defaults ─────────────────────────────────────────

_VOL_DEFAULTS = dict(
    nx=150, ny=150, nz=150,
    voxel_nm=10.0,
    porosity=0.492,
    max_iters=10,
    seed=42,
    sem_blur=0.8,
    sem_poisson=10.0,
)


def _add_volume_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared volume/simulation arguments to a subparser."""
    g = parser.add_argument_group("volume")
    g.add_argument("--nx", type=int, default=_VOL_DEFAULTS["nx"],
                   help="X voxels (default: %(default)s)")
    g.add_argument("--ny", type=int, default=_VOL_DEFAULTS["ny"],
                   help="Y voxels (default: %(default)s)")
    g.add_argument("--nz", type=int, default=_VOL_DEFAULTS["nz"],
                   help="Z voxels / slice count (default: %(default)s)")
    g.add_argument("--voxel-nm", type=float, default=_VOL_DEFAULTS["voxel_nm"],
                   metavar="NM",
                   help="Physical voxel edge length in nm (default: %(default)s)")
    g.add_argument("--porosity", type=float, default=_VOL_DEFAULTS["porosity"],
                   help="Target void fraction 0–1 (default: %(default)s)")
    g.add_argument("--max-iters", type=int, default=_VOL_DEFAULTS["max_iters"],
                   help="Generation attempts to find best porosity match (default: %(default)s)")
    g.add_argument("--seed", type=int, default=_VOL_DEFAULTS["seed"],
                   help="RNG seed for reproducibility (default: %(default)s)")

    g2 = parser.add_argument_group("SEM-like output")
    g2.add_argument("--sem-blur", type=float, default=_VOL_DEFAULTS["sem_blur"],
                    metavar="SIGMA",
                    help="Gaussian blur sigma applied to PNG slices (default: %(default)s)")
    g2.add_argument("--sem-poisson", type=float, default=_VOL_DEFAULTS["sem_poisson"],
                    metavar="SCALE",
                    help="Poisson noise scale applied to PNG slices (default: %(default)s)")
    g2.add_argument("--no-png", action="store_true",
                    help="Skip per-slice PNG export")
    g2.add_argument("--no-tiff", action="store_true",
                    help="Skip 3-D TIFF export")


# ── PSD helpers ───────────────────────────────────────────────────────────────

def _lognorm_bins(sigma: float, mu: float, n_bins: int):
    """Return (bin_centers_nm, normalized_weights) for a unimodal log-normal PSD."""
    lo = max(mu * np.exp(-2 * sigma), 100.0)   # floor at 100 nm = 10 voxels @ 10 nm/vox
    hi = mu * np.exp(2 * sigma)
    bins = np.geomspace(lo, hi, n_bins)
    weights = lognorm.pdf(bins, s=sigma, scale=mu)
    weights /= weights.sum()
    mask = weights >= MIN_BIN_WEIGHT
    bins, weights = bins[mask], weights[mask]
    return bins, weights / weights.sum()


def _bimodal_bins(mu1: float, mu2: float, s1: float, s2: float,
                  w1: float, w2: float, n_bins: int):
    """Return (bin_centers_nm, normalized_weights) for a bimodal log-normal PSD."""
    lo = max(mu1 * np.exp(-2 * s1), 100.0)
    hi = mu2 * np.exp(2 * s2)
    bins = np.geomspace(lo, hi, n_bins)
    raw = w1 * lognorm.pdf(bins, s=s1, scale=mu1) + \
          w2 * lognorm.pdf(bins, s=s2, scale=mu2)
    raw /= raw.sum()
    mask = raw >= MIN_BIN_WEIGHT
    bins, raw = bins[mask], raw[mask]
    return bins, raw / raw.sum()


# ── config writer + runner ────────────────────────────────────────────────────

def _write_temp_config(bins: np.ndarray, weights: np.ndarray,
                       out_dir: Path, args) -> Path:
    """Serialize a config understood by generate_synthetic_pfib.py to a temp file."""
    bins_str = ", ".join(f"{b:.2f}" for b in bins)
    wts_str  = ", ".join(f"{w:.4f}" for w in weights)
    lines = [
        f"nx: {args.nx}",
        f"ny: {args.ny}",
        f"nz: {args.nz}",
        f"voxel_size_nm: {args.voxel_nm}",
        f"overall_porosity: {args.porosity}",
        f"generator: sphere",
        f"max_iters: {args.max_iters}",
        f"seed: {args.seed}",
        f"slice_axis: z",
        f"psd_bins_nm: {bins_str}",
        f"psd_weights: {wts_str}",
        f"out_dir: {out_dir}",
        f"export_png: {'false' if args.no_png else 'true'}",
        f"export_tiff3d: {'false' if args.no_tiff else 'true'}",
        f"export_mesh: false",
        f"sem_like: true",
        f"sem_blur_sigma: {args.sem_blur}",
        f"sem_poisson_scale: {args.sem_poisson}",
    ]
    tmp = Path(tempfile.mktemp(suffix=".txt"))
    tmp.write_text("\n".join(lines))
    return tmp


def _run(cfg_path: Path) -> None:
    """Call generate_synthetic_pfib.py with the given config, then clean up."""
    subprocess.run(
        [sys.executable, str(GEN_SCRIPT), "--spec", str(cfg_path)],
        check=True,
    )
    cfg_path.unlink(missing_ok=True)


# ── subcommand handlers ───────────────────────────────────────────────────────

def _run_lognormal(args) -> None:
    base_out = Path(args.out_dir)
    for sigma in args.sigma:
        # Auto-name: sigma_050 for sigma=0.5, etc.
        tag  = f"{sigma:.2f}".replace(".", "")           # "050"
        name = args.name if (len(args.sigma) == 1 and args.name) \
               else f"sigma_{tag}"
        out_dir = base_out / name
        bins, weights = _lognorm_bins(sigma, args.mu, args.n_bins)

        print(f"\n{'='*60}")
        print(f"lognormal  sigma={sigma}  mu={args.mu} nm  → {out_dir}")
        print(f"  bins (nm) : {np.round(bins).astype(int).tolist()}")
        print(f"  weights   : {np.round(weights, 3).tolist()}")
        print(f"{'='*60}")

        cfg = _write_temp_config(bins, weights, out_dir, args)
        _run(cfg)


def _run_bimodal(args) -> None:
    # Normalize weights so they sum to 1
    total = args.w1 + args.w2
    w1, w2 = args.w1 / total, args.w2 / total

    tag  = f"mu{int(args.mu1)}_{int(args.mu2)}"
    name = args.name or f"bimodal_{tag}"
    out_dir = Path(args.out_dir) / name
    bins, weights = _bimodal_bins(args.mu1, args.mu2,
                                  args.sigma1, args.sigma2,
                                  w1, w2, args.n_bins)

    print(f"\n{'='*60}")
    print(f"bimodal  mu1={args.mu1} nm (w={w1:.2f})  "
          f"mu2={args.mu2} nm (w={w2:.2f})  → {out_dir}")
    print(f"  bins (nm) : {np.round(bins).astype(int).tolist()}")
    print(f"  weights   : {np.round(weights, 3).tolist()}")
    print(f"{'='*60}")

    cfg = _write_temp_config(bins, weights, out_dir, args)
    _run(cfg)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True,
                                title="generation mode")

    # ── lognormal ─────────────────────────────────────────────────────────────
    p_ln = sub.add_parser(
        "lognormal",
        help="Unimodal log-normal PSD (single or batch sigma sweep)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ln.add_argument(
        "--sigma", type=float, nargs="+", required=True,
        help="Log-space std (shape parameter). Multiple values run one dataset each.",
    )
    p_ln.add_argument(
        "--mu", type=float, default=150.0,
        help="Median pore diameter in nm (default: 150)",
    )
    p_ln.add_argument(
        "--n-bins", type=int, default=9,
        help="Number of PSD histogram bins (default: 9)",
    )
    p_ln.add_argument(
        "--name", type=str, default=None,
        help="Dataset name override (single-sigma only; auto-generated otherwise)",
    )
    p_ln.add_argument(
        "--out-dir", type=str, default=str(DEFAULT_OUT),
        help="Root output directory (default: output/psd_gen/)",
    )
    _add_volume_args(p_ln)

    # ── bimodal ───────────────────────────────────────────────────────────────
    p_bm = sub.add_parser(
        "bimodal",
        help="Two-peak log-normal sum PSD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_bm.add_argument("--mu1", type=float, required=True,
                      help="Peak 1 median pore diameter (nm)")
    p_bm.add_argument("--mu2", type=float, required=True,
                      help="Peak 2 median pore diameter (nm)")
    p_bm.add_argument("--sigma1", type=float, default=0.3,
                      help="Peak 1 log-space std (default: 0.3)")
    p_bm.add_argument("--sigma2", type=float, default=0.3,
                      help="Peak 2 log-space std (default: 0.3)")
    p_bm.add_argument("--w1", type=float, default=0.5,
                      help="Relative weight of peak 1 (default: 0.5)")
    p_bm.add_argument("--w2", type=float, default=0.5,
                      help="Relative weight of peak 2 (default: 0.5)")
    p_bm.add_argument("--n-bins", type=int, default=9,
                      help="Number of PSD histogram bins (default: 9)")
    p_bm.add_argument("--name", type=str, default=None,
                      help="Dataset name override (auto-generated if omitted)")
    p_bm.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT),
                      help="Root output directory (default: output/psd_gen/)")
    _add_volume_args(p_bm)

    args = parser.parse_args()

    if args.mode == "lognormal":
        _run_lognormal(args)
    elif args.mode == "bimodal":
        _run_bimodal(args)


if __name__ == "__main__":
    main()
