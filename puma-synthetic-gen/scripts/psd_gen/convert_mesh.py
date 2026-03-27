#!/usr/bin/env python3
"""
Convert psd_gen volume_stack.tif outputs to STL meshes via marching cubes.

Reads the clean binary volume (volume_stack.tif) directly — NOT png_slices/,
which have SEM blur/noise applied and are not geometrically accurate.

Examples
--------
  # Single dataset
  python psd_gen/convert_mesh.py --dataset sigma_050

  # Multiple datasets
  python psd_gen/convert_mesh.py --dataset sigma_025 sigma_050 sigma_075

  # All datasets in the output directory
  python psd_gen/convert_mesh.py --all

  # Custom data directory
  python psd_gen/convert_mesh.py --all --data-dir /path/to/output
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
import trimesh
from skimage.measure import marching_cubes

DEFAULT_DATA = Path(__file__).parent.parent.parent / "output" / "psd_gen"


def convert(dataset_dir: Path, voxel_nm: float, step_size: int) -> None:
    tif_path = dataset_dir / "volume_stack.tif"
    out_path  = dataset_dir / "synthetic_volume.stl"

    if not tif_path.exists():
        print(f"  [SKIP] {dataset_dir.name}: no volume_stack.tif found")
        return

    vol = tifffile.imread(str(tif_path))          # (nz, nx, ny), values 0 or 255
    solid = (vol > 127).astype(np.float32)        # already binary, skip Otsu

    spacing = (voxel_nm, voxel_nm, voxel_nm)      # mesh units = nm
    verts, faces, normals, _ = marching_cubes(
        solid, level=0.5, spacing=spacing, step_size=step_size
    )

    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces,
        vertex_normals=normals,
        process=True,
    )
    mesh.export(str(out_path))

    print(f"  {dataset_dir.name}: "
          f"{verts.shape[0]:,} verts  {faces.shape[0]:,} faces  → {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="convert_mesh.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-dir", type=str, default=str(DEFAULT_DATA),
        help="Root directory containing dataset subdirs (default: output/psd_gen/)",
    )
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--dataset", type=str, nargs="+", metavar="NAME",
        help="One or more dataset subdirectory names",
    )
    target.add_argument(
        "--all", action="store_true",
        help="Convert every subdirectory found in --data-dir",
    )
    ap.add_argument(
        "--voxel-nm", type=float, default=10.0,
        help="Voxel edge length in nm — sets mesh spatial scale (default: 10)",
    )
    ap.add_argument(
        "--step-size", type=int, default=1,
        help="Marching cubes step size; >1 reduces mesh density (default: 1)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        ap.error(f"--data-dir does not exist: {data_dir}")

    if args.all:
        datasets = sorted(d for d in data_dir.iterdir() if d.is_dir())
    else:
        datasets = [data_dir / n for n in args.dataset]

    missing = [d for d in datasets if not d.exists()]
    if missing:
        ap.error(f"Dataset directories not found: {[d.name for d in missing]}")

    print(f"Converting {len(datasets)} dataset(s) in {data_dir} ...")
    for d in datasets:
        convert(d, args.voxel_nm, args.step_size)
    print("Done.")


if __name__ == "__main__":
    main()
