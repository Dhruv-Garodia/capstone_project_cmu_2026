"""Combine hand-drawn borders with the pipeline's phases.

    python -m pore_pipeline.apply_borders data/comp_full.tif outputs/comp \
        outputs/comp/manual_annotation --out outputs/comp/manual_borders

The border folder holds slice_XXXX.png masks where 128 means "outside the layer".
Only that border is used; the pore/solid values in those files are ignored.
Phases come from the finished pipeline run (de-striped, normalized, smoothed):

  interior (inside the pipeline's own ROI)  -> the run's consensus trimap
  rim (inside the hand border, outside ROI) -> pore if clearly dark, solid if clearly
                                               bright, otherwise uncertain; dark pixels within
                                               EDGE_DARK_PX of the border are ignored (gap/membrane)

Writes three label sets and borders.csv:
  trimap/   uncertain kept
  high/     interior uncertain -> pore   (working assumption), rim uncertain -> ignore
  low/      interior uncertain -> solid,                       rim uncertain -> ignore
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import labels as L
from . import preprocess as P
from . import segment as S

EDGE_DARK_PX = 12     # px from the hand border within which dark rim pixels are ignored
VARIANTS = ("trimap", "high", "low")


def label_slice(frame, region, consensus, cal):
    """(trimap, high, low) for one slice from its smoothed frame, hand region and consensus mask."""
    inner = consensus != L.EXCLUDED
    tri = np.full(region.shape, L.UNCERTAIN, np.uint8)
    tri[frame < cal["low"]] = L.PORE
    tri[frame > cal["high"]] = L.SOLID
    tri[inner] = consensus[inner]
    tri[~region] = L.EXCLUDED
    edge = ndi.distance_transform_edt(region) <= EDGE_DARK_PX
    tri[edge & ~inner & (tri == L.PORE)] = L.EXCLUDED

    unc = tri == L.UNCERTAIN
    high, low = tri.copy(), tri.copy()
    high[unc & inner], low[unc & inner] = L.PORE, L.SOLID
    high[unc & ~inner] = low[unc & ~inner] = L.EXCLUDED
    return tri, high, low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("run")
    ap.add_argument("borders")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    cal = json.loads((run / "summary.json").read_text())["calibration"]
    frames = P.RunFrames(args.tif, run)
    borders = L.slice_files(args.borders)
    for name in VARIANTS:
        (out / name).mkdir(parents=True, exist_ok=True)

    rows = []
    for z, path in sorted(borders.items()):
        region = np.array(Image.open(path).convert("L")) != L.EXCLUDED
        consensus = L.read_mask(run / "masks" / "consensus", z)
        tri, high, low = label_slice(S.smooth(frames(z)), region, consensus, cal)
        for name, mask in zip(VARIANTS, (tri, high, low)):
            L.write_mask(out / name, z, mask)
        rows.append({"slice": z, "region_fraction": region.mean(),
                     "rim_fraction_of_region": (region & (consensus == L.EXCLUDED)).sum() / region.sum(),
                     "uncertain_fraction": (tri == L.UNCERTAIN).sum() / region.sum(),
                     "porosity_low": L.porosity(low), "porosity_high": L.porosity(high)})

    with open(out / "borders.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"{len(rows)} slices -> {out}  porosity low {mean('porosity_low'):.3f}  high {mean('porosity_high'):.3f}  "
          f"uncertain {mean('uncertain_fraction'):.3f}  rim {mean('rim_fraction_of_region'):.3f}")


if __name__ == "__main__":
    main()
