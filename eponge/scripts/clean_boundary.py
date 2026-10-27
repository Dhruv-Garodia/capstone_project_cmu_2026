"""Boundary clean-up for layers with overhangs and pockets (uncomp_full, pristine_full).

The v7 boundary correction (auto_correct_real_masks.py) describes the layer as the region between one top
and one bottom surface per column and smooths those surfaces. That suits the flat comp_full band with
hand-brushed boundaries, but on uncomp_full the layer hangs into the PTL void with pockets and overhangs,
and the surface model fills real cavities (up to ~1500 px per frame). This version works on the region
itself instead:

  1. slice consensus over z-2..z+2: a layer pixel is dropped when 3 or more of the 5 slices call it
     outside, and an outside pixel is added only when 4 of 5 call it layer (removes one-slice glitches;
     the stricter rule for adding keeps material from creeping into cavities of the neighbouring slices);
  2. light in-plane smoothing: opening then closing with a radius-2 disk (removes jaggies and specks);
  3. one layer per frame: the largest connected region is kept; optionally (--side-band) thin strips
     in the outer columns are removed, e.g. layer "tongues" climbing the trench walls on pristine_full;
  4. topology: small outside specks enclosed by material (< --max-hole px) become inside and get
     pore/solid from the grey threshold. Unlike v7, large enclosed regions stay outside: on uncomp they
     are PTL-void pockets sealed off in-plane by a thin bridge, not part of the layer.

Pore/solid labels of the input are kept wherever the region is unchanged.

    python eponge/scripts/clean_boundary.py --masks ../runs/uncomp/initial_masks --stack uncomp_full.tif --out ../runs/uncomp/boundary
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=int, default=2, help="slices each side for the consensus (2 = 5 slices)")
    ap.add_argument("--pore-below", type=float, default=140.0)
    ap.add_argument("--max-hole", type=int, default=400, help="largest enclosed outside region filled (px)")
    ap.add_argument("--side-band", type=int, default=0,
                    help="px from each side of the layer in which thin strips are removed (0 = off)")
    ap.add_argument("--side-open-radius", type=int, default=8,
                    help="strips narrower than about twice this radius are removed in the side bands")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "corrected_masks").mkdir(parents=True, exist_ok=True)
    masks = {int(re.fullmatch(r"slice_(\d+)\.png", p.name).group(1)): np.array(Image.open(p))
             for p in Path(a.masks).glob("slice_*.png")}
    frames = sorted(masks)
    stack = tifffile.imread(a.stack, key=frames)
    roi = np.stack([masks[z] != 128 for z in frames])
    votes = ndi.uniform_filter(roi.astype(np.float32), size=(2 * a.radius + 1, 1, 1), mode="nearest")
    new_roi = np.where(roi, votes > 0.5, votes >= 0.8 - 1e-6)
    se = disk(2)
    rows = []
    for i, z in enumerate(frames):
        r = ndi.binary_closing(ndi.binary_opening(new_roi[i], se), se)
        lab, n = ndi.label(r)
        if n:
            r = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
        if a.side_band:                               # thin tongues climbing the trench walls / cap
            cols = np.nonzero(r.any(0))[0]
            if len(cols):
                opened = ndi.binary_opening(r, disk(a.side_open_radius))
                band = np.zeros_like(r)
                band[:, cols.min():cols.min() + a.side_band] = True
                band[:, max(cols.max() - a.side_band + 1, 0):cols.max() + 1] = True
                r = np.where(band, opened, r)
                lab, n = ndi.label(r)
                if n:
                    r = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
        outside = ~r
        ol, on = ndi.label(outside)
        border = np.unique(np.r_[ol[0], ol[-1], ol[:, 0], ol[:, -1]])
        sizes = np.bincount(ol.ravel())
        small = np.nonzero(sizes < a.max_hole)[0]
        enclosed = outside & ~np.isin(ol, border) & np.isin(ol, small)
        r |= enclosed
        old = masks[z]
        sm = ndi.gaussian_filter(stack[i].astype(np.float32), 1.0)
        m = np.full(old.shape, 128, np.uint8)
        keep = r & (old != 128)
        m[keep] = old[keep]
        newin = r & (old == 128)
        m[newin] = np.where(sm[newin] < a.pore_below, 0, 255)
        Image.fromarray(m).save(out / "corrected_masks" / f"slice_{z:04d}.png")
        o = old != 128
        rows.append({"frame": z, "added": float((r & ~o).sum() / o.sum()), "removed": float((o & ~r).sum() / o.sum()),
                     "enclosed_filled_px": int(enclosed.sum())})
    dice = lambda A, B: 2 * (A & B).sum() / (A.sum() + B.sum())
    after = np.stack([np.array(Image.open(out / "corrected_masks" / f"slice_{z:04d}.png")) != 128 for z in frames])
    adj = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1) if frames[i + 1] - frames[i] == 1]
    idx = {z: i for i, z in enumerate(frames)}
    summary = {"frames": len(frames), "radius": a.radius,
               "mean_added": float(np.mean([r["added"] for r in rows])),
               "mean_removed": float(np.mean([r["removed"] for r in rows])),
               "max_changed": float(max(r["added"] + r["removed"] for r in rows)),
               "neighbour_roi_dice_before": float(np.mean([dice(roi[idx[p]], roi[idx[q]]) for p, q in adj])),
               "neighbour_roi_dice_after": float(np.mean([dice(after[idx[p]], after[idx[q]]) for p, q in adj]))}
    (out / "summary.json").write_text(json.dumps({**summary, "per_frame": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
