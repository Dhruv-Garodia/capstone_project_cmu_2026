#!/usr/bin/env python3
"""Tune an automatic labeller on the REAL stacks, for use on synthetic ones.

The synthetic start masks used to be built by bisecting a threshold until
porosity hit 0.97 x truth. That made "the start masks are 1-2% low" an input
rather than a finding.

Here the locally-adaptive threshold is tuned on the real data only: its level
is chosen so that, on real frames, it reproduces the porosity of the team's
real start masks. That level is then applied unchanged to synthetic stacks,
whose grey scale is calibrated to the real one. Whatever porosity error comes
out on synthetic data is an outcome; truth is never consulted.
"""
import json
from pathlib import Path
import numpy as np, tifffile
from PIL import Image
from scipy import ndimage as ndi

from paths import REAL_DIR as LAB
from paths import LABELS_DIR as DL
SETS = {"comp": (LAB / "comp_full.tif", DL / "team_review"),
        "uncomp": (LAB / "uncomp_full.tif", DL / "team_review_uncomp"),
        "pristine": (LAB / "pristine_full.tif", DL / "team_review_pristine")}
ADAPT, WIN = 0.30, 251


def adaptive_offset(stack, roi):
    data = stack.astype(np.float32)
    size = (0, WIN, WIN)
    tot = ndi.uniform_filter(np.where(roi, data, 0), size)
    shr = ndi.uniform_filter(roi.astype(np.float32), size)
    field = np.where(shr > 0.02, tot / np.maximum(shr, 1e-6), 0).astype(np.float32)
    for z in range(len(field)):
        if roi[z].any():
            field[z][~roi[z]] = data[z][roi[z]].mean()
    return data, ADAPT * (field - float(field[roi].mean()))


def main():
    out = {}
    for name, (tif, pkg) in SETS.items():
        frames = sorted(int(p.stem[6:]) for p in (pkg / "start_masks").glob("slice_*.png"))
        pick = frames[:: max(1, len(frames) // 12)][:12]
        masks = np.stack([np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png"))
                          for z in pick])
        stack = np.stack([np.asarray(tifffile.imread(tif, key=z)) for z in pick])
        roi = masks != 128
        target = float((masks == 0).sum() / roi.sum())
        data, off = adaptive_offset(stack, roi)
        n = float(roi.sum())
        por = lambda L: float(((data <= L + off) & roi).sum() / n)
        lo, hi = 0, 255
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if por(mid) < target else (lo, mid)
        level = min((lo, hi), key=lambda L: abs(por(L) - target))
        pred = (data <= level + off) & roi
        agree = float((pred[roi] == (masks == 0)[roi]).mean())
        out[name] = {"level": int(level), "adapt": ADAPT, "win": WIN,
                     "real_start_porosity": round(target, 4),
                     "labeller_porosity_on_real": round(por(level), 4),
                     "pixel_agreement_with_real_start_masks": round(agree, 4)}
        print(f"{name:<9} level {level:>3}  real start porosity {target:.4f}  "
              f"labeller {por(level):.4f}  agrees with real start masks on {agree:.2%}")
    Path("real_labeller.json").write_text(json.dumps(out, indent=2))
    print("\nwrote real_labeller.json")


if __name__ == "__main__":
    main()
