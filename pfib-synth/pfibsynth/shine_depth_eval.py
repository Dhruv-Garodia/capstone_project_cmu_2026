#!/usr/bin/env python3
"""Shine-through by depth: which pore pixels does each method get right?

On a synthetic package every frame has truth, so for each pore pixel of a
frame we can say how far down the beam the first solid is: follow the same
pixel into frames z+1 .. z+8, shifted by (drift + parallax) * k rows (the
whole picture drifts, and buried material sits a further parallax * k
along +y). Depth 1 = solid right under the pore -- the strongest
shine-through, a lit floor that looks like solid. 'none' = no solid within
8 slices, a deep hole.

For each method: share of pore pixels called pore (recall) per depth bin,
and share of solid pixels called solid. Methods: the threshold rule (start
masks), and the checkpoints given.

    python shine_depth_eval.py --keys zwin_random_v3_s0 ...   (torch-env)
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
import eval_shine as ev
import adabn_eval as ab

HERE = Path(__file__).resolve().parent
from paths import OUT as DL
BINS = [("1", 1, 1), ("2-3", 2, 3), ("4-8", 4, 8), ("none<=8", 9, 9)]


def depth_map(truth, z, shift_per_slice, kmax=8):
    t0 = truth[z]
    pore = t0 == 0
    d = np.full(t0.shape, kmax + 1, np.int16); d[t0 == 255] = 0
    open_ = pore.copy()
    for k in range(1, kmax + 1):
        if z + k >= len(truth):
            break
        tk = truth[z + k]
        s = int(round(shift_per_slice * k))
        sh = np.full_like(tk, 128)
        if s >= 0:
            sh[:tk.shape[0] - s] = tk[s:]
        else:
            sh[-s:] = tk[:tk.shape[0] + s]
        hit = open_ & (sh == 255)
        d[hit] = k; open_ &= ~hit
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", default=str(HERE / "shine_depth_eval.json"))
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = {k: ev.load(HERE / "checkpoints" / f"{k}.pt", dev) for k in args.keys}
    res = {}
    for name in ab.PKGS:
        pkg = DL / f"team_review_synth_{name}"
        meta = json.loads((pkg / "stack.json").read_text())
        tj = json.loads((pkg / "truth" / "truth.json").read_text())
        shift = float(tj["optics"]["drift"]) + float(tj["optics"]["parallax"])
        tif = pkg / meta["stack_file"]
        zl = ev.package_frames(16)
        truth = {z: np.array(Image.open(pkg / "truth" / "masks" / f"slice_{z:04d}.png"))
                 for z in sorted({z + k for z in zl for k in range(0, 9)}) if z < 118}
        tl = [truth[z] for z in range(min(truth), max(truth) + 1)]
        off = min(truth)
        depths = [depth_map(tl, z - off, shift) for z in zl]
        keep = [ndi.binary_erosion(truth[z] != 128, iterations=ev.MARGIN) for z in zl]
        preds = {"rule (start masks)": [np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png")) == 0 for z in zl]}
        drift = ab.band_drift(tif, truth[zl[0]] != 128)
        X = ab.stack_windows(tif, zl, drift, [truth[z] != 128 for z in zl])
        for k, (m, ck) in models.items():
            preds[k] = list(ev.predict(m, ck, X, dev) > 0.5)
        r = {}
        for meth, P in preds.items():
            row = {}
            for lab, lo, hi in BINS:
                sel = [(d >= lo) & (d <= hi) & kp for d, kp in zip(depths, keep)]
                n = sum(s.sum() for s in sel)
                row[f"recall_{lab}"] = float(sum((p & s).sum() for p, s in zip(P, sel)) / max(n, 1))
                row[f"share_{lab}"] = float(n)
            sol = [(d == 0) & kp for d, kp in zip(depths, keep)]
            row["solid_correct"] = float(sum((~p & s).sum() for p, s in zip(P, sol)) / max(sum(s.sum() for s in sol), 1))
            r[meth] = row
        tot = sum(r["rule (start masks)"][f"share_{b[0]}"] for b in BINS)
        for meth in r:
            for b in BINS:
                r[meth][f"share_{b[0]}"] = r["rule (start masks)"][f"share_{b[0]}"] / tot
        res[name] = r
        print(name, {b[0]: round(r["rule (start masks)"][f"share_{b[0]}"], 3) for b in BINS}, flush=True)
        for meth, row in r.items():
            print(f"   {meth:<22} " + " ".join(f"d{b[0]}={row['recall_' + b[0]]:.2f}" for b in BINS) + f"  solid={row['solid_correct']:.2f}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
