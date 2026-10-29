#!/usr/bin/env python3
"""Test-time AdaBN (Dhruv's finding) on the synthetic-trained models.

For each checkpoint and each target stack, BatchNorm running statistics are
re-estimated on that stack's frames without using any label, then the model
is scored again. Everything else -- per-stack median/IQR normalisation,
registration, threshold 0.5 -- is unchanged, so the difference isolates
AdaBN.

  v2 packages   scored against truth on 16 frames; BN is re-estimated on
                other frames of the same package (not the scored ones)
  real stacks   scored against the team labels on the hand-labelled frames;
                BN is re-estimated on unlabelled slices >= 125. Labels are
                rule masks, not truth, so "closer to labels" is all this says.

    python adabn_eval.py --keys zwin_random_s0 ... --out adabn_eval.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi

import eval_shine as ev
import shine_data as sd

HERE = Path(__file__).resolve().parent
from paths import OUT as DL
PKGS = ["comp_like", "uncomp_like", "pristine_like", "mudcrack", "fibre_mat", "fibre_coated", "faceted"]


def adapt(model, ck, Xu, dev, crop=384, iters=60, bs=4, seed=0):
    """Copy of the model with BN statistics re-estimated on windows Xu."""
    m = copy.deepcopy(model)
    bns = [mod for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
    for b in bns:
        b.reset_running_stats(); b.momentum = None      # cumulative average
    m.train()
    rng = np.random.default_rng(seed)
    n, _, h, w = Xu.shape
    c = min(crop, h // 16 * 16, w // 16 * 16)
    with torch.no_grad():
        for _ in range(iters):
            idx = rng.integers(0, n, bs)
            ys = rng.integers(0, h - c + 1, bs); xs = rng.integers(0, w - c + 1, bs)
            xb = np.stack([Xu[i, :, y:y + c, x:x + c] for i, y, x in zip(idx, ys, xs)]).astype(np.float32)
            xb = np.clip(np.nan_to_num(xb), -20, 20)
            if not ck["window"]:
                xb = xb[:, 3:4]
            m(torch.from_numpy(xb).to(dev))
    m.eval()
    return m


def stack_windows(tif, zl, drift, masks):
    need = sorted({z + k for z in zl for k in sd.WIN})
    st = {z: np.asarray(tifffile.imread(tif, key=z)) for z in need}
    X = np.empty((len(zl), 9) + st[zl[0]].shape, np.float16)
    for n, z in enumerate(zl):
        X[n] = sd.windows(np.stack([st[z + k] for k in sd.WIN]), [3], drift, norm_mask=masks[n])[0]
    return X


def band_drift(tif, roi):
    rr = np.where(roi.any(1))[0]
    band = (rr[0] + (rr[-1] - rr[0]) // 4, rr[0] + 3 * (rr[-1] - rr[0]) // 4)
    run = np.stack([np.asarray(tifffile.imread(tif, key=40 + i)) for i in range(20)])
    return sd.estimate_drift(run, rows=band)


def packages(models, dev):
    res = {}
    for name in PKGS:
        pkg = DL / f"team_review_synth_{name}"
        tif = pkg / json.loads((pkg / "stack.json").read_text())["stack_file"]
        zl = ev.package_frames(16)
        zu = [z for z in range(4, 112, 3) if all(abs(z - q) > 1 for q in zl)][:16]
        truth = [np.array(Image.open(pkg / "truth" / "masks" / f"slice_{z:04d}.png")) for z in zl]
        # the ROI is only used to pick which pixels set the grey normalisation;
        # for the unscored frames use the start masks' ROI, as on real data
        su = [np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png")) != 128 for z in zu]
        drift = band_drift(tif, truth[0] != 128)
        X = stack_windows(tif, zl, drift, [t != 128 for t in truth])
        Xu = stack_windows(tif, zu, drift, su)
        keep = np.stack([ndi.binary_erosion(t != 128, iterations=ev.MARGIN) for t in truth])
        pore = np.stack([t == 0 for t in truth])
        start = np.stack([np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png")) == 0 for z in zl])
        res.setdefault("start_masks", {})[name] = ev.scores(start, pore, keep=keep)
        for key, (m, ck) in models.items():
            res.setdefault(key, {})[name] = ev.scores(ev.predict(m, ck, X, dev) > 0.5, pore, keep=keep)
            ma = adapt(m, ck, Xu, dev)
            res.setdefault(key + "+adabn", {})[name] = ev.scores(ev.predict(ma, ck, X, dev) > 0.5, pore, keep=keep)
            del ma
        print(f"  package {name} done", flush=True)
    return res


def real(models, dev):
    res = {}
    for name, (tif, pkg, published) in ev.REAL.items():
        hl = ev.hand_labels(pkg)
        cand = [z for z in sorted(hl) if z >= 3 and (pkg / "start_masks" / f"slice_{z:04d}.png").exists()]
        zl = cand[:: max(1, len(cand) // 12)][:12]
        starts = [np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png")) for z in zl]
        labs = [np.array(Image.open(hl[z])) for z in zl]
        drift = band_drift(tif, np.array(Image.open(pkg / "start_masks" / "slice_0040.png")) != 128)
        npages = len(tifffile.TiffFile(tif).pages)
        zu = list(range(125, min(npages - 6, 125 + 4 * 16), 4))
        # unlabelled slices have no ROI; normalise over the band the labelled
        # frames' ROI covers, the closest label-free stand-in
        band = np.zeros_like(starts[0], bool)
        rr = np.where((starts[-1] != 128).any(1))[0]; band[rr[0]:rr[-1] + 1] = True
        X = stack_windows(tif, zl, drift, [s != 128 for s in starts])
        Xu = stack_windows(tif, zu, drift, [band] * len(zu))
        keep = np.stack([ndi.binary_erosion(s != 128, iterations=ev.MARGIN) for s in starts])
        lab = np.stack([l == 0 for l in labs])
        r = {"published": published, "labels_porosity": float(lab[keep].mean())}
        for key, (m, ck) in models.items():
            for tag, mm in ((key, m), (key + "+adabn", adapt(m, ck, Xu, dev))):
                pred = ev.predict(mm, ck, X, dev) > 0.5
                r[tag] = {"porosity": float(pred[keep].mean()),
                          "disagree_with_labels": float((pred[keep] != lab[keep]).mean())}
        res[name] = r
        print(f"  real {name} done", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", default=str(HERE / "adabn_eval.json"))
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = {k: ev.load(HERE / "checkpoints" / f"{k}.pt", dev) for k in args.keys}
    out = {"packages": packages(models, dev), "real": real(models, dev)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    P = out["packages"]
    print(f"\n{'':<26}" + "".join(f"{n:>14}" for n in PKGS))
    for key in P:
        print(f"{key:<26}" + "".join(f"{P[key][n]['por_err']:>+8.3f} {P[key][n]['pore_iou']:.2f}" for n in PKGS))
    print("(porosity error vs truth, pore IoU)\n")
    for name, r in out["real"].items():
        print(f"{name}: published {r['published']:.3f} labels {r['labels_porosity']:.3f}")
        for key in r:
            if isinstance(r[key], dict):
                print(f"   {key:<26} porosity {r[key]['porosity']:.3f}  disagree {r[key]['disagree_with_labels']:.3f}")


if __name__ == "__main__":
    main()
