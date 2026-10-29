#!/usr/bin/env python3
"""Evaluate trained shine-through models.

Three test beds, each answering a different question.

  synthetic   calibrated comp / uncomp / pristine physics (never trained on
              by the randomised configs), six material families each, three
              of them never seen in training. Recall is broken out by how far
              below the first buried solid sits: those are the pixels
              shine-through makes look solid.
  packages    the six full-size review packages: real ROI outlines and real
              context, drift estimated from the images, scored against truth
              with a 10 px margin inside the ROI edge excluded (the
              synthetic/real seam is a cue a model could use). The comparison
              is against the truth-blind start masks scored the same way.
  real        the three real stacks. There is no truth: report porosity and
              disagreement with the team's labels, and export the z-window
              predictions as mask PNGs so the disagreements can be judged by
              eye in napari.

    python eval_shine.py --cache <dir> --ckpts checkpoints/*.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from scipy import ndimage as ndi

import shine_data as sd
from train_shine import make_model

HERE = Path(__file__).resolve().parent
from paths import OUT as DL
from paths import REAL_DIR as LAB
REAL = {"comp": (LAB / "comp_full.tif", DL / "team_review", 0.417),
        "uncomp": (LAB / "uncomp_full.tif", DL / "team_review_uncomp", 0.558),
        "pristine": (LAB / "pristine_full.tif", DL / "team_review_pristine", 0.515)}
PACKAGES = ["comp_like", "uncomp_like", "pristine_like", "mudcrack", "fibre_felt", "faceted"]
BINS = {"d1": (1, 1), "d2-3": (2, 3), "d4-8": (4, 8), "none<=8": (9, 9)}
MARGIN = 10


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = make_model(ck["window"], ck["aux"])
    m.load_state_dict(ck["state"]); m.to(dev).eval()
    return m, ck


@torch.no_grad()
def predict(model, ck, X, dev, bs=None):
    """X: (n, 9, h, w) normalised windows -> pore probability (n, h, w)."""
    n, _, h, w = X.shape
    bs = bs or (1 if h * w > 1_000_000 else 4)
    H, W = (h + 15) // 16 * 16, (w + 15) // 16 * 16
    out = np.empty((n, h, w), np.float32)
    for i in range(0, n, bs):
        xb = X[i:i + bs].astype(np.float32)
        if not ck["window"]:
            xb = xb[:, 3:4]
        xb = np.pad(xb, ((0, 0), (0, 0), (0, H - h), (0, W - w)), mode="reflect")
        p = torch.sigmoid(model(torch.from_numpy(xb).to(dev))[:, 0]).cpu().numpy()
        out[i:i + bs] = p[:, :h, :w]
    return out


def scores(pred, pore, depth=None, keep=None):
    keep = np.ones(pore.shape, bool) if keep is None else keep
    p, y = pred[keep], pore[keep]
    pr = (p & y).sum() / max(y.sum(), 1); sr = (~p & ~y).sum() / max((~y).sum(), 1)
    r = {"bal_acc": float(0.5 * (pr + sr)), "por_err": float(p.mean() - y.mean()),
         "pore_iou": float((p & y).sum() / max((p | y).sum(), 1))}
    if depth is not None:
        d = depth[keep]
        for name, (lo, hi) in BINS.items():
            sel = y & (d >= lo) & (d <= hi)
            r[f"recall_{name}"] = float((p & sel).sum() / max(sel.sum(), 1))
    return r


def eval_synthetic(models, cache, dev):
    res = {}
    for folder in sorted(Path(cache).glob("test_*")):
        _, regime, fam = folder.name.split("_", 2)
        d = np.load(folder / "vol_000.npz")
        X, pore, depth = d["X"], d["pore"], d["depth"]
        for key, (m, ck) in models.items():
            pred = predict(m, ck, X, dev) > 0.5
            res.setdefault(key, {})[f"{regime}:{fam}"] = scores(pred, pore, depth)
        print(f"  synthetic {regime:<9} {fam:<11} done", flush=True)
    return res


def package_frames(n_frames, total=118):
    return list(np.linspace(3, total - 6, n_frames).astype(int))


def eval_packages(models, dev, n_frames=16):
    res = {}
    for name in PACKAGES:
        pkg = DL / f"team_review_synth_{name}"
        meta = json.loads((pkg / "stack.json").read_text())
        tif = pkg / meta["stack_file"]
        zl = package_frames(n_frames)
        need = sorted({z + k for z in zl for k in sd.WIN})
        stack = {z: np.asarray(tifffile.imread(tif, key=z)) for z in need}
        truth = {z: np.array(Image.open(pkg / "truth" / "masks" / f"slice_{z:04d}.png")) for z in zl}
        start = {z: np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png")) for z in zl}
        roi0 = truth[zl[0]] != 128
        rr = np.where(roi0.any(1))[0]
        band = (rr[0] + (rr[-1] - rr[0]) // 4, rr[0] + 3 * (rr[-1] - rr[0]) // 4)
        run = np.stack([np.asarray(tifffile.imread(tif, key=40 + i)) for i in range(20)])
        drift = sd.estimate_drift(run, rows=band)
        X = np.empty((len(zl), 9) + roi0.shape, np.float16)
        for n, z in enumerate(zl):
            img = np.stack([stack[z + k] for k in sd.WIN])
            roi = truth[z] != 128
            X[n] = sd.windows(img, [3], drift, norm_mask=roi)[0]
        keep = np.stack([ndi.binary_erosion(truth[z] != 128, iterations=MARGIN) for z in zl])
        pore = np.stack([truth[z] == 0 for z in zl])
        st = np.stack([start[z] == 0 for z in zl])
        res.setdefault("start_masks", {})[name] = scores(st, pore, keep=keep)
        for key, (m, ck) in models.items():
            pred = predict(m, ck, X, dev) > 0.5
            res.setdefault(key, {})[name] = scores(pred, pore, keep=keep)
        res.setdefault("_drift", {})[name] = drift
        print(f"  package {name:<13} drift est {drift:+.2f}  done", flush=True)
    return res


def hand_labels(pkg):
    out = {}
    for d in sorted((pkg / "annotators").glob("*")):
        if not d.is_dir() or any(t in d.name for t in ("_before", "_boundary", "_restore")):
            continue
        for f in d.glob("slice_*.png"):
            out.setdefault(int(f.stem[6:]), f)
    return out


def eval_real(models, dev, export_key, export_dir, n_frames=12):
    res = {}
    for name, (tif, pkg, published) in REAL.items():
        hl = hand_labels(pkg)
        cand = [z for z in sorted(hl) if z >= 3 and (pkg / "start_masks" / f"slice_{z:04d}.png").exists()]
        zl = cand[:: max(1, len(cand) // n_frames)][:n_frames]
        m0 = np.array(Image.open(pkg / "start_masks" / "slice_0040.png")) != 128
        rr = np.where(m0.any(1))[0]
        band = (rr[0] + (rr[-1] - rr[0]) // 4, rr[0] + 3 * (rr[-1] - rr[0]) // 4)
        run = np.stack([np.asarray(tifffile.imread(tif, key=40 + i)) for i in range(20)])
        drift = sd.estimate_drift(run, rows=band)
        need = sorted({z + k for z in zl for k in sd.WIN})
        stack = {z: np.asarray(tifffile.imread(tif, key=z)) for z in need}
        starts = {z: np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png")) for z in zl}
        hands = {z: np.array(Image.open(hl[z])) for z in zl}
        X = np.empty((len(zl), 9) + m0.shape, np.float16)
        for n, z in enumerate(zl):
            img = np.stack([stack[z + k] for k in sd.WIN])
            X[n] = sd.windows(img, [3], drift, norm_mask=starts[z] != 128)[0]
        keep = np.stack([ndi.binary_erosion(starts[z] != 128, iterations=MARGIN) for z in zl])
        lab = np.stack([hands[z] == 0 for z in zl])
        r = {"drift_est": drift, "frames": [int(z) for z in zl], "published": published,
             "labels_porosity": float(lab[keep].mean()),
             "start_porosity": float(np.stack([starts[z] == 0 for z in zl])[keep].mean())}
        for key, (m, ck) in models.items():
            prob = predict(m, ck, X, dev)
            pred = prob > 0.5
            r[key] = {"porosity": float(pred[keep].mean()),
                      "disagree_with_labels": float((pred[keep] != lab[keep]).mean()),
                      "model_pore_label_solid": float((pred & ~lab)[keep].mean()),
                      "model_solid_label_pore": float((~pred & lab)[keep].mean())}
            if key == export_key:
                out = Path(export_dir) / name
                out.mkdir(parents=True, exist_ok=True)
                for n, z in enumerate(zl):
                    roi = starts[z] != 128
                    mk = np.where(roi, np.where(pred[n], 0, 255), 128).astype(np.uint8)
                    Image.fromarray(mk, mode="L").save(out / f"slice_{z:04d}.png")
                    # disagreement overlay: red = model pore / label solid,
                    # blue = model solid / label pore
                    g = stack[z].astype(np.float32)
                    rgb = np.stack([g, g, g], -1)
                    a = roi & pred[n] & ~lab[n]; b = roi & ~pred[n] & lab[n]
                    rgb[a] = 0.45 * rgb[a] + np.array([140, 0, 0])
                    rgb[b] = 0.45 * rgb[b] + np.array([0, 40, 140])
                    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(
                        out / f"disagree_{z:04d}.png")
        res[name] = r
        print(f"  real {name:<9} drift est {drift:+.2f}  done", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--export-key", default="zwin_random_s0")
    ap.add_argument("--export-dir", default=str(HERE / "model_masks"))
    ap.add_argument("--out", default=str(HERE / "eval_shine.json"))
    args = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = {Path(p).stem: load(p, dev) for p in args.ckpts}
    print("models:", ", ".join(models), flush=True)
    res = {"synthetic": eval_synthetic(models, args.cache, dev),
           "packages": eval_packages(models, dev),
           "real": eval_real(models, dev, args.export_key, args.export_dir)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
