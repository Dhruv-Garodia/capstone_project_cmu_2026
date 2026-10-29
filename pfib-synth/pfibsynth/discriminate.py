#!/usr/bin/env python3
"""Can a classifier tell the synthetic images from the real ones?

The synth-to-real work used this test and credited it with throwing out
changes that "only looked better". The useful comparison is not the raw
score but the score against a control: a classifier trying to separate real
patches from *other real patches* also scores above chance, because
different parts of a stack genuinely differ. Synthetic data is as good as
the data allows when it scores no better than that control.

Features are deliberately simple and interpretable, so that when the
classifier does win you can see on which feature:

  * grey histogram over 16 bins
  * gradient magnitude percentiles (edge sharpness)
  * residual after a 3x3 median (grain)
  * radial power spectrum in 8 bands (how energy is spread over scales)

Logistic regression, trained with plain gradient descent, scored by AUC on
held-out patches.

    python discriminate.py --style comp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from paths import REAL_DIR as LAB
from paths import LABELS_DIR as DL
REAL = {"comp": (LAB / "comp_full.tif", DL / "team_review"),
        "uncomp": (LAB / "uncomp_full.tif", DL / "team_review_uncomp"),
        "pristine": (LAB / "pristine_full.tif", DL / "team_review_pristine")}
OUTSIDE = 128
P = 96          # patch size


def features(p: np.ndarray) -> np.ndarray:
    a = p.astype(np.float32)
    f = []
    h, _ = np.histogram(a, bins=16, range=(0, 256), density=True)
    f += list(h * 100)
    gy, gx = np.gradient(a)
    g = np.hypot(gy, gx)
    f += [float(np.percentile(g, q)) for q in (25, 50, 75, 90, 99)]
    res = a - ndi.median_filter(a, size=3)
    f += [float(res.std()), float(np.percentile(np.abs(res), 90))]
    F = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
    cy, cx = np.array(F.shape) // 2
    yy, xx = np.ogrid[:F.shape[0], :F.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    rm = r.max() / 2
    tot = F.sum() + 1e-9
    for i in range(8):
        band = (r >= i * rm / 8) & (r < (i + 1) * rm / 8)
        f.append(float(F[band].sum() / tot))
    f += [float(a.mean()), float(a.std())]
    return np.array(f, np.float32)


def patches_from(img, mask, n, rng, exclude_outside=True):
    out = []
    h, w = img.shape
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        y = rng.integers(0, h - P)
        x = rng.integers(0, w - P)
        m = mask[y:y + P, x:x + P]
        if exclude_outside and (m == OUTSIDE).any():
            continue
        out.append(features(img[y:y + P, x:x + P]))
    return out


def auc(y, s):
    o = np.argsort(s)
    y = np.asarray(y)[o]
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def logistic_auc(X, y, rng, iters=900, lr=0.12):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    idx = rng.permutation(len(X))
    X, y = X[idx], np.asarray(y)[idx]
    cut = int(0.65 * len(X))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        z = Xtr @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        gw = Xtr.T @ (p - ytr) / len(Xtr) + 1e-3 * w
        gb = float((p - ytr).mean())
        w -= lr * gw; b -= lr * gb
    return auc(yte, Xte @ w + b), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="comp", choices=list(REAL))
    ap.add_argument("--synth", default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tif, pkg = REAL[args.style]
    frames = sorted(int(p.stem[6:]) for p in (pkg / "start_masks").glob("slice_*.png"))

    # two disjoint halves of the real stack -> the control comparison
    lo = frames[: len(frames) // 2][:: max(1, len(frames) // 16)][:8]
    hi = frames[len(frames) // 2:][:: max(1, len(frames) // 16)][:8]

    def real_patches(zs, n_each):
        out = []
        for z in zs:
            m = np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png"))
            img = np.asarray(tifffile.imread(tif, key=z))
            out += patches_from(img, m, n_each, rng)
        return out

    A = real_patches(lo, args.n // len(lo) + 1)
    B = real_patches(hi, args.n // len(hi) + 1)

    from paths import OUT
    sp = Path(args.synth) if args.synth else OUT / f"syn_{args.style}"
    meta = json.loads((sp / "stack.json").read_text())
    S = []
    nsl = len(list((sp / "truth" / "masks").glob("*.png")))
    zs = list(range(0, nsl, max(1, nsl // 8)))[:8]
    for z in zs:
        img = np.asarray(tifffile.imread(sp / meta["stack_file"], key=z))
        m = np.array(Image.open(sp / "truth" / "masks" / f"slice_{z:04d}.png"))
        S += patches_from(img, m, args.n // len(zs) + 1, rng)

    n = min(len(A), len(B), len(S))
    A, B, S = np.array(A[:n]), np.array(B[:n]), np.array(S[:n])
    print(f"style {args.style}: {n} patches of {P}x{P} per group\n")

    ctrl, _ = logistic_auc(np.vstack([A, B]), [0] * n + [1] * n, rng)
    test, w = logistic_auc(np.vstack([np.vstack([A, B]), S]),
                           [0] * (2 * n) + [1] * n, rng)

    print(f"  real-vs-real  (control) AUC {ctrl:.3f}")
    print(f"  real-vs-synth           AUC {test:.3f}")
    print(f"\n  0.5 = indistinguishable, 1.0 = perfectly separable.")
    print(f"  The control is the floor worth aiming at, not 0.5: two halves")
    print(f"  of the same real stack already differ.")
    if test <= ctrl + 0.03:
        print("\n  -> synthetic is as hard to pick out as another part of the real stack")
    else:
        names = (["hist%02d" % i for i in range(16)]
                 + ["grad25", "grad50", "grad75", "grad90", "grad99"]
                 + ["grain_std", "grain_p90"]
                 + ["fft%d" % i for i in range(8)] + ["mean", "std"])
        order = np.argsort(-np.abs(w))[:6]
        print("\n  -> still separable. Features carrying the most weight:")
        for i in order:
            print(f"       {names[i]:<12} weight {w[i]:+.2f}")


if __name__ == "__main__":
    main()
