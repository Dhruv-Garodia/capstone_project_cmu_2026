#!/usr/bin/env python3
"""Fast realism loop: small synthetic volumes against the real ROI.

Building a full package takes minutes; a 16-slice 320 x 640 crop takes a few
seconds, which is what trying material ideas needs. Real and synthetic are
measured the same way -- pore = grey at or below the real-tuned labeller
level, no truth used on either side -- so the numbers compare like for like.

    python lab.py --style comp --family catalyst
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

import materials
from microscope import Optics, render, add_noise
from discriminate import REAL, features, logistic_auc, P

HERE = Path(__file__).resolve().parent
LAB = json.loads((HERE / "real_labeller.json").read_text())


def metrics(imgs, rois, level):
    out = {k: [] for k in ("solid_local_sd", "detail", "mid_grey", "ac_y", "ac_x",
                           "far_pore", "p10", "p50", "p90")}
    for a, roi in zip(imgs, rois):
        a = a.astype(np.float32)
        roi = ndi.binary_erosion(roi, iterations=12)
        if roi.sum() < 4000:
            continue
        sm = ndi.gaussian_filter(a, 1.0)
        pore = (sm <= level) & roi
        sol = ndi.binary_erosion((sm > level) & roi, iterations=3)
        mu = ndi.uniform_filter(a, 5)
        sd = np.sqrt(np.maximum(ndi.uniform_filter(a * a, 5) - mu * mu, 0))
        out["solid_local_sd"].append(np.median(sd[sol]))
        out["detail"].append((ndi.gaussian_filter(a, 1.5) - ndi.gaussian_filter(a, 4))[roi].std())
        g = ndi.gaussian_filter(a, 1.5)[roi]
        p10, p90 = np.percentile(g, [10, 90])
        out["mid_grey"].append(((g > p10 + .33 * (p90 - p10)) & (g < p90 - .33 * (p90 - p10))).mean())
        ys, xs = np.where(roi)
        sub = a[ys.min():ys.max(), xs.min():xs.max()]
        rr = roi[ys.min():ys.max(), xs.min():xs.max()]
        sub = np.where(rr, sub - sub[rr].mean(), 0)
        for ax, key in ((0, "ac_y"), (1, "ac_x")):
            A = np.take(sub, range(0, sub.shape[ax] - 6), ax)
            B = np.take(sub, range(6, sub.shape[ax]), ax)
            out[key].append((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))
        dt = ndi.distance_transform_edt(pore | ~roi)
        out["far_pore"].append((dt[pore] > 6).mean())
        v = a[roi]
        for q in (10, 50, 90):
            out[f"p{q}"].append(np.percentile(v, q))
    return {k: float(np.mean(v)) for k, v in out.items()}


def real_set(style, n=10):
    tif, pkg = REAL[style]
    zs = sorted(int(p.stem[6:]) for p in (pkg / "start_masks").glob("slice_*.png"))
    zs = zs[:: max(1, len(zs) // n)][:n]
    imgs = [np.asarray(tifffile.imread(tif, key=z)) for z in zs]
    rois = [np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png")) != 128 for z in zs]
    return imgs, rois


def fake_set(style, family, seed=1, n=16, shape=(320, 640), optics="optics.json", **kw):
    cal = json.loads((HERE / optics).read_text())[style]
    if family == "v2":   # the recipe stored with optics_v2.json
        m = dict(cal["material"]); family = m.pop("family")
        m["corr"] = tuple(m["corr"]); kw = {**m, **kw}
    opt = Optics(**cal["optics"])
    sh = cal["standin"]
    vol = (n + opt.depth + 1, shape[0] + 16, shape[1])
    kw.setdefault("corr", (sh["corr_z"], sh["corr_xy"], sh["corr_xy"]))
    solid, tex = materials.build(family, vol, kw.pop("porosity", sh["porosity"]), seed=seed,
                                 with_texture=True, **kw)
    rng = np.random.default_rng(seed)
    bright = materials.bright_phase(solid, opt.bright_frac, seed) if opt.bright_frac else None
    grey = add_noise(render(solid, opt, list(range(n)), rng, texture=tex, bright=bright), opt, rng)
    imgs = [g[8:-8] for g in grey]
    return imgs, [np.ones(i.shape, bool) for i in imgs]


def patch_auc(real, fake, seed=0, k=240):
    rng = np.random.default_rng(seed)

    def pats(imgs, rois, n):
        f = []
        for _ in range(n * 40):
            if len(f) >= n:
                break
            i = rng.integers(len(imgs)); a, r = imgs[i], rois[i]
            y = rng.integers(0, a.shape[0] - P); x = rng.integers(0, a.shape[1] - P)
            if r[y:y + P, x:x + P].all():
                f.append(features(a[y:y + P, x:x + P]))
        return f
    R, S = pats(*real, k), pats(*fake, k)
    m = min(len(R), len(S))
    R, S = np.array(R[:m]), np.array(S[:m])
    a, _ = logistic_auc(np.vstack([R, S]), [0] * m + [1] * m, rng)
    # which features differ most, in units of the real spread
    d = (S.mean(0) - R.mean(0)) / (R.std(0) + 1e-6)
    names = (["hist%02d" % i for i in range(16)]
             + ["grad25", "grad50", "grad75", "grad90", "grad99", "grain_std", "grain_p90"]
             + ["fft%d" % i for i in range(8)] + ["mean", "std"])
    top = [(names[i], round(float(d[i]), 2)) for i in np.argsort(-np.abs(d))[:6]]
    return a, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="comp")
    ap.add_argument("--family", default="agglomerate")
    ap.add_argument("--kw", default="{}", help="json kwargs for the family")
    ap.add_argument("--save", default=None)
    ap.add_argument("--optics", default="optics.json")
    args = ap.parse_args()
    real = real_set(args.style)
    fake = fake_set(args.style, args.family, optics=args.optics, **json.loads(args.kw))
    lv = LAB[args.style]["level"]
    mr, mf = metrics(*real, lv), metrics(*fake, lv)
    print(f"{args.style} {args.family} {args.kw}")
    for k in mr:
        print(f"  {k:<15} real {mr[k]:7.3f}  fake {mf[k]:7.3f}")
    a, top = patch_auc(real, fake)
    print(f"  patch AUC real-vs-fake {a:.3f}; biggest gaps (fake-real, in real sd): {top}")
    if args.save:
        Image.fromarray(np.concatenate([fake[0][0], fake[0][len(fake[0]) // 2]], 1)).save(args.save)


if __name__ == "__main__":
    main()
