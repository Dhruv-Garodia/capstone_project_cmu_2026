#!/usr/bin/env python3
"""v3 see-through rendering: helpers and a fast realism loop.

depth_curve(style): mean pore grey for d = 1..depth (+ 'nothing hit'),
built from the measured real curve (optics.json targets: baseline + excess
for d = 1..8). Beyond 8 the real data says only "none within 8"; an
exponential fitted to d = 4..8 is extended to d = depth, and 'nothing hit'
is set to the v2 g_empty -- so the deepest holes can be darker than the d~8
ones, as real deep holes are (p10 of real ROIs is 71-121).

    python see_through.py --style comp --save crop.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import materials
from microscope import Optics, render_surface, add_noise

HERE = Path(__file__).resolve().parent


def depth_curve(style, depth=24, optics="optics_v2.json"):
    t = json.loads((HERE / "optics.json").read_text())[style]["targets"]
    g_empty = json.loads((HERE / optics).read_text())[style]["optics"]["g_empty"]
    base = float(t["baseline"])
    ex = {int(k): float(v) for k, v in t["excess"].items()}
    c = np.zeros(depth + 2, np.float32)
    for k in range(1, 9):
        c[k] = base + ex[k]
    ks = np.arange(4, 9); y = np.log(np.maximum([c[k] - g_empty for k in ks], 1.0))
    slope, icpt = np.polyfit(ks, y, 1)
    for k in range(9, depth + 1):
        c[k] = g_empty + np.exp(icpt + slope * k)
    c[depth + 1] = g_empty
    return c


def material_kw(cal, params):
    """(family, kwargs) for materials.build from a fitted see-through entry."""
    mk = dict(params["mat"])
    if params["family"] == "catalyst":
        m = dict(cal["material"]); m.pop("family"); m.pop("porosity", None)
        sc = mk.pop("corr_scale", 1.0)
        m["corr"] = tuple(c * sc for c in m["corr"]); m.update(mk)
        return "catalyst", m
    if "cluster_sigma" in mk:
        mk["cluster_sigma"] = tuple(mk["cluster_sigma"])
    return params["family"], mk


def renderer_kw(params, opt):
    """kwargs for render_surface from a fitted entry (opt.depth is set by the caller)."""
    return dict(smooth=params["smooth"], secant=params["secant"], cos_floor=params["cos_floor"],
                hit_albedo=True, defocus=params["defocus"],
                hole_floor=params["hole_floor_frac"] * opt.g_empty, hole_len=params["hole_len"],
                face_smooth=params["face_smooth"], escape=params["escape"], sigma0=params["sigma0"],
                shade_floor=params.get("shade_floor", 0.0))


def crop(style, family="v2", seed=1, n=12, shape=(320, 640), optics="optics_v2.json", **kw):
    cal = json.loads((HERE / optics).read_text())[style]
    opt = Optics(**cal["optics"])
    if "depth" in kw:
        opt.depth = int(kw.pop("depth"))
    m = dict(cal["material"]); fam = m.pop("family"); m.pop("porosity")
    m["corr"] = tuple(m["corr"]); m.update(kw.pop("mat", {}))
    vol = (n + opt.depth + 2, shape[0] + 16, shape[1])
    solid, tex = materials.build(fam if family == "v2" else family, vol, cal["material"]["porosity"],
                                 seed=seed, with_texture=True, **m)
    if tex is None:
        tex = materials.se_texture(solid, np.random.default_rng(seed), 0.15, 1.0)
    alb = np.where(solid, tex, 0).astype(np.float32)
    t = json.loads((HERE / "optics.json").read_text())[style]["targets"]
    rng = np.random.default_rng(seed)
    grey, depth = render_surface(solid, alb, opt, list(range(n)), depth_curve(style, opt.depth, optics), rng,
                                 face_level=kw.pop("face_level", float(t["solid_mean"])), **kw)
    img = add_noise(grey, opt, rng)
    return [g[8:-8] for g in img], solid[:n, 8:-8]


def main():
    import lab
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="comp")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    imgs, _ = crop(a.style, **json.loads(a.kw))
    rois = [np.ones(i.shape, bool) for i in imgs]
    real = lab.real_set(a.style)
    lv = lab.LAB[a.style]["level"]
    mr, mf = lab.metrics(*real, lv), lab.metrics(imgs, rois, lv)
    for k in mr:
        print(f"  {k:<15} real {mr[k]:7.3f}  fake {mf[k]:7.3f}")
    au, top = lab.patch_auc(real, (imgs, rois))
    print(f"  patch AUC {au:.3f}; gaps {top}")
    if a.save:
        Image.fromarray(np.concatenate([imgs[0], imgs[len(imgs) // 2]], 1)).save(a.save)


if __name__ == "__main__":
    main()
