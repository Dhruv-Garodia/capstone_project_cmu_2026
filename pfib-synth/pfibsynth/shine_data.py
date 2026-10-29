#!/usr/bin/env python3
"""Training and test volumes for shine-through-aware segmentation.

Every volume is stored the way a model would meet a real stack:

  * in the IMAGE frame -- content drifts +y between slices, by the volume's
    own drift -- with truth and the depth-of-signal label carried into the
    same frame, so labels sit exactly under the pixels they describe;
  * with the drift ESTIMATED from the images (label-free, as on real data)
    and used only to register neighbours to a target slice. The target slice
    itself is never moved, so an estimation error can blur the context but
    can never misalign a label. (The first version registered the whole stack
    with the estimate and read labels in the sample frame, which drifted
    labels 3-5 px off their pixels by slice 30.)

Physics for training is drawn from generic ranges, not from the three
calibrated regimes, so the calibrated comp / uncomp / pristine optics are
genuinely unseen at test time. The ranges were chosen to span the measured
values with margin; that is disclosed, not hidden.

Labels per pixel:
  pore      1 = pore, 0 = solid
  depth     0 = solid, 1..8 = slices down to the first buried solid along
            the beam path, 9 = none within 8. Only synthetic data can provide
            this; it is the auxiliary target.

    python shine_data.py --out ~/.../shine_cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import materials
from microscope import Optics, render, add_noise, depth_to_solid

HERE = Path(__file__).resolve().parent
SHAPE = (64, 192, 384)
WIN = list(range(-3, 6))            # z-3 .. z+5, target at index 3
ZLO, ZHI = 3, SHAPE[0] - 26         # need z-3 and 24 slices of shine-through
TRAIN_FAMILIES = ["agglomerate", "compressed", "spinodal"]
HELDOUT_FAMILIES = ["mudcrack", "faceted", "fiber_felt"]


def calibrated(name):
    cal = json.loads((HERE / "optics.json").read_text())[name]
    return Optics(**cal["optics"]), dict(cal["standin"])


def generic_physics(rng):
    """Randomised optics from fixed ranges; no calibrated regime as a base."""
    o = Optics()
    a1, l1 = rng.uniform(0.10, 0.60), rng.uniform(0.5, 4.0)
    a2, l2 = rng.uniform(0.0, 0.15), rng.uniform(10.0, 80.0)
    k = np.arange(1, o.depth + 1)
    o.weights = [float(x) for x in a1 * np.exp(-(k - 1) / l1) + a2 * np.exp(-(k - 1) / l2)]
    o.drift = float(rng.uniform(-0.3, 1.5))
    o.parallax = float(rng.uniform(-0.1, 0.45))
    o.sigma0 = float(rng.uniform(0.5, 1.3))
    o.sigma_k = float(rng.uniform(0.0, 0.5))
    o.edge_gain = float(rng.uniform(0.0, 0.6))
    o.g_empty = float(rng.uniform(55, 150))
    o.gain = float(rng.uniform(30, 90))
    o.illum_amp = float(rng.uniform(5, 30))
    o.curtain_amp = float(rng.uniform(0, 6))
    o.dose = float(rng.uniform(80, 800))
    o.read_noise = float(rng.uniform(2, 18))
    if rng.random() < 0.5:
        o.bright_frac = float(rng.uniform(0.1, 0.5))
        o.bright_boost = float(rng.uniform(1.5, 5.0))
    else:
        o.bright_frac, o.bright_boost = 0.0, 1.0
    sh = {"corr_z": float(rng.uniform(2.5, 8.0)), "corr_xy": float(rng.uniform(3.0, 8.0))}
    return o, sh


TRAIN_FAMILIES_V2 = ["catalyst", "agglomerate", "spinodal"]   # fibre stays held out


def generic_material_v2(rng):
    """Randomised v2 material recipe, from ranges -- never the recipes fitted
    to comp / uncomp / pristine (relevel.py), which sit inside these ranges."""
    fam = str(rng.choice(TRAIN_FAMILIES_V2))
    kw = {"corr": (float(rng.uniform(2.5, 8.0)), float(rng.uniform(3.0, 7.0)),
                   float(rng.uniform(3.0, 7.0)))}
    tex = dict(tex_amp=float(rng.uniform(0.0, 0.45)), rim=float(rng.uniform(0.0, 1.5)),
               rim_len=float(rng.uniform(1.0, 2.5)))
    if fam == "catalyst":
        kw.update(y_stretch=float(rng.uniform(0.8, 1.8)), macro_frac=float(rng.uniform(0.0, 0.3)),
                  macro_amp=float(rng.uniform(0.2, 1.2)), rough=float(rng.uniform(0.2, 0.8)),
                  rough_sigma=float(rng.uniform(1.0, 1.6)), hetero=float(rng.uniform(0.0, 0.6)),
                  **tex)
        tex = None
    return fam, kw, tex


def make_v2(opt, family, kw, tex_kw, porosity, seed):
    """As make(), with the v2 material and its emission texture rendered."""
    solid, tex = materials.build(family, SHAPE, porosity, seed=seed, with_texture=True, **kw)
    rng = np.random.default_rng(seed + 7)
    if tex is None and tex_kw:
        tex = materials.se_texture(solid, rng, **tex_kw)
    zs = list(range(ZHI + 6))
    img = add_noise(render(solid, opt, zs, rng, texture=tex), opt, rng)
    shifts = [int(round(opt.drift * i)) for i in zs]
    img = np.stack([np.roll(img[i], shifts[i], axis=0) for i in zs])
    pore = np.zeros((len(zs), SHAPE[1], SHAPE[2]), bool)
    depth = np.zeros_like(pore, dtype=np.uint8)
    for z in range(ZLO, ZHI + 1):
        pore[z] = np.roll(~solid[z], shifts[z], axis=0)
        d = depth_to_solid(solid, z, 8, opt.parallax).astype(np.uint8)
        depth[z] = np.roll(d, shifts[z], axis=0)
    return img, pore, depth


SHAPE_V3 = (128, 192, 384)       # deep see-through needs ~80 slices below a frame
ZHI_V3 = SHAPE_V3[0] - 82


def generic_render_v3(rng, opt):
    """Random see-through renderer settings and depth curve (v3), from ranges
    spanning the three fitted styles (optics_v3.json, deep refit) with margin."""
    D = int(rng.choice([48, 64, 80]))
    g0 = opt.g_empty
    a = rng.uniform(30, 110); lam = rng.uniform(3, 12)
    curve = np.zeros(D + 2, np.float32)
    for k in range(1, D + 1):
        curve[k] = g0 + 15.0 + a * np.exp(-(k - 1) / lam)
    curve[D + 1] = g0
    rk = dict(smooth=float(rng.uniform(1.2, 3.2)), secant=float(rng.uniform(0.4, 1.6)),
              cos_floor=0.2, hit_albedo=True, defocus=float(rng.uniform(0.0, 0.06)),
              hole_floor=float(rng.uniform(0.4, 0.9) * g0), hole_len=float(rng.uniform(2.5, 12)),
              face_smooth=float(rng.uniform(0.3, 1.8)), escape=float(rng.uniform(1.0, 4.0)),
              sigma0=float(rng.uniform(0.8, 1.6)), shade_floor=float(rng.uniform(0.25, 0.6)))
    face = float(curve[1] + rng.uniform(10, 60))
    return D, curve, face, rk


def generic_material_v3(rng):
    fam = str(rng.choice(["spheres", "catalyst", "agglomerate"]))
    if fam == "spheres":
        sc = rng.uniform(0.7, 1.7)
        kw = dict(r_med=float(rng.uniform(1.8, 3.0)), r_spread=float(rng.uniform(0.25, 0.75)),
                  cluster_sigma=tuple(float(x * sc) for x in (12, 10, 8)),
                  per_cluster=float(rng.uniform(50, 500)), film=float(rng.uniform(1.0, 2.5)),
                  film_cover=float(rng.uniform(0.7, 0.95)), fill=float(rng.uniform(0.4, 1.1)),
                  macro_frac=float(rng.uniform(0.0, 0.35)), rim=float(rng.uniform(0.2, 1.0)))
        return fam, kw, None
    return generic_material_v2(rng)


def make_v3(opt, family, kw, tex_kw, porosity, seed, rng):
    """As make_v2, rendered with the v3 see-through renderer."""
    from microscope import render_surface
    D, curve, face, rk = generic_render_v3(rng, opt)
    opt.depth = D
    solid, tex = materials.build(family, SHAPE_V3, porosity, seed=seed, with_texture=True, **kw)
    r2 = np.random.default_rng(seed + 7)
    if tex is None:
        tex = materials.se_texture(solid, r2, **(tex_kw or {"tex_amp": 0.15, "rim": 0.8}))
    alb = np.where(solid, tex, 0).astype(np.float32)
    zs = list(range(ZHI_V3 + 6))
    grey, _ = render_surface(solid, alb, opt, zs, curve, r2, face_level=face, **rk)
    img = add_noise(grey, opt, r2)
    shifts = [int(round(opt.drift * i)) for i in zs]
    img = np.stack([np.roll(img[i], shifts[i], axis=0) for i in zs])
    pore = np.zeros((len(zs), SHAPE_V3[1], SHAPE_V3[2]), bool)
    depth = np.zeros_like(pore, dtype=np.uint8)
    for z in range(ZLO, ZHI_V3 + 1):
        pore[z] = np.roll(~solid[z], shifts[z], axis=0)
        d = depth_to_solid(solid, z, 8, opt.parallax).astype(np.uint8)
        depth[z] = np.roll(d, shifts[z], axis=0)
    return img, pore, depth, rk


_REF = {}


def make_v5(rng, seed):
    """v5 physics with settings jittered around the three blind-tuned styles."""
    import gen2, v4
    from microscope import render_surface
    cal = json.loads((HERE / "optics_v5.json").read_text())
    style = str(rng.choice(["comp", "uncomp", "pristine"]))
    P = json.loads(json.dumps(cal[style]["v5"]))
    j = lambda x, r=0.25: float(x * np.exp(rng.normal(0, r)))
    for k in ("smooth", "escape", "comp_amp", "comp_sig", "fun_lam", "edge", "esc", "lump_amp", "sigma_y", "sigma_x"):
        if k in P: P[k] = j(P[k])
    P["hole_floor"] = float(np.clip(j(P["hole_floor"]), 0.15, 0.9)); P["fun_mix"] = float(np.clip(j(P["fun_mix"]), 0, 0.9))
    opt, _ = generic_physics(rng)
    opt.depth = 64; opt.bright_frac = 0.0
    por = float(rng.uniform(0.35, 0.65))
    fam = str(rng.choice(["voro", "voro", "web", "spheres"]))
    if fam == "voro":
        vk = dict(P["voro"]); vk["warp_sig"] = tuple(vk["warp_sig"])
        for k in ("big", "small", "warp"): vk[k] = j(vk[k], 0.2)
        vk["p_big"] = float(np.clip(j(vk["p_big"], 0.3), 0.05, 0.5))
        solid = gen2.vorofoam(SHAPE_V3, por, rng, **vk)
    elif fam == "web":
        solid = v4.web(SHAPE_V3, por, rng, wall=(j(4), j(6), j(4)), fine=0.12, mix=0.95, agg_amp=0.3, macro_frac=0.1)
    else:
        solid, _ = materials.build("spheres", SHAPE_V3, por, seed=seed, with_texture=True)
    n = ZHI_V3 + 6
    a = materials._grf(SHAPE_V3, (P["comp_sig"],) * 3, rng, octaves=2, gain=0.5)
    a = np.exp(P["comp_amp"] * a / (a.std() + 1e-9)).astype(np.float32); a /= float(a[solid].mean())
    alb = np.where(solid, a, 0).astype(np.float32)
    fy = v4.face_yield(solid, list(range(n)), iv=(0.8, 0.8, 0.8), esc=P["esc"], edge=P["edge"]) * a[:n]
    D = 64
    curve = np.zeros(D + 2, np.float32); g0 = float(rng.uniform(60, 130)); A = rng.uniform(40, 110); lam = rng.uniform(3, 12)
    curve[1:D + 1] = g0 + 15 + A * np.exp(-np.arange(D) / lam); curve[D + 1] = g0
    g, _ = render_surface(solid, alb, opt, list(range(n)), curve, rng, illum=np.zeros(SHAPE_V3[1:], np.float32),
                          face_level=float(curve[1] + rng.uniform(10, 60)), face_map=fy, smooth=P["smooth"],
                          secant=P["secant"], cos_floor=0.2, hit_albedo=True, defocus=0.03,
                          hole_floor=P["hole_floor"] * g0, hole_len=6.0, escape=P["escape"], sigma0=0.5, shade_floor=0.4)
    g = v4.funnel(g, solid[:n], {"mix": P["fun_mix"], "lam": P["fun_lam"], "rim": float(curve[1]), "floor": P["fun_floor"], "soft": 1.0}, opt)
    g = ndi.gaussian_filter(g, (0, P["sigma_y"], P["sigma_x"]))
    # finish exactly like the packages: monotone map onto the real layer's
    # grey distribution (with a mild random gamma so the model does not
    # learn one tone curve), then the measured real grain
    ref = _REF.setdefault(style, np.sort(v4.real_hist(style)))
    qs = np.linspace(0, 1, 512) ** float(np.exp(rng.normal(0, 0.15)))
    src = np.sort(g[:, ::3, ::3].ravel())
    g = np.interp(g, np.quantile(src, np.linspace(0, 1, 512)), np.quantile(ref, qs)).astype(np.float32)
    g = g + P["grain"] * rng.uniform(0.85, 1.15) * rng.standard_normal(g.shape) * np.sqrt(np.clip(g, 1, 255) / P["grain_ref"])
    img = np.clip(g, 0, 255).astype(np.uint8)
    zs = list(range(n))
    shifts = [int(round(opt.drift * i)) for i in zs]
    img = np.stack([np.roll(img[i], shifts[i], axis=0) for i in zs])
    pore = np.zeros((n, SHAPE_V3[1], SHAPE_V3[2]), bool); depth = np.zeros_like(pore, dtype=np.uint8)
    for z in range(ZLO, ZHI_V3 + 1):
        pore[z] = np.roll(~solid[z], shifts[z], axis=0)
        depth[z] = np.roll(depth_to_solid(solid, z, 8, opt.parallax).astype(np.uint8), shifts[z], axis=0)
    return img, pore, depth, {"style": style, "family": fam, "porosity": por, "drift": opt.drift}


def make_v7(rng, seed, width=384, rows=192):
    """Training volume from the v7 generator itself (make_full.build).

    v5 training data came from a separate, simpler renderer; v7 builds each
    volume with the same code as the packages and full stacks -- slice
    thickness, micropores in the truth, directional edges, scan-correlated
    noise, 3-D context -- with the v7 settings of a random style jittered
    about +-25% (lognormal) and a random porosity, then crops a `rows`-high
    band that lies inside the layer. Labels come straight from the exact
    truth, already in the image frame; depth is measured along the beam path
    (drift + parallax) like everywhere else.
    """
    import make_full
    style = str(rng.choice(["comp", "uncomp", "pristine"]))
    sl = json.loads((HERE / "slicing.json").read_text())[style]
    ov = json.loads(json.dumps(sl.get("v7", {})))
    j = lambda x, r=0.25: float(x * np.exp(rng.normal(0, r)))
    for k in ("edge_dy", "micropore", "grain", "sigma_y", "sigma_x", "curtain_illum", "incl_per_kpx"):
        if k in ov: ov[k] = j(ov[k])
    if "voro" in ov:
        for k in ("big", "small", "warp"): ov["voro"][k] = j(ov["voro"][k], 0.2)
    por = float(rng.uniform(0.35, 0.65))
    fam = "v5" if rng.random() < 0.8 else "spheres"
    n = ZHI_V3 + 9                                    # depth label looks 8 slices down
    r = make_full.build(style, n, int(seed), fam, por, None, width=width, overrides=ov)
    img, tru = r["image"], r["truth"]
    valid = (tru != 128).mean(axis=(0, 2))
    best = int(np.argmax(np.convolve(valid, np.ones(rows), "valid")))
    img, tru = img[:, best:best + rows], tru[:, best:best + rows]
    pore = tru == 0                                   # stray outside-layer pixels count as solid
    opt = r["optics"]; s1 = float(opt.drift + opt.parallax)
    depth = np.zeros(pore.shape, np.uint8)
    for z in range(ZLO, ZHI_V3 + 1):
        d = np.full(pore[z].shape, 9, np.uint8); d[~pore[z]] = 0; open_ = pore[z].copy()
        for k in range(1, 9):
            sh = int(round(s1 * k)); nxt = ~pore[z + k]
            b = np.zeros_like(nxt); b[:rows - sh] = nxt[sh:] if sh > 0 else nxt[:rows - sh]
            hit = open_ & b; d[hit] = k; open_ &= ~hit
        depth[z] = d
    return img, pore, depth, {"style": style, "family": fam, "porosity": por, "drift": float(opt.drift)}


def make(opt, sh, family, porosity, seed):
    """Image-frame stack, truth and depth labels for target frames ZLO..ZHI."""
    aniso = 0.55 if family == "compressed" else 1.0
    cxy = sh.get("corr_xy", 5.0)
    solid = materials.build(family, SHAPE, porosity, seed=seed,
                            corr=(sh["corr_z"], cxy * aniso, cxy))
    bright = materials.bright_phase(solid, opt.bright_frac, seed) if opt.bright_frac else None
    rng = np.random.default_rng(seed + 7)
    zs = list(range(ZHI + 6))
    img = add_noise(render(solid, opt, zs, rng, bright=bright), opt, rng)
    shifts = [int(round(opt.drift * i)) for i in zs]
    img = np.stack([np.roll(img[i], shifts[i], axis=0) for i in zs])
    pore = np.zeros((len(zs), SHAPE[1], SHAPE[2]), bool)
    depth = np.zeros_like(pore, dtype=np.uint8)
    for z in range(ZLO, ZHI + 1):
        pore[z] = np.roll(~solid[z], shifts[z], axis=0)
        d = depth_to_solid(solid, z, 8, opt.parallax).astype(np.uint8)
        depth[z] = np.roll(d, shifts[z], axis=0)
    return img, pore, depth


def estimate_drift(img, gap=6, rows=None):
    """Label-free drift: NCC peak between slices `gap` apart, sub-pixel."""
    h = img.shape[1]
    y0, y1 = (h // 4, 3 * h // 4) if rows is None else rows
    est = []
    for i in range(0, img.shape[0] - gap, 4):
        a = img[i, y0:y1].astype(np.float32); a -= a.mean()
        vals = []
        for dy in range(-12, 13):
            b = img[i + gap, y0 + dy:y1 + dy].astype(np.float32); b -= b.mean()
            vals.append(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        j = int(np.argmax(vals)); off = 0.0
        if 0 < j < len(vals) - 1:
            l, c, r = vals[j - 1], vals[j], vals[j + 1]
            den = l - 2 * c + r
            off = 0.5 * (l - r) / den if den else 0.0
        est.append((j - 12 + off) / gap)
    return float(np.median(est))


def windows(img, z_list, drift, norm_mask=None):
    """Registered 9-slice windows, target slice untouched.

    Neighbour z+k is shifted by -drift*k so it lines up with slice z. Grey is
    normalised per stack (median / IQR) over `norm_mask` if given.
    """
    ref = img[z_list[0]] if norm_mask is None else img[z_list[0]][norm_mask]
    med = float(np.median(ref))
    # a saturated stack can have IQR 0; never divide by less than 2 grey levels
    iqr = max(float(np.subtract(*np.percentile(ref, [75, 25]))), 0.5 * float(np.std(ref)), 2.0)
    out = np.empty((len(z_list), len(WIN), img.shape[1], img.shape[2]), np.float16)
    for n, z in enumerate(z_list):
        for j, k in enumerate(WIN):
            s = img[z + k].astype(np.float32)
            if k:
                s = ndi.shift(s, (-drift * k, 0), order=1, mode="nearest")
            out[n, j] = (s - med) / iqr
    return out


def build_set(kind, n, seed0, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed0)
    meta = []
    for i in range(n):
        seed = seed0 * 1000 + i
        if kind == "random_v5":
            img, pore, depth, info = make_v5(rng, seed)
            est = estimate_drift(img)
            zl = list(range(ZLO, ZHI_V3 + 1, 2))
            X = windows(img, zl, est)
            np.savez_compressed(out / f"vol_{i:03d}.npz", X=X, pore=pore[zl], depth=depth[zl])
            meta.append({"i": i, **info, "drift_est": est})
            print(f"  {out.name}/vol_{i:03d}  {info['family']:<8} {info['style']:<9} drift {info['drift']:+.2f} est {est:+.2f}", flush=True)
            continue
        if kind == "random_v3":
            opt, _ = generic_physics(rng)
            opt.edge_gain = float(rng.uniform(0.0, 1.0))
            opt.bright_frac, opt.bright_boost = 0.0, 1.0
            fam, kw, tex_kw = generic_material_v3(rng)
            por = float(rng.uniform(0.35, 0.65))
            img, pore, depth, rk = make_v3(opt, fam, kw, tex_kw, por, seed, rng)
            est = estimate_drift(img)
            zl = list(range(ZLO, ZHI_V3 + 1, 2))
            X = windows(img, zl, est)
            np.savez_compressed(out / f"vol_{i:03d}.npz", X=X, pore=pore[zl], depth=depth[zl])
            meta.append({"i": i, "family": fam, "porosity": por, "drift": opt.drift,
                         "drift_est": est, "parallax": opt.parallax, "render": rk})
            print(f"  {out.name}/vol_{i:03d}  {fam:<11} drift {opt.drift:+.2f} est {est:+.2f}",
                  flush=True)
            continue
        if kind == "random_v2":
            opt, _ = generic_physics(rng)
            # the v2 relevel put edge_gain at 0-1.0; v1's range stopped at 0.6
            opt.edge_gain = float(rng.uniform(0.0, 1.0))
            opt.bright_frac, opt.bright_boost = 0.0, 1.0   # v2 grain/rim replaces it
            fam, kw, tex_kw = generic_material_v2(rng)
            por = float(rng.uniform(0.35, 0.65))
            img, pore, depth = make_v2(opt, fam, kw, tex_kw, por, seed)
            est = estimate_drift(img)
            zl = list(range(ZLO, ZHI + 1, 2))
            X = windows(img, zl, est)
            np.savez_compressed(out / f"vol_{i:03d}.npz", X=X, pore=pore[zl], depth=depth[zl])
            meta.append({"i": i, "family": fam, "porosity": por, "drift": opt.drift,
                         "drift_est": est, "parallax": opt.parallax,
                         "material": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
                         "texture": tex_kw})
            print(f"  {out.name}/vol_{i:03d}  {fam:<11} drift {opt.drift:+.2f} est {est:+.2f}",
                  flush=True)
            continue
        if kind == "random":
            opt, sh = generic_physics(rng)
            fam = str(rng.choice(TRAIN_FAMILIES)); por = float(rng.uniform(0.35, 0.65))
        elif kind.startswith("narrow_"):
            opt, sh = calibrated(kind.split("_", 1)[1])
            fam = str(rng.choice(TRAIN_FAMILIES)); por = float(rng.uniform(0.35, 0.65))
        else:                                   # test: "<regime>:<family>"
            regime, fam = kind.split(":")
            opt, sh = calibrated(regime)
            por = float(sh["porosity"])
        img, pore, depth = make(opt, sh, fam, por, seed)
        est = estimate_drift(img)
        step = 2 if kind in ("random",) or kind.startswith("narrow_") else 1
        zl = list(range(ZLO, ZHI + 1, step))
        X = windows(img, zl, est)
        np.savez_compressed(out / f"vol_{i:03d}.npz", X=X, pore=pore[zl], depth=depth[zl])
        meta.append({"i": i, "family": fam, "porosity": por, "drift": opt.drift,
                     "drift_est": est, "parallax": opt.parallax})
        print(f"  {out.name}/vol_{i:03d}  {fam:<11} drift {opt.drift:+.2f} est {est:+.2f}",
              flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--only-v5", action="store_true")
    ap.add_argument("--only-v3", action="store_true",
                    help="build just train_random_v3 (see-through renderer)")
    ap.add_argument("--only-v2", action="store_true",
                    help="build just train_random_v2 (v2 material, same physics ranges)")
    args = ap.parse_args()
    root = Path(args.out).expanduser()
    if args.only_v5:
        build_set("random_v5", args.n_train, 51, root / "train_random_v5")
        return
    if args.only_v3:
        build_set("random_v3", args.n_train, 31, root / "train_random_v3")
        return
    if args.only_v2:
        build_set("random_v2", args.n_train, 21, root / "train_random_v2")
        return
    build_set("random", args.n_train, 11, root / "train_random")
    build_set("narrow_comp", args.n_train, 12, root / "train_narrow_comp")
    t = 30
    for regime in ("comp", "uncomp", "pristine"):
        for fam in TRAIN_FAMILIES + HELDOUT_FAMILIES:
            t += 1
            build_set(f"{regime}:{fam}", 1, t, root / f"test_{regime}_{fam}")


if __name__ == "__main__":
    main()
