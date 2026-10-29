#!/usr/bin/env python3
"""Build a synthetic pFIB-SEM dataset as a team_review package.

Output is byte-compatible with the packages the team has been annotating:
same mask encoding (0 pore / 128 outside / 255 solid), same assignment.csv
columns, same start_masks/ and annotators/ layout, the same review_tool.py.
So it opens in the same napari launcher and merges with the same combine.py,
and anything measured on it can be compared with the real rounds directly.

What the real packages cannot give you is the answer. These carry truth/,
the exact mask read off the same voxel grid the image was rendered from, so
porosity error is a number rather than a disagreement.

    python make_package.py --style comp --family compressed --frames 118
    python make_package.py --style pristine --family mudcrack --name mud_a
    python make_package.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

import materials
import context
from microscope import Optics, render, render_surface, add_noise, _illumination
import see_through

HERE = Path(__file__).resolve().parent
from paths import LABELS_DIR, OUT as OUT_ROOT
TEAM_PKG = LABELS_DIR / "team_review"      # the team's review tools, copied in when present

PORE, OUTSIDE, SOLID = 0, 128, 255
PAD_Z = 26          # slices past the last frame; shine-through reaches ~24


def adaptive_fixed(stack, roi, level, adapt=0.30, win=251):
    """The adaptive threshold at a level tuned on REAL data (real_labeller.py).

    Truth is never consulted, so the porosity this produces on a synthetic
    stack is a measurement of the labeller, not something chosen in advance.
    """
    data = stack.astype(np.float32)
    size = (0, win, win)
    tot = ndi.uniform_filter(np.where(roi, data, 0), size)
    shr = ndi.uniform_filter(roi.astype(np.float32), size)
    field = np.where(shr > 0.02, tot / np.maximum(shr, 1e-6), 0).astype(np.float32)
    for z in range(len(field)):
        if roi[z].any():
            field[z][~roi[z]] = data[z][roi[z]].mean()
    offset = adapt * (field - float(field[roi].mean()))
    return (data <= (level + offset)) & roi


def render_v5(style, P, zone, solid, opt, n, rng, lev, Hv, W):
    """v5: Voronoi foam + see-through physics + composition + funnels.

    Same physics as evolve.render (where it was tuned on crops by blind
    tests), applied to the whole generated frame. Returns noiseless grey,
    already tone-mapped (one monotone map, fitted on layer pixels) onto the
    real layer's grey distribution.
    """
    import v4, see_through
    from microscope import render_surface
    t = json.loads((HERE / "optics.json").read_text())[style]["targets"]
    face = float(t["solid_mean"])
    cl_solid = ((zone == context.CL) | (zone == context.FRINGE)) & solid
    occ = (zone == context.CAP) | (zone == context.MATRIX) | cl_solid
    alb = np.zeros(zone.shape, np.float32)
    a = materials._grf(zone.shape, (P["comp_sig"],) * 3, rng, octaves=2, gain=0.5)
    a = np.exp(P["comp_amp"] * a / (a.std() + 1e-9)).astype(np.float32)
    a /= float(a[cl_solid].mean())
    alb[cl_solid] = a[cl_solid]; del a
    alb[zone == context.CAP] = lev["cap_mean"] / face
    alb[zone == context.MATRIX] = lev["sub_p50"] / face
    fy = v4.face_yield(occ, list(range(n)), iv=(0.8, 0.8, 0.8), esc=P["esc"], edge=P["edge"])
    fy = fy * np.where(cl_solid[:n], alb[:n], 1.0)
    if P.get("lump_amp"):
        lf = materials._grf((n,) + zone.shape[1:], (2.0, 15.0, 15.0), rng); lf /= lf.std() + 1e-9
        fy = fy * (1.0 + P["lump_amp"] * lf)
    if P.get("curtain_amp"):
        k = np.abs(np.fft.fftfreq(W)) + 1.0 / W
        c = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal((n, W)), axis=1) * k ** -0.7
                                * np.exp(-(k * P.get("curtain_w", 8.0)) ** 2), axis=1))
        c /= c.std() + 1e-9
        pf = 1.0 - occ[:n].astype(np.float32)
        cs = np.cumsum(pf, 1); above = (cs - np.roll(cs, 40, 1)) / 40.0
        cur = 1.0 + P["curtain_amp"] * c[:, None, :] * (0.5 + np.clip(above, 0, 1))
        fy = np.where(cl_solid[:n], fy * cur, fy)        # milling grooves in the layer only
    opt.depth = int(P.get("depth", 80))
    curve = see_through.depth_curve(style, opt.depth, optics="optics_v3.json")
    g, _ = render_surface(occ, alb, opt, list(range(n)), curve, rng, illum=np.zeros((Hv, W), np.float32),
                          face_level=face, face_map=fy, smooth=P["smooth"], secant=P["secant"], cos_floor=0.2,
                          hit_albedo=True, defocus=0.03, hole_floor=P["hole_floor"] * opt.g_empty, hole_len=6.0,
                          escape=P["escape"], sigma0=0.5, shade_floor=0.4)
    del alb, fy
    g = v4.funnel(g, occ[:n], {"mix": P["fun_mix"], "lam": P["fun_lam"], "rim": 180,
                               "floor": P["fun_floor"], "soft": 1.0}, opt)
    g = ndi.gaussian_filter(g, (0, P["sigma_y"], P["sigma_x"]))
    # monotone tone map: layer pixels -> real layer quantiles; applied to all
    lay = (zone[:n] == context.CL)
    src = np.sort(g[lay].ravel()[::7]); ref = np.sort(v4.real_hist(style))
    qs = np.linspace(0, 1, 512)
    g = np.interp(g, np.quantile(src, qs), np.quantile(ref, qs)).astype(np.float32)
    return g, None


def build(args, cal, geo):
    """Generate one stack: layout, material, render, frame, label.

    v1 (make_package_v1.py) borrowed the ROI outline and all context from
    real frames. Here the layout is drawn fresh per seed (context.py) and the
    whole frame -- cap, layer, support -- is rendered as one volume.
    """
    style = args.style
    opt = Optics(**cal[style]["optics"])
    g = geo[style]
    H, W = g["height"], g["width"]
    rng = np.random.default_rng(args.seed)
    n = args.frames
    # the see-through renderer looks up to `depth` slices past the last frame
    st_depth = int(cal[style].get("see_through", {}).get("params", {}).get("depth", 0))
    nz = n + max(PAD_Z, st_depth + 2)
    lev = json.loads((HERE / "real_context_levels.json").read_text())[style]

    drift = float(opt.drift)
    margin = int(np.ceil(abs(opt.parallax) * (PAD_Z + 2))) + 6
    travel = int(np.ceil(abs(drift) * (nz + 1)))
    Hv = H + travel + 2 * margin
    base = margin + travel
    st = np.array([base - int(round(drift * z)) for z in range(nz)])
    print(f"style {style}  family {args.family}  porosity target {args.porosity}")
    print(f"frame {H} x {W}, drift {drift:+.2f} px/slice, parallax {opt.parallax:+.2f}, "
          f"volume {(nz, Hv, W)} = {nz * Hv * W / 1e6:.0f}M voxels", flush=True)

    # --- layout, generated (nothing copied from real frames)
    t0 = time.time()
    lay = context.draw_layout(style, nz, H, W, rng)
    zone = np.empty((nz, Hv, W), np.uint8)
    for z in range(nz):
        zf, _, _ = context.frame_zones(lay, z, H, W)
        zone[z, :st[z]] = context.CAP
        zone[z, st[z]:st[z] + H] = zf
        zone[z, st[z] + H:] = context.MATRIX
    print(f"  layout drawn in {time.time() - t0:.1f}s", flush=True)

    # --- material
    t0 = time.time()
    kw = json.loads(args.mat_kw) if args.mat_kw else {}
    st_fit = cal[style].get("see_through")
    family = args.family
    if family == "fitted":           # the material fitted with the v3 renderer
        family, fkw = see_through.material_kw(cal[style], st_fit["params"]); kw = {**fkw, **kw}
    elif family == "catalyst":
        rec = dict(cal[style]["material"]); rec.pop("family"); rec.pop("porosity", None)
        rec["corr"] = tuple(rec["corr"]); kw = {**rec, **kw}
    V5 = cal[style].get("v5")            # v5 physics for every family
    if family == "v5":
        import gen2
        vk = dict(V5["voro"]); vk["warp_sig"] = tuple(vk.get("warp_sig", (3.0, 4.0, 3.0)))
        solid = gen2.vorofoam(zone.shape, args.porosity, rng, **vk); tex = None
    else:
        solid, tex = materials.build(family, zone.shape, args.porosity, seed=args.seed,
                                     with_texture=True, **kw)
    if tex is None:     # families without their own: same SE edge glow and grain
        tex = materials.se_texture(solid, rng, tex_amp=0.15, rim=1.0)
    print(f"  material {family} built in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    if V5 is not None:
        raw, roi_tone = render_v5(style, V5, zone, solid, opt, n, rng, lev, Hv, W)
        del tex
    elif st_fit is not None:
        # v3: see-through rendering. Everything that is not open space is a
        # surface the beam can hit; albedo sets its relative brightness.
        occ = (zone == context.CAP) | (zone == context.MATRIX) | ((zone == context.CL) & solid) \
            | ((zone == context.FRINGE) & solid)
        alb = np.zeros(zone.shape, np.float32)
        clz = ((zone == context.CL) | (zone == context.FRINGE)) & solid
        alb[clz] = tex[clz]
        t_lv = json.loads((HERE / "optics.json").read_text())[style]["targets"]
        face = float(t_lv["solid_mean"])
        alb[zone == context.CAP] = lev["cap_mean"] / face
        alb[zone == context.MATRIX] = lev["sub_p50"] / face
        del tex
        p = st_fit["params"]
        opt.depth = int(p["depth"])
        curve = see_through.depth_curve(style, opt.depth, optics=args.optics)
        raw, _ = render_surface(occ, alb, opt, list(range(n)), curve, rng,
                                illum=np.zeros((Hv, W), np.float32), face_level=face,
                                **see_through.renderer_kw(p, opt))
        a_t, b_t = st_fit["tone"]
        raw = a_t * raw + b_t
        del occ, alb
    else:
        # v2: depth-weighted glow (render)
        wsum = 1.0 + float(np.sum(opt.weights[:opt.depth])) if opt.weights else 3.0
        em = np.zeros(zone.shape, np.float32)
        cl = zone == context.CL
        em[cl & solid] = tex[cl & solid]
        e_cap = max(lev["cap_mean"] - opt.g_empty, 5.0) / (opt.gain * wsum)
        cap_tex = 1.0 + 0.06 * materials._grf(zone.shape, (8.0, 30.0, 30.0), rng)
        capz = (zone == context.CAP) | ((zone == context.FRINGE) & ~solid)
        em[capz] = (e_cap * cap_tex)[capz]
        fr = (zone == context.FRINGE) & solid
        em[fr] = tex[fr]
        del capz, fr, cap_tex
        e_sub = 0.25 * e_cap
        em[zone == context.MATRIX] = e_sub
        for z in range(nz):   # soft glowing shell around support voids
            v = zone[z] == context.VOID
            if v.any():
                d = ndi.distance_transform_edt(~v)
                shell = (zone[z] == context.MATRIX) & (d < 7)
                em[z][shell] += 3.5 * e_sub * np.exp(-(d[shell] - 1.0) / 2.0)
                din = ndi.distance_transform_edt(v)
                em[z][v] = 0.9 * e_sub * np.exp(-din[v] / 7.0)
        del tex
        raw = render(np.ones(zone.shape, bool), opt, list(range(n)), rng,
                     illum=np.zeros((Hv, W), np.float32), texture=em)
        del em
    print(f"  rendered {n} slices in {time.time() - t0:.1f}s", flush=True)

    # --- frame, tone the context zones, overlays, noise
    illum = _illumination((H, W), opt, rng)
    streak = materials._grf((1, 1, W), (1.0, 1.0, 1.5), rng)[0, 0]
    image = np.empty((n, H, W), np.uint8)
    truth = np.full((n, H, W), OUTSIDE, np.uint8)
    roi_all = np.zeros((n, H, W), bool)
    yy = np.arange(H, dtype=np.float32)[:, None]
    tone = None
    for i in range(n):
        f = raw[i, st[i]:st[i] + H]
        zf, top, bot = context.frame_zones(lay, i, H, W)
        if st_fit is not None:
            # support holes: outside the ROI and never scored. A hole is a
            # tube running along z, so the slanted beam keeps hitting its far
            # wall at stepped depths (stripes); real support holes read as a
            # dark opening that lightens toward a bright rim, so draw that
            vd = zf == context.VOID
            if vd.any():
                f = f.copy()
                mlev = float(np.median(f[zf == context.MATRIX])) if (zf == context.MATRIX).any() else float(f.mean())
                din = ndi.distance_transform_edt(vd)
                dout = ndi.distance_transform_edt(~vd)
                yy_ = np.arange(f.shape[0], dtype=np.float32)[:, None]
                inner = 0.62 * mlev + 0.22 * mlev * np.exp(-(din - 1) / 9.0)
                # lit far wall: lighter toward the bottom of the opening
                lab_, nl = ndi.label(vd)
                if nl:
                    tops = ndi.minimum(yy_ * np.ones_like(f), lab_, index=np.arange(1, nl + 1))
                    bots = ndi.maximum(yy_ * np.ones_like(f), lab_, index=np.arange(1, nl + 1))
                    t_ = np.r_[0, tops][lab_]; b_ = np.r_[1, bots][lab_]
                    inner = inner + 0.12 * mlev * np.clip((yy_ - t_) / np.maximum(b_ - t_, 1), 0, 1)
                f[vd] = inner[vd]
                rim = (zf == context.MATRIX) & (dout < 6)
                f[rim] += 0.45 * mlev * np.exp(-(dout[rim] - 1) / 1.8)
                soft = ndi.gaussian_filter(f, 1.2)
                band = (dout < 4) | ((din < 4) & vd)
                f = np.where(band, soft, f)
        below = ndi.gaussian_filter((yy > bot[None]).astype(np.float32), 2.0)
        above = ndi.gaussian_filter((zf == context.CAP).astype(np.float32), 2.0)
        if tone is None:   # one affine per context zone, fixed for the stack
            sb = f[(yy > bot[None] + 60) & (zf != context.LENS)]
            r5, r50 = np.percentile(sb, [5, 50])
            a_s = min((lev["sub_p50"] - lev["sub_p5"]) / max(r50 - r5, 1.0), 2.5)
            b_s = lev["sub_p5"] - a_s * r5
            ca = f[(yy < top[None] - 6) & (zf == context.CAP)]
            b_c = lev["cap_mean"] - float(ca.mean())
            tone = (a_s, b_s, b_c)
            print(f"  context tone: support a={a_s:.2f} b={b_s:.1f}, cap shift {b_c:+.1f}", flush=True)
        a_s, b_s, b_c = tone
        f = f + below * ((a_s * f + b_s) - f) + above * b_c
        dist = np.maximum(yy - bot[None], 0)
        f = f + below * ((lev["near_mean"] - lev["sub_p50"]) * np.exp(-dist / 25.0)
                         + 3.0 * streak[None] * np.exp(-dist / 60.0))
        blk, side = context.frame_overlays(lay, i, H, W)
        if side.any():
            wall = lev["cap_mean"] * 0.8 + 10.0 * streak[None] + 0 * yy
            edge = ndi.gaussian_filter(side.astype(np.float32), 1.5)
            f = np.where(side, wall, f) + 40.0 * (edge * (1 - edge)) * 4
        f = f + illum
        if V5 is not None:
            # measured real grain, signal-dependent (shot-like)
            gsd = V5["grain"]
            img = np.clip(f + gsd * rng.standard_normal(f.shape) * np.sqrt(np.clip(f, 1, 255) / V5["grain_ref"]),
                          0, 255).astype(np.uint8)
        else:
            img = add_noise(f[None], opt, rng)[0]
        img[blk] = 0
        image[i] = img
        roi = (zf == context.CL) & ~blk & ~side
        roi_all[i] = roi
        sol = solid[i, st[i]:st[i] + H]
        truth[i][roi] = np.where(sol[roi], SOLID, PORE)
    del raw

    true_por = float((truth == PORE).sum() / ((truth == PORE).sum() + (truth == SOLID).sum()))
    print(f"  true porosity inside ROI: {true_por:.4f}, excluded fraction "
          f"{1 - roi_all.mean():.3f}", flush=True)

    lab = json.loads((HERE / "real_labeller.json").read_text())[style]
    pred = adaptive_fixed(image, roi_all, lab["level"], lab["adapt"], lab["win"])
    start = np.where(roi_all, np.where(pred, PORE, SOLID), OUTSIDE).astype(np.uint8)
    start_por = float((start == PORE).sum() / ((start == PORE).sum() + (start == SOLID).sum()))
    agree = float((start[roi_all] == truth[roi_all]).mean())
    print(f"  start masks (real-tuned level {lab['level']}, truth-blind): porosity "
          f"{start_por:.4f} vs truth {true_por:.4f} ({start_por - true_por:+.4f}), "
          f"pixel agreement {agree:.4f}", flush=True)

    gp = image[roi_all & (truth == PORE)].astype(np.float32)
    gs = image[roi_all & (truth == SOLID)].astype(np.float32)
    stats = {"pore_mean": round(float(gp.mean()), 2), "solid_mean": round(float(gs.mean()), 2),
             "pore_std": round(float(gp.std()), 2), "solid_std": round(float(gs.std()), 2),
             "excluded_fraction": round(float(1 - roi_all.mean()), 4)}
    rows = np.where(roi_all.any(axis=(0, 2)))[0]
    return dict(image=image, truth=truth, start=start, roi=roi_all, optics=opt,
                true_porosity=true_por, start_porosity=start_por,
                start_agreement=agree, start_level=lab["level"], stats=stats,
                band=(int(rows[0]), int(rows[-1]) + 1),
                top=np.array([int(np.where(r.any(1))[0][0]) for r in roi_all]),
                bot=np.array([int(np.where(r.any(1))[0][-1]) for r in roi_all]),
                layout_seed=args.seed)


# ----------------------------------------------------------------- writing
def write_package(args, res, dest: Path):
    n = args.frames
    if dest.exists():
        if any((dest / "annotators").glob("*/*.png")) and not args.force:
            raise SystemExit(f"{dest} already has saved annotator work; "
                             f"refusing to overwrite (pass --force)")
        shutil.rmtree(dest)
    (dest / "start_masks").mkdir(parents=True)
    (dest / "annotators").mkdir()
    (dest / "truth" / "masks").mkdir(parents=True)

    stack_name = f"{args.name}_full.tif"
    tifffile.imwrite(dest / stack_name, res["image"], compression="zlib")

    for z in range(n):
        Image.fromarray(res["start"][z], mode="L").save(
            dest / "start_masks" / f"slice_{z:04d}.png")
        Image.fromarray(res["truth"][z], mode="L").save(
            dest / "truth" / "masks" / f"slice_{z:04d}.png")

    # stack.json, exactly the shape the team's launchers read
    (dest / "stack.json").write_text(json.dumps({
        "name": args.name,
        "stack_file": stack_name,
        "published_porosity": round(res["true_porosity"], 4),
        "start_porosity": round(res["start_porosity"], 4),
    }, indent=2))

    # assignment.csv: 4 annotators, round-robin, plus shared ALL frames
    pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    rows = []
    shared = set(range(5, n, 10))
    for z in range(n):
        if z in shared:
            rows.append({"frame": z, "annotator": "ALL", "cross_checker": "-"})
        else:
            a, c = pairs[(z // 1) % 4]
            rows.append({"frame": z, "annotator": a, "cross_checker": c})
    with open(dest / "assignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "annotator", "cross_checker"])
        w.writeheader()
        w.writerows(rows)

    # the team's own tooling, unmodified, so output stays comparable
    copied = []
    for fn in ("review_tool.py", "combine.py"):
        src = TEAM_PKG / fn
        if src.exists():
            shutil.copy(src, dest / fn)
            copied.append(fn)

    # my helpers and launchers, retargeted at this stack
    tif_abs = (dest / stack_name).resolve()
    for fn in ("helpers.py",):
        src = TEAM_PKG / fn
        if src.exists():
            shutil.copy(src, dest / fn)
            copied.append(fn)
    for fn, extra in (("start_annotating.py", ""), ("start_crosscheck.py", ' "--check"')):
        src = TEAM_PKG / fn
        if not src.exists():
            continue
        txt = src.read_text()
        txt = txt.replace(
            # the path written inside the team's start_annotating.py
            '"/Users/houkinwan/Desktop/Workspace/CMU/capstone/pfib-labeling/comp_full.tif"',
            f'"{tif_abs}"')
        (dest / fn).write_text(txt)
        copied.append(fn)

    truth_json = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "style_calibrated_to": args.style,
        "material_family": args.family,
        "requested_porosity": args.porosity,
        "true_porosity": round(res["true_porosity"], 6),
        "start_mask_porosity": round(res["start_porosity"], 6),
        "start_mask_pixel_agreement": round(res["start_agreement"], 6),
        "start_mask_porosity_error": round(res["start_porosity"] - res["true_porosity"], 6),
        "start_mask_threshold_level": res["start_level"],
        "start_mask_method": "adaptive threshold at a level tuned on the real stack "
                             "(real_labeller.py); truth not consulted",
        "layout": "generated (context.py), seed " + str(res["layout_seed"]) + "; nothing copied from real frames",
        "seed": args.seed,
        "frames": n,
        "optics": res["optics"].as_dict(),
        "appearance_stats": {k: v for k, v in res["stats"].items()
                             if not k.startswith("_")},
        "roi_band": {"top_rows": res["top"].tolist(), "bottom_rows": res["bot"].tolist()},
    }
    (dest / "truth" / "truth.json").write_text(json.dumps(truth_json, indent=2))
    return copied, truth_json


README = """# Synthetic review package -- {name}

Generated {when}. This is **not** real data. It is a simulated pFIB-SEM
stack whose appearance was fitted to the measured statistics of the real
{style} region, carrying a material of type **{family}**.

It is laid out exactly like the real review packages, so it opens with the
same launcher and merges with the same `combine.py`.

## Why this exists

On the real stacks nobody knows the right answer. The published porosity for
the three regions is 0.417 / 0.558 / 0.515, the automatic start masks sat at
0.404 / 0.513 / 0.500, the pseudo-labels before that were at 0.205, and the
team's four-way hand correction of comp finished at 0.401. Those numbers
disagree and there is no way to settle which is right from the data.

Here the answer is known exactly, because the image and the mask were read
off the same voxel grid. `truth/masks/` is that answer.

- true porosity inside the ROI: **{true_por:.4f}**
- the automatic start masks in `start_masks/`: **{start_por:.4f}**
  ({start_err:+.4f}), agreeing with truth on {agree:.2%} of ROI pixels

The start masks come from a threshold tuned on the *real* stack to
reproduce the real start masks; truth was never consulted, so that error is
a measurement, not a setting.

Everything in the frame is synthetic, including the ROI outline, the cap
above it and the support layer below it. The layout is drawn fresh for this
package from statistics measured on the real {style} stack, so no real
outline is reused. After a review round, `score_against_truth.py` says whether the
correction moved toward the answer or away from it.

## What the image contains

Shine-through is simulated, not drawn on. Each slice accumulates signal from
the cut face and from material below it, weighted by depth:

{weights}

- whole-image drift: **{drift:+.2f} px per slice**
- beam parallax: **{parallax:+.2f} px per slice of depth**

{shine_rule}

## Reviewing it

Mask values are 0 = pore, 128 = outside, 255 = solid. In napari: 1 = pore
(red), 2 = solid (clear), 3 = outside (blue).

Material band sits around rows {band_lo}-{band_hi} of {H}; the boundary
wanders by a few px from frame to frame, so it needs checking per frame
rather than once.

Run `start_annotating.py` to annotate, `start_crosscheck.py --check` to
cross-check. Shift+A / Shift+D flips the image one slice without moving the
mask -- the shine-through test.
"""

SHINE_RULE = {
    1.0: ("The whole picture drifts about 1 px per slice; that moves the cut face\n"
          "and buried material alike. Buried material moves a further ~0.2-0.3 px\n"
          "per slice of depth relative to the cut face, looks soft, and brightens\n"
          "as the cut face approaches it. Still and sharp means solid."),
    0.0: ("There is almost no whole-picture drift here. Buried material moves\n"
          "~0.2 px per slice of depth relative to the cut face, looks soft and\n"
          "grey, and **brightens over the next few slices** as the cut face\n"
          "approaches it. Still and sharp means solid."),
}


SEE_THROUGH_TEXT = """Shine-through is simulated, not drawn on. The sample is not infiltrated,
so the microscope looks down into open pores: every pore pixel shows the
first surface the beam meets below the cut face (searched up to 80 slices
deep), shaded by how steeply that surface faces the beam and by how much
signal can escape back out of the pore (narrow crevices dark, open walls
lit). Mean brightness per depth follows the curve measured on the real
stack. This is the v3 "see-through" renderer (microscope.render_surface).
"""
SEE_THROUGH_RULE = """Buried surfaces move ~0.2-0.3 px per slice of depth relative to the cut
face and brighten as the cut face approaches them. They are sharp and
detailed, as in the real images, so sharpness alone does not mark the cut
face; flip slices and watch whether a surface stays put.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="comp", choices=["comp", "uncomp", "pristine"],
                    help="which real region's measured appearance to match")
    ap.add_argument("--mat-kw", default=None, help="json kwargs for the material family")
    ap.add_argument("--optics", default="optics_v2.json")
    ap.add_argument("--family", default="catalyst",
                    help=f"material: {', '.join(sorted(materials.FAMILIES))}")
    ap.add_argument("--name", default=None)
    ap.add_argument("--porosity", type=float, default=None)
    ap.add_argument("--frames", type=int, default=118)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("material families:", ", ".join(sorted(materials.FAMILIES)))
        print("styles           : comp, uncomp, pristine")
        return

    cal = json.loads((HERE / args.optics).read_text())
    geo = json.loads((HERE / "geometry.json").read_text())
    if args.style not in cal:
        raise SystemExit(f"no calibration for {args.style}; run calibrate.py --all")

    if args.porosity is None:
        args.porosity = cal[args.style].get("material", cal[args.style]["standin"])["porosity"]
    if args.name is None:
        args.name = f"synth_{args.family}_{args.style}"
    dest = Path(args.out) if args.out else OUT_ROOT / f"team_review_{args.name}"

    res = build(args, cal, geo)
    copied, tj = write_package(args, res, dest)

    g = geo[args.style]
    opt = res["optics"]
    w = opt.weights or []
    (dest / "README.md").write_text(README.format(
        name=args.name, when=time.strftime("%Y-%m-%d"), style=args.style,
        family=args.family, true_por=res["true_porosity"],
        start_por=res["start_porosity"],
        start_err=res["start_porosity"] - res["true_porosity"],
        agree=res["start_agreement"],
        weights="\n".join(f"  - {i+1} slice{'s' if i else ' '} below: "
                          f"{x:.3f} of the cut-face weight"
                          for i, x in enumerate(w[:6])),
        drift=opt.drift, parallax=opt.parallax,
        shine_rule=SHINE_RULE[1.0 if abs(opt.drift) > 0.5 else 0.0],
        band_lo=res["band"][0], band_hi=res["band"][1], H=g["height"]))

    if cal[args.style].get("see_through"):
        # the template text describes the v2 depth-weighted renderer
        txt = (dest / "README.md").read_text()
        a = txt.index("Shine-through is simulated"); b = txt.index("- whole-image drift")
        txt = txt[:a] + SEE_THROUGH_TEXT + "\n" + txt[b:]
        a = txt.index("## Reviewing it")
        rule_start = txt.rfind("\n\n", 0, txt.rfind("\n\n", 0, a)) + 2
        txt = txt[:rule_start] + SEE_THROUGH_RULE + "\n" + txt[a:]
        (dest / "README.md").write_text(txt)
    print(f"\nwrote {dest}")
    print(f"  stack        {args.name}_full.tif  "
          f"({args.frames} x {g['height']} x {g['width']})")
    print(f"  start_masks  {args.frames} frames at porosity {res['start_porosity']:.4f}")
    print(f"  truth/masks  {args.frames} frames at porosity {res['true_porosity']:.4f}")
    print(f"  copied from the team package: {', '.join(copied) or 'nothing found'}")
    print(f"\n  start masks are off truth by "
          f"{res['start_porosity'] - res['true_porosity']:+.4f} porosity "
          f"and agree on {res['start_agreement']:.2%} of ROI pixels")


if __name__ == "__main__":
    main()
