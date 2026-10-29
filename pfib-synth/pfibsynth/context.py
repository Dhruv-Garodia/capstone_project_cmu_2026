#!/usr/bin/env python3
"""Generated frame layout: ROI outline, cap above, support layer below.

v1 copied the ROI outline and every pixel outside it from real frames. That
made the frame look right but tied each synthetic package to one stretch of
one real stack, and the seam between pasted real context and synthetic
material was visible. Here nothing is copied. Each package draws its own
layout from distributions measured on the real stacks
(measure_boundaries.py -> real_boundaries.json, real_context_levels.json):

  top / bottom edge  median row, long-scale wander, mid-scale wiggle and
                     pixel-scale roughness, each with its measured size, and
                     a slow per-slice trend; different every seed
  support voids      rounded holes (superellipse tubes running roughly along
                     z, so they evolve slice to slice) in a dark support
                     layer. The layer above rests on them, so the bottom edge
                     arches over voids and drops into the gaps between them
  rims               a thin shell of high emission around each void; walls
                     running along z stack up through the depth response,
                     which is why rims glow in the real images
  cap                dense deposited layer, with occasional flat lens voids
  overlays           no-data rows at the top and trench side walls, for
                     styles whose real frames have them (pristine)

Everything except the overlays is built as voxels and rendered through the
same microscope as the material, so edges get the same blur, depth signal
and noise as everything else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
CL, CAP, MATRIX, VOID, LENS, FRINGE = 0, 1, 2, 3, 4, 5


def _field2(shape, corr, rng):
    """Unit-sd smooth random field over (z, x)."""
    f = rng.standard_normal(shape).astype(np.float32)
    f = ndi.gaussian_filter(f, corr, mode="reflect")
    f -= f.mean()
    return f / (f.std() + 1e-9)


@dataclass
class Layout:
    top: np.ndarray        # (nz, W) frame rows of the top edge
    bot: np.ndarray        # (nz, W) frame rows of the bottom edge (before voids)
    voids: list            # per void: dict of tube parameters
    black: np.ndarray      # (nz,) no-data rows at the top of each frame
    side_l: np.ndarray     # (nz,) dead columns on the left
    side_r: np.ndarray     # (nz,) dead columns on the right
    lenses: list           # cap lens voids
    fringe: np.ndarray     # (nz, W) how far material pokes up into the cap


def draw_layout(style, nz, H, W, rng, v7=False) -> Layout:
    rb = json.loads((HERE / "real_boundaries.json").read_text())[style]
    jit = lambda: float(rng.uniform(0.7, 1.3))
    z = np.arange(nz, dtype=np.float32)[:, None]
    trend = float(rng.uniform(-0.6, 0.1))

    def per_frame(f, smooth):
        """Scale so the within-frame sd, measured the way
        measure_boundaries.py measures it, is 1. Normalising over the
        whole volume left the within-frame wander at ~40% of real."""
        g = ndi.gaussian_filter1d(f, smooth, axis=1, mode="nearest") if smooth else f
        return f / (g.std(1).mean() + 1e-9)

    def edge(e, centre_shift=0.0, boost=1.0):
        s = rb[e]
        med = s["median"] * float(rng.uniform(0.95, 1.05)) + centre_shift
        long = per_frame(_field2((nz, W), (60.0, 120.0), rng), 60)
        mid = _field2((nz, W), (10.0, 22.0), rng)
        mid = mid - ndi.gaussian_filter1d(mid, 60, axis=1, mode="nearest")
        mid = per_frame(mid, 0)
        # pixel-scale roughness above ~4 px reads as a comb of single-column
        # spikes; the measured excess (pristine's bottom: 19 px) is pockets,
        # i.e. mid-scale shape, so it is moved into the mid component
        rough = min(s["rough_sd"], 4.0)
        mid_sd = float(np.sqrt(s["mid_sd"] ** 2 + s["rough_sd"] ** 2 - rough ** 2))
        f = (boost * s["long_sd"] * jit() * long + mid_sd * jit() * mid
             + (0.0 if v7 else rough * jit()) * _field2((nz, W), (4.0, 3.0), rng))
        return med + f + trend * (z - nz / 2)

    top = np.clip(edge("top"), 4, H * 0.6)
    # boost stays 1.0: after moving excess roughness into the mid scale the
    # wander already lands near the measured value (checked on layouts alone)
    # v7: the measured bottom wander includes the arches over support voids,
    # which v7 now draws as caverns; keeping it all made the bottom swing in
    # 60 px waves where real bottoms are fairly flat
    bot = edge("bottom", centre_shift=float(rng.uniform(0, 20)), boost=0.5 if v7 else 1.0)
    bot = np.maximum(bot, top + 60)

    # support voids: tubes along z with a slight tilt, below the bottom edge,
    # kept apart by walls of matrix -- overlapping superellipses merged into
    # blocky clumps that real supports never show
    voids = []
    med_bot = float(np.median(bot))
    gap = 18.0

    def clear(v):
        for u in voids:
            if (abs(u["cx"] - v["cx"]) < u["a"] + v["a"] + gap
                    and abs(u["cy"] - v["cy"]) < u["b"] + v["b"] + gap):
                return False
        return True

    def new_void(cx, cy, a, b):
        return dict(cx=cx, cy=cy, a=a, b=b, p=float(rng.uniform(1.9, 2.4) if v7 else rng.uniform(2.2, 3.6)),
                    lz=(float(rng.uniform(0.8, 2.0)) * a if v7 else None),
                    # lopsided outlines: low harmonics of the radius
                    harm=[(k, float(rng.normal(0, 0.09 / k ** 0.5)), float(rng.uniform(0, 2 * np.pi)))
                          for k in (1, 2, 3)],
                    vx=float(rng.normal(0, 0.06)), vy=float(rng.normal(0, 0.04)),
                    ga=float(rng.normal(0, 0.12)), z0=float(rng.uniform(-40, nz)))

    if v7:
        # finite cavities need more of them to fill the support: scatter
        # extra ones along z as well
        pass

    x = float(rng.uniform(-60, 20))
    while x < W + 100:
        a = float(rng.uniform(35, 120))
        b = float(a * rng.uniform(0.45, 0.85))
        xi = int(np.clip(x + a, 0, W - 1))
        local = float(np.median(bot[:, xi]))       # the support follows the layer
        v = new_void(x + a, local + b * float(rng.uniform(0.6, 1.5)), a, b)
        if clear(v):
            voids.append(v)
        x = v["cx"] + a + gap + float(rng.uniform(0, 60))
    if v7:
        for _ in range(int(W / 70 * max(1.0, nz / 150))):
            a = float(rng.uniform(30, 120)); b = float(a * rng.uniform(0.5, 0.85))
            cx = float(rng.uniform(0, W)); local = float(np.median(bot[:, int(np.clip(cx, 0, W - 1))]))
            v = new_void(cx, local + float(rng.uniform(0.3, 3.5)) * b, a, b)
            v["z0"] = float(rng.uniform(-a, nz + a))
            if all(abs(u["cx"] - v["cx"]) > u["a"] + v["a"] + gap or abs(u["cy"] - v["cy"]) > u["b"] + v["b"] + gap
                   or (u.get("lz") and abs(u["z0"] - v["z0"]) > u["lz"] + v["lz"]) for u in voids):
                voids.append(v)
    if v7:
        # caverns right under the layer: the support's open surface pores,
        # bridged by the layer, so their ceiling is the layer's underside
        # (dark, flat-topped holes along the bottom edge of real comp)
        for _ in range(int(W / 160 * max(1.0, nz / 150))):
            a = float(rng.uniform(25, 75)); b = float(a * rng.uniform(0.3, 0.6))
            cx = float(rng.uniform(0, W)); local = float(np.median(bot[:, int(np.clip(cx, 0, W - 1))]))
            v = new_void(cx, local + float(rng.uniform(-0.2, 0.4)) * b, a, b)
            v["z0"] = float(rng.uniform(-a, nz + a)); v["cavern"] = True
            voids.append(v)
    for _ in range(int(W / 60)):
        a = float(rng.uniform(30, 130)); b = float(a * rng.uniform(0.4, 0.8))
        cx = float(rng.uniform(0, W))
        local = float(np.median(bot[:, int(np.clip(cx, 0, W - 1))]))
        v = new_void(cx, local + float(rng.uniform(1.2, 4.0)) * b, a, b)
        if clear(v):
            voids.append(v)
    # material poking into the cap is solid 3-D stuff, persistent over
    # several slices; redrawn every slice it shone up through the cap all
    # along the edge as a 30 px halo (real: ~8 px)
    # v7: x-correlation 2.5 px drew a comb of needle spikes along the top
    # edge; real particles poking into the cap are 6-15 px wide lumps
    fringe = np.clip(_field2((nz, W), (6.0, 7.0 if v7 else 2.5), rng), 0, None) * float(rng.uniform(6, 12))

    blk = rb["black_top_rows"]
    if blk["frac_frames"] > 0.5:
        b0 = float(rng.uniform(0.8, 1.05)) * blk["mean"]
        black = np.clip(b0 + 6 * _field2((nz, 1), (25.0, 1.0), rng)[:, 0]
                        + float(rng.normal(0, 0.1)) * (np.arange(nz) - nz / 2), 0, H * 0.4)
    else:
        black = np.zeros(nz)
    sl = rb["dead_cols_left"]; sr = rb["dead_cols_right"]
    side_l = np.clip(sl * jit() + 4 * _field2((nz, 1), (20.0, 1.0), rng)[:, 0], 0, None) if sl > 10 else np.zeros(nz)
    side_r = np.clip(sr * jit() + 6 * _field2((nz, 1), (20.0, 1.0), rng)[:, 0], 0, None) if sr > 10 else np.zeros(nz)

    lenses = []
    lay_v7 = v7
    n_lens = int(rng.poisson(5 if style == "pristine" else 1.5) * nz / 100) + 1
    for _ in range(n_lens):
        lenses.append(dict(cx=float(rng.uniform(0, W)), z0=float(rng.uniform(0, nz)),
                           frac=float(rng.uniform(0.3, 0.8)), a=float(rng.uniform(25, 90)),
                           b=float(rng.uniform(1.5, 3.5)), lz=float(rng.uniform(6, 25)),
                           curv=float(rng.uniform(-0.004, 0.004))))
    L = Layout(top, bot, voids, black, side_l, side_r, lenses, fringe)
    L.v7 = v7
    return L


def void_mask(lay: Layout, zi, H, W):
    """Support voids at slice zi in frame rows, plus each void's upper surface."""
    m = np.zeros((H, W), bool)
    vtop = np.full(W, np.inf, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for v in lay.voids:
        t = zi - v["z0"]
        a = v["a"] * (1 + v["ga"] * np.tanh(t / 60))
        b = v["b"]
        if v.get("lz"):
            # a cavity, not an endless tube: it opens, grows and closes as
            # the cut passes through it
            q = 1.0 - (t / v["lz"]) ** 2
            if q <= 0.02:
                continue
            a = a * np.sqrt(q); b = b * np.sqrt(q)
        cx = v["cx"] + v["vx"] * t
        cy = v["cy"] + v["vy"] * t
        x0, x1 = int(max(0, cx - 1.3 * a - 1)), int(min(W, cx + 1.3 * a + 2))
        y0, y1 = int(max(0, cy - 1.3 * b - 1)), int(min(H, cy + 1.3 * b + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        ux = (xx[y0:y1, x0:x1] - cx) / a
        uy = (yy[y0:y1, x0:x1] - cy) / b
        r = (np.abs(ux) ** v["p"] + np.abs(uy) ** v["p"]) ** (1 / v["p"])
        th = np.arctan2(uy, ux)
        lim = 1.0 + sum(c * np.cos(k * th + ph) for k, c, ph in v.get("harm", []))
        inside = r <= lim
        m[y0:y1, x0:x1] |= inside
        col = inside.any(0)
        first = np.where(col, inside.argmax(0) + y0, np.inf)
        vtop[x0:x1] = np.minimum(vtop[x0:x1], first)
    return m, vtop


def void_hits(lay: Layout, zi, H, W, slant, zlen=1.0):
    """Where the beam meets the far wall of each support cavity (v7).

    In slice-frame coordinates the beam advances `slant` px down per slice
    (pore path = drift + parallax) and a cavity is an ellipsoid of half-
    length lz slices. Solving the ray-ellipsoid exit exactly gives, per
    cavity pixel, the depth k (slices) at which the beam hits the far wall
    and the cosine between that wall and the beam -- at any depth, unlike
    the voxel renderer, which gives up after `depth` slices and fell back to
    a flat dark 'hole'. Returns (k, cos) maps, NaN outside cavities.
    """
    K = np.full((H, W), np.nan, np.float32); C = np.full((H, W), np.nan, np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.array([1.0 / zlen, slant, 0.0]); d /= np.linalg.norm(d)        # (t, y, x) in px
    for v in lay.voids:
        if not v.get("lz"):
            continue
        t0 = zi - v["z0"]
        if abs(t0) >= v["lz"]:
            continue
        a = v["a"] * (1 + v["ga"] * np.tanh(t0 / 60)); b = v["b"]; lz = v["lz"]
        cx = v["cx"] + v["vx"] * t0; cy = v["cy"] + v["vy"] * t0
        x0, x1 = int(max(0, cx - a - 2)), int(min(W, cx + a + 3)); y0, y1 = int(max(0, cy - b - 2)), int(min(H, cy + b + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        Y = yy[y0:y1, x0:x1] - cy; X = xx[y0:y1, x0:x1] - cx
        s_rel = slant - v["vy"]
        A = 1 / lz ** 2 + s_rel ** 2 / b ** 2
        B = 2 * (t0 / lz ** 2 + Y * s_rel / b ** 2)
        Cq = t0 ** 2 / lz ** 2 + Y ** 2 / b ** 2 + X ** 2 / a ** 2 - 1
        inside = Cq < 0
        k = (-B + np.sqrt(np.clip(B * B - 4 * A * Cq, 0, None))) / (2 * A)
        te, ye = t0 + k, Y + s_rel * k
        # outward normal of the ellipsoid in px units (t scaled by zlen)
        n = np.stack([te / lz ** 2 * zlen, ye / b ** 2, X / a ** 2])
        n /= np.linalg.norm(n, axis=0) + 1e-9
        cos = np.clip(n[0] * d[0] + n[1] * d[1] + n[2] * d[2], 0.0, 1.0)  # wall faces back up the beam
        sub_k = K[y0:y1, x0:x1]; sub_c = C[y0:y1, x0:x1]
        sel = inside & (np.isnan(sub_k) | (k < sub_k))
        sub_k[sel] = k[sel]; sub_c[sel] = cos[sel]
    return K, C


def frame_zones(lay: Layout, zi, H, W):
    """Zone map for slice zi in frame coordinates, and the effective edges."""
    vm, vtop = void_mask(lay, zi, H, W)
    top = np.rint(lay.top[zi]).astype(int)
    # the layer rests on the support: it cannot reach into a void
    bot = lay.bot[zi] if getattr(lay, "v7", False) else np.minimum(lay.bot[zi], np.where(np.isfinite(vtop), vtop - 2, np.inf))
    bot = np.rint(np.maximum(bot, top + 40)).astype(int)
    yy = np.arange(H)[:, None]
    z = np.full((H, W), MATRIX, np.uint8)
    z[yy < top[None]] = CAP
    z[(yy >= top[None]) & (yy <= bot[None])] = CL
    # material pokes up into the cap: solid there stays material, pore is
    # filled by the cap (the renderer decides per voxel); outside the ROI
    fr = np.rint(lay.fringe[zi]).astype(int)
    z[(yy < top[None]) & (yy >= (top - fr)[None])] = FRINGE
    z[vm & (z == MATRIX)] = VOID
    # lens voids in the cap
    for L in lay.lenses:
        dz = (zi - L["z0"]) / L["lz"]
        if abs(dz) >= 1:
            continue
        cx = L["cx"]; a = L["a"] * np.sqrt(1 - dz * dz)
        xs = np.arange(max(0, int(cx - a)), min(W, int(cx + a) + 1))
        if xs.size == 0:
            continue
        # a smooth thin crescent, soft-edged in thickness
        ty = float(np.median(top)) * L["frac"] + L["curv"] * (xs - cx) ** 2
        th = L["b"] * np.sqrt(np.clip(1 - ((xs - cx) / max(a, 1)) ** 2, 0, 1))
        for x, y, t in zip(xs, ty, th):
            if t < 0.6:
                continue
            y0, y1 = int(round(y - t)), int(round(y + t)) + 1
            seg = z[max(0, y0):max(0, y1), x]
            seg[seg == CAP] = LENS
    return z, top, bot


def frame_overlays(lay: Layout, zi, H, W):
    """(no-data rows mask, side-wall mask) for slice zi."""
    blk = np.zeros((H, W), bool)
    blk[:int(round(lay.black[zi]))] = True
    side = np.zeros((H, W), bool)
    l, r = int(round(lay.side_l[zi])), int(round(lay.side_r[zi]))
    if l:
        side[:, :l] = True
    if r:
        side[:, W - r:] = True
    return blk, side
