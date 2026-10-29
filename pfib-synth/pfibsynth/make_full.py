#!/usr/bin/env python3
"""Full-length synthetic pFIB stacks (v5), same slice count as the real data.

One continuous 3-D volume per stack, so paging through slices behaves like
milling: pores open and close, buried material slides with the measured
parallax, the whole picture drifts by the measured drift. Built in z-chunks
so a 665-slice pristine stack fits in memory, with every field defined
globally (no seams between chunks):

  * Voronoi cells: global seeds, nearest-seed lookup by KD-tree
  * smooth random fields (warp, patchiness, composition, lumps): white
    noise on a coarse global lattice, filtered once, interpolated per chunk
  * rendering: each chunk carries `depth` extra slices below its frames;
    the per-depth shading normalisation is fixed from the first chunk and
    reused, and the tone map is fitted once on all frames

Output is a team-format package (stack, start_masks, truth/masks, launcher).

    python make_full.py --style pristine            # 665 slices
    python make_full.py --style comp --frames 354
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import context, materials, see_through, v4, paths
from microscope import Optics, render_surface, _illumination
import make_package as mp

HERE = Path(__file__).resolve().parent
REAL_SLICES = {"comp": 354, "uncomp": 410, "pristine": 665}
CH = 48            # frames per render chunk


# ---------------------------------------------------------------- fields
def _lin(arr, axis, pos):
    """Linear interpolation of arr along axis at fractional positions pos."""
    i0 = np.clip(np.floor(pos).astype(int), 0, arr.shape[axis] - 1)
    i1 = np.clip(i0 + 1, 0, arr.shape[axis] - 1)
    t = (pos - np.floor(pos)).astype(np.float32)
    sh = [1] * arr.ndim; sh[axis] = -1; t = t.reshape(sh)
    return np.take(arr, i0, axis) * (1 - t) + np.take(arr, i1, axis) * t


class Micro:
    """Fine 3-D texture fixed in the sample, cheap at any size: two random
    tiles (periods 61 and 67 voxels, incommensurate so no visible repeat),
    Gaussian-filtered with wrap, summed. The carbon agglomerates in a
    catalyst layer are built from ~2-4 px primary particles, which gives
    real solid its granular 'cauliflower' look; v5/v6 solid was flat inside
    with crisp outlines (scorecard: too little energy at 4-8 px)."""

    def __init__(self, rng, sig=(1.2, 1.2, 1.2)):
        self.t = []
        for n in (61, 67):
            g = rng.standard_normal((n, n, n)).astype(np.float32)
            g = ndi.gaussian_filter(g, sig, mode="wrap"); g /= g.std() + 1e-9
            self.t.append(g)

    def slab(self, z0, z1, ny, nx, L=1.0):
        out = np.zeros((z1 - z0, ny, nx), np.float32)
        for k, g in enumerate(self.t):
            n = g.shape[0]
            pz = np.arange(z0, z1) / L + 17 * k
            z0i = np.floor(pz).astype(int); t = (pz - z0i).astype(np.float32)[:, None, None]
            yi = (np.arange(ny) + 29 * k) % n; xi = (np.arange(nx) + 5 * k) % n
            a0 = g[z0i % n][:, yi][:, :, xi]; a1 = g[(z0i + 1) % n][:, yi][:, :, xi]
            out += (1 - t) * a0 + t * a1
        return out / np.sqrt(2.0)


class Field:
    """Continuous Gaussian random field over a big volume, sampled by slabs."""

    def __init__(self, shape, sigma, rng, octaves=1, gain=0.5, lac=2.5):
        self.shape = shape; self.levels = []
        for o in range(octaves):
            s = np.maximum(np.asarray(sigma, float) / lac ** o, 0.6)
            fac = np.maximum(np.floor(s / 2.0), 1).astype(int)
            small = tuple(int(np.ceil(n / f)) + 2 for n, f in zip(shape, fac))
            g = rng.standard_normal(small).astype(np.float32)
            ndi.gaussian_filter(g, s / fac, output=g, mode="wrap")
            g /= g.std() + 1e-9
            self.levels.append((fac, g, gain ** o))
        z = self.slab(0, min(shape[0], 8)); self.norm = float(z.std()) + 1e-9

    def at(self, pts):
        """Field values at (N, 3) points (trilinear on each lattice)."""
        out = 0.0
        for fac, g, amp in self.levels:
            out = out + amp * ndi.map_coordinates(g, (np.asarray(pts, float) / fac).T, order=1, mode="nearest")
        return out / self.norm

    def slab(self, z0, z1, norm=True):
        out = None
        for fac, g, amp in self.levels:
            pz = np.arange(z0, z1) / fac[0]
            lo = int(np.floor(pz.min())); hi = int(np.ceil(pz.max())) + 2
            sub = g[lo:hi]
            sub = _lin(sub, 2, np.arange(self.shape[2]) / fac[2])
            sub = _lin(sub, 1, np.arange(self.shape[1]) / fac[1])
            sub = _lin(sub, 0, pz - lo)
            out = amp * sub if out is None else out + amp * sub
        return out / self.norm if norm and hasattr(self, "norm") else out


# -------------------------------------------------------------- material
def sponge_full(shape, porosity, rng, radii=(2.5, 4.0, 6.0, 9.0, 14.0), vfrac=(0.15, 0.3, 0.3, 0.17, 0.08),
                zlen=1.0, y_el=1.15, rough=0.08, **kw):
    """Porous carbon as a sponge: the PORES are the objects (v7).

    Real catalyst layers at this resolution read as densely packed rounded
    holes with thin bright ligaments between them. The Voronoi foam chose
    whole cells as pore or solid, which drew irregular pore clusters and
    thick solid plateaus. Here the pore space is a Boolean model: overlapping
    rounded voids with a size distribution (`radii`, volume shares `vfrac`),
    stretched `zlen` along the milling axis and `y_el` in y, edges roughened
    by a small 3-D field. Porosity of a Boolean model is 1 - exp(-sum n V),
    which fixes the number of voids per class; the result is then trimmed to
    the exact target.
    """
    nz, ny, nx = shape
    vol = float(nz) * ny * nx
    lam = -np.log(1 - porosity)                            # sum n_i V_i
    samp = (1.0 / zlen, 1.0 / y_el, 1.0)
    centres = []
    for r, vf in zip(radii, vfrac):
        V = 4 / 3 * np.pi * r ** 3 * zlen * y_el
        n = rng.poisson(lam * vf * vol / V)
        centres.append((r, np.stack([rng.uniform(0, s_, n) for s_ in shape], 1).astype(np.float32)))
    pad = int(np.ceil(max(radii) * max(zlen, y_el) * 1.4)) + 2
    field = np.empty(shape, np.float16)
    noise = Micro(rng, (1.5 * zlen, 1.5, 1.5))
    for z0 in range(0, nz, 24):
        z1 = min(nz, z0 + 24); a0, a1 = max(0, z0 - pad), min(nz, z1 + pad)
        best = np.full((a1 - a0, ny, nx), np.inf, np.float32)
        for r, c in centres:
            m = np.ones((a1 - a0, ny, nx), bool)
            sel = c[(c[:, 0] >= a0) & (c[:, 0] < a1)]
            if len(sel) == 0:
                continue
            ci = np.clip(np.rint(sel).astype(int) - [a0, 0, 0], 0, [a1 - a0 - 1, ny - 1, nx - 1])
            m[ci[:, 0], ci[:, 1], ci[:, 2]] = False
            d = ndi.distance_transform_edt(m, sampling=samp).astype(np.float32) / r
            np.minimum(best, d, out=best)
        f = best[z0 - a0:z1 - a0] * (1.0 + rough * noise.slab(z0, z1, ny, nx, 1.0))
        field[z0:z1] = f.astype(np.float16)
    thr = float(np.quantile(field[::3, ::5, ::5].astype(np.float32), porosity))
    return field > thr                                     # solid = outside every void


def round_solid(solid, sig, ch=32):
    """Round the cells' flat facets: blur the solid in 3-D and re-threshold at
    the level that keeps the porosity. Cutting a flat facet that lies near
    the slice plane switches a whole patch at one slice, which shows up as a
    horizontal step in a reslice; real particles are rounded and appear and
    vanish gradually."""
    nz = solid.shape[0]; pad = int(np.ceil(3 * sig[0])) + 1
    sm = np.empty(solid.shape, np.float16)
    for z0 in range(0, nz, ch):
        z1 = min(nz, z0 + ch); a0 = max(0, z0 - pad); a1 = min(nz, z1 + pad)
        sm[z0:z1] = ndi.gaussian_filter(solid[a0:a1].astype(np.float32), sig)[z0 - a0:z1 - a0]
    frac = float(solid.mean())
    t = float(np.quantile(sm[::3, ::5, ::5].astype(np.float32), 1 - frac))
    return sm > t


def vorofoam_full(shape, porosity, rng, big=40.0, small=10.0, stretch=1.6, p_big=0.3, warp=4.0,
                  warp_sig=(4.0, 5.0, 4.0), patch=0.0, patch_sig=(10.0, 25.0, 22.0), zlen=1.0, zlen_big=None,
                  zlen_small=None, **kw):
    """gen2.vorofoam at any size: same construction, global seeds, chunked.

    zlen: slices per pixel-width of material. A slice is thinner than a
    pixel, so one pixel of material spans `zlen` slices; every length along
    the milling axis (cells, warp, patches) is multiplied by it.
    """
    # scale-dependent elongation (v7c): in the real stacks coarse structure
    # persists along the milling axis longer, and fine structure shorter,
    # than one stretch factor gives (band-pass correlation: coarse 8 slices
    # apart real 0.59 vs 0.48; fine 4 apart 0.14 vs 0.20), so the big cells
    # and the small cells get their own factor
    zb = zlen_big or zlen; zs_ = zlen_small or zlen
    nz, ny, nx = shape; vol = float(nz) * ny * nx / zlen
    samp = np.array([1.0 / zlen, 1.0 / stretch, 1.0])
    samp_b = np.array([1.0 / zb, 1.0 / stretch, 1.0]); samp_s = np.array([1.0 / zs_, 1.0 / stretch, 1.0])
    warp_sig = (warp_sig[0] * zlen,) + tuple(warp_sig[1:]); patch_sig = (patch_sig[0] * zlen,) + tuple(patch_sig[1:])
    nb = max(4, int(float(nz) * ny * nx / zb / big ** 3 * stretch)); ns = max(8, int(float(nz) * ny * nx / zs_ / small ** 3 * stretch))
    sb = np.stack([rng.uniform(0, s, nb) for s in shape], 1); ss = np.stack([rng.uniform(0, s, ns) for s in shape], 1)
    tb, ts = cKDTree(sb * samp_b), cKDTree(ss * samp_s)
    W = [Field(shape, warp_sig, rng, octaves=2) for _ in range(3)]
    big_keep = rng.random(nb) >= p_big
    r = rng.random(ns)
    if patch:
        pf = Field(shape, patch_sig, rng)
        fz = pf.at(ss)
        lg = np.log(r / (1 - r + 1e-9) + 1e-12)
        r = 1.0 / (1.0 + np.exp(-(lg + patch * np.asarray(fz))))
    Ls = np.empty(shape, np.int32); keepb = np.empty(shape, bool)
    yy, xx = np.meshgrid(np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing="ij")
    for z0 in range(0, nz, 16):
        z1 = min(nz, z0 + 16)
        dz, dy, dx = (warp * w.slab(z0, z1) for w in W)
        zc = np.arange(z0, z1, dtype=np.float32)[:, None, None]
        P0 = np.stack([(zc + 0.3 * zlen * dz).ravel(), (yy[None] + dy).ravel(), (xx[None] + dx).ravel()], 1)
        Ls[z0:z1] = ts.query(P0 * samp_s, workers=-1)[1].reshape(z1 - z0, ny, nx)
        keepb[z0:z1] = big_keep[tb.query(P0 * samp_b, workers=-1)[1]].reshape(z1 - z0, ny, nx)
        P = P0
        del P, dz, dy, dx
    cnt = np.bincount(Ls[keepb], minlength=ns).astype(np.float64)       # kept-big voxels per small cell
    order = np.argsort(r); csum = np.cumsum(cnt[order][::-1])[::-1]     # solid if r >= q
    target = (1 - porosity) * float(nz) * ny * nx                        # in voxels, not scaled
    k = int(np.searchsorted(-csum, -target))                            # first index with csum <= target
    q = r[order][min(k, ns - 1)]
    solid = keepb & (r >= q)[Ls]
    del Ls, keepb
    if kw.get("round"):
        solid = round_solid(solid, (kw["round"] * zlen, kw["round"], kw["round"]))
    return solid


def inclusions(shape, zone, rng, per_kpx=1.0, L=1.0):
    """Sparse large dense particles inside the layer (v7).

    Every real frame has one or two smooth, dense chunks 50-150 px across in
    the layer -- the striped region at the right of comp, the bright wedge in
    pristine -- milled flat, so curtaining shows on them as vertical
    stripes. v6 had none, which left the layer too uniform. Each is a warped
    ellipsoid (the warp makes rounded-polygon outlines, not discs), long
    along the milling axis in proportion to its size.
    """
    nz, ny, nx = shape
    inc = np.zeros(shape, bool)
    rows = np.where((zone == context.CL).any(axis=(0, 2)))[0]
    n = rng.poisson(per_kpx * nx / 1000.0 * max(1.0, nz / 120.0))
    for _ in range(n):
        ax = rng.uniform(25, 75); ay = ax * rng.uniform(0.5, 1.0); az = ax * L * rng.uniform(0.6, 1.2)
        cz = rng.uniform(0, nz); cy = rng.uniform(rows[0] + 0.3 * (rows[-1] - rows[0]), rows[-1] - 0.2 * (rows[-1] - rows[0]))
        cx = rng.uniform(0, nx)
        z0, z1 = int(max(0, cz - az * 1.6 - 2)), int(min(nz, cz + az * 1.6 + 3))     # box wide enough for the warp:
        y0, y1 = int(max(0, cy - ay * 1.6 - 2)), int(min(ny, cy + ay * 1.6 + 3))     # 1.3x clipped warped particles
        x0, x1 = int(max(0, cx - ax * 1.6 - 2)), int(min(nx, cx + ax * 1.6 + 3))     # into rectangles
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        g = rng.standard_normal((z1 - z0, y1 - y0, x1 - x0)).astype(np.float32)
        g = ndi.gaussian_filter(g, (max(2.0, az / 3), max(2.0, ay / 3), max(2.0, ax / 3))); g /= g.std() + 1e-9
        r = ((zz - cz) / az) ** 2 + ((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2
        r = r + 0.35 * g
        r[0], r[-1], r[:, 0], r[:, -1], r[:, :, 0], r[:, :, -1] = 9, 9, 9, 9, 9, 9      # never touch the box
        inc[z0:z1, y0:y1, x0:x1] |= r < 1.0
    return inc & (zone == context.CL)


# ------------------------------------------------------------ rendering
def render_full(style, P, zone, solid, opt, n, rng, lev, Hv, W, inc=None):
    t = json.loads((HERE / "optics.json").read_text())[style]["targets"]
    face = float(t["solid_mean"])
    D = int(P.get("depth", 80)); opt.depth = D
    curve = see_through.depth_curve(style, D, optics="optics_v3.json")
    if P.get("depth_gamma"):
        # flatter fall-off with depth: buried surfaces stay visible as
        # coherent shapes for more slices before they are cut
        c1 = float(curve[1]); g_ = float(P["depth_gamma"])
        curve = np.asarray(curve, np.float32).copy()
        curve[1:] = c1 * (np.clip(curve[1:], 1e-3, None) / c1) ** g_
    nz = zone.shape[0]; L = float(P.get("zlen", 1.0))
    # every 3-D field lives in the sample, so its milling-axis length is in
    # slices = pixels x L; v5 gave lumps a 2-slice life, which flickered
    comp = Field(zone.shape, (P["comp_sig"] * L, P["comp_sig"], P["comp_sig"]), rng, octaves=2)
    lump = Field((n, Hv, W), (15.0 * L, 15.0, 15.0), rng) if P.get("lump_amp") else None
    if P.get("curtain_amp"):
        k = np.abs(np.fft.fftfreq(W)) + 1.0 / W
        cz = Field((n, 1, W), (6.0 * L, 1.0, 1.0), rng)          # slow change between slices
        base = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal((1, W)), axis=1) * k ** -0.7
                                   * np.exp(-(k * P.get("curtain_w", 8.0)) ** 2), axis=1))
        base /= base.std() + 1e-9
    raw = np.empty((n, Hv, W), np.float16)
    micro = Micro(rng, (P.get("micro_sig", 1.2),) * 3) if P.get("micro_amp") else None
    if inc is not None:
        kk = np.abs(np.fft.fftfreq(W)) + 1.0 / W
        stripe = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal(W)) * kk ** -0.3 * np.exp(-(kk * 3.0) ** 2)))
        stripe = (stripe / (stripe.std() + 1e-9)).astype(np.float32)
        cz2 = Field((n, 1, W), (20.0 * L, 1.0, 6.0), rng)
    rk = dict(smooth=P["smooth"], secant=P["secant"], cos_floor=0.2, hit_albedo=True, defocus=0.03,
              hole_floor=P["hole_floor"] * opt.g_empty, hole_len=6.0, escape=P["escape"], sigma0=0.5,
              shade_floor=0.4, approach=P.get("approach"), open_level=P.get("open_level", 0.8))
    for z0 in range(0, n, CH):
        z1 = min(n, z0 + CH); ze = min(nz, z1 + D + 2); m = z1 - z0
        zn, sd = zone[z0:ze], solid[z0:ze]
        cl_solid = ((zn == context.CL) | (zn == context.FRINGE)) & sd
        occ = (zn == context.CAP) | (zn == context.MATRIX) | cl_solid
        a = np.exp(P["comp_amp"] * comp.slab(z0, ze)).astype(np.float32)
        a /= float(a[cl_solid].mean()) if cl_solid.any() else 1.0
        alb = np.zeros(zn.shape, np.float32); alb[cl_solid] = a[cl_solid]
        alb[zn == context.CAP] = lev["cap_mean"] / face; alb[zn == context.MATRIX] = lev["sub_p50"] / face
        fy = v4.face_yield(occ, list(range(m)), iv=(0.8 * L, 0.8, 0.8), esc=P["esc"], edge=P["edge"],
                          dy=P.get("edge_dy", 0.0))
        fy = fy * np.where(cl_solid[:m], a[:m], 1.0)
        if P.get("_gtex") is not None:
            gt = 1.0 + float(P.get("grain_contrast", 0.18)) * P["_gtex"][z0:ze]
            alb = np.where(cl_solid, alb * gt, alb)
            fy = np.where(cl_solid[:m], fy * gt[:m], fy)
            del gt
        if inc is not None:
            ic = inc[z0:ze]
            ia = float(P.get("incl_alb", 1.08))
            alb = np.where(ic & cl_solid, ia, alb)
            # dense and milled flat: curtaining ridges show as vertical stripes
            st = (1.0 + P.get("incl_stripe", 0.12) * stripe[None, None, :]
                  * (1.0 + 0.5 * cz2.slab(z0, z1)[:, :1, :]))
            fy = np.where(ic[:m], ia * st, fy)
        if P.get("micro_amp"):
            mt = micro.slab(z0, ze, Hv, W, L)
            if inc is not None:
                mt = np.where(inc[z0:ze], 0.0, mt)
            fy = fy * np.where(cl_solid[:m], 1.0 + P["micro_amp"] * mt[:m], 1.0)
            alb = np.where(cl_solid, alb * (1.0 + P["micro_amp"] * mt), alb)
            del mt
        del a
        if lump is not None:
            lp = 1.0 + P["lump_amp"] * lump.slab(z0, z1)
            fy = fy * (np.where(cl_solid[:m], lp, 1.0) if P.get("ctx3d") else lp)   # v7: the cap is homogeneous
        if P.get("curtain_amp"):
            c = base * (1.0 + 0.4 * cz.slab(z0, z1)[:, 0, :])
            pf = 1.0 - occ[:m].astype(np.float32); cs = np.cumsum(pf, 1)
            above = (cs - np.roll(cs, 40, 1)) / 40.0
            cur = 1.0 + P["curtain_amp"] * c[:, None, :] * (0.5 + np.clip(above, 0, 1))
            fy = np.where(cl_solid[:m], fy * cur, fy)
        oh = (zn == context.MATRIX) if P.get("ctx3d") else None
        g, dep = render_surface(occ, alb, opt, list(range(m)), curve, rng, illum=np.zeros((Hv, W), np.float32),
                                face_level=face, face_map=fy, open_hits=oh, **rk)
        gf = v4.funnel(g, occ[:m], {"mix": P["fun_mix"], "lam": P["fun_lam"], "rim": 180,
                                    "floor": P["fun_floor"], "soft": 1.0}, opt)
        if P.get("approach"):
            # the funnel is how a deep floor looks through the rim; a floor a
            # slice below the cut is not seen through a funnel, so it fades out
            b_, l_ = P["approach"]
            w = np.where(dep >= 1, b_ * np.exp(-(dep.astype(np.float32) - 1.0) / l_), 0.0)
            gf = gf + ndi.gaussian_filter(w, (0, 1.0, 1.0)) * (g - gf)
        g = gf; del dep, gf
        if P.get("halo"):
            # edge blooming: electrons leaving a cut edge also excite the
            # pore wall just inside the rim, so a pore brightens from its rim
            # inwards over a couple of pixels instead of stepping from the
            # face straight down to its floor
            h_, lam_ = P["halo"]
            for i_ in range(m):
                fc = occ[i_]
                if fc.all() or not fc.any():
                    continue
                d_, idx_ = ndi.distance_transform_edt(~fc, return_indices=True)
                near = g[i_][idx_[0], idx_[1]]                  # brightness of the nearest face pixel
                w_ = np.where(fc, 0.0, h_ * np.exp(-(d_ - 1.0) / lam_)).astype(np.float32)
                g[i_] = g[i_] + w_ * (near - g[i_])
        g = ndi.gaussian_filter(g, (0, P["sigma_y"], P["sigma_x"]))
        if P.get("flare"):
            # probe tail: a real beam is a sharp core plus a faint wide skirt
            # of scattered electrons, which lays a soft glow around bright
            # regions (point-spread function = core + a * wide Gaussian)
            a_, s_ = P["flare"]
            g = (1 - a_) * g + a_ * ndi.gaussian_filter(g, (0, s_, s_))
        raw[z0:z1] = g.astype(np.float16)
        print(f"  rendered frames {z0}-{z1}", flush=True)
        del g, occ, alb, fy, cl_solid
    # one monotone tone map for the whole stack: layer pixels -> real layer quantiles
    lay = zone[:n] == context.CL
    src = raw[lay][::31].astype(np.float32); ref = v4.real_hist(style)
    if P.get("sat_spread"):
        # the detector saturates AFTER the shot noise: where the true signal
        # is well over the 8-bit range, pixels sit flat at 255 even with
        # noise. Tone-mapping onto the clipped real histogram and then adding
        # noise pulled that pile down to ~240 and broke its flat plateaus,
        # so the saturated share is spread above 255 here, clipped later
        ref = ref.copy(); sat = ref >= 253.5
        ref[sat] = 254.0 + np.random.default_rng(5).exponential(float(P["sat_spread"]), int(sat.sum()))
    qs = np.linspace(0, 1, 512)
    xs, ys = np.quantile(src, qs), np.quantile(ref, qs)
    for z in range(n):
        raw[z] = np.interp(raw[z].astype(np.float32), xs, ys).astype(np.float16)
    return raw


def build(style, n, seed, family, porosity, mat_kw, zlen=None, width=None, round_px=None, approach=None,
          overrides=None, v6=False):
    cal = json.loads((HERE / "optics_v5.json").read_text())[style]
    opt = Optics(**cal["optics"]); P = dict(cal["v5"])
    sl = HERE / "slicing.json"           # v6: slice thickness and rounding, fitted to the real z-statistics
    if sl.exists():
        slc = json.loads(sl.read_text())[style]
        P.update({k: v for k, v in slc.items() if k in ("zlen", "round", "approach")})
        if not v6 and "v7" in slc:
            P.update(slc["v7"])
    if zlen is not None:
        P["zlen"] = zlen
    if round_px is not None:
        P["round"] = round_px
    if approach is not None:
        P["approach"] = approach
    if overrides:
        P.update(overrides)
    geo = json.loads((HERE / "geometry.json").read_text())[style]
    H, W = geo["height"], width or geo["width"]
    rng = np.random.default_rng(seed)
    lev = json.loads((HERE / "real_context_levels.json").read_text())[style]
    D = int(P.get("depth", 80)); nz = n + D + 2
    drift = float(opt.drift)
    margin = int(np.ceil(abs(opt.parallax) * (D + 2))) + 6
    travel = int(np.ceil(abs(drift) * (nz + 1)))
    Hv = H + travel + 2 * margin; base = margin + travel
    # the frame window moves by drift*z exactly: cut at floor, then shift the
    # remaining fraction by interpolation, so the picture glides instead of
    # jumping a whole pixel every ~10 slices
    sf = np.array([base - drift * z for z in range(nz)])
    st = np.floor(sf).astype(int); fr = (sf - st).astype(np.float32)
    print(f"{style}: {n} frames, volume {(nz, Hv, W)} = {nz * Hv * W / 1e9:.2f} G voxels", flush=True)
    t0 = time.time()
    lay = context.draw_layout(style, nz, H, W, rng, v7=bool(P.get("ctx3d")))
    zone = np.empty((nz, Hv, W), np.uint8)
    for z in range(nz):
        zf, _, _ = context.frame_zones(lay, z, H, W)
        zone[z, :st[z]] = context.CAP; zone[z, st[z]:st[z] + H] = zf; zone[z, st[z] + H:] = context.MATRIX
    print(f"  layout {time.time() - t0:.0f}s", flush=True); t0 = time.time()
    if family == "sponge":
        sk = dict(P.get("sponge", {})); sk["zlen"] = float(P.get("zlen", 1.0))
        pf = 1 - (1 - porosity) / (1 - float(P.get("micropore", 0.0)))
        solid = sponge_full(zone.shape, pf, rng, **sk)
    elif family == "v5":
        vk = dict(P["voro"]); vk["warp_sig"] = tuple(vk["warp_sig"]); vk["zlen"] = float(P.get("zlen", 1.0)); vk["round"] = P.get("round")
        pf = 1 - (1 - porosity) / (1 - float(P.get("micropore", 0.0)))
        solid = vorofoam_full(zone.shape, pf, rng, **vk)
    else:
        # other families: build on a grid compressed along the milling axis and
        # stretch it back, so each keeps its own 3-D shape at the instrument's
        # slice thickness (zlen of the style it is rendered in)
        L = float(P.get("zlen", 1.0))
        nzc = int(np.ceil(nz / L)) + 2
        sc, tc = materials.build(family, (nzc, Hv, W), porosity, seed=seed, with_texture=True, **(mat_kw or {}))
        if L == 1.0:
            solid = sc[:nz]; gtex = tc[:nz] if isinstance(tc, np.ndarray) and tc.shape == sc.shape else None
        else:
            pos = np.arange(nz) / L; i0 = np.floor(pos).astype(int); t = (pos - i0).astype(np.float32)
            solid = np.empty((nz, Hv, W), bool)
            for z in range(nz):
                solid[z] = ((1 - t[z]) * sc[i0[z]] + t[z] * sc[i0[z] + 1]) >= 0.5
            gtex = (tc[np.rint(pos).astype(int)] if isinstance(tc, np.ndarray) and tc.shape == sc.shape else None)
        del sc, tc
    print(f"  material {family} {time.time() - t0:.0f}s, porosity {1 - solid.mean():.3f}", flush=True); t0 = time.time()
    cl_family = family in ("v5", "sponge")       # micropores and dense particles are catalyst-layer features
    if P.get("macro_per_kpx") and cl_family:
        # sparse macro-voids (v7d): real layers are a fine porous network with
        # a few large caverns; looking down one, you watch the structure at
        # its far end come forward for many slices. Making every pore larger
        # gave that look but too much contrast in every patch, so they are
        # discrete objects: warped ellipsoids, long along the milling axis,
        # carved out of the layer only
        lo_, hi_ = P.get("macro_r", [18.0, 40.0])
        rows_ = np.where((zone == context.CL).any(axis=(0, 2)))[0]
        nm = rng.poisson(P["macro_per_kpx"] * W / 1000.0 * max(1.0, nz / 120.0))
        for _ in range(nm):
            ax_ = rng.uniform(lo_, hi_); ay_ = ax_ * rng.uniform(0.6, 1.1); az_ = ax_ * float(P.get("zlen", 1.0)) * rng.uniform(1.0, 2.0)
            cz_, cx_ = rng.uniform(0, nz), rng.uniform(0, W)
            cy_ = rng.uniform(rows_[0] + 0.15 * (rows_[-1] - rows_[0]), rows_[-1] - 0.15 * (rows_[-1] - rows_[0]))
            z0_, z1_ = int(max(0, cz_ - 1.6 * az_)), min(nz, int(cz_ + 1.6 * az_) + 1)
            y0_, y1_ = int(max(0, cy_ - 1.6 * ay_)), min(Hv, int(cy_ + 1.6 * ay_) + 1)
            x0_, x1_ = int(max(0, cx_ - 1.6 * ax_)), min(W, int(cx_ + 1.6 * ax_) + 1)
            if z1_ <= z0_ or y1_ <= y0_ or x1_ <= x0_: continue
            zz_, yy_, xx_ = np.meshgrid(np.arange(z0_, z1_), np.arange(y0_, y1_), np.arange(x0_, x1_), indexing="ij")
            g_ = ndi.gaussian_filter(rng.standard_normal(zz_.shape).astype(np.float32), (max(2, az_ / 3), max(2, ay_ / 3), max(2, ax_ / 3)))
            g_ /= g_.std() + 1e-9
            r_ = ((zz_ - cz_) / az_) ** 2 + ((yy_ - cy_) / ay_) ** 2 + ((xx_ - cx_) / ax_) ** 2 + 0.3 * g_
            hole_ = (r_ < 1.0) & (zone[z0_:z1_, y0_:y1_, x0_:x1_] == context.CL)
            solid[z0_:z1_, y0_:y1_, x0_:x1_] &= ~hole_
        print(f"  macro-voids: {nm}, layer porosity now {1 - solid[zone == context.CL].mean():.3f}", flush=True)
    if P.get("micropore") and cl_family:
        # primary pores inside agglomerates, as voxels in the truth (v7).
        # Real 'solid-looking' regions carry ~2% persistent dark specks and
        # persistent 1-2 px texture; v6 painted texture onto solid instead,
        # which the truth did not know about
        f = float(P["micropore"]); mps = P.get("micropore_sig", 0.9); mpf = Micro(rng, tuple(mps) if isinstance(mps, (list, tuple)) else (mps,) * 3)
        from scipy.stats import norm
        thr = float(norm.ppf(1 - f))
        lay_ = zone == context.CL
        for z0_ in range(0, nz, 32):
            z1_ = min(nz, z0_ + 32)
            hole = mpf.slab(z0_, z1_, Hv, W, float(P.get("zlen", 1.0))) > thr
            solid[z0_:z1_] &= ~(hole & lay_[z0_:z1_])
        print(f"  micropores: porosity now {1 - solid[zone == context.CL].mean():.3f}", flush=True)
    inc = None
    if P.get("incl_per_kpx") and cl_family:
        inc = inclusions(zone.shape, zone, rng, P["incl_per_kpx"], float(P.get("zlen", 1.0)))
        # dense but not solid: fine pores at a quarter of the layer's
        # micro-porosity (the striped region in real comp is textured)
        mpi = Micro(rng, (3.0, 1.2, 1.2))
        for z0_ in range(0, nz, 32):
            z1_ = min(nz, z0_ + 32)
            ih = mpi.slab(z0_, z1_, Hv, W, float(P.get("zlen", 1.0))) > 1.75
            solid[z0_:z1_] |= inc[z0_:z1_] & ~ih
        print(f"  inclusions: {inc.mean():.4f} of the volume", flush=True)
    if family not in ("v5", "sponge") and gtex is not None and P.get("grain_contrast", 0.18) and family == "faceted":
        P["_gtex"] = gtex
    raw = render_full(style, P, zone, solid, opt, n, rng, lev, Hv, W, inc=inc)
    P.pop("_gtex", None)
    print(f"  render {time.time() - t0:.0f}s", flush=True)

    if P.get("curtain_illum") is not None:     # v7: measured stripe strength in cap and support (~8)
        opt.curtain_amp = float(P["curtain_illum"])
    cur_c = None
    if P.get("curtain_regional"):
        cur_amp = float(opt.curtain_amp); opt.curtain_amp = 0.0           # stripes drawn per region below
        cur_capk, cur_rate = float(P["curtain_regional"][0]), float(P["curtain_regional"][1])
        cur_c = []
        for _ in range(2):
            c_ = ndi.gaussian_filter1d(rng.standard_normal(W).astype(np.float32), 3.0, mode="reflect"); c_ -= c_.mean(); cur_c.append(c_ / (c_.std() + 1e-9))
    illum = _illumination((H, W), opt, rng)
    image = np.empty((n, H, W), np.uint8); truth = np.full((n, H, W), 128, np.uint8)
    roi_all = np.zeros((n, H, W), bool)
    yy = np.arange(H, dtype=np.float32)[:, None]; tone = None
    gsd, gref = P["grain"], P["grain_ref"]
    jit = (rng.normal(0, 1, (n, 2)) * np.array(P["jitter"], np.float32)) if P.get("jitter") else None
    nsig = tuple(P["noise_sig"]) if P.get("noise_sig") else None
    nfix = {0.42: 1.025, 0.45: 1.03}.get(max(nsig), 1.03) if nsig else 1.0   # keep median-residual sd
    for i in range(n):
        if P.get("cubic_frame"):
            # cubic resampling: linear interpolation blurred frames by an
            # amount that cycled with the drift fraction, so sharpness pulsed
            # from slice to slice (a temporal tell)
            w_ = raw[i, max(0, st[i] - 3):st[i] + H + 4].astype(np.float32); o_ = st[i] - max(0, st[i] - 3)
            f = ndi.shift(w_, (-fr[i], 0), order=3, mode="nearest")[o_:o_ + H]
        else:
            a0 = raw[i, st[i]:st[i] + H].astype(np.float32); a1 = raw[i, st[i] + 1:st[i] + 1 + H].astype(np.float32)
            f = (1 - fr[i]) * a0 + fr[i] * a1
        zf, top, bot = context.frame_zones(lay, i, H, W)
        vd = zf == context.VOID
        if vd.any() and P.get("ctx3d"):
            # v7: cavity interiors from the exact ray-ellipsoid exit (secant
            # shading, weak fade with depth, depth blur), and a thin bright
            # rim where the cut face meets the cavity (SE edge effect)
            mlev = float(np.median(f[zf == context.MATRIX])) if (zf == context.MATRIX).any() else float(f.mean())
            Kh, Ch = context.void_hits(lay, i, H, W, float(opt.drift + opt.parallax), float(P.get("zlen", 1.0)))
            hv = vd & ~np.isnan(Kh)
            if hv.any():
                sh = (1.0 / np.clip(Ch[hv], 0.25, 1.0)) ** (0.5 * float(P["secant"]))
                sh = sh / np.median(sh)
                val = np.full((H, W), np.nan, np.float32)
                val[hv] = mlev * float(P.get("cav_level", 0.8)) * sh * np.exp(-Kh[hv] / float(P.get("cav_fade", 250.0)))
                filled = np.where(hv, val, mlev * 0.6)
                filled = ndi.gaussian_filter(filled, 1.5)
                f = np.where(vd, filled, f)
            dout = ndi.distance_transform_edt(~vd)
            rim = (~vd) & (dout < 4)
            f[rim] += float(P.get("cav_rim", 0.5)) * mlev * np.exp(-(dout[rim] - 1) / 0.9)
            # the overhang over a cavity glows: electrons escape out of the
            # thin lip of face above it into the void (same physics as the
            # pore top-edge cue); real cavities show a broad bright band
            # under their top edge fading downwards
            run = np.zeros((H, W), np.float32); acc = np.zeros(W, np.float32)
            for yy_ in range(H):
                acc = np.where(vd[yy_], acc + 1, 0); run[yy_] = acc
            f = np.where(vd, f + float(P.get("cav_glow", 0.7)) * mlev * np.exp(-(run - 1) / float(P.get("cav_glow_len", 6.0))), f)
        if vd.any() and not P.get("ctx3d"):
            mlev = float(np.median(f[zf == context.MATRIX])) if (zf == context.MATRIX).any() else float(f.mean())
            din = ndi.distance_transform_edt(vd); dout = ndi.distance_transform_edt(~vd)
            inner = 0.62 * mlev + 0.22 * mlev * np.exp(-(din - 1) / 9.0)
            f[vd] = inner[vd]
            rim = (zf == context.MATRIX) & (dout < 6)
            f[rim] += 0.45 * mlev * np.exp(-(dout[rim] - 1) / 1.8)
            band = (dout < 4) | ((din < 4) & vd)
            f = np.where(band, ndi.gaussian_filter(f, 1.2), f)
        below = ndi.gaussian_filter((yy > bot[None]).astype(np.float32), 2.0)
        above = ndi.gaussian_filter((zf == context.CAP).astype(np.float32), 2.0)
        if tone is None:
            sb = f[(yy > bot[None] + 60) & (zf != context.LENS)]
            r5, r50 = np.percentile(sb, [5, 50])
            a_s = min((lev["sub_p50"] - lev["sub_p5"]) / max(r50 - r5, 1.0), 2.5)
            ca = f[(yy < top[None] - 6) & (zf == context.CAP)]
            tone = (a_s, lev["sub_p5"] - a_s * r5, lev["cap_mean"] - float(ca.mean()))
        a_s, b_s, b_c = tone
        f = f + below * ((a_s * f + b_s) - f) + above * b_c
        if P.get("sup_lift"):        # v7: support level (real support median ~75, highlights ~133)
            f = f * (1.0 + (float(P["sup_lift"]) - 1.0) * below)
        blk, side = context.frame_overlays(lay, i, H, W)
        if side.any():
            f = np.where(side, lev["cap_mean"] * 0.8, f)
        f = f + illum
        if cur_c is not None:
            # curtaining forms under the porous layer and in the support, not
            # in the milled-smooth cap (real cap stripes: corr 0.35 between
            # slices; support: 0.85 at 16 slices); it evolves slowly
            th = cur_rate * i
            cc_ = np.cos(th) * cur_c[0] + np.sin(th) * cur_c[1]
            capw = ndi.gaussian_filter((zf == context.CAP).astype(np.float32), 3.0)
            f = f + cur_amp * cc_[None, :] * (1.0 - (1.0 - cur_capk) * capw)
        if jit is not None:
            # stage/beam position noise: each frame lands a fraction of a
            # pixel off the smooth drift, in y and x (real: frame-to-frame
            # residual sd ~0.3 px; v7 synthetic ~0.02). Image, truth and ROI
            # move together, so the truth stays pixel-exact.
            f = ndi.shift(f, jit[i], order=3 if P.get("cubic_frame") else 1, mode="nearest")
        nz_ = rng.standard_normal(f.shape).astype(np.float32)
        if nsig:
            # the detector/scan correlates noise along the fast-scan axis
            # (real comp/uncomp: x; pristine: y); white noise left the
            # median residual anti-correlated where real is not
            nz_ = ndi.gaussian_filter(nz_, nsig); nz_ *= nfix / (nz_.std() + 1e-9)
        img = np.clip(f + gsd * nz_ * np.sqrt(np.clip(f, 1, 255) / gref), 0, 255).astype(np.uint8)
        img[blk] = 0
        image[i] = img
        roi = (zf == context.CL) & ~blk & ~side
        sol = ((1 - fr[i]) * solid[i, st[i]:st[i] + H] + fr[i] * solid[i, st[i] + 1:st[i] + 1 + H])
        if jit is not None:
            roi = ndi.shift(roi.astype(np.float32), jit[i], order=1, mode="nearest") >= 0.5
            sol = ndi.shift(sol.astype(np.float32), jit[i], order=1, mode="nearest")
        sol = sol >= 0.5
        roi_all[i] = roi
        truth[i][roi] = np.where(sol[roi], 255, 0)
    del raw, solid, zone
    tp = float((truth == 0).sum() / ((truth == 0).sum() + (truth == 255).sum()))
    lab = json.loads((HERE / "real_labeller.json").read_text())[style]
    start = np.full_like(truth, 128)
    for z0 in range(0, n, 60):   # labeller per block (memory); 251 px window is in-plane only
        z1 = min(n, z0 + 60)
        pred = mp.adaptive_fixed(image[z0:z1], roi_all[z0:z1], lab["level"], lab["adapt"], lab["win"])
        start[z0:z1] = np.where(roi_all[z0:z1], np.where(pred, 0, 255), 128)
    sp = float((start == 0).sum() / ((start == 0).sum() + (start == 255).sum()))
    agree = float((start[roi_all] == truth[roi_all]).mean())
    print(f"  true porosity {tp:.4f}; rule start masks {sp:.4f} ({sp - tp:+.4f}), agreement {agree:.3f}", flush=True)
    rows = np.where(roi_all.any(axis=(0, 2)))[0]
    return dict(image=image, truth=truth, start=start, roi=roi_all, optics=opt, true_porosity=tp,
                start_porosity=sp, start_agreement=agree, start_level=lab["level"],
                stats={"excluded_fraction": round(float(1 - roi_all.mean()), 4)},
                band=(int(rows[0]), int(rows[-1]) + 1),
                top=np.array([int(np.where(r.any(1))[0][0]) if r.any() else 0 for r in roi_all]),
                bot=np.array([int(np.where(r.any(1))[0][-1]) if r.any() else 0 for r in roi_all]),
                layout_seed=seed, zlen=float(P.get("zlen", 1.0)), round_px=P.get("round"))


README = """# Full-length synthetic stack -- {name}

Not real data. A simulated pFIB-SEM stack with the same frame size and the
same number of slices ({n}) as the real {style} stack, rendered from ONE
continuous 3-D volume, so paging through the slices behaves like milling:
pores open and close, buried surfaces slide with the measured parallax,
and the picture drifts by the measured drift ({drift:+.2f} px/slice).

- `{stack}`: the image stack ({n} x {H} x {W}, 8-bit)
- `truth/masks/slice_NNNN.png`: exact answer for every slice
  (0 = pore, 255 = solid, 128 = outside the layer)
- `start_masks/`: what the threshold rule behind our labels gives
  (truth-blind), so you can see where shine-through fools it

True porosity inside the layer: **{tp:.4f}**; rule start masks: {sp:.4f}
({err:+.4f}), pixel agreement {agree:.1%}.

Pore pixels show the surface the beam hits below the cut face (see-through
renderer, v5: Voronoi-foam material, secant + escape shading, composition
contrast, measured noise). Generated by `pfib-synth/make_full.py`.
Open with `start_annotating.py` like the review packages.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, choices=list(REAL_SLICES))
    ap.add_argument("--family", default="v5")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--porosity", type=float, default=None)
    ap.add_argument("--mat-kw", default=None)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--name", default=None)
    ap.add_argument("--zlen", type=float, default=None, help="slices per pixel-width of material")
    ap.add_argument("--v6", action="store_true", help="v6 look: skip the v7 overrides in slicing.json")
    ap.add_argument("--round", type=float, default=None, help="facet rounding, px")
    ap.add_argument("--width", type=int, default=None, help="narrower frame, for quick tests")
    ap.add_argument("--out", default=None)
    ap.add_argument("--override", default=None, help="JSON dict of render settings to override (keys of slicing.json v7 / optics_v5.json v5)")
    a = ap.parse_args()
    n = a.frames or REAL_SLICES[a.style]
    por = a.porosity or {"comp": 0.417, "uncomp": 0.558, "pristine": 0.515}[a.style]
    name = a.name or f"synth_full_{a.style}"
    dest = Path(a.out) if a.out else paths.OUT / f"full_{name}"
    res = build(a.style, n, a.seed, a.family, por, json.loads(a.mat_kw) if a.mat_kw else None,
                zlen=a.zlen, width=a.width, round_px=a.round, v6=a.v6,
                overrides=json.loads(a.override) if a.override else None)

    class A: pass
    args = A(); args.frames = n; args.name = name; args.force = True; args.style = a.style
    args.family = a.family; args.porosity = por; args.seed = a.seed
    mp.write_package(args, res, dest)
    geo = json.loads((HERE / "geometry.json").read_text())[a.style]
    (dest / "README.md").write_text(README.format(
        name=name, n=n, style=a.style, drift=res["optics"].drift, stack=f"{name}_full.tif",
        H=geo["height"], W=geo["width"], tp=res["true_porosity"], sp=res["start_porosity"],
        err=res["start_porosity"] - res["true_porosity"], agree=res["start_agreement"]))
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
