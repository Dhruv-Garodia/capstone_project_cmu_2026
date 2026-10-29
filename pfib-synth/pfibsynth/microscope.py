#!/usr/bin/env python3
"""A model of the microscope, not a model of a photograph.

The image is formed by accumulating signal from the cut face and from the
material buried under it, because that is what a pFIB-SEM slice actually is.
Measured on the real stacks (measure_real.py), only a minority of the signal
in a pore pixel comes from the cut face at all.

Two motions, measured separately by path_scan.py on the real stacks:

  drift     the whole image moves +y per slice. Applied when frames are cut
            out of the volume (make_package), never inside the physics.
  parallax  material k slices below the cut face appears parallax*k px
            further along +y. This is the only thing that moves buried
            material relative to the cut face; drift moves both alike.

Measured drift 1.10 / 1.09 / 0.10 and parallax 0.24 / 0.19 / 0.20 px per
slice for comp / uncomp / pristine (path_scan.py; parallax_check.py gets
0.31 / 0.21 / 0.17 with a finer method). An earlier version merged both into
one `shear` with the sign reversed.

Signal for slice z, before noise:

    S(y,x) = sum_k  w_k * G_k[ em(z+k, y + parallax*k, x) ]     w_0 = 1
             w_k = a1 * exp(-(k-1)/lam)        for k >= 1
             G_k  = Gaussian blur, sigma = s0 + sk*k

G_k widening with depth is why shine-through looks soft and the cut face
looks sharp; it is the beam spreading as it goes deeper.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import ndimage as ndi

PORE, OUTSIDE, SOLID = 0, 128, 255


@dataclass
class Optics:
    """Everything the simulated microscope needs. Units are grey levels."""
    depth: int = 24            # slices below the cut face that contribute.
                               # Rendering only 8 made pore pixels with deep
                               # solid too dark, and the fitter answered with
                               # a weight on slice 8 as large as the cut face.
    a1: float = 0.55           # weight of the first buried slice
    lam: float = 4.3           # attenuation length, in slices
    drift: float = 0.0         # whole-image +y motion per slice (framing only)
    parallax: float = 0.2      # +y offset per slice of depth for buried material
    sigma0: float = 0.9        # blur of the cut face itself
    sigma_k: float = 0.45      # extra blur per slice of depth
    edge_gain: float = 0.35    # secondary-electron edge brightening
    edge_width: float = 1.6
    g_empty: float = 132.0     # grey where nothing is near the cut face
    gain: float = 60.0         # grey per unit of accumulated signal
    illum_amp: float = 16.0    # smooth across-slice illumination drift
    illum_scale: float = 0.33  # as a fraction of the frame
    curtain_amp: float = 3.5   # FIB curtaining, vertical streaks
    bright_frac: float = 0.0   # share of solid that is the bright phase
    bright_boost: float = 1.0  # its emission relative to ordinary solid
    dose: float = 220.0        # electrons per pixel; sets shot noise
    read_noise: float = 4.0    # detector read noise, grey levels
    weights: list | None = None  # explicit w_1..w_depth, overriding a1/lam

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------- rendering
def _shift_y(a: np.ndarray, dy: float) -> np.ndarray:
    """Shift a 2-D array along y by dy (may be fractional), edge-padded."""
    if dy == 0:
        return a
    if float(dy).is_integer():
        dy = int(dy)
        out = np.empty_like(a)
        if dy > 0:
            out[dy:] = a[:-dy]
            out[:dy] = a[0]
        else:
            out[:dy] = a[-dy:]
            out[dy:] = a[-1]
        return out
    return ndi.shift(a, (dy, 0), order=1, mode="nearest")


def _illumination(shape, opt: Optics, rng) -> np.ndarray:
    """Smooth multiplicative-looking drift, plus vertical curtaining."""
    ny, nx = shape
    cy = max(3.0, ny * opt.illum_scale)
    cx = max(3.0, nx * opt.illum_scale)
    f = rng.standard_normal((ny, nx)).astype(np.float32)
    f = ndi.gaussian_filter(f, (cy, cx), mode="reflect")
    # Centre before scaling. Smoothed over a third of the frame, the field is
    # nearly constant, so its mean is a random offset; left in, it shifted
    # the whole image by tens of grey levels depending on the seed, and the
    # calibration absorbed one seed's offset into the baseline.
    f -= f.mean()
    sd = f.std()
    f = (f / sd) if sd > 0 else f
    field = opt.illum_amp * f

    # curtaining: streaks constant down y, correlated over a few px in x
    c = rng.standard_normal(nx).astype(np.float32)
    c = ndi.gaussian_filter1d(c, 3.0, mode="reflect")
    c -= c.mean()
    sd = c.std()
    if sd > 0:
        field = field + opt.curtain_amp * (c / sd)[None, :]
    return field


def emission(solid: np.ndarray, bright: np.ndarray | None, opt: Optics):
    """Signal each voxel returns: 1 for solid, `bright_boost` for the bright
    phase, 0 for pore. Kept separate from `solid` so the answer key stays a
    two-class mask however many materials the structure contains."""
    em = solid.astype(np.float32)
    if bright is not None and opt.bright_boost != 1.0:
        em[bright] = opt.bright_boost
    return em


def render(solid: np.ndarray, opt: Optics, z_frames, rng,
           illum=None, bright=None, texture=None) -> np.ndarray:
    """Render the given slices. Returns float32 grey, unclipped, no noise.

    `texture`, if given, multiplies each solid voxel's emission: real solid
    is made of particles, and its grey varies within the solid on a 2-4 px
    scale; uniform emission renders solid as a flat plateau.

    `solid` must extend at least opt.depth slices past max(z_frames), and far
    enough in y to cover parallax*depth for the deepest contributing slice.
    """
    nz, ny, nx = solid.shape
    if opt.weights is not None:
        weights = [1.0] + list(opt.weights)[:opt.depth]
    else:
        weights = [1.0] + [opt.a1 * np.exp(-(k - 1) / opt.lam)
                           for k in range(1, opt.depth + 1)]

    em = emission(solid, bright, opt)
    if texture is not None:
        em *= texture
    out = np.zeros((len(z_frames), ny, nx), np.float32)

    # Blur as a cascade rather than a fresh filter per (slice, depth).
    # Gaussians compose -- blurring by sigma_a then by sqrt(sigma_b^2 -
    # sigma_a^2) is exactly a blur by sigma_b -- so each depth only needs a
    # small incremental blur of the previous one. A 118-slice stack at 11
    # depths is 1298 filters, most of them wide; this is 11 passes with
    # narrow kernels and one volume held at a time.
    work = em.copy()
    prev_sigma = 0.0
    for k, w in enumerate(weights):
        sigma_k = opt.sigma0 + opt.sigma_k * k
        step = np.sqrt(max(sigma_k ** 2 - prev_sigma ** 2, 0.0))
        if step > 1e-3:
            ndi.gaussian_filter(work, (0, step, step), output=work, mode="reflect")
        prev_sigma = sigma_k
        if w == 0:
            continue
        for i, z in enumerate(z_frames):
            zk = z + k
            if zk >= nz:
                continue
            # read row y + parallax*k: _shift_y(a, d)[y] = a[y - d]
            out[i] += w * _shift_y(work[zk], -opt.parallax * k)
    del work

    # edge brightening on the cut face only
    if opt.edge_gain:
        for i, z in enumerate(z_frames):
            gy, gx = np.gradient(ndi.gaussian_filter(em[z], opt.edge_width,
                                                     mode="reflect"))
            out[i] += opt.edge_gain * np.hypot(gy, gx)

    out *= opt.gain
    out += opt.g_empty

    if illum is None:
        illum = _illumination((ny, nx), opt, rng)
    out += illum[None]
    return out


def add_noise(grey: np.ndarray, opt: Optics, rng) -> np.ndarray:
    """Shot noise then read noise, then clip to 8-bit.

    The synth-to-real iteration log calls detector noise "the single biggest
    giveaway, and a surprise". It is applied after the signal is in grey
    levels so the dose sets how coarse the quantisation looks.
    """
    out = grey.astype(np.float32)
    if opt.dose > 0:
        lam = np.clip(out / 255.0 * opt.dose, 0.0, None)
        shot = rng.poisson(lam).astype(np.float32)
        out = shot * (255.0 / opt.dose)
    if opt.read_noise > 0:
        out = out + rng.normal(0.0, opt.read_noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------------------------------------------- statistics
def depth_to_solid(solid: np.ndarray, z: int, max_depth: int, parallax: float):
    """Slices down to the first solid below, per pixel, in slice z's frame.

    Matches measure_real.py's definition so the real and synthetic
    attenuation curves are the same measurement, not two similar ones.
    """
    face = solid[z]
    pore = ~face
    depth = np.zeros(face.shape, np.int16)
    found = ~pore
    for k in range(1, max_depth + 1):
        if z + k >= solid.shape[0]:
            break
        below = _shift_y(solid[z + k].astype(np.float32), -parallax * k) > 0.5
        newly = pore & ~found & below
        depth[newly] = k
        found |= newly
    depth[pore & ~found] = max_depth + 1
    depth[~pore] = 0
    return depth


def response_matrix(solid, z_frames, max_depth, parallax, roi=None, k_max=None,
                    sigma0=0.0, sigma_k=0.0):
    """M[d, k] = P(solid at depth k | first solid at depth d), over pore pixels.

    The measured shine-through curve is the mean grey of pore pixels grouped
    by how far below them the first solid sits. In the forward model that
    grey is sum_k w_k * (solid at depth k), so with M in hand the depth
    weights can be solved for directly instead of assumed to be a single
    exponential.

    With sigma0/sigma_k given, each buried layer is blurred exactly as
    render() blurs it, so M describes what the renderer will produce rather
    than an unblurred idealisation of it. The blur matters: by 24 slices
    down it is ~10 px wide, and leaving it out made fitted curves that the
    renderer could not reproduce.
    """
    K = max_depth
    KM = k_max or max_depth
    num = np.zeros((K + 2, KM + 1))
    cnt = np.zeros(K + 2)
    for i, z in enumerate(z_frames):
        if z + KM >= solid.shape[0]:
            continue
        keep = np.ones(solid[z].shape, bool) if roi is None else roi[i]
        d = depth_to_solid(solid, z, K, parallax)
        below = []
        for k in range(1, KM + 1):
            lay = solid[z + k].astype(np.float32)
            sig = sigma0 + sigma_k * k
            if sig > 0:
                lay = ndi.gaussian_filter(lay, sig, mode="reflect")
            below.append(_shift_y(lay, -parallax * k))
        for dd in range(1, K + 2):
            sel = (d == dd) & keep
            n = int(sel.sum())
            if n < 50:
                continue
            cnt[dd] += n
            for k in range(1, KM + 1):
                num[dd, k] += float(below[k - 1][sel].sum())
    M = np.zeros((K + 2, KM + 1))
    for dd in range(1, K + 2):
        if cnt[dd] > 0:
            M[dd] = num[dd] / cnt[dd]
    return M, cnt


def measure(image: np.ndarray, solid: np.ndarray, z_frames, opt: Optics,
            roi=None, max_depth=8) -> dict:
    """The same statistics measure_real.py pulls off the real stacks."""
    gp, gs, bg_spread, hf = [], [], [], []
    sat_lo = sat_hi = npx = 0
    depth_grey = {k: [] for k in range(1, max_depth + 2)}

    for i, z in enumerate(z_frames):
        img = image[i]
        face = solid[z]
        keep = np.ones(face.shape, bool) if roi is None else roi[i]
        gp.append(img[(~face) & keep].astype(np.float32))
        gs.append(img[face & keep].astype(np.float32))

        v = img[keep]
        npx += v.size
        sat_lo += int((v == 0).sum())
        sat_hi += int((v == 255).sum())

        win = min(251, max(21, min(img.shape) // 2 * 2 + 1))
        tot = ndi.uniform_filter(np.where(keep, img, 0).astype(np.float32), win)
        shr = ndi.uniform_filter(keep.astype(np.float32), win)
        b = np.where(shr > 0.02, tot / np.maximum(shr, 1e-6), np.nan)
        vals = b[np.isfinite(b) & keep]
        if vals.size:
            bg_spread.append(float(np.percentile(vals, 95) - np.percentile(vals, 5)))

        # grain: residual after a 3x3 median, inside solid away from edges.
        # 1.16 undoes the median's own attenuation of white noise.
        core = ndi.binary_erosion(face & keep, np.ones((5, 5)))
        if core.sum() > 500:
            res = img.astype(np.float32) - ndi.median_filter(img, size=3)
            hf.append(float(np.std(res[core]) * 1.16))

        d = depth_to_solid(solid, z, max_depth, opt.parallax)
        for k in range(1, max_depth + 2):
            sel = (d == k) & keep
            if sel.sum() > 50:
                depth_grey[k].append(float(img[sel].mean()))

    gp = np.concatenate(gp)
    gs = np.concatenate(gs)
    allv = np.concatenate([gp, gs])
    lo, hi = int(min(gp.min(), gs.min())), int(max(gp.max(), gs.max()))
    best_acc, best_t = 0.0, None
    for t in range(lo, hi + 1, 2):
        acc = 0.5 * ((gp < t).mean() + (gs >= t).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t

    curve = {k: float(np.mean(v)) for k, v in depth_grey.items() if v}
    return {
        "pore_mean": round(float(gp.mean()), 2), "pore_std": round(float(gp.std()), 2),
        "solid_mean": round(float(gs.mean()), 2), "solid_std": round(float(gs.std()), 2),
        "contrast_mean_gap": round(float(gs.mean() - gp.mean()), 2),
        "best_global_threshold": best_t,
        "best_global_balanced_acc": round(best_acc, 4),
        "saturated_either_frac": round((sat_lo + sat_hi) / npx, 5),
        "local_background_spread_p5_p95": round(float(np.mean(bg_spread)), 1),
        "hf_noise_solid": round(float(np.mean(hf)), 2) if hf else 0.0,
        "p10": round(float(np.percentile(allv, 10)), 1),
        "p50": round(float(np.percentile(allv, 50)), 1),
        "p90": round(float(np.percentile(allv, 90)), 1),
        "frac_above_239": round(float((allv >= 239).mean()), 5),
        "iqr": round(float(np.percentile(allv, 75) - np.percentile(allv, 25)), 1),
        "depth_curve": {str(k): round(v, 2) for k, v in curve.items()},
        "_curve_raw": curve,
    }


# ------------------------------------------------------- surface renderer (v3)
def first_hit(occ: np.ndarray, z: int, max_depth: int, parallax: float):
    """Depth along the beam to the first occupied voxel, and its row.

    The beam enters at the cut face of slice z and, k slices deeper, sits
    parallax*k px further along +y (the same convention as render()).
    Returns depth (0 = cut face occupied, max_depth+1 = nothing hit) and
    the y coordinate of the hit in slice z+depth, per pixel of slice z.
    """
    ny, nx = occ.shape[1:]
    depth = np.full((ny, nx), max_depth + 1, np.int16)
    depth[occ[z]] = 0
    open_ = ~occ[z]
    yy = np.arange(ny, dtype=np.float32)[:, None]
    for k in range(1, max_depth + 1):
        if z + k >= occ.shape[0] or not open_.any():
            break
        below = _shift_y(occ[z + k].astype(np.float32), -parallax * k) > 0.5
        newly = open_ & below
        depth[newly] = k
        open_ &= ~newly
    hit_y = np.clip(np.rint(yy + parallax * depth), 0, ny - 1).astype(np.int32)
    return depth, np.broadcast_to(hit_y, (ny, nx))


def render_surface(occ, albedo, opt: Optics, z_frames, curve, rng, illum=None,
                   smooth=1.8, cos_floor=0.18, secant=1.0, open_glow=None,
                   face_level=None, hit_albedo=True, defocus=None,
                   hole_floor=None, hole_len=8.0, face_smooth=0.0,
                   escape=0.0, cone_steps=(2.0, 4.0, 8.0, 16.0), cone_spread=0.6,
                   sigma0=None, shade_floor=0.0, face_map=None, approach=None,
                   open_hits=None, open_level=0.8):
    """See-through rendering: each pore pixel shows the surface it looks at.

    v1/v2 summed buried material as a soft, depth-weighted glow, which reads
    as a flat cut with haze underneath. The real stacks are not infiltrated,
    so the SEM looks down into open pores and images the pore walls and
    particles below the cut face, shaded by their orientation. Here:

      cut face (depth 0)  albedo, as before (grain and edge rim live there)
      pore, depth d>=1    the first surface along the beam, with SE yield
                          rising as the surface turns away from the beam
                          (secant law, 1/cos^secant, floored at cos_floor),
                          and a mean brightness per depth taken from
                          `curve` -- the measured grey-vs-depth curve.
                          Matching that curve is an input here, not a check.
      nothing within depth  `curve[-1]`, plus `open_glow` if given (light
                          escaping from walls not resolved as a hit)

    Which wall of a pore is visible follows from the parallax alone: rays
    slant +y going deeper, so the +y wall is hit and the -y wall undercut.
    Deeper surfaces are blurred more (sigma0 + sigma_k * d), by depth band.

    curve: array of length depth+2, mean grey for d = 0 .. depth+1 where
    curve[0] is unused (the face uses `face_level` * albedo).
    Returns float32 grey (n, ny, nx), illumination added, no noise.
    """
    D = opt.depth
    if sigma0 is not None:          # probe blur of the cut face, per style
        opt = Optics(**dict(opt.as_dict(), sigma0=sigma0))
    nz, ny, nx = occ.shape
    S = ndi.gaussian_filter(occ.astype(np.float32), smooth)
    # Escape: secondary electrons from a buried wall reach the detector only
    # if they get back out of the pore, i.e. up through the cut face (-z).
    # Voxel cone tracing: occupancy pre-blurred at the cone's footprint for
    # each step (a Gaussian mip chain), sampled along a few cones around -z;
    # transmittance is the product of (1 - occupancy) along each cone. Narrow
    # crevices come out dark and open walls lit -- the ambient-occlusion
    # approximation used by fast SEM simulators in place of tracing every SE.
    mips = {}
    if escape:
        for t in cone_steps:
            mips[t] = ndi.gaussian_filter(occ.astype(np.float32), max(0.7, cone_spread * t / 2))
        dirs = [np.array(v, float) for v in ((-1, 0, 0), (-1, cone_spread, 0), (-1, -cone_spread, 0),
                                             (-1, 0, cone_spread), (-1, 0, -cone_spread))]
        dirs = [v / np.linalg.norm(v) for v in dirs]
    b = np.array([1.0, opt.parallax, 0.0]); b /= np.linalg.norm(b)
    out = np.zeros((len(z_frames), ny, nx), np.float32)
    xx = np.broadcast_to(np.arange(nx)[None, :], (ny, nx))
    shade_all, depth_all, alb_all, open_all = [], [], [], []
    for i, z in enumerate(z_frames):
        d, hy = first_hit(occ, z, D, opt.parallax)
        zh = np.clip(z + d, 1, nz - 2)
        hy1 = np.clip(hy, 1, ny - 2); xh = np.clip(xx, 1, nx - 2)
        gz = S[zh + 1, hy1, xh] - S[zh - 1, hy1, xh]
        gy = S[zh, hy1 + 1, xh] - S[zh, hy1 - 1, xh]
        gx = S[zh, hy1, xh + 1] - S[zh, hy1, xh - 1]
        g = np.sqrt(gz * gz + gy * gy + gx * gx) + 1e-6
        # outward normal n = -grad S; incidence against the beam: n . (-b)
        cosi = np.clip((gz * b[0] + gy * b[1] + gx * b[2]) / g, cos_floor, 1.0)
        shade = (1.0 / cosi) ** secant
        if escape:
            nn = np.stack([-gz / g, -gy / g, -gx / g])          # outward normal
            p0 = np.stack([zh, hy1, xh]).astype(np.float32) + 1.5 * nn
            vis = np.zeros((ny, nx), np.float32)
            for v in dirs:
                tr = np.ones((ny, nx), np.float32)
                for t in cone_steps:
                    q = p0 + (v * t)[:, None, None]
                    qz = np.clip(np.rint(q[0]), 0, nz - 1).astype(np.int32)
                    qy = np.clip(np.rint(q[1]), 0, ny - 1).astype(np.int32)
                    qx = np.clip(np.rint(q[2]), 0, nx - 1).astype(np.int32)
                    above = q[0] < z - 0.5     # back above the cut face: open
                    tr *= np.where(above, 1.0, 1.0 - mips[t][qz, qy, qx])
                vis += tr
            vis /= len(dirs)
            shade = shade * (0.15 + 0.85 * vis) ** escape
        hit_alb = albedo[zh, hy1, xh] if hit_albedo else np.where(occ[zh, hy1, xh], 1.0, 0.0)
        if open_hits is not None:
            open_all.append(open_hits[zh, hy1, xh] & (d <= D))
        shade_all.append(shade); depth_all.append(d); alb_all.append(hit_alb.astype(np.float32))
    # normalise the geometric shading per depth so the mean grey at each depth
    # is curve[d] for a surface of albedo 1 (the layer's mean); the hit
    # surface's own albedo then scales it, so a dark support wall stays dark
    # (normalising shading x albedo pulled support walls up to the layer's
    # curve and lit the support holes brighter than the support)
    dstack = np.stack(depth_all); sstack = np.stack(shade_all)
    norm = np.ones(D + 2, np.float32)
    for k in range(1, D + 1):
        sel = dstack == k
        if sel.any():
            norm[k] = float(sstack[sel].mean())
    for i, z in enumerate(z_frames):
        d, sh = depth_all[i], shade_all[i] * alb_all[i]
        face = occ[z]
        img = np.zeros((ny, nx), np.float32)
        lv = face_level if face_level is not None else curve[1]
        fa = albedo[z] if not face_smooth else ndi.gaussian_filter(albedo[z], face_smooth)
        if face_map is not None:          # v4: physical cut-face yield, precomputed
            fa = face_map[i]
        layers = [(face, lv * fa, opt.sigma0)]
        for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, D)):
            m = (d >= lo) & (d <= hi)
            if m.any():
                rel = sh / norm[d]
                if shade_floor:
                    # deep hits keep detail and never go black: real deep
                    # surfaces stay visibly lit; strong escape occlusion
                    # alone drove some of them to 0
                    rel = shade_floor + (1.0 - shade_floor) * rel
                val = np.asarray(curve, np.float32)[d] * rel
                if approach:
                    # a surface a slice or two below the cut is under well
                    # under a pixel of material: its SE yield is close to
                    # the face's, so it brightens smoothly into the face
                    # instead of jumping when cut (real stacks: abrupt
                    # changes are darkenings ~11:1)
                    w = approach[0] * np.exp(-(d.astype(np.float32) - 1.0) / approach[1])
                    val = (1.0 - w) * val + w * lv * alb_all[i]
                if open_hits is not None:
                    # walls of a large open cavity (support voids): electrons
                    # escape freely, so no porous-medium depth falloff; only
                    # orientation and the cone-traced escape shade them
                    val = np.where(open_all[i], open_level * lv * rel, val)
                sk = opt.sigma_k if defocus is None else defocus
                layers.append((m, val, opt.sigma0 + sk * (lo + hi) / 2))
        none = d > D
        nv = np.full((ny, nx), curve[D + 1], np.float32)
        if hole_floor is not None and none.any():
            # nothing hit within `depth` slices: a deep hole. Its middle is
            # dark (hole_floor); towards its rim, walls just out of reach
            # still send light up, so it brightens to the deepest-hit level.
            # Flat 'nothing hit' patches were the most obviously fake thing.
            din = ndi.distance_transform_edt(none).astype(np.float32)
            nv = hole_floor + (curve[D] - hole_floor) * np.exp(-(din - 1.0) / hole_len)
        if open_glow is not None:
            nv = nv + open_glow(z)
        layers.append((none, nv, opt.sigma0 + (opt.sigma_k if defocus is None else defocus) * D))
        # back-to-front compositing, each band with its own defocus: a band's
        # blurred mask is its coverage, its colour is normalised by that
        # coverage, and it is laid over everything deeper. Blur crosses band
        # edges (normalised blending alone kept seams sharp, a mosaic) and no
        # gaps open between bands (summing blurred bands left black seams).
        img[:] = curve[D + 1]
        for m, val, sg in reversed(layers):
            a = ndi.gaussian_filter(m.astype(np.float32), sg)
            c = ndi.gaussian_filter(np.where(m, val, 0).astype(np.float32), sg) / np.maximum(a, 1e-3)
            img = img * (1.0 - a) + c * a
        # cut-face edge brightening, as in render()
        if opt.edge_gain:
            gy2, gx2 = np.gradient(ndi.gaussian_filter(face.astype(np.float32), opt.edge_width))
            img += opt.edge_gain * opt.gain * np.hypot(gy2, gx2)
        out[i] = img
    if illum is None:
        illum = _illumination((ny, nx), opt, rng)
    out += illum[None]
    return out, np.stack(depth_all)
