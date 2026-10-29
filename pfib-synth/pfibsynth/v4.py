#!/usr/bin/env python3
"""v4: material and cut-face physics aimed at being indistinguishable.

Two changes over v3, both leaving the voxel truth exact:

1. Material: 'web' -- a hierarchical sponge built from the ZERO SET of a
   band-limited random field. |f| < w is a network of thin walls (sheets)
   around closed cavities, the shape of real uncomp; with the field
   stretched along y and a fine octave it becomes the ragged fibrous
   network of real comp. A coarse field modulates density (agglomerates)
   and carves macropores. Spheres gave beads, plain level sets gave slabs.

2. Cut-face yield from the interaction volume, not a flat plateau:
     Y = (solid within the interaction volume below the face)
         x (1 + edge * escape),   escape = open space within SE range
   so dense material is mid-grey, thin walls and edges glow, and every
   2-4 px of geometry produces relief. Real cut faces are never flat.

Plus: fine normals for buried surfaces (smooth ~1.2 px instead of a 5 px
blur), and a final monotone tone map onto the real ROI histogram (cannot
move an edge: it is the same map for every pixel).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
import materials
from materials import _grf, _level_for_porosity
from microscope import Optics, render_surface, add_noise

HERE = Path(__file__).resolve().parent


def web(shape, porosity, rng, wall=(3.0, 4.0, 3.0), stretch_y=1.3, fine=0.35,
        agg=(12.0, 20.0, 16.0), agg_amp=0.6, macro_frac=0.15, macro_amp=1.5,
        mix=0.6, **kw):
    """Sheet/ligament sponge from the zero set of a random field."""
    cz, cy, cx = wall
    f = _grf(shape, (cz, cy * stretch_y, cx), rng, octaves=2, lacunarity=(1.0, 2.2, 2.2), gain=0.35)
    f /= f.std() + 1e-9
    sheet = -np.abs(f)                      # 0 on the walls, negative in cavities
    blob = _grf(shape, (cz * 1.5, cy * stretch_y * 1.5, cx * 1.5), rng, octaves=1)
    blob /= blob.std() + 1e-9
    field = mix * sheet + (1 - mix) * 0.5 * blob
    if fine:
        g = _grf(shape, (1.0, 1.2, 1.0), rng, octaves=1); field += fine * g / (g.std() + 1e-9)
    if agg_amp:
        a = _grf(shape, agg, rng, octaves=1); field += agg_amp * a / (a.std() + 1e-9)
    if macro_frac > 0:
        m = _grf(shape, (agg[0], agg[1] * 1.5, agg[2] * 1.5), rng, octaves=1)
        t = _level_for_porosity(m, 1 - macro_frac)
        field -= macro_amp * np.clip(m - t, 0, None) / (m.std() + 1e-9) * 3
    return field >= _level_for_porosity(field, porosity)


materials.FAMILIES["web"] = lambda shape, porosity, rng, **kw: web(shape, porosity, rng, **kw)


def face_yield(solid, z_frames, iv=(2.5, 1.2, 1.2), esc=2.0, edge=1.0, dy=0.0):
    """Cut-face SE yield per frame: interaction-volume density x escape.

    iv: interaction volume extent (z below the face, y, x), voxels.
    The density term looks only DOWN from the face (z..z+3*iv_z).
    """
    out = []
    sf = solid.astype(np.float32)
    dens = ndi.gaussian_filter(sf, (0, iv[1], iv[2]))
    for z in z_frames:
        k = np.exp(-np.arange(0, int(3 * iv[0]) + 1) / iv[0]); k /= k.sum()
        d = sum(w * dens[min(z + j, len(dens) - 1)] for j, w in enumerate(k))
        open_near = 1.0 - ndi.gaussian_filter(sf[z], esc)
        if dy:
            # the interaction volume runs along the slanted beam, so a face
            # pixel with open pore just below it (+y) loses more electrons
            # out of that side than one with pore above it: overhangs glow
            open_near = ndi.shift(open_near, (-dy, 0), order=1, mode="nearest")
        y = (0.55 + 0.45 * d) * (1.0 + edge * open_near)
        out.append(y.astype(np.float32))
    return np.stack(out)


def real_hist(style, n=10):
    """Real ROI grey, lightly denoised (3x3 median): the noiseless target.

    Read from real_tone.json (a snapshot of this function on the real stacks)
    so generation works without them; quantiles are identical.
    """
    tone = HERE / "real_tone.json"
    if n == 10 and tone.exists():
        c = json.loads(tone.read_text())[style]
        return np.repeat(np.arange(256), c).astype(np.float32)
    import lab
    imgs, rois = lab.real_set(style, n)
    # eroded ROI: the raw outline includes dark support pixels a patch fully
    # inside the layer never has
    return np.concatenate([ndi.median_filter(a, 3)[ndi.binary_erosion(r, iterations=12)]
                           for a, r in zip(imgs, rois)]).astype(np.float32)


def real_hist_temporal(style, n=10):
    """Real ROI grey with the noise removed by TIME, not space (v7).

    real_hist() denoises with a 3x3 median, which also erases 1-2 px pores:
    the generator was then tone-matched to a histogram with its smallest
    pores missing. Detector noise is independent from slice to slice while
    structure persists, so a median over three consecutive slices,
    registered by drift, removes the noise and keeps the pores.
    """
    import lab
    tif, pkg = lab.REAL[style]
    imgs, rois = lab.real_set(style, n)
    zs = sorted(int(p.stem[6:]) for p in (pkg / "start_masks").glob("slice_*.png"))
    zs = zs[:: max(1, len(zs) // n)][:n]
    out = []
    for a, r, z in zip(imgs, rois, zs):
        a = a.astype(np.float32); nb = []
        for dz in (-1, 1):
            try:
                b = np.asarray(tifffile.imread(tif, key=z + dz)).astype(np.float32)
            except Exception:
                continue
            best = max(range(-4, 5), key=lambda dy: np.corrcoef(a[40:-40:2, ::4].ravel(),
                                                                 np.roll(b, -dy, 0)[40:-40:2, ::4].ravel())[0, 1])
            nb.append(np.roll(b, -best, 0))
        if len(nb) == 2:
            m = np.median(np.stack([a] + nb), 0)
            out.append(m[ndi.binary_erosion(r, iterations=12)])
    return np.concatenate(out).astype(np.float32)


def real_grain(style):
    """Pixel noise sd of the real ROI: residual after a 3x3 median, x1.16."""
    import lab
    imgs, rois = lab.real_set(style, 6)
    r = [(a.astype(np.float32) - ndi.median_filter(a, 3))[m] for a, m in zip(imgs, rois)]
    return float(np.std(np.concatenate(r)) * 1.16)


def tone_match(img, ref, rng):
    """Monotone map of img (all pixels) onto the distribution of ref."""
    flat = img.reshape(-1)
    order = np.argsort(flat + rng.uniform(0, 1e-3, flat.size), kind="stable")
    q = np.sort(ref)[np.linspace(0, ref.size - 1, flat.size).astype(int)]
    out = np.empty_like(flat, dtype=np.float32); out[order] = q
    return out.reshape(img.shape)


def render_crop(style, P, seed=1, n=10, shape=(288, 576), ref=None):
    cal = json.loads((HERE / "optics_v3.json").read_text())[style]
    opt = Optics(**cal["optics"]); opt.depth = int(P.get("depth", 64))
    por = cal["material"]["porosity"]
    vol = (n + opt.depth + 2, shape[0] + 16, shape[1])
    rng = np.random.default_rng(seed)
    solid = web(vol, por, rng, **P.get("mat", {}))
    fy = face_yield(solid, list(range(n)), **P.get("face", {}))
    import see_through
    curve = see_through.depth_curve(style, opt.depth, optics="optics_v3.json")
    alb = solid.astype(np.float32)
    t = json.loads((HERE / "optics.json").read_text())[style]["targets"]
    rk = dict(smooth=P.get("smooth", 1.2), secant=P.get("secant", 1.0), cos_floor=0.2, hit_albedo=True,
              defocus=P.get("defocus", 0.03), hole_floor=P.get("hole_floor_frac", 0.6) * opt.g_empty,
              hole_len=P.get("hole_len", 6.0), escape=P.get("escape", 2.0), sigma0=P.get("sigma0", 1.1),
              shade_floor=P.get("shade_floor", 0.4))
    grey, _ = render_surface(solid, alb, opt, list(range(n)), curve, rng,
                             face_level=float(t["solid_mean"]), face_map=fy, **rk)
    grey = grey[:, 8:-8]
    if ref is not None:
        # map the NOISELESS image onto the denoised real distribution, then
        # add noise at the measured real grain: matching after noise had
        # stretched the noise into a fake grain
        grey = tone_match(grey, ref, rng)
        g = P.get("grain", None)
        sd = g if g is not None else real_grain(style)
        corr = P.get("grain_corr", 0.6)
        nz = rng.standard_normal(grey.shape).astype(np.float32)
        if corr:
            nz = ndi.gaussian_filter(nz, (0, corr, corr)); nz /= nz.std()
        img = grey + sd * nz * np.sqrt(np.clip(grey, 1, 255) / max(float(np.median(grey)), 1))
    else:
        img = add_noise(grey, opt, rng).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), solid[:n, 8:-8]


def funnel(grey, face_solid, F, opt):
    """Pore interiors as funnels: brightness falls with distance from the rim.

    Looking into an open pore at an angle, a point d px inside the rim sees
    wall that is deeper the larger d, and less of its signal escapes; the
    real pores are darkest at a core and brighten smoothly to the rim. The
    first-hit render alone gives flat pools. F = {mix, lam, rim, floor} in
    grey levels; only pore pixels change, so the truth is untouched.
    """
    if not F:
        return grey
    out = grey.copy()
    for i in range(len(grey)):
        pore = ~face_solid[i]
        d = ndi.distance_transform_edt(pore).astype(np.float32)
        f = F["floor"] + (F["rim"] - F["floor"]) * np.exp(-(d - 1) / F["lam"])
        f = ndi.gaussian_filter(np.where(pore, f, F["rim"]), F.get("soft", 1.5))
        out[i] = np.where(pore, (1 - F["mix"]) * grey[i] + F["mix"] * f, grey[i])
    return out


DRIFT = {"comp": 1.103, "uncomp": 1.09, "pristine": 0.10}


def real_geometry(style, z0=0, nz=76, clean=0.0, rough=0.0, rough_sigma=(0.8, 1.2, 1.0),
                  micro=0.0, micro_sigma=(1.0, 1.4, 1.2), seed=0, erode=0.0,
                  web_detail=None):
    """3D solid volume from the real rule masks, registered for drift.

    Returns (solid[nz, H, W], inside[H, W]) in the sample frame; `inside` is
    where every slice of the range is inside the layer. The masks are the
    threshold rule's, so this geometry carries the rule's own shape bias,
    but appearance is rendered from it by physics, never learned from it,
    and truth for the rendered stack is this volume exactly.
    """
    from discriminate import REAL
    from PIL import Image
    tif, pkg = REAL[style]
    zs = sorted(int(p.stem[6:]) for p in (pkg / "start_masks").glob("*.png"))[z0:z0 + nz]
    S, R = [], []
    for z in zs:
        m = np.array(Image.open(pkg / "start_masks" / f"slice_{z:04d}.png"))
        s = int(round(DRIFT[style] * z))
        S.append(np.roll(m == 255, -s, axis=0)); R.append(np.roll(m != 128, -s, axis=0))
    S, R = np.stack(S), np.stack(R)
    if clean:
        S = ndi.gaussian_filter(S.astype(np.float32), (clean, 0.5, 0.5)) > 0.5
    if erode:
        # thin the mask solid by `erode` px (signed distance): tests whether
        # the rule masks overstate cut-face solid (shine-through labelled solid)
        sd = ndi.distance_transform_edt(S).astype(np.float32) - ndi.distance_transform_edt(~S).astype(np.float32)
        S = sd > erode
    if web_detail:
        # hybrid: real large-scale shapes + generated fine sponge. Thin
        # cavities carved into the solid, thin strands left in the pores.
        W = web_detail
        rng2 = np.random.default_rng(seed + 1)
        f = _grf(S.shape, tuple(W.get("wall", (2.0, 2.5, 2.0))), rng2, octaves=2, lacunarity=(1.0, 2.0, 2.0), gain=0.4)
        f = -np.abs(f / (f.std() + 1e-9))       # 0 on a fine wall network
        sd = ndi.distance_transform_edt(S).astype(np.float32) - ndi.distance_transform_edt(~S).astype(np.float32)
        inner = sd > W.get("keep", 1.5)          # keep a solid rim at the mask outline
        carve = inner & (f < np.quantile(f[inner], W.get("carve", 0.25))) if inner.any() else np.zeros_like(S)
        strands = (~S) & (sd < -W.get("gap", 2.0)) & (f > np.quantile(f, 1 - W.get("strand", 0.05)))
        S = (S & ~carve) | strands
    if rough or micro:
        # sub-mask-resolution structure: the rule masks are blurred-threshold
        # outlines; the material has pores and roughness finer than that.
        # Signed distance to the mask surface + fine noise -> ragged surface;
        # a second fine field carves micro-pores inside the solid.
        rng = np.random.default_rng(seed)
        sd = ndi.distance_transform_edt(S).astype(np.float32) - ndi.distance_transform_edt(~S).astype(np.float32)
        if rough:
            sd += rough * materials._grf(S.shape, rough_sigma, rng, octaves=2, gain=0.5)
        S = sd > 0
        if micro:
            m = materials._grf(S.shape, micro_sigma, rng, octaves=1)
            S &= ~(m > _level_for_porosity(m, 1 - micro))
    return S, R.all(0)


def render_real_geo(style, P, z0=0, n=10, ref=None, seed=1, box=None):
    cal = json.loads((HERE / "optics_v3.json").read_text())[style]
    opt = Optics(**cal["optics"]); opt.depth = int(P.get("depth", 64))
    S, inside = real_geometry(style, z0, n + opt.depth + 2, P.get("clean", 0.0), **P.get("geo", {}))
    if box is None:
        # widest run of columns that stays inside the layer over the most rows
        best = None
        for thr in (0.97, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            rows = np.where(inside.mean(1) > thr)[0]
            if len(rows) < 40:
                continue
            # longest contiguous row run
            runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
            r = max(runs, key=len)
            if len(r) < 40:
                continue
            cols = np.where(inside[r[0]:r[-1] + 1].mean(0) > 0.995)[0]
            if len(cols) < 200:
                continue
            cr = max(np.split(cols, np.where(np.diff(cols) > 1)[0] + 1), key=len)
            if len(cr) >= 200:
                best = (int(r[0]), int(r[-1]) + 1, int(cr[0]), int(cr[-1]) + 1); break
        if best is None:
            raise SystemExit(f"no region stays inside the {style} layer for these slices")
        box = best
    y0, y1, x0, x1 = box
    solid = S[:, y0:y1, x0:x1].copy()
    rng = np.random.default_rng(seed)
    fy = face_yield(solid, list(range(n)), **P.get("face", {}))
    alb = solid.astype(np.float32)
    C = P.get("comp")
    if C:
        # compositional contrast: Pt-rich agglomerates bright, carbon and
        # ionomer dimmer, varying on the agglomerate scale. Multiplies the
        # emission of solid only (cut face and buried surfaces alike).
        f = materials._grf(solid.shape, tuple(C.get("sigma", (3.0, 4.0, 4.0))), rng, octaves=2, gain=0.5)
        f = f / (f.std() + 1e-9)
        a = np.exp(C["amp"] * f).astype(np.float32)
        a /= float(a[solid].mean())
        alb = np.where(solid, a, 0).astype(np.float32)
        fy = fy * a[:n]
    import see_through
    curve = see_through.depth_curve(style, opt.depth, optics="optics_v3.json")
    t = json.loads((HERE / "optics.json").read_text())[style]["targets"]
    rk = dict(smooth=P.get("smooth", 1.2), secant=P.get("secant", 1.0), cos_floor=0.2, hit_albedo=True,
              defocus=P.get("defocus", 0.03), hole_floor=P.get("hole_floor_frac", 0.6) * opt.g_empty,
              hole_len=P.get("hole_len", 6.0), escape=P.get("escape", 2.0), sigma0=P.get("sigma0", 1.4),
              shade_floor=P.get("shade_floor", 0.4))
    grey, _ = render_surface(solid, alb, opt, list(range(n)), curve, rng,
                             face_level=float(t["solid_mean"]), face_map=fy, **rk)
    grey = funnel(grey, solid[:n], P.get("funnel"), opt)
    K = P.get("curtain")
    if K:
        # FIB curtaining: thin vertical streaks, constant down a column over
        # long stretches, slowly evolving between slices; multiplicative
        n_, h_, w_ = grey.shape
        c0 = materials._grf((n_, 1, w_), (K.get("zc", 6.0), 1.0, K.get("w", 0.8)), rng)[:, 0, :]
        c0 = c0 / (c0.std() + 1e-9)
        mod = materials._grf((n_, h_, w_), (2.0, K.get("len", 25.0), 0.6), rng)
        mod = 1.0 + 0.5 * mod / (mod.std() + 1e-9)
        grey = grey * (1.0 + K["amp"] * c0[:, None, :] * mod)
    if P.get("psf") is not None:
        grey = np.stack([apply_psf(g, np.asarray(P["psf"])) for g in grey])
    if ref is not None:
        grey = tone_match(grey, ref, rng)
        nz_ = rng.standard_normal(grey.shape).astype(np.float32)
        c = P.get("grain_corr", 0.6)
        if c:
            nz_ = ndi.gaussian_filter(nz_, (0, c, c)); nz_ /= nz_.std()
        sd = P.get("grain") or real_grain(style)
        grey = grey + sd * nz_ * np.sqrt(np.clip(grey, 1, 255) / max(float(np.median(grey)), 1))
    return np.clip(grey, 0, 255).astype(np.uint8), solid[:n], box


def render_super(style, P, seed=1, n=8, shape=(96, 192), k=3, ref=None):
    """Render at k x finer in-plane resolution, then average k x k blocks.

    Particles near the pixel size give partial-volume grey levels; real
    solid looks like soft mottling for this reason. Truth per pixel is the
    majority of its k x k fine voxels (exact for that definition).
    """
    cal = json.loads((HERE / "optics_v3.json").read_text())[style]
    opt = Optics(**cal["optics"]); opt.depth = int(P.get("depth", 32))
    opt.parallax *= k
    por = cal["material"]["porosity"]
    H, W = shape[0] * k + 16 * k, shape[1] * k
    rng = np.random.default_rng(seed)
    m = dict(P.get("mat", {}))
    m["wall"] = tuple(np.array(m.get("wall", (4, 6, 4)), float) * np.array([1, k, k]))
    m["agg"] = tuple(np.array(m.get("agg", (12, 20, 16)), float) * np.array([1, k, k]))
    solid = web((n + opt.depth + 2, H, W), por, rng, **m)
    fy = face_yield(solid, list(range(n)), iv=(P.get("iv_z", 1.0), k * 0.8, k * 0.8),
                    esc=P.get("esc", 2.0) * k, edge=P.get("edge", 0.8))
    import see_through
    curve = see_through.depth_curve(style, opt.depth, optics="optics_v3.json")
    t = json.loads((HERE / "optics.json").read_text())[style]["targets"]
    rk = dict(smooth=P.get("smooth", 1.0) * k, secant=P.get("secant", 0.8), cos_floor=0.2, hit_albedo=True,
              defocus=P.get("defocus", 0.03) * k, hole_floor=0.6 * opt.g_empty, hole_len=6.0 * k,
              escape=P.get("escape", 3.0), sigma0=P.get("sigma0", 1.4) * k, shade_floor=0.4,
              cone_steps=tuple(np.array([2.0, 4.0, 8.0, 16.0]) * 1.0))
    grey, _ = render_surface(solid, solid.astype(np.float32), opt, list(range(n)), curve, rng,
                             face_level=float(t["solid_mean"]), face_map=fy, **rk)
    g = grey[:, 8 * k:-8 * k].reshape(n, shape[0], k, shape[1], k).mean((2, 4))
    tr = solid[:n, 8 * k:-8 * k].reshape(n, shape[0], k, shape[1], k).mean((2, 4)) > 0.5
    if ref is not None:
        g = tone_match(g, ref, rng)
        g = g + (P.get("grain") or real_grain(style)) * rng.standard_normal(g.shape) \
            * np.sqrt(np.clip(g, 1, 255) / max(float(np.median(g)), 1))
    return np.clip(g, 0, 255).astype(np.uint8), tr


def spectrum(patches):
    """Mean 2-D power spectrum of 64x64 patches (mean removed, Hann window)."""
    w = np.outer(np.hanning(64), np.hanning(64)).astype(np.float32)
    F = [np.abs(np.fft.fft2((p - p.mean()) * w)) ** 2 for p in patches]
    return np.mean(F, 0)


def fit_psf(real_p, fake_p, noise_sd_real, noise_sd_fake, prev=None, lo=0.25, hi=4.0):
    """Linear filter H (frequency domain, 64x64) mapping fake signal spectrum to real.

    Noise power (white, sd known) is removed from both before the ratio, so
    H shapes the signal only: an effective, possibly anisotropic, point
    spread function plus scan blur. Smoothed and clipped for stability.
    """
    w2 = (np.hanning(64)[:, None] * np.hanning(64)[None, :]) ** 2
    nfloor = lambda sd: sd ** 2 * w2.sum()
    Pr = np.maximum(spectrum(real_p) - nfloor(noise_sd_real), 1e-3)
    Pf = np.maximum(spectrum(fake_p) - nfloor(noise_sd_fake), 1e-3)
    H = np.sqrt(Pr / Pf)
    H = np.fft.ifftshift(ndi.gaussian_filter(np.fft.fftshift(H), 1.5))
    H = np.clip(H, lo, hi); H[0, 0] = 1.0
    return H if prev is None else prev * H


def apply_psf(img, H):
    """Apply a 64x64 frequency response to an image of any size (resampled)."""
    h, w = img.shape[-2:]
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None, :]
    iy = np.clip(np.round(fy * 64).astype(int), -32, 31) % 64; ix = np.clip(np.round(fx * 64).astype(int), -32, 31) % 64
    Hf = H[iy, ix]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * Hf))
