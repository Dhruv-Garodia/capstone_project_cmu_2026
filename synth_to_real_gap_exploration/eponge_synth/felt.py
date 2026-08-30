"""
Sedimented felt of crumpled sheets (or fibres) — a deposition model, not random placement.

Why: an SEM of a flake/fibre felt (e.g. the 5 kV x1000 micrograph) shows thousands of thin,
crumpled, wavy-edged sheets RESTING ON EACH OTHER in layers, seen plan-view with bright
secondary-electron edges.  Random placement of curved ellipsoids in space (the old flake_network)
produces interpenetrating, isolated, smooth ribbons and looks nothing like it.

Model
-----
1. each sheet = thin slab of half-thickness c around a crumpled mid-surface
       h(u, v) = curl * u^2 / a  +  crumple * b * sum_k A_k sin(2 pi (f_u,k u/a + f_v,k v/b) + phi_k)
   with a wavy elliptical outline  (u/a)^2 + (v/b)^2 <= (1 + waviness * sum_m w_m sin(m theta + psi_m))^2
2. orientation: sheet normal tilted from +z by |N(0, tilt_spread)| (felts are anisotropic: sheets lie flat-ish)
3. ballistic deposition: the sheet is lowered along z until it touches the pile (column height map),
   so sheets never interpenetrate and the top surface is a genuine, rough, layered surface
4. compaction: z is scaled by `compaction` (<1) at the end, as the real felt is pressed

`mode="fiber"` gives round-section curved fibres (b = c) with the same deposition physics.
Returns Microstructure with material 1 = sheet/fibre.  z index 0 is the TOP surface (beam side).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .microstructure import Microstructure


def _frame(tilt_spread_deg, rng):
    tilt = abs(rng.normal(0, np.deg2rad(tilt_spread_deg)))
    phi = rng.uniform(0, 2 * np.pi)
    n = np.array([np.cos(tilt), np.sin(tilt) * np.cos(phi), np.sin(tilt) * np.sin(phi)])  # (z, y, x)
    ref = rng.standard_normal(3)
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.stack([u, v, n])  # rows: local u, v, w axes in world (z, y, x) coordinates


def sheet_felt(
    shape: Tuple[int, int, int] = (128, 320, 320),
    voxel_um: float = 0.25,
    mode: str = "sheet",                 # 'sheet' (flakes / platelets) or 'fiber'
    n_sheets: int = 8000,
    length_um: Tuple[float, float] = (8.0, 0.45),      # lognormal (median, sigma) of the long axis
    fragment_fraction: float = 0.0,                     # bimodal population: fraction of small fragments
    fragment_length_um: Tuple[float, float] = (2.5, 0.4),
    width_ratio: Tuple[float, float] = (0.25, 0.7),    # uniform range, short axis / long axis (sheet mode)
    thickness_um: Tuple[float, float] = (0.35, 0.25),  # lognormal (median, sigma)
    crumple: float = 0.35,               # amplitude of the crumpling height field, relative to the half-width
    n_modes: int = 3,
    mode_freq: Tuple[float, float] = (0.4, 1.6),   # crumple wavelengths, cycles per sheet
    curl: float = 0.25,                  # parabolic curl along the long axis
    roll_ratio: Tuple[float, float] = (1.2, 4.0),   # cylindrical rolling about the long axis: radius = ratio * half-width (petal/ribbon curl)
    fold_prob: float = 0.35,             # probability of a sharp crease across the sheet
    fold_amp: float = 0.5,               # crease slope (height per unit distance from the crease line)
    outline_modes: int = 6,              # torn / irregular outline: number of angular modes
    edge_waviness: float = 0.25,
    tilt_spread_deg: float = 35.0,       # std of the sheet-normal tilt from the stacking axis
    compaction: float = 0.6,             # final z-scaling of the pile (pressing)
    settle_quantile: float = 0.25,       # sheet rests at this quantile of the pile height under it (0 = rigid ballistic; >0 = bends over obstacles)
    drape: float = 0.0,                  # flexibility: fraction of the (smoothed) gap to the pile that each part of the sheet sags down by
    drape_sigma: float = 0.35,           # smoothing of the sag field, as a fraction of the sheet half-width (keeps the sheet's own shape)
    micro_crumple: float = 0.0,          # fine wrinkles: amplitude relative to the half-width (sub-micron surface texture)
    micro_freq: Tuple[float, float] = (2.5, 6.0),
    max_failures: int = 6000,
    fill_top_fraction: float = 0.85,     # stop when this fraction of columns has pile top above 0.3 Z
    coverage_stop: float = 0.92,         # ... or when this fraction of columns is covered by at least one sheet
    seed: int = 0,
) -> Microstructure:
    rng = np.random.default_rng(seed)
    Z, Y, X = (int(s) for s in shape)
    H = np.full((Y, X), Z, np.int32)     # top of pile per column (Z = empty column)
    vox_z, vox_y, vox_x, vox_e = [], [], [], []
    deposited = failed = 0
    for _ in range(n_sheets):
        Lm, Ls = (fragment_length_um if rng.uniform() < fragment_fraction else length_um)
        L = min(np.exp(rng.normal(np.log(Lm), Ls)) / voxel_um, 0.35 * min(Y, X))
        a = max(L / 2, 1.5)
        c = max(np.exp(rng.normal(np.log(thickness_um[0]), thickness_um[1])) / voxel_um / 2, 0.55)
        b = a * rng.uniform(*width_ratio) if mode == "sheet" else c
        R = _frame(tilt_spread_deg, rng)
        A = rng.uniform(0.3, 1.0, n_modes); A /= A.sum()
        fu, fv, ph = rng.uniform(*mode_freq, n_modes), rng.uniform(*mode_freq, n_modes), rng.uniform(0, 2 * np.pi, n_modes)
        wm = rng.uniform(0.2, 1.0, outline_modes); wm /= wm.sum(); psi = rng.uniform(0, 2 * np.pi, outline_modes)
        Rr = b * rng.uniform(*roll_ratio) * rng.choice([-1, 1])       # signed roll radius
        h_roll_max = abs(Rr) - np.sqrt(max(Rr * Rr - b * b, 0.0))
        do_fold = rng.uniform() < fold_prob
        f_ang, f_off = rng.uniform(0, np.pi), rng.uniform(-0.5, 0.5)
        hmax = crumple * b + curl * a + h_roll_max + (fold_amp * 1.6 * min(a, b) if do_fold else 0.0)
        ext = np.sqrt((a * R[0]) ** 2 + (b * R[1]) ** 2 + ((c + hmax) * R[2]) ** 2)   # bbox of the tilted slab
        ez, ey, ex = (int(np.ceil(e)) + 1 for e in ext)
        cy, cx = int(rng.integers(0, Y)), int(rng.integers(0, X))
        zr = np.arange(-ez, ez + 1, dtype=np.float32)[:, None, None]
        yr = np.arange(-ey, ey + 1, dtype=np.float32)[None, :, None]
        xr = np.arange(-ex, ex + 1, dtype=np.float32)[None, None, :]
        Rf = R.astype(np.float32)
        uu = Rf[0, 0] * zr + Rf[0, 1] * yr + Rf[0, 2] * xr      # broadcasting: no int64 grids
        vv = Rf[1, 0] * zr + Rf[1, 1] * yr + Rf[1, 2] * xr
        ww = Rf[2, 0] * zr + Rf[2, 1] * yr + Rf[2, 2] * xr
        h = curl * uu ** 2 / a
        h = h + np.sign(Rr) * (abs(Rr) - np.sqrt(np.maximum(Rr * Rr - vv ** 2, 0.0)))     # cylindrical roll
        if do_fold:                                                                         # sharp crease
            h = h + fold_amp * np.abs(np.cos(f_ang) * uu / a + np.sin(f_ang) * vv / b - f_off) * min(a, b)
        for k in range(n_modes):
            h = h + crumple * b * A[k] * np.sin(2 * np.pi * (fu[k] * uu / a + fv[k] * vv / b) + ph[k])
        if micro_crumple > 0:
            for _k in range(3):
                mf = rng.uniform(*micro_freq, 2); mp = rng.uniform(0, 2 * np.pi)
                h = h + micro_crumple * b / 3 * np.sin(2 * np.pi * (mf[0] * uu / a + mf[1] * vv / b) + mp)
        if mode == "sheet":
            theta = np.arctan2(vv / b, uu / a)
            outline = 1.0 + edge_waviness * sum(wm[m] * np.sin((m + 2) * theta + psi[m]) for m in range(outline_modes))
            rr = (uu / a) ** 2 + (vv / b) ** 2
            inside = (np.abs(ww - h) <= c) & (rr <= outline ** 2)
            edge = rr >= (0.92 * outline) ** 2                       # outline band: the physical sheet edge
            if do_fold:
                edge |= np.abs(np.cos(f_ang) * uu / a + np.sin(f_ang) * vv / b - f_off) < 0.05   # crease line
            del theta, outline, rr
        else:  # fibre: circular cross-section around the (curled, crumpled) centre line
            inside = ((vv ** 2 + (ww - h) ** 2) <= c ** 2) & (np.abs(uu) <= a)
            edge = np.abs(uu) >= 0.95 * a
        idx = np.nonzero(inside)
        e_flag = edge[idx]
        del uu, vv, ww, h, inside, edge
        if len(idx[0]) == 0:
            continue
        dz = idx[0] - ez; dz = dz - dz.min()
        yv, xv = idx[1] - ey + cy, idx[2] - ex + cx
        inb = (yv >= 0) & (yv < Y) & (xv >= 0) & (xv < X)
        if inb.sum() < 8:
            continue
        dz, yv, xv, e_flag = dz[inb], yv[inb], xv[inb], e_flag[inb]
        z0 = int(np.quantile(H[yv, xv] - dz, settle_quantile)) - 1   # rests on / bends over the pile
        if z0 < 0:
            failed += 1
            if failed > max_failures:
                break
            continue
        zv = z0 + dz
        if drape > 0:                                   # flexible sheet: sag toward the pile below, smoothly over its footprint
            gap = (H[yv, xv] - zv - 1).astype(np.float32)
            y0_, x0_ = yv.min(), xv.min()
            Fm = np.full((yv.max() - y0_ + 1, xv.max() - x0_ + 1), np.inf, np.float32)
            np.minimum.at(Fm, (yv - y0_, xv - x0_), gap)
            filled = np.isfinite(Fm)
            if filled.any():
                Fm[~filled] = np.median(Fm[filled])
                from scipy import ndimage as _ndi
                Fs = _ndi.gaussian_filter(Fm, max(drape_sigma * b, 1.0))
                sag = np.clip(drape * Fs[yv - y0_, xv - x0_], 0, None)
                zv = np.minimum(np.round(zv + sag).astype(np.int32), H[yv, xv] - 1)
                zv = np.maximum(zv, 0)
        vox_z.append(zv.astype(np.int32)); vox_y.append(yv.astype(np.int32)); vox_x.append(xv.astype(np.int32)); vox_e.append(e_flag)
        np.minimum.at(H, (yv, xv), zv)
        deposited += 1
        if deposited % 100 == 0 and ((H < 0.3 * Z).mean() > fill_top_fraction or (H < Z).mean() > coverage_stop):
            break
    solid = np.zeros((Z, Y, X), bool)
    edges = np.zeros((Z, Y, X), bool)
    if vox_z:
        zv = np.concatenate(vox_z); yv = np.concatenate(vox_y); xv = np.concatenate(vox_x); ev = np.concatenate(vox_e)
        # compaction: press the pile toward the floor (z = Z-1); keep the top surface rough
        zc = np.round(Z - 1 - (Z - 1 - zv) * compaction).astype(np.int32)
        zc = zc - zc.min() + 1                    # pile top at z = 1: the beam side, where depth attenuation starts
        keep = zc < Z
        solid[zc[keep], yv[keep], xv[keep]] = True
        edges[zc[keep & ev], yv[keep & ev], xv[keep & ev]] = True
    mat = solid.astype(np.uint8)
    meta = dict(generator="sheet_felt", mode=mode, voxel_um=voxel_um, realized_porosity=float(1 - solid.mean()),
                n_deposited=deposited, length_um=list(length_um), width_ratio=list(width_ratio),
                thickness_um=list(thickness_um), crumple=crumple, n_modes=n_modes, curl=curl,
                edge_waviness=edge_waviness, tilt_spread_deg=tilt_spread_deg, compaction=compaction, settle_quantile=settle_quantile, drape=drape, micro_crumple=micro_crumple, n_failed=failed, seed=seed,
                shape=[Z, Y, X], top_surface_mean_z=float(np.mean(np.argmax(solid, axis=0))))
    ms = Microstructure(solid=mat, material=mat, meta=meta)
    ms.edges = edges          # geometric edge map (outline bands + creases) for the renderer's SE edge effect
    return ms
