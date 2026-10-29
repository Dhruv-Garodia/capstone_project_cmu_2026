#!/usr/bin/env python3
"""v5 geometry: hierarchical anisotropic Voronoi foam (angular pores).

Big cells = agglomerates; a fraction removed -> angular cavities.
Small cells inside the rest; a fraction removed -> fine angular pores.
Thin gaps along small-cell walls. Distances are anisotropic (sampling),
so cells are elongated along y like the real texture. Porosity is hit by
bisection on the removal fraction of small cells. Truth = this volume.
"""
import json
import numpy as np
from scipy import ndimage as ndi


def cells(shape, n, rng, sampling, cluster=None):
    seeds = np.zeros(shape, bool)
    if cluster:
        npar = max(1, n // cluster[0])
        par = np.stack([rng.uniform(0, s, npar) for s in shape], 1)
        kids = par[rng.integers(0, npar, n)] + rng.normal(0, 1, (n, 3)) * np.array(cluster[1])
        idx = tuple(np.clip(np.rint(kids[:, i]).astype(int), 0, shape[i] - 1) for i in range(3))
    else:
        idx = tuple(rng.integers(0, s, n) for s in shape)
    seeds[idx] = True
    lab = np.zeros(shape, np.int32); lab[idx] = np.arange(1, n + 1)
    d, ind = ndi.distance_transform_edt(~seeds, sampling=sampling, return_indices=True)
    L = lab[ind[0], ind[1], ind[2]]
    # wall distance: distance to a voxel with a different label
    wall = np.zeros(shape, bool)
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3
        a[ax] = slice(1, None); b[ax] = slice(None, -1)
        diff = L[tuple(a)] != L[tuple(b)]
        wall[tuple(a)] |= diff        # one side only: 1-voxel walls
    wd = ndi.distance_transform_edt(~wall, sampling=sampling).astype(np.float32)
    return L, wd


def vorofoam(shape, porosity, rng, big=28.0, small=7.0, stretch=1.6, p_big=0.25, gap=0.6,
             gap_frac=0.0, rough=0.0, warp=2.0, warp_sig=(3.0, 4.0, 3.0), debris=0.0, patch=0.0, patch_sig=(10.0, 25.0, 22.0), clustered=False,
             cell_spread=0.0, **kw):
    sampling = (1.0, 1.0 / stretch, 1.0)            # y counts less -> cells longer in y
    vol = np.prod(shape)
    Lb, wdb = cells(shape, max(4, int(vol / big ** 3 * stretch)), rng, sampling)
    Ls, wds = cells(shape, max(8, int(vol / small ** 3 * stretch)), rng, sampling,
                    cluster=(12, (small * 1.5, small * 2.0, small * 1.5)) if clustered else None)
    nb = Lb.max() + 1; ns = Ls.max() + 1
    if warp:
        # domain warp: smooth random displacement of where each voxel looks
        # up its cell, so facets bend and fray instead of tiling
        from materials import _grf
        zz, yy, xx = np.meshgrid(*[np.arange(n_) for n_ in shape], indexing="ij")
        def w(sig):
            g = _grf(shape, sig, rng, octaves=2, gain=0.5); return g / (g.std() + 1e-9)
        dz, dy, dx = (warp * w(warp_sig) for _ in range(3))
        co = [np.clip(zz + 0.3 * dz, 0, shape[0] - 1), np.clip(yy + dy, 0, shape[1] - 1), np.clip(xx + dx, 0, shape[2] - 1)]
        Lb = ndi.map_coordinates(Lb, co, order=0); wdb = ndi.map_coordinates(wdb, co, order=1)
        Ls = ndi.map_coordinates(Ls, co, order=0); wds = ndi.map_coordinates(wds, co, order=1)
        del zz, yy, xx, co
    big_keep = rng.random(nb) >= p_big
    solid0 = big_keep[Lb]
    if debris:
        # real cavities hold loose particles: keep a sparse fraction of the
        # small cells inside removed agglomerates
        solid0 = solid0 | ((~big_keep[Lb]) & (rng.random(ns) < debris)[Ls])
    # gaps only around a fraction of small cells (cracks/contacts open there)
    open_cell = rng.random(ns) < gap_frac
    thr = gap
    if rough:
        from materials import _grf
        g = _grf(shape, (1.2, 1.5, 1.2), rng, octaves=1); g /= g.std()
        thr = gap + rough * g
    gaps = (wds < thr) & open_cell[Ls]
    r = rng.random(ns)
    if patch:
        # removal likelihood varies smoothly in space: some agglomerates full
        # of mesopores, some nearly dense (real is patchy, not a uniform lace)
        from materials import _grf
        f = _grf(shape, patch_sig, rng, octaves=1); f /= f.std() + 1e-9
        cnt = np.bincount(Ls.ravel(), minlength=ns).astype(np.float32)
        fm = np.bincount(Ls.ravel(), weights=f.ravel(), minlength=ns) / np.maximum(cnt, 1)
        r = 1.0 / (1.0 + np.exp(-(np.log(r / (1 - r + 1e-9) + 1e-9) + patch * fm)))
        del f
    lo, hi = 0.0, 0.95
    for _ in range(18):                                # bisection on small-cell removal
        q = 0.5 * (lo + hi)
        s = solid0 & (r >= q)[Ls] & ~gaps
        if 1 - s.mean() < porosity:
            lo = q
        else:
            hi = q
    return s


if __name__ == "__main__":
    import time
    from PIL import Image
    t = time.time()
    s = vorofoam((40, 250, 500), 0.42, np.random.default_rng(1))
    print("built", round(time.time() - t, 1), "s porosity", round(1 - s.mean(), 3))
    Image.fromarray((s[10] * 255).astype(np.uint8)).save(
        "voro_mask.png")
