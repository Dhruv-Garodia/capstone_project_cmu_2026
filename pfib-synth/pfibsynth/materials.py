#!/usr/bin/env python3
"""Material families, built directly as a 3D voxel grid.

The principle carried over from the synth-to-real work: no mesh, no renderer.
The structure is voxels, the image is read from those voxels, and the answer
key is read from the same voxels, so the two cannot fail to line up. The old
PuMA -> .obj -> Blender route broke exactly there: a 1080x720 render at 38
degrees tilt against a 150x150 straight-down mask.

Every family takes a target porosity and hits it by bisection on its own
level/threshold, so porosity is an input you can trust rather than an output
you measure afterwards. The old generator asked for 0.6 and realized 0.728.

All families return a bool array of shape (nz, ny, nx), True = solid.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

__all__ = ["FAMILIES", "build"]


# --------------------------------------------------------------- utilities
def _level_for_porosity(field: np.ndarray, porosity: float,
                        max_sample: int = 4_000_000) -> float:
    """Threshold putting `porosity` of the field below it.

    Estimated from a random subsample on large volumes: an exact quantile of
    10^8 floats is a full partition, and the sampling error on 4 million
    draws is about 1e-4 in porosity, well inside the 0.01 tolerance the
    caller checks against.
    """
    flat = field.reshape(-1)
    if flat.size > max_sample:
        idx = np.random.default_rng(0).integers(0, flat.size, max_sample)
        flat = flat[idx]
    return float(np.quantile(flat, porosity))


def _grf(shape, sigma, rng, octaves=1, lacunarity=2.5, gain=0.30):
    """Gaussian random field: smoothed white noise, optionally multi-scale.

    sigma is a per-axis correlation length in voxels. Several octaves give
    clumped-nanoparticle texture -- coarse agglomerates with fine structure
    on them -- rather than the single smooth blob a one-scale field gives.

    lacunarity may be a scalar or per-axis. Per-axis matters: the real stacks
    have pore channels that run far along z (38-70% of pore pixels have no
    solid within 8 slices below), and a scalar lacunarity shortens the z
    correlation on every octave, which caps how elongated the pores can get
    no matter how large sigma_z is.
    """
    sigma = np.asarray(sigma, float)
    lac = np.broadcast_to(np.asarray(lacunarity, float), (len(shape),)).astype(float)
    out = np.zeros(shape, np.float32)
    amp = 1.0
    for o in range(octaves):
        n = _one_octave(shape, sigma / (lac ** o), rng)
        sd = n.std()
        if sd > 0:
            out += amp * (n / sd)
        amp *= gain
    return out


def _one_octave(shape, sigma, rng):
    """Correlated noise at one scale, generated on a decimated grid.

    A Gaussian of sigma 14 is a ~113-tap kernel; applying it along two axes
    of a 212-million-voxel volume is minutes of work per octave. Since the
    result is by construction smooth on the scale of sigma, it can be built
    at a coarser resolution and interpolated up with no visible difference
    and roughly two orders of magnitude less arithmetic.
    """
    sigma = np.maximum(np.asarray(sigma, float), 1e-3)
    fac = np.maximum(np.floor(sigma / 2.0), 1.0)
    fac = np.minimum(fac, np.maximum(np.array(shape) // 8, 1))
    if np.all(fac <= 1):
        n = rng.standard_normal(shape).astype(np.float32)
        ndi.gaussian_filter(n, sigma, output=n, mode="wrap")
        return n
    small = tuple(max(4, int(np.ceil(s / f))) for s, f in zip(shape, fac))
    n = rng.standard_normal(small).astype(np.float32)
    ndi.gaussian_filter(n, sigma / fac, output=n, mode="wrap")
    zoom = tuple(s / t for s, t in zip(shape, small))
    return ndi.zoom(n, zoom, order=1, mode="grid-wrap").astype(np.float32)


def _smooth_profile(n, rng, scale, sigma):
    """A smooth 1-D wander of length n, zero mean, roughly +-scale."""
    w = rng.standard_normal(n).astype(np.float32)
    w = ndi.gaussian_filter1d(w, sigma, mode="nearest")
    sd = w.std()
    return (w / sd) * scale if sd > 0 else w


# ------------------------------------------------------------- the families
def agglomerate(shape, porosity, rng, corr=(12.0, 3.0, 3.0), octaves=3,
                lacunarity=(1.0, 2.5, 2.5), octave_gain=0.30, hetero_amp=0.0,
                hetero_corr=(30.0, 70.0, 70.0), **kw):
    """Clumped nanoparticle agglomerates -- the catalyst-layer base.

    Two components. The fine one, at `corr`, is the agglomerate and particle
    scale. The coarse one, at `hetero_corr`, makes some regions of the layer
    denser than others.

    The coarse component is not decoration. Porosity measured over 64 px
    blocks of the real stacks has a spread of 0.17 to 0.22, where counting
    statistics alone would give 0.008 -- real material is about 25 times more
    heterogeneous than a statistically uniform field. Without it, too few
    pore pixels sit over a lot of buried material and too few over none, and
    the spread of pore grey values comes out 5 to 8 levels short of measured
    however the optics are tuned.
    """
    f = _grf(shape, corr, rng, octaves=octaves, lacunarity=lacunarity,
             gain=octave_gain)
    if hetero_amp > 0:
        f = f + hetero_amp * _grf(shape, hetero_corr, rng, octaves=1)
    return f >= _level_for_porosity(f, porosity)


def compressed(shape, porosity, rng, corr=(12.0, 2.0, 3.6), **kw):
    """Agglomerate flattened in y, as a compressed region is.

    Shorter correlation length along y than x makes pores laterally wide and
    vertically squat, which is what compression does to a catalyst layer.
    """
    return agglomerate(shape, porosity, rng, corr=corr, octaves=3,
                       lacunarity=(1.0, 2.5, 2.5),
                       hetero_amp=kw.get("hetero_amp", 0.0),
                       hetero_corr=kw.get("hetero_corr", (30.0, 50.0, 90.0)))


def spinodal(shape, porosity, rng, corr=(10.0, 4.0, 4.0), **kw):
    """Bicontinuous ionomer/binder phase -- a single-scale level set.

    Both phases percolate, which is the structural signature that separates a
    spinodal morphology from a sphere pack and changes tortuosity a lot.
    """
    f = _grf(shape, corr, rng, octaves=1)
    return f >= _level_for_porosity(f, porosity)


def catalyst(shape, porosity, rng, corr=(4.3, 6.3, 6.3), y_stretch=1.3,
             macro_frac=0.25, macro_corr=(10.0, 14.0, 14.0), macro_amp=2.0,
             rough=0.35, rough_sigma=1.0, tex_amp=0.15, rim=0.0, rim_len=1.5,
             hetero=0.0, hetero_corr=(20.0, 60.0, 60.0), **kw):
    """Catalyst layer with three scales, and an emission texture on the solid.

    Built to close the gaps measured between the real ROI and the single-scale
    agglomerate (lab.py, why_fake):

      macropores   26-49% of real pore pixels sit more than 6 px from any
                   solid; the single-scale field gave 9-21%. A coarse field
                   carves large voids out of the fine agglomerate, and the
                   porosity bisection runs on the combined field so the
                   target still holds.
      y elongation real texture is longer along y than x on all three
                   stacks; the old comp stand-in squashed y and came out the
                   other way round.
      roughness    a particle-scale octave, so solid edges are ragged
                   rather than the smooth contours of a blurred field.
      texture      per-voxel emission 1 +- tex_amp on a 1.5 px scale: real
                   solid is made of particles and is not a flat plateau.
      rim          secondary-electron edge effect: solid within ~rim_len px
                   of a pore emits up to (1 + rim) times more, so thin
                   ligaments and edges glow and thick bodies stay mid-grey.
                   Uniform emission made every solid body a flat white blob.
                   Texture and rim change the image only, never the truth.

    Returns (solid, texture).
    """
    cz, cy, cx = corr
    f = _grf(shape, (cz, cy * y_stretch, cx), rng, octaves=3,
             lacunarity=(1.0, 2.5, 2.5), gain=0.30)
    f /= f.std() + 1e-9
    if rough:
        f += rough * _grf(shape, (rough_sigma, rough_sigma * 1.1, rough_sigma), rng, octaves=1)
    if hetero:   # denser and looser regions, as real 64 px blocks show
        f += hetero * _grf(shape, hetero_corr, rng, octaves=1)
    if macro_frac > 0:
        g = _grf(shape, macro_corr, rng, octaves=1)
        t = _level_for_porosity(g, 1.0 - macro_frac)
        f -= macro_amp * np.clip(g - t, 0, None) / (g.std() + 1e-9) * 3.0
        del g
    solid = f >= _level_for_porosity(f, porosity)
    del f
    return solid, se_texture(solid, rng, tex_amp, rim, rim_len)


def spheres(shape, porosity, rng, r_med=3.0, r_spread=0.25, cluster_sigma=(9.0, 7.0, 6.0),
            per_cluster=40.0, film=0.8, film_cover=0.6, tex_amp=0.12, rim=0.6, rim_len=1.5,
            fill=1.25, macro_frac=0.0, macro_corr=(10.0, 12.0, 10.0), macro_amp=3.0,
            **kw):
    """Catalyst layer as clustered, overlapping carbon spheres with an ionomer film.

    The established route for catalyst-layer reconstructions (sphere packing
    / sphere-based stochastic reconstruction, e.g. the Waterloo and Alberta
    CL generators): carbon-support particles are spheres, they sit in
    agglomerates, and ionomer coats them as a thin, patchy film.

      particles    sphere radii lognormal around r_med px
      agglomerates a Thomas cluster process: parent points Poisson, each with
                   Poisson(per_cluster) children Gaussian-scattered by
                   cluster_sigma (z, y, x) -- longer in depth than across,
                   as measured on the real stacks
      ionomer      the solid grows by `film` px where a smooth random mask
                   is on (`film_cover` of the surface), merging nearby
                   particles into necks the way binder does
      macropores   emerge from the gaps between agglomerates; with
                   macro_frac > 0 a secondary pore network (a thresholded
                   smooth field, the usual two-scale CL picture) is also
                   carved through dense packing, which keeps the solid one
                   connected sponge instead of separate clumps

    Spheres are drawn by one distance transform from their centres, so cost
    does not depend on how many there are; porosity is hit by shifting the
    level of (radius - distance), i.e. growing or shrinking every sphere a
    little, after which the film is added and the level re-solved.
    Returns (solid, texture).
    """
    nz, ny, nx = shape
    vol = float(nz * ny * nx)
    target = (1.0 - porosity) * fill      # <1: fewer spheres, grown and fused
    v_sphere = 4.0 / 3.0 * np.pi * r_med ** 3
    n_pts = int(target * vol / v_sphere)
    n_par = max(1, int(n_pts / per_cluster))
    par = np.stack([rng.uniform(-10, nz + 10, n_par), rng.uniform(-10, ny + 10, n_par),
                    rng.uniform(-10, nx + 10, n_par)], 1)
    kids = rng.poisson(per_cluster, n_par)
    pts = np.repeat(par, kids, 0) + rng.normal(0, 1, (int(kids.sum()), 3)) * np.array(cluster_sigma)
    idx = np.rint(pts).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(shape)), 1)
    idx = idx[ok]
    rad = r_med * np.exp(rng.normal(0, r_spread, len(idx))).astype(np.float32)
    centre = np.zeros(shape, bool)
    rmap = np.zeros(shape, np.float32)
    centre[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    rmap[idx[:, 0], idx[:, 1], idx[:, 2]] = np.maximum(rmap[idx[:, 0], idx[:, 1], idx[:, 2]], rad)
    dist, ind = ndi.distance_transform_edt(~centre, return_indices=True)
    field = (rmap[ind[0], ind[1], ind[2]] - dist).astype(np.float32)
    del dist, ind, centre, rmap
    if macro_frac > 0:
        g = _grf(shape, macro_corr, rng, octaves=2, gain=0.4)
        t = _level_for_porosity(g, 1.0 - macro_frac)
        field -= macro_amp * np.clip(g - t, 0, None) / (g.std() + 1e-9) * 4.0
        del g
    if film > 0:
        mask = _grf(shape, (4.0, 6.0, 6.0), rng, octaves=1)
        on = mask > np.quantile(mask.reshape(-1)[::97], 1.0 - film_cover)
        field = np.where(on, field + film, field)
        del mask, on
    solid = field >= _level_for_porosity(field, porosity)
    del field
    return solid, se_texture(solid, rng, tex_amp, rim, rim_len)


def se_texture(solid, rng, tex_amp=0.0, rim=0.0, rim_len=1.5):
    """Per-voxel emission multiplier for the solid (None if neither term).

    tex_amp  particle-scale grain, 1 +- tex_amp on a ~2 px scale
    rim      edge effect: within ~rim_len px of a pore, up to (1 + rim) x
    Normalised so the mean over solid is 1, leaving the calibrated grey
    levels where they were on average.
    """
    tex = None
    if tex_amp:
        tex = _grf(solid.shape, (1.0, 2.0, 2.0), rng, octaves=1)
        tex = np.clip(1.0 + tex_amp * tex, 0.4, 1.8).astype(np.float32)
    if rim:
        d = ndi.distance_transform_edt(solid).astype(np.float32)
        e = 1.0 + rim * np.exp(-(d - 1.0) / rim_len)
        tex = e if tex is None else tex * e
        del d
    if tex is not None:
        tex /= float(tex[solid].mean())
    return tex


def _voronoi_cells(shape, seeds_zyx, rng):
    """Nearest-seed id per voxel, and distance to the nearest cell wall.

    Done with two O(N) distance transforms rather than an N x n_seeds
    distance matrix, which is what makes this usable at real stack sizes:
    a 1860-wide ROI band has ~10^8 voxels.
    """
    mask = np.ones(shape, bool)
    idx = tuple(np.clip(np.rint(seeds_zyx[:, i]).astype(int), 0, shape[i] - 1)
                for i in range(3))
    mask[idx] = False
    # nearest seed, as the coordinates of that seed
    _, inds = ndi.distance_transform_edt(mask, return_indices=True)
    # an id that is unique per seed voxel
    sid = (inds[0].astype(np.int64) * shape[1] + inds[1]) * shape[2] + inds[2]
    # cell walls: where the nearest seed changes between neighbours
    wall = np.zeros(shape, bool)
    wall[:-1] |= sid[:-1] != sid[1:]
    wall[:, :-1] |= sid[:, :-1] != sid[:, 1:]
    wall[:, :, :-1] |= sid[:, :, :-1] != sid[:, :, 1:]
    wall_dist = ndi.distance_transform_edt(~wall).astype(np.float32)
    return sid, wall_dist


def _mudcrack_2d_impl(shape, porosity, rng, cell_px=55.0, **kw):
    """A dense slab broken by a through-thickness crack network.

    Cracks are cell walls of seeds scattered in x-y only, so each crack runs
    straight through z the way a drying crack in a catalyst layer does.
    Crack width is solved for the target porosity, not guessed.
    """
    nz, ny, nx = shape
    n_seeds = max(4, int(ny * nx / (cell_px ** 2)))
    sy = rng.uniform(0, ny, n_seeds)
    sx = rng.uniform(0, nx, n_seeds)
    # one z-plane of cells, then extruded with a slow wander
    mask = np.ones((1, ny, nx), bool)
    mask[0,
         np.clip(np.rint(sy).astype(int), 0, ny - 1),
         np.clip(np.rint(sx).astype(int), 0, nx - 1)] = False
    _, inds = ndi.distance_transform_edt(mask, return_indices=True)
    sid = inds[1][0].astype(np.int64) * nx + inds[2][0]
    wall = np.zeros((ny, nx), bool)
    wall[:-1] |= sid[:-1] != sid[1:]
    wall[:, :-1] |= sid[:, :-1] != sid[:, 1:]
    wall_dist = ndi.distance_transform_edt(~wall).astype(np.float32)

    wob = _smooth_profile(nz, rng, 1.5, 4.0)
    field = np.empty(shape, np.float32)
    for z in range(nz):
        field[z] = wall_dist + wob[z]
    field += 0.6 * _grf(shape, (2.0, 2.0, 2.0), rng, octaves=2)
    return field >= _level_for_porosity(field, porosity)


def mudcrack(shape, porosity, rng, cell_px=70.0, crack_frac=0.20, wander=6.0, pinch=0.5, **kw):
    """A porous catalyst layer broken by a drying-crack network (v6).

    The layer lies in the x-z plane (y is depth through it, z the milling
    axis), and drying cracks form a polygon network *in that plane* and run
    down through the thickness. The v5 version had it the wrong way round:
    one polygon pattern in the image plane, copied unchanged along z, with
    featureless solid between the cracks, so every slice showed the same
    flat picture. Here:

      network   Voronoi walls of seeds scattered in x-z
      depth     the walls wander in x and z as they go down through y (and
                with position), so a crack face is a tilted, curving sheet
                and the cut sees it move sideways from slice to slice
      opening   crack width varies smoothly, pinching shut in places, so
                cracks branch off, close into bridges and reopen
      blocks    ordinary porous catalyst layer between the cracks, at the
                porosity left over once the cracks take `crack_frac`
    """
    nz, ny, nx = shape
    n_seeds = max(4, int(nz * nx / cell_px ** 2))
    sz = rng.uniform(0, nz, n_seeds); sx = rng.uniform(0, nx, n_seeds)
    mask = np.ones((nz, nx), bool)
    mask[np.clip(np.rint(sz).astype(int), 0, nz - 1), np.clip(np.rint(sx).astype(int), 0, nx - 1)] = False
    _, inds = ndi.distance_transform_edt(mask, return_indices=True)
    sid = inds[0].astype(np.int64) * nx + inds[1]
    wall = np.zeros((nz, nx), bool)
    wall[:-1] |= sid[:-1] != sid[1:]
    wall[:, :-1] |= sid[:, :-1] != sid[:, 1:]
    wd = ndi.distance_transform_edt(~wall).astype(np.float32)          # (z, x)
    # smooth 3-D wander and opening fields, on a coarse grid
    c = 8
    cs = (int(np.ceil(nz / c)) + 2, int(np.ceil(ny / c)) + 2, int(np.ceil(nx / c)) + 2)
    wz, wx, op = (_grf(cs, (3.0, 6.0, 3.0), rng, octaves=2) for _ in range(3))
    zz, xx = np.meshgrid(np.arange(nz, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing="ij")
    crack_d = np.empty(shape, np.float32); widen = np.empty(shape, np.float32)
    for y in range(ny):
        cz, cy, cx = zz / c, np.full_like(zz, y / c), xx / c
        co = np.stack([cz, cy, cx])
        dz = wander * ndi.map_coordinates(wz, co, order=1, mode="nearest")
        dx = wander * ndi.map_coordinates(wx, co, order=1, mode="nearest")
        crack_d[:, y] = ndi.map_coordinates(wd, np.stack([zz + dz, xx + dx]), order=1, mode="nearest")
        widen[:, y] = ndi.map_coordinates(op, co, order=1, mode="nearest")
    # crack where the wall distance is under a locally varying half-width
    score = crack_d / np.exp(pinch * widen)
    crack = score <= np.quantile(score[::2, ::4, ::4], crack_frac)
    del crack_d, widen, score
    m = min(0.85, max(0.05, (porosity - crack_frac) / (1.0 - crack_frac)))
    block = catalyst(shape, m, rng)[0]
    return block & ~crack


def mudcrack_2d(shape, porosity, rng, cell_px=55.0, **kw):
    """v5 mudcrack, kept for comparison: one x-y crack pattern extruded along z."""
    return _mudcrack_2d_impl(shape, porosity, rng, cell_px, **kw)


def faceted(shape, porosity, rng, crystal_px=28.0, junction_sig=2.2, open_frac=0.35, **kw):
    """Sintered polycrystal (v7): closed grain boundaries, pores at junctions.

    v6 opened a uniform gap along every grain boundary, which drew a mesh of
    thin dark lines -- a 2-D drawing, flat. In a sintered body most
    boundaries are closed and show only as a step in signal between grains
    (channeling contrast, one level per grain); the pores sit where three or
    more grains meet, as rounded voids, and a random subset of boundaries
    (`open_frac`, by a smooth 3-D field) opens into channels. Pore score =
    boundary density blurred over `junction_sig` px (high where boundaries
    meet), boosted on the open boundaries; thresholded at the porosity.
    """
    nz, ny, nx = shape
    n = max(6, int(nz * ny * nx / (crystal_px ** 3)))
    seeds = np.stack([rng.uniform(0, nz, n), rng.uniform(0, ny, n), rng.uniform(0, nx, n)], 1).astype(np.float32)
    sid, wall_dist = _voronoi_cells(shape, seeds, rng)
    wall = (wall_dist < 0.75).astype(np.float32)
    dens = ndi.gaussian_filter(wall, junction_sig)
    opn = _grf(shape, (6.0, 8.0, 8.0), rng, octaves=1)
    opn = (opn > np.quantile(opn[::3, ::3, ::3], 1 - open_frac)).astype(np.float32)
    dn = dens / (np.quantile(dens[::3, ::3, ::3], 0.995) + 1e-9)
    score = dn ** 3 + 0.45 * opn * np.exp(-wall_dist / 1.2) + 0.02 * _grf(shape, (2.0, 2.0, 2.0), rng, octaves=1)
    solid = score < np.quantile(score[::2, ::3, ::3], 1 - porosity)
    h = (sid.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)
    tex = (h.astype(np.float64) / float(1 << 32) * 2 - 1).astype(np.float32)
    return solid, tex


def faceted_mesh(shape, porosity, rng, crystal_px=28.0, **kw):
    """Faceted crystals: solid Voronoi polyhedra with pore walls between them.

    Euclidean Voronoi cells are convex polyhedra with flat faces and sharp
    edges by construction -- the look a sphere pack cannot produce. Shrinking
    each cell away from its walls leaves the crystal and opens the pore.
    """
    nz, ny, nx = shape
    n = max(6, int(nz * ny * nx / (crystal_px ** 3)))
    seeds = np.stack([rng.uniform(0, nz, n), rng.uniform(0, ny, n),
                      rng.uniform(0, nx, n)], 1).astype(np.float32)
    sid, wall_dist = _voronoi_cells(shape, seeds, rng)
    wall_dist = wall_dist + 0.4 * _grf(shape, (1.5, 1.5, 1.5), rng, octaves=1)
    solid = wall_dist >= _level_for_porosity(wall_dist, porosity)
    # channeling contrast: each crystal's lattice is oriented differently,
    # so each grain returns its own signal level -- the strongest cue that a
    # polycrystal is 3-D grains and not a drawn mesh (v7). Texture in
    # [-1, 1], one value per grain, from a hash of the grain id.
    h = (sid.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)
    tex = (h.astype(np.float64) / float(1 << 32) * 2 - 1).astype(np.float32)
    return solid, tex


def fiber_felt(shape, porosity, rng, radius=3.0, in_plane=True, **kw):
    """Random cylinders -- the PTL / gas-diffusion-layer fibre mat.

    Fibres lie mostly in-plane, as a calendered felt does, so a z-slice cuts
    them as long streaks rather than circles.

    Each fibre is a finite segment rasterised inside its own bounding box, so
    cost scales with fibre volume rather than stack volume. The segment
    length is capped: a fibre spanning the full 1860 px width of a real frame
    gives every one of them a bounding box the size of the whole volume, and
    filling the stack then takes thousands of full-volume passes.
    """
    nz, ny, nx = shape
    solid = np.zeros(shape, bool)
    target_solid = 1.0 - porosity
    span = float(min(max(ny, nx), max(60.0, 36.0 * radius)))

    check_every = 64
    for i in range(200000):
        p0 = np.array([rng.uniform(0, nz), rng.uniform(0, ny), rng.uniform(0, nx)],
                      np.float32)
        if in_plane:
            d = np.array([rng.normal(0, 0.18), rng.normal(0, 1.0), rng.normal(0, 1.0)],
                         np.float32)
        else:
            d = rng.standard_normal(3).astype(np.float32)
        d /= np.linalg.norm(d) + 1e-9

        a = p0 - d * span / 2
        b = p0 + d * span / 2
        lo = np.floor(np.minimum(a, b) - radius - 1).astype(int)
        hi = np.ceil(np.maximum(a, b) + radius + 1).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(shape))
        if np.any(hi <= lo):
            continue

        zz, yy, xx = np.meshgrid(np.arange(lo[0], hi[0]),
                                 np.arange(lo[1], hi[1]),
                                 np.arange(lo[2], hi[2]), indexing="ij")
        rel = np.stack([zz - p0[0], yy - p0[1], xx - p0[2]], -1).astype(np.float32)
        t = (rel * d).sum(-1)
        perp = rel - t[..., None] * d
        hit = (perp * perp).sum(-1) <= radius * radius
        solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= hit

        # counting the whole volume every fibre is itself a full-volume pass
        if i % check_every == check_every - 1 and solid.mean() >= target_solid:
            break
    return _match_porosity(solid, porosity, rng)


def fibre(shape, porosity, rng, r_med=4.0, r_spread=0.35, thick_frac=0.08,
          thick_r=(9.0, 16.0), seg=1.0, bend=0.035, tilt=0.10, decorate=0.0,
          binder=0.0, tex_amp=0.08, rim=0.8, rim_len=2.0, **kw):
    """Fibre mat: curved fibres of varying radius lying in the layer plane.

    Replaces fiber_felt, which had three things wrong with it:

      orientation  the layer plane is x-z (y is depth through the layer), so
                   a laid-down mat has fibres with a small y component. The
                   old fibres lay in the y-x image plane, i.e. stood across
                   the layer, and every slice showed full-length sticks.
                   Here a slice cuts most fibres obliquely: round or oval
                   sections where a fibre runs along z, long streaks only
                   where one runs along x, and they evolve slice to slice.
      shape        fibres follow a smooth random walk in direction (`bend`
                   rad per px, mostly in-plane) instead of being straight
                   rods, and radius is lognormal per fibre with a few thick
                   ones, instead of one radius for all.
      appearance   SE edge glow and a faint grain (se_texture), so a fibre
                   section reads as a bright-rimmed disc, not a flat stick.

    decorate > 0 coats fibres with a rough particle shell of that thickness
    (catalyst on fibres); binder > 0 webs fibres together where they nearly
    touch. Built by drawing centrelines and growing them with one distance
    transform, so cost does not depend on how many fibres there are.
    Returns (solid, texture).
    """
    nz, ny, nx = shape
    vol = float(nz * ny * nx)
    target = (1.0 - porosity) * 1.15            # overlaps eat some volume
    centre = np.zeros(shape, bool)
    rad_id = np.zeros(shape, np.float32)
    filled = 0.0
    while filled < target * vol:
        r = (rng.uniform(*thick_r) if rng.random() < thick_frac
             else r_med * float(np.exp(rng.normal(0, r_spread))))
        th = rng.uniform(0, np.pi)               # in-plane heading, x-z
        ph = rng.normal(0, tilt)                 # small climb in y
        p = np.array([rng.uniform(-0.2 * nz, 1.2 * nz), rng.uniform(0, ny),
                      rng.uniform(-0.2 * nx, 1.2 * nx)])
        L = rng.uniform(0.6, 1.4) * 2.0 * max(nz, nx)
        steps = int(L / seg)
        dth = np.cumsum(rng.normal(0, bend * np.sqrt(seg), steps))
        dph = np.cumsum(rng.normal(0, 0.3 * bend * np.sqrt(seg), steps))
        a = th + dth
        b = np.clip(ph + dph, -0.5, 0.5)
        d = np.stack([np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)], 1)
        pts = p + np.cumsum(d * seg, 0)
        # fibres reflect off the layer's top and bottom rather than leaving
        pts[:, 1] = np.abs(np.mod(pts[:, 1], 2 * ny) - ny) * -1 + ny
        idx = np.rint(pts).astype(int)
        ok = ((idx[:, 0] >= 0) & (idx[:, 0] < nz) & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
              & (idx[:, 2] >= 0) & (idx[:, 2] < nx))
        idx = idx[ok]
        if len(idx) == 0:
            continue
        centre[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        rad_id[idx[:, 0], idx[:, 1], idx[:, 2]] = np.maximum(
            rad_id[idx[:, 0], idx[:, 1], idx[:, 2]], r)
        filled += np.pi * r * r * len(idx) * seg
    dist, ind = ndi.distance_transform_edt(~centre, return_indices=True)
    rnear = rad_id[ind[0], ind[1], ind[2]]
    del ind, centre, rad_id
    field = (rnear - dist).astype(np.float32)    # >0 inside a fibre
    del dist, rnear
    if decorate > 0:
        shell = _grf(shape, (1.2, 1.2, 1.2), rng, octaves=2, gain=0.5)
        field = np.maximum(field, np.minimum(field + decorate, 0.0) + decorate * 0.6 * shell)
        del shell
    if binder > 0:
        sm = ndi.gaussian_filter(np.clip(field, -3, 3), 2.0)
        field = np.maximum(field, (sm + 3.0 - 3.0 / max(binder, 1e-3)) * 2.0)
    solid = field >= _level_for_porosity(field, porosity)
    del field
    return solid, se_texture(solid, rng, tex_amp, rim, rim_len)


def sheet_felt(shape, porosity, rng, thickness=2.5, n_sheets=None, **kw):
    """Bent sheets dropped and allowed to settle.

    The synth-to-real iteration log records this as a change that worked:
    placing sheets at random looks wrong, letting them fall and rest on what
    is already there gives the stacked, draped arrangement a real felt has.
    """
    nz, ny, nx = shape
    n = n_sheets or max(6, int(nz / (2.5 * thickness)))
    solid = np.zeros(shape, bool)
    height = np.zeros((ny, nx), np.float32)   # top of the pile, per column

    for _ in range(n):
        # a gently bent sheet: smooth height offset over y-x
        bend = _grf((ny, nx), (max(ny / 8, 4.0), max(nx / 8, 4.0)), rng, octaves=2)
        bend = bend / (np.abs(bend).max() + 1e-9) * rng.uniform(2.0, 6.0)
        # drop it until it just rests on the current pile
        rest = (height + bend).max()
        z0 = height + bend + (nz * 0.04) - (rest - height.max())
        z0 = np.clip(z0, 0, nz - 1)
        zz = np.arange(nz)[:, None, None]
        sheet = (zz >= z0[None]) & (zz < (z0 + thickness)[None])
        solid |= sheet
        height = np.maximum(height, z0 + thickness)
        if 1.0 - solid.mean() <= porosity:
            break

    # settle: close the few voxel-scale gaps contact would close
    solid = ndi.binary_closing(solid, np.ones((2, 2, 2)))
    # trim or grow to hit the porosity target exactly
    return _match_porosity(solid, porosity, rng)


def _match_porosity(solid: np.ndarray, porosity: float, rng) -> np.ndarray:
    """Nudge a binary structure onto the target porosity.

    Uses the distance transform rather than repeated dilation so the
    correction is a smooth offset of the interface, not a blocky one.
    """
    cur = 1.0 - solid.mean()
    if abs(cur - porosity) < 0.002:
        return solid
    din = ndi.distance_transform_edt(solid)
    dout = ndi.distance_transform_edt(~solid)
    signed = din - dout            # >0 inside solid
    return signed >= _level_for_porosity(signed, porosity)


FAMILIES = {
    "agglomerate": agglomerate,
    "compressed": compressed,
    "spinodal": spinodal,
    "catalyst": catalyst,
    "spheres": spheres,
    "mudcrack": mudcrack,
    "mudcrack_2d": mudcrack_2d,
    "faceted": faceted,
    "faceted_mesh": faceted_mesh,
    "fiber_felt": fiber_felt,
    "fibre": fibre,
    "sheet_felt": sheet_felt,
}


def build(family: str, shape, porosity: float, seed: int, with_texture=False, **kw):
    """Solid mask (True = solid); with_texture=True returns (solid, texture),
    texture None for families that do not make one."""
    if family not in FAMILIES:
        raise SystemExit(f"unknown family {family!r}; have {sorted(FAMILIES)}")
    rng = np.random.default_rng(seed)
    out = FAMILIES[family](shape, porosity, rng, **kw)
    solid, tex = out if isinstance(out, tuple) else (out, None)
    got = 1.0 - solid.mean()
    if abs(got - porosity) > 0.01:
        solid = _match_porosity(solid, porosity, rng)
    return (solid, tex) if with_texture else solid


def bright_phase(solid: np.ndarray, frac: float, seed: int,
                 corr=(3.0, 2.2, 2.2)) -> np.ndarray:
    """A brighter minority phase occupying `frac` of the solid.

    A catalyst layer is not one material. Platinum sits on carbon and returns
    far more signal, so the solid phase in a real micrograph is a mix of very
    bright specks and dimmer grey, not a single level. That mixture is a
    large part of why a single grey threshold tops out near 0.89 balanced
    accuracy on the real stacks: some solid is darker than some shine-through.

    Both phases are still solid in the answer key -- this changes how the
    material looks, not what it is.
    """
    if frac <= 0:
        return np.zeros_like(solid)
    rng = np.random.default_rng(seed + 977)
    f = _grf(solid.shape, corr, rng, octaves=2, gain=0.45)
    if not solid.any():
        return np.zeros_like(solid)
    # sample inside the solid phase rather than materialising all of it
    flat_f, flat_s = f.reshape(-1), solid.reshape(-1)
    if flat_f.size > 4_000_000:
        idx = rng.integers(0, flat_f.size, 8_000_000)
        sel = flat_s[idx]
        inside = flat_f[idx][sel]
    else:
        inside = flat_f[flat_s]
    if inside.size == 0:
        return np.zeros_like(solid)
    cut = float(np.quantile(inside, 1.0 - frac))
    return solid & (f >= cut)


def shear_structure(solid: np.ndarray, shear: float) -> np.ndarray:
    """Tilt the structure so its long axis follows the beam's slanted path.

    The simulator reads slice z's buried signal along the ray
    (z+k, y + shear*k), so for pore channels to run *along* that ray -- which
    is what makes shine-through slide smoothly instead of flickering -- the
    structure's elongation axis has to be tilted to match. Rolling slice z by
    -shear*z does exactly that, and because the generating field is periodic
    the wrap introduces no seam.

    Without this, a shear of 1 px/slice makes the ray cut across pore
    channels, and the depth-to-first-solid statistic collapses onto the
    in-plane correlation length (measured: caps at ~5.2 slices, where the
    real stacks reach 5.7 to 7.6).
    """
    if not shear:
        return solid
    out = np.empty_like(solid)
    for z in range(solid.shape[0]):
        out[z] = np.roll(solid[z], -int(round(shear * z)), axis=0)
    return out
