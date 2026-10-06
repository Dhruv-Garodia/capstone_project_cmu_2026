"""Microstructure metrics computed from a segmentation.

All functions accept a boolean ``pore`` array (2D image or 3D volume) and an optional
boolean ``roi`` (catalyst-layer region). Lengths are returned in pixels unless a
``pixel_nm`` is given, in which case nanometres are returned as well.

Conventions for the comp_full stack: y (rows) is the through-plane direction of the
catalyst layer, x (columns) is in-plane, z is the milling direction.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve


# --------------------------------------------------------------------------- basics
def porosity(pore: np.ndarray, roi: np.ndarray | None = None) -> float:
    roi = np.ones_like(pore, bool) if roi is None else roi
    n = roi.sum()
    return float((pore & roi).sum() / n) if n else float("nan")


def local_thickness(pore: np.ndarray, max_radius: int | None = None) -> np.ndarray:
    """Local thickness (diameter of the largest inscribed sphere/disc covering each pore voxel).

    Uses the classic Hildebrand-Rüegsegger definition implemented with distance
    transforms: for each radius r, voxels within r of a centre with EDT >= r get 2r.
    """
    pore = pore.astype(bool)
    if not pore.any():
        return np.zeros(pore.shape, np.float32)
    dt = ndi.distance_transform_edt(pore)
    rmax = int(np.ceil(dt.max())) if max_radius is None else int(max_radius)
    out = np.zeros(pore.shape, np.float32)
    radii = np.unique(np.clip(np.round(np.geomspace(1, max(rmax, 1), num=min(rmax, 40))), 1, None)).astype(int)
    for r in radii:
        seeds = dt >= r
        if not seeds.any():
            break
        covered = ndi.distance_transform_edt(~seeds) < r
        out[covered & pore] = 2.0 * r
    # pixels thinner than the smallest disc still have thickness ~ 2*EDT
    thin = pore & (out == 0)
    out[thin] = 2.0 * dt[thin]
    return out


def pore_size_distribution(pore: np.ndarray, roi: np.ndarray | None = None, bins: int = 30,
                           pixel_nm: float = 1.0) -> dict:
    """Volume-weighted pore size distribution from local thickness (porespy convention)."""
    lt = local_thickness(pore)
    vals = lt[pore & (roi if roi is not None else True)] * pixel_nm
    if vals.size == 0:
        return {"bin_edges": [], "volume_fraction": [], "d50": float("nan"), "mean": float("nan")}
    edges = np.linspace(0, vals.max() + 1e-6, bins + 1)
    hist, _ = np.histogram(vals, edges)
    frac = hist / hist.sum()
    return {
        "bin_edges": edges.tolist(),
        "volume_fraction": frac.tolist(),
        "d10": float(np.percentile(vals, 10)),
        "d50": float(np.percentile(vals, 50)),
        "d90": float(np.percentile(vals, 90)),
        "mean": float(vals.mean()),
        "_local_thickness": lt,
    }


def chord_lengths(phase: np.ndarray, axis: int, roi: np.ndarray | None = None) -> np.ndarray:
    """All chord (run) lengths of ``phase`` along ``axis``; chords cut by the ROI edge are dropped."""
    phase = phase.astype(bool)
    valid = np.ones_like(phase) if roi is None else roi.astype(bool)
    p = np.moveaxis(phase & valid, axis, -1).reshape(-1, phase.shape[axis])
    v = np.moveaxis(valid, axis, -1).reshape(-1, phase.shape[axis])
    pad = np.zeros((p.shape[0], 1), bool)
    d = np.diff(np.concatenate([pad, p, pad], 1).astype(np.int8), axis=1)
    rows_s, cols_s = np.nonzero(d == 1)
    rows_e, cols_e = np.nonzero(d == -1)
    lengths = cols_e - cols_s
    # discard chords touching the ROI boundary or array edge (truncated)
    vpad = np.concatenate([pad, v, pad], 1)
    ok = vpad[rows_s, cols_s] & vpad[rows_e, cols_e + 1]
    return lengths[ok]


def specific_surface(pore: np.ndarray, roi: np.ndarray | None = None, pixel_nm: float = 1.0) -> float:
    """Pore/solid interface per unit ROI area (2D, 1/nm) or volume (3D, 1/nm).

    Counts phase changes between face neighbours and applies the Cauchy-Crofton
    correction (pi/4 in 2D, 2/3 in 3D) for staircase over-estimation.
    """
    roi = np.ones_like(pore, bool) if roi is None else roi
    faces = 0
    for ax in range(pore.ndim):
        a = np.swapaxes(pore, 0, ax)
        r = np.swapaxes(roi, 0, ax)
        both = r[1:] & r[:-1]
        faces += int(((a[1:] != a[:-1]) & both).sum())
    corr = np.pi / 4 if pore.ndim == 2 else 2.0 / 3.0
    return float(faces * corr / max(roi.sum(), 1) / pixel_nm)


def connectivity(pore: np.ndarray, roi: np.ndarray | None = None, axis: int = 0) -> dict:
    """Connected-component statistics and percolation along ``axis``."""
    pore = pore & (roi if roi is not None else True)
    lab, n = ndi.label(pore)
    if n == 0:
        return {"n_components": 0, "largest_fraction": 0.0, "percolating_fraction": 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    first = np.take(lab, 0, axis=axis)
    last = np.take(lab, -1, axis=axis)
    if roi is not None:
        # percolation measured between the ROI's own faces along `axis`
        idx = np.arange(pore.shape[axis]).reshape([-1 if i == axis else 1 for i in range(pore.ndim)])
        roi_b = np.broadcast_to(roi, pore.shape)
        top = np.where(roi_b, idx, np.iinfo(np.int32).max).min(axis=axis, keepdims=True)
        bot = np.where(roi_b, idx, -1).max(axis=axis, keepdims=True)
        near_top = (idx <= top + 2) & roi_b
        near_bot = (idx >= bot - 2) & roi_b
        first = lab[near_top & pore]
        last = lab[near_bot & pore]
    perc = np.intersect1d(np.unique(first), np.unique(last))
    perc = perc[perc > 0]
    return {
        "n_components": int(n),
        "largest_fraction": float(sizes.max() / sizes.sum()),
        "percolating_fraction": float(sizes[perc - 1].sum() / sizes.sum()) if perc.size else 0.0,
        "isolated_fraction": float(sizes[sizes < 10].sum() / sizes.sum()),
    }


# ----------------------------------------------------------------- tortuosity factor
def tortuosity_factor(pore: np.ndarray, axis: int = 0, max_iter: int = 4000) -> dict:
    """Diffusive tortuosity factor (TauFactor definition) along ``axis``.

    Solves Laplace's equation for concentration inside the pore phase with c=1 on the
    first face and c=0 on the last face (no-flux elsewhere), then
    D_eff/D = flux * L / (A * dc) and tau = porosity / (D_eff/D).
    Works on 2D images and modest 3D volumes (finite-volume, 2-/6-neighbour stencil).
    """
    pore = np.moveaxis(pore.astype(bool), axis, 0)
    eps = pore.mean()
    shape = pore.shape
    # keep only pore connected to both faces (dead ends carry no flux but slow CG)
    lab, _ = ndi.label(pore)
    good = np.intersect1d(np.unique(lab[0]), np.unique(lab[-1]))
    good = good[good > 0]
    if good.size == 0:
        return {"tau": float("inf"), "deff_ratio": 0.0, "porosity": float(eps), "percolates": False}
    active = np.isin(lab, good)
    idx = -np.ones(shape, np.int64)
    idx[active] = np.arange(active.sum())
    n = int(active.sum())
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    diag = np.zeros(n)
    coords = np.nonzero(active)
    me = idx[coords]
    for ax in range(pore.ndim):
        for step in (-1, 1):
            nb = list(coords)
            nb[ax] = nb[ax] + step
            inside = (nb[ax] >= 0) & (nb[ax] < shape[ax])
            nbi = np.full(me.shape, -1)
            sel = [c[inside] for c in nb]
            nbi[inside] = idx[tuple(sel)]
            conn = nbi >= 0
            rows.append(me[conn]); cols.append(nbi[conn]); vals.append(-np.ones(conn.sum()))
            diag[me[conn]] += 1
            if ax == 0:
                # Dirichlet faces: half-cell to the boundary (conductance 2)
                at_face = ~inside
                if step == -1:
                    diag[me[at_face]] += 2; b[me[at_face]] += 2 * 1.0
                else:
                    diag[me[at_face]] += 2
    A = sparse.csr_matrix((np.concatenate(vals + [diag]),
                           (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                          shape=(n, n))
    if n < 250_000 and pore.ndim == 2:
        c = spsolve(A.tocsc(), b)
    else:
        c, _ = cg(A, b, maxiter=max_iter, rtol=1e-6) if "rtol" in cg.__code__.co_varnames else cg(A, b, maxiter=max_iter)
    conc = np.zeros(shape)
    conc[active] = c
    inflow = (2 * (1.0 - conc[0][active[0]])).sum()
    L = shape[0]
    area = np.prod(shape[1:])
    deff = inflow * L / area  # dc = 1
    tau = eps / deff if deff > 0 else float("inf")
    return {"tau": float(tau), "deff_ratio": float(deff), "porosity": float(eps), "percolates": True}


# ---------------------------------------------------------------- ROI-aware helpers
def largest_inner_rectangle(roi: np.ndarray, min_frac: float = 0.98) -> tuple[slice, slice]:
    """Crude axis-aligned box well inside a band-shaped ROI (rows where most columns are ROI)."""
    rows = np.nonzero(roi.mean(1) >= min_frac)[0]
    if rows.size < 8:
        rows = np.nonzero(roi.mean(1) >= 0.8)[0]
    if rows.size == 0:
        return slice(0, 0), slice(0, 0)
    # longest contiguous run of good rows
    splits = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
    run = max(splits, key=len)
    rs = slice(int(run[0]), int(run[-1]) + 1)
    cols = np.nonzero(roi[rs].all(0))[0]
    if cols.size == 0:
        return rs, slice(0, 0)
    splits = np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1)
    cc = max(splits, key=len)
    return rs, slice(int(cc[0]), int(cc[-1]) + 1)


def depth_profile(pore: np.ndarray, roi: np.ndarray, bins: int = 20) -> dict:
    """Porosity versus normalised depth through the catalyst layer (0 = top face, 1 = bottom face).

    Works column by column, so a wavy CL is unrolled before averaging.
    """
    h = roi.shape[0]
    yy = np.arange(h)[:, None]
    has = roi.any(0)
    top = np.where(roi, yy, h).min(0)
    bot = np.where(roi, yy, -1).max(0)
    thick = np.where(has, bot - top + 1, 0)
    depth = (yy - top[None]) / np.maximum(thick[None], 1)
    sel = roi & has[None]
    b = np.clip((depth[sel] * bins).astype(int), 0, bins - 1)
    num = np.bincount(b, weights=pore[sel].astype(float), minlength=bins)
    den = np.bincount(b, minlength=bins)
    prof = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return {
        "depth": ((np.arange(bins) + 0.5) / bins).tolist(),
        "porosity": prof.tolist(),
        "thickness_px_mean": float(thick[has].mean()) if has.any() else 0.0,
        "thickness_px_std": float(thick[has].std()) if has.any() else 0.0,
    }


def rev_curve(pore: np.ndarray, roi: np.ndarray | None = None, sizes=(16, 32, 64, 96, 128, 192, 256),
              n_samples: int = 40, seed: int = 0) -> dict:
    """Representative elementary volume: porosity mean/std of random boxes of growing size."""
    rng = np.random.default_rng(seed)
    roi = np.ones_like(pore, bool) if roi is None else roi
    out = {"size": [], "mean": [], "std": []}
    for s in sizes:
        if any(s > d for d in pore.shape[-2:]):
            break
        vals = []
        for _ in range(n_samples * 4):
            y = rng.integers(0, pore.shape[-2] - s + 1)
            x = rng.integers(0, pore.shape[-1] - s + 1)
            r = roi[..., y:y + s, x:x + s]
            if r.mean() < 0.999:
                continue
            vals.append(pore[..., y:y + s, x:x + s].mean())
            if len(vals) >= n_samples:
                break
        if len(vals) >= 5:
            out["size"].append(int(s)); out["mean"].append(float(np.mean(vals))); out["std"].append(float(np.std(vals)))
    return out


# --------------------------------------------------------------------- one-call API
def analyze(classes: np.ndarray, pixel_nm: float = 6.0, compute_tortuosity: bool = True) -> dict:
    """Full metric set for a class map (0 solid, 1 pore, 2 outside), 2D or 3D."""
    roi = classes != 2
    pore = classes == 1
    res: dict = {"pixel_nm": pixel_nm, "dims": int(classes.ndim)}
    res["porosity"] = porosity(pore, roi)
    res["roi_fraction"] = float(roi.mean())
    psd = pore_size_distribution(pore, roi, pixel_nm=pixel_nm)
    psd.pop("_local_thickness", None)
    res["pore_size_nm"] = psd
    ax_names = ["y_through_plane", "x_in_plane"] if classes.ndim == 2 else ["z_milling", "y_through_plane", "x_in_plane"]
    res["mean_chord_nm"] = {}
    for ax, name in enumerate(ax_names):
        pc = chord_lengths(pore, ax, roi)
        sc = chord_lengths(~pore & roi, ax, roi)
        res["mean_chord_nm"][name] = {"pore": float(pc.mean() * pixel_nm) if pc.size else float("nan"),
                                      "solid": float(sc.mean() * pixel_nm) if sc.size else float("nan")}
    res["specific_surface_per_um"] = specific_surface(pore, roi, pixel_nm) * 1000.0
    res["connectivity"] = connectivity(pore, roi, axis=classes.ndim - 2)
    plane_roi = roi if classes.ndim == 2 else roi.all(0)
    if classes.ndim == 2:
        prof = depth_profile(pore, roi)
    else:
        profs = [depth_profile(p, r) for p, r in zip(pore, roi)]
        prof = {"depth": profs[0]["depth"],
                "porosity": np.nanmean([p["porosity"] for p in profs], 0).tolist(),
                "thickness_px_mean": float(np.mean([p["thickness_px_mean"] for p in profs])),
                "thickness_px_std": float(np.mean([p["thickness_px_std"] for p in profs]))}
    prof["thickness_nm_mean"] = prof["thickness_px_mean"] * pixel_nm
    res["depth_profile"] = prof
    res["rev"] = rev_curve(pore, roi)
    if compute_tortuosity:
        rs, cs = largest_inner_rectangle(plane_roi)
        cap = 640 if classes.ndim == 2 else 192  # keep the linear solve tractable (sparse direct in 2D, CG in 3D)
        if cs.stop - cs.start > cap:
            mid = (cs.start + cs.stop) // 2
            cs = slice(mid - cap // 2, mid + cap // 2)
        res["tortuosity"] = {}
        if rs.stop - rs.start >= 16 and cs.stop - cs.start >= 16:
            box = pore[..., rs, cs]
            if classes.ndim == 2:
                res["tortuosity"]["through_plane_y"] = tortuosity_factor(box, axis=0)
                res["tortuosity"]["in_plane_x"] = tortuosity_factor(box, axis=1)
                # a 2D section rarely has a pore path across the layer (2D percolation threshold ~0.59),
                # so also report the solid (electron-conduction) phase
                res["tortuosity"]["solid_through_plane_y"] = tortuosity_factor(~box, axis=0)
            else:
                res["tortuosity"]["through_plane_y"] = tortuosity_factor(box, axis=1)
                res["tortuosity"]["in_plane_x"] = tortuosity_factor(box, axis=2)
                res["tortuosity"]["milling_z"] = tortuosity_factor(box, axis=0)
            res["tortuosity"]["box_px"] = [int(rs.start), int(rs.stop), int(cs.start), int(cs.stop)]
    return res
