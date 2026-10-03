"""Catalyst-layer region of interest.

The catalyst layer is the only strongly textured band in the frame: the Pt cap
above it and the membrane below it are smooth. We threshold a local-texture map,
take per-column top/bottom boundaries, smooth them along x and z, and shrink the
band by a margin so the Pt-infiltrated top fringe and the membrane edge stay out.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

TEXTURE_WIN = 21      # px, local-std window
TEXTURE_SMOOTH = 6    # px, gaussian on the texture map
X_MEDIAN = 51         # px, boundary smoothing along x
Z_MEDIAN = 9          # slices, boundary smoothing along z
MARGIN = 30           # px, inward shrink of the band (~180 nm at 6 nm/px)
SIDE_MARGIN = 40      # px, trimmed from each lateral end (trench side walls)
EDGE_VOID_MIN = 1500  # px, dark regions this big that touch the band edge are gaps, not pores
EDGE_VOID_NECK = 7    # px, opening radius that separates a gap from pores connected to it
EDGE_VOID_DEPTH = 90  # px, gaps are only removed this far into the band
MIN_THICKNESS = 60    # px, columns thinner than this are dropped


def texture_map(img):
    g = ndi.gaussian_filter(img.astype(np.float32), 1.0)
    mu = ndi.uniform_filter(g, TEXTURE_WIN)
    var = ndi.uniform_filter(g * g, TEXTURE_WIN) - mu * mu
    return ndi.gaussian_filter(np.sqrt(np.maximum(var, 0)), TEXTURE_SMOOTH)


def texture_threshold(sample_slices):
    """One texture threshold per stack, from a sample of slices."""
    vals = np.concatenate([texture_map(s)[::4, ::4].ravel() for s in sample_slices])
    return float(threshold_otsu(vals))


def raw_boundaries(img, tex_thr):
    """Per-column (top, bottom) rows of the textured band; NaN where absent."""
    valid = img > 0                                   # zero padding in some stacks
    tex = (texture_map(img) > tex_thr) & valid
    tex = ndi.binary_closing(tex, structure=np.ones((15, 15)), border_value=0)
    tex = ndi.binary_fill_holes(tex)
    tex = ndi.binary_opening(tex, structure=np.ones((11, 11)))
    lab, n = ndi.label(tex)
    if n == 0:
        nan = np.full(img.shape[1], np.nan)
        return nan, nan.copy()
    sizes = ndi.sum(tex, lab, range(1, n + 1))
    tex = lab == (1 + int(np.argmax(sizes)))
    any_col = tex.any(axis=0)
    top = np.where(any_col, tex.argmax(axis=0), np.nan).astype(np.float32)
    bot = np.where(any_col, img.shape[0] - 1 - tex[::-1].argmax(axis=0), np.nan).astype(np.float32)
    return top, bot


def _nanmedian_filter(arr, size, axis):
    pad = size // 2
    padded = np.pad(arr, [(pad, pad) if a == axis else (0, 0) for a in range(arr.ndim)],
                    mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis)
    with np.errstate(all="ignore"):
        return np.nanmedian(win, axis=-1)


def smooth_boundaries(tops, bots):
    """tops/bots: (Z, W) arrays. Median along x then along z."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tops = _nanmedian_filter(_nanmedian_filter(tops, X_MEDIAN, 1), Z_MEDIAN, 0)
        bots = _nanmedian_filter(_nanmedian_filter(bots, X_MEDIAN, 1), Z_MEDIAN, 0)
    return tops, bots


def roi_from_boundaries(top, bot, shape, margin=MARGIN):
    rows = np.arange(shape[0], dtype=np.float32)[:, None]
    t, b = top[None, :] + margin, bot[None, :] - margin
    ok = np.isfinite(t) & np.isfinite(b) & ((b - t) >= MIN_THICKNESS)
    cols = np.flatnonzero(ok[0])
    if cols.size:
        ok[0, :cols[0] + SIDE_MARGIN] = False
        ok[0, cols[-1] - SIDE_MARGIN + 1:] = False
    return (rows >= t) & (rows <= b) & ok


def drop_edge_voids(roi, dark):
    """Remove delamination gaps: large dark regions opening onto the band edge.

    The dark mask is opened first so a gap does not drag connected interior pores
    out with it, and removal is capped at EDGE_VOID_DEPTH px from the edge.
    """
    core = ndi.binary_opening(dark & roi, structure=_disk(EDGE_VOID_NECK))
    lab, n = ndi.label(core)
    if n == 0:
        return roi
    edge = roi & ~ndi.binary_erosion(roi, iterations=2)
    touching = [l for l in np.unique(lab[edge]) if l != 0]
    if not touching:
        return roi
    sizes = ndi.sum(core, lab, touching)
    kill = [l for l, sz in zip(touching, sizes) if sz >= EDGE_VOID_MIN]
    if not kill:
        return roi
    void = ndi.binary_dilation(np.isin(lab, kill), structure=_disk(EDGE_VOID_NECK)) & dark
    near_edge = ndi.distance_transform_edt(roi) <= EDGE_VOID_DEPTH
    return roi & ~(void & near_edge)


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r
