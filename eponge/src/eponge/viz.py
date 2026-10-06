"""The three post-segmentation visualisations.

1. ``plot_segmentation_map``  - raw image, class overlay, and a pore-size (local thickness) map.
2. ``plot_quantitative``      - pore size distribution, porosity depth profile through the CL
                                (with uncertainty band) and the REV convergence curve.
3. ``render_3d``              - interactive 3D reconstruction of the pore network (plotly HTML);
                                for a single 2D image a
                                connectivity / percolation map is produced instead
                                (``plot_connectivity``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

from . import metrics as M

CLASS_COLORS = {"solid": "#d9d4c7", "pore": "#1f6feb", "outside": "#00000000"}


def overlay(image: np.ndarray, classes: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgb = np.repeat(image[..., None], 3, -1).astype(np.float32) / 255.0
    pore = classes == 1
    out = classes == 2
    rgb[pore] = rgb[pore] * (1 - alpha) + np.array([0.12, 0.44, 0.92]) * alpha
    rgb[out] = rgb[out] * 0.45
    edge = (classes != 2) ^ ndi.binary_erosion(classes != 2, iterations=2)
    rgb[edge] = [1.0, 0.75, 0.1]
    return np.clip(rgb, 0, 1)


def plot_segmentation_map(image: np.ndarray, classes: np.ndarray, pixel_nm: float = 6.0,
                          path: str | Path | None = None, title: str = ""):
    """Visualisation 1."""
    lt = M.local_thickness(classes == 1) * pixel_nm
    lt_masked = np.ma.masked_where(classes != 1, lt)
    h = image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(3, 1, figsize=(12, 12 * h * 3 + 1.4))
    fig.subplots_adjust(left=0.01, right=0.93, top=0.95, bottom=0.01, hspace=0.12)
    ax[0].imshow(image, cmap="gray"); ax[0].set_title("pFIB-SEM input")
    ax[1].imshow(overlay(image, classes)); ax[1].set_title("Segmentation: pore (blue), solid, outside CL (dimmed), CL boundary (amber)")
    ax[2].imshow(image, cmap="gray", alpha=0.6)
    im = ax[2].imshow(lt_masked, cmap="viridis", vmin=0, vmax=np.percentile(lt[classes == 1], 99) if (classes == 1).any() else 1)
    ax[2].set_title("Local pore diameter (nm)")
    cax = ax[2].inset_axes([1.01, 0.1, 0.015, 0.8])
    fig.colorbar(im, cax=cax, label="nm")
    for a in ax:
        a.axis("off")
    if title:
        fig.suptitle(title)
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return fig


def plot_quantitative(results: dict, path: str | Path | None = None, porosity_band: dict | None = None):
    """Visualisation 2: PSD, depth profile, REV. ``results`` is the output of ``metrics.analyze``."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    psd = results["pore_size_nm"]
    if psd["bin_edges"]:
        e = np.array(psd["bin_edges"]); c = (e[1:] + e[:-1]) / 2
        ax[0].bar(c, np.array(psd["volume_fraction"]) * 100, width=np.diff(e), color="#1f6feb", alpha=0.85)
        ax[0].axvline(psd["d50"], color="k", ls="--", lw=1, label=f"d50 = {psd['d50']:.0f} nm")
        ax[0].legend()
    ax[0].set(xlabel="pore diameter (nm)", ylabel="pore volume (%)", title="Pore size distribution (local thickness)")
    prof = results["depth_profile"]
    d = np.array(prof["depth"]); p = np.array(prof["porosity"])
    ax[1].plot(p, d, "-o", color="#1f6feb", ms=3)
    if porosity_band is not None and "profile_p10" in porosity_band:
        ax[1].fill_betweenx(d, porosity_band["profile_p10"], porosity_band["profile_p90"], color="#1f6feb", alpha=0.2,
                            label="10-90 % band")
        ax[1].legend()
    ax[1].axvline(results["porosity"], color="k", ls="--", lw=1)
    ax[1].invert_yaxis()
    ax[1].set(xlabel="porosity", ylabel="normalised depth (0 = top of CL)",
              title=f"Porosity through CL (thickness {prof.get('thickness_nm_mean', 0)/1000:.2f} um)")
    rev = results["rev"]
    if rev["size"]:
        s = np.array(rev["size"]) * results["pixel_nm"]
        ax[2].errorbar(s, rev["mean"], yerr=rev["std"], fmt="-o", color="#1f6feb", capsize=3)
    ax[2].set(xlabel="window size (nm)", ylabel="porosity", title="Representative elementary area / volume")
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return fig


def plot_connectivity(classes: np.ndarray, path: str | Path | None = None):
    """2D stand-in for visualisation 3: pore clusters, percolating (top-to-bottom) ones highlighted."""
    roi = classes != 2
    pore = classes == 1
    lab, n = ndi.label(pore)
    yy = np.arange(classes.shape[0])[:, None]
    top = np.where(roi, yy, 10 ** 6).min(0); bot = np.where(roi, yy, -1).max(0)
    near_t = roi & (yy <= top + 2); near_b = roi & (yy >= bot - 2)
    perc = np.intersect1d(lab[near_t & pore], lab[near_b & pore]); perc = perc[perc > 0]
    rgb = np.zeros(classes.shape + (3,))
    rgb[roi] = 0.85
    rng = np.random.default_rng(1)
    pal = rng.uniform(0.25, 0.65, (n + 1, 3)); pal[:, 2] = 0.5
    rgb[pore] = pal[lab[pore]] * 0.6
    rgb[np.isin(lab, perc)] = [0.9, 0.3, 0.1]
    fig, ax = plt.subplots(figsize=(12, 12 * classes.shape[0] / classes.shape[1] + 0.6), constrained_layout=True)
    ax.imshow(rgb); ax.axis("off")
    ax.set_title(f"Pore connectivity: {n} clusters; through-CL percolating clusters in orange")
    if path:
        fig.savefig(path, dpi=150); plt.close(fig)
    return fig


def render_3d(classes: np.ndarray, path: str | Path, pixel_nm: float = 6.0, z_nm: float | None = None,
              max_side: int = 120) -> Path:
    """Visualisation 3: interactive 3D pore network (plotly isosurface of the pore phase)."""
    import plotly.graph_objects as go

    z_nm = z_nm or pixel_nm
    roi_plane = (classes != 2).all(0)
    rs, cs = M.largest_inner_rectangle(roi_plane)
    width = min(cs.stop - cs.start, 480)
    c0 = (cs.start + cs.stop - width) // 2
    vol = classes[:, rs, c0:c0 + width] == 1
    full_shape, full_por = vol.shape, float(vol.mean())
    step = max(1, int(np.ceil(max(vol.shape[1:]) / max_side)))
    if step > 1:  # block-average downsampling keeps thin pores better than striding
        Y, X = vol.shape[1] // step * step, vol.shape[2] // step * step
        vol = vol[:, :Y, :X].reshape(len(vol), Y // step, step, X // step, step).mean((2, 4)) > 0.5
    smooth = ndi.gaussian_filter(vol.astype(np.float32), 0.7)
    zz, yy, xx = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
    fig = go.Figure(go.Isosurface(
        x=(xx * pixel_nm * step).ravel(), y=(zz * z_nm).ravel(), z=(-yy * pixel_nm * step).ravel(),
        value=smooth.ravel(), isomin=0.5, isomax=1.0, surface_count=1,
        caps=dict(x_show=True, y_show=True, z_show=True),
        colorscale=[[0, "#1f6feb"], [1, "#58a6ff"]], showscale=False, opacity=1.0,
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6),
    ))
    fig.update_layout(
        title=f"3D pore network: interior {full_shape[0]}x{full_shape[1]}x{full_shape[2]} voxels "
              f"(z, y, x), porosity {full_por:.3f}; drawn at 1/{step} in-plane",
        scene=dict(xaxis_title="x in-plane (nm)", yaxis_title="z milling (nm)", zaxis_title="y through-plane (nm)",
                   aspectmode="data", camera=dict(eye=dict(x=1.9, y=-2.1, z=1.3))),
        margin=dict(l=0, r=0, t=40, b=0))
    path = Path(path)
    fig.write_html(path, include_plotlyjs="cdn")
    return path
