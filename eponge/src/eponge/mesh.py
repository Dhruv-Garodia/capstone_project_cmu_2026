"""Surface meshes of the 3D pore network for high-quality rendering.

``pore_mesh`` turns a segmented volume into a smooth, closed triangle mesh of the pore phase
(marching cubes on a lightly smoothed occupancy field, Taubin smoothing, quadric decimation)
with two per-vertex attributes:

* ``diameter_nm``  local pore diameter (local thickness) at the vertex;
* ``percolating``  1 if the vertex belongs to a pore cluster that connects the top and the
                   bottom of the catalyst layer (through-plane, y), else 0.

``save_glb`` writes a standard glTF binary (opens in Blender, ParaView, MeshLab, any glTF
viewer) coloured by pore diameter. ``save_web`` writes the compact format read by the web
app's three.js viewer: quantised positions, indices and both attributes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from . import metrics as M


def _viridis(t: np.ndarray) -> np.ndarray:
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], float) / 255
    t = np.clip(t, 0, 1) * 4
    i = np.minimum(t.astype(int), 3)
    f = (t - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def pore_mesh(pore: np.ndarray, spacing_nm=(6.0, 6.0, 6.0), smooth_sigma: float = 0.7,
              target_faces: int = 400_000) -> dict:
    """Mesh of the pore phase of a (Z, Y, X) boolean volume. Coordinates in nm, origin at the corner.

    The volume is padded with solid so the block is closed at its faces (a cut-away block).
    """
    import fast_simplification
    import trimesh

    pore = pore.astype(bool)
    lt = M.local_thickness(pore) * float(np.mean(spacing_nm[1:]))
    lab, n = ndi.label(pore)
    top, bot = np.unique(lab[:, 0, :]), np.unique(lab[:, -1, :])
    perc_labels = np.intersect1d(top, bot)
    perc_labels = perc_labels[perc_labels > 0]
    perc = np.isin(lab, perc_labels)

    field = ndi.gaussian_filter(np.pad(pore, 1).astype(np.float32), smooth_sigma)
    verts, faces, _, _ = marching_cubes(field, 0.5, spacing=spacing_nm)
    verts -= np.array(spacing_nm, dtype=np.float32)  # undo the padding offset
    mesh = trimesh.Trimesh(verts, faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=8)
    if len(mesh.faces) > target_faces:
        v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64),
                                            target_reduction=1 - target_faces / len(mesh.faces))
        mesh = trimesh.Trimesh(v, f, process=True)

    # sample attributes at the nearest pore voxel of each vertex
    idx = np.round(mesh.vertices / np.array(spacing_nm)).astype(int)
    for a in range(3):
        idx[:, a] = np.clip(idx[:, a], 0, pore.shape[a] - 1)
    dist, (iz, iy, ix) = ndi.distance_transform_edt(~pore, return_indices=True)
    nz, ny, nx = iz[tuple(idx.T)], iy[tuple(idx.T)], ix[tuple(idx.T)]
    diam = lt[nz, ny, nx]
    perc_v = perc[nz, ny, nx]
    return {"mesh": mesh, "diameter_nm": diam.astype(np.float32), "percolating": perc_v.astype(np.uint8),
            "shape": pore.shape, "spacing_nm": list(map(float, spacing_nm)),
            "porosity": float(pore.mean()), "percolating_fraction": float(perc.sum() / max(pore.sum(), 1)),
            "n_clusters": int(n)}


def save_glb(res: dict, path: str | Path, dmax: float | None = None) -> Path:
    mesh = res["mesh"].copy()
    d = res["diameter_nm"]
    dmax = dmax or float(np.percentile(d, 98))
    rgb = _viridis(d / dmax)
    mesh.visual.vertex_colors = np.c_[(rgb * 255).astype(np.uint8), np.full(len(d), 255, np.uint8)]
    path = Path(path)
    mesh.export(path)
    return path


def save_web(res: dict, path_prefix: str | Path, extra_meta: dict | None = None) -> dict:
    """``<prefix>.bin`` = [uint16 positions (quantised to the bounding box) | uint32 indices |
    uint8 diameter (0-255 of dmax) | uint8 percolating]; ``<prefix>.json`` = layout + metadata."""
    mesh = res["mesh"]
    v = mesh.vertices.astype(np.float64)
    lo, hi = v.min(0), v.max(0)
    q = np.round((v - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype(np.uint16)
    f = mesh.faces.astype(np.uint32)
    dmax = float(np.percentile(res["diameter_nm"], 98))
    d8 = np.clip(np.round(res["diameter_nm"] / dmax * 255), 0, 255).astype(np.uint8)
    p8 = res["percolating"].astype(np.uint8)
    blobs = [q.tobytes(), f.tobytes(), d8.tobytes(), p8.tobytes()]
    prefix = Path(path_prefix)
    prefix.with_suffix(".bin").write_bytes(b"".join(blobs))
    meta = {"n_vertices": int(len(v)), "n_faces": int(len(f)), "bbox_min_nm": lo.tolist(), "bbox_max_nm": hi.tolist(),
            "axes": ["z_milling", "y_through_plane", "x_in_plane"], "diameter_max_nm": dmax,
            "layout": ["pos_u16x3", "index_u32x3", "diameter_u8", "percolating_u8"],
            "volume_shape": list(map(int, res["shape"])), "spacing_nm": res["spacing_nm"],
            "porosity": res["porosity"], "percolating_fraction": res["percolating_fraction"],
            "n_clusters": res["n_clusters"], **(extra_meta or {})}
    prefix.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta
