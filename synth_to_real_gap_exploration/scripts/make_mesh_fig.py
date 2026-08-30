"""3-D ground-truth meshes (marching cubes) for felt, fibre mat and mud-crack CL -> gallery/12_mesh_groundtruth_3d.png
Run from a directory containing the material .npy volumes (see make_layer_sheets.py for how to regenerate them)."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

def pool_max(a, f):
    a = a[:(a.shape[0]//f[0])*f[0], :(a.shape[1]//f[1])*f[1], :(a.shape[2]//f[2])*f[2]]
    return a.reshape(a.shape[0]//f[0], f[0], a.shape[1]//f[1], f[1], a.shape[2]//f[2], f[2]).max((1,3,5))

def panel(ax, vol, vox, title):
    for step in (2, 3, 4):
        verts, faces, _, _ = marching_cubes(vol.astype(np.float32), level=0.5, step_size=step)
        if len(faces) <= 90000: break
    tri = verts[faces]
    depth = tri[:, :, 0].mean(1)
    col = plt.cm.cividis(1 - (depth - depth.min()) / (np.ptp(depth) + 1e-6))
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    lam = (0.45 + 0.55 * np.abs(n @ np.array([0.6, 0.5, 0.62])))[:, None]
    ax.add_collection3d(Poly3DCollection(tri[:, :, ::-1], facecolors=np.clip(col[:, :3] * lam, 0, 1), linewidths=0))
    ax.set_xlim(0, vol.shape[2]); ax.set_ylim(0, vol.shape[1]); ax.set_zlim(vol.shape[0], 0)
    ax.set_box_aspect((vol.shape[2] * vox[2], vol.shape[1] * vox[1], vol.shape[0] * vox[0]))
    ax.view_init(elev=28, azim=-58); ax.set_axis_off(); ax.set_title(f"{title}\n{len(faces):,} faces", fontsize=10)

if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 6))
    panel(fig.add_subplot(131, projection='3d'), pool_max(np.load('felt_material_v12.npy')[:80, 100:356, 100:356], (1, 2, 2)), (0.125, 0.25, 0.25), 'sheet_felt')
    panel(fig.add_subplot(132, projection='3d'), pool_max(np.load('fiber_material.npy')[:70, 60:260, 60:260], (1, 2, 2)), (0.25, 0.5, 0.5), 'fiber_felt')
    panel(fig.add_subplot(133, projection='3d'), (np.load('mudcrack_material.npy')[40:120, 30:130, 30:130] > 0).astype(np.uint8), (0.006,) * 3, 'catalyst mud-crack')
    fig.suptitle('Ground-truth geometry as meshes (colour = depth from the beam side)')
    fig.tight_layout(); fig.savefig('12_mesh_groundtruth_3d.png', dpi=110)
