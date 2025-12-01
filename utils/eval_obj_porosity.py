import trimesh
import os

import trimesh
import os
import numpy as np
from scipy.ndimage import distance_transform_edt

base = "data/model-recon-output"
VOXEL_PITCH = 0.5
BIN_EDGES = np.geomspace(0.05, 5.0, 12)  # diameters in microns (example)

for i in range(0, 11):
    path = os.path.join(base, f"synthetic_{i}", "mc_smooth.obj")
    print(f"\n=== synthetic_{i} ===")

    try:
        # Porosity
        mesh = trimesh.load(path)
        vox = mesh.voxelized(pitch=VOXEL_PITCH)
        mask = vox.matrix    # Boolean 3D array

        solid_vol = mask.sum() * VOXEL_PITCH**3
        
        x,y,z=vox.shape
        print("x:",x, "y:",y, "z:", z)
        total_vol = x*y*z* VOXEL_PITCH**3
        
        porosity = 1 - solid_vol / total_vol
        print("total_vol:", total_vol)
        print("solid_vol:", solid_vol)

        print("porosity:", porosity)
                
        # # PSD
        # vox = mesh.voxelized(pitch=VOXEL_PITCH)
        # solid = vox.matrix.astype(np.uint8)  # 1 = solid, 0 = pore
        # pore = 1 - solid

        # dist_vox = distance_transform_edt(pore)

        # # Convert radius(vox) → diameter(nm)
        # # pitch (micron/voxel) → multiply by 1000 for nm
        # diam_nm = dist_vox * 2 * VOXEL_PITCH * 1000.0

        # # Keep only non-zero pores
        # diam_nm = diam_nm[diam_nm > 0]

        # # PSD histogram by diameter
        # hist, bins = np.histogram(diam_nm, bins=BIN_EDGES * 1000)  # convert μm→nm

        # # Normalize → weights
        # weights = hist / hist.sum()

        # bins_out = ", ".join(f"{b:.2f}" for b in bins[:-1])
        # w_out = ", ".join(f"{w:.3f}" for w in weights)

        # print(f"psd_bins_nm: {bins_out}")
        # print(f"psd_weights: {w_out}")

    except Exception as e:
        print("Error:", e)
