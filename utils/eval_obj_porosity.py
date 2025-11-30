import trimesh
import os

base = "data/model-recon-output"

for i in range(0, 11):
    path = os.path.join(base, f"synthetic_{i}", "voxel_direct.obj")
    
    print(f"\n=== synthetic_{i} ===")
    try:
        mesh = trimesh.load(path)
        solid_volume = mesh.volume
        bbox_volume = mesh.bounding_box.volume
        pore_volume = bbox_volume - solid_volume

        porosity = pore_volume / bbox_volume
        print("porosity:", porosity)

    except Exception as e:
        print("Error loading mesh:", e)
