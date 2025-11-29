import trimesh

mesh = trimesh.load("data/model-recon-output/synthetic_5/voxel_direct.obj")
solid_volume = mesh.volume
bbox_volume = mesh.bounding_box.volume
pore_volume = bbox_volume - solid_volume

porosity = pore_volume / bbox_volume
print(porosity)

# evalated on voxel_direct.obj
# (pred, gt)
#0 0.3516312222222222 0.425199
#1 0.39724590123456793 0.471001
#2 0.43498564197530865 0.506801
#3 0.4768557901234568 0.54663
#4 0.5254917530864198 0.593496
#5 0.5664025679012346 0.630187