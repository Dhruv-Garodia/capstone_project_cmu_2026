import bpy
from math import *
import bmesh
import random
import json
import sys
from pathlib import Path

FLAT = False
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OBJ_PATH = REPO_ROOT / "puma-synthetic-gen" / "output" / "synthetic_pfibs" / "synthetic_8" / "synthetic_volume.obj"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "no_scale_demo_output"


def normalize_scene():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"


def set_active_object(obj):
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def get_single_mesh_object():
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(mesh_objects) != 1:
        raise RuntimeError(
            f"Expected exactly one mesh object in scene, found {len(mesh_objects)}: "
            f"{[obj.name for obj in mesh_objects]}"
        )
    return mesh_objects[0]


def render(n, num_slices=2, flat=FLAT, obj_path=DEFAULT_OBJ_PATH, output_dir=DEFAULT_OUTPUT_DIR):
    normalize_scene()

    porous_name = f"synthetic_{n}"
    seed = 11632 + n
    random.seed(seed)

    obj_path = Path(obj_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.obj_import(filepath=str(obj_path))

    porous = get_single_mesh_object()
    porous.scale[1] *= -1
    porous.rotation_euler[1] = radians(90)

    px, py, pz = porous.dimensions
    porous.location = (px / 2, py / 2, pz / 2)
    length = porous.dimensions[1]

    light = bpy.data.objects["Light"]
    light.data.type = "SPOT"

    lx, ly, lz = (0, 3, 0)
    light.rotation_euler = (-1.5708, 0, 0)
    light.data.color = (1.0, 1.0, 1.0)
    light.data.energy = 200 * random.random() + 300
    light.data.spot_size = radians(65)

    r = 0.5 * random.random() + 2
    theta = 10 * random.random() + 85
    h = 1.4 * r

    camera = bpy.data.objects["Camera"]

    if flat:
        camera.rotation_euler = (radians(90), 0.0, radians(180))
        cx, cy, cz = (0, 4.5, 0)
        lx, ly, lz = (0, 4.5, 0)
        light.data.spot_size = radians(50)
    else:
        camera.rotation_euler = (radians(38), 0.0, radians(90 + theta))
        cx, cy, cz = (r * cos(radians(theta)), length / 2 + r * sin(radians(theta)), h)
        light.data.energy = 900

    bpy.context.scene.render.resolution_x = 1080
    bpy.context.scene.render.resolution_y = 720

    if "LightMaterial" not in bpy.data.materials:
        bpy.data.materials.new(name="LightMaterial")

    for i in range(0, num_slices):
        dy = i / num_slices * length

        camera.location = (cx, cy - dy, cz)
        light.location = (lx, ly - dy, lz)

        ez = 0.0001
        mat = bpy.data.materials["LightMaterial"]
        mat.use_nodes = True

        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(0, (-dy) / 2, pz / 2),
            scale=(px, py - dy, ez),
        )
        top = bpy.context.active_object
        top.name = "topLayer"
        top.data.materials.append(mat)

        set_active_object(porous)
        bpy.ops.object.mode_set(mode="EDIT")
        mesh = bmesh.from_edit_mesh(porous.data)

        bmesh.ops.bisect_plane(
            mesh,
            geom=mesh.verts[:] + mesh.edges[:] + mesh.faces[:],
            plane_co=(0, 0, dy),
            plane_no=(0, 0, 1),
            clear_outer=False,
            clear_inner=True,
        )

        bmesh.update_edit_mesh(porous.data)
        bpy.ops.object.mode_set(mode="OBJECT")

        folder = output_dir / f"slice_{n}"
        if flat:
            folder = output_dir / f"slice_{n}_flat"
        folder.mkdir(parents=True, exist_ok=True)

        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.filepath = str(folder / f"{porous_name}_{i}.png")
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"].default_value = (
            0.06,
            0.06,
            0.06,
            1,
        )

        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(top, do_unlink=True)

    params = {
        "radius": r,
        "height": h,
        "theta": theta,
        "light_intensity": light.data.energy,
        "obj_path": str(obj_path),
        "scale_applied": False,
    }

    log_file = folder / f"{porous_name}.json"
    with open(log_file, "w") as f:
        json.dump(params, f, indent=4)

    bpy.data.objects.remove(porous, do_unlink=True)
    print("Image captured and saved.")


def run():
    num_slices = 2
    flat = FLAT
    obj_path = DEFAULT_OBJ_PATH
    output_dir = DEFAULT_OUTPUT_DIR
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if args:
            num_slices = int(args[0])
        if len(args) > 1:
            flat = args[1].lower() in ("1", "true", "yes", "y", "flat")
        if len(args) > 2:
            obj_path = Path(args[2])
        if len(args) > 3:
            output_dir = Path(args[3])
    render(1, num_slices, flat, obj_path=obj_path, output_dir=output_dir)


if __name__ == "__main__":
    run()
