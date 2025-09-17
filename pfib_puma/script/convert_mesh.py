# pip install numpy imageio scikit-image trimesh
import os, re, argparse
import numpy as np
import imageio.v3 as iio
from skimage import filters, morphology, measure
import trimesh

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(png_dir):
    fnames = [f for f in os.listdir(png_dir) if f.lower().endswith(".png")]
    fnames.sort(key=natural_key)
    return [os.path.join(png_dir, f) for f in fnames]

def build_binary_volume_memmap(files, mmap_path, solid_is_bright=True,
                               per_slice_otsu=True, min_obj_size=1000,
                               close_ball_radius=1, normalize_p1p99=True):
    # read one slice to infer shape
    first = iio.imread(files[0])
    if first.ndim == 3:
        first = first[..., 0]
    z, y, x = len(files), first.shape[0], first.shape[1]

    # boolean memmap on disk (1 byte/voxel; still tiny vs float32)
    vol = np.memmap(mmap_path, dtype=np.bool_, mode='w+', shape=(z, y, x))

    # If not using per-slice Otsu, compute a global threshold on a sample
    global_thr = None
    if not per_slice_otsu:
        sample_idx = np.linspace(0, z-1, num=min(z, 50), dtype=int)
        sample_vals = []
        for idx in sample_idx:
            sl = iio.imread(files[idx])
            if sl.ndim == 3: sl = sl[..., 0]
            sl = sl.astype(np.float32)
            if normalize_p1p99:
                p1, p99 = np.percentile(sl, (1, 99))
                sl = np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1)
            sample_vals.append(sl)
        sample_stack = np.stack(sample_vals, axis=0)
        global_thr = filters.threshold_otsu(sample_stack)
        del sample_vals, sample_stack

    for zi, f in enumerate(files):
        sl = iio.imread(f)
        if sl.ndim == 3:
            sl = sl[..., 0]
        sl = sl.astype(np.float32)

        # light per-slice robust normalization to stabilize Otsu
        if normalize_p1p99:
            p1, p99 = np.percentile(sl, (1, 99))
            sl = np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1)

        # threshold
        if per_slice_otsu:
            t = filters.threshold_otsu(sl)
        else:
            t = global_thr

        mask = (sl > t) if solid_is_bright else (sl < t)

        # write mask to disk immediately (no 3-D float array kept in RAM)
        vol[zi] = mask

        if zi % 50 == 0:
            print(f"[seg] slice {zi+1}/{z}")

    vol.flush()

    # reopen in r+ to run 3D morphology—operates chunk-wise via memmap
    vol = np.memmap(mmap_path, dtype=np.bool_, mode='r+', shape=(z, y, x))

    # small 3D closing to connect slice boundaries
    if close_ball_radius and close_ball_radius > 0:
        vol[:] = morphology.binary_closing(vol, morphology.ball(close_ball_radius))

    # remove tiny blobs in 3D (comment out for speed on first pass)
    if min_obj_size and min_obj_size > 0:
        vol[:] = morphology.remove_small_objects(vol, min_obj_size)

    return vol  # memmap

def largest_component3d(vol_bool):
    # returns a boolean memmap/array with only the largest component
    labeled = morphology.label(vol_bool, connectivity=3)
    if labeled.max() == 0:
        return vol_bool
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (labeled == keep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png_dir", required=True, help="Directory containing PNG slices")
    ap.add_argument("--out_mesh", default="pfib_mesh.stl")
    ap.add_argument("--mmap", default="binary_volume.dat")
    ap.add_argument("--solid_is_bright", action="store_true",
                    help="Set if the solid/material is brighter than pores")
    ap.add_argument("--global_otsu", action="store_true",
                    help="Use global Otsu from a sample instead of per-slice")
    ap.add_argument("--min_obj", type=int, default=1000)
    ap.add_argument("--close_r", type=int, default=1)
    ap.add_argument("--voxel_xy", type=float, default=0.01) # µm
    ap.add_argument("--voxel_z",  type=float, default=0.02) # µm
    ap.add_argument("--step_size", type=int, default=1,
                    help="Subsample factor for marching_cubes (>=1). 2 or 3 reduces memory/time.")
    args = ap.parse_args()

    files = list_images(args.png_dir)
    assert files, f"No PNGs found under {args.png_dir}"
    print(f"Found {len(files)} slices. Example: {os.path.basename(files[0])}")

    # Build disk-backed binary volume
    vol = build_binary_volume_memmap(
        files, args.mmap,
        solid_is_bright=args.solid_is_bright,
        per_slice_otsu=(not args.global_otsu),
        min_obj_size=args.min_obj,
        close_ball_radius=args.close_r,
        normalize_p1p99=True
    )

    # (Optional) keep only the largest connected component to ensure manifold
    print("Keeping largest 3D component…")
    solid = largest_component3d(vol)

    # Marching cubes directly on the boolean array (memmap OK)
    print("Meshing…")
    verts, faces, normals, _ = measure.marching_cubes(
        solid.astype(np.uint8),
        level=0.5,
        spacing=(args.voxel_z, args.voxel_xy, args.voxel_xy),
        allow_degenerate=False,
        step_size=max(1, args.step_size)  # >1 for faster/lower-res preview
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    # Optional quick decimation for heavy meshes
    if len(mesh.faces) > 2_000_000:
        target = int(len(mesh.faces) * 0.5)
        print(f"Decimating {len(mesh.faces):,} → {target:,} faces…")
        try:
            mesh = mesh.simplify_quadratic_decimation(target)
        except Exception:
            pass

    mesh.export(args.out_mesh)
    print(f"Saved mesh to {args.out_mesh} with {len(mesh.vertices)} verts / {len(mesh.faces)} faces.")

if __name__ == "__main__":
    main()
