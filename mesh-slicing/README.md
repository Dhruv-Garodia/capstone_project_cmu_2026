# Mesh Slicing

This folder contains the Blender scripts used to render slice images from 3D meshes.

The most important workflow in this folder is the one connected to the PuMA synthetic data generation pipeline:

1. generate synthetic data with `puma-synthetic-gen/scripts/generate_synthetic_pfib.py`
2. convert the exported slice stack into an STL with `utils/convert_mesh.py`
3. render mesh slices from that STL with `render_stl_mesh_slices.py`

## Blender Installation

Install Blender first before running any script in this folder.

Recommended Windows setup:

1. Download Blender from the official site and install the normal desktop build.
2. During install, note the full path to `blender.exe`.
3. Test it once from PowerShell with:

```powershell
& "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --version
```

If your Blender install lives somewhere else, replace every example path in this README with your own `blender.exe` path.

Important notes:

- the examples in this README use `"D:\Program Files\Blender Foundation\Blender 5.0\blender.exe"` only as an example
- if your Blender path contains spaces, keep the surrounding quotes
- on Windows PowerShell, use `&` before the quoted Blender executable path
- if `blender.exe --version` works but the slicing command does not, the most common cause is a wrong script path or mesh path after the `--`

## Path And Command Notes

All three slicing scripts support command-line overrides after Blender's `--` separator.

General command shape:

```powershell
& "<path-to-blender.exe>" --background --python "<path-to-script.py>" -- <num_slices> <flat> <mesh_path> [output_dir] [intermediate_obj_path]
```

Argument meaning:

- `<num_slices>`: number of rendered slices
- `<flat>`: `true` or `false`
- `<mesh_path>`: OBJ path for `render_mesh_slices.py` and `render_mesh_slices_no_scale.py`, STL path for `render_stl_mesh_slices.py`
- `[output_dir]`: optional override for where rendered PNGs are written
- `[intermediate_obj_path]`: optional extra argument only for `render_stl_mesh_slices.py`

Path behavior in the scripts:

- all paths passed on the command line are resolved to absolute paths inside the script
- if you do not pass a mesh path, each script falls back to its built-in default path
- `render_mesh_slices.py` defaults to `mesh-slicing/voxel_direct_syn1.obj`
- `render_mesh_slices_no_scale.py` defaults to `puma-synthetic-gen/output/synthetic_pfibs/synthetic_8/synthetic_volume.obj`
- `render_stl_mesh_slices.py` defaults to `real_groundtruth.stl` at the repo root
- `render_stl_mesh_slices.py` also writes an intermediate OBJ; by default that is `mesh-slicing/test.obj`

Recommended path practice:

- run commands from the repo root when possible
- prefer absolute paths for Blender, the Python script, and the input mesh
- if you copy commands from this README, replace `D:\capstone_project_cmu_2026` with your own repo location first
- if your mesh path points to a newly generated sample, make sure you update both the folder name and the file name, for example `synthetic_8` to `synthetic_12`

Common path mistakes:

- using `adjust_config.txt` when the file is actually `adjusted_config.txt`
- running from a different working directory and assuming relative paths still point to the same files
- forgetting that `render_stl_mesh_slices.py` expects an STL, while the other two scripts expect OBJ
- forgetting quotes around a path that contains spaces
- editing the hard-coded defaults in the Python file and then also passing a different path on the command line, which makes it unclear which input is being used
- `utils/convert_mesh.py` currently contains a local absolute input path in its config section, so anyone else using it must update `INPUT_FOLDER` for their own machine and dataset before running it

## Script-Specific Path Notes

- `render_mesh_slices.py` is the old reconstruction path and intentionally applies a `0.01` scale normalization to the imported OBJ
- `render_mesh_slices_no_scale.py` does not apply that normalization, so it should be used for direct slicing of `synthetic_volume.obj`
- `render_stl_mesh_slices.py` assumes the STL came from the current reconstruction pipeline and rescales it to match the expected `150 x 150 x 150` PuMA volume
- `render_stl_mesh_slices.py` exports an intermediate OBJ as part of processing; if you want to keep multiple runs, pass a different `[intermediate_obj_path]` instead of reusing `test.obj`

## Files Kept In This Folder

- `render_mesh_slices.py`
  Reconstruction OBJ slicing script.
  Use this for model reconstruction meshes such as `voxel_direct_syn1.obj`, where the historical `0.01` scale normalization is still needed.

- `render_stl_mesh_slices.py`
  Main pipeline script for the STL workflow.
  Use this when you want to connect PuMA synthetic data generation to mesh slicing through `utils/convert_mesh.py`.

- `render_mesh_slices_no_scale.py`
  Direct OBJ slicing script.
  Use this when you want to slice `synthetic_volume.obj` directly without the old `0.01` scale step.

- `voxel_direct_syn1.obj`
  Legacy demo OBJ kept as a reference mesh.

## Main Use Cases

There are three main use cases in this folder:

### 1. Reconstruction mesh slicing

Use `render_mesh_slices.py` for reconstruction meshes like:

```text
D:\capstone_project_cmu_2026\mesh-slicing\voxel_direct_syn1.obj
```

This path keeps the older `0.01` scale step because that normalization is part of the expected reconstruction-mesh workflow.

### 2. Direct synthetic OBJ slicing

Use `render_mesh_slices_no_scale.py` when you want to slice:

```text
D:\capstone_project_cmu_2026\puma-synthetic-gen\output\synthetic_pfibs\synthetic_8\synthetic_volume.obj
```

This path is the direct baseline and does not apply the old `0.01` scale.

### 3. PuMA -> convert_mesh -> STL -> mesh slicing

Use `render_stl_mesh_slices.py` for the full pipeline connected to PuMA data generation and `utils/convert_mesh.py`.

This is the most important workflow in this folder.

## Recommended Workflow

The recommended workflow is:

### Step 1: Generate synthetic data with PuMA

Activate the PuMA environment first:

```powershell
conda activate puma
```

From:

```text
D:\capstone_project_cmu_2026\puma-synthetic-gen
```

run:

```powershell
python scripts/generate_synthetic_pfib.py --spec input/config/adjusted_config.txt
```

For the full details of synthetic data generation, refer to:

```text
D:\capstone_project_cmu_2026\puma-synthetic-gen\README.md
```

With the current setup, this creates outputs such as:

```text
D:\capstone_project_cmu_2026\puma-synthetic-gen\output\synthetic_pfibs\synthetic_8\png_slices
D:\capstone_project_cmu_2026\puma-synthetic-gen\output\synthetic_pfibs\synthetic_8\synthetic_volume.obj
```

Important note:

- for the STL reconstruction pipeline, `sem_like: false` is strongly recommended in `adjusted_config.txt`
- the current workflow assumes a `150 x 150 x 150` volume

### Step 2: Convert the slice stack into STL

From the repo root:

```powershell
cd D:\capstone_project_cmu_2026
python utils\convert_mesh.py
```

This converts the generated PNG slice stack into:

```text
D:\capstone_project_cmu_2026\real_groundtruth.stl
```

`utils/convert_mesh.py` is the bridge from the image stack back to a mesh.

### Step 3: Render slices from the STL

Run:

```powershell
cd D:\capstone_project_cmu_2026
& "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python "D:\capstone_project_cmu_2026\mesh-slicing\render_stl_mesh_slices.py" -- 5 false "D:\capstone_project_cmu_2026\real_groundtruth.stl"
```

This script:

- imports `real_groundtruth.stl`
- matches its orientation and scale to the expected slicing setup
- exports an intermediate OBJ during processing
- renders slice images

Default output:

```text
D:\capstone_project_cmu_2026\mesh-slicing\pipeline_demo_output\slice_1\
```

## Reconstruction OBJ Workflow

If you want to slice a reconstruction mesh such as `voxel_direct_syn1.obj`, use:

```powershell
cd D:\capstone_project_cmu_2026
& "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python "D:\capstone_project_cmu_2026\mesh-slicing\render_mesh_slices.py" -- 5 false "D:\capstone_project_cmu_2026\mesh-slicing\voxel_direct_syn1.obj"
```

Default output:

```text
D:\capstone_project_cmu_2026\mesh-slicing\demo_output\slice_1\
```

## Direct Synthetic OBJ Workflow

If you want to slice the original PuMA mesh directly instead of going through STL reconstruction, use:

```powershell
cd D:\capstone_project_cmu_2026
& "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python "D:\capstone_project_cmu_2026\mesh-slicing\render_mesh_slices_no_scale.py" -- 5 false "D:\capstone_project_cmu_2026\puma-synthetic-gen\output\synthetic_pfibs\synthetic_8\synthetic_volume.obj"
```

This is the direct comparison path for:

```text
synthetic_8/synthetic_volume.obj
```

Default output:

```text
D:\capstone_project_cmu_2026\mesh-slicing\no_scale_demo_output\slice_1\
```

## Which Script To Use

Use `render_mesh_slices.py` when:

- you are slicing reconstruction meshes such as `voxel_direct_syn1.obj`
- the mesh was built for the older workflow that expects the `0.01` scale normalization
- you want the reconstruction-dataset slicing path

Use `render_stl_mesh_slices.py` when:

- you want the full pipeline connected to PuMA data generation
- you want to render slices from `real_groundtruth.stl`
- you are comparing reconstructed STL results against the original synthetic mesh

Use `render_mesh_slices_no_scale.py` when:

- you already have a good OBJ mesh
- you want to slice `synthetic_volume.obj` directly
- you want a baseline result to compare against the STL reconstruction path

## Demo Commands

### Demo A: Reconstruction mesh slicing

```powershell
cd D:\capstone_project_cmu_2026
& "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python "D:\capstone_project_cmu_2026\mesh-slicing\render_mesh_slices.py" -- 5 false "D:\capstone_project_cmu_2026\mesh-slicing\voxel_direct_syn1.obj"
```

### Demo B: Full PuMA -> STL -> mesh slicing pipeline

```powershell
cd D:\capstone_project_cmu_2026\puma-synthetic-gen
conda activate puma
python scripts/generate_synthetic_pfib.py --spec input/config/adjusted_config.txt

cd D:\capstone_project_cmu_2026
conda activate puma
python utils\convert_mesh.py

& "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python "D:\capstone_project_cmu_2026\mesh-slicing\render_stl_mesh_slices.py" -- 5 false "D:\capstone_project_cmu_2026\real_groundtruth.stl"
```

### Demo C: Direct synthetic_volume.obj slicing

```powershell
cd D:\capstone_project_cmu_2026
& "D:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python "D:\capstone_project_cmu_2026\mesh-slicing\render_mesh_slices_no_scale.py" -- 5 false "D:\capstone_project_cmu_2026\puma-synthetic-gen\output\synthetic_pfibs\synthetic_8\synthetic_volume.obj"
```

## Why Two Scripts Exist

The scripts are all useful, but they answer different questions:

- `render_mesh_slices.py` shows how reconstruction meshes should be sliced in the older normalized workflow
- `render_mesh_slices_no_scale.py` shows what the original synthetic mesh looks like when sliced directly
- `render_stl_mesh_slices.py` shows what the reconstructed mesh looks like after going through the image stack and `convert_mesh.py`

That comparison is useful when you want to measure how much geometry changes after stack-to-mesh reconstruction.

## Notes

- all scripts currently use `render(1, ...)`, so output filenames are still based on `synthetic_1`
- all scripts rely on Blender default objects such as `Camera` and `Light`
- the synthetic-data workflows assume the current mesh volume is `150 x 150 x 150`
- STL reconstruction and direct PuMA mesh export may still differ slightly because they are generated by different meshing paths
