# PuMA Synthetic Data Generation

Generates synthetic pFIB-SEM datasets using [PuMA](https://puma-nasa.readthedocs.io/) (NASA).
Outputs 3D porous microstructures with known ground-truth porosity for U-Net training.

**Pipeline position:**
```
puma-synthetic-gen  →  Task C (Blender slicing)  →  Task A (U-Net training)
  (this module)           2D SEM-like images            segmentation model
```

---

## Setup

```bash
conda create -n puma conda-forge::puma
conda activate puma
pip install -r ../requirements.txt
```

---

## Directory Structure

```
puma-synthetic-gen/
├── scripts/
│   ├── generate_synthetic_pfib.py   # core generation engine (do not modify)
│   ├── mask_highlighted_segments.py # mask utility
│   ├── visualize.py                 # interactive single-volume slice viewer
│   └── psd_gen/                     # PSD-parameterized generation tools
│       ├── generate.py              # unified CLI: lognormal + bimodal
│       ├── convert_mesh.py          # volume_stack.tif → STL
│       └── visualize_results.py     # multi-dataset summary figure
├── utils/
│   ├── eval.py                      # porosity analysis utilities
│   ├── convert.py                   # binary volume visualisation helper
│   ├── cropping.py                  # volume cropping
│   └── plot.py                      # plotting helpers
├── input/
│   └── config/                      # example spec files for generate_synthetic_pfib.py
│       ├── example_config.txt
│       ├── 9bin_lognomal.txt
│       ├── adjusted_config.txt
│       └── log_normal_auto.txt
└── output/
    ├── synthetic_pfibs/             # original fixed-PSD datasets (synthetic_0..10)
    └── psd_gen/                     # PSD-sweep datasets (lognormal + bimodal)
        ├── sigma_025_150/
        │   └── realized_stats.json
        └── ...
```

---

## Original generation script

`generate_synthetic_pfib.py` is the low-level engine. Pass it a key-value spec file:

```bash
python scripts/generate_synthetic_pfib.py --spec input/config/example_config.txt
```

Key spec parameters:

| Key | Description |
|-----|-------------|
| `nx`, `ny`, `nz` | Volume dimensions in voxels |
| `voxel_size_nm` | Physical voxel size in nm |
| `overall_porosity` | Target void fraction (0–1) |
| `psd_bins_nm` | Pore diameter bin centres in nm |
| `psd_weights` | Normalized bin weights (must sum to ~1) |
| `generator` | `sphere` or `fibers` |
| `seed` | RNG seed |
| `out_dir` | Output directory |
| `sem_like` | Apply Gaussian blur + Poisson noise to PNG slices |

See `input/config/example_config.txt` for a full annotated example.

---

## Interactive volume viewer

`scripts/visualize.py` — inspect any TIFF or NPY volume slice-by-slice with a slider.

```bash
# Single volume
python scripts/visualize.py output/psd_gen/sigma_050_150/volume_stack.tif

# Side-by-side comparison
python scripts/visualize.py volume_a.tif volume_b.tif --title1 "sigma=0.25" --title2 "sigma=1.25"
```

---

## psd_gen — PSD-parameterized generation

All three tools live in `scripts/psd_gen/` and are run from the `puma-synthetic-gen/` root.
All paths default to `output/psd_gen/` relative to the repo root.

### generate.py

Generates synthetic volumes parameterized by pore size distribution.
Internally writes a temporary config and calls `generate_synthetic_pfib.py`.

#### Subcommand: `lognormal`

Single unimodal log-normal PSD, or a batch sweep over multiple sigma values.

```bash
# Single dataset
python scripts/psd_gen/generate.py lognormal --sigma 0.5

# Batch sweep (one dataset per sigma)
python scripts/psd_gen/generate.py lognormal --sigma 0.25 0.50 0.75 1.00 1.25

# Custom resolution, porosity, and output directory
python scripts/psd_gen/generate.py lognormal \
    --sigma 0.5 \
    --mu 150 \
    --nx 150 --ny 150 --nz 150 \
    --porosity 0.492 \
    --voxel-nm 10 \
    --seed 42 \
    --out-dir output/psd_gen
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--sigma` | required | Log-space std (shape parameter). Multiple values run one dataset each |
| `--mu` | 150 | Median pore diameter in nm |
| `--n-bins` | 9 | Number of PSD histogram bins |
| `--name` | auto | Dataset name (single-sigma only; ignored for batch) |

#### Subcommand: `bimodal`

Two-peak log-normal sum PSD representing a material with two pore populations.

```bash
# Equal mix of 150 nm and 500 nm populations
python scripts/psd_gen/generate.py bimodal --mu1 150 --mu2 500

# Large-pore dominated (30/70 weight)
python scripts/psd_gen/generate.py bimodal \
    --mu1 150 --mu2 700 \
    --w1 0.3 --w2 0.7 \
    --sigma1 0.3 --sigma2 0.3 \
    --nx 150 --ny 150 --nz 150
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mu1`, `--mu2` | required | Peak median pore diameters in nm |
| `--sigma1`, `--sigma2` | 0.3 | Log-space std for each peak |
| `--w1`, `--w2` | 0.5 | Relative weights (auto-normalized) |
| `--n-bins` | 9 | Number of PSD histogram bins |

#### Shared volume arguments (both subcommands)

| Argument | Default | Description |
|----------|---------|-------------|
| `--nx`, `--ny`, `--nz` | 150 | Volume dimensions in voxels |
| `--voxel-nm` | 10.0 | Physical voxel edge length in nm |
| `--porosity` | 0.492 | Target void fraction |
| `--max-iters` | 10 | Generation attempts to find best porosity match |
| `--seed` | 42 | RNG seed |
| `--sem-blur` | 0.8 | Gaussian blur sigma for SEM-like PNG output |
| `--sem-poisson` | 10.0 | Poisson noise scale for SEM-like PNG output |
| `--no-png` | off | Skip per-slice PNG export |
| `--no-tiff` | off | Skip 3-D TIFF export |
| `--out-dir` | output/psd_gen/ | Root output directory |

---

### convert_mesh.py

Converts `volume_stack.tif` → `synthetic_volume.stl` via marching cubes.
**Always reads `volume_stack.tif` directly** — not `png_slices/`, which have SEM
blur applied and are geometrically inaccurate.

```bash
# Single dataset
python scripts/psd_gen/convert_mesh.py --dataset sigma_050_150

# Multiple datasets
python scripts/psd_gen/convert_mesh.py --dataset sigma_025_150 sigma_050_150 sigma_075_150

# All datasets in output/psd_gen/
python scripts/psd_gen/convert_mesh.py --all

# Custom data directory
python scripts/psd_gen/convert_mesh.py --all --data-dir /path/to/output
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | — | One or more dataset names (mutually exclusive with `--all`) |
| `--all` | off | Convert every subdir in `--data-dir` |
| `--data-dir` | output/psd_gen/ | Root directory of datasets |
| `--voxel-nm` | 10.0 | Voxel edge in nm (sets mesh spatial scale) |
| `--step-size` | 1 | Marching cubes step; >1 reduces mesh density |

Output: `<dataset>/synthetic_volume.stl`

---

### visualize_results.py

Generates a static summary figure (`psd_summary.png`) across multiple datasets:
- Achieved vs target porosity (bar chart)
- Slice-to-slice porosity std dev (line chart)
- Realized statistics table
- Sample mid-slice per dataset (requires `png_slices/`)

```bash
# All datasets in output/psd_gen/
python scripts/psd_gen/visualize_results.py --all

# Specific datasets
python scripts/psd_gen/visualize_results.py \
    --datasets sigma_025_150 sigma_050_150 sigma_075_150 sigma_100_150 sigma_125_150

# Custom paths
python scripts/psd_gen/visualize_results.py \
    --all \
    --data-dir output/psd_gen \
    --out-dir output/figures
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | — | Dataset names to include (mutually exclusive with `--all`) |
| `--all` | off | Include every subdir in `--data-dir` |
| `--data-dir` | output/psd_gen/ | Root directory of datasets |
| `--porosity-target` | 0.492 | Reference line on porosity bar chart |
| `--out-dir` | same as `--data-dir` | Where to save `psd_summary.png` |

---

## Output structure

Each generated dataset contains:

```
output/psd_gen/<name>/
├── png_slices/
│   ├── slice_0000.png   # SEM-like (Gaussian blur + Poisson noise). Input to U-Net.
│   └── ...
├── volume_stack.tif     # Clean binary volume (void=0, solid=255). Ground truth.
├── synthetic_volume.stl # 3-D mesh (generated by convert_mesh.py)
└── realized_stats.json  # Achieved porosity statistics
```

**Convention:** voxel value `0` = void/pore (black), `255` = solid (white).

> **Important:** `volume_stack.tif` is ground truth. `png_slices/` are for model
> input only — SEM filtering inflates apparent solid fraction by ~10 percentage
> points and should never be used as mesh input.

---

## Existing datasets (`output/psd_gen/`)

| Dataset | Type | sigma | mu1 / mu2 | Porosity |
|---------|------|-------|-----------|----------|
| sigma_025_150 | lognormal | 0.25 | 150 nm | ~49.1% |
| sigma_050_150 | lognormal | 0.50 | 150 nm | ~49.3% |
| sigma_075_150 | lognormal | 0.75 | 150 nm | ~49.3% |
| sigma_100_150 | lognormal | 1.00 | 150 nm | ~49.0% |
| sigma_125_150 | lognormal | 1.25 | 150 nm | ~49.1% |
| bimodal_1_150 | bimodal | — | 150 / 500 nm (50/50) | ~49.2% |
| bimodal_2_150 | bimodal | — | 150 / 700 nm (30/70) | ~49.3% |

All at 150×150×150 voxels, 10 nm/vox, seed=42.
