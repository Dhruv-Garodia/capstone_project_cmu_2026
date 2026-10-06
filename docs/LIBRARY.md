# Using the `eponge` library

`eponge` takes a pFIB-SEM image or stack to a segmentation, a set of microstructure
numbers with uncertainty, and three figures.

## Install

```bash
cd eponge
pip install -e ".[export,viz3d,dev]"   # Python 3.10+; PyTorch is installed as a dependency
```

The trained model ships in `eponge/src/eponge/weights/eponge_unet2d.pt`. Without it, train one
(see [MODELS.md](MODELS.md)) or copy a `runs/*/best.pt` there.

## Analyse an image or a stack

```bash
eponge analyze slice.png  --out results/          # one image
eponge analyze stack.tif  --out results/          # multi-page TIFF (or a folder of PNGs)
eponge analyze slice.png  --pixel-nm 4.5          # other pixel sizes
```

```python
import eponge

res = eponge.analyze_file("stack.tif", pixel_nm=6.0)
res.metrics["porosity"]                        # pore fraction inside the catalyst layer
res.metrics["pore_size_nm"]["d50"]             # median pore diameter
res.metrics["tortuosity"]["through_plane_y"]   # {"tau": ..., "deff_ratio": ..., ...}
res.uncertainty                                # MC-dropout band and threshold sensitivity
res.save("results/")
```

### What comes out

| file | content |
|---|---|
| `mask.png` or `masks/slice_XXXX.png` | `0` pore, `255` solid, `128` outside the catalyst layer |
| `metrics.json` | everything below, plus uncertainty |
| `viz1_segmentation_map.png` | input, overlay with the layer boundary, local pore-diameter map |
| `viz2_quantitative.png` | pore size distribution, porosity through the layer, REV curve |
| `viz3_3d_pore_network.html` (stacks) | interactive 3D pore network |
| `viz3_connectivity.png` (single images) | pore clusters; clusters spanning the layer highlighted |

### Metrics

| key | meaning |
|---|---|
| `porosity` | pore pixels / catalyst-layer pixels |
| `pore_size_nm` | local-thickness pore size distribution (volume weighted), d10/d50/d90 |
| `mean_chord_nm` | mean pore and solid chord lengths along each axis (anisotropy check) |
| `specific_surface_per_um` | pore–solid interface per unit area (2D) or volume (3D) |
| `connectivity` | clusters, largest-cluster share, share of pore that spans the layer |
| `tortuosity` | diffusive tortuosity factor τ = ε / (D_eff/D), finite-volume solve, per direction; solid phase too in 2D |
| `depth_profile` | porosity versus normalised depth through the layer, layer thickness |
| `rev` | porosity mean and spread in windows of growing size |

Uncertainty: `porosity_mc` (Monte-Carlo dropout, 10th–90th percentile), `porosity_vs_threshold`
(porosity when the pore probability threshold moves from 0.3 to 0.7), and for stacks the spread
of per-slice porosity.

## Other commands

```bash
eponge audit-annotations --stack comp_full.tif --masks data/manual_annotation   # what rule produced the masks?
eponge fix-annotations   --stack comp_full.tif --masks data/manual_annotation --out annotations_v2
eponge train             --stack comp_full.tif --labels annotations_v2 --arch transunet --out runs/x
eponge export-onnx       runs/x/best.pt model.onnx
eponge export-web        runs/x/best.pt ../webapp/model       # browser weights (ResUNet only)
```

Scripts in `eponge/scripts/`: `compare_models.py` (comparison table), `make_demo_mesh.py`
(3D mesh for the website), `make_annotations_v2.py`, `napari_review.py` (inspect and hand-edit
labels in napari).

## Conventions

* Class indices inside the code: `0` solid, `1` pore, `2` outside the catalyst layer.
* Axes for comp_full: rows (y) run through the layer, columns (x) along it, slices (z) are the milling direction.
* Stacks are used as acquired. The ~1 px/slice offset seen when correlating neighbouring
  slices is sub-surface material sliding behind the cut face, not drift; see [FINDINGS.md](FINDINGS.md).

Tests: `pytest -q eponge/tests`.
