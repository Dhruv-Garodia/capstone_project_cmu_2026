# Éponge

Segmentation, porosity and microstructure analysis for pFIB-SEM images and stacks of porous
catalyst layers (IrO₂ PEM water-electrolysis anodes). One call takes a slice or a stack to a
3-class segmentation (solid / pore / outside the catalyst layer), physical metrics with
uncertainty, and three figures.

```bash
pip install -e ".[export,viz3d]"          # from this folder
eponge analyze slice.png --out results/   # single image
eponge analyze stack.tif --out results/   # multi-page TIFF: registered, 3D metrics + 3D view
```

```python
import eponge
res = eponge.analyze_file("stack.tif", pixel_nm=6.0)
res.metrics["porosity"], res.metrics["tortuosity"], res.uncertainty
res.save("results/")      # masks, metrics.json, viz1/viz2/viz3
```

## What is in the package

| module | contents |
|---|---|
| `eponge.io` | TIFF/PNG/folder loading (8/16/32-bit), mask encodings (`0` pore, `255` solid, `128` outside) |
| `eponge.preprocess` | chained sub-pixel phase-correlation registration, FFT de-curtaining, percentile normalisation |
| `eponge.annotation` | audit of napari masks (implied-threshold detector), v1 → v2 correction |
| `eponge.models` | residual U-Net, 2D (k = 0) or 2.5D (2k+1 adjacent slices), MC-dropout ready, 2.4 M params |
| `eponge.data`, `eponge.train` | random-crop dataset with scale / gamma / noise / curtaining augmentation; training CLI |
| `eponge.inference` | tiled inference, flip TTA, Monte-Carlo-dropout porosity bands, threshold sensitivity |
| `eponge.metrics` | porosity, local-thickness PSD, chord lengths, specific surface, connectivity / percolation, diffusive tortuosity factor (TauFactor definition, finite volume), depth profile through the layer, REV curve |
| `eponge.baselines` | fixed threshold, Otsu, local threshold (for comparison tables) |
| `eponge.viz` | the three visualisations (below) |
| `eponge.export` | ONNX export and browser weights (BatchNorm folded, float16) |

### The three visualisations

1. **Segmentation map** (`viz1_segmentation_map.png`): input, class overlay with the detected
   catalyst-layer boundary, and a local pore-diameter map in nm.
2. **Quantitative panel** (`viz2_quantitative.png`): volume-weighted pore size distribution,
   porosity versus normalised depth through the layer, and the REV convergence curve.
3. **3D pore network** (`viz3_3d_pore_network.html`, stacks): interactive isosurface of the
   registered pore phase. For a single 2D image a connectivity map is produced instead, with
   clusters that span the layer highlighted (`viz3_connectivity.png`).

## Commands

```bash
eponge audit-annotations --stack comp_full.tif --masks data/manual_annotation      # implied threshold per frame
eponge fix-annotations   --stack comp_full.tif --masks data/manual_annotation --out annotations_v2
eponge train --stack comp_full.tif --labels annotations_v2 --out runs/unet2d --k 0
eponge train --stack comp_full.tif --labels annotations_v2 --out runs/unet25d --k 2 --register
eponge export-onnx runs/unet2d/best.pt model.onnx
eponge export-web  runs/unet2d/best.pt ../webapp/model
python scripts/napari_review.py --stack comp_full.tif --v1 data/manual_annotation --v2 annotations_v2
python scripts/compare_models.py          # comparison table on held-out frames 100-117
```

## Conventions

* Network classes: `0` solid, `1` pore, `2` outside the catalyst layer. PNG masks: `255`, `0`, `128`.
* Axes for comp_full: `y` (rows) runs through the layer, `x` along it, `z` is the milling direction.
* Default pixel size is 6 nm. Pass `pixel_nm` for other data; the model tolerates 0.6–1.6× scale changes.
* Porosity is reported inside the catalyst layer only.

Run the tests with `pytest -q tests`.
