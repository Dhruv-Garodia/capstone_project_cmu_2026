# Éponge: pFIB-SEM catalyst-layer segmentation and porosity

Capstone 2026 (CMU). We segment Xe-pFIB-SEM images of IrO₂ PEM-water-electrolysis catalyst
layers into **pore**, **solid** and **outside the layer**, then measure porosity, pore sizes,
connectivity and tortuosity with uncertainty, as a Python library and as a website that runs in
the browser.

**Website:** upload a slice or a TIFF stack and get the segmentation, the numbers and a 3D view.
See [docs/WEBSITE.md](docs/WEBSITE.md) to run or deploy it (Vercel).

## Results at a glance

Held-out test frames 100–117 of `comp_full.tif` (never used for training or model selection):

| model | params | mIoU | pore IoU | layer IoU | porosity error |
|---|---:|---:|---:|---:|---:|
| **MillNet 2D** (ours; website model) | 2.5 M | **0.934** | **0.884** | 0.977 | 0.011 |
| **MillNet 2.5D** (ours; library default for stacks) | 2.5 M | **0.936** | **0.887** | **0.978** | **0.004** |
| Éponge ResUNet | 2.4 M | 0.911 | 0.842 | 0.974 | 0.005 |
| Garodia U-Net, 2.5D | 7.9 M | 0.912 | 0.844 | 0.972 | 0.005 |
| UNet++ ResNet-34, ImageNet | 26.1 M | 0.903 | 0.828 | 0.973 | 0.010 |
| TransUNet R50-ViT-B/16, ImageNet-21k | 105.3 M | 0.902 | 0.826 | 0.975 | 0.004 |
| SegFormer-B2, ImageNet | 24.7 M | 0.894 | 0.812 | 0.975 | 0.007 |
| Same ResUNet trained on the original labels | 2.4 M | 0.706 | 0.435 | 0.951 | 0.182 |

* **MillNet, our own architecture, is the best model**: +0.02 mIoU and +0.04 pore IoU over everything
  else, robust across seeds. Its advantage is frame-level context (whole-frame histogram conditioning,
  axial attention). Design reasoning and ablations: [docs/MILLNET.md](docs/MILLNET.md).
* **Correcting the labels mattered most.** The original napari masks were a raw grey threshold that
  changed between annotation sessions; they gave porosity 0.18. Corrected labels give 0.354, inside an
  independent classical pipeline's bracket (0.32–0.51) on every test frame.
* Standard architectures tie within 0.02 mIoU; larger pretrained models (incl. TransUNet) did not help.
* comp_full catalyst layer: porosity ≈ 0.37–0.40, median pore diameter 60–72 nm, through-plane pore
  tortuosity τ ≈ 3.3 (3D), > 95 % of pore volume connected across the layer.

Full table and discussion: [docs/MODELS.md](docs/MODELS.md). What we learned: [docs/FINDINGS.md](docs/FINDINGS.md).

## Quick start

```bash
pip install -e "eponge[export,viz3d,dev]"

# put the data in place (not in git):
#   data/raw/comp_full.tif          the pFIB-SEM stack
#   data/manual_annotation/*.png    the napari masks

eponge fix-annotations --stack data/raw/comp_full.tif --masks data/manual_annotation --out data/annotations_v2
eponge train   --stack data/raw/comp_full.tif --labels data/annotations_v2 --out runs/millnet_k0 --arch millnet
eponge analyze data/raw/comp_full.tif --out results/        # or any PNG / TIFF slice or stack
```

## Documentation

| doc | read it for |
|---|---|
| [docs/FINDINGS.md](docs/FINDINGS.md) | the five things we learned about the data, labels and earlier pipelines |
| [docs/ANNOTATIONS.md](docs/ANNOTATIONS.md) | how the napari masks were audited and corrected |
| [docs/MILLNET.md](docs/MILLNET.md) | our MillNet architecture: design reasoning, results, ablations |
| [docs/MODELS.md](docs/MODELS.md) | architectures, training protocol, full comparison |
| [docs/LIBRARY.md](docs/LIBRARY.md) | Python API, CLI, metrics, the three visualisations |
| [docs/WEBSITE.md](docs/WEBSITE.md) | using, running and deploying the website |
| [docs/literature_review_and_proposal.md](docs/literature_review_and_proposal.md) | literature review and proposed method |
| [pore_pipeline/README.md](pore_pipeline/README.md) | D. Garodia's classical pipeline, trimap labels, label-free checks |

## Repository layout

| path | contents |
|---|---|
| `eponge/` | the library: preprocessing, annotation audit, model zoo, training, inference with uncertainty, metrics, 3D meshes, visualisations, CLI, tests |
| `webapp/` | the website (static; TensorFlow.js + three.js) |
| `pore_pipeline/` | classical segmentation, trimap labels and label-free evaluation (D. Garodia) |
| `runs/` | training summaries and logs for every model (`run_zoo.sh` reproduces them) |
| `data/annotations_v2/` | correction log for the corrected labels (masks are rebuilt by one command) |
| `results/` | `pore_pipeline` results on the comp, uncomp and pristine stacks |
| `docs/` | documentation; `LEGACY_README.md` describes the earlier synthetic-data code |
| `model/`, `utils/`, `scripts/`, `puma-synthetic-gen/` | earlier synthetic-data pipeline (see `docs/LEGACY_README.md`) |

Raw TIFFs, masks and model checkpoints are git-ignored. The website's 4.8 MB model and its demo
mesh are committed so the site works from a fresh clone.

## Team

Aaditya Vikram, Dhruv Garodia, Hou Kin, Yunzhi.
