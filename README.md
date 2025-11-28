# PFIB-SEM 3D Segmentation & Synthetic Microstructure Pipeline

This repository contains a full pipeline for **synthetic microstructure generation**, **2D/3D PFIB-SEM image preprocessing**, **UNet-based segmentation**, and **evaluation/visualization utilities**. 
---

## 🌳 Project Structure

.
├── checkpoints/             # Stores trained model weights & checkpoints
│
├── data/                    # All datasets used in the project
│   ├── __MACOSX/            # (Ignore) macOS metadata directory
│   ├── model-recon-output/  # Model-based reconstructed volume outputs
│   ├── model-seg-output/    # Model-predicted segmentation masks
│   ├── paper-seg-output/    # Reproduced segmentation from original paper
│   ├── pFIB-real-data/      # Real experimental PFIB-SEM images
│   └── synthetic-train-test/# Synthetic training/testing dataset
│
├── model/                   # Core UNet model + training pipeline
│   ├── dataset.py
│   ├── transforms.py
│   └── unet.py
│
├── scripts/                 # High-level bash scripts
│   ├── crop.sh              # Crop real stacks before preprocessing
│   ├── start_training.sh    # Launch model training end-to-end
│   └── validate_model.sh    # Run inference + evaluation on validation data
│
├── puma-synthetic-gen/      # Synthetic dataset generator (has its own README)
│
├── requirements.txt         # Python dependency list (pip)
│
└── utils/                   # Core utilities for preprocessing & evaluation
    ├── convert_mesh.py
    ├── crop_real_stack.py
    ├── cropping.py
    ├── eval_porosity.py
    ├── eval_pore_distribution.py
    ├── process_filename.py
    ├── reproduce.py
    ├── resize_pngs.py
    ├── test_model.py
    ├── tif_to_png.py
    └── visualize.py

---

## 🔧 Utilities (utils/ folder)

The `utils/` directory contains helper scripts used throughout the workflow:

| Script | Description |
|--------|-------------|
| `train.py` | Main training script for UNet / other models |
| `test_model.py` | Run inference on a folder of images |
| `eval_porosity.py` | Compute porosity statistics per-slice & volume |
| `eval_pore_distribution.py` | Analyze pore-size distribution |
| `visualize.py` | Interactive viewer for 3D stacks with a slice slider |
| `convert_mesh.py` | Convert mesh → voxel → PNG pipeline for synthetic generation |
| `reproduce.py` | Reproduce paper segmentation results for quality evaluation |
| `process_filename.py` | Normalizes naming patterns for image stacks |
| `crop_real_stack.py` | Crop real PFIB stacks into smaller training tiles |
| `cropping.py` | Shared helper functions for cropping tasks |
| `resize_pngs.py` | Resize image folders while maintaining structure |
| `tif_to_png.py` | Convert .tif stacks into numbered .png slices |

---

## ▶️ Quick Start

### 1. Install dependencies (recommended: fresh conda env)

```
conda create -n pfib_sem python=3.10
conda activate pfib_sem
pip install -r requirements.txt

```
---

## 📊 Data Preparation

### 1. Cropping real PFIB stacks

Use the helper script:
```
bash scripts/crop.sh
```

### 2. Synthetic data  
Located in `puma-synthetic-gen/` → includes mesh conversion, lightening, PNG export, and mask generation.  
*(Has its own README; not documented here.)*

---

## ⚙️ Model Training
```
bash scripts/start_training.sh
```

or directly:

```
python utils/train.py 
--data_dir data/synthetic_lightened 
--mask_dir data/synthetic_mask 
--save_dir checkpoints/
```
You may provide custom parameters if needed

---

## 📈 Validation

```
bash scripts/validate_model.sh
```

Or manually:
```
python utils/test_model.py 
--checkpoint checkpoints/best_model.pth 
--input data/real/
```

---

## 📊 Evaluation

### Porosity
```
python utils/eval_porosity.py --input data/model_segmented/
```

### Pore size distribution
```
python utils/eval_pore_distribution.py --input data/model_segmented/
```

---

## 🧩 Model Checkpoints

All trained models are saved inside:

```

checkpoints/
best_model.pth
last_epoch.pth
...

```