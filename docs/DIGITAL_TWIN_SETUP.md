# Digital twin: setup, stored data and how to use it

The digital twin is a simulator of our pFIB-SEM measurement. It takes a 3D pore/solid structure whose phase is
known exactly, mills it slice by slice, and renders each cut face the way the microscope would see it,
shine-through included. Because the structure is known, every simulated slice comes with an **exact mask**,
which the real data can never have. We use it to (1) score segmentation methods against exact truth and
(2) pretrain networks before fine-tuning on the hand-annotated real data.

There is one calibrated twin per acquisition (comp, uncomp, pristine), because the three were imaged
differently (slide rate, contrast, shine-through depth).

## Current version (v2): read this first

Sections 1–6 below describe the first, physics-only twin (now `twin_dataset_v1_physics/`). Its images did not look like
the real stacks, and its SliceGAN structures were about half the real pore size. Version 2 replaces both parts:

| Part | v1 (physics only) | **v2 (current, `twin_dataset/`)** |
|---|---|---|
| Structure | one SliceGAN trained on a model segmentation of an earlier comp stack | **one SliceGAN per acquisition**, trained on that acquisition's hand masks (train slices 0–83), learned at half in-plane resolution and scaled back up (`runs/slicegan_<stack>/`, `slicegan_twin.py --stack <stack> --sf 2`) |
| Where shine-through happens | ray tracing along the measured slide | unchanged (verified: pores with solid right behind them are 33–42 grey levels brighter in the real images) |
| Appearance | hand-modelled grey levels, blur and noise | **conditional GAN** (pix2pixHD-style, `twin_refiner.py`, `runs/twin_refiner_v2/`) learned from paired hand masks and real images |
| Files per volume | `images.tif`, `masks.tif` | `images.tif` (physics), **`images_gan.tif`** (GAN), `masks.tif` (exact), and for hand-label volumes `real_images.tif` |
| Volumes | 78 | 60: 14 SliceGAN + 6 hand-label per acquisition, 40 × 256 × 256 |

**Structure fidelity** (porosity, real → SliceGAN): comp 0.393 → 0.404, uncomp 0.55 (layer interior) → 0.549,
pristine 0.500 → 0.481. Features along the layer come out about 25 % short for comp and pristine, and uncomp's pores are
20–30 % long in z and y (`runs/slicegan_<stack>/validation.json`).

**Appearance fidelity.** MillNet-SA2's test mIoU on real vs GAN-twin images of the same structures: comp 0.915 / 0.913,
uncomp 0.949 / 0.927, pristine 0.949 / 0.895. The physics twin scored 0.779 / 0.860 / 0.829. A pristine-only GAN
fine-tune was rejected because it was worse on validation slices.

**Exact-truth result.** MillNet-SA2 trained on synthetic data (36 GAN-twin volumes) scores 0.926 / 0.956 / 0.909 pore
IoU (comp / uncomp / pristine, mean 0.930) on held-out synthetic volumes, against 0.883 for U-Net ResNet-34 trained on real
data, 0.782 for Otsu and 0.754 for the Ferner rule (`runs/twin_gan_ps/millnet_sa2_k3/`). Models trained on real data only
read whole-frame statistics, so their synthetic crops must be pasted into a real frame (`twin_context_check.py`).

---

## 1. Where everything is stored

All paths are relative to `Capstone_Project_Final/`.

| What | Path | Contents |
|---|---|---|
| **Real data + hand masks** | `final_dataset/{comp,uncomp,pristine}/` | `images.tif` (raw slices), `masks/slice_XXXX.png` (final 4-annotator masks, slices 0–117), `split.json`, `stats.json`, `agreement_report.md` |
| **Twin dataset** (exported) | `twin_dataset/` | 78 simulated volumes with exact masks, 26 per acquisition (details in section 2) |
| Twin calibration per acquisition | `repo/runs/twin_{comp,uncomp,pristine}/calibration.json` (also copied to `twin_dataset/{stack}/calibration.json`) | fitted imaging parameters and the grey-level percentiles they were fitted to |
| SliceGAN generator | `repo/runs/slicegan/iro2_Gen.pt`, `iro2_params.data` | trained generator (Kench & Cooper 2021) |
| SliceGAN volumes | `repo/runs/slicegan/volumes.npz` | 14 generated 3D structures, 64 × 160 × 160 voxels; `validation.json` compares their porosity and chord lengths with real geometry |
| Real-derived geometry source | `repo/runs/twin/comp_full_millnet_classes.npy` | MillNet segmentation of the full comp stack (354 slices), used to cut "real" geometry blocks |
| Ferner rule, best-case parameters | `repo/runs/twin_real/ferner_oracle.json` | Ferner et al. 2024 parameters tuned on the twin against exact truth |
| Twin-pretrained model | `repo/runs/twin_gan3/millnet_sa2_k3/best.pt` | MillNet-SA2 pretrained on SliceGAN twin volumes (val pore IoU 0.930 on held-out twin volumes) |
| Fine-tuned on real data | `runs/final_models/millnet_sa2_k3_twin_s0/` | the same model fine-tuned on `final_dataset` |

**Code** (in `repo/eponge/`):

| File | Role |
|---|---|
| `src/eponge/simulate.py` | the twin: `Imaging` parameters, `render_slice`, `render_stack`, `real_geometry`, `make_dataset`, `calibrate` |
| `scripts/slicegan_twin.py` | trains SliceGAN on real cut faces and generates volumes |
| `scripts/twin_per_stack.py` | calibrates the twin separately for comp, uncomp and pristine |
| `scripts/twin_pretrain.py` | pretrains a network on twin volumes |
| `scripts/export_twin_dataset.py` | writes `twin_dataset/` |
| `scripts/report_final.py` | scores every trained model against twin exact truth, next to Otsu and the Ferner rule |

---

## 2. The twin dataset (`twin_dataset/`)

```
twin_dataset/
├── README.md, index.json          index.json: one metadata entry per volume
├── comp/  uncomp/  pristine/
│   ├── calibration.json           the imaging parameters of this acquisition's twin
│   ├── {stack}_slicegan_00..13/   SliceGAN geometry
│   ├── {stack}_real_00..05/       geometry cut from the comp segmentation
│   └── {stack}_handlabel_00..05/  geometry built from our hand-annotated masks (paired with real images)
│       ├── images.tif             simulated detector images, (40, H, W) uint8
│       ├── masks.tif              exact phase of each cut plane: 0 = pore, 255 = solid
│       ├── real_images.tif        hand-label volumes only: the real images of the same crop and slices
│       └── meta.json              geometry source, true porosity, imaging parameters, crop location
```

| Geometry | Volumes per acquisition | Slice size | True porosity | Source |
|---|---:|---|---|---|
| SliceGAN | 14 | 160 × 160 | 0.36–0.43 | generator trained on comp cut faces (slices 0–179) |
| Real-derived | 6 | 126–161 × 256 | 0.32–0.45 | blocks of the comp segmentation |
| **Hand-label** | 6 | 106–239 × 256 | comp 0.37–0.43, uncomp 0.48–0.68, pristine 0.40–0.56 | final hand masks of that acquisition, slices 0–117 stacked into 3D |

Each volume is 64 slices of structure; the last 24 provide material behind the cut plane, so 40 slices are
rendered. The same geometry is rendered once per acquisition, so `comp_slicegan_03` and
`uncomp_slicegan_03` share their masks and differ only in imaging. Hand-label volumes are the exception:
each acquisition uses its own masks.

**Mask convention.** The twin has no "outside the layer" class: every pixel is pore (0) or solid (255).
`final_dataset` uses 0 / 255 / 128 (outside).

**Hand-label volumes and real images.** `meta.json → crop_in_final_dataset` gives the slice range and the
y/x window in `final_dataset/{stack}/images.tif`, and `real_images.tif` holds exactly that crop. The real
and twin images show the same structure, so the pair shows directly what the twin gets right and wrong.
These volumes include test slices 101–117. Do not train on them if you report results on the real test split.

### Loading

```python
import json, tifffile
d = "twin_dataset/uncomp/uncomp_handlabel_00"
twin  = tifffile.imread(f"{d}/images.tif")        # (40, H, W) uint8, simulated
truth = tifffile.imread(f"{d}/masks.tif") == 0    # True = pore, exact
real  = tifffile.imread(f"{d}/real_images.tif")   # same crop, real microscope
meta  = json.load(open(f"{d}/meta.json"))
```

---

## 3. How the twin renders a slice

1. **Geometry.** A 3D boolean array, solid or pore per voxel (z = milling direction, y = through the layer,
   x = along it).
2. **Viewing ray with slide.** For cut plane z and pixel (y, x), the ray reaches voxel (z + t, y + s·t, x) at
   depth t slices behind the plane. s is the measured sub-surface slide rate of that acquisition.
3. **Cut face.** Solid at the plane (t = 0) appears bright (`cut`), with brighter pore rims (`edge`), smooth
   texture and fine grain.
4. **Shine-through.** A pore at the plane shows the first solid behind it, at depth t\*, with brightness
   `floor + (back − floor) · exp(−(t* − 1) / depth_decay)`. Deeper material is darker; if nothing lies
   within `t_max` = 24 slices, the pixel is `floor`.
5. **Detector.** Vertical streak texture, Gaussian PSF (0.8 px), signal-dependent noise, curtaining stripes,
   8-bit clipping.
6. **Truth.** The phase of voxel layer z itself.

`make_dataset` also jitters `depth_decay` (×0.7–1.4), `back` (×0.92–1.05) and the slide (×0.85–1.15)
per volume, so a model cannot memorise one exact setting. Each volume's actual values are in its `meta.json`.

### Calibrated parameters

| Parameter | comp | uncomp | pristine | Meaning |
|---|---:|---:|---:|---|
| slide s (px/slice) | 1.20 | 1.15 | 0.025 | measured on the real stacks |
| cut | 212 | 212 | 180 | cut-face brightness |
| back | 185 | 185 | 160 | material right behind the plane |
| floor | 90 | 90 | 70 | deep pores |
| depth_decay (slices) | 18 | 10 | 14 | how fast shine-through fades with depth |
| fit error (grey levels) | 7.3 | 7.0 | 6.5 | mean percentile error against the real layer |

cut, back, floor and depth_decay are fitted by grid search so that the twin's grey-level percentiles
(1/5/25/50/75/95 %) match the real catalyst layer of that stack. Physically, material behind the plane is
constrained to be at least 20 grey levels darker than the cut face. The other parameters are shared.

---

## 4. How faithful is it?

The hand-label volumes let us compare twin and real images of the **same structure**, separately inside
pores and on solid (10th / 50th / 90th percentile of grey level, 6 volumes per acquisition):

| | Pore, twin | Pore, real | Solid, twin | Solid, real |
|---|---|---|---|---|
| comp | 87 / 138 / 187 | 84 / 145 / 188 | 185 / 212 / 240 | 195 / 234 / 255 |
| uncomp | 76 / 100 / 156 | 67 / 136 / 190 | 184 / 212 / 240 | 202 / 246 / 255 |
| pristine | 62 / 97 / 150 | 43 / 94 / 132 | 153 / 180 / 208 | 121 / 153 / 186 |

What this shows:

* **comp pores match closely.** The grey shine-through inside pores, the effect the twin exists to model,
  has nearly the same distribution as in the real images.
* **uncomp pores are too dark in the twin** (median 100 vs 136): the real uncomp stack has more
  shine-through than its twin. Its fitted depth_decay (10 slices) is probably too short.
* **The cut face is too dark for comp and uncomp** (median 212 vs 234–246; the real face saturates near 255)
  and **too bright for pristine** (180 vs 153).

The cause is known. The calibration matched percentiles of the whole layer, without knowing which pixels are
pore and which are solid. Now that the hand masks exist, the twin can be calibrated on pore and solid
distributions separately. That is the next improvement (section 6).

Pixel-by-pixel correlation between the twin and real images (0.72–0.85) is **not** evidence of fidelity.
The blurred mask alone correlates as well or better (0.76–0.85), because the correlation mostly measures
"pores dark, solid bright". The per-phase distributions above are the meaningful check.

---

## 5. How the twin is used in the project

1. **Exact-truth evaluation** (`report_final.py`). Held-out SliceGAN volumes 12–13 and real-derived blocks
   from comp slices 250–353 are rendered through each acquisition's twin. Every model, plus Otsu and the
   Ferner et al. 2024 rule (best-case parameters), is scored by pore IoU against the exact phase. This is the
   only place where methods are compared against a truth that does not come from annotators.
2. **Pretraining** (`twin_pretrain.py`). MillNet-SA2 is pretrained on SliceGAN volumes 0–11 rendered through
   all three twins, then fine-tuned on `final_dataset` (`train_final.py --init ...`). This tests whether
   physics-based synthetic data improves the real-data result.
3. **Website demo.** The "How it works" page animates the same imaging model per acquisition.

### Reproduce

```bash
cd repo
python eponge/scripts/slicegan_twin.py prepare && python eponge/scripts/slicegan_twin.py train --minutes 60
python eponge/scripts/slicegan_twin.py generate --n 14          # -> runs/slicegan/volumes.npz
python eponge/scripts/twin_per_stack.py                         # -> runs/twin_{stack}/calibration.json
python eponge/scripts/export_twin_dataset.py                    # -> ../twin_dataset/
python eponge/scripts/twin_pretrain.py --arch millnet --learned-slide --k 3 --out runs/twin_gan3/millnet_sa2_k3
```

---

## 6. Known limitations and next steps

* **Calibration ignores the phase** (section 4). Next step: fit cut/back/floor/depth_decay per acquisition so
  that pore and solid grey distributions on the hand-label volumes match the real images separately. This
  would fix the uncomp shine-through and the cut-face brightness.
* **SliceGAN and real-derived geometry come from comp.** SliceGAN was trained on comp cut faces, and the
  real-derived blocks come from a model segmentation of comp, so uncomp and pristine use comp-like
  structure with their own imaging. The hand-label volumes are the exception: each uses its own
  acquisition's masks. Training SliceGAN on the uncomp and pristine hand masks would give each acquisition
  its own geometry.
* **Isotropic voxels assumed.** The ray steps one slice in z per voxel. If the slice thickness differs from
  the pixel size, depth_decay and t_max are in slice units, not physical length.
* **No layer boundary.** Twin volumes lie entirely inside the catalyst layer, so they test pore/solid
  decisions only, not finding the layer edge.
* **Simple shine-through model.** Brightness depends only on depth to the first solid. Real detectors also
  respond to the angle and size of the opening, so very small and very large pores are approximated.
