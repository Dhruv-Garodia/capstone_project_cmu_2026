# pore_pipeline

Pore/solid segmentation of pFIB-SEM catalyst-layer stacks: a knob-free classical
pipeline, label-free checks, and a 2.5D U-Net trained on its output.

Run everything from the `Capstone/` folder. Needs numpy, scipy, scikit-image, tifffile,
Pillow, PyWavelets, matplotlib, and PyTorch for the U-Net.

## Layout

    labels.py         label values, mask file helpers, overlays
    preprocess.py     de-curtaining, histogram normalization, preprocessed-frame cache
    roi.py            catalyst-layer detection
    segment.py        Otsu, Ferner-style, consensus trimap
    run.py            classical pipeline on one stack
    apply_borders.py  hand-drawn borders + pipeline phases -> training labels
    evaluate.py       label-free checks (z-profile, x-z isotropy)
    unet/             model.py, data.py, train.py, predict.py

Every mask is an 8-bit PNG named `slice_XXXX.png` (zero-based TIFF page):
**0 pore, 64 uncertain, 128 excluded, 255 solid.**

## 1. Classical pipeline

    python -m pore_pipeline.run data/comp_full.tif --out outputs/comp

1. **De-curtain** every frame (wavelet-FFT, Muench 2009).
2. **Catalyst-layer ROI**: the layer is the only textured band; boundaries are smoothed along
   x and z, shrunk by `--margin`, and delamination gaps opening onto the edge are removed.
3. **Normalize** each slice's ROI histogram to a reference slice (removes brightness drift in z).
4. **Calibrate** per stack: Otsu and 3-class thresholds, streak direction, gradient threshold.
5. **Segment**: `otsu`, and `ferner_style` (approximation of Ferner et al. 2024: gaussian,
   gradient along the streak, low/high grey rules, 3D open/close, 3D gaussian).
6. **Consensus**: pore or solid where both agree, uncertain where they differ.

Slices are deliberately **not** registered by cross-correlation: what correlates between
neighbouring slices is sub-surface material sliding behind a nearly stationary cut face.

Outputs: `masks/{otsu,ferner_style,consensus}/`, `stats.csv`, `summary.json`, `preview.png`.

## 2. Labels for training

    python -m pore_pipeline.apply_borders data/comp_full.tif outputs/comp \
        outputs/comp/manual_annotation --out outputs/comp/manual_borders

Uses the team's hand-drawn layer borders and the pipeline's phases. Writes `trimap/`
(uncertain kept), `high/` (uncertain = pore, the working assumption) and `low/`
(uncertain = solid). Carry both `high` and `low` through any result: the spread is the
error bar on the assumption.

## 3. Label-free checks

    python -m pore_pipeline.evaluate outputs/comp
    python -m pore_pipeline.evaluate outputs/comp --tag unseen --z0 128 \
        --masks ferner_style=outputs/comp/masks/ferner_style unet_k3=outputs/comp/unet_k3/masks

`zprofile.png`: porosity and brightness per slice (should be flat). `isotropy*.png`:
two-point correlation of the pore phase along x and z. Both are in-plane directions of the
electrode, so an x-z difference is an imaging or segmentation artifact (shine-through).

## 4. 2.5D U-Net

    python -m pore_pipeline.unet.train data/comp_full.tif outputs/comp \
        --labels outputs/comp/manual_borders/high --train-z 0-90 --val-z 100-117 \
        --k 3 --out outputs/comp/unet_k3
    python -m pore_pipeline.unet.predict data/comp_full.tif outputs/comp \
        --ckpt outputs/comp/unet_k3/best.pt --out outputs/comp/unet_k3/masks

The network segments slice z from slices z-k..z+k of the preprocessed stack (`--k 0` is a
plain 2D U-Net). Excluded and uncertain pixels are masked out of the loss. Train and
validation slices are separated along z, because neighbouring slices are near-duplicates.

`outputs/comp/train_unets.sh` reproduces the k=3 and k=0 runs, prediction and evaluation.

## Caveats

- `ferner_style` is not Ferner's code; the MATLAB source and supplementary were unavailable.
- The training labels are rule-derived. Agreement between a network and these labels shows
  the network learned the rule, not that the rule is right. The evidence that matters is
  label-free (x-z isotropy, flatness along z) or independent (mass balance, other stacks).
