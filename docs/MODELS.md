# Models: what we trained and which one to use

## Short answer

* **Single images and the website:** Éponge **ResUNet** (2.4 M parameters). It ties for the best
  accuracy, is the smallest model, and is the only one small enough to run in a browser.
* **Stacks in the Python library:** **Garodia U-Net, 2.5D** (7 adjacent slices as input). It ties
  for best accuracy with 3× lower porosity error and the best z-consistency among the top models.
* Larger pretrained models (UNet++, SegFormer, TransUNet) did **not** improve accuracy here.

## Setup (identical for every model)

| | |
|---|---|
| data | `comp_full.tif`, corrected labels v2 (see [ANNOTATIONS.md](ANNOTATIONS.md)) |
| split | train frames 0–74 · checkpoint selection 79–90 · **test 100–117, evaluated once** · buffers 75–78, 91–99 |
| budget | 4,000 iterations × batch 8, random 256 px crops |
| augmentation | horizontal flip, scale 0.6–1.6×, gamma, contrast, blur, noise, curtaining stripes |
| loss | class-weighted cross-entropy + Dice over 3 classes (solid, pore, outside the layer) |
| optimiser | AdamW + one-cycle; learning rate 2e-3 (ResUNet), 1e-3 (Garodia), 5e-4 (UNet++), 2e-4 (SegFormer), 1e-4 (TransUNet) |
| inference | full frame (tiles of 1024 px; 256 px for TransUNet, matching its position embeddings), flip TTA |
| hardware | Apple M4, PyTorch MPS; 17–81 min per model |

## Architectures

| name | what it is | params | pretrained |
|---|---|---:|---|
| ResUNet | Éponge residual U-Net, 5 levels, channels capped at 128, dropout for MC uncertainty | 2.4 M | no |
| Garodia U-Net | D. Garodia's U-Net from `pore_pipeline/unet/model.py` (3-class head) | 7.8 M | no |
| UNet++ | nested U-Net (Zhou et al. 2018) with a ResNet-34 encoder | 26.1 M | ImageNet |
| SegFormer-B2 | hierarchical transformer + all-MLP decoder (Xie et al. 2021) | 24.7 M | ImageNet |
| TransUNet | R50 + ViT-B/16 hybrid encoder, cascaded upsampler (Chen et al. 2021, [arXiv:2102.04306](https://arxiv.org/abs/2102.04306)) | 105.3 M | ImageNet-21k |

All live in `eponge/src/eponge/models/zoo.py`; train any of them with
`eponge train ... --arch {resunet,unet_garodia,unetpp_r34,segformer,transunet} [--k K]`.

## Results on the held-out test frames

| model | params | mIoU | pore IoU | layer IoU | pore F1 ±1px | porosity error | z-consistency | pore IoU vs Garodia trimap | in Garodia bracket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResUNet (Éponge, 2D, batch 16) · on the website | 2.4 M | 0.912 | 0.845 | 0.975 | 0.977 | 0.016 | 0.867 | 0.770 | 100 % |
| Garodia U-Net (2.5D, 7 slices) | 7.9 M | 0.912 | 0.844 | 0.972 | 0.980 | 0.005 | 0.872 | 0.771 | 100 % |
| ResUNet (Éponge, 2D) | 2.4 M | 0.911 | 0.842 | 0.974 | 0.978 | 0.005 | 0.864 | 0.773 | 100 % |
| Garodia U-Net (2D) | 7.8 M | 0.904 | 0.831 | 0.972 | 0.974 | 0.011 | 0.869 | 0.774 | 100 % |
| UNet++ ResNet-34 (2D, ImageNet) | 26.1 M | 0.903 | 0.828 | 0.973 | 0.980 | 0.010 | 0.870 | 0.776 | 100 % |
| TransUNet R50-ViT-B/16 (2D, ImageNet-21k) | 105.3 M | 0.902 | 0.826 | 0.975 | 0.980 | 0.004 | 0.867 | 0.759 | 100 % |
| SegFormer-B2 (2D, ImageNet) | 24.7 M | 0.894 | 0.812 | 0.975 | 0.976 | 0.007 | 0.874 | 0.758 | 100 % |
| ResUNet trained on v1 labels | 2.4 M | 0.706 | 0.435 | 0.951 | 0.737 | 0.182 | 0.817 | 0.541 | 0 % |
| Previous 2D U-Net (v1 labels, earlier experiment) | 3.3 M | 0.678 | 0.393 | 0.938 | 0.694 | 0.201 | 0.781 | 0.501 | 0 % |
| *Classical baselines (given the true layer outline)* | | | | | | | | | |
| Otsu on raw image* | – | 0.922 | 0.855 | 1.000 | 0.962 | 0.016 | 0.809 | 0.757 | 100 % |
| Local threshold* | – | 0.797 | 0.644 | 1.000 | 0.819 | 0.068 | 0.841 | 0.594 | 100 % |
| Threshold I < 140* | – | 0.724 | 0.422 | 1.000 | 0.699 | 0.211 | 0.763 | 0.469 | 0 % |

Test frames 100–117 (never used for training or checkpoint selection). Reference porosity (v2): 0.366. * Baselines are given the reference catalyst-layer outline (layer IoU 1.000); the networks find the layer themselves. Raw Otsu scores high on mIoU because the v2 labels are themselves a denoised Otsu rule; its boundary noise shows in the lowest z-consistency. z-consistency: pore Dice between neighbouring predicted slices (labels themselves: 0.855 v2, 0.764 v1 on frames 61–90). Garodia trimap: D. Garodia's independent Otsu + Ferner-style consensus, scored on its confident pixels; 'in bracket' = share of test frames whose predicted porosity lies between his low and high estimates.

How to read the columns:

* **mIoU, pore IoU, layer IoU, pore F1 ±1px, porosity error** compare against the v2 labels.
  "±1px" ignores disagreements within one pixel of a pore boundary.
* **z-consistency** is label-free: pore Dice between neighbouring predicted slices. Real structure
  persists over a few 6 nm slices; noise does not.
* **vs Garodia trimap** scores against an independent rule (Otsu + Ferner-style consensus from
  `pore_pipeline`) on the pixels that rule is confident about. **In Garodia bracket** is the share of
  test frames whose porosity lies between that pipeline's low and high estimates.

## What the comparison shows

1. **Labels matter far more than architecture.** The same ResUNet trained on the original v1 masks
   drops from 0.911 to 0.706 mIoU and predicts porosity 0.18. That lies outside the independent
   Garodia bracket on 0 % of frames, while every model trained on v2 lies inside it on 100 %.
2. **The architectures are statistically tied.** On v2, all five lie within 0.018 mIoU; against
   the independent trimap their ranking reshuffles (0.758–0.776). Nearly all remaining
   disagreement is one-pixel boundary placement (pore F1 0.974–0.980 with ±1 px tolerance).
3. **Bigger and pretrained did not help.** The target is a local grey-level/texture decision on 75
   frames of one stack, which a 2.4 M-parameter convolutional network captures fully. ImageNet
   features and global self-attention (TransUNet, SegFormer) add parameters, not accuracy.
   TransUNet does give the lowest porosity error (0.004).
4. **Slice context helps a little.** Garodia's U-Net with ±3 neighbouring slices gains +0.008 mIoU
   and has the best z-consistency of the top models. No registration is applied (see FINDINGS.md).
5. **Previous model.** The earlier 2D U-Net from the real-data experiment scores 0.678 mIoU against
   v2. It learned the v1 threshold rule faithfully, so its low score reflects the labels it was given.

## Reproduce

```bash
bash runs/run_zoo.sh                                   # all architecture runs (~4 h on an M4)
python eponge/scripts/compare_models.py                # table above + webapp/model/results.json
python eponge/scripts/compare_models.py --from-json    # rebuild the table only
```

Per-run outputs: `runs/<name>/summary.json` (history, test metrics), `runs/<name>.log`,
`runs/<name>/test_predictions/`. Checkpoints (`best.pt`) are not in git.
