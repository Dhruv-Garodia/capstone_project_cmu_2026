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
