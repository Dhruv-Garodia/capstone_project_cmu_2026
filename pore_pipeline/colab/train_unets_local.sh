#!/bin/sh
# Trains the 2.5D (k=3) and 2D (k=0) U-Nets on comp, predicts the whole stack, and runs the label-free checks
# on slices 128+ (never used for training or validation).
cd "$(dirname "$0")/../.."
for K in 3 0; do
  O=outputs/comp/unet_k$K
  python3 -u -m pore_pipeline.unet.train data/comp_full.tif outputs/comp --labels outputs/comp/manual_borders/high \
      --train-z 0-90 --val-z 100-117 --k $K --iters 2000 --val-every 250 --out $O > $O.train.log 2>&1
  python3 -u -m pore_pipeline.unet.predict data/comp_full.tif outputs/comp --ckpt $O/best.pt --out $O/masks \
      --region outputs/comp/manual_borders/high >> $O.train.log 2>&1
done
python3 -m pore_pipeline.evaluate outputs/comp --tag unseen --z0 128 \
    --masks otsu=outputs/comp/masks/otsu ferner_style=outputs/comp/masks/ferner_style \
            unet_k0=outputs/comp/unet_k0/masks unet_k3=outputs/comp/unet_k3/masks > outputs/comp/unet_eval.log 2>&1
echo done > outputs/comp/unet_done.flag
