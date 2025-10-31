#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PFIB_FORCE_DEVICE=cpu

rm -rf test_out

python scripts/test_model.py \
  --img_dir data/synthetic_pfibs_resized/synthetic_10 \
  --ckpt checkpoints/unet_best.pt \
  --out_dir test_out

