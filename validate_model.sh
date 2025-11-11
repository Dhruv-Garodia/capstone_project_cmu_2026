#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PFIB_FORCE_DEVICE=cpu


python scripts/test_model.py \
  --img_dir data/lightened_pfibs_300/synthetic_10 \
  --ckpt checkpoints/unet_best_300_final.pt \
  --out_dir test_out_300_final

