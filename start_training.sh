#!/bin/bash
# python scripts/process_filename.py
# python scripts/resize_pngs.py

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PFIB_FORCE_DEVICE=cuda
python scripts/train.py \
  --img_dir data/lightened_pfibs_500 \
  --mask_dir data/synthetic_pfibs_500 
