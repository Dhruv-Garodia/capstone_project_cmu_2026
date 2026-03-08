#!/bin/bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PFIB_FORCE_DEVICE="${PFIB_FORCE_DEVICE:-cuda}"

IMG_DIR="${1:-data/lightened_pfibs_500}"
MASK_DIR="${2:-data/synthetic_pfibs_500}"
OUT_DIR="${3:-checkpoints}"

if [[ ! -d "$IMG_DIR" ]]; then
  echo "[ERROR] Image directory not found: $IMG_DIR"
  exit 1
fi
if [[ ! -d "$MASK_DIR" ]]; then
  echo "[ERROR] Mask directory not found: $MASK_DIR"
  exit 1
fi

python utils/train.py \
  --img_dir "$IMG_DIR" \
  --mask_dir "$MASK_DIR" \
  --out "$OUT_DIR"
