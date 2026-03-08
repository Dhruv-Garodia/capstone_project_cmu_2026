#!/bin/bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PFIB_FORCE_DEVICE="${PFIB_FORCE_DEVICE:-cpu}"

IMG_DIR="${1:-data/pFIB-pristine}"
CKPT="${2:-checkpoints/unet_best.pt}"
OUT_DIR="${3:-test_out_real_nocropping}"
THRESH="${4:-0.6}"

if [[ ! -d "$IMG_DIR" ]]; then
  echo "[ERROR] Input image directory not found: $IMG_DIR"
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[ERROR] Checkpoint file not found: $CKPT"
  exit 1
fi

python utils/test_model.py \
  --img_dir "$IMG_DIR" \
  --ckpt "$CKPT" \
  --out_dir "$OUT_DIR" \
  --thresh "$THRESH"

# Optional evaluations:
# python utils/eval.py \
#   "$OUT_DIR" \
#   --pores-are-black \
#   --label Prediction
#
# python utils/eval_porosity.py \
#   "$OUT_DIR" \
#   --pores-are-black \
#   --label Prediction
