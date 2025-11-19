#!/bin/bash

python scripts/tif_to_png.py \
  --input data/pFIB/pristine_full.tif \
  --out-dir data/pFIB-pristine \
  --prefix real

# python scripts/crop_real_stack.py \
#   --input-tif data/pFIB/comp_full.tif \
#   --out-dir data/pFIB-comp-crops-300 \
#   --cx 930.0 \
#   --cy 200.0
