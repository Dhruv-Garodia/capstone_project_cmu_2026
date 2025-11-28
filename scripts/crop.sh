#!/bin/bash

# python utils/tif_to_png.py \
#   --input data/pFIB/pristine_full.tif \
#   --out-dir data/pFIB-pristine \
#   --prefix real

# python utils/visualize.py data/pFIB-segmented_resized/ 

python utils/crop_real_stack.py \
  --input-tif data/pFIB-segmented_resized/ \
  --out-dir data/pFIB-segmented_resized_cropped \
  --cx 400.0 \
  --cy 200.0

# python utils/crop_real_stack.py \
#   --input-tif data/pFIB-segmented_resized/ \
#   --out-dir data/pFIB-segmented_resized_cropped \
#   --cx 400.0 \
#   --cy 200.0
