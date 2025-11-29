#!/bin/bash

# python utils/tif_to_png.py \
#   --input data/pFIB/pristine_full.tif \
#   --out-dir data/pFIB-pristine \
#   --prefix real

# python utils/visualize.py data/pFIB-segmented_resized/ 

rm -rf data/paper-seg-output/pFIB-seg_resized_cropped
python utils/crop_real_stack.py \
  --input-tif data/paper-seg-output/pFIB-seg_resized \
  --out-dir data/paper-seg-output/pFIB-seg_resized_cropped \
  --cx 664.7 \
  --cy 197.2 \
  --crop-size 150
rm data/paper-seg-output/pFIB-seg_resized_cropped/ref*

rm -rf data/model-seg-output/real/pFIB-seg_resized_cropped
python utils/crop_real_stack.py \
  --input-tif data/model-seg-output/real/pFIB-seg_resized \
  --out-dir data/model-seg-output/real/pFIB-seg_resized_cropped \
  --cx 736.9 \
  --cy 200 \
  --crop-size 150

rm data/model-seg-output/real/pFIB-seg_resized_cropped/ref*
python utils/visualize.py data/model-seg-output/real/pFIB-seg_resized_cropped data/paper-seg-output/pFIB-seg_resized_cropped