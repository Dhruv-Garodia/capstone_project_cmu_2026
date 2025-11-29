#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PFIB_FORCE_DEVICE=cpu


python utils/test_model.py \
  --img_dir data/pFIB-pristine \
  --ckpt checkpoints/celoss-single/unet_best_300_final.pt \
  --out_dir test_out_real_nocropping \
  --thresh 0.6

# python utils/eval.py \
#   test_out_single_300_nopadding/ \
#   --pores-are-black \
#   --label Pred300

# python utils/eval_porosity.py \
#   data/synthetic_pfibs_300/synthetic_10/ \
#   --pores-are-black \
#   --label GroundTruth300

python utils/eval_porosity.py \
  data/model-recon-output/synthetic_0/convex_hull.obj \
  --pores-are-black

# python utils/convert.py 
# python utils/eval_pore_distribution.py --mesh ground_truth.stl --axis z --n_slices 150