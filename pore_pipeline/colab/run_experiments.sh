#!/bin/sh
# Full experiment grid on a GPU box. Run from the Eponge folder:  sh run_experiments.sh
# Trains on comp (labels high and low, k=3 and k=0, three seeds), predicts all three stacks with
# each k=3 seed-0 model, and runs the label-free checks. Everything is written under outputs/.
set -e
pip install -q PyWavelets tifffile scikit-image 2>/dev/null || true
python3 - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
for LAB in high low; do
  for K in 3 0; do
    for SEED in 0 1 2; do
      O=outputs/comp/unet_${LAB}_k${K}_s${SEED}
      [ -f $O/best.pt ] && { echo "skip $O"; continue; }
      python3 -u -m pore_pipeline.unet.train data/comp_full.tif outputs/comp --labels outputs/comp/manual_borders/$LAB \
          --train-z 0-90 --val-z 100-117 --k $K --iters 2000 --val-every 250 --batch 16 --seed $SEED --out $O 2>&1 | tee $O.train.log
    done
  done
done
for LAB in high low; do
  O=outputs/comp/unet_${LAB}_k3_s0
  for STACK in comp uncomp pristine; do
    REGION=""; [ $STACK = comp ] && REGION="--region outputs/comp/manual_borders/$LAB"
    python3 -u -m pore_pipeline.unet.predict data/${STACK}_full.tif outputs/$STACK --ckpt $O/best.pt \
        --out outputs/$STACK/pred_unet_${LAB}_k3_s0/masks $REGION
  done
done
python3 -m pore_pipeline.evaluate outputs/comp --tag unseen --z0 128 \
    --masks otsu=outputs/comp/masks/otsu ferner_style=outputs/comp/masks/ferner_style \
            unet_high=outputs/comp/pred_unet_high_k3_s0/masks unet_low=outputs/comp/pred_unet_low_k3_s0/masks
for STACK in uncomp pristine; do
  python3 -m pore_pipeline.evaluate outputs/$STACK --tag unet \
      --masks otsu=outputs/$STACK/masks/otsu ferner_style=outputs/$STACK/masks/ferner_style \
              unet_high=outputs/$STACK/pred_unet_high_k3_s0/masks unet_low=outputs/$STACK/pred_unet_low_k3_s0/masks
done
echo ALL DONE
