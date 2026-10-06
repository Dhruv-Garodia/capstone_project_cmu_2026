#!/bin/bash
# Separating MillNet's gain: scale augmentation vs factorised head vs axial attention vs histogram FiLM.
cd "$(dirname "$0")/.."
PY=${PY:-python}
C="--stack data/raw/comp_full.tif --labels data/annotations_v2 --eval-labels data/manual_annotation --iters 4000 --bs 8 --eval-every 500"
run() { name=$1; shift; [ -f runs/$name/summary.json ] && return; $PY -m eponge.train $C --out runs/$name "$@" > runs/$name.log 2>&1; }
run resunet_k0_narrowscale     --arch resunet --scale-range 0.8 1.25                         # is it the augmentation?
run millnet_k0_wide            --arch millnet --scale-range 0.6 1.6                          # MillNet 2D with the baseline augmentation
run millnet_k0_headonly        --arch millnet --no-stats --no-axial --scale-range 0.6 1.6    # factorised head alone
run millnet_k0_noaxial         --arch millnet --no-axial --scale-range 0.6 1.6               # head + FiLM, no attention
echo done
