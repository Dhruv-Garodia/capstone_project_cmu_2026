#!/bin/bash
# MillNet experiments (docs/MILLNET.md): full model, ablations, and second seeds for the
# strongest existing models so that differences can be judged against seed-to-seed noise.
# Same budget and split as run_zoo.sh.
cd "$(dirname "$0")/.."
PY=${PY:-python}
C="--stack data/raw/comp_full.tif --labels data/annotations_v2 --eval-labels data/manual_annotation --iters 4000 --bs 8 --eval-every 500"
run() { name=$1; shift; [ -f runs/$name/summary.json ] && return; $PY -m eponge.train $C --out runs/$name "$@" > runs/$name.log 2>&1; }
run millnet_k3            --arch millnet --k 3
run millnet_k3_s1         --arch millnet --k 3 --seed 1
run millnet_k3_nobank     --arch millnet --k 3 --no-bank        # ablation: plain neighbour channels
run millnet_k3_nostats    --arch millnet --k 3 --no-stats       # ablation: no histogram conditioning
run millnet_k0            --arch millnet --k 0                  # 2D variant (single images)
run resunet_k0_s1         --arch resunet --seed 1
run garodia_unet_k3_s1    --arch unet_garodia --k 3 --seed 1
echo done
