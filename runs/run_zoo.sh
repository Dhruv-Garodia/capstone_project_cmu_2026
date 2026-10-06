#!/bin/bash
# Architecture comparison (docs/MODELS.md). Identical budget for every model:
# 4000 iterations x batch 8, 256 px crops, same augmentation, same split
# (train 0-74 | checkpoint selection 79-90 | test 100-117, evaluated once). No registration.
# Needs data/raw/comp_full.tif, data/manual_annotation/ and data/annotations_v2/ (see README).
cd "$(dirname "$0")/.."
PY=${PY:-python}
C="--stack data/raw/comp_full.tif --labels data/annotations_v2 --eval-labels data/manual_annotation --iters 4000 --bs 8 --eval-every 500"
run() { name=$1; shift; [ -f runs/$name/summary.json ] && return; $PY -m eponge.train $C --out runs/$name "$@" > runs/$name.log 2>&1; }
run unet2d_v2       --arch resunet --bs 16          # the model deployed on the website
run resunet_k0      --arch resunet
run garodia_unet_k0 --arch unet_garodia
run garodia_unet_k3 --arch unet_garodia --k 3       # 2.5D, 7 slices: library default for stacks
run segformer_k0    --arch segformer
run unetpp_r34_k0   --arch unetpp_r34
run transunet_k0    --arch transunet
run resunet_v1labels_k0 --arch resunet --labels data/manual_annotation --eval-labels data/annotations_v2   # label ablation
echo done
