#!/bin/bash
# Per-acquisition twin calibration (physical constraint) + fine-tune, then cross-sample validation.
cd "$(dirname "$0")/.."
source ../.venv/bin/activate
D="/Users/aadityavikram/Downloads/Experimental data"
python eponge/scripts/twin_per_stack.py --name comp --stack data/raw/comp_full.tif
python eponge/scripts/twin_per_stack.py --name uncomp --stack "$D/uncomp_full.tif"
python eponge/scripts/twin_per_stack.py --name pristine --stack "$D/pristine_full.tif"
python eponge/scripts/cross_sample.py
