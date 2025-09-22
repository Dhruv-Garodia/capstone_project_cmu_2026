#!/usr/bin/env bash
set -e

SPEC_FILE=${1:-data/config/log_normal_auto.txt}
PNG_DIR=${2:-output/lognormal/png_slices}
OUT_MESH=${3:-pfib_mesh_auto_200bin.stl}

python script/generate_synthetic_pfib.py --spec "$SPEC_FILE"
python script/convert_mesh.py --png_dir "$PNG_DIR" --out_mesh "$OUT_MESH"