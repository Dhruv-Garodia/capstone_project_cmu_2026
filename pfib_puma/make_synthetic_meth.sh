#!/usr/bin/env bash
set -e

SPEC_FILE=${1:-data/config/adjusted_config.txt}
PNG_DIR=${2:-output/synthetic_pfibs/synthetic_7/png_slices}
OUT_MESH=${3:-synthetic_7.stl}

python script/generate_synthetic_pfib.py --spec "$SPEC_FILE"
cp "$SPEC_FILE" "$(dirname "$PNG_DIR")/"