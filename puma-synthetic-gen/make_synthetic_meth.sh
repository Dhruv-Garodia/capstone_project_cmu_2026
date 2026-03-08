#!/usr/bin/env bash
set -e

SPEC_FILE=${1:-input/config/adjusted_config.txt}
OUT_DIR_OVERRIDE=${2:-}

python scripts/generate_synthetic_pfib.py --spec "$SPEC_FILE"

if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
  mkdir -p "$OUT_DIR_OVERRIDE"
  cp "$SPEC_FILE" "$OUT_DIR_OVERRIDE/"
else
  OUT_DIR=$(awk -F: '/^[[:space:]]*out_dir[[:space:]]*:/ {sub(/^[[:space:]]+/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}' "$SPEC_FILE")
  if [[ -n "$OUT_DIR" ]]; then
    mkdir -p "$OUT_DIR"
    cp "$SPEC_FILE" "$OUT_DIR/"
  fi
fi
