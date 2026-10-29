#!/bin/sh
# the same comp-like material at five porosities, quick width; each with viewer file and GIFs
for p in 0.30 0.37 0.44 0.51 0.58; do
  python generate.py comp_like --porosity $p --frames 40 --width 512 --name sweep_por_$p --view
done
