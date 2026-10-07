#!/bin/bash
# Publication experiments on the public benchmark (docs/MODELS.md, docs/DIGITAL_TWIN.md):
# self-training (MillNet-ST), seeds for every headline number.
cd "$(dirname "$0")/.."
PY=${PY:-python}
B="eponge/scripts/external_benchmark.py --iters 2500 --only millnet_7slice"
run() { log=$1; shift; [ -f runs/$log.done ] && return; caffeinate -i $PY $B "$@" > runs/$log.log 2>&1 && touch runs/$log.done; }
run ext_ssl_s0          --fractions 0.1 --ssl
run ext_twin_ssl_s0     --fractions 0.1 --ssl --init twin
run ext_full_s1         --fractions 1.0 --seed 1
run ext_full_s2         --fractions 1.0 --seed 2
caffeinate -i $PY eponge/scripts/external_benchmark.py --iters 2500 --only resunet_7slice --seed 1 > runs/ext_resunet_s1.log 2>&1
caffeinate -i $PY eponge/scripts/external_benchmark.py --iters 2500 --only resunet_7slice --seed 2 > runs/ext_resunet_s2.log 2>&1
run ext_scratch10_s1    --fractions 0.1 --seed 1
run ext_twin10_s1       --fractions 0.1 --init twin --seed 1
run ext_ssl10_s1        --fractions 0.1 --ssl --seed 1
run ext_twinssl10_s1    --fractions 0.1 --ssl --init twin --seed 1
echo QUEUE_DONE
