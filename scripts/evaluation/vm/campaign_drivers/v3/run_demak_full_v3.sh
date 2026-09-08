#!/usr/bin/env bash
# 6 v3 Demak full-series arms (213 scenes each), strictly serial.
# Runs alongside the CPU-bound SDS scorer: inference is GPU-bound and the two
# do not contend for the same resource. One writer per output dir either way.
set -u
cd ~/segwater_v2
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib

ok=0; fail=0
for arch in swinb cnxb; do
  for seed in s42 s19 s58; do
    a="${arch}_${seed}"
    echo "=== demak_full $a start $(date -u +%H:%M:%S)"
    python scripts/run_inference_sweep.py \
      scripts/evaluation/vm/configs/demak/inference_sweep_v3_demak_full_${a}.yaml \
      > ~/logs_v3/demak_full_${a}.log 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    echo "=== demak_full $a exit $rc $(date -u +%H:%M:%S)"
  done
done
echo "DEMAK FULL DONE $(date -u +%H:%M:%S)  ok=$ok fail=$fail"
