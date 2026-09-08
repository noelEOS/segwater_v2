#!/usr/bin/env bash
# 6 v3 Demak concurrent-gate arms (6 scenes each), strictly serial.
set -u
cd ~/segwater_v2
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
ok=0; fail=0
for arch in swinb cnxb; do
  for seed in s42 s19 s58; do
    a="${arch}_${seed}"
    echo "=== $a start $(date -u +%H:%M:%S)"
    python scripts/run_inference_sweep.py \
      scripts/evaluation/vm/configs/demak/inference_sweep_v3_demak_concurrent_${a}.yaml \
      > ~/logs_v3/demak_conc_${a}.log 2>&1
    rc=$?; [ $rc -eq 0 ] && ok=$((ok+1)) || fail=$((fail+1))
    echo "=== $a exit $rc $(date -u +%H:%M:%S)"
  done
done
echo "DEMAK CONC DONE $(date -u +%H:%M:%S)  ok=$ok fail=$fail"
