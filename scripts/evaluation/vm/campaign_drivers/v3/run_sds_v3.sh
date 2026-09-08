#!/usr/bin/env bash
# 24 v3 SDS arms (4 frames x 2 archs x 3 seeds), strictly serial: one GPU,
# one writer per output dir. Failures are logged and the loop continues, so a
# single bad arm cannot silently abort the rest; the tally at the end is the
# thing to read.
set -u
cd ~/segwater_v2
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib

ok=0; fail=0
for site in narrabeen duck torreypines trucvert; do
  for arch in swinb cnxb; do
    for seed in s19 s42 s58; do
      a="${site}_${arch}_${seed}"
      cfg="scripts/evaluation/vm/configs/sds/inference_sweep_v3_sds_${a}.yaml"
      echo "=== $a start $(date -u +%H:%M:%S)"
      python scripts/run_inference_sweep.py "$cfg" > ~/logs_v3/sds_${a}.log 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
      echo "=== $a exit $rc $(date -u +%H:%M:%S)"
    done
  done
done
echo "SDS ALL DONE $(date -u +%H:%M:%S)  ok=$ok fail=$fail"
