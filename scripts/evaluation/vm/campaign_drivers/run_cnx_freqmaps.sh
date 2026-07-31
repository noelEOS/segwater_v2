#!/bin/bash
# Frequency maps for the two ConvNeXtV2 campaigns: 2 lineages x {best,last} x {s32,s112}.
# CPU-only. Each cell reads its own campaign's demak_full run dirs (3 seeds) and
# writes into that campaign's own freq_maps/ dir -- never a shared one.
export CUDA_VISIBLE_DEVICES=""
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
mkdir -p ~/workspace/logs/freqmaps
ok=0; fail=0
for tag in cnxb cnxt; do
  OUT=~/workspace/results/ship_decision_${tag}_2026-07/freq_maps
  for variant in best last; do
    for stride in 112 32; do
      n="${tag}_${variant}_s${stride}"
      s=$(date +%s)
      if python scripts/evaluation/vm/ship/run_ship_freqmaps.py \
           --variant "$variant" --stride "$stride" --tag "$tag" --out-dir "$OUT" \
           > ~/workspace/logs/freqmaps/$n.log 2>&1; then
        ok=$((ok+1)); echo "  OK   $n ($(( $(date +%s)-s ))s)"
      else
        fail=$((fail+1)); echo "  FAIL $n"; tail -6 ~/workspace/logs/freqmaps/$n.log
      fi
    done
  done
done
echo "=== FREQMAPS DONE: ok=$ok fail=$fail (expect 8) $(date -u +%H:%M:%SZ) ==="
