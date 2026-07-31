#!/bin/bash
# Calendar-matched (and same-window) frequency maps for ALL THREE lineages,
# so each can be compared against the registry's matching estimand.
export CUDA_VISIBLE_DEVICES=""
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
mkdir -p ~/workspace/logs/freqmaps
ok=0; fail=0
for spec in "ship:ship_decision_2026-07" "cnxb:ship_decision_cnxb_2026-07" "cnxt:ship_decision_cnxt_2026-07"; do
  tag=${spec%%:*}; dir=${spec##*:}
  OUT=~/workspace/results/$dir/freq_maps
  for ss in matched samewin; do
    for variant in best last; do
      n="${tag}_${variant}_s32_${ss}"
      if python scripts/evaluation/vm/ship/run_ship_freqmaps.py \
           --variant "$variant" --stride 32 --tag "$tag" --scene-set "$ss" --out-dir "$OUT" \
           > ~/workspace/logs/freqmaps/$n.log 2>&1; then
        ok=$((ok+1)); echo "  OK   $n"
      else fail=$((fail+1)); echo "  FAIL $n"; tail -5 ~/workspace/logs/freqmaps/$n.log; fi
    done
  done
done
echo "=== MATCHED/SAMEWIN DONE: ok=$ok fail=$fail (expect 12) $(date -u +%H:%M:%SZ) ==="
