#!/bin/bash
# swa5 arm, 3 seeds, 4 gates. Same phase order and settings as the best/last
# campaign (cheap gates first; s112 before s32 in the full series).
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
C=~/configs/ship_decision_2026-07
run_phase () {
  local phase="$1"; shift
  echo "=== PHASE $phase START $(date -u +%H:%M:%S) ==="
  for cfg in "$@"; do
    n=$(basename "$cfg" .yaml); s=$(date +%s)
    if python scripts/run_inference_sweep.py "$cfg" > ~/workspace/logs/ship/$n.log 2>&1; then
      e=$(date +%s)
      echo "  DONE $n ($((e-s))s) $(grep -oE 'Successful scenes: [0-9]+ \| Failed scenes: [0-9]+' ~/workspace/logs/ship/$n.log | tail -1)"
    else e=$(date +%s); echo "  FAILED $n ($((e-s))s)"; tail -5 ~/workspace/logs/ship/$n.log; fi
  done
  echo "=== PHASE $phase COMPLETE $(date -u +%H:%M:%S) ==="
}
run_phase 1-demak_gate $C/demak_gate/*_swa5.yaml
run_phase 2-hampyeong  $C/hampyeong/*_swa5.yaml
run_phase 3-narrabeen  $C/narrabeen/*_swa5.yaml
run_phase 4-demak_full_s112 $C/demak_full/*_swa5_s112.yaml
run_phase 5-demak_full_s32  $C/demak_full/*_swa5_s32.yaml
echo "=== ALL SWA5 INFERENCE FINISHED $(date -u +%H:%M:%S) ==="
