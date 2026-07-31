#!/bin/bash
# EXPERIMENT (2026-07-29): inference perf levers ON vs OFF.
# amp_dtype=bfloat16 + tf32=true, vs the shipped fp32 default.
# Model: Swin-B mx630 stage-2, arm=last, s42. Sites: Demak concurrent, Narrabeen.
# Baseline is RE-RUN here (not taken from old logs) so both arms see the same
# idle GPU -- the earlier baseline logs were recorded under GPU contention.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
run () {  # $1=label $2=cfg
  echo "=== START $1 ==="
  s=$(date +%s)
  if python scripts/run_inference_sweep.py "$2" > ~/workspace/logs/logs_perf_$1.log 2>&1; then
    e=$(date +%s); echo "=== DONE $1 ($((e-s))s) $(grep -oE 'Successful scenes: [0-9]+ \| Failed scenes: [0-9]+' ~/workspace/logs/logs_perf_$1.log|tail -1) ==="
  else
    e=$(date +%s); echo "=== FAILED $1 ($((e-s))s) ==="; tail -5 ~/workspace/logs/logs_perf_$1.log
  fi
}
run demak_BASE ~/configs/mx630s2/demak_gate_mx630s2_s42_last.yaml
run demak_PERF ~/configs/perf_experiment/demak_gate_mx630s2_s42_last_PERF.yaml
run narra_BASE ~/configs/mx630s2/narrabeen_mx630s2_s42_last.yaml
run narra_PERF ~/configs/perf_experiment/narrabeen_mx630s2_s42_last_PERF.yaml
echo "=== PERF EXPERIMENT FINISHED ==="
