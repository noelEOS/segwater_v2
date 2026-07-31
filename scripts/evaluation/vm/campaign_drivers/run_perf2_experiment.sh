#!/bin/bash
# EXPERIMENT (2026-07-29): bf16 + device-stitch (4th row of INFERENCE_SPEEDUP_NOTES
# table, expected 1.69x loop). Adds stitching.accumulate_on_device=true on top of
# the PERF arm (bf16+tf32). Compare against BOTH the fp32 baseline and PERF.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
run () {
  echo "=== START $1 ==="; s=$(date +%s)
  if python scripts/run_inference_sweep.py "$2" > ~/workspace/logs/logs_perf2_$1.log 2>&1; then
    e=$(date +%s); echo "=== DONE $1 ($((e-s))s) $(grep -oE 'Successful scenes: [0-9]+ \| Failed scenes: [0-9]+' ~/workspace/logs/logs_perf2_$1.log|tail -1) ==="
  else e=$(date +%s); echo "=== FAILED $1 ($((e-s))s) ==="; tail -8 ~/workspace/logs/logs_perf2_$1.log; fi
}
run demak_PERF2 ~/configs/perf_experiment/demak_gate_mx630s2_s42_last_PERF2.yaml
run narra_PERF2 ~/configs/perf_experiment/narrabeen_mx630s2_s42_last_PERF2.yaml
echo "=== PERF2 EXPERIMENT FINISHED ==="
