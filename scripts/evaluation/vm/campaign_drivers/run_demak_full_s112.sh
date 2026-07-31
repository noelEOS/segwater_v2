#!/bin/bash
# Demak FULL SERIES (213 scenes), stride 112, bf16 + tf32 + device-stitch.
# Swin-B mx630 stage-2, arm=last, s42. Trend at BOTH the full 206-scene window
# and the S2-date-matched (n=48) estimand.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
s=$(date +%s); echo "=== START demak_full_s112_bf16dev ==="
if python scripts/run_inference_sweep.py ~/configs/perf_experiment/demak_full_mx630s2_last_s112_bf16dev.yaml \
     > ~/workspace/logs/logs_demak_full_s112_bf16dev.log 2>&1; then
  e=$(date +%s); echo "=== DONE ($((e-s))s) $(grep -oE 'Successful scenes: [0-9]+ \| Failed scenes: [0-9]+' ~/workspace/logs/logs_demak_full_s112_bf16dev.log|tail -1) ==="
else e=$(date +%s); echo "=== FAILED ($((e-s))s) ==="; tail -8 ~/workspace/logs/logs_demak_full_s112_bf16dev.log; fi
echo "=== S112 FULL SERIES FINISHED ==="
