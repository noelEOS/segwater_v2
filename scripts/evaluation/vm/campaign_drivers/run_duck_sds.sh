#!/bin/bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
for cfg in ~/configs/duck_sds/duck_s42*.yaml; do
  n=$(basename "$cfg" .yaml)
  echo "=== START $n ==="
  if python scripts/run_inference_sweep.py "$cfg" > ~/logs_$n.log 2>&1; then
    echo "=== DONE $n ($(grep -oE "Successful scenes: [0-9]+ \| Failed scenes: [0-9]+" ~/logs_$n.log | tail -1)) ==="
  else
    echo "=== FAILED $n ==="; tail -5 ~/logs_$n.log
  fi
done
echo "=== ALL DUCK INFERENCE FINISHED ==="
