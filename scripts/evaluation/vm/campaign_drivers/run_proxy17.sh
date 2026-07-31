#!/bin/bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
for cfg in ~/configs/demak_proxy17/demak_proxy17_*.yaml; do
  n=$(basename "$cfg" .yaml)
  echo "=== START $n ==="
  if python scripts/run_inference_sweep.py "$cfg" > ~/logs_$n.log 2>&1; then
    echo "=== DONE $n ($(grep -oE "Successful scenes: [0-9]+ \| Failed scenes: [0-9]+" ~/logs_$n.log | tail -1)) ==="
  else
    echo "=== FAILED $n ==="
  fi
done
echo "=== ALL PROXY17 FINISHED ==="
