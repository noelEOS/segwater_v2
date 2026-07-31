#!/bin/bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
# Demak first (fast, and it is the acceptance gate), then the SDS sites
for cfg in ~/configs/noaug_last_3seed/demak_*.yaml ~/configs/noaug_last_3seed/narrabeen_*.yaml \
           ~/configs/noaug_last_3seed/duck_*.yaml ~/configs/noaug_last_3seed/trucvert_*.yaml \
           ~/configs/noaug_last_3seed/torreypines_*.yaml; do
  n=$(basename "$cfg" .yaml)
  echo "=== START $n ==="
  if python scripts/run_inference_sweep.py "$cfg" > ~/logs_3seed_$n.log 2>&1; then
    echo "=== DONE $n ($(grep -oE "Successful scenes: [0-9]+ \| Failed scenes: [0-9]+" ~/logs_3seed_$n.log | tail -1)) ==="
  else
    echo "=== FAILED $n ==="; tail -4 ~/logs_3seed_$n.log
  fi
done
echo "=== ALL 3SEED INFERENCE FINISHED ==="
