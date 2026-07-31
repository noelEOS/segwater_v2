#!/bin/bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
for cfg in ~/segwater_v2/configs/narrabeen/narrabeen_s42_best.yaml ~/segwater_v2/configs/narrabeen/narrabeen_s42_last.yaml \
           ~/segwater_v2/configs/narrabeen/narrabeen_s42noaug_best.yaml ~/segwater_v2/configs/narrabeen/narrabeen_s42noaug_last.yaml \
           ~/configs/duck_sds/duck_s42aug_best.yaml ~/configs/duck_sds/duck_s42aug_last.yaml \
           ~/configs/duck_sds/duck_s42noaug_best.yaml ~/configs/duck_sds/duck_s42noaug_last.yaml; do
  n=$(basename "$cfg" .yaml)
  echo "=== START $n ==="
  if python scripts/run_inference_sweep.py "$cfg" > ~/logs_rerun_$n.log 2>&1; then
    echo "=== DONE $n ($(grep -oE "Successful scenes: [0-9]+ \| Failed scenes: [0-9]+" ~/logs_rerun_$n.log | tail -1)) ==="
  else
    echo "=== FAILED $n ==="; tail -4 ~/logs_rerun_$n.log
  fi
done
echo "=== ALL RERUN INFERENCE FINISHED ==="
