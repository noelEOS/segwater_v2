#!/bin/bash
# Waits for the in-flight 3seed SDS batch to release the GPU, then runs the
# Demak FULL SERIES (213 scenes) for all three seeds, sequentially.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
while ! grep -q "ALL 3SEED INFERENCE FINISHED" /home/noel/3seed_batch.log 2>/dev/null; do sleep 30; done
echo "=== GPU released, starting demak full series ==="
for cfg in ~/configs/demak_full_3seed/demak_full_*.yaml; do
  n=$(basename "$cfg" .yaml)
  echo "=== START $n ==="
  if python scripts/run_inference_sweep.py "$cfg" > ~/logs_demakfull_$n.log 2>&1; then
    echo "=== DONE $n ($(grep -oE "Successful scenes: [0-9]+ \| Failed scenes: [0-9]+" ~/logs_demakfull_$n.log | tail -1)) ==="
  else
    echo "=== FAILED $n ==="; tail -4 ~/logs_demakfull_$n.log
  fi
done
echo "=== ALL DEMAK FULL FINISHED ==="
