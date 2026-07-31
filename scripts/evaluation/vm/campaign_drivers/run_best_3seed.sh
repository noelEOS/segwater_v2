#!/bin/bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
# gate first (fast, 6 scenes), then the 213-scene full series
for cfg in ~/configs/demak_best_3seed/demak_gate_*.yaml ~/configs/demak_best_3seed/demak_full_*.yaml; do
  n=$(basename "$cfg" .yaml); echo "=== START $n ==="
  if python scripts/run_inference_sweep.py "$cfg" > ~/logs_best_$n.log 2>&1; then
    echo "=== DONE $n ($(grep -oE "Successful scenes: [0-9]+ \| Failed scenes: [0-9]+" ~/logs_best_$n.log | tail -1)) ==="
  else echo "=== FAILED $n ==="; tail -4 ~/logs_best_$n.log; fi
done
echo "=== ALL BEST-CKPT INFERENCE FINISHED ==="
