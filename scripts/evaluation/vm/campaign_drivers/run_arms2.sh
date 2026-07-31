#!/bin/bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
ORDER="demak_gate_mx630s2_s42_swa5 demak_gate_mx630k_s42_best hampyeong_mx630s2_s42_swa5 hampyeong_mx630k_s42_best narrabeen_mx630s2_s42_swa5 narrabeen_mx630k_s42_best demak_full_mx630s2_swa5 demak_full_mx630k_best"
for n in $ORDER; do
  cfg=~/configs/mx630_arms2/$n.yaml
  echo "=== START $n ==="
  if python scripts/run_inference_sweep.py "$cfg" > ~/logs_arms2_$n.log 2>&1; then
    echo "=== DONE $n ($(grep -oE 'Successful scenes: [0-9]+ \| Failed scenes: [0-9]+' ~/logs_arms2_$n.log | tail -1)) ==="
  else echo "=== FAILED $n ==="; tail -5 ~/logs_arms2_$n.log; fi
done
echo "=== ALL ARMS2 FINISHED ==="
