#!/usr/bin/env bash
# Remaining 5 hamp24 v3 arms, strictly serial: one GPU, one writer per output dir.
# No self-wait: arm 1 is already finished (24/24) and this script is the only
# thing driving the GPU now. An earlier version pgrep-ed for a pattern its own
# launching shell matched, and waited on itself forever.
set -u
cd ~/segwater_v2
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib

for a in swinb_s19 swinb_s58 cnxb_s42 cnxb_s19 cnxb_s58; do
  echo "=== $a start $(date -u +%H:%M:%S)"
  python scripts/run_inference_sweep.py \
    scripts/evaluation/vm/configs/hampyeong/inference_sweep_v3_hamp24_${a}.yaml \
    > ~/logs_v3/hamp24_${a}.log 2>&1
  echo "=== $a exit $? $(date -u +%H:%M:%S)"
done
echo "ALL ARMS DONE $(date -u +%H:%M:%S)"
