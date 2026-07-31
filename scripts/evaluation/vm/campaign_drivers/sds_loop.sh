#!/bin/bash
# Re-run the idempotent SDS pass until all 18 sweeps are scored or inference ends
# AND nothing new appears. One writer only: this is the sole SDS scoring process.
for i in $(seq 1 90); do
  bash ~/workspace/scripts/run_cnxb_sds.sh
  n=$(ls -d ~/workspace/results/ship_decision_cnxb_2026-07/narrabeen/raw/*_sweep 2>/dev/null | wc -l)
  [ "$n" -ge 18 ] && { echo "ALL 18 SDS SWEEPS SCORED"; break; }
  sleep 60
done
