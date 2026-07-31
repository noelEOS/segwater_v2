#!/bin/bash
# Ship-campaign closeout. Idempotent; safe to re-run.
set -e
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
S=~/workspace/results/ship_decision_2026-07
T=$S/demak_trend
echo "=== 1. s32 areas ==="
python scripts/evaluation/vm/ship/build_ship_areas.py --stride 32
echo "=== 2. s32 trend ==="
python scripts/evaluation/vm/ship/fit_ship_trend.py --stride 32 $T/demak_full_ship_areas_s32.csv
echo "=== 3. S2-matched, BOTH strides (12 rows) ==="
python scripts/evaluation/vm/ship/fit_ship_s2matched.py \
  --area-csv $T/demak_full_ship_areas_s112.csv --stride 112 \
  --area-csv $T/demak_full_ship_areas_s32.csv  --stride 32
echo "=== 4. provenance manifest (all 42 dirs) ==="
python scripts/evaluation/vm/ship/build_ship_manifest.py
echo "=== 5. consolidate ==="
python scripts/evaluation/vm/ship/consolidate_ship_results.py
echo "=== FINALIZE COMPLETE ==="
