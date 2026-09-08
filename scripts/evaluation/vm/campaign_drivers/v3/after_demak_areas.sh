#!/usr/bin/env bash
# Wait for the Demak full-series inference chain, then build the water-area
# time series. Waits on the chain LOG sentinel, not a process pattern.
set -u
until grep -q "DEMAK FULL DONE" ~/logs_v3/demak_full_chain.log 2>/dev/null; do sleep 60; done
echo "inference: $(tail -1 ~/logs_v3/demak_full_chain.log)"

cd ~/segwater_v2
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib

# Completeness first: a short run dir would yield a plausible trend on the
# wrong sample. The builder also asserts 213/206, but fail early and loudly.
python scripts/evaluation/vm/completion.py --gate demak_full \
  --check outputs/inference/runs/v3_demak_full_* || { echo "ABORT: short run dir"; exit 1; }

python scripts/evaluation/vm/analysis/build_v3_demak_areas.py
echo "AREAS DONE $(date -u +%H:%M:%S)"
