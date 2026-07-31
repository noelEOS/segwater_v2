#!/bin/bash
# SDS threshold sweep for one Narrabeen arm (best|last), all strides.
# 82-scene scorable set only (see docs/RUNBOOK_sds_vm_eval.md scene coverage).
ARM="${1:?usage: run_sds_narra_s42.sh best|last}"
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim
THRESHOLDS="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
OK=0; FAIL=0
for run_path in ~/segwater_v2/outputs/inference/runs/narrabeen_s42noaug_${ARM}_*; do
  [ -d "$run_path" ] || continue
  name=$(basename "$run_path")
  echo "=== SDS START $name ==="
  if python scripts/sds/run_sds_from_rasters.py \
      --site NARRABEEN --raster-dir "$run_path" \
      --out-dir ~/sds_vm_eval/"${name}_sweep" \
      --thresholds "$THRESHOLDS" \
      --segwater-root ~/segwater_v2 \
      --no-figures > ~/logs_sds_$name.log 2>&1; then
    OK=$((OK+1)); echo "=== SDS DONE $name ==="
  else
    FAIL=$((FAIL+1)); echo "=== SDS FAILED $name (see ~/logs_sds_$name.log) ==="
  fi
done
echo "=== SDS $ARM FINISHED: $OK ok, $FAIL failed ==="
