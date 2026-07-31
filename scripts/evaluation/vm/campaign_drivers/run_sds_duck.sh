#!/bin/bash
# Duck SDS threshold sweep, all 4 arms x 3 strides.
# TRAP: DUCK requires --keep-top-k 999 (default 1 fails the whole run).
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim
THRESHOLDS="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
OK=0; FAIL=0
for run_path in ~/segwater_v2/outputs/inference/runs/duck_s42*_*; do
  [ -d "$run_path" ] || continue
  n=$(basename "$run_path")
  echo "=== SDS START $n ==="
  if python scripts/sds/run_sds_from_rasters.py \
      --site DUCK --raster-dir "$run_path" \
      --out-dir ~/sds_vm_eval_duck/"${n}_sweep" \
      --thresholds "$THRESHOLDS" \
      --keep-top-k 999 \
      --segwater-root ~/segwater_v2 \
      --no-figures > ~/logs_sds_$n.log 2>&1; then
    OK=$((OK+1)); echo "=== SDS DONE $n ==="
  else
    FAIL=$((FAIL+1)); echo "=== SDS FAILED $n ==="; tail -4 ~/logs_sds_$n.log
  fi
done
echo "=== ALL DUCK SDS FINISHED: $OK ok, $FAIL failed ==="
