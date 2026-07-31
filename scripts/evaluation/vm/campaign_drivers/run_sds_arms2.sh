#!/bin/bash
# Narrabeen SDS for the two new mx630 arms (mx630s2 swa5, mx630k best).
# Scene set = every scene in the GT calendar window (87 staged), NOT a max_days filter.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim
THRESHOLDS="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
OK=0; FAIL=0
for run_path in ~/segwater_v2/outputs/inference/runs/narrabeen_mx630s2_s42_swa5_* \
                ~/segwater_v2/outputs/inference/runs/narrabeen_mx630k_s42_best_*; do
  [ -d "$run_path" ] || continue
  name=$(basename "$run_path")
  echo "=== SDS START $name ==="
  if python scripts/sds/run_sds_from_rasters.py \
      --site NARRABEEN --raster-dir "$run_path" \
      --out-dir ~/sds_vm_eval_mx630_arms2/"${name}_sweep" \
      --thresholds "$THRESHOLDS" \
      --segwater-root ~/segwater_v2 \
      --no-figures > ~/logs_sds_arms2_$name.log 2>&1; then
    OK=$((OK+1)); echo "=== SDS DONE $name ==="
  else
    FAIL=$((FAIL+1)); echo "=== SDS FAILED $name (see ~/logs_sds_arms2_$name.log) ==="
    tail -5 ~/logs_sds_arms2_$name.log
  fi
done
echo "=== SDS ARMS2 FINISHED: $OK ok, $FAIL failed ==="
