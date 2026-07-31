#!/bin/bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim
THR="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
OK=0; FAIL=0
for rp in ~/segwater_v2/outputs/inference/runs/duck_s19noaug_last_* ~/segwater_v2/outputs/inference/runs/duck_s58noaug_last_*; do
  [ -d "$rp" ] || continue
  n=$(basename "$rp"); echo "=== SDS START $n ==="
  if python scripts/sds/run_sds_from_rasters.py --site DUCK --raster-dir "$rp" \
       --out-dir ~/sds_vm_eval_duck/"${n}_sweep" --thresholds "$THR" --keep-top-k 999 \
       --segwater-root ~/segwater_v2 --no-figures > ~/logs_sds3s_$n.log 2>&1; then
    OK=$((OK+1)); echo "=== SDS DONE $n ==="
  else FAIL=$((FAIL+1)); echo "=== SDS FAILED $n ==="; tail -3 ~/logs_sds3s_$n.log; fi
done
echo "=== DUCK 2SEED SDS FINISHED: $OK ok, $FAIL failed ==="
