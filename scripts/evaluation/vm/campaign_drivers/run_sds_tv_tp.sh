#!/bin/bash
# SDS sweeps for Trucvert + Torreypines.
# ⚠️ TRUCVERT REQUIRES --no-min-chainage-length, else the transect set comes back EMPTY.
#    A ~-61 m bias at Trucvert is normal, not a bug.
# TORREYPINES needs no extra flag.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim
THR="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
OK=0; FAIL=0
sweep () {  # $1=SITE $2=run_path $3=outroot  $4...=extra flags
  local site="$1" rp="$2" outroot="$3"; shift 3
  local n=$(basename "$rp")
  echo "=== SDS START $n ==="
  if python scripts/sds/run_sds_from_rasters.py --site "$site" --raster-dir "$rp" \
       --out-dir "$outroot/${n}_sweep" --thresholds "$THR" \
       --segwater-root ~/segwater_v2 --no-figures "$@" > ~/logs_sdstvtp_$n.log 2>&1; then
    OK=$((OK+1)); echo "=== SDS DONE $n ==="
  else
    FAIL=$((FAIL+1)); echo "=== SDS FAILED $n ==="; tail -4 ~/logs_sdstvtp_$n.log
  fi
}
for rp in ~/segwater_v2/outputs/inference/runs/trucvert_s42*_*; do
  [ -d "$rp" ] && sweep TRUCVERT "$rp" ~/sds_vm_eval_trucvert --no-min-chainage-length
done
for rp in ~/segwater_v2/outputs/inference/runs/torreypines_s42*_*; do
  [ -d "$rp" ] && sweep TORREYPINES "$rp" ~/sds_vm_eval_torreypines --no-min-chainage-length
done
echo "=== ALL TV_TP SDS FINISHED: $OK ok, $FAIL failed ==="
