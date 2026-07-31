#!/bin/bash
# MHWS + RAW sweeps over the SAME no-aug/last rasters already on disk (no inference).
#   MHWS: --reference MHWS  (do NOT set --submission-type; it is relabelled automatically)
#   RAW : --submission-type raw_timeseries  (--reference stays at its MSL default,
#         which controls the GROUNDTRUTH contour -> groundtruth_MSL, as registered)
# Site traps carried over: DUCK --keep-top-k 999; TRUCVERT + TORREYPINES --no-min-chainage-length.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim
THR="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
OK=0; FAIL=0
R=~/segwater_v2/outputs/inference/runs

sweep () {  # $1=SITE $2=run_path $3=outroot $4=tag  $5...=flags
  local site="$1" rp="$2" outroot="$3" tag="$4"; shift 4
  local n=$(basename "$rp")
  echo "=== SDS START [$tag] $n ==="
  if python scripts/sds/run_sds_from_rasters.py --site "$site" --raster-dir "$rp" \
       --out-dir "$outroot/${n}_sweep" --thresholds "$THR" \
       --segwater-root ~/segwater_v2 --no-figures "$@" > ~/logs_sds_${tag}_$n.log 2>&1; then
    OK=$((OK+1)); echo "=== SDS DONE [$tag] $n ==="
  else
    FAIL=$((FAIL+1)); echo "=== SDS FAILED [$tag] $n ==="; tail -3 ~/logs_sds_${tag}_$n.log
  fi
}

for ref in mhws raw; do
  if [ "$ref" = "mhws" ]; then REFFLAGS=(--reference MHWS); else REFFLAGS=(--submission-type raw_timeseries); fi
  for rp in $R/narrabeen_s*noaug_last_*;   do [ -d "$rp" ] && sweep NARRABEEN   "$rp" ~/sds_vm_eval_$ref        "$ref" "${REFFLAGS[@]}"; done
  for rp in $R/duck_s*noaug_last_*;        do [ -d "$rp" ] && sweep DUCK        "$rp" ~/sds_vm_eval_duck_$ref   "$ref" "${REFFLAGS[@]}" --keep-top-k 999; done
  for rp in $R/trucvert_s*noaug_last_*;    do [ -d "$rp" ] && sweep TRUCVERT    "$rp" ~/sds_vm_eval_tv_$ref     "$ref" "${REFFLAGS[@]}" --no-min-chainage-length; done
  for rp in $R/torreypines_s*noaug_last_*; do [ -d "$rp" ] && sweep TORREYPINES "$rp" ~/sds_vm_eval_tp_$ref     "$ref" "${REFFLAGS[@]}" --no-min-chainage-length; done
done
echo "=== ALL MHWS+RAW SDS FINISHED: $OK ok, $FAIL failed ==="
