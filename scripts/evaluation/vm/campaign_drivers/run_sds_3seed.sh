#!/bin/bash
# SDS sweeps for the s19+s58 no-aug/last runs, all 4 canonical sites.
# Site traps: TRUCVERT + TORREYPINES need --no-min-chainage-length; DUCK needs --keep-top-k 999.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim
THR="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
OK=0; FAIL=0
sweep () {  # $1=SITE $2=run_path $3=outroot $4...=flags
  local site="$1" rp="$2" outroot="$3"; shift 3
  local n=$(basename "$rp")
  echo "=== SDS START $n ==="
  if python scripts/sds/run_sds_from_rasters.py --site "$site" --raster-dir "$rp" \
       --out-dir "$outroot/${n}_sweep" --thresholds "$THR" \
       --segwater-root ~/segwater_v2 --no-figures "$@" > ~/logs_sds3s_$n.log 2>&1; then
    OK=$((OK+1)); echo "=== SDS DONE $n ==="
  else
    FAIL=$((FAIL+1)); echo "=== SDS FAILED $n ==="; tail -3 ~/logs_sds3s_$n.log
  fi
}
R=~/segwater_v2/outputs/inference/runs
for rp in $R/narrabeen_s{19,58}noaug_last_*;   do [ -d "$rp" ] && sweep NARRABEEN   "$rp" ~/sds_vm_eval; done
for rp in $R/duck_s{19,58}noaug_last_*;        do [ -d "$rp" ] && sweep DUCK        "$rp" ~/sds_vm_eval_duck --keep-top-k 999; done
for rp in $R/trucvert_s{19,58}noaug_last_*;    do [ -d "$rp" ] && sweep TRUCVERT    "$rp" ~/sds_vm_eval_trucvert --no-min-chainage-length; done
for rp in $R/torreypines_s{19,58}noaug_last_*; do [ -d "$rp" ] && sweep TORREYPINES "$rp" ~/sds_vm_eval_torreypines --no-min-chainage-length; done
echo "=== ALL 3SEED SDS FINISHED: $OK ok, $FAIL failed ==="
