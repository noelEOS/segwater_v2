#!/bin/bash
# SDS sweeps for the re-run Narrabeen (87) + Duck (109) arms.
# Site traps: DUCK needs --keep-top-k 999. NARRABEEN needs nothing extra.
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim
THR="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
OK=0; FAIL=0
run_one () {  # $1=site $2=run_path $3=outroot $4...=extra flags
  local site="$1" rp="$2" outroot="$3"; shift 3
  local n=$(basename "$rp")
  echo "=== SDS START $n ==="
  if python scripts/sds/run_sds_from_rasters.py --site "$site" --raster-dir "$rp" \
       --out-dir "$outroot/${n}_sweep" --thresholds "$THR" \
       --segwater-root ~/segwater_v2 --no-figures "$@" > ~/logs_sdsrerun_$n.log 2>&1; then
    OK=$((OK+1)); echo "=== SDS DONE $n ==="
  else
    FAIL=$((FAIL+1)); echo "=== SDS FAILED $n ==="; tail -4 ~/logs_sdsrerun_$n.log
  fi
}
for rp in ~/segwater_v2/outputs/inference/runs/narrabeen_s42*_*; do
  [ -d "$rp" ] && run_one NARRABEEN "$rp" ~/sds_vm_eval
done
for rp in ~/segwater_v2/outputs/inference/runs/duck_s42*_*; do
  [ -d "$rp" ] && run_one DUCK "$rp" ~/sds_vm_eval_duck --keep-top-k 999
done
echo "=== ALL SDS RERUN FINISHED: $OK ok, $FAIL failed ==="
