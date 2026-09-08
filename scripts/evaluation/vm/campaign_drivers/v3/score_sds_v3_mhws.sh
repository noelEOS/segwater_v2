#!/usr/bin/env bash
# Score the 24 v3 SDS run dirs against MHWS -> ~/sds_vm_eval_v3_mhws/
#
# --reference MHWS moves BOTH the tidal-correction reference contour and the
# groundtruth contour together. It is therefore a DIFFERENT ESTIMAND, not a
# re-check of MSL: never pool or average MHWS and MSL numbers.
#
# --submission-type is deliberately NOT set: on an MHWS run the scorer relabels
# it automatically to the MHWS folder name, so setting it by hand mislabels
# provenance in metrics.{csv,json}.
#
# Site-required flags are orthogonal to the reference and still apply.
set -u
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim

RUNS=~/segwater_v2/outputs/inference/runs
OUT=~/sds_vm_eval_v3_mhws
mkdir -p "$OUT" ~/logs_v3/sds_score_mhws

ok=0; fail=0
for site in NARRABEEN DUCK TORREYPINES TRUCVERT; do
  case "$site" in
    TRUCVERT|TORREYPINES) FLAGS="--no-min-chainage-length" ;;
    DUCK)                 FLAGS="--keep-top-k 999" ;;
    *)                    FLAGS="" ;;
  esac
  low=$(echo "$site" | tr "[:upper:]" "[:lower:]")
  for run_path in "$RUNS"/v3_sds_${low}_*; do
    name=$(basename "$run_path")
    echo "$name" | grep -qE "_[0-9]{8}T[0-9]{6}Z_" || { echo "SKIP $name"; continue; }
    echo "=== $site $name  $(date -u +%H:%M:%S)"
    python scripts/sds/run_sds_from_rasters.py \
      --site "$site" \
      --raster-dir "$run_path" \
      --out-dir "$OUT/${name}_sweep" \
      --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
      --segwater-root ~/segwater_v2 \
      --reference MHWS \
      --no-figures $FLAGS \
      > ~/logs_v3/sds_score_mhws/${name}.log 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    echo "    exit $rc  $(date -u +%H:%M:%S)"
  done
done
echo "MHWS SCORING DONE $(date -u +%H:%M:%S)  ok=$ok fail=$fail"
