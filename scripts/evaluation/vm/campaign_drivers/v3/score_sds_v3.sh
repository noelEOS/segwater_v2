#!/usr/bin/env bash
# Score the 24 v3 SDS run dirs -> ~/sds_vm_eval_v3/<run>_sweep/
#
# Site-required flags come from sds_core.SITE_REQUIRED_FLAGS, which the scorer
# VALIDATES: omit one and the run refuses to start, naming the flag. They are
# set here explicitly rather than relying on that error:
#   TRUCVERT / TORREYPINES  --no-min-chainage-length  (per-transect timestep
#       gate of 30; both sites fall below it on EVERY transect)
#   DUCK                    --keep-top-k 999          (default 1 fails the run)
#   NARRABEEN               none
#
# Thresholds 0.1-0.9 = the canonical sweep grid (0.5 is the operating point).
# Run dirs are resolved by the _<UTC stamp>_ that always follows the sweep name,
# never a bare prefix glob -- a bare glob matches longer siblings.
#
# The scorer exits NON-ZERO if any scene extraction failed, AFTER writing every
# output, so a partial sweep stays inspectable. Failures are tallied, not fatal.
set -u
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/SDS_Benchmark_slim

RUNS=~/segwater_v2/outputs/inference/runs
OUT=~/sds_vm_eval_v3
mkdir -p "$OUT" ~/logs_v3/sds_score

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
    # Guard: only real run dirs carrying a UTC stamp.
    echo "$name" | grep -qE "_[0-9]{8}T[0-9]{6}Z_" || { echo "SKIP (no stamp) $name"; continue; }
    echo "=== $site $name  $(date -u +%H:%M:%S)"
    python scripts/sds/run_sds_from_rasters.py \
      --site "$site" \
      --raster-dir "$run_path" \
      --out-dir "$OUT/${name}_sweep" \
      --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
      --segwater-root ~/segwater_v2 \
      --no-figures $FLAGS \
      > ~/logs_v3/sds_score/${name}.log 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    echo "    exit $rc  $(date -u +%H:%M:%S)"
  done
done
echo "SDS SCORING DONE $(date -u +%H:%M:%S)  ok=$ok fail=$fail"
