#!/bin/bash
# Narrabeen SDS for the swa5 arm. CPU-only; CUDA hidden so it cannot contend
# with the GPU inference job. Idempotent: skips sweeps that already have
# sweep_metrics.csv, so it is safe to re-run as dirs land.
export CUDA_VISIBLE_DEVICES=""
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
OUT=~/workspace/results/ship_decision_2026-07/narrabeen/raw
mkdir -p "$OUT"; cd ~/SDS_Benchmark_slim
ok=0; skip=0; fail=0
for d in ~/segwater_v2/outputs/inference/runs/narrabeen_ship_*_swa5_*_b0_s*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  t=$(ls "$d"/*/*_probability_water.tif 2>/dev/null | wc -l)
  [ "$t" -eq 87 ] || { echo "  skip(incomplete $t/87) ${n:0:52}"; continue; }
  [ -f "$OUT/${n}_sweep/sweep_metrics.csv" ] && { skip=$((skip+1)); continue; }
  rm -rf "$OUT/${n}_sweep"
  if python scripts/sds/run_sds_from_rasters.py --site NARRABEEN --raster-dir "$d" \
       --out-dir "$OUT/${n}_sweep" --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
       --segwater-root ~/segwater_v2 --no-figures > ~/workspace/logs/ship/sds_$n.log 2>&1; then
    ok=$((ok+1)); echo "  OK   ${n:0:60}"
  else fail=$((fail+1)); echo "  FAIL ${n:0:60}"; tail -3 ~/workspace/logs/ship/sds_$n.log; fi
done
echo "=== swa5 SDS pass: ok=$ok already=$skip fail=$fail  total=$(ls -d $OUT/*swa5*_sweep 2>/dev/null|wc -l)/9 ==="
