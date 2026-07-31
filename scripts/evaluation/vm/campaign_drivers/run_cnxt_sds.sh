#!/bin/bash
# Narrabeen SDS sweeps, ConvNeXtV2-Tiny cnxt campaign. CPU-only; CUDA hidden so
# this can never contend with the GPU inference job. Idempotent.
# Skip logic keyed on the output ARTIFACT (sweep_metrics.csv), gated on input
# COMPLETENESS via completion.py -- never on directory existence.
export CUDA_VISIBLE_DEVICES=""
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
OUT=~/workspace/results/ship_decision_cnxt_2026-07/narrabeen/raw
mkdir -p "$OUT" ~/workspace/logs/cnxt
EXPECT=$(python ~/segwater_v2/scripts/evaluation/vm/completion.py --gate narrabeen --print-expected)
cd ~/SDS_Benchmark_slim
ok=0; skip=0; fail=0; inc=0
for d in ~/segwater_v2/outputs/inference/runs/narrabeen_cnxt_*_b0_s*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  t=$(ls "$d"/*/*_probability_water.tif 2>/dev/null | grep -cv '/\._')
  [ "$t" -eq "$EXPECT" ] || { inc=$((inc+1)); continue; }
  [ -f "$OUT/${n}_sweep/sweep_metrics.csv" ] && { skip=$((skip+1)); continue; }
  if python scripts/sds/run_sds_from_rasters.py --site NARRABEEN --raster-dir "$d" \
       --out-dir "$OUT/${n}_sweep" --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
       --segwater-root ~/segwater_v2 --no-figures > ~/workspace/logs/cnxt/sds_$n.log 2>&1; then
    ok=$((ok+1)); echo "  OK   $n"
  else fail=$((fail+1)); echo "  FAIL $n"; tail -3 ~/workspace/logs/cnxt/sds_$n.log; fi
done
echo "=== SDS pass: ok=$ok already=$skip incomplete=$inc fail=$fail  total=$(ls -d $OUT/*_sweep 2>/dev/null|wc -l)/18 ==="
