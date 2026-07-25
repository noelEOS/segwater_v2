#!/usr/bin/env bash
# check_inputs.sh — verify a VM has everything needed for Demak/Hampyeong eval.
# Read-only. Prints one PASS/MISS line per required input; exits non-zero if any
# input is missing. Run this first on a fresh VM (the "is this VM ready?" check).
#
#   bash scripts/evaluation/vm/check_inputs.sh [demak|hampyeong|all|sds]
#
# `sds` is opt-in and NOT covered by `all`: it is a separate job using the
# SDS_Benchmark_slim tree (see docs/RUNBOOK_sds_vm_eval.md).
#
# Re-stage sources for anything MISSing are in MANIFEST.md.
set -u

SITE="${1:-all}"
REPO="${SEGWATER_REPO:-$HOME/segwater_v2}"
ANC="${SEGWATER_ANCILLARY:-$HOME/ancillary}"
INF_ENV="${SEGWATER_INF_ENV:-$HOME/miniforge3/envs/torch211_cu128_inference}"
DEMAK_DATA="${SEGWATER_DEMAK_DATA:-$HOME/data_demak_concurrent}"
SDS_ROOT="${SEGWATER_SDS_ROOT:-$HOME/SDS_Benchmark_slim}"

fail=0
chk_file() { if [ -f "$1" ]; then echo "  PASS  $2"; else echo "  MISS  $2 -> $1"; fail=1; fi; }
chk_dir()  { if [ -d "$1" ]; then echo "  PASS  $2"; else echo "  MISS  $2 -> $1"; fail=1; fi; }
chk_count() { # dir glob expected label
  n=$(ls -1 $1/$2 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" = "$3" ]; then echo "  PASS  $4 ($n)"; else echo "  MISS  $4: found $n, expect $3 in $1/$2"; fail=1; fi
}

echo "== common =="
chk_dir  "$REPO" "repo checkout"
chk_file "$INF_ENV/bin/python" "inference env python"

if [ "$SITE" = "demak" ] || [ "$SITE" = "all" ]; then
  echo "== demak =="
  chk_count "$DEMAK_DATA" "S1_*.tif" 6 "6 concurrent S1 scenes"
  chk_count "$ANC/demak_semarang/reference_s2" "*.tif" 6 "6 S2 reference rasters"
  chk_file  "$ANC/demak_semarang/valid_mask/GSHHG_GlobalSurfaceWater_combined_mask.tif" "Demak valid mask"
fi

if [ "$SITE" = "hampyeong" ] || [ "$SITE" = "all" ]; then
  echo "== hampyeong =="
  chk_count "$ANC/hampyeong/nas_root/sen12coast/Validation/Hampyeong/DEM_FLOOD_MASKS/Descending_reproj" "*.tif" 5 "5 DEM flood masks"
  chk_file  "$ANC/hampyeong/nas_root/Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM/DEM_VALID_MASK_aoi.tif" "tide-gauge DEM valid mask"
  chk_file  "$ANC/hampyeong/val_split_calibration/val_thresholds.csv" "val thresholds"
  chk_file  "$ANC/hampyeong/val_split_calibration/platt_params.csv" "platt params"
fi

# SDS is a separate job with its own tree; not included in "all" because most
# eval work does not need it. See docs/RUNBOOK_sds_vm_eval.md.
if [ "$SITE" = "sds" ]; then
  echo "== sds =="
  chk_dir  "$SDS_ROOT" "SDS_Benchmark_slim tree"
  chk_file "$SDS_ROOT/scripts/sds/sds_core.py" "sds_core.py"
  chk_file "$SDS_ROOT/datasets/sites_info.txt" "sites_info.txt"
  for s in NARRABEEN DUCK TORREYPINES TRUCVERT; do
    chk_dir "$SDS_ROOT/datasets/$s" "dataset $s"
  done
  # osgeo is conda-only and the usual missing piece on a fresh VM.
  if "$INF_ENV/bin/python" -c "import osgeo, pytz, astropy" >/dev/null 2>&1; then
    echo "  PASS  SDS python deps (osgeo/pytz/astropy)"
  else
    echo "  MISS  SDS python deps -> see RUNBOOK_sds_vm_eval.md Step 0b"; fail=1
  fi
fi

echo ""
if [ "$fail" = "0" ]; then echo "READY ($SITE)"; else echo "NOT READY ($SITE) — see MANIFEST.md for re-stage sources"; fi
exit $fail
