#!/usr/bin/env bash
# check_inputs.sh — verify a VM has everything needed for Demak/Hampyeong eval.
# Read-only. Prints one PASS/MISS line per required input; exits non-zero if any
# input is missing. Run this first on a fresh VM (the "is this VM ready?" check).
#
#   bash scripts/evaluation/vm/check_inputs.sh [demak|hampyeong|narrabeen|demak_full|all|sds]
#
# `sds` is opt-in and NOT covered by `all`: it is a separate job using the
# SDS_Benchmark_slim tree (see docs/RUNBOOK_sds_vm_eval.md).
#
# Site data dirs are env-overridable (SEGWATER_*_DATA). A site whose dir is
# absent AND whose env var is unset is SKIPped, not MISSed, so `all` still
# reports READY on a VM that was never staged with that site's scenes.
#
# Re-stage sources for anything MISSing are in MANIFEST.md.
set -u

SITE="${1:-all}"
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SEGWATER_REPO:-$HOME/segwater_v2}"
ANC="${SEGWATER_ANCILLARY:-$HOME/ancillary}"
INF_ENV="${SEGWATER_INF_ENV:-$HOME/miniforge3/envs/torch211_cu128_inference}"
DEMAK_DATA="${SEGWATER_DEMAK_DATA:-$HOME/data_demak_concurrent}"
SDS_ROOT="${SEGWATER_SDS_ROOT:-$HOME/SDS_Benchmark_slim}"

fail=0
chk_file() { if [ -f "$1" ]; then echo "  PASS  $2"; else echo "  MISS  $2 -> $1"; fail=1; fi; }
chk_dir()  { if [ -d "$1" ]; then echo "  PASS  $2"; else echo "  MISS  $2 -> $1"; fail=1; fi; }
chk_count() { # dir glob expected label
  # `find` (not `ls $1/$2`): the old unquoted glob broke on spaces and counted
  # AppleDouble `._*` stubs, which rsync-from-macOS plants and which match any
  # suffix glob. -type f so a same-named directory cannot inflate the count.
  n=$(find "$1" -maxdepth 1 -name "$2" ! -name '._*' -type f 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" = "$3" ]; then echo "  PASS  $4 ($n)"; else echo "  MISS  $4: found $n, expect $3 in $1/$2"; fail=1; fi
}
chk_import() { # label python-expression miss-hint
  # SUBSHELL + `cd /tmp`: the cwd must NOT leak back to the caller, and the
  # import must be tested from OUTSIDE the repo — importing `src` from inside
  # the checkout succeeds via cwd even when the editable install is absent,
  # which is precisely the exit-0-do-nothing failure this check exists for.
  if ( cd /tmp && "$INF_ENV/bin/python" -c "$2" ) >/dev/null 2>&1; then
    echo "  PASS  $1"
  else
    echo "  MISS  $1 -> $3"; fail=1
  fi
}
exp_count() { # gate literal-fallback
  # Expected counts come from completion.py (single source of truth). Fall back
  # to the literal when that call fails: a broken env is exactly when this
  # checker is run, and it must still report the data-staging results.
  local n
  n=$("$INF_ENV/bin/python" "$KIT_DIR/completion.py" --gate "$1" --print-expected 2>/dev/null)
  case "$n" in
    ''|*[!0-9]*) echo "$2" ;;
    *) echo "$n" ;;
  esac
}

echo "== common =="
chk_dir  "$REPO" "repo checkout"
chk_file "$INF_ENV/bin/python" "inference env python"
chk_import "editable install (import src from /tmp)" \
  "from src.utils.vectorizer import ShorelineVectorizer" \
  "pip install -e \$REPO in $INF_ENV — see MANIFEST.md, editable-install section"
chk_import "opencv (import cv2)" "import cv2" \
  "pip install opencv-python-headless in $INF_ENV — see MANIFEST.md"

if [ "$SITE" = "demak" ] || [ "$SITE" = "all" ]; then
  echo "== demak =="
  chk_count "$DEMAK_DATA" "S1_*.tif" "$(exp_count demak_gate 6)" "concurrent S1 scenes"
  chk_count "$ANC/demak_semarang/reference_s2" "*.tif" 6 "6 S2 reference rasters"
  chk_file  "$ANC/demak_semarang/valid_mask/GSHHG_GlobalSurfaceWater_combined_mask.tif" "Demak valid mask"
  # sklearn is the AUC scorer's dependency (score_demak_gate_aucroc.py).
  chk_import "sklearn (Demak AUC scorer)" "import sklearn" \
    "pip install scikit-learn in $INF_ENV — see MANIFEST.md"
fi

if [ "$SITE" = "hampyeong" ] || [ "$SITE" = "all" ]; then
  echo "== hampyeong =="
  chk_count "$ANC/hampyeong/nas_root/sen12coast/Validation/Hampyeong/DEM_FLOOD_MASKS/Descending_reproj" "*.tif" 5 "5 DEM flood masks"
  chk_file  "$ANC/hampyeong/nas_root/Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM/DEM_VALID_MASK_aoi.tif" "tide-gauge DEM valid mask"
  chk_file  "$ANC/hampyeong/val_split_calibration/val_thresholds.csv" "val thresholds"
  chk_file  "$ANC/hampyeong/val_split_calibration/platt_params.csv" "platt params"
fi

if [ "$SITE" = "narrabeen" ] || [ "$SITE" = "all" ]; then
  echo "== narrabeen =="
  NARRABEEN_DATA="${SEGWATER_NARRABEEN_DATA:-$HOME/NARRABEEN_ron_147_ts_9_sn_16}"
  if [ -d "$NARRABEEN_DATA" ]; then
    chk_count "$NARRABEEN_DATA" "*.tif" "$(exp_count narrabeen 87)" "Narrabeen S1 scenes"
  elif [ -n "${SEGWATER_NARRABEEN_DATA:-}" ]; then
    chk_dir "$NARRABEEN_DATA" "Narrabeen scene dir"
  else
    echo "  SKIP  narrabeen (dir absent; set SEGWATER_NARRABEEN_DATA to check)"
  fi
fi

if [ "$SITE" = "demak_full" ] || [ "$SITE" = "all" ]; then
  echo "== demak_full =="
  DEMAK_FULL_DATA="${SEGWATER_DEMAK_FULL_DATA:-$HOME/data_demak}"
  if [ -d "$DEMAK_FULL_DATA" ]; then
    chk_count "$DEMAK_FULL_DATA" "S1_*.tif" "$(exp_count demak_full 213)" "Demak full-series S1 scenes"
  elif [ -n "${SEGWATER_DEMAK_FULL_DATA:-}" ]; then
    chk_dir "$DEMAK_FULL_DATA" "Demak full-series scene dir"
  else
    echo "  SKIP  demak_full (dir absent; set SEGWATER_DEMAK_FULL_DATA to check)"
  fi
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
