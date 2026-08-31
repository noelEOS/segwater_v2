#!/usr/bin/env bash
# Run ONE shard of the hamp24 stride-8 sweep on this VM, then score it.
#
# The stride-8 family is 6 sweep configs (3 seeds x {main, fillin}) and takes
# ~7.4 h per main config on a single g4-standard-48. This script lets each
# config run on its own machine, so the family finishes in the time of its
# slowest shard instead of the sum.
#
# NOTHING about the computation changes: each shard runs the same config file,
# the same checkpoints and the same runner it would have run locally, so the
# outputs are identical to the single-machine ordering. Only the placement
# differs. Scoring is deliberately NOT done per shard -- it needs all 21 arms
# present, so it runs once on the collector node after the run dirs are
# gathered (see --collect below).
#
# Usage, on each worker VM:
#     run_hamp24_stride_shard.sh <stride> <seed> [main|fillin]
#   e.g.
#     run_hamp24_stride_shard.sh 8 19 main
#     run_hamp24_stride_shard.sh 8 19 fillin
#
# Preflight only (verifies staging without running anything):
#     run_hamp24_stride_shard.sh --check <stride> <seed> [main|fillin]
#
# What must be staged on the worker (see scripts/evaluation/vm/MANIFEST.md):
#   ~/segwater_v2                     repo checkout (a plain checkout is enough:
#                                     the runner is invoked as
#                                     scripts/run_inference_sweep.py with the
#                                     repo as cwd, so no pip install is needed
#                                     for inference -- the campaign VM has none)
#   ~/miniforge3/envs/torch211_cu128_inference
#   ~/hampyeong_ron_134_ts_16_sn_15_all24/   24 scenes + pairing_metadata.csv
#   outputs/stage2/<arch>/s<seed>/best.pth   ONLY the seed this shard runs;
#                                            copy with `cp -L` / `rsync -L` so
#                                            the symlink is resolved.
# Ancillary GT/masks are NOT needed on a worker -- only the collector scores.
set -uo pipefail

CHECK_ONLY=0
if [ "${1:-}" = "--check" ]; then CHECK_ONLY=1; shift; fi

STRIDE="${1:?usage: run_hamp24_stride_shard.sh [--check] <stride> <seed> [main|fillin]}"
SEED="${2:?usage: run_hamp24_stride_shard.sh [--check] <stride> <seed> [main|fillin]}"
KIND="${3:-main}"

REPO="${SEGWATER_REPO:-$HOME/segwater_v2}"
ENV_DIR="${SEGWATER_INF_ENV:-$HOME/miniforge3/envs/torch211_cu128_inference}"
SCENES="${SEGWATER_HAMP_SCENES:-$HOME/hampyeong_ron_134_ts_16_sn_15_all24}"

case "$KIND" in
  main)   SUFFIX="" ;;
  fillin) SUFFIX="-fillin" ;;
  *) echo "third argument must be 'main' or 'fillin', got '$KIND'"; exit 2 ;;
esac

CFG="$REPO/scripts/evaluation/vm/configs/hampyeong24/inference_sweep_hamp24-str${STRIDE}-s${SEED}${SUFFIX}.yaml"
LOG="$HOME/hamp24_str${STRIDE}_s${SEED}${SUFFIX}.log"
TAG="stride=${STRIDE} seed=s${SEED} ${KIND}"

fail() { echo "PREFLIGHT FAIL ($TAG): $*"; exit 1; }

# ---- preflight: everything this shard needs, checked before the GPU warms ---
[ -d "$REPO" ]     || fail "no repo at $REPO"
[ -d "$ENV_DIR" ]  || fail "no inference env at $ENV_DIR"
[ -f "$CFG" ]      || fail "no config at $CFG"
[ -d "$SCENES" ]   || fail "no scenes at $SCENES"

N_SCENES=$(ls "$SCENES"/S1B_*.tif 2>/dev/null | wc -l)
[ "$N_SCENES" -eq 24 ] || fail "expected 24 scenes in $SCENES, found $N_SCENES"

export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib"     # CXXABI trap: required, see MANIFEST
cd "$REPO" || fail "cannot cd $REPO"

# The config names its checkpoints; verify each one is present AND is a real
# file with weights, not a dangling symlink copied without -L.
MISSING=0
while read -r ck; do
  [ -z "$ck" ] && continue
  if [ ! -e "$REPO/$ck" ]; then
    echo "  MISSING ckpt: $ck"; MISSING=1; continue
  fi
  sz=$(stat -Lc %s "$REPO/$ck" 2>/dev/null || echo 0)
  if [ "$sz" -lt 1000000 ]; then
    echo "  BAD ckpt (${sz} B -- dangling symlink? copy with cp -L): $ck"; MISSING=1
  fi
done < <(grep -oE 'outputs/stage2/[^"]+\.pth' "$CFG" | sort -u)
[ "$MISSING" -eq 0 ] || fail "checkpoints missing or unresolved (see above)"

# Import what the runner actually needs. The sweep invokes
# scripts/run_inference.py with the repo as cwd, so the repo's own modules are
# imported by path -- there is deliberately NO `import segwater` here, because
# the package is not installed on the campaign VM either and requiring it would
# reject a correctly-staged worker.
python -c "import torch, rasterio, omegaconf, yaml" 2>/dev/null \
  || fail "env broken: torch/rasterio/omegaconf/yaml not importable from $ENV_DIR"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'" 2>/dev/null \
  || fail "torch cannot see a CUDA device"
[ -f "$REPO/scripts/run_inference_sweep.py" ] || fail "no runner at $REPO/scripts/run_inference_sweep.py"

N_CKPT=$(grep -cE 'checkpoint_path:' "$CFG")
echo "PREFLIGHT OK ($TAG): $N_CKPT arms x $N_SCENES scenes"
echo "  config: $CFG"
echo "  gpu:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"

if [ "$CHECK_ONLY" -eq 1 ]; then echo "(--check: not running)"; exit 0; fi

# ---- run ------------------------------------------------------------------
echo "=== $(date -u +%FT%TZ) START $TAG"
python scripts/run_inference_sweep.py "$CFG" > "$LOG" 2>&1
rc=$?
echo "=== $(date -u +%FT%TZ) DONE  $TAG rc=$rc"
tail -3 "$LOG"

if [ $rc -ne 0 ]; then
  echo "SHARD FAILED ($TAG) -- see $LOG"
  exit $rc
fi

# Marker names the shard, so the collector can tell which are finished.
touch "$HOME/hamp24_str${STRIDE}_s${SEED}${SUFFIX}_done.marker"
N_TIF=$(ls "$REPO"/outputs/inference/dev_sweep_hamp24_str${STRIDE}_s${SEED}_*/*/*probability_water.tif 2>/dev/null | wc -l)
echo "SHARD COMPLETE ($TAG): $N_TIF probability rasters"
echo
echo "Next: copy the run dirs to the collector, e.g. from the collector run"
echo "  rsync -av --include='dev_sweep_hamp24_str${STRIDE}_s${SEED}_*/***' --exclude='*' \\"
echo "    <this-vm>:~/segwater_v2/outputs/inference/ ~/segwater_v2/outputs/inference/"
echo "Then, once ALL 21 arms are present on the collector:"
echo "  ~/score_stride.sh ${STRIDE}"
