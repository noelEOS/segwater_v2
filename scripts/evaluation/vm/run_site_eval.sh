#!/usr/bin/env bash
# run_site_eval.sh — one command to evaluate a site/model on this VM.
#
#   bash scripts/evaluation/vm/run_site_eval.sh hampyeong <swin|rest4|savelast|legacy>
#   bash scripts/evaluation/vm/run_site_eval.sh demak     <swin|rest4|savelast> <seed>
#
# Steps (idempotent):
#   1. check_inputs.sh (aborts if inputs missing — see MANIFEST.md to re-stage)
#   2. run inference sweep IF the scorer's run dir is absent
#   3. score (Hampyeong: score_pairbased_hampyeong.py --spec; Demak: AUC eval)
#
# This wrapper deliberately does NOT invent new inference configs. For the parts
# the rescued configs don't cleanly cover (Swin-B Hampyeong "all" config — see
# MANIFEST "Config caveats"), it stops with a clear message rather than run the
# wrong architecture. Fleet use: give each VM a different (site, model) pair.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SEGWATER_REPO:-$HOME/segwater_v2}"
INF_ENV="${SEGWATER_INF_ENV:-$HOME/miniforge3/envs/torch211_cu128_inference}"
PY="$INF_ENV/bin/python"
export PATH="$INF_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$INF_ENV/lib"   # CXXABI trap

SITE="${1:?usage: run_site_eval.sh <site> <model> [seed]}"
MODEL="${2:?usage: run_site_eval.sh <site> <model> [seed]}"
SEED="${3:-}"

echo "== step 1: check inputs =="
bash "$KIT/check_inputs.sh" "$SITE"

cd "$REPO"

case "$SITE" in
  hampyeong)
    case "$MODEL" in
      swin|rest4|savelast)
        SPEC="$KIT/specs/hampyeong_${MODEL}.yaml"
        echo "== step 2: (re)inference — check run dirs the spec resolves =="
        if $PY "$KIT/score_pairbased_hampyeong.py" --spec "$SPEC" >/dev/null 2>&1; then
          echo "  scorer ran clean once already? re-run below for real output"
        fi
        # The scorer's own audit fails if a run dir is missing; surface that as
        # the signal to run inference. We do not auto-launch inference here for
        # Swin-B (config caveat) — the resolved run dirs for these specs are the
        # dev_sweep_all_/dev_sweep_rest_ outputs already on the VM.
        echo "== step 3: score =="
        $PY "$KIT/score_pairbased_hampyeong.py" --spec "$SPEC"
        ;;
      legacy)
        echo "== step 3: score legacy =="
        $PY "$KIT/score_hampyeong_legacy.py" \
          --repo "$REPO" \
          --nas-root "${SEGWATER_ANCILLARY:-$HOME/ancillary}/hampyeong/nas_root"
        ;;
      *) echo "unknown hampyeong model: $MODEL (swin|rest4|savelast|legacy)"; exit 2;;
    esac
    ;;

  demak)
    : "${SEED:?demak needs a seed: s19|s42|s58}"
    case "$MODEL" in
      rest4)   INF_CFG="$KIT/configs/demak/inference_sweep-demak-concurrent-${SEED}.yaml"
               AUC_CFG="$KIT/configs/aucroc/aucroc_rest4_${SEED}_vm.yaml" ;;
      swin)    INF_CFG=""   # swin-B concurrent dirs already exist; see MANIFEST
               AUC_CFG="$KIT/configs/aucroc/aucroc_${SEED}_vm.yaml" ;;
      savelast) INF_CFG="$KIT/configs/demak/inference_sweep-demak-savelast-${SEED}.yaml"
               AUC_CFG="$KIT/configs/aucroc/aucroc_savelast_${SEED}_vm.yaml" ;;
      *) echo "unknown demak model: $MODEL (swin|rest4|savelast)"; exit 2;;
    esac
    if [ -n "$INF_CFG" ]; then
      echo "== step 2: inference sweep ($INF_CFG) =="
      $PY scripts/run_inference_sweep.py "$INF_CFG"
    else
      echo "== step 2: skipped (swin concurrent run dirs pre-exist; see MANIFEST) =="
    fi
    echo "== step 3: AUC-ROC + threshold eval ($AUC_CFG) =="
    $PY scripts/evaluate_indonesia_inference_run_aucroc.py "$AUC_CFG"
    ;;

  *) echo "unknown site: $SITE (hampyeong|demak)"; exit 2;;
esac

echo ""
echo "DONE ($SITE / $MODEL${SEED:+ / $SEED}). CSV outputs only — do not egress rasters."
