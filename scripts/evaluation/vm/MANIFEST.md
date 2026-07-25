# VM eval kit — manifest & runbook

One home for the Demak + Hampyeong VM-side evaluation. Everything here was
previously loose in a single VM's `~/` (the only copy); it is now tracked so it
survives VM cleanups and works on a fresh VM or a fleet.

Goal: an agent bringing up a VM to evaluate one site/model should only need to
(1) verify/scp inputs, (2) run inference if the run dir is absent, (3) score.

## Contents

| File | What |
|---|---|
| `score_pairbased_hampyeong.py` | ONE config-driven Hampyeong scorer, `--spec <yaml>`. Replaces the 3 hand-forked VM scorers. |
| `specs/hampyeong_{swin,rest4,savelast}.yaml` | the three scoring targets as data. Proven byte-identical to the original forks (2026-07-25). |
| `specs/hampyeong_gate.yaml` | 9-entry spec (3 seeds × best/last/SWA-5) for the 2026-07-25 checkpoint-selection gate. |
| `score_hampyeong_legacy.py` | separate job: scores the 2 legacy Sen1Coast rasters + Table-2 repro check. |
| `configs/demak/*.yaml` | `run_inference_sweep.py` configs (Demak, per seed). |
| `configs/hampyeong/*.yaml` | Hampyeong inference sweep configs. ⚠️ see "Config caveats". |
| `configs/aucroc/*.yaml` | Demak AUC-ROC + threshold-sweep eval configs (per variant/seed). |
| `gen_rest4_configs.py` | generator that emitted the rest4 sweep configs. |
| `gate/` | checkpoint-selection-gate config generators (Demak + Hampyeong, one per arm×seed) — see `gate/README.md`. |
| `check_inputs.sh` | read-only readiness check; PASS/MISS per required input. |
| `run_site_eval.sh` | 3-step entry point (check → infer-if-absent → score). |

## Required inputs on the VM (and where to re-stage from)

`check_inputs.sh [demak|hampyeong|all]` verifies all of these. Override default
locations with env vars: `SEGWATER_REPO`, `SEGWATER_ANCILLARY`,
`SEGWATER_INF_ENV`, `SEGWATER_DEMAK_DATA`.

| Input | Default VM path | Re-stage source (ingress is free) |
|---|---|---|
| repo checkout | `~/segwater_v2` | git |
| inference env | `~/miniforge3/envs/torch211_cu128_inference` | conda env (see env note) |
| **Demak** 6 concurrent S1 | `~/data_demak_concurrent/S1_*.tif` | NAS / Mac (ask user) |
| Demak 6 S2 reference | `~/ancillary/demak_semarang/reference_s2/*.tif` | Mac `…/Concurrent_S2_ndwi_mndwi_awei_ndvi/*.tif` (1.9 MB) |
| Demak valid mask | `~/ancillary/demak_semarang/valid_mask/GSHHG_GlobalSurfaceWater_combined_mask.tif` | Mac `…/S1_S2_Accuracy_Assessment/…` (244 KB) |
| **Hampyeong** DEM flood masks (5) | `~/ancillary/hampyeong/nas_root/sen12coast/Validation/Hampyeong/DEM_FLOOD_MASKS/Descending_reproj/` | NAS mirror |
| Hampyeong tide-gauge DEM mask | `~/ancillary/hampyeong/nas_root/Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM/DEM_VALID_MASK_aoi.tif` | NAS mirror |
| Hampyeong val calibration | `~/ancillary/hampyeong/val_split_calibration/{val_thresholds,platt_params}.csv` | Mac |

## Environment note (CXXABI trap)

Conda activation does not work in non-interactive SSH shells. Always use the
absolute env python, and export the lib path or model import dies on CXXABI:

```bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
```

`torch211_cu128_inference` may be missing `scikit-learn` after a VM cleanup
(`pip install scikit-learn` into that env) — needed by the Demak AUC scorer.

## HARD RULE — no heavy-TIFF egress

Score VM-side; only the KB-scale CSV outputs come back. Never scp probability
rasters off the VM without explicit user authorization (egress cost).

## Scoring specs — status

All three Hampyeong specs were verified **byte-identical** to their original VM
forks on 2026-07-25 (same MD5 on the output CSV):

| spec | md5(csv) | rows |
|---|---|---|
| rest4 | `9fe2d570c89e…` | 36 |
| swin | `c765ab6f3382…` | 9 |
| savelast | `90ad4c5d4efb…` | 6 |

The scorer's provenance audit asserts config==summary==manifest checkpoint,
stride 32, threshold 0.5, per-entry expected checkpoint, and no shared
checkpoint across entries — so a mis-pointed spec fails loudly, it does not
silently mis-score.

## Config caveats (read before running inference)

- The Hampyeong `inference_sweep_hampyeong-s{19,42,58}.yaml` rescued here are
  **Swin-LARGE** configs (Swin-L is a dropped model; its checkpoints are not on
  the VM). They did **not** produce the Swin-**Base** `dev_sweep_all_hampyeong`
  outputs the swin spec scores. The Swin-B "all" config was edited in place and
  is not among these files — regenerate it from the demak swin-B pattern +
  hampyeong input_dir if you need to re-infer Swin-B Hampyeong.
- The `configs/` rescue covered demak + hampyeong only. The VM also has
  `~/configs/{torreypines,trucvert,duck,namibia,…}` (SDS sites) not pulled here.
- Specs default to a VM `output:` path; redirect it before running elsewhere.
