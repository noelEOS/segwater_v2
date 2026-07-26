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

## ⚠️ Fresh VM: the segwater package must be pip-installed editable

```bash
cd ~/segwater_v2 && python -m pip install -e . --no-deps --no-build-isolation
```

Without it `import src` resolves only when cwd *is* the repo root.
`run_inference_sweep.py` spawns per-group subprocesses that do not inherit that
cwd, so **every group dies with `ModuleNotFoundError: No module named 'src'`** —
and the sweep still **exits 0**, printing only `FAILED GROUP` lines. It looks
like a config error, not an env error. Seen on a fresh VM 2026-07-26.

Check it the way the failure actually manifests — from *outside* the repo:

```bash
cd /tmp && python -c "from src.utils.vectorizer import ShorelineVectorizer"
```

`--no-deps` keeps pip from disturbing the conda-installed torch/GDAL stack.
Other fresh-VM gaps seen the same day: `cv2` (`pip install
opencv-python-headless`) and the SDS extras in `docs/RUNBOOK_sds_vm_eval.md` §0b.

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

## SDS is a separate job — see `docs/RUNBOOK_sds_vm_eval.md`

This kit covers **Demak + Hampyeong** only. SDS (satellite-derived shoreline:
NARRABEEN / DUCK / TORREYPINES / TRUCVERT) runs from the `SDS_Benchmark_slim/`
tree, which is intentionally **not** part of this kit — it is a 461 MB
third-party benchmark with its own layout and vendored CoastSat, untracked in
this repo. Do not copy it in here.

Short version for an agent asked to do SDS on the VM:

1. `rsync -az --exclude='__pycache__' SDS_Benchmark_slim/ gcp-vm:~/SDS_Benchmark_slim/`
   (461 MB, ~15 s, ingress free). The tree is relocatable — `sds_core.py`
   derives `REPO_ROOT` from its own path.
2. Deps go in the **same** `torch211_cu128_inference` env (already done
   2026-07-25; only needed on a fresh VM):
   `pip install pytz PyQt5 bs4 wget astropy` **plus** conda-only
   `gdal=3.12.3` pinned to the env's `libgdal-core`. No `eda_coastsat` env
   is needed on the VM; `ee` is not needed.
3. `cd ~/SDS_Benchmark_slim && python scripts/sds/run_sds_from_rasters.py
   --site <SITE> --raster-dir <run dir> --out-dir ~/sds_vm_eval/<label>
   --thresholds 0.5 --segwater-root ~/segwater_v2`
   (`--segwater-root` is required — it imports Segwater's ShorelineVectorizer.)
   For a real threshold sweep pass the comma list `0.1,…,0.9`; a single
   `--thresholds 0.5` still prints a "Sweep summary" header with ONE row.
4. Egress CSVs only.

⚠️ Multi-run sweeps: `batch_sds_sweep_runs.sh` takes **every** subdir of the
runs folder with no site filter (the shared VM runs dir mixes all sites), and
its `find -type d` does not follow symlinks, so a staging dir of symlinks finds
nothing. Loop `run_sds_from_rasters.py` over an explicit glob instead — pattern
in the runbook.

⚠️ Site traps: TRUCVERT needs `--no-min-chainage-length` (else empty), DUCK
needs `--keep-top-k 999` (else whole-run fail). Full detail, including the
verified Narrabeen sanity numbers, is in the runbook.
