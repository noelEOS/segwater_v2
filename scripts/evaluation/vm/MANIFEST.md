# VM eval kit — manifest & runbook

One home for the Demak + Hampyeong VM-side evaluation. Everything here was
previously loose in a single VM's `~/` (the only copy); it is now tracked so it
survives VM cleanups and works on a fresh VM or a fleet.

Goal: an agent bringing up a VM to evaluate one site/model should only need to
(1) verify/scp inputs, (2) run inference if the run dir is absent, (3) score.

> **Which runbook do I read?** → **`docs/RUNBOOK_INDEX.md`** routes any task to
> the (usually two) documents it needs, and collects the rules that apply across
> all of them. Start there if you are not already sure.

## Contents

| File | What |
|---|---|
| `score_pairbased_hampyeong.py` | ONE config-driven Hampyeong scorer, `--spec <yaml>`. Replaces the 3 hand-forked VM scorers. |
| `runsel.py` | **load-bearing — copy with the kit.** Unambiguous run-dir resolution: a bare prefix glob matches longer siblings (`demak_full_mx630k_*` also matches `…_mx630k_best_…`), so every lookup anchors on the `_<UTC stamp>_` the sweep always emits. Stdlib-only, no editable install needed. Tests: `tests/test_runsel.py`. |
| `completion.py` | **load-bearing.** The one definition of "this run finished": expected scene count per gate + a `*/*_probability_water.tif` count (AppleDouble `._*` stubs excluded). Exists because every other signal lies — `run_metadata.json` is an *eval* output that never appears in a run dir, `run_summary.json` is rewritten per scene, and `continue_on_error: true` lets a short sweep exit 0. Shell reads the numbers via `--print-expected` instead of keeping a second copy. Tests: `tests/test_completion.py`. |
| `ckptsel.py` | **load-bearing.** One checkpoint-selection implementation, replacing four divergent copies. `best` = the `best.pth` **symlink** only (never glob `*_pmwiou*.pth` — several per seed dir, so a glob picks by sort order) and refuses a `best.pth` pointing at a `*_last.pth`; `last` = the single `*_last.pth`, raising on 0 or >1. Plus `require_seed_token` (mis-copied checkpoint) and `assert_distinct_weights` (sha256 across arms). Keeps the deliberate filename-mIoU vs full-float rules as separate named functions. Tests: `tests/test_ckptsel.py`. |
| `naming.py` | **load-bearing.** `require_no_prefix_collisions` — the generation-time check for the hazard `runsel` catches at read time (a sweep name that is a prefix of another makes every later lookup ambiguous) — plus `atomic_write`/`atomic_to_csv`, so a killed writer never leaves a partial artifact that looks complete. Docstring records the deliberate **no-lockfile** decision. Tests: `tests/test_naming.py`. |
| `analysis/` | area builders, trend fitters and comparison tools — see `analysis/README.md`. |
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

The scorer's provenance audit checks config==summary==manifest checkpoint,
stride 32, threshold 0.5, per-entry expected checkpoint, and no shared
checkpoint across entries — so a mis-pointed spec fails loudly, it does not
silently mis-score. Those guards are `raise ProvenanceError`, not `assert`
(2026-07-30): under `python -O` the old asserts were stripped and the "fails
loudly" claim above was false.

Spec **schema** is now checked Mac-side by `tests/test_spec_schema.py`: every
tracked spec must carry the keys the scorer dereferences (`label`, `slug`,
`seed`, `run_dir`, `checkpoint` per entry). Previously a spec missing `slug`
raised a bare `KeyError` *after* the audit had already printed "passed".

## Rules now enforced by code (and where)

Most of this kit's correctness rules used to live only as prose here or in the
runbooks. As of **2026-07-30** each of the following has an enforcing call site;
the prose is now a description of a check, not the check itself. If you are
tempted to re-document one of these as a manual step, look at the code first.

| Rule | Enforced by |
|---|---|
| The repo must be pip-installed **editable** (else sweeps exit 0 doing nothing) | `check_inputs.sh` → `chk_import` in the `common` block, run from a subshell `cd /tmp` so cwd cannot mask it. Also checks `cv2` (common) and `sklearn` (demak). |
| A run is complete only at its **expected scene count** | `completion.EXPECTED_SCENES` + `python completion.py --gate <g> --check DIR...` (exit 1 if any short). Counts `*/*_probability_water.tif`, excluding AppleDouble `._*` stubs. `--print-expected` feeds the shell so no second copy of the numbers exists. |
| **Per-site SDS flags** (Duck `--keep-top-k 999`; Trucvert / Torrey Pines `--no-min-chainage-length`) | `sds_core.SITE_REQUIRED_FLAGS` + validation in `run_sds_from_rasters.main()`, which `p.error`s naming the missing flag. Override: `--i-know-this-site-needs-flags`. Deliberately a **validation error, not an auto-applied default** — auto-defaults would change the estimand of an already-logged command. Lives in the `SDS_Benchmark_slim/` **nested** repo (`SITE_REQUIRED_FLAGS` added in `0c788c7`; the `run_sds_from_rasters` hardening in `42c3167`). |
| No **sweep name may prefix another** (else every later run-dir lookup is ambiguous) | `naming.require_no_prefix_collisions`, called by `ship/gen_ship_configs.py` over the generated config stems. This is the generation-time half of the rule; `runsel` is the read-time half. |
| **best/last checkpoint resolution** | `ckptsel.resolve_best` (the `best.pth` symlink only — never glob `*_pmwiou*.pth`; refuses a target ending `_last.pth`) and `ckptsel.resolve_last` (exactly one `*_last.pth`). Adopted by `ensure_best_ckpts.py`, `relink_best_ckpts.py`, `eval_stratified_ladder.py`, `build_swa_checkpoint.py`, `ship/gen_ship_configs.py`. |
| **Run-dir resolution** — never a bare prefix glob, never `hits[0]` / `head -1` | `runsel.resolve_run_dir` / `resolve_run_dirs(expect=)` / `resolve_glob_spec`, anchored on the `_<UTC stamp>_`. |
| A sweep that lost scenes must be **machine-visible** | `run_inference_sweep.py`: unconditional `WARNING: N scene job(s) not successful` after the summary, plus `--strict` → non-zero exit. Not non-zero by default, because `benchmark_inference_config_sweep.py` raises on any non-zero child and would abort mid-benchmark. |
| Hampyeong **provenance guards** must survive `python -O` | `score_pairbased_hampyeong.py`: `ProvenanceError` + `require()`, replacing 11 bare `assert`s (message strings verbatim, including the "refusing to clobber" text). |
| **SDS scene-extraction failures** must not look like success | `run_sds_from_rasters.py`: `failed_scenes_by_threshold` + `n_failed_scenes_total` in `sweep_summary.json`, and a non-zero exit **after** all outputs are written (a partial run stays inspectable). `sweep_metrics.csv` unchanged. |
| **Consolidation joins** must not silently drop arms | `ship/consolidate_ship_results.py`: seed-key normalization in all four join loops (`"19"`/`19`/`"s19"`), per-source "N read, M matched, K unmatched" accounting, always exit 1 when a gate CSV was read and **zero** rows matched, and `--strict` for the final pass. |
| **Collation completeness** | `analysis/collate_sds_arms2.py --expect-arms` (default `len(SPEC)`): exits non-zero unless collation ends with that many distinct `model` values, so an unfinished sweep root cannot produce a short table that looks complete. Multi-match spec keys are fatal rather than silently dropping both arms. |

Tests for all of the above are Mac-side and VM-free: `tests/test_completion.py`,
`test_ckptsel.py`, `test_naming.py`, `test_runsel.py`, `test_spec_schema.py`,
`test_build_ship_manifest.py`, `test_consolidate_ship.py`,
`test_collate_sds_arms2.py`, `test_sds_site_flags.py`, `test_hampyeong_guards.py`,
`test_raster_verifiers.py`, `test_manifest_row_append.py`.

### Test environment

The repo suite runs with the **`segwater-test` env python** — the base conda
`python` has no pytest:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/segwater-test/bin/python -m pytest -q
```

`pytest.ini` collects `tests` **and** `scripts/tests` (the latter holds the
backup and stitcher-parity harnesses; `test_backup_runs_to_drive.py` is a
`main()`-style harness wrapped in one pytest test so it is collected coverage
rather than something to remember to run by hand).

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
third-party benchmark with its own layout and vendored CoastSat. Do not copy it
in here.

### ⚠️ `SDS_Benchmark_slim/` is a NESTED git repo

It has **its own git history and no remote** (like `docs/`), so the parent repo
cannot track files inside it. Consequences:

- **Commit with `git -C SDS_Benchmark_slim`**, never `git add -f` from the parent.
- **Scorer provenance:** an SDS result is only reproducible together with the
  slim tree's commit. Record it alongside the results CSV:

  ```bash
  git -C ~/SDS_Benchmark_slim rev-parse --short HEAD
  ```

  The non-zero-exit hardening is commit `42c3167` (2026-07-30); results
  produced before it ran under the old exit-0-on-failed-extraction behaviour.
  `sds_core.SITE_REQUIRED_FLAGS` landed later, in `0c788c7` (2026-07-31).

  ⚠️ **The two copies of this tree diverged.** The Mac held `42c3167` while the
  VM held the same changes only as *uncommitted working-tree edits* on top of
  `5c04772`. Both are now committed with byte-identical content — Mac `0c788c7`,
  VM `cae0805` — but the hashes differ because the histories do. Record the hash
  of the copy that actually ran the scoring, and check `git status` in BOTH
  before trusting either.
- **Edits go in the slim tree only** — never in
  `SDS_Benchmark-0.0-reproducidibility/`, which is the untouched upstream
  reproducibility tree.

Short version for an agent asked to do SDS on the VM:

1. Sync the tree (461 MB, ~15 s, ingress free). The tree is relocatable —
   `sds_core.py` derives `REPO_ROOT` from its own path.

   ```bash
   # refresh: the external IP changes on restart. --project is REQUIRED — bare
   # config-ssh refreshes whatever project gcloud is pointed at, which may be a
   # different one that ALSO has a `gpu-rtx-hpo-west` (see RUNBOOK_INDEX.md).
   gcloud compute config-ssh --project spring-ember-503606
   rsync -az --exclude='__pycache__' SDS_Benchmark_slim/ \
     <instance>.<zone>.<project>:~/SDS_Benchmark_slim/
   ```

   ⚠️ **Use the full `config-ssh` hostname, never a `gcp-vm` alias.**
   `docs/RUNBOOK_INDEX.md` forbids the alias: it has twice resolved to a
   *different project's* instance sharing the hostname `gpu-rtx-hpo-west`, once
   producing a false data-loss report. An earlier revision of this very line
   used the alias.
2. Deps go in the **same** `torch211_cu128_inference` env (already done
   2026-07-25; only needed on a fresh VM):
   `pip install pytz PyQt5 bs4 wget astropy` **plus** conda-only
   `gdal=3.12.3` pinned to the env's `libgdal-core`. No `eda_coastsat` env
   is needed on the VM; `ee` is not needed.
3. **Stage the scenes correctly — see the scene-staging rule below.** Get this
   wrong and everything downstream is a plausible result on the wrong sample.
4. `cd ~/SDS_Benchmark_slim && python scripts/sds/run_sds_from_rasters.py
   --site <SITE> --raster-dir <run dir> --out-dir ~/sds_vm_eval/<label>
   --thresholds 0.5 --segwater-root ~/segwater_v2`
   (`--segwater-root` is required — it imports Segwater's ShorelineVectorizer.)
   For a real threshold sweep pass the comma list `0.1,…,0.9`; a single
   `--thresholds 0.5` still prints a "Sweep summary" header with ONE row.
5. Egress CSVs only.

### ⚠️ Scene staging — the rule (full detail in the runbook)

> **Stage every scene of the frame whose date falls inside the in-situ
> groundtruth's CALENDAR WINDOW. Nothing more, nothing less.**

SDS **silently ignores** scenes it cannot pair with a survey, so a mis-staged
input dir never announces itself — `n_shorelines` counts rasters *fed in*, not
rasters *scored*. Check `n`.

- **Nothing more**: at 3 of the 4 canonical frames the in-situ record ends years
  before the imagery, so most scenes can never contribute.
- **Nothing less**: do **NOT** pre-filter to scenes with a survey within
  `max_days` (10). That is a *framework knob* applied at scoring time, not a
  property of the data. Staging on it makes the staged set depend on a mutable
  setting and breaks the provenance claim *"we used all scenes from frame X
  within the survey period"*. ⚠️ Such scenes are *nearly* inert but **not
  exactly**: measured 2026-07-27, they move RMSE by ≤0.16 m at thr 0.5 (6 of 12
  cells identical to <0.001 m) because pairing is nearest-survey-within-tolerance,
  so extra candidates can change which scene wins a survey. Far inside seed-SD,
  but re-staging means **re-running**, not relabelling.
- ⚠️ **Never reconstruct the split with a date cutoff**; compute it against the
  groundtruth dates and write the keep/drop lists to `~/<site>_sds_split.json`.
- Parked scenes are **moved, never deleted**, into a sibling
  `*_no_groundtruth/` dir with a README + restore command.

Canonical frames, measured 2026-07-27 (staged / archive; scorable at the default
10-day tolerance in parentheses):

| site | canonical frame | staged / archive | (scorable @10 d) |
|---|---|---|---|
| NARRABEEN | `ron_147_ts_9_sn_16` | **87** / 215 | (82) |
| DUCK | `ron_4_ts_24_sn_19` | **109** / 109 — no filtering needed | (79) |
| TORREYPINES | `ron_71` | **25** / 50 ⚠️ thin | (15) |
| TRUCVERT | `ron_8_ts_30_sn_20` | **78** / 139 | (73) |

⚠️ Multi-run sweeps: `batch_sds_sweep_runs.sh` takes **every** subdir of the
runs folder with no site filter (the shared VM runs dir mixes all sites), and
its `find -type d` does not follow symlinks, so a staging dir of symlinks finds
nothing. Loop `run_sds_from_rasters.py` over an explicit glob instead — pattern
in the runbook.

⚠️ Site traps — **now enforced by the scorer** (`sds_core.SITE_REQUIRED_FLAGS`,
validated in `run_sds_from_rasters.main()`; a violation is a `p.error` naming the
missing flag, overridable with `--i-know-this-site-needs-flags`):
**TRUCVERT and TORREYPINES** both need `--no-min-chainage-length` (Trucvert →
empty transect set; Torreypines → `No matched points` at every threshold, after
shorelines extract fine). DUCK needs `--keep-top-k 999` (else whole-run fail).
NARRABEEN needs nothing.

General rule behind the min-chainage flag: it drops transects with fewer than
**30 timesteps**, so **any site with under ~30 staged scenes trips it** —
Torreypines has 25. Check the scene count before assuming a site is flag-free.
Full detail, including the
verified Narrabeen sanity numbers, is in the runbook.
