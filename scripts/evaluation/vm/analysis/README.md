# VM-side analysis scripts

Area builders, trend fitters and comparison tools that ran on the GCP VM during
the mx630 evaluation (2026-07-29). They lived only in `~/workspace/scripts/` on
the VM; tracked here so the analysis is reproducible if that machine goes away.

**These are working scripts, not a library.** They hardcode VM absolute paths
(`/home/noel/...`). Treat them as the executable record of *what was computed*;
adjust paths before re-running elsewhere.

## Area builders — per-scene water area from probability rasters

| script | scope |
|---|---|
| `build_demak_full_areas.py` | Demak full series, pair-based `last`, 3 seeds |
| `build_demak_best_areas.py` | same, `best` arm (asserts `_pmwiou` in ckpt, rejects `_last`) |
| `build_full3_areas.py` | Demak full series, first 3 mx630 arms |
| `build_full5_areas.py` | Demak full series, all 5 mx630 arms |
| `build_s112_areas.py` | Demak full series, stride 112 + bf16/device-stitch |
| `build_proxy17_areas.py` | 17-scene proxy subset |

Shared guards (do not relax — they exist because of specific past incidents):

- `assert enc == EXPECT_ENC[arm]` — encoder must match the arm. Two arms of
  *different architectures* share the checkpoint filename `step23930_last`.
- `assert ck not in seen` — checkpoints distinct across arms.
- run-dir resolution anchored on the `_<UTC timestamp>_` that follows the arm
  name. A bare prefix glob is ambiguous once one arm name is a prefix of
  another (`mx630k` also matches `mx630k_best`).
- `assert len(scenes) == 213`; `A_PRIORI_EXCLUSIONS` drops the orbit-127 scene;
  `in_analysis_window` flags `<= 2024-12-31` (206 scenes).

## Trend fitters

| script | estimand |
|---|---|
| `fit_trend_by_arm.py` | full 206-scene window, grouped by `arm` |
| `fit_s2matched.py` | S2-date-matched (n=48), all arms |
| `fit_s112.py` | both estimands for a single-arm CSV |
| `fit_demak_trend.py` | earlier per-seed version |

Canonical spec (replicates `experiments/demak_semarang/outputs/window2024/run_2024win.py`
and `experiments/PAIRBASED_EVALUATION/demak_trend/scripts/trend_variants_csvonly.py`):

    area ~ 1 + t_years + annual + semiannual harmonics
    OLS, Newey-West/HAC SEs, maxlags = max(1, floor(4*(n/100)**(2/9)))
    window <= 2024-12-31 23:59:59 UTC, orbit-127 dropped
    date-match: nearest S1 scene within +/-6 d of each gated optical date
                (valid_frac_aoi >= 0.90), deduplicated by positional index

⚠️ **Use HAC, never plain OLS.** The fitters print plain-OLS alongside for
reference only; quoting it against a registered HAC figure overstates deltas by
~20 ha/yr. Registered controls these reproduce exactly: chip full thr0.5
+340.39 ± 30.81 (n=206), chip S2-matched +320.75 ± 19.12 (n=48).

⚠️ Single seed per arm ⇒ the HAC SE is the uncertainty. This is **not**
comparable in kind to the 3-seed seed-SD carried by the registered rows.

## Comparison tools

- `collate_sds_arms2.py` — Narrabeen SDS sweep metrics across arms into one
  tidy CSV, tagging `arch` / `lineage` / `arm` / `stride`. Skips (loudly)
  rather than guessing when a dir does not match exactly one spec entry.
- `perf_raster_diff.py` — probability-raster divergence between two runs:
  max/mean |Δ| and the fraction of pixels whose 0.5 decision flips. Used for
  the bf16/tf32/device-stitch experiment. `--base-ts` / `--perf-ts` pin a run
  when several timestamps exist for the same stride.
- `verify_rerun_equality.py` — byte-equality check between re-runs.

## configs/

`spec_hampyeong_mx630_arms2.yaml` — 5-arm Hampyeong spec spanning two
architectures and two lineages; per-entry `lineage_slug` / `training_data`.

`*_PERF*.yaml`, `*_s112_bf16dev.yaml` — the perf-lever experiment. Each differs
from its baseline **only** in `compute.amp_dtype`, `compute.tf32`,
`stitching.accumulate_on_device` (and stride, for the s112 run).
