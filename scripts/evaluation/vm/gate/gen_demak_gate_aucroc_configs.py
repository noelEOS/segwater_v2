"""DEPRECATED (2026-07-30) — superseded by ``ship/gen_ship_configs.py``.

Kept as the **executable record** of the tracked `configs/aucroc/demak_gate_*`
scoring configs it produced (2026-07-25 checkpoint-selection gate). Do not
generate NEW configs with it.

Unlike the three sibling gate generators this one resolves RUN DIRS, not
checkpoints — it has no `resolve()` copy to diverge, and it already goes through
`runsel.resolve_run_dirs` (anchored on the UTC stamp). What it still lacks:
  * **No prefix-collision check** on the emitted `demak_gate_{arm}_{seed}_s32`
    config/run names (`naming.require_no_prefix_collisions`, which
    `gen_ship_configs.py` calls). None of the five deprecated generators has one.
  * **Hardcoded expected scene count.** `if n_tif != 6` inlines the Demak-gate
    completion contract as a literal. That number now lives in one place,
    `completion.EXPECTED_SCENES["demak_gate"]`, alongside the reasons the naive
    completion signals (run_metadata.json, run_summary.json, sweep exit code)
    are all wrong; a literal here can drift from it silently.
  * **Its own probability-raster glob** (`*/*_probability_water.tif`) rather than
    `completion.count_probability_rasters`, so it does not get the AppleDouble
    (`._*`) stub filtering that inflated tif counts elsewhere.
  * **No atomic write** — configs go out via plain `write_text`.

Original docstring: generate AUC-ROC scoring configs, one per Demak gate
inference run; discover run dirs by the `gate_{arm}_demak_{seed}_...` prefix so
it stays in sync with whatever the inference batch actually produced; fail
loudly if an expected arm is missing or ambiguous rather than silently scoring
8 of 9.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runsel import resolve_run_dirs  # noqa: E402

RUNS = Path("/home/noel/segwater_v2/outputs/inference/runs")
OUT = Path("/home/noel/configs/aucroc_gate")
SEEDS = ["s19", "s42", "s58"]
ARMS = ["best", "last", "swa5"]

TEMPLATE = """\
# Demak concurrent gate scoring — Swin-B stage2 seed {seed}, arm "{arm}".
# Same estimator as every registered accuracy table: S2 vote-and-veto reference,
# GSHHG+GSW valid mask, threshold sweep 0->1 step 0.01, greater_than.

evaluation:
  name: "{name}"

  reference_dir: "/home/noel/ancillary/demak_semarang/reference_s2"
  reference_glob: "*.tif"
  s1_id_regex: "(S1_\\\\d{{8}}_\\\\d{{6}}_\\\\d+_\\\\d+_\\\\d+)"

  spatial_policy: "evaluate_geospatial_overlap"
  resolution_atol: 1.0e-12

  valid_mask_path: "/home/noel/ancillary/demak_semarang/valid_mask/GSHHG_GlobalSurfaceWater_combined_mask.tif"
  valid_mask_value: 1

  reference_water_values: [1]
  reference_nodata_values: [255]

  prediction:
    type: "probability_map"
    path_template: "{{run_dir}}/{{s1_id}}/{{s1_id}}_probability_water.tif"

  threshold_sweep:
    enabled: true
    start: 0.0
    stop: 1.0
    step: 0.01
    comparison: "greater_than"
    optimize_metrics: ["iou", "f1", "mcc"]

  model_runs:
    Upernet_Swin_Base_224_{arm}_native224_weighted:
      run_dir: "{run_dir}"

  missing_prediction_policy: "record_and_continue"

output:
  root: "/home/noel/pairbased_vm_eval"
  run_name: "{name}"
  add_timestamp: false
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written, problems = [], []
    for seed in SEEDS:
        for arm in ARMS:
            # Anchored on the UTC stamp: a bare prefix glob would also match a
            # sweep whose name extends this one (e.g. an `..._s42_PERF` re-run).
            hits = resolve_run_dirs(RUNS, "gate_%s_demak_%s" % (arm, seed))
            if len(hits) != 1:
                problems.append("%s/%s: %d run dirs%s" % (
                    seed, arm, len(hits),
                    "".join("\n      " + h.name for h in hits)))
                continue
            run_dir = hits[0]
            n_tif = len(list(run_dir.glob("*/*_probability_water.tif")))
            if n_tif != 6:
                problems.append("%s/%s: %d/6 probability tifs" % (seed, arm, n_tif))
                continue
            name = "demak_gate_%s_%s_s32" % (arm, seed)
            (OUT / (name + ".yaml")).write_text(
                TEMPLATE.format(seed=seed, arm=arm, name=name, run_dir=str(run_dir)))
            written.append((name, run_dir.name))
    print("=== WROTE %d scoring configs ===" % len(written))
    for name, rd in written:
        print("  %-28s <- %s" % (name, rd))
    if problems:
        print("=== PROBLEMS ===")
        for p in problems:
            print("  !! " + p)
        sys.exit(1)


if __name__ == "__main__":
    main()
