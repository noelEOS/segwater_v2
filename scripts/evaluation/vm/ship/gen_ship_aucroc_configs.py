"""Generate the Demak-concurrent AUC-ROC + threshold-sweep eval configs, one per arm.

This is the *scoring* side of the Demak concurrent gate: it consumes the run dirs
that ``gen_ship_configs.py`` + ``run_*_inference.sh`` produced and feeds
``scripts/evaluate_indonesia_inference_run_aucroc.py`` (note: at the ``scripts/``
root, NOT under ``scripts/evaluation/`` where the rest of the eval kit lives).

Supersedes the DEPRECATED ``vm/gate/gen_demak_gate_aucroc_configs.py``, whose own
docstring lists what it lacks: no prefix-collision check, a hardcoded expected
scene count, its own probability-raster glob without AppleDouble filtering. This
script resolves run dirs through ``runsel`` and asserts they are distinct; the
scene-count contract stays in ``completion.py``, checked separately before
scoring.

WHY THIS SCRIPT EXISTS
----------------------
The Swin-B ``ship`` campaign's equivalent configs were hand-written loose in the
VM's ``~/`` and were never committed, so that campaign's gate scoring is not
reproducible from the repo -- exactly the exposure
``RUNBOOK_multiagent_vm_campaign.md`` §9 ("commit the toolchain, not just the
results") describes. Generating them makes this campaign's gate leg reproducible
and keeps the arm->run_dir mapping machine-derived rather than transcribed.

RUN-DIR RESOLUTION
------------------
Every run dir is resolved through ``runsel.resolve_run_dir``, which anchors on
the ``_<UTC stamp>_`` the sweep always emits. A bare ``<name>_*`` prefix glob
would also match longer sibling sweep names, and this campaign shares a runs root
with 199 pre-existing dirs from other lineages. Never ``hits[0]`` / ``head -1``.

The resolved dirs are also asserted DISTINCT across arms: two arms pointed at one
run dir would produce six plausible rows that are really fewer, which is
indistinguishable from success in the output CSV.

Usage:
    python scripts/evaluation/vm/ship/gen_ship_aucroc_configs.py \\
        --tag cnxb --out-root ~/configs/ship_decision_cnxb_2026-07/aucroc \\
        --output-root ~/workspace/results/ship_decision_cnxb_2026-07/demak_gate/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runsel import RunDirError, resolve_run_dir  # noqa: E402

SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]
RUNS = Path.home() / "segwater_v2/outputs/inference/runs"
ANCILLARY = Path.home() / "ancillary/demak_semarang"

# Reference/mask/sweep block copied verbatim from the pair-based aucroc configs
# so this gate's estimand is unchanged. Do not "tidy" the sweep step: 0.01 is what
# the registered tau* values were computed on.
TEMPLATE = """\
# Demak concurrent AUC-ROC + threshold sweep — campaign "{tag}", {seed}/{variant}.
# {label}. Stride 32, bf16 + TF32 + device-stitch.
# Reference/mask/sweep block is verbatim from the pair-based aucroc configs, so
# tau* and area_bias stay comparable to the registered values.
# Run dir was resolved via runsel (UTC-stamp anchored), never a prefix glob.

evaluation:
  name: "{name}"

  reference_dir: "{ancillary}/reference_s2"
  reference_glob: "*.tif"
  s1_id_regex: "(S1_\\\\d{{8}}_\\\\d{{6}}_\\\\d+_\\\\d+_\\\\d+)"

  spatial_policy: "evaluate_geospatial_overlap"
  resolution_atol: 1.0e-12

  valid_mask_path: "{ancillary}/valid_mask/GSHHG_GlobalSurfaceWater_combined_mask.tif"
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
    {model_key}:
      run_dir: "{run_dir}"

  missing_prediction_policy: "record_and_continue"

output:
  root: "{output_root}"
  run_name: "{name}"
  add_timestamp: false
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True,
                    help="campaign tag, as it appears in the MIDDLE of the "
                         "sweep names (e.g. cnxb)")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="where to write the generated eval configs")
    ap.add_argument("--output-root", type=Path, required=True,
                    help="`output.root` the scorer writes its results under")
    ap.add_argument("--label", default="ConvNeXtV2-Base mx630_stage2",
                    help="human-readable lineage label for the header comment")
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=["best", "last", "swa5"])
    a = ap.parse_args()

    # --- resolve every run dir FIRST; refuse before writing anything ----------
    resolved: dict[str, Path] = {}
    for seed in SEEDS:
        for variant in a.variants:
            sweep = "demak_gate_%s_%s_%s" % (a.tag, seed, variant)
            try:
                resolved["%s/%s" % (seed, variant)] = resolve_run_dir(RUNS, sweep)
            except RunDirError as exc:
                raise SystemExit("%s: %s" % (sweep, exc)) from exc

    # two arms must never share a run dir
    seen: dict[Path, str] = {}
    for key, d in resolved.items():
        if d in seen:
            raise SystemExit("DUPLICATE RUN DIR: %s and %s both resolve to %s"
                             % (key, seen[d], d.name))
        seen[d] = key

    a.out_root.mkdir(parents=True, exist_ok=True)
    for key, run_dir in resolved.items():
        seed, variant = key.split("/")
        name = "demak_gate_%s_%s_%s_aucroc" % (a.tag, seed, variant)
        (a.out_root / (name + ".yaml")).write_text(TEMPLATE.format(
            tag=a.tag, seed=seed, variant=variant, label=a.label, name=name,
            ancillary=ANCILLARY, run_dir=run_dir,
            model_key="%s_%s_%s" % (a.tag, seed, variant),
            output_root=a.output_root))
        print("  %-3s %-4s  %s" % (seed, variant, run_dir.name))
    print("=== WROTE %d aucroc configs under %s ===" % (len(resolved), a.out_root))
    print("=== RUN DIR DISTINCTNESS: %d distinct / %d arms  OK ==="
          % (len(seen), len(resolved)))


if __name__ == "__main__":
    main()
