"""Score pair-based-memmap Hampyeong runs on the VM — one config-driven scorer.

This replaces the family of near-duplicate hand-forked scorers that used to live
loose in the VM home directory (score_pairbased_hampyeong_vm.py,
..._vm_rest4.py, ..._vm_savelast.py). They shared one identical metric spine and
differed ONLY in *which* runs they targeted and how the run dir was resolved.
That variation is now data: a ``--spec`` YAML. The spine below — loader, guards,
provenance audit, metrics — is the canonical one, unchanged, so outputs are
directly comparable to results/pairbased_per_date_metrics.csv.

Run:
    python scripts/evaluation/vm/score_pairbased_hampyeong.py --spec <spec.yaml>

Guards kept intact from every original scorer (these are the point of the
scorer, not decoration; do not relax them):
  * threshold 0.5, greater_equal, resolution_atol 1e-6.
  * EXPECTED_VALID_PIXELS assertion (post all masks).
  * identical ground-truth across every entry on a given date.
  * prediction-digest DISTINCTNESS across entries (catches seed/ckpt miswiring).
  * provenance audit: run_config.yaml == run_summary.json == run_manifest.csv
    checkpoint; stride 32; threshold 0.5; each entry's checkpoint is the one the
    spec expects; no two entries share a checkpoint.

Spec schema (YAML) — see specs/ for the three canonical instances:
    repo:        abs path to the segwater_v2 checkout (default /home/noel/segwater_v2)
    nas_root:    abs path to the ancillary NAS mirror (DEM masks + tide-gauge mask)
    pair_root:   run root, relative to repo (default outputs/inference)
    expected_valid_pixels: int
    variant_column: bool         # emit a `variant` column (savelast A/B)
    lineage_slug: str            # dataset/training lineage in the `model` column
                                 # (default "pairbased" -- MUST be set for any
                                 #  other lineage, e.g. "mx630s2", "mx630k")
    training_data: str           # `training_data` column (default "pair-based")
    output: abs path for the CSV (refuses to clobber)
    entries:                     # one row-source per arch/seed/variant
      - label:      human arch label, e.g. Swin-B
        slug:       model-slug used in the `model` column. MUST name the
                    architecture under test (e.g. `swinb`, `cnxv2t`) -- the
                    `model` column is often read on its own, and two arms of
                    different architectures can share a checkpoint FILENAME.
        seed:       19|42|58
        variant:    optional string (only with variant_column)
        run_dir:    EITHER an explicit path relative to pair_root
                    OR a `glob:` pattern relative to pair_root (must match 1).
                    `glob:` is AUTO-ANCHORED: the first `*` must begin with the
                    UTC timestamp (`\\d{8}T\\d{6}Z`) that every sweep emits, so
                    a longer sibling run name cannot be absorbed. To pin one
                    exact run instead, give the explicit dir name.
        checkpoint: expected checkpoint path relative to repo
        lineage_slug:  optional per-entry override of the file-wide value
        training_data: optional per-entry override of the file-wide value
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runsel import RunDirError, resolve_glob_spec


class ProvenanceError(Exception):
    """A provenance/consistency guard failed. Raised, never asserted: `assert`
    is stripped under `python -O`, and these guards are the point of the scorer
    (see the module docstring) -- they must fail loudly under every interpreter
    flag, not silently mis-score."""


def require(cond, msg: str) -> None:
    """Guard helper: raise :class:`ProvenanceError` when `cond` is falsey."""
    if not cond:
        raise ProvenanceError(msg)


DATES = ["20210305", "20210422", "20210621"]
SCENE_IDS = {
    "20210305": "S1B_IW_GRDH_1SDV_20210305T213224_20210305T213249_025885_031658_6933_Clipped",
    "20210422": "S1B_IW_GRDH_1SDV_20210422T213225_20210422T213250_026585_032CBA_CF3C_Clipped",
    "20210621": "S1B_IW_GRDH_1SDV_20210621T213228_20210621T213253_027460_034782_6C41_Clipped",
}


def load_spec(spec_path: Path) -> dict:
    spec = yaml.safe_load(spec_path.read_text())
    spec.setdefault("repo", "/home/noel/segwater_v2")
    spec.setdefault("nas_root", "/home/noel/ancillary/hampyeong/nas_root")
    spec.setdefault("pair_root", "outputs/inference")
    spec.setdefault("variant_column", False)
    # Lineage tags. These default to the pair-based values so pre-existing specs
    # keep producing byte-identical output, but ANY non-pair-based spec must set
    # them -- `dataset` alone is not sufficient provenance, and a wrong lineage
    # string in the `model` column is exactly the kind of thing that later gets
    # pooled by mistake.
    spec.setdefault("lineage_slug", "pairbased")
    spec.setdefault("training_data", "pair-based")
    if "expected_valid_pixels" not in spec:
        raise ValueError("spec must set expected_valid_pixels")
    if "output" not in spec:
        raise ValueError("spec must set output")
    if not spec.get("entries"):
        raise ValueError("spec must list at least one entry")
    return spec


def resolve_run_dir(pair_root: Path, run_dir: str) -> str:
    """Return run_dir relative to pair_root. Accepts an explicit path or a
    `glob:<pattern>` that must match exactly one dir (keeps the resolved
    timestamped dir name, so provenance is preserved in the output/logs).

    `glob:` patterns are AUTO-ANCHORED: the first `*` must begin with the UTC
    timestamp the sweep always emits, so a run whose name merely extends this
    one (`…_last` vs `…_last_PERF`) can no longer be absorbed. This is a
    tightening only -- it can reject a dir that violates the naming contract,
    never one that satisfies it. See runsel.resolve_glob_spec.
    """
    if run_dir.startswith("glob:"):
        pattern = run_dir[len("glob:"):]
        hits = resolve_glob_spec(pair_root, pattern)
        if len(hits) != 1:
            raise RunDirError(
                "glob %r: expected 1 run dir under %s, got %d\n%s"
                % (pattern, pair_root, len(hits),
                   "\n".join("  " + h.name for h in hits) or "  (none)")
            )
        return str(hits[0].relative_to(pair_root))
    return run_dir


def audit(spec: dict, repo: Path, pair_root: Path) -> list[dict]:
    """Resolve every entry's run dir and verify its provenance. Returns the
    entries with `resolved_run_dir` filled in."""
    seen: dict[str, str] = {}
    resolved = []
    for e in spec["entries"]:
        tag = _tag(e, spec["variant_column"])
        run_dir = resolve_run_dir(pair_root, e["run_dir"])
        base = pair_root / run_dir
        cfg = yaml.safe_load((base / "run_config.yaml").read_text())
        summary = json.loads((base / "run_summary.json").read_text())
        ckpt = cfg["inference"]["checkpoint_path"]
        require(ckpt == summary["checkpoint_path"], f"{tag}: config/summary checkpoint mismatch")
        require(ckpt == e["checkpoint"], f"{tag}: expected {e['checkpoint']}, got {ckpt}")
        man = pd.read_csv(base / "run_manifest.csv")
        require(set(man["checkpoint_path"].unique()) == {ckpt}, f"{tag}: manifest disagrees")
        require(ckpt not in seen, f"checkpoint collision: {tag} / {seen.get(ckpt)}")
        seen[ckpt] = tag
        require(cfg["inference"]["data"]["stride"] == 32, f"{tag}: stride != 32")
        require(cfg["inference"]["post_processing"]["threshold"] == 0.5, f"{tag}: threshold != 0.5")
        resolved.append({**e, "resolved_run_dir": run_dir})
    print(f"Provenance audit passed: {len(seen)} distinct checkpoints")
    for ckpt, tag in seen.items():
        print(f"  {tag:20s} -> {ckpt}")
    print()
    return resolved


def _tag(entry: dict, variant_column: bool) -> str:
    if variant_column and entry.get("variant"):
        return f"{entry['label']} s{entry['seed']} {entry['variant']}"
    return f"{entry['label']} s{entry['seed']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path, help="scoring spec YAML")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    repo = Path(spec["repo"])
    sys.path.insert(0, str(repo / "scripts"))
    # imported after repo is on the path — same modules every original used
    from evaluation.metrics import METRIC_NAMES, compute_binary_metrics  # noqa: E402
    from inference_overlap_utils import load_overlap_reference_and_prediction  # noqa: E402

    nas = Path(spec["nas_root"])
    pair_root = repo / spec["pair_root"]
    expected_valid = int(spec["expected_valid_pixels"])
    variant_column = bool(spec["variant_column"])
    lineage_slug = str(spec["lineage_slug"])
    training_data = str(spec["training_data"])
    dest = Path(spec["output"])
    require(not dest.exists(), f"refusing to clobber existing {dest}")

    entries = audit(spec, repo, pair_root)
    mask = nas / "Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM/DEM_VALID_MASK_aoi.tif"

    rows = []
    for date in DATES:
        ref = (nas / "sen12coast/Validation/Hampyeong/DEM_FLOOD_MASKS/Descending_reproj"
               / f"DEM_FLOOD_S1_DESC_{date}_VAL_AOI.tif")
        scene = SCENE_IDS[date]
        digests: dict[str, str] = {}
        y_true_reference = None
        for e in entries:
            tag = _tag(e, variant_column)
            pred = pair_root / e["resolved_run_dir"] / scene / f"{scene}_probability_water.tif"
            require(pred.exists(), f"missing prediction {pred}")
            y_true, y_pred, diag = load_overlap_reference_and_prediction(
                reference_path=str(ref),
                prediction_path=str(pred),
                reference_water_values=[1],
                reference_nodata_values=None,
                probability_threshold=0.5,
                probability_comparison="greater_equal",
                resolution_atol=1e-6,
                valid_mask_path=str(mask),
                valid_mask_value=1,
            )
            n_valid = diag["valid_pixels_after_all_masks"]
            require(n_valid == expected_valid, f"{date} {tag}: valid px {n_valid} != {expected_valid}")
            if y_true_reference is None:
                y_true_reference = y_true
            else:
                require(np.array_equal(y_true, y_true_reference), f"{date}: y_true differs for {tag}")
            d = hashlib.sha1(y_pred.tobytes()).hexdigest()
            require(d not in digests, f"{date}: {tag} identical prediction to {digests[d]} -- miswiring")
            digests[d] = tag
            m = compute_binary_metrics(y_true, y_pred, include_counts=True)
            # Per-entry override: one spec may span lineages (e.g. an mx630s2
            # Swin-B arm next to an mx630k ConvNeXtV2-T arm), so the file-wide
            # default is not always right.
            e_lineage = str(e.get("lineage_slug", lineage_slug))
            model_name = f"{e['slug']}_s{e['seed']}_{e_lineage}"
            if variant_column and e.get("variant"):
                model_name += f"_{e['variant']}"
            # Column order matches the original forks: `variant` (when present)
            # sits right after `date`; every other column is unchanged.
            row = {"date": int(date)}
            if variant_column:
                row["variant"] = e.get("variant", "")
            row.update({
                "model": model_name,
                "arch": e["label"],
                "training_data": str(e.get("training_data", training_data)),
                "seed": e["seed"],
                **{k: m[k] for k in METRIC_NAMES},
                **{c: m[c] for c in ("tn", "fp", "fn", "tp")},
                "n_valid": n_valid,
                "water_fraction_gt": float(y_true.mean()),
            })
            rows.append(row)
            print(f"  {date}  {tag:24s}  IoU={m['iou']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest} ({len(out)} rows)")


if __name__ == "__main__":
    main()
