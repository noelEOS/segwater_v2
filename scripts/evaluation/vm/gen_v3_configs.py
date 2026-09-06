#!/usr/bin/env python3
"""Generate the dataset_v3 lineage's inference sweep configs.

Seven sites: Demak (6-scene concurrent gate), Demak full series (213 scenes),
Hampyeong (24-scene bay) and the four SDS frames (Narrabeen, Duck, Torrey
Pines, Trucvert).

The v3 stage-2 runs differ from every earlier lineage in four ways that all have
to be stated in the config, because nothing downstream can infer them:

* **checkpoints** live at ``outputs/stage2_v3/<arch>/<HH-MM-SS>_s<seed>/`` and
  the deployed arm is ``*_last.pth`` -- the rule adopted on the 3-seed Demak
  gate (docs/RUNBOOK_checkpoint_selection_gate.md), and what
  docs/dataset_v3/HPO_V3_STUDIES.md specifies for this lineage.
* **s1_encoding: q55s005** -- v3 was trained on quantized dB, the evaluation
  rasters are continuous dB, and the key is mandatory so a config that omits it
  cannot run at all. See ``build_s1_encoding`` in scripts/run_inference.py.
* **normalization** comes from configs/inference.yaml, which now carries the v3
  constants; nothing per-config is needed, but a config generated against a
  different base would silently disagree -- hence the assertion below.
* **perf flags on** (decided 2026-09-06): ``amp_dtype: bfloat16`` + ``tf32`` +
  ``accumulate_on_device``, measured 1.69x on this GPU class
  (docs/INFERENCE_SPEEDUP_NOTES.md A.5).

  Note this is NOT the training recipe's four flags: ``torch.compile`` is not
  wired into the inference entry point (``maybe_compile`` is never imported
  there) and ``cudnn.benchmark`` is already set unconditionally for CUDA, so
  the inference surface is exactly the three knobs above.

  bf16 and tf32 are numerics-affecting. The speedup notes gate bf16 adoption on
  a one-scene ``null`` vs ``bfloat16`` check; that gate was waived by decision,
  so v3 probability rasters are not bit-comparable with the fp32 lineage runs.
  Metrics comparisons remain valid; a byte-level diff against an old run is not.

Written to be the executable record of the configs it emits, with the three
properties gen_rest4_configs.py was deprecated for lacking: emission from a
single parsed template (not string concatenation), a prefix-collision check on
the emitted sweep names, and an atomic write.

Usage::

    python scripts/evaluation/vm/gen_v3_configs.py --check   # diff, write nothing
    python scripts/evaluation/vm/gen_v3_configs.py           # write the configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from naming import atomic_write_text, require_no_prefix_collisions  # noqa: E402

REPO = HERE.parents[2]
CONFIG_ROOT = HERE / "configs"

SEEDS = ["s19", "s42", "s58"]

# arch dir token -> (encoder_name, short slug for the sweep name). The run dir
# under outputs/stage2_v3/<arch>/ is discovered, never hardcoded: it carries a
# wall-clock stamp that is not derivable from the seed.
ARCHS = {
    "upernet_tu-swin_base_patch4_window7_224": ("tu-swin_base_patch4_window7_224", "swinb"),
    "upernet_tu-convnextv2_base": ("tu-convnextv2_base", "cnxb"),
}

# The v3 constants, asserted against configs/inference.yaml so a config can
# never be generated against a base that disagrees with the lineage.
V3_MEANS = [-15.435106, -24.856323]
V3_STDS = [6.517790, 9.117346]

SITES = {
    # The 6-scene concurrent gate: S1 scenes paired with same-window S2, scored
    # against the S2 vote-and-veto reference. This is the accuracy gate.
    "demak": {
        "subdir": "demak",
        "input_dir": "/home/noel/data_demak_concurrent",
        "input_glob": "S1_*.tif",
        "name": "v3_demak_concurrent",
        # Demak keeps the length filter: the gate scores a coastline, and short
        # spurious rings inland are not shoreline. Mirrors the pair-based
        # concurrent configs exactly so only lineage and perf differ.
        "post": {
            "inference.post_processing.filtering.apply_length_filter": True,
            "inference.post_processing.filtering.min_length_meters": 10000.0,
        },
    },
    "hampyeong": {
        "subdir": "hampyeong",
        "input_dir": "/home/noel/hampyeong_ron_134_ts_16_sn_15_all24",
        "input_glob": "S1B_*.tif",
        "name": "v3_hamp24",
        # Hampyeong keeps keep_top_k and NO length filter: the bay's waterline
        # is not one long line, and the 24-scene campaign scored it this way.
        "post": {
            "inference.post_processing.filtering.apply_length_filter": False,
            "inference.post_processing.filtering.min_length_meters": 10000.0,
            "inference.post_processing.filtering.keep_top_k": 5,
        },
    },
    # The 213-scene full time series (2017-03-16 .. 2025-04-03, frame 76_8_9,
    # one scene per date). Feeds the shoreline-change/trend analysis rather than
    # the accuracy gate -- there is no concurrent S2 reference for these dates.
    #
    # Same post-processing as the concurrent gate, deliberately: the two products
    # must be the same measurement at different sampling, or a trend built from
    # the series cannot be read against the gate's accuracy.
    #
    # ⚠️ 213 scenes x 2175x2160 px per arm -- by far the largest run here.
    # ⚠️ The old lineage's config pointed at ~/data_demak_full_series; on this VM
    # the series lives at ~/data_demak (what check_inputs' demak_full gate
    # resolves by default, and what SEGWATER_DEMAK_FULL_DATA overrides).
    "demak_full": {
        "subdir": "demak",
        "input_dir": "/home/noel/data_demak",
        "input_glob": "S1_*.tif",
        "name": "v3_demak_full",
        "post": {
            "inference.post_processing.filtering.apply_length_filter": True,
            "inference.post_processing.filtering.min_length_meters": 10000.0,
        },
    },
    # --- SDS sites (satellite-derived shoreline) ---
    #
    # No length filter and no keep_top_k: SDS extracts a waterline per scene and
    # scores it against in-situ transects itself, so a shoreline post-filter here
    # would discard the very geometry the benchmark measures. This matches the
    # pair-based SDS configs, which set apply_length_filter false and nothing else.
    #
    # Stride stays s32 via the shared preset -- the operational stride (the
    # manuscript reports s32 throughout; s8 is the Supplementary S11
    # compute-for-precision option, not a different default).
    #
    # ⚠️ input_dir is the STAGED dir. Scenes whose acquisition date falls outside
    # the site's in-situ groundtruth window (padded by max_days at both ends) are
    # parked in a sibling *_no_groundtruth/. SDS silently ignores scenes it cannot
    # pair, so a mis-staged dir yields a plausible result on the wrong sample.
    # Staging is computed from the groundtruth dates (~/stage_sds_splits.py) and
    # verified by scripts/evaluation/vm/sds_scene_check.py -- never a date cutoff.
    **{
        key: {
            "subdir": "sds",
            "input_dir": f"/home/noel/Inference_input/{src}",
            "input_glob": "*.tif",
            "name": f"v3_sds_{key}",
            "post": {
                "inference.post_processing.filtering.apply_length_filter": False,
                "inference.post_processing.filtering.min_length_meters": 10000.0,
            },
        }
        for key, src in [
            ("narrabeen", "NARRABEEN_ron_147_ts_9_sn_16"),
            ("duck", "Duck_ron_4_ts_24_sn_19"),
            ("torreypines", "Torreypines_ron_71"),
            ("trucvert", "TRUCVERT_ron_8_ts_30_sn_20"),
        ]
    },
}


def find_run_dir(arch: str, seed: str) -> str:
    """The single stage-2 run dir for one arch/seed, by its ``_s<seed>`` suffix.

    Raises rather than picking when the directory is absent or ambiguous: a
    wrong checkpoint here mis-scores the whole campaign silently.
    """
    root = REPO / "outputs" / "stage2_v3" / arch
    hits = sorted(p for p in root.glob(f"*_{seed}") if p.is_dir()) if root.is_dir() else []
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one run dir for {arch}/{seed} under {root}, found {len(hits)}"
            + (f": {[h.name for h in hits]}" if hits else "")
        )
    return hits[0].name


def checkpoint_path(arch: str, seed: str, run: str) -> str:
    """The deployed arm: the run's single ``*_last.pth``.

    Path is relative to the repo root, matching how every other sweep config
    spells a checkpoint. Existence is checked when generating on the VM; the
    Mac has no checkpoints, so absence there is not an error.
    """
    rel = f"outputs/stage2_v3/{arch}/{run}"
    local = REPO / rel
    if local.is_dir():
        lasts = sorted(local.glob("*_last.pth"))
        if len(lasts) != 1:
            raise SystemExit(f"expected one *_last.pth in {rel}, found {len(lasts)}")
        return f"{rel}/{lasts[0].name}"
    # Off-VM: the builder's own naming contract.
    return f"{rel}/{arch}_{seed}_step36140_last.pth"


def build_config(site_key: str, arch: str, seed: str) -> tuple[str, dict]:
    site = SITES[site_key]
    encoder, slug = ARCHS[arch]
    run = find_run_dir(arch, seed)
    sweep_name = f"{site['name']}_{slug}_{seed}"

    overrides = {
        # Lineage. Mandatory key -- see build_s1_encoding in run_inference.py.
        "inference.data.s1_encoding": "q55s005",
        # Perf (numerics-affecting; decided 2026-09-06).
        "inference.compute.amp_dtype": "bfloat16",
        "inference.compute.tf32": True,
        "inference.stitching.accumulate_on_device": True,
        # Geometry/post-processing, carried over from the scored configs so the
        # lineage is the only thing that changes.
        "inference.post_processing.smoothing.apply_simplification": True,
        "inference.post_processing.smoothing.simplify_tolerance_meters": 1.0,
        "inference.data.edge_policy": "shift_inward",
        "inference.stitching.min_weight": 0.001,
        "inference.data.num_workers": 8,
        "inference.data.batch_size": 256,
        "inference.output.probability_precision": "float16",
    }
    overrides.update(site["post"])

    doc = {
        "sweep": {
            "name": sweep_name,
            "dry_run": False,
            "continue_on_error": True,
            "input_dir": site["input_dir"],
            "input_glob": site["input_glob"],
            "common_overrides": overrides,
            "checkpoints": [
                {
                    "name": f"{arch}_{seed}",
                    "checkpoint_path": checkpoint_path(arch, seed, run),
                    "model": {"arch": "upernet", "encoder_name": encoder},
                }
            ],
            "presets": [
                {
                    "name": "native224_weighted_224_b0_s32",
                    "overrides": {
                        "inference.data.tile_size": 224,
                        "inference.data.buffer_size": 0,
                        "inference.data.stride": 32,
                        "inference.stitching.mode": "weighted_blend",
                        "inference.stitching.blend_window": "hann",
                    },
                }
            ],
        }
    }
    return sweep_name, doc


HEADER = """\
# GENERATED by scripts/evaluation/vm/gen_v3_configs.py -- do not hand-edit.
#
# dataset_v3 lineage, {site}, {arch} seed {seed}.
#   checkpoint : *_last.pth (the deployed arm for this lineage)
#   encoding   : q55s005 -- eval rasters are re-quantized onto the training
#                lattice before normalization (mandatory key)
#   perf       : bf16 + tf32 + device-stitch (~1.69x). Numerics-affecting:
#                these rasters are NOT bit-comparable with fp32 lineage runs.
"""


def assert_base_config_is_v3() -> None:
    """The generated configs inherit normalization from configs/inference.yaml."""
    base = yaml.safe_load((REPO / "configs" / "inference.yaml").read_text())
    norm = base["inference"]["data"]["normalization"]
    bad = []
    if [round(v, 6) for v in norm["means"]] != V3_MEANS:
        bad.append(f"means {norm['means']} != {V3_MEANS}")
    if [round(v, 6) for v in norm["stds"]] != V3_STDS:
        bad.append(f"stds {norm['stds']} != {V3_STDS}")
    if bad:
        raise SystemExit("configs/inference.yaml is not on the v3 constants: " + "; ".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args()

    assert_base_config_is_v3()

    emitted: dict[str, tuple[Path, str]] = {}
    for site_key, site in SITES.items():
        for arch in ARCHS:
            for seed in SEEDS:
                name, doc = build_config(site_key, arch, seed)
                text = HEADER.format(site=site_key, arch=arch, seed=seed) + yaml.safe_dump(
                    doc, sort_keys=False, default_flow_style=False, width=100
                )
                # Parse what we are about to write, not what we meant to write.
                round_trip = yaml.safe_load(text)
                assert round_trip["sweep"]["name"] == name, name
                out = CONFIG_ROOT / site["subdir"] / f"inference_sweep_{name}.yaml"
                emitted[name] = (out, text)

    # Generation-time half of the prefix rule: check the new names against each
    # other AND against every sweep name the tracked configs already carry.
    existing = []
    for f in CONFIG_ROOT.rglob("*.yaml"):
        doc = yaml.safe_load(f.read_text()) or {}
        if isinstance(doc, dict) and "sweep" in doc:
            nm = (doc["sweep"] or {}).get("name")
            if nm and nm not in emitted:
                existing.append(nm)
    require_no_prefix_collisions(sorted(emitted) + existing)

    for name in sorted(emitted):
        out, text = emitted[name]
        status = "unchanged" if out.exists() and out.read_text() == text else (
            "update" if out.exists() else "new")
        print(f"  {status:9s} {out.relative_to(REPO)}")
        if not args.check and status != "unchanged":
            out.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(out, text)

    print(f"\n{len(emitted)} configs {'checked' if args.check else 'written'}"
          f" ({len(SITES)} sites x {len(ARCHS)} archs x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
