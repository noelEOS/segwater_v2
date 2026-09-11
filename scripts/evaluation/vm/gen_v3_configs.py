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

Seven architectures, not two. Swin-B and ConvNeXtV2-B were evaluated on GCP;
the other five (UNet, UNet++, DeepLabV3+, SegFormer, DPT) finished stage-2 on
2026-09-08/09 and have never been evaluated. ``--archs new5`` emits only those.

⚠️ **The decoder arch is per-architecture.** Only the Swin and ConvNeXt arms are
UPerNet. The ``model.arch`` token in each emitted config must match
``SegmentationModelFactory.build`` (src/models/factory.py) or the state_dict will
not load. Every arch/encoder pair in ``ARCHS`` is read from the driver that
trained that arm -- ``campaign_drivers/v3/run_stage2_arch.sh`` and
``run_stage2_unet.sh`` -- because the checkpoints carry no architecture metadata
(only step/model_state_dict/optimizer_state_dict/val_miou). The run directory
name plus those drivers are the entire record.

Two host layouts, since the campaign moved from a GCP VM to a RunPod box:
``--host gcp`` (default, ``/home/noel/...`` + repo-relative checkpoints)
reproduces the original campaign; ``--host runpod`` emits ``/workspace/...``.

Usage::

    python scripts/evaluation/vm/gen_v3_configs.py --check   # diff, write nothing
    python scripts/evaluation/vm/gen_v3_configs.py           # write the configs

    # the five new architectures, for the RunPod box
    python scripts/evaluation/vm/gen_v3_configs.py --host runpod --archs new5

    # Isolated dense-stride performance experiment; default configs unchanged.
    python scripts/evaluation/vm/gen_v3_configs.py --host runpod --sites demak \\
        --archs unet --stride 8 --cache-scene --batch-accumulation
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

# --- Host layout -------------------------------------------------------------
# The GCP eval VM and the RunPod box disagree on where data and checkpoints live,
# so neither is baked in. Defaults reproduce the GCP campaign byte-for-byte; the
# RunPod layout is selected with --host runpod (see HOSTS).
#
#   data_root   : parent of data_demak/, data_demak_concurrent/, Inference_input/
#   ckpt_root   : filesystem dir holding <arch>/<run>/*_last.pth (for discovery)
#   ckpt_prefix : how that same dir is SPELLED inside the config, which
#                 run_inference resolves relative to the repo root on GCP but
#                 needs absolute on the pod
HOSTS = {
    "gcp": {
        "data_root": "/home/noel",
        "ckpt_root": REPO / "outputs" / "stage2_v3",
        "ckpt_prefix": "outputs/stage2_v3",
        # Spelled exactly as the completed 2026-09-06/08 campaign ran it. Do not
        # "fix" this to match the pod: these configs are that campaign's record.
        "hamp24_dir": "/home/noel/hampyeong_ron_134_ts_16_sn_15_all24",
    },
    # The 2026-09-11 stride-8 fleet: four spot VMs across four GCP projects,
    # staged by rclone into ~/inputs and ~/ckpts rather than the 2026-09 eval
    # VM's flat ~/ layout. Same $HOME, different tree.
    "gcp_fleet": {
        "data_root": "/home/noel/inputs",
        "ckpt_root": Path("/home/noel/ckpts"),
        "ckpt_prefix": "/home/noel/ckpts",
        "hamp24_dir": "/home/noel/inputs/Inference_input/hampyeong_gee_frame_scenes_24",
    },
    "runpod": {
        "data_root": "/workspace/inputs",
        "ckpt_root": Path("/workspace/ckpts"),
        "ckpt_prefix": "/workspace/ckpts",
        # The 24 GEE re-downloads (3810x3131), uploaded 2026-09-10 and verified
        # byte-identical to the Mac. NOT the pack's 6-scene
        # Inference_input/hampyeong_ron_134_ts_16_sn_15.
        "hamp24_dir": "/workspace/inputs/Inference_input/hampyeong_gee_frame_scenes_24",
    },
}

DATA_ROOT = HOSTS["gcp"]["data_root"]
CKPT_ROOT = HOSTS["gcp"]["ckpt_root"]
CKPT_PREFIX = HOSTS["gcp"]["ckpt_prefix"]
HAMP24_DIR = HOSTS["gcp"]["hamp24_dir"]

# Tiling stride. 32 = operational (tile 224 -> 7x overlap per axis); the
# manuscript reports s32 throughout and s8 is the Supplementary S11
# compute-for-precision option. A coarser stride trades stitching quality for
# speed: 112 gives 2x overlap and ~12x fewer tiles per scene.
STRIDE = 32
CACHE_SCENE = False
BATCH_ACCUMULATION = False

SEEDS = ["s19", "s42", "s58"]

# arch dir token -> (decoder arch, encoder_name, short slug for the sweep name).
# The run dir under outputs/stage2_v3/<arch>/ is discovered, never hardcoded: it
# carries a wall-clock stamp that is not derivable from the seed.
#
# ⚠️ The decoder arch is NOT "upernet" for all seven. Only the two ViT/ConvNeXt
# encoders use UPerNet; the other five are their own decoders, and the token
# must match SegmentationModelFactory.build (src/models/factory.py) exactly or
# the state_dict will not load.
#
# Every triple below is read from the driver that actually trained the arm --
# campaign_drivers/v3/run_stage2_arch.sh (segformer/unetpp/dpt/deeplab) and
# run_stage2_unet.sh (unet). The checkpoints themselves carry no architecture
# metadata (only step/model_state_dict/optimizer_state_dict/val_miou), so the
# directory name plus these drivers are the sole record.
ARCHS = {
    "upernet_tu-swin_base_patch4_window7_224": (
        "upernet", "tu-swin_base_patch4_window7_224", "swinb"),
    "upernet_tu-convnextv2_base": (
        "upernet", "tu-convnextv2_base", "cnxb"),
    "unet_resnet50": (
        "unet", "resnet50", "unet"),
    "unetplusplus_resnet50": (
        "unetplusplus", "resnet50", "unetpp"),
    "deeplabv3plus_resnet50": (
        "deeplabv3plus", "resnet50", "deeplab"),
    "segformer_mit_b4": (
        "segformer", "mit_b4", "segformer"),
    "dpt_tu-vit_base_patch16_224.mae": (
        "dpt", "tu-vit_base_patch16_224.mae", "dpt"),
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
        "input_dir": "{data_root}/data_demak_concurrent",
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
        "input_dir": "{hamp24_dir}",
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
        "input_dir": "{data_root}/data_demak",
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
            "input_dir": f"{{data_root}}/Inference_input/{src}",
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


# Stage-2 run directory stamps, read off the checkpoint tree on 2026-09-10
# (`rclone check` verified against gdrive:segwater_v2_dataset_v3_results/stage2_v3/,
# 21/21 matching). Wall-clock stamps cannot be derived from the seed, so this is
# the record used when generating on a host without the checkpoints. When the tree
# IS present it is discovered instead and this table is not consulted.
RUN_DIRS = {
    "deeplabv3plus_resnet50": {
        "s19": "16-30-22_s19",
        "s42": "13-37-44_s42",
        "s58": "19-15-06_s58",
    },
    "dpt_tu-vit_base_patch16_224.mae": {
        "s19": "21-12-20_s19",
        "s42": "15-57-04_s42",
        "s58": "02-23-11_s58",
    },
    "segformer_mit_b4": {
        "s19": "02-34-58_s19",
        "s42": "21-59-42_s42",
        "s58": "07-01-33_s58",
    },
    "unet_resnet50": {
        "s19": "09-32-39_s19",
        "s42": "06-33-13_s42",
        "s58": "12-32-14_s58",
    },
    "unetplusplus_resnet50": {
        "s19": "22-27-55_s19",
        "s42": "15-33-00_s42",
        "s58": "05-20-17_s58",
    },
    "upernet_tu-convnextv2_base": {
        "s19": "01-34-44_s19",
        "s42": "18-50-35_s42",
        "s58": "06-37-19_s58",
    },
    "upernet_tu-swin_base_patch4_window7_224": {
        "s19": "09-00-42_s19",
        "s42": "04-01-53_s42",
        "s58": "13-54-57_s58",
    },
}


# The five architectures trained 2026-09-08/09 that have never been evaluated.
# Swin-B and ConvNeXtV2-B were evaluated on GCP and are excluded by this token.
NEW5 = [
    "unet_resnet50",
    "unetplusplus_resnet50",
    "deeplabv3plus_resnet50",
    "segformer_mit_b4",
    "dpt_tu-vit_base_patch16_224.mae",
]


SDS_SITES = ["narrabeen", "duck", "torreypines", "trucvert"]


def select_sites(spec: str) -> list[str]:
    """Resolve --sites to site keys. 'sds' expands to the four SDS frames.

    Raises on an unknown key rather than silently emitting a smaller campaign.
    """
    if spec == "all":
        return list(SITES)
    out = []
    for raw in spec.split(","):
        name = raw.strip()
        if name == "sds":
            out.extend(SDS_SITES)
        elif name in SITES:
            out.append(name)
        else:
            raise SystemExit(
                f"unknown site {name!r}; choose from {sorted(SITES)}, "
                f"or 'sds' for the four frames, or 'all'"
            )
    seen = set()
    return [s for s in out if not (s in seen or seen.add(s))]


def select_archs(spec: str) -> list[str]:
    """Resolve --archs to arch dir tokens, accepting slugs as an alias.

    Raises on an unknown name rather than silently emitting a smaller campaign:
    a typo here means missing arms nobody notices until the ranking table is
    short a row.
    """
    if spec == "all":
        return list(ARCHS)
    if spec == "new5":
        return list(NEW5)
    by_slug = {slug: token for token, (_, _, slug) in ARCHS.items()}
    out = []
    for raw in spec.split(","):
        name = raw.strip()
        if name in ARCHS:
            out.append(name)
        elif name in by_slug:
            out.append(by_slug[name])
        else:
            raise SystemExit(
                f"unknown arch {name!r}; choose from dir tokens {sorted(ARCHS)} "
                f"or slugs {sorted(by_slug)}, or 'all'/'new5'"
            )
    return out


def find_run_dir(arch: str, seed: str) -> str:
    """The single stage-2 run dir for one arch/seed, by its ``_s<seed>`` suffix.

    Raises rather than picking when the directory is present but ambiguous: a
    wrong checkpoint here mis-scores the whole campaign silently.

    When the checkpoint tree is absent -- generating on the Mac, where no
    checkpoints live -- falls back to RUN_DIRS, the recorded stamps. That is a
    lookup of a known answer, not a guess: the stamp is wall-clock and cannot be
    derived, so an unrecorded arch/seed is an error, never an invention.
    """
    root = CKPT_ROOT / arch
    if root.is_dir():
        hits = sorted(p for p in root.glob(f"*_{seed}") if p.is_dir())
        if len(hits) != 1:
            raise SystemExit(
                f"expected exactly one run dir for {arch}/{seed} under {root}, found {len(hits)}"
                + (f": {[h.name for h in hits]}" if hits else "")
            )
        return hits[0].name
    try:
        return RUN_DIRS[arch][seed]
    except KeyError:
        raise SystemExit(
            f"{root} does not exist and no run dir is recorded for {arch}/{seed}; "
            f"generate on a host that has the checkpoints, or add the stamp to RUN_DIRS"
        ) from None


def checkpoint_path(arch: str, seed: str, run: str) -> str:
    """The deployed arm: the run's single ``*_last.pth``.

    Path is relative to the repo root, matching how every other sweep config
    spells a checkpoint. Existence is checked when generating on the VM; the
    Mac has no checkpoints, so absence there is not an error.
    """
    rel = f"{CKPT_PREFIX}/{arch}/{run}"
    local = CKPT_ROOT / arch / run
    if local.is_dir():
        lasts = sorted(local.glob("*_last.pth"))
        if len(lasts) != 1:
            raise SystemExit(f"expected one *_last.pth in {rel}, found {len(lasts)}")
        return f"{rel}/{lasts[0].name}"
    # Off-VM: the builder's own naming contract.
    return f"{rel}/{arch}_{seed}_step36140_last.pth"


def build_config(site_key: str, arch: str, seed: str) -> tuple[str, dict]:
    site = SITES[site_key]
    decoder, encoder, slug = ARCHS[arch]
    run = find_run_dir(arch, seed)
    # Stride 32 is the operational default and its sweep names are unsuffixed --
    # 42 tracked configs and every scored v3 result depend on that spelling.
    #
    # A non-default stride is tagged BEFORE the seed, not after. A trailing tag
    # makes the operational name a strict prefix of the variant
    # ("..._s19" prefixes "..._s19_stride112"), and run dirs are selected by
    # glob: `*_s19_*` would then match both strides and silently pool them.
    # require_no_prefix_collisions() rejects that shape, by design.
    stride_tag = "" if STRIDE == 32 else f"_stride{STRIDE}"
    optimization_tag = ("_cache" if CACHE_SCENE else "") + ("_batchstitch" if BATCH_ACCUMULATION else "")
    sweep_name = f"{site['name']}_{slug}{stride_tag}{optimization_tag}_{seed}"

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
    if CACHE_SCENE:
        overrides["inference.data.cache_scene"] = True
    if BATCH_ACCUMULATION:
        overrides["inference.stitching.batch_accumulation"] = True

    doc = {
        "sweep": {
            "name": sweep_name,
            "dry_run": False,
            "continue_on_error": True,
            "input_dir": site["input_dir"].format(
                data_root=DATA_ROOT, hamp24_dir=HAMP24_DIR),
            "input_glob": site["input_glob"],
            "common_overrides": overrides,
            "checkpoints": [
                {
                    "name": f"{arch}_{seed}",
                    "checkpoint_path": checkpoint_path(arch, seed, run),
                    "model": {"arch": decoder, "encoder_name": encoder},
                }
            ],
            "presets": [
                {
                    "name": f"native224_weighted_224_b0_s{STRIDE}",
                    "overrides": {
                        "inference.data.tile_size": 224,
                        "inference.data.buffer_size": 0,
                        "inference.data.stride": STRIDE,
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
    ap.add_argument("--cache-scene", action="store_true", help="opt in to preprocessing each scene once in RAM")
    ap.add_argument("--batch-accumulation", action="store_true", help="opt in to ordered Triton weighted blending")
    ap.add_argument("--host", choices=sorted(HOSTS), default="gcp",
                    help="filesystem layout to emit paths for (default: gcp)")
    ap.add_argument("--stride", type=int, default=32,
                    help="tiling stride (default 32, the operational value). A "
                         "non-32 stride suffixes every sweep name with _stride<N> "
                         "so it cannot collide with the operational configs.")
    ap.add_argument("--sites", default="all",
                    help="comma-separated site keys, or 'all' (default). "
                         "e.g. hampyeong,demak,sds  ('sds' expands to the 4 frames)")
    ap.add_argument("--archs", default="all",
                    help="comma-separated arch dir tokens or slugs, or 'all' (default), "
                         "or 'new5' for the five never-evaluated architectures")
    args = ap.parse_args()

    global DATA_ROOT, CKPT_ROOT, CKPT_PREFIX, HAMP24_DIR, STRIDE, CACHE_SCENE, BATCH_ACCUMULATION
    CACHE_SCENE = args.cache_scene
    BATCH_ACCUMULATION = args.batch_accumulation
    STRIDE = args.stride
    if STRIDE < 1 or STRIDE > 224:
        raise SystemExit(f"--stride {STRIDE} outside 1..224 (tile_size is 224)")
    host = HOSTS[args.host]
    DATA_ROOT = host["data_root"]
    CKPT_ROOT = host["ckpt_root"]
    CKPT_PREFIX = host["ckpt_prefix"]
    HAMP24_DIR = host["hamp24_dir"]

    selected = select_archs(args.archs)
    selected_sites = select_sites(args.sites)

    assert_base_config_is_v3()

    emitted: dict[str, tuple[Path, str]] = {}
    for site_key in selected_sites:
        site = SITES[site_key]
        for arch in selected:
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
          f" ({len(selected_sites)} sites x {len(selected)} archs x {len(SEEDS)} seeds)"
          f"  stride={STRIDE}"
          f"  host={args.host}  data_root={DATA_ROOT}  ckpt_prefix={CKPT_PREFIX}")


if __name__ == "__main__":
    main()
