"""Generate a SHIP-decision config matrix (3 seeds x N variants x 4 gates).

Arms = 3 seeds (s19, s42, s58) x checkpoint variants (default best, last). Every
run uses bf16 + TF32 + device-stitch so precision is uniform and the seed is the
only variable.

LINEAGE IS A PARAMETER, NOT A CONSTANT
--------------------------------------
``--lineage-root`` (the dir holding the per-seed subdirs), ``--encoder`` and
``--tag`` are flags. They used to be module constants pinned to Swin-B under
``outputs/mx630_stage2/<seed>/``, which meant a second lineage could not be run
without editing the script -- and an edited-in-place generator is exactly how two
lineages end up sharing sweep names.

Two lineages evaluated so far:

* Swin-B      ``outputs/mx630_stage2``                          tag ``ship``
* ConvNeXtV2-B ``outputs/mx630_stage2/upernet_tu-convnextv2_base`` tag ``cnxb``

⚠️ Checkpoint FILENAMES collide across lineages -- ``step23930_last.pth`` exists
in both with entirely different weights -- so the tag is the only thing keeping
the run dirs apart. Always pass a tag not already used on the VM.

NAMING (this is load-bearing, do not "tidy" it)
-----------------------------------------------
Sweep names are ``<gate>_<tag>_<seed>_<variant>[_<stride>]``.

* The campaign tag sits in the MIDDLE, never appended. Appending a tag
  to an existing name is exactly what produced the ``..._last`` vs
  ``..._last_PERF`` collision: a bare prefix glob then matches both. With the
  tag in the middle, every name in a campaign is disjoint from the legacy
  ``..._mx630s2_...`` dirs and from other campaigns' dirs.
* ``best``/``last`` are equal length and neither prefixes the other; seeds are
  fixed width. So no arm name can prefix another arm name.
* Demak-full is split into one sweep per stride so a crash in the ~23 min s32
  leg does not cost the ~4 min s112 leg.

CHECKPOINT RESOLUTION
---------------------
``best`` is resolved through the ``best.pth`` SYMLINK, not by globbing
``*_pmwiou*.pth`` -- each seed dir holds THREE pmwiou checkpoints (the HPO
selection kept several), so a glob is ambiguous and would silently pick by sort
order. ``last`` is the single ``*_last.pth``.

Guards, all of which run before any GPU time is spent:
  * the resolved filename must contain ``_<seed>_`` (catches a mis-copied file
    sitting in the wrong seed dir -- which would yield a plausible 6-row table
    that is really a 4-arm table);
  * ``best`` must not resolve to a ``_last.pth``;
  * all 6 checkpoints must have DISTINCT sha256. Two arms sharing weights is the
    one failure mode whose output is indistinguishable from success.

Usage:
    # Swin-B (the original campaign; these remain the defaults)
    python scripts/evaluation/vm/ship/gen_ship_configs.py [--out-root DIR]

    # ConvNeXtV2-Base
    python scripts/evaluation/vm/ship/gen_ship_configs.py \\
        --lineage-root ~/segwater_v2/outputs/mx630_stage2/upernet_tu-convnextv2_base \\
        --encoder tu-convnextv2_base --tag cnxb \\
        --out-root ~/configs/ship_decision_cnxb_2026-07
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Shared kit modules (stdlib-only, no editable install needed). Checkpoint
# resolution and the sha256/distinctness guard live in ckptsel (the local sha256
# helper is gone -- ckptsel.assert_distinct_weights hashes and compares in one
# call); the sweep-name prefix-collision rule lives in naming. One implementation,
# one test each.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ckptsel  # noqa: E402
from naming import NameCollisionError, require_no_prefix_collisions  # noqa: E402

SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]
ARCH = "upernet"
REPO = Path.home() / "segwater_v2"

# Defaults reproduce the original Swin-B `ship` campaign verbatim. Override all
# three together for another lineage -- an encoder that disagrees with the
# lineage root would produce runs that load the wrong architecture's weights.
DEFAULT_ENCODER = "tu-swin_base_patch4_window7_224"
DEFAULT_LINEAGE_ROOT = REPO / "outputs/mx630_stage2"
DEFAULT_TAG = "ship"

# bf16 + TF32 + device-stitch, spliced into every gate's common_overrides.
BF16DEV = [
    'inference.compute.amp_dtype: "bfloat16"',
    "inference.compute.tf32: true",
    "inference.stitching.accumulate_on_device: true",
]

# Per-gate post-processing. These genuinely differ and are copied verbatim from
# the configs that produced the existing s42 numbers -- do not homogenise them,
# or the new s42 rows stop being comparable to the old ones.
SMOOTH = [
    "inference.post_processing.smoothing.apply_simplification: true",
    "inference.post_processing.smoothing.simplify_tolerance_meters: 1.0",
]
EDGE = [
    'inference.data.edge_policy: "shift_inward"',
    "inference.stitching.min_weight: 0.001",
]

GATES = {
    "demak_gate": dict(
        input_dir="/home/noel/data_demak_concurrent",
        input_glob="S1_*.tif",
        strides=[32],
        split_by_stride=False,
        extra=SMOOTH + [
            "inference.post_processing.filtering.apply_length_filter: true",
            "inference.post_processing.filtering.min_length_meters: 10000.0",
        ] + EDGE,
    ),
    "hampyeong": dict(
        input_dir="/home/noel/Inference_input/hampyeong_ron_134_ts_16_sn_15",
        input_glob="S1B_*.tif",
        strides=[32],
        split_by_stride=False,
        extra=SMOOTH + [
            "inference.post_processing.filtering.apply_length_filter: false",
            "inference.post_processing.filtering.min_length_meters: 10000.0",
            "inference.post_processing.filtering.keep_top_k: 5",
        ] + EDGE,
    ),
    "narrabeen": dict(
        input_dir="/home/noel/NARRABEEN_ron_147_ts_9_sn_16",
        input_glob="*.tif",
        strides=[112, 32, 8],          # all three in ONE sweep
        split_by_stride=False,
        extra=SMOOTH + [
            "inference.post_processing.filtering.apply_length_filter: false",
            "inference.post_processing.filtering.min_length_meters: 10000.0",
            "inference.post_processing.filtering.keep_top_k: 999",
        ] + EDGE,
    ),
    "demak_full": dict(
        input_dir="/home/noel/data_demak",
        input_glob="S1_*.tif",
        strides=[112, 32],             # one sweep EACH (split_by_stride)
        split_by_stride=True,
        extra=SMOOTH + [
            "inference.post_processing.filtering.apply_length_filter: true",
            "inference.post_processing.filtering.min_length_meters: 10000.0",
        ] + EDGE + [
            "inference.data.num_workers: 8",
            "inference.data.batch_size: 256",
            'inference.output.probability_precision: "float16"',
            "inference.output.keep_probability_memmap: false",
        ],
    ),
}

HEADER = """\
# {gate} — SHIP decision campaign "{tag}", {seed} arm "{variant}".
# {arch} / {encoder}, canonical 2-stage on mx630.
# Lineage root: {lineage_root}
# bf16 + TF32 + device-stitch (uniform across ALL arms of this campaign, so
# precision is not confounded with seed).
# ⚠️ Checkpoint FILENAMES collide across lineages (step23930_last exists in
#    mx630k / ConvNeXtV2-Tiny, in mx630_stage2 / Swin-B, and in
#    mx630_stage2/upernet_tu-convnextv2_base / ConvNeXtV2-Base — all different
#    weights). Never pool on filename or on `dataset=` alone; carry the
#    architecture AND the campaign tag.
"""


def resolve_ckpt(lineage_root: Path, seed: str, variant: str) -> Path:
    """Resolve one arm's checkpoint via ckptsel, keeping this script's SystemExit
    contract (every failure here is a clean refusal before any GPU time)."""
    d = lineage_root / seed
    if not d.is_dir():
        raise SystemExit("missing seed dir: %s" % d)
    resolver = {"best": ckptsel.resolve_best,
                "last": ckptsel.resolve_last,
                "swa5": ckptsel.resolve_swa}[variant]
    try:
        p = resolver(d)
        ckptsel.require_seed_token(p, seed)
    except ckptsel.CkptSelError as exc:
        raise SystemExit("%s/%s: %s" % (seed, variant, exc)) from exc
    return p


def preset_block(stride: int) -> list[str]:
    return [
        '    - name: "native224_weighted_224_b0_s%d"' % stride,
        "      overrides:",
        "        inference.data.tile_size: 224",
        "        inference.data.buffer_size: 0",
        "        inference.data.stride: %d" % stride,
        '        inference.stitching.mode: "weighted_blend"',
        '        inference.stitching.blend_window: "hann"',
    ]


def write_config(path: Path, gate: str, seed: str, variant: str,
                 ckpt: Path, strides: list[int], sweep_name: str,
                 encoder: str, tag: str, lineage_root: Path) -> None:
    g = GATES[gate]
    rel = ckpt.relative_to(REPO)
    lines = [HEADER.format(gate=gate.upper(), seed=seed, variant=variant,
                           tag=tag, arch=ARCH, encoder=encoder,
                           lineage_root=lineage_root), ""]
    lines += [
        "sweep:",
        '  name: "%s"' % sweep_name,
        "  dry_run: false",
        "  continue_on_error: true",
        "",
        '  input_dir: "%s"' % g["input_dir"],
        '  input_glob: "%s"' % g["input_glob"],
        "",
        "  common_overrides:",
    ]
    lines += ["    " + o for o in g["extra"] + BF16DEV]
    lines += [
        "",
        "  checkpoints:",
        '    - name: "%s_%s_%s"' % (tag, seed, variant),
        '      checkpoint_path: "%s"' % rel,
        "      model:",
        '        arch: "%s"' % ARCH,
        '        encoder_name: "%s"' % encoder,
        "",
        "  presets:",
    ]
    for i, s in enumerate(strides):
        if i:
            lines.append("")
        lines += preset_block(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path,
                    default=Path.home() / "configs/ship_decision_2026-07")
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=["best", "last", "swa5"],
                    help="arms to emit (default: %(default)s). Pass a subset to "
                         "add an arm to a campaign whose other arms are already "
                         "run — regenerating them would mint new sweep names.")
    ap.add_argument("--lineage-root", type=Path, default=DEFAULT_LINEAGE_ROOT,
                    help="dir holding the per-seed checkpoint subdirs "
                         "(default: %(default)s)")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help="timm encoder name; MUST match --lineage-root's "
                         "architecture (default: %(default)s)")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="campaign tag, placed in the MIDDLE of every sweep "
                         "name. Must not already be in use on the VM — it is "
                         "the only thing separating this campaign's run dirs "
                         "from another lineage's (default: %(default)s)")
    a = ap.parse_args()
    variants = list(a.variants)
    lineage_root = a.lineage_root.expanduser()
    if not lineage_root.is_dir():
        raise SystemExit("missing lineage root: %s" % lineage_root)

    print("=== LINEAGE ===")
    print("  root    %s" % lineage_root)
    print("  arch    %s / %s" % (ARCH, a.encoder))
    print("  tag     %s   (sweep names: <gate>_%s_<seed>_<variant>)"
          % (a.tag, a.tag))

    # --- resolve + audit every arm BEFORE writing anything --------------------
    # One dict, ONE key type ("<seed>/<variant>"). It used to be double-keyed with
    # both a tuple and a string per arm, which made len(arms) twice the arm count
    # and left the duplicate-weights message indexing the wrong key shape.
    # ckptsel.assert_distinct_weights now owns the digest/collision logic.
    arms = {}
    for seed in SEEDS:
        for variant in variants:
            arms["%s/%s" % (seed, variant)] = resolve_ckpt(
                lineage_root, seed, variant)
    try:
        digests = ckptsel.assert_distinct_weights(arms)
    except ckptsel.CkptSelError as exc:
        raise SystemExit(str(exc)) from exc

    print("=== ARMS (%d) ===" % (len(arms)))
    for seed in SEEDS:
        for variant in variants:
            key = "%s/%s" % (seed, variant)
            print("  %-3s %-4s  %s  %s" % (seed, variant, digests[key][:16], arms[key].name))
    print("=== sha256 DISTINCTNESS: %d distinct / %d arms  OK ==="
          % (len(set(digests.values())), len(arms)))

    # --- write configs -------------------------------------------------------
    n = 0
    for gate, g in GATES.items():
        for seed in SEEDS:
            for variant in variants:
                ckpt = arms["%s/%s" % (seed, variant)]
                if g["split_by_stride"]:
                    for s in g["strides"]:
                        name = "%s_%s_%s_%s_s%d" % (gate, a.tag, seed, variant, s)
                        write_config(a.out_root / gate / (name + ".yaml"),
                                     gate, seed, variant, ckpt, [s], name,
                                     a.encoder, a.tag, lineage_root)
                        n += 1
                else:
                    name = "%s_%s_%s_%s" % (gate, a.tag, seed, variant)
                    write_config(a.out_root / gate / (name + ".yaml"),
                                 gate, seed, variant, ckpt, g["strides"], name,
                                 a.encoder, a.tag, lineage_root)
                    n += 1
    print("=== WROTE %d sweep configs under %s ===" % (n, a.out_root))

    # --- no sweep name may prefix another ------------------------------------
    # naming.require_no_prefix_collisions is the one implementation of this rule
    # (a bare `<name>_*` glob would match both members of a colliding pair). Its
    # message already lists every pair, so the old per-pair stderr loop is gone;
    # the exit-non-zero-naming-the-pairs behavior is preserved.
    names = sorted(p.stem for p in a.out_root.rglob("*.yaml"))
    try:
        require_no_prefix_collisions(names, what="sweep name")
    except NameCollisionError as exc:
        raise SystemExit(str(exc)) from exc
    print("=== NAME COLLISION CHECK: %d names, no prefixes  OK ===" % len(names))


if __name__ == "__main__":
    main()
