"""Generate the mx630_stage2 SHIP-decision config matrix (6 arms x 4 gates).

Arms = 3 seeds (s19, s42, s58) x 2 checkpoint variants (best, last). Every run
uses bf16 + TF32 + device-stitch, including s42 -- which already had fp32
results -- so precision is uniform and the seed is the only variable.

NAMING (this is load-bearing, do not "tidy" it)
-----------------------------------------------
Sweep names are ``<gate>_ship_<seed>_<variant>[_<stride>]``.

* The campaign tag ``ship`` sits in the MIDDLE, never appended. Appending a tag
  to an existing name is exactly what produced the ``..._last`` vs
  ``..._last_PERF`` collision: a bare prefix glob then matches both. With the
  tag in the middle, every name in this campaign is disjoint from the legacy
  ``..._mx630s2_...`` dirs.
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
    python scripts/evaluation/vm/ship/gen_ship_configs.py [--out-root DIR]
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
ENCODER = "tu-swin_base_patch4_window7_224"
STAGE2 = Path.home() / "segwater_v2/outputs/mx630_stage2"
REPO = Path.home() / "segwater_v2"

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
# {gate} — mx630_stage2 SHIP decision, {seed} arm "{variant}".
# Swin-B (upernet / tu-swin_base_patch4_window7_224), canonical 2-stage on mx630.
# bf16 + TF32 + device-stitch (uniform across ALL 6 arms of this campaign, incl.
# s42, so precision is not confounded with seed).
# ⚠️ Lineage tag mx630_stage2. Checkpoint FILENAMES collide across lineages
#    (step23930_last also exists in mx630k / ConvNeXtV2-Tiny, different weights).
#    Never pool on filename or on `dataset=` alone.
"""


def resolve_ckpt(seed: str, variant: str) -> Path:
    """Resolve one arm's checkpoint via ckptsel, keeping this script's SystemExit
    contract (every failure here is a clean refusal before any GPU time)."""
    d = STAGE2 / seed
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
                 ckpt: Path, strides: list[int], sweep_name: str) -> None:
    g = GATES[gate]
    rel = ckpt.relative_to(REPO)
    lines = [HEADER.format(gate=gate.upper(), seed=seed, variant=variant), ""]
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
        '    - name: "mx630s2_%s_%s"' % (seed, variant),
        '      checkpoint_path: "%s"' % rel,
        "      model:",
        '        arch: "%s"' % ARCH,
        '        encoder_name: "%s"' % ENCODER,
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
    a = ap.parse_args()
    variants = list(a.variants)

    # --- resolve + audit every arm BEFORE writing anything --------------------
    # One dict, ONE key type ("<seed>/<variant>"). It used to be double-keyed with
    # both a tuple and a string per arm, which made len(arms) twice the arm count
    # and left the duplicate-weights message indexing the wrong key shape.
    # ckptsel.assert_distinct_weights now owns the digest/collision logic.
    arms = {}
    for seed in SEEDS:
        for variant in variants:
            arms["%s/%s" % (seed, variant)] = resolve_ckpt(seed, variant)
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
                        name = "%s_ship_%s_%s_s%d" % (gate, seed, variant, s)
                        write_config(a.out_root / gate / (name + ".yaml"),
                                     gate, seed, variant, ckpt, [s], name)
                        n += 1
                else:
                    name = "%s_ship_%s_%s" % (gate, seed, variant)
                    write_config(a.out_root / gate / (name + ".yaml"),
                                 gate, seed, variant, ckpt, g["strides"], name)
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
