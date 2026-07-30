"""DEPRECATED (2026-07-30) — superseded by ``ship/gen_ship_configs.py``.

Kept as the **executable record** of the tracked `specs/hampyeong_gate.yaml`
(9 entries: 3 seeds x best/last/swa5). Do not emit NEW specs with it; the ship
campaign's spec (`ship/spec_hampyeong_ship.yaml`) is hand-written and covered by
`tests/test_spec_schema.py`.

What its local `resolve()` (below) lacks that `gen_ship_configs.py` has via
``ckptsel``:
  * **Returns None instead of raising**, collapsing "missing" and "ambiguous"
    (`c[0] if len(c) == 1 else None`) into one silent outcome with no candidate
    list. `ckptsel.resolve_last` raises `CkptSelError` naming the candidates.
  * **No `_last.pth` guard on the `best` arm** — `best.pth` is `.resolve()`d but
    its target is never checked, so a `best.pth` pointing at a `*_last.pth`
    would emit a spec whose `early_best` and `save_last` entries name the SAME
    checkpoint. The scorer's own distinctness audit would then be the only thing
    catching it, at score time rather than generation time.
    `ckptsel.resolve_best` refuses it outright.
  * **No seed-token check** (`ckptsel.require_seed_token`) and **no sha256
    distinctness check** (`ckptsel.assert_distinct_weights`) across the 9 arms.
  * **No prefix-collision check** on the `glob:` run-dir patterns it writes into
    the spec (`naming.require_no_prefix_collisions`); the anchoring in
    `runsel.resolve_glob_spec` is what protects those at score time.

Original docstring: emit the Hampyeong gate scoring spec (9 entries: 3 seeds x
best/last/swa5).
"""
from pathlib import Path
import sys

STAGE2 = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
SEEDS = ["s19", "s42", "s58"]
ARMS = [("best", "early_best"), ("last", "save_last"), ("swa5", "swa5")]
OUT = Path("/home/noel/configs/hampyeong_gate/spec_hampyeong_gate.yaml")

HEAD = """\
# Hampyeong gate — Swin-B three-seed checkpoint-selection comparison.
# 9 entries = 3 seeds x {early_best, save_last, swa5}. variant_column keeps the
# three arms distinguishable per seed. The scorer's distinctness + provenance
# audit enforces that all 9 really are different checkpoints.
repo: /home/noel/segwater_v2
nas_root: /home/noel/ancillary/hampyeong/nas_root
pair_root: outputs/inference
expected_valid_pixels: 1179967
variant_column: true
output: /home/noel/hampyeong_gate_per_date_metrics.csv

entries:
"""

ENTRY = """\
  - label: Swin-B
    slug: swin_b
    seed: {seedn}
    variant: {variant}
    run_dir: "glob:runs/gate_{arm}_hampyeong_{seed}_*_upernet_swin_base_224_{arm}_native224_weighted_224_b0_s32"
    checkpoint: {ckpt}
"""


def resolve(seed_dir: Path, arm: str):
    if arm == "best":
        p = seed_dir / "best.pth"
        return p.resolve().relative_to(Path.cwd()) if p.exists() else None
    pat = "*_last.pth" if arm == "last" else "*_swa5_step*.pth"
    c = sorted(seed_dir.glob(pat))
    return c[0] if len(c) == 1 else None


def main() -> None:
    parts, missing = [HEAD], []
    for seed in SEEDS:
        for arm, variant in ARMS:
            ck = resolve(STAGE2 / seed, arm)
            if ck is None:
                missing.append("%s/%s" % (seed, arm))
                continue
            parts.append(ENTRY.format(seedn=seed[1:], variant=variant,
                                      arm=arm, seed=seed, ckpt=str(ck)))
    if missing:
        print("MISSING: %s" % ", ".join(missing))
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts))
    print("Wrote %s (%d entries)" % (OUT, len(parts) - 1))
    print("".join(parts))


if __name__ == "__main__":
    main()
