"""Re-derive the ship-campaign provenance manifest from the run dirs themselves.

This deliberately reads each run's ``run_config.yaml`` rather than trusting what
``gen_ship_configs.py`` intended to write. Construction and verification then
draw on independent sources, so a generator bug cannot certify itself.

Asserts, per run dir:
  * encoder is the expected Swin-B encoder;
  * the checkpoint path lives under ``outputs/mx630_stage2/<seed>/`` AND its
    filename carries ``_<seed>_`` (a file copied into the wrong seed dir passes
    the first check but not the second);
  * the variant matches the checkpoint kind (``last`` -> ``*_last.pth``,
    ``best`` -> a ``_pmwiou`` file that is not ``_last``);
  * all three bf16 compute flags are actually set -- a config key that never
    reached the code would leave the run silently fp32;
  * the stride recorded in the config matches the stride in the dir name;
  * scene count matches the gate's expectation (from ``completion.py``);
  * the dir name carries a UTC stamp (the manifest's ``stamp`` column).

Every one of those guards lives in the pure :func:`check_run_dir`, so
``tests/test_build_ship_manifest.py`` exercises them without a run dir.

Also emits a per-arm checkpoint sha256 so two arms cannot silently share weights.

Usage:
    python scripts/evaluation/vm/ship/build_ship_manifest.py [--out CSV]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from completion import count_probability_rasters, expected_scenes  # noqa: E402
from naming import atomic_write  # noqa: E402
from runsel import TIMESTAMP_PATTERN, RunDirError, resolve_run_dirs  # noqa: E402

REPO = Path.home() / "segwater_v2"
RUNS = REPO / "outputs/inference/runs"
SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]

# Defaults reproduce the original Swin-B `ship` campaign. ``--encoder`` /
# ``--ckpt-subdir`` / ``--tag`` override them for another lineage; ``ENCODER``
# stays exported because tests import it as the known-good value.
ENCODER = "tu-swin_base_patch4_window7_224"
DEFAULT_CKPT_SUBDIR = "mx630_stage2"
DEFAULT_TAG = "ship"
# arm-label prefix per encoder, so the manifest's `arm` column names the
# architecture rather than hardcoding "swinb" for every lineage.
ARM_PREFIX = {
    "tu-swin_base_patch4_window7_224": "swinb",
    "tu-convnextv2_base": "cnxb",
    "tu-convnextv2_tiny": "cnxt",
}

# gate -> (sweep-name template AFTER tag substitution, completion-table gate key,
# expected strides)
#
# The scene count is NOT inlined here: ``completion.EXPECTED_SCENES`` is the one
# source of truth for it, so a re-staged gate cannot leave this file disagreeing
# with the pollers and the scorers. Two gates share one key on purpose --
# ``demak_full_s112`` and ``demak_full_s32`` are the same 213-scene series run at
# two strides.
def gates_for_tag(tag: str) -> dict:
    """Sweep-name templates for one campaign tag.

    The tag sits in the MIDDLE of the name (``<gate>_<tag>_<seed>_<variant>``),
    matching ``gen_ship_configs.py``. Built per call rather than as a module
    constant so two lineages on one VM cannot read each other's run dirs.
    """
    return {
        "demak_gate": ("demak_gate_%s_%%s_%%s" % tag, "demak_gate", [32]),
        "hampyeong": ("hampyeong_%s_%%s_%%s" % tag, "hampyeong", [32]),
        "narrabeen": ("narrabeen_%s_%%s_%%s" % tag, "narrabeen", [8, 32, 112]),
        "demak_full_s112": ("demak_full_%s_%%s_%%s_s112" % tag, "demak_full", [112]),
        "demak_full_s32": ("demak_full_%s_%%s_%%s_s32" % tag, "demak_full", [32]),
    }


GATES = gates_for_tag(DEFAULT_TAG)

STRIDE_RE = re.compile(r"_b0_s(\d+)$")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cond, msg, problems):
    if not cond:
        problems.append(msg)
    return cond


def run_dir_stamp(run_dir_name: str) -> str:
    """The UTC stamp inside a run-dir name, or ``""`` if the name carries none.

    Searched for, never indexed: the old ``name.split("_")[-6][:16]`` assumed a
    fixed underscore count in the preset/checkpoint tail, so any preset with a
    different token count silently yielded a neighbouring token as the "stamp".
    """
    m = TIMESTAMP_PATTERN.search(run_dir_name)
    return m.group(0) if m else ""


def check_run_dir(gate, seed, variant, run_dir_name, cfg, n_tif, strides,
                  expected_scenes, encoder=ENCODER,
                  ckpt_subdir=DEFAULT_CKPT_SUBDIR) -> list[str]:
    """Every per-run-dir provenance guard, as a pure function.

    Takes the parsed ``run_config.yaml`` dict plus the counts main() measured;
    touches no filesystem. Returns the problem strings (empty list = all pass),
    so each guard is unit-testable without a run dir or a VM.

    ``encoder`` and ``ckpt_subdir`` default to the Swin-B lineage. ``ckpt_subdir``
    is the path segment the checkpoint must sit under, ABOVE the seed dir --
    ``mx630_stage2`` for Swin-B, ``mx630_stage2/upernet_tu-convnextv2_base`` for
    ConvNeXtV2-Base. Checking it matters because checkpoint filenames collide
    across lineages, so the path is the only thing distinguishing them.
    """
    problems: list[str] = []
    inf, mdl = cfg["inference"], cfg["model"]
    ck = inf["checkpoint_path"]
    enc = mdl["encoder_name"]
    stride = int(inf["data"]["stride"])
    comp = inf.get("compute", {}) or {}
    stitch = inf.get("stitching", {}) or {}
    m = STRIDE_RE.search(run_dir_name)
    dir_stride = int(m.group(1)) if m else -1

    tag = "%s/%s/%s/s%d" % (gate, seed, variant, stride)
    check(enc == encoder, "%s: encoder %s" % (tag, enc), problems)
    check("/%s/%s/" % (ckpt_subdir, seed) in ck,
          "%s: ckpt not under %s/%s: %s" % (tag, ckpt_subdir, seed, ck), problems)
    check("_%s_" % seed in Path(ck).name,
          "%s: ckpt filename lacks _%s_: %s" % (tag, seed, Path(ck).name), problems)
    if variant == "last":
        check(Path(ck).name.endswith("_last.pth"),
              "%s: variant=last but ckpt is %s" % (tag, Path(ck).name), problems)
    else:
        check("_pmwiou" in Path(ck).name and not Path(ck).name.endswith("_last.pth"),
              "%s: variant=best but ckpt is %s" % (tag, Path(ck).name), problems)
    check(str(comp.get("amp_dtype")) == "bfloat16",
          "%s: amp_dtype=%r" % (tag, comp.get("amp_dtype")), problems)
    check(comp.get("tf32") is True, "%s: tf32=%r" % (tag, comp.get("tf32")), problems)
    check(stitch.get("accumulate_on_device") is True,
          "%s: accumulate_on_device=%r" % (tag, stitch.get("accumulate_on_device")),
          problems)
    check(stride == dir_stride,
          "%s: config stride %d != dir stride %d" % (tag, stride, dir_stride), problems)
    check(stride in strides, "%s: unexpected stride %d" % (tag, stride), problems)
    check(n_tif == expected_scenes,
          "%s: %d probability tifs, expected %d" % (tag, n_tif, expected_scenes), problems)
    if not run_dir_stamp(run_dir_name):
        problems.append("%s: no UTC stamp in run dir name: %s" % (tag, run_dir_name))
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path.home() / "workspace/results/ship_decision_2026-07/MANIFEST.csv")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="campaign tag in the MIDDLE of every sweep name "
                         "(default: %(default)s)")
    ap.add_argument("--encoder", default=ENCODER,
                    help="encoder every run_config.yaml must declare "
                         "(default: %(default)s)")
    ap.add_argument("--ckpt-subdir", default=DEFAULT_CKPT_SUBDIR,
                    help="path segment above the seed dir that every checkpoint "
                         "must sit under (default: %(default)s)")
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=["best", "last", "swa5"],
                    help="arms to expect (default: %(default)s)")
    a = ap.parse_args()
    variants = list(a.variants)
    arm_prefix = ARM_PREFIX.get(a.encoder, a.encoder.replace("tu-", ""))
    n_expected_arms = len(SEEDS) * len(variants)

    rows, problems, digests = [], [], {}
    for gate, (tmpl, gate_key, strides) in gates_for_tag(a.tag).items():
        n_scenes = expected_scenes(gate_key)
        for seed in SEEDS:
            for variant in variants:
                name = tmpl % (seed, variant)
                try:
                    dirs = resolve_run_dirs(RUNS, name, expect=len(strides))
                except RunDirError as e:
                    problems.append("%s: %s" % (name, str(e).splitlines()[0]))
                    continue
                for d in dirs:
                    cfg = yaml.safe_load((d / "run_config.yaml").read_text())
                    inf, mdl = cfg["inference"], cfg["model"]
                    ck = inf["checkpoint_path"]
                    ckp = REPO / ck
                    enc = mdl["encoder_name"]
                    stride = int(inf["data"]["stride"])
                    comp = inf.get("compute", {}) or {}
                    stitch = inf.get("stitching", {}) or {}
                    n_tif = count_probability_rasters(d)

                    problems.extend(check_run_dir(
                        gate, seed, variant, d.name, cfg, n_tif, strides,
                        n_scenes, encoder=a.encoder, ckpt_subdir=a.ckpt_subdir))

                    key = "%s/%s" % (seed, variant)
                    if key not in digests:
                        digests[key] = sha256(ckp) if ckp.exists() else "MISSING"
                    rows.append(dict(
                        gate=gate, seed=seed, variant=variant,
                        arm="%s_%s_%s_%s" % (arm_prefix, seed, a.tag, variant),
                        stride=stride, run_dir=d.name, stamp=run_dir_stamp(d.name),
                        checkpoint=ck, ckpt_sha256=digests[key], encoder=enc,
                        amp_dtype=comp.get("amp_dtype"), tf32=comp.get("tf32"),
                        accumulate_on_device=stitch.get("accumulate_on_device"),
                        n_scenes=n_tif))

    # two arms must never share weights
    seen = {}
    for k, v in digests.items():
        if v in seen and v != "MISSING":
            problems.append("DUPLICATE WEIGHTS: %s and %s share sha256 %s" % (k, seen[v], v[:16]))
        seen[v] = k

    import csv
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with atomic_write(a.out, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("wrote %s (%d run dirs)" % (a.out, len(rows)))
    n_distinct = len({v for v in digests.values()})
    print("distinct checkpoints: %d (expect %d)" % (n_distinct, n_expected_arms))
    if n_distinct != n_expected_arms:
        problems.append("expected %d distinct checkpoints, found %d"
                        % (n_expected_arms, n_distinct))
    if problems:
        print("\n=== %d PROBLEM(S) ===" % len(problems))
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("=== ALL PROVENANCE CHECKS PASSED ===")


if __name__ == "__main__":
    main()
