"""Regression tests for unambiguous run-dir resolution.

Covers the bug class that motivated ``scripts/evaluation/vm/runsel.py``: a
prefix glob matching a longer sibling run dir. Stdlib + pytest only -- no
torch, no rasterio, no GPU, no VM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "evaluation" / "vm"))

from runsel import (  # noqa: E402
    TIMESTAMP_RE,
    RunDirError,
    resolve_glob_spec,
    resolve_run_dir,
    resolve_run_dirs,
    run_dir_regex,
)

# The two dirs from the real incident: `mx630k` is a prefix of `mx630k_best`.
SIBLINGS = [
    "demak_full_mx630k_20260729T020404Z_mx630k_native224_weighted_224_b0_s32",
    "demak_full_mx630k_best_20260729T040713Z_mx630k_best_native224_weighted_224_b0_s32",
]

SPEC_DIRS = [
    REPO / "scripts" / "evaluation" / "vm" / "specs",
    REPO / "scripts" / "evaluation" / "vm" / "analysis" / "configs",
]


def mkdirs(root: Path, names) -> Path:
    for n in names:
        (root / n).mkdir(parents=True)
    return root


# --------------------------------------------------------------------------- #
# 1. the bug class itself
# --------------------------------------------------------------------------- #
def test_prefix_sibling_is_not_matched(tmp_path):
    """`mx630k` must resolve to its own dir, never to `mx630k_best`."""
    mkdirs(tmp_path, SIBLINGS)

    got = resolve_run_dir(tmp_path, "demak_full_mx630k")
    assert got.name == SIBLINGS[0]

    got = resolve_run_dir(tmp_path, "demak_full_mx630k_best")
    assert got.name == SIBLINGS[1]


def test_bare_prefix_glob_would_have_matched_both(tmp_path):
    """Pin the premise: the old approach really was ambiguous."""
    mkdirs(tmp_path, SIBLINGS)
    naive = sorted(p.name for p in tmp_path.glob("demak_full_mx630k_*"))
    assert len(naive) == 2, "premise broken: the bare glob is no longer ambiguous"


def test_duplicate_runs_of_same_name_raise_and_list_candidates(tmp_path):
    """Two runs of one sweep => refuse to guess, and show both names."""
    dupes = [
        "demak_gate_x_20260729T010101Z_a_native224_weighted_224_b0_s32",
        "demak_gate_x_20260729T020202Z_a_native224_weighted_224_b0_s32",
    ]
    mkdirs(tmp_path, dupes)
    with pytest.raises(RunDirError) as e:
        resolve_run_dir(tmp_path, "demak_gate_x")
    for d in dupes:
        assert d in str(e.value), "error message must list the candidates"


def test_timestamp_pin_disambiguates(tmp_path):
    mkdirs(tmp_path, [
        "demak_gate_x_20260729T010101Z_a_native224_weighted_224_b0_s32",
        "demak_gate_x_20260729T020202Z_a_native224_weighted_224_b0_s32",
    ])
    got = resolve_run_dir(tmp_path, "demak_gate_x", timestamp="20260729T020202Z")
    assert got.name.startswith("demak_gate_x_20260729T020202Z_")


def test_missing_run_raises(tmp_path):
    mkdirs(tmp_path, SIBLINGS)
    with pytest.raises(RunDirError):
        resolve_run_dir(tmp_path, "demak_full_nonexistent")


def test_name_without_timestamp_is_rejected(tmp_path):
    """A dir violating the naming contract is not a candidate."""
    mkdirs(tmp_path, ["demak_full_mx630k_notatimestamp_native224_b0_s32"])
    with pytest.raises(RunDirError):
        resolve_run_dir(tmp_path, "demak_full_mx630k")


# --------------------------------------------------------------------------- #
# 2. backward compatibility over the REAL spec patterns
# --------------------------------------------------------------------------- #
def collect_glob_patterns():
    """Every `glob:` run_dir value in the tracked spec YAMLs."""
    out = []
    for d in SPEC_DIRS:
        for f in sorted(d.glob("*.yaml")):
            try:
                spec = yaml.safe_load(f.read_text())
            except yaml.YAMLError:
                continue
            if not isinstance(spec, dict):
                continue
            for e in spec.get("entries") or []:
                rd = e.get("run_dir", "")
                if isinstance(rd, str) and rd.startswith("glob:"):
                    out.append((f.name, rd[len("glob:"):]))
    return out


PATTERNS = collect_glob_patterns()


def test_spec_patterns_were_found():
    assert PATTERNS, "no glob: patterns found -- spec layout changed?"


@pytest.mark.parametrize("srcfile,pattern", PATTERNS,
                         ids=[f"{s}:{i}" for i, (s, _) in enumerate(PATTERNS)])
def test_existing_spec_pattern_still_matches_its_target(tmp_path, srcfile, pattern):
    """Auto-anchoring must not change what an existing spec resolves to.

    Synthesise the dir the pattern intends to match by substituting a plausible
    timestamp for its `*`, then assert the anchored matcher still finds it.
    """
    stamp = "20260729T020404Z"
    head, star, rest = pattern.partition("*")
    assert star, "pattern has no wildcard: %r" % pattern
    # `rest` may start with `_s32` (bare tail) or `_upernet_...` (spelled out);
    # either way the timestamp goes where the `*` was.
    target = head + stamp + rest
    (tmp_path / target).mkdir(parents=True)

    hits = resolve_glob_spec(tmp_path, pattern)
    assert [p.relative_to(tmp_path).as_posix() for p in hits] == [target], (
        "pattern from %s no longer resolves to its target" % srcfile
    )


@pytest.mark.parametrize("srcfile,pattern", PATTERNS,
                         ids=[f"{s}:{i}" for i, (s, _) in enumerate(PATTERNS)])
def test_existing_spec_pattern_rejects_prefix_sibling(tmp_path, srcfile, pattern):
    """A sibling whose name extends this one must not be absorbed."""
    stamp = "20260729T020404Z"
    head, _, rest = pattern.partition("*")
    # `head` ends with "_"; append an extra name segment before the timestamp,
    # which is exactly the mx630k -> mx630k_best shape.
    decoy = head + "sibling_" + stamp + rest
    (tmp_path / decoy).mkdir(parents=True)

    hits = resolve_glob_spec(tmp_path, pattern)
    assert hits == [], (
        "pattern from %s absorbed a prefix sibling: %s" % (srcfile, decoy)
    )


def test_naive_anchoring_rule_would_break_a_real_pattern(tmp_path):
    """Guard the subtlety: `* -> <stamp>` (no trailing `(?:_.*)?`) is wrong.

    A single `*` may span the timestamp AND the checkpoint/preset middle. The
    naive substitution demands the stamp be followed immediately by the tail
    and matches nothing. This test fails if someone "simplifies" the rule.
    """
    pattern = "runs/hampyeong_mx630k_s42_last_*_s32"
    target = ("runs/hampyeong_mx630k_s42_last_20260729T020404Z"
              "_upernet_convnextv2_tiny_mx630k_last_native224_weighted_224_b0_s32")
    (tmp_path / target).mkdir(parents=True)

    assert len(resolve_glob_spec(tmp_path, pattern)) == 1

    naive = re.compile("^" + re.escape("runs/hampyeong_mx630k_s42_last_")
                       + TIMESTAMP_RE + re.escape("_s32") + "$")
    assert not naive.match(target), "the naive rule is no longer distinguishable"


# --------------------------------------------------------------------------- #
# 3. cardinality / multi-stride
# --------------------------------------------------------------------------- #
STRIDES = [
    "narrabeen_x_20260729T055717Z_swin_native224_weighted_224_b0_s8",
    "narrabeen_x_20260729T055717Z_swin_native224_weighted_224_b0_s32",
    "narrabeen_x_20260729T055717Z_swin_native224_weighted_224_b0_s112",
]


def test_multi_stride_expect_three(tmp_path):
    mkdirs(tmp_path, STRIDES)
    hits = resolve_run_dirs(tmp_path, "narrabeen_x", expect=3)
    assert [p.name for p in hits] == sorted(STRIDES)


def test_multi_stride_singular_raises_listing_all(tmp_path):
    mkdirs(tmp_path, STRIDES)
    with pytest.raises(RunDirError) as e:
        resolve_run_dir(tmp_path, "narrabeen_x")
    for d in STRIDES:
        assert d in str(e.value)


def test_suffix_narrows_to_one(tmp_path):
    mkdirs(tmp_path, STRIDES)
    got = resolve_run_dir(tmp_path, "narrabeen_x", suffix="_b0_s32")
    assert got.name.endswith("_b0_s32")


def test_suffix_separates_prefix_tokens(tmp_path):
    """`unet_resnet50` must not absorb `unetplusplus_resnet50`."""
    names = [
        "dev_sweep_rest_hampyeong_s58_20260729T010101Z_unet_resnet50_b0_s32",
        "dev_sweep_rest_hampyeong_s58_20260729T010101Z_unetplusplus_resnet50_b0_s32",
    ]
    mkdirs(tmp_path, names)
    got = resolve_run_dir(tmp_path, "dev_sweep_rest_hampyeong_s58",
                          suffix="_unet_resnet50_b0_s32")
    assert got.name == names[0]


def test_expect_none_accepts_any_count(tmp_path):
    mkdirs(tmp_path, STRIDES)
    assert len(resolve_run_dirs(tmp_path, "narrabeen_x")) == 3


# --------------------------------------------------------------------------- #
# 4. must survive `python -O`
# --------------------------------------------------------------------------- #
def test_raises_rundirerror_not_assertionerror(tmp_path):
    """`assert` is stripped under -O; the guard must be a real raise."""
    mkdirs(tmp_path, SIBLINGS)
    with pytest.raises(RunDirError):
        resolve_run_dir(tmp_path, "demak_full_absent")
    assert not issubclass(RunDirError, AssertionError)


def test_regex_shape():
    rx = run_dir_regex("demak_full_mx630k")
    assert rx.match("demak_full_mx630k_20260729T020404Z_a_b0_s32")
    assert not rx.match("demak_full_mx630k_best_20260729T020404Z_a_b0_s32")
    assert not rx.match("demak_full_mx630k_20260729T020404_a_b0_s32")  # no Z
