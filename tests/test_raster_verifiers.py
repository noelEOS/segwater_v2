"""Contract tests for the two raster/result verifier scripts in vm/analysis/.

Both scripts exist to catch silently-wrong comparisons, so their plumbing --
run-dir resolution, cell keying, duplicate detection -- is what needs pinning.
Stdlib + pytest only: no rasterio, no torch, no VM. The rasterio import in
``perf_raster_diff`` lives inside the function that reads rasters precisely so
these pure helpers stay importable here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VM = REPO / "scripts" / "evaluation" / "vm"
sys.path.insert(0, str(VM))
sys.path.insert(0, str(VM / "analysis"))

from runsel import RunDirError  # noqa: E402


def _load(name: str):
    """Import a vm/analysis script by path (the dir is not a package)."""
    path = VM / "analysis" / (name + ".py")
    spec = importlib.util.spec_from_file_location("t_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prd = _load("perf_raster_diff")
vre = _load("verify_rerun_equality")


def mkdirs(root: Path, names):
    for n in names:
        (root / n).mkdir(parents=True)
    return root


# --------------------------------------------------------------------------- #
# perf_raster_diff: run-dir resolution
# --------------------------------------------------------------------------- #
BASE = "demak_perf_base_20260729T010101Z_ckpt_native224_weighted_224_b0_s32"
# `demak_perf_base` is a prefix of `demak_perf_base_PERF`: the bug class.
PERF = "demak_perf_base_PERF_20260729T020202Z_ckpt_native224_weighted_224_b0_s32"


def test_baseline_prefix_does_not_swallow_its_perf_sibling(tmp_path):
    mkdirs(tmp_path, [BASE, PERF])

    b_dirs, p_dirs, problem = prd.resolve_pair(
        tmp_path, "demak_perf_base", "demak_perf_base_PERF"
    )
    assert problem is None
    assert [Path(d).name for d in b_dirs] == [BASE]
    assert [Path(d).name for d in p_dirs] == [PERF]


def test_bare_prefix_glob_would_have_matched_both(tmp_path):
    """Pin the premise the anchoring defends against."""
    mkdirs(tmp_path, [BASE, PERF])
    assert len(sorted(tmp_path.glob("demak_perf_base_*"))) == 2


def test_timestamp_pinning_selects_one_of_several_baselines(tmp_path):
    dirs = [
        "demak_perf_base_20260729T010101Z_ckpt_native224_weighted_224_b0_s32",
        "demak_perf_base_20260730T030303Z_ckpt_native224_weighted_224_b0_s32",
    ]
    mkdirs(tmp_path, dirs + [PERF])

    b_all, _, _ = prd.resolve_pair(tmp_path, "demak_perf_base", "demak_perf_base_PERF")
    assert len(b_all) == 2, "premise: unpinned resolution is ambiguous here"
    assert prd.duplicate_strides(b_all), "both baselines share stride s32"

    b_pin, _, problem = prd.resolve_pair(
        tmp_path, "demak_perf_base", "demak_perf_base_PERF",
        base_ts="20260730T030303Z",
    )
    assert problem is None
    assert [Path(d).name for d in b_pin] == [dirs[1]]
    assert not prd.duplicate_strides(b_pin)


def test_missing_prefix_resolves_to_nothing(tmp_path):
    mkdirs(tmp_path, [BASE, PERF])
    b_dirs, _, problem = prd.resolve_pair(tmp_path, "nonexistent_sweep", "demak_perf_base_PERF")
    assert problem is None
    assert b_dirs == []


def test_dir_without_timestamp_is_not_a_candidate(tmp_path):
    """A dir violating the naming contract must not be compared."""
    mkdirs(tmp_path, ["demak_perf_base_notastamp_native224_b0_s32"])
    b_dirs, _, _ = prd.resolve_pair(tmp_path, "demak_perf_base", "demak_perf_base_PERF")
    assert b_dirs == []


def test_resolve_pair_reports_rundirerror_instead_of_raising(monkeypatch, tmp_path):
    """Ambiguity/missing must stay print-and-continue, not abort a batch."""
    def boom(*_a, **_k):
        raise RunDirError("synthetic resolution failure")

    monkeypatch.setattr(prd, "find", boom)
    b_dirs, p_dirs, problem = prd.resolve_pair(tmp_path, "a", "b")
    assert (b_dirs, p_dirs) == ([], [])
    assert "synthetic resolution failure" in problem


def test_runs_root_default_is_the_vm_path():
    assert prd.DEFAULT_RUNS_ROOT == "/home/noel/segwater_v2/outputs/inference/runs"


def test_self_comparison_raises_without_assert(monkeypatch, tmp_path):
    """`python -O` strips asserts; the self-comparison guard must survive it."""
    same = tmp_path / BASE
    same.mkdir()
    monkeypatch.setattr(prd, "resolve_pair",
                        lambda *a, **k: ([str(same)], [str(same)], None))
    monkeypatch.setattr(sys, "argv",
                        ["perf_raster_diff.py", "demak_perf_base", "demak_perf_base", "lbl"])
    with pytest.raises(SystemExit) as e:
        prd.main()
    assert "self-comparison" in str(e.value)


# --------------------------------------------------------------------------- #
# verify_rerun_equality: cell keying and duplicate detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dirname,expected", [
    ("narrabeen_s42aug_last_20260101T000000Z_ckpt_b0_s8_sweep",
     ("narrabeen", "aug", "last", "s8")),
    ("narrabeen_s42noaug_best_20260101T000000Z_ckpt_b0_s32_sweep",
     ("narrabeen", "no_aug", "best", "s32")),
    ("duck_s42_last_20260101T000000Z_ckpt_mid_bits_b0_s16_sweep",
     ("duck", "aug", "last", "s16")),
])
def test_key_parses_site_aug_arm_stride(dirname, expected):
    assert vre.key(dirname) == expected


@pytest.mark.parametrize("dirname", [
    "narrabeen_s42aug_last_20260101T000000Z_ckpt_b0_s8",       # no _sweep
    "torreypines_s42aug_last_20260101T000000Z_ckpt_b0_s8_sweep",  # other site
    "narrabeen_s42aug_swa5_20260101T000000Z_ckpt_b0_s8_sweep",  # unmodelled arm
    "narrabeen_s42aug_last_ckpt_b0_s8_sweep",                   # no timestamp
    "sweep",
])
def test_key_rejects_non_matching_names(dirname):
    assert vre.key(dirname) is None


def test_duplicate_cell_raises_naming_both_dirs(tmp_path):
    """Two dirs differing only in timestamp collapse to one cell -- refuse."""
    a = "narrabeen_s42aug_last_20260101T000000Z_ckpt_b0_s8_sweep"
    b = "narrabeen_s42aug_last_20260202T000000Z_ckpt_b0_s8_sweep"
    mkdirs(tmp_path, [a, b])

    with pytest.raises(SystemExit) as e:
        vre.collect(tmp_path, "narrabeen_s42*_sweep")
    msg = str(e.value)
    assert "duplicate cell" in msg
    assert a in msg and b in msg, "both colliding dirs must be named"


def test_collect_is_sorted_and_keyed(tmp_path):
    names = [
        "narrabeen_s42aug_last_20260101T000000Z_ckpt_b0_s8_sweep",
        "narrabeen_s42aug_best_20260101T000000Z_ckpt_b0_s8_sweep",
        "narrabeen_s42noaug_last_20260101T000000Z_ckpt_b0_s32_sweep",
        "not_a_sweep_dir",
    ]
    mkdirs(tmp_path, names)
    got = vre.collect(tmp_path, "narrabeen_s42*_sweep")
    assert set(got) == {
        ("narrabeen", "aug", "last", "s8"),
        ("narrabeen", "aug", "best", "s8"),
        ("narrabeen", "no_aug", "last", "s32"),
    }


def test_load_skips_dirs_without_metrics_csv(tmp_path):
    d = tmp_path / "narrabeen_s42aug_last_20260101T000000Z_ckpt_b0_s8_sweep"
    d.mkdir()
    assert vre.load(tmp_path, "narrabeen_s42*_sweep") == {}

    (d / "sweep_metrics.csv").write_text(
        "threshold,n,rmse,bias,std,q90,R2,n_shorelines\n0.5,10,1.0,0.1,2.0,3.0,0.9,82\n"
    )
    got = vre.load(tmp_path, "narrabeen_s42*_sweep")
    assert list(got) == [("narrabeen", "aug", "last", "s8")]
    assert got[("narrabeen", "aug", "last", "s8")][0.5]["n_shorelines"] == "82"


def test_root_defaults_preserved():
    a = vre.parse_args([])
    assert a.old_root == "/home/noel/_superseded_82_79"
    assert a.old_duck_subdir == "sds_vm_eval_duck_79"
    assert a.new_narrabeen_root == "/home/noel/sds_vm_eval"
    assert a.new_duck_root == "/home/noel/sds_vm_eval_duck"


def test_roots_are_overridable():
    a = vre.parse_args(["--old-root", "/x", "--new-duck-root", "/y"])
    assert a.old_root == "/x"
    assert a.new_duck_root == "/y"


# --------------------------------------------------------------------------- #
# aucroc scorer: the removed silent default (parse level only -- no sklearn here)
# --------------------------------------------------------------------------- #
AUCROC = REPO / "scripts" / "evaluate_indonesia_inference_run_aucroc.py"


def _aucroc_parse_args():
    """Extract the scorer's parse_args without importing sklearn/omegaconf.

    The module body imports sklearn, which the test env does not carry, so the
    function is compiled from source in a namespace holding only what it needs.
    """
    import argparse as _argparse
    import ast

    tree = ast.parse(AUCROC.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
    ns = {"argparse": _argparse, "Path": Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(AUCROC), "exec"), ns)
    return ns["parse_args"]


def test_aucroc_positional_and_flag_are_equivalent():
    parse_args = _aucroc_parse_args()
    assert parse_args(["cfg.yaml"]) == Path("cfg.yaml")
    assert parse_args(["--config", "cfg.yaml"]) == Path("cfg.yaml")


def test_aucroc_no_args_exits_2_and_explains_removed_fallback(capsys):
    parse_args = _aucroc_parse_args()
    with pytest.raises(SystemExit) as e:
        parse_args([])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "config is required" in err
    assert "removed deliberately" in err
