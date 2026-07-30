"""Contract tests for the run-completion helper (completion.py).

The completion contract exists because every other signal lies:
run_metadata.json never appears in run dirs, run_summary.json is rewritten per
scene, and continue_on_error lets a short sweep exit 0. Stdlib + pytest only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "evaluation" / "vm"))

from completion import (  # noqa: E402
    EXPECTED_SCENES,
    CompletionError,
    _main,
    completion_report,
    count_probability_rasters,
    expected_scenes,
    is_run_complete,
    require_run_complete,
)


def make_run_dir(root: Path, name: str, scenes, extra_files=()) -> Path:
    """Fabricate a run dir with per-scene subdirs, the real layout."""
    run = root / name
    for s in scenes:
        d = run / s
        d.mkdir(parents=True)
        (d / f"{s}_probability_water.tif").touch()
    for rel in extra_files:
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return run


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
def test_complete_run_counts_exactly(tmp_path):
    run = make_run_dir(tmp_path, "gate_run", [f"S1_{i}" for i in range(6)])
    assert count_probability_rasters(run) == 6
    assert is_run_complete(run, 6)
    assert require_run_complete(run, 6) == 6


def test_short_run_is_incomplete_and_raises(tmp_path):
    run = make_run_dir(tmp_path, "gate_run", [f"S1_{i}" for i in range(4)])
    assert not is_run_complete(run, 6)
    with pytest.raises(CompletionError) as e:
        require_run_complete(run, 6)
    assert "expected 6" in str(e.value) and "found 4" in str(e.value)


def test_overcomplete_is_an_error_not_complete(tmp_path):
    # Over-complete means a stale expectation or two sweeps in one out-dir.
    run = make_run_dir(tmp_path, "gate_run", [f"S1_{i}" for i in range(7)])
    assert not is_run_complete(run, 6)
    with pytest.raises(CompletionError):
        require_run_complete(run, 6)


def test_auxiliary_rasters_do_not_count(tmp_path):
    # Run dirs also hold confusion/frequency rasters etc.; only the
    # probability product counts (runbook: never count bare *.tif).
    run = make_run_dir(
        tmp_path, "r", ["S1_a"],
        extra_files=["S1_a/S1_a_confusion.tif", "S1_a/S1_a_water_mask.tif"],
    )
    assert count_probability_rasters(run) == 1


def test_appledouble_stub_is_not_counted(tmp_path):
    # rsync from macOS plants ._<name> stubs that DO match a suffix glob.
    run = make_run_dir(
        tmp_path, "r", ["S1_a"],
        extra_files=["S1_b/._S1_b_probability_water.tif"],
    )
    assert count_probability_rasters(run) == 1


def test_empty_run_dir(tmp_path):
    run = tmp_path / "empty_run"
    run.mkdir()
    assert count_probability_rasters(run) == 0
    assert not is_run_complete(run, 6)


# --------------------------------------------------------------------------- #
# the gate table
# --------------------------------------------------------------------------- #
def test_gate_table_values():
    # The decision-gating four; changing any of these is a campaign-level event.
    assert expected_scenes("demak_gate") == 6
    assert expected_scenes("hampyeong") == 6
    assert expected_scenes("narrabeen") == 87
    assert expected_scenes("demak_full") == 213


def test_unknown_gate_raises_and_lists_known():
    with pytest.raises(CompletionError) as e:
        expected_scenes("atlantis")
    msg = str(e.value)
    assert "atlantis" in msg
    for known in EXPECTED_SCENES:
        assert known in msg


# --------------------------------------------------------------------------- #
# report + CLI
# --------------------------------------------------------------------------- #
def test_completion_report_is_sorted_and_flags(tmp_path):
    b = make_run_dir(tmp_path, "b_run", ["S1_a", "S1_b"])
    a = make_run_dir(tmp_path, "a_run", ["S1_a"])
    rep = completion_report([b, a], 2)
    assert [r[0].name for r in rep] == ["a_run", "b_run"]
    assert rep[0][1:] == (1, False)
    assert rep[1][1:] == (2, True)


def test_cli_print_expected(capsys):
    assert _main(["--gate", "narrabeen", "--print-expected"]) == 0
    assert capsys.readouterr().out.strip() == "87"


def test_cli_check_exit_codes(tmp_path, capsys):
    ok = make_run_dir(tmp_path, "ok_run", [f"S1_{i}" for i in range(6)])
    short = make_run_dir(tmp_path, "short_run", ["S1_0"])
    assert _main(["--gate", "demak_gate", "--check", str(ok)]) == 0
    assert _main(["--gate", "demak_gate", "--check", str(ok), str(short)]) == 1
    out = capsys.readouterr().out
    assert "SHORT" in out and "1/6" in out


def test_cli_explicit_expected(tmp_path):
    run = make_run_dir(tmp_path, "r", ["S1_a", "S1_b", "S1_c"])
    assert _main(["--expected", "3", "--check", str(run)]) == 0


# --------------------------------------------------------------------------- #
# smoke over real local run dirs, when present
# --------------------------------------------------------------------------- #
def test_smoke_real_runs_dir():
    runs = REPO / "outputs" / "inference" / "runs"
    if not runs.is_dir():
        pytest.skip("no local outputs/inference/runs")
    for d in sorted(runs.iterdir())[:5]:
        if d.is_dir():
            assert count_probability_rasters(d) >= 0
