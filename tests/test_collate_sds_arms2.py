"""Regression tests for the SDS arm-collation plumbing (plan 2.4).

Every failure mode this script had was SILENT: an overlapping SPEC pair dropped
both arms with an innocuous-looking message, a multi-match labelled rows by coin
flip, an empty collection surfaced as ``pd.concat([])``'s opaque ValueError, and
the seed lived only in a format string. These tests pin the loud behaviour, and
pin the OUTPUT SCHEMA against the pre-hardening code so the hardening cannot
have moved a number or renamed a column.

Stdlib + pandas + pytest only -- no torch, no rasterio, no GPU, no VM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "evaluation" / "vm"))
sys.path.insert(0, str(REPO / "scripts" / "evaluation" / "vm" / "analysis"))

import collate_sds_arms2 as mod  # noqa: E402

# ---------------------------------------------------------------------------
# The output schema, derived by RUNNING the pre-hardening script on a fixture
# (git show HEAD:...collate_sds_arms2.py). Hard-coded here so a plumbing change
# that silently reorders or renames a column fails this test.
# ---------------------------------------------------------------------------
GOLDEN_COLUMNS = [
    "threshold", "n", "rmse", "bias", "std", "q90", "R2", "n_shorelines",
    "arch", "lineage", "arm", "model", "stride", "site", "reference", "seed",
    "run_dir",
]

#: Columns of a real sweep_metrics.csv
#: (experiments/model_ensemble_sds/NARRABEEN_ens_convnext_swin_s42/sweep_metrics.csv).
METRICS_HEADER = "threshold,n,rmse,bias,std,q90,R2,n_shorelines"
METRICS_ROW = "0.5,361,15.85,3.31,15.49,26.39,0.0223,87"

STAMP = "20260729T020404Z"


def make_sweep_dir(root: Path, base: str, *, metrics: bool = True) -> Path:
    """Create one ``<base>`` sweep dir, optionally holding a sweep_metrics.csv."""
    d = root / base
    d.mkdir(parents=True)
    if metrics:
        (d / "sweep_metrics.csv").write_text(METRICS_HEADER + "\n" + METRICS_ROW + "\n")
    return d


def sweep_name(key: str, stride: int, *, stamp: str = STAMP) -> str:
    """A realistic run-dir name carrying ``key``, a UTC stamp and a stride tail."""
    return "NARRABEEN_%s_%s_ckpt_native224_weighted_224_b0_s%d_sweep" % (
        key, stamp, stride)


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "sds_vm_eval"


def run(root, tmp_path, *extra):
    """Invoke ``main`` on ``root``, returning (exit_code_or_0, out_path)."""
    out = tmp_path / "collated.csv"
    rc = mod.main(["--roots", str(root), "--out", str(out), *extra])
    return rc, out


# ---------------------------------------------------------------------------
# resolve_spec
# ---------------------------------------------------------------------------

def test_exact_single_match_resolves():
    hit = mod.resolve_spec(sweep_name("mx630s2_s42_best", 32), mod.SPEC)
    assert hit == ("swinb", "mx630s2", "best", 42)


def test_zero_spec_matches_returns_none():
    assert mod.resolve_spec(sweep_name("some_other_campaign_s42_last", 32), mod.SPEC) is None


def test_multi_match_is_fatal_naming_both_keys():
    """Two arms claiming one dir must abort, not silently pick one."""
    spec = [("mx630k_s42_last", ("cnxv2t", "mx630k", "last", 42)),
            ("s42_last", ("swinb", "other", "last", 42))]
    base = sweep_name("mx630k_s42_last", 32)
    with pytest.raises(SystemExit) as ei:
        mod.resolve_spec(base, spec)
    msg = str(ei.value)
    assert "mx630k_s42_last" in msg and "'s42_last'" in msg and base in msg


# ---------------------------------------------------------------------------
# SPEC self-consistency checks (run at startup)
# ---------------------------------------------------------------------------

def test_shipped_spec_passes_both_checks():
    mod.check_spec_disjoint(mod.SPEC)
    mod.check_spec_seeds(mod.SPEC)


def test_overlapping_spec_keys_raise_at_load():
    spec = [("mx630k_s42_last", ("cnxv2t", "mx630k", "last", 42)),
            ("s42_last", ("swinb", "other", "last", 42))]
    with pytest.raises(mod.SpecError) as ei:
        mod.check_spec_disjoint(spec)
    msg = str(ei.value)
    assert "'s42_last'" in msg and "mx630k_s42_last" in msg


def test_seed_parsed_from_key_and_mismatch_raises():
    good = [("mx630k_s19_last", ("cnxv2t", "mx630k", "last", 19))]
    mod.check_spec_seeds(good)  # agreement: no raise

    typo = [("mx630k_s19_last", ("cnxv2t", "mx630k", "last", 42))]
    with pytest.raises(mod.SpecError) as ei:
        mod.check_spec_seeds(typo)
    assert "embeds seed 19" in str(ei.value) and "42" in str(ei.value)


def test_main_refuses_an_inconsistent_spec(root, tmp_path, monkeypatch):
    make_sweep_dir(root, sweep_name("mx630k_s42_last", 32))
    monkeypatch.setattr(mod, "SPEC",
                        [("mx630k_s42_last", ("cnxv2t", "mx630k", "last", 19))])
    with pytest.raises(mod.SpecError):
        run(root, tmp_path, "--expect-arms", "1")


# ---------------------------------------------------------------------------
# Skips: counted, non-fatal, reported on stderr
# ---------------------------------------------------------------------------

def test_dir_matching_no_spec_key_is_skipped_and_counted(root, tmp_path, capsys):
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    make_sweep_dir(root, sweep_name("unrelated_campaign_s42_last", 32))
    run(root, tmp_path, "--expect-arms", "1")
    err = capsys.readouterr().err
    assert "SKIP (0 spec matches)" in err
    assert "no-spec-match 1" in err


def test_missing_sweep_metrics_is_skipped_and_counted(root, tmp_path, capsys):
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    make_sweep_dir(root, sweep_name("mx630k_s42_last", 32), metrics=False)
    run(root, tmp_path, "--expect-arms", "1")
    err = capsys.readouterr().err
    assert "SKIP (no metrics)" in err
    assert "no-metrics 1" in err


def test_dir_without_utc_stamp_is_rejected(root, tmp_path, capsys):
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    # Hand-made / copied dir: right key, right stride tail, NO stamp.
    make_sweep_dir(root, "NARRABEEN_mx630k_s42_last_copy_224_b0_s32_sweep")
    rc, out = run(root, tmp_path, "--expect-arms", "1")
    err = capsys.readouterr().err
    assert "SKIP (no UTC stamp)" in err
    assert "no-stamp 1" in err
    # ...and it contributed no rows.
    assert set(pd.read_csv(out)["model"]) == {"swinb_s42_mx630s2_best"}


def test_missing_stride_tail_is_skipped_and_counted(root, tmp_path, capsys):
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    make_sweep_dir(root, "NARRABEEN_mx630k_s42_last_%s_ckpt_sweep" % STAMP)
    run(root, tmp_path, "--expect-arms", "1")
    err = capsys.readouterr().err
    assert "SKIP (no stride)" in err
    assert "no-stride 1" in err


def test_no_rows_raises_a_clear_message_not_pandas_valueerror(root, tmp_path):
    root.mkdir(parents=True)
    with pytest.raises(SystemExit) as ei:
        run(root, tmp_path)
    msg = str(ei.value)
    assert "no sweep_metrics.csv collected" in msg
    assert str(root) in msg


def test_nonexistent_roots_fail_cleanly(tmp_path):
    """Pass criterion 3: running against absent /home/noel roots on a Mac."""
    with pytest.raises(SystemExit) as ei:
        mod.main(["--roots", str(tmp_path / "nope1"), str(tmp_path / "nope2"),
                  "--out", str(tmp_path / "o.csv")])
    assert "no sweep_metrics.csv collected" in str(ei.value)


# ---------------------------------------------------------------------------
# Expected-arm assertion
# ---------------------------------------------------------------------------

def test_expect_arms_mismatch_raises_listing_found_arms(root, tmp_path):
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    make_sweep_dir(root, sweep_name("mx630k_s42_last", 32))
    with pytest.raises(SystemExit) as ei:
        run(root, tmp_path)  # default --expect-arms == len(SPEC) == 5
    msg = str(ei.value)
    assert "expected 5 distinct arm(s)" in msg and "found 2" in msg
    assert "swinb_s42_mx630s2_best" in msg and "cnxv2t_s42_mx630k_last" in msg
    assert "SKIP" in msg  # points the reader at the dropped-dir lines


def test_expect_arms_counts_distinct_models_not_dirs(root, tmp_path):
    """Two strides of one arm are ONE arm."""
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 8))
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    rc, out = run(root, tmp_path, "--expect-arms", "1")
    assert rc == 0
    assert sorted(pd.read_csv(out)["stride"]) == [8, 32]


# ---------------------------------------------------------------------------
# Happy path: schema + values pinned against the pre-hardening code
# ---------------------------------------------------------------------------

def test_successful_run_writes_pinned_schema_and_values(root, tmp_path):
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    make_sweep_dir(root, sweep_name("mx630k_s42_last", 8))
    rc, out = run(root, tmp_path, "--expect-arms", "2")
    assert rc == 0

    df = pd.read_csv(out)
    assert list(df.columns) == GOLDEN_COLUMNS
    assert len(df) == 2

    # Hand-computed expectation. Row order follows sorted(glob) per root, so the
    # cnxv2t dir ("NARRABEEN_mx630k_...") sorts before the swinb one.
    got = df.sort_values("model").reset_index(drop=True)
    assert list(got["model"]) == ["cnxv2t_s42_mx630k_last", "swinb_s42_mx630s2_best"]
    assert list(got["arch"]) == ["cnxv2t", "swinb"]
    assert list(got["lineage"]) == ["mx630k", "mx630s2"]
    assert list(got["arm"]) == ["last", "best"]
    assert list(got["stride"]) == [8, 32]
    assert list(got["seed"]) == [42, 42]
    assert list(got["site"]) == ["NARRABEEN", "NARRABEEN"]
    assert list(got["reference"]) == ["MSL", "MSL"]
    assert list(got["run_dir"]) == [sweep_name("mx630k_s42_last", 8),
                                    sweep_name("mx630s2_s42_best", 32)]
    # metric columns pass through untouched
    assert list(got["threshold"]) == [0.5, 0.5]
    assert list(got["n"]) == [361, 361]
    assert list(got["rmse"]) == [15.85, 15.85]
    assert list(got["bias"]) == [3.31, 3.31]
    assert list(got["std"]) == [15.49, 15.49]
    assert list(got["q90"]) == [26.39, 26.39]
    assert list(got["R2"]) == [0.0223, 0.0223]
    assert list(got["n_shorelines"]) == [87, 87]


def test_site_and_reference_flags_reach_the_columns(root, tmp_path):
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    rc, out = run(root, tmp_path, "--expect-arms", "1",
                  "--site", "DUCK", "--reference", "MHWS")
    df = pd.read_csv(out)
    assert list(df["site"]) == ["DUCK"] and list(df["reference"]) == ["MHWS"]
    # ...without disturbing the schema
    assert list(df.columns) == GOLDEN_COLUMNS


def test_multiple_roots_are_all_scanned(tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    make_sweep_dir(r1, sweep_name("mx630s2_s42_best", 32))
    make_sweep_dir(r2, sweep_name("mx630k_s42_last", 32))
    out = tmp_path / "o.csv"
    rc = mod.main(["--roots", str(r1), str(r2), "--out", str(out),
                   "--expect-arms", "2"])
    assert rc == 0
    assert len(pd.read_csv(out)) == 2


def test_no_tmp_file_left_behind(root, tmp_path):
    """atomic_to_csv writes via <path>.tmp and replaces it."""
    make_sweep_dir(root, sweep_name("mx630s2_s42_best", 32))
    rc, out = run(root, tmp_path, "--expect-arms", "1")
    assert out.exists()
    assert not out.with_name(out.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# Defaults preserved verbatim from the pre-hardening constants
# ---------------------------------------------------------------------------

def test_defaults_match_the_pre_hardening_hardcoded_values():
    a = mod.parse_args([])
    assert a.roots == ["/home/noel/sds_vm_eval_mx630s2",
                       "/home/noel/sds_vm_eval_mx630k",
                       "/home/noel/sds_vm_eval_mx630_arms2"]
    assert a.out == "/home/noel/sds_narrabeen_mx630_arms_msl.csv"
    assert a.site == "NARRABEEN" and a.reference == "MSL"
    assert a.expect_arms == len(mod.SPEC) == 5


def test_default_roots_list_is_not_mutated_by_argparse():
    """`default=list(DEFAULT_ROOTS)` keeps the module constant safe."""
    mod.parse_args([]).roots.append("/tmp/injected")
    assert mod.DEFAULT_ROOTS == ["/home/noel/sds_vm_eval_mx630s2",
                                 "/home/noel/sds_vm_eval_mx630k",
                                 "/home/noel/sds_vm_eval_mx630_arms2"]


def test_key_seed_regex_matches_every_shipped_key():
    for key, val in mod.SPEC:
        m = re.search(r"_s(\d+)_", key)
        assert m is not None and int(m.group(1)) == val[3]
