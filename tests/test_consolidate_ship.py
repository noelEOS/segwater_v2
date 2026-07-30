"""Regression tests for the ship-campaign consolidator's join and exit policy.

The bug class: the decision table is joined on ``(seed, variant)`` with
``"s19"``-style seed keys, but two of the four ingest loops compared the raw CSV
cell. A gate CSV writing its seed as a bare ``19`` matched nothing and left
``gate_iou`` -- the ship criterion -- all-NaN, with no warning and no non-zero
exit. These tests pin: (a) byte-identical outputs to the pre-change script for
inputs whose tokens were already correct, (b) seed-token robustness, (c)
unmatched-row accounting, (d) the exit policy.

Stdlib + pandas + pytest only -- no torch, no rasterio, no GPU, no VM.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
VM = REPO / "scripts" / "evaluation" / "vm"
sys.path.insert(0, str(VM))
sys.path.insert(0, str(VM / "ship"))

from consolidate_ship_results import _seed_key  # noqa: E402

SCRIPT = VM / "ship" / "consolidate_ship_results.py"
SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]


# --------------------------------------------------------------------------
# fixture builder: a complete --root with all four gate CSVs
# --------------------------------------------------------------------------
def _arms():
    """Deterministic (seed, variant) order matching the script's row order."""
    return [(s, v) for v in VARIANTS for s in SEEDS]


def write_gate(root: Path, seed_fmt=lambda s: s) -> Path:
    """demak_gate/demak_gate_ship_summary.csv -- one row per arm.

    ``seed_fmt`` maps the canonical ``"s19"`` token to whatever the CSV should
    literally contain, so a test can inject ``19`` / ``"19"`` / ``"s99"``.
    """
    p = root / "demak_gate" / "demak_gate_ship_summary.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (s, v) in enumerate(_arms()):
        recs.append({
            "seed": seed_fmt(s), "variant": v,
            "iou_at_0p5": 0.70 + 0.01 * i, "tau_star": 0.40 + 0.01 * i,
            "iou_at_tau_star": 0.72 + 0.01 * i, "area_bias": -0.05 + 0.01 * i,
            "roc_auc": 0.90 + 0.001 * i,
        })
    pd.DataFrame(recs).to_csv(p, index=False)
    return p


def write_hampyeong(root: Path, seed_fmt=lambda s: s) -> Path:
    """hampyeong/hampyeong_ship_per_date_metrics.csv -- 3 dates per arm."""
    p = root / "hampyeong" / "hampyeong_ship_per_date_metrics.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (s, v) in enumerate(_arms()):
        for d, date in enumerate(("2021-03-01", "2021-06-01", "2021-09-01")):
            recs.append({"seed": seed_fmt(s), "variant": v, "date": date,
                         "iou": 0.60 + 0.01 * i + 0.002 * d})
    pd.DataFrame(recs).to_csv(p, index=False)
    return p


def write_narrabeen(root: Path, seed_fmt=lambda s: s) -> Path:
    """narrabeen/sds_narrabeen_ship_msl.csv -- 3 strides x 2 thresholds per arm.

    The off-0.5 threshold rows exist so the script's threshold filter is
    exercised (they must not reach the table).
    """
    p = root / "narrabeen" / "sds_narrabeen_ship_msl.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (s, v) in enumerate(_arms()):
        for thr in (0.5, 0.6):
            for stride in (8, 32, 112):
                recs.append({
                    "seed": seed_fmt(s), "variant": v, "threshold": thr,
                    "stride": stride,
                    "rmse": 14.0 + 0.1 * i + 0.01 * stride + 10 * (thr - 0.5),
                    "bias": -1.0 + 0.1 * i, "std": 13.0 + 0.1 * i,
                })
    pd.DataFrame(recs).to_csv(p, index=False)
    return p


def write_trend(root: Path, stride: int, seed_fmt=lambda s: s) -> Path:
    """demak_trend/demak_full_ship_trend_s{stride}.csv -- one row per arm."""
    p = root / "demak_trend" / ("demak_full_ship_trend_s%d.csv" % stride)
    p.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (s, v) in enumerate(_arms()):
        recs.append({"seed": seed_fmt(s), "variant": v,
                     "slope_ha_yr": 300.0 + 5 * i + stride,
                     "hac_se": 30.0 + 0.5 * i})
    pd.DataFrame(recs).to_csv(p, index=False)
    return p


def build_root(root: Path, **fmts) -> Path:
    """Full synthetic campaign root. Per-source ``*_fmt`` kwargs inject tokens."""
    root.mkdir(parents=True, exist_ok=True)
    write_gate(root, fmts.get("gate_fmt", lambda s: s))
    write_hampyeong(root, fmts.get("hamp_fmt", lambda s: s))
    write_narrabeen(root, fmts.get("sds_fmt", lambda s: s))
    for stride in (32, 112):
        write_trend(root, stride, fmts.get("trend_fmt", lambda s: s))
    return root


def run(root: Path, *args, script: Path = SCRIPT):
    return subprocess.run(
        [sys.executable, str(script), "--root", str(root), *args],
        capture_output=True, text=True,
    )


OUTPUTS = ("SHIP_DECISION_TABLE.csv", "SHIP_DECISION_SUMMARY.md")


def pristine_script(tmp_path: Path) -> Path:
    """The pre-change script from git HEAD~ ancestry, for the golden contrast.

    Extracted from git rather than vendored so the golden reference cannot
    silently drift from history. Skips if the blob is unavailable (shallow
    clone / archive export).
    """
    rel = "scripts/evaluation/vm/ship/consolidate_ship_results.py"
    for rev in ("HEAD", "HEAD~1", "HEAD~2"):
        cp = subprocess.run(["git", "-C", str(REPO), "show", "%s:%s" % (rev, rel)],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            continue
        if "_seed_key" in cp.stdout:  # already the hardened version
            continue
        p = tmp_path / "orig_consolidate.py"
        p.write_text(cp.stdout)
        return p
    pytest.skip("pre-change consolidate_ship_results.py not reachable from git")


# --------------------------------------------------------------------------
# _seed_key unit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,want", [
    ("s19", "s19"), (19, "s19"), ("19", "s19"),
    ("s42", "s42"), (42, "s42"), (58, "s58"), ("s99", "s99"),
])
def test_seed_key_normalizes(raw, want):
    assert _seed_key(raw) == want


# --------------------------------------------------------------------------
# golden byte-identity on the all-correct fixture
# --------------------------------------------------------------------------
def test_golden_byte_identical_to_pristine(tmp_path):
    """Correct-token inputs must produce byte-identical artifacts pre/post."""
    orig = pristine_script(tmp_path)
    a, b = tmp_path / "old", tmp_path / "new"
    build_root(a)
    build_root(b)

    r_old = run(a, script=orig)
    r_new = run(b)
    assert r_old.returncode == 0, r_old.stderr
    assert r_new.returncode == 0, r_new.stdout + r_new.stderr

    for name in OUTPUTS:
        assert (a / name).read_bytes() == (b / name).read_bytes(), \
            "%s differs pre/post change" % name


# --------------------------------------------------------------------------
# seed-token robustness -- the incident itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fmt,label", [
    (lambda s: s, "s19"),
    (lambda s: int(s[1:]), "19-int"),
    (lambda s: s[1:], "19-str"),
])
def test_gate_seed_token_forms_all_populate_gate_iou(tmp_path, fmt, label):
    root = build_root(tmp_path / label, gate_fmt=fmt)
    r = run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_csv(root / "SHIP_DECISION_TABLE.csv")
    assert df["gate_iou"].notna().all(), "gate_iou not populated for %s" % label


@pytest.mark.parametrize("fmt", [lambda s: int(s[1:]), lambda s: s[1:]])
def test_pristine_code_loses_gate_iou_on_bare_seed(tmp_path, fmt):
    """Golden contrast: the bug is real in the pre-change code (bare seeds)."""
    orig = pristine_script(tmp_path)
    root = build_root(tmp_path / "bare", gate_fmt=fmt)
    r = run(root, script=orig)
    assert r.returncode == 0
    df = pd.read_csv(root / "SHIP_DECISION_TABLE.csv")
    # Either the column is absent entirely or it is all-NaN -- silently, and
    # with exit 0. Both are the incident.
    assert "gate_iou" not in df.columns or df["gate_iou"].isna().all()


# --------------------------------------------------------------------------
# unmatched-row accounting
# --------------------------------------------------------------------------
def test_unmatched_row_is_counted_and_flagged(tmp_path):
    """One s99 gate row: counted as unmatched, '!!' printed, exit 0 by default."""
    root = build_root(tmp_path / "u")
    g = pd.read_csv(root / "demak_gate" / "demak_gate_ship_summary.csv")
    g.loc[0, "seed"] = "s99"
    g.to_csv(root / "demak_gate" / "demak_gate_ship_summary.csv", index=False)

    r = run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "!!" in r.stdout
    assert "demak_gate: 6 rows read, 5 matched, 1 unmatched" in r.stdout
    assert "s99" in r.stdout


def test_unmatched_row_fails_under_strict(tmp_path):
    root = build_root(tmp_path / "us")
    g = pd.read_csv(root / "demak_gate" / "demak_gate_ship_summary.csv")
    g.loc[0, "seed"] = "s99"
    g.to_csv(root / "demak_gate" / "demak_gate_ship_summary.csv", index=False)

    r = run(root, "--strict")
    assert r.returncode == 1, r.stdout
    assert "unmatched" in r.stdout


def test_missing_source_fails_only_under_strict(tmp_path):
    root = build_root(tmp_path / "m")
    (root / "hampyeong" / "hampyeong_ship_per_date_metrics.csv").unlink()

    assert run(root).returncode == 0
    r = run(root, "--strict")
    assert r.returncode == 1, r.stdout
    assert "missing source" in r.stdout


# --------------------------------------------------------------------------
# zero-matched: fatal regardless of --strict
# --------------------------------------------------------------------------
def test_zero_matched_source_exits_one_without_strict(tmp_path):
    """Gate CSV present but no seed joinable -> join failure, not incompleteness."""
    root = build_root(tmp_path / "z", gate_fmt=lambda s: "s9" + s[1:])
    r = run(root)
    assert r.returncode == 1, r.stdout
    assert "ZERO rows joined" in r.stdout
    assert "demak_gate" in r.stdout


def test_zero_matched_still_writes_table(tmp_path):
    root = build_root(tmp_path / "zw", gate_fmt=lambda s: "s9" + s[1:])
    run(root)
    assert (root / "SHIP_DECISION_TABLE.csv").exists()


def test_zero_matched_on_only_source_exits_one_on_skeleton_path(tmp_path):
    """Gate-only root, all seeds unjoinable: skeleton written but exit 1."""
    root = tmp_path / "zs"
    root.mkdir()
    write_gate(root, seed_fmt=lambda s: "s9" + s[1:])
    r = run(root)
    assert r.returncode == 1, r.stdout
    assert (root / "SHIP_DECISION_TABLE.csv").exists()


# --------------------------------------------------------------------------
# skeleton path (mid-campaign) stays exit 0
# --------------------------------------------------------------------------
def test_empty_root_writes_skeleton_and_exits_zero(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    r = run(root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skeleton written" in r.stdout
    df = pd.read_csv(root / "SHIP_DECISION_TABLE.csv")
    assert list(df["seed"]) == [s for v in VARIANTS for s in SEEDS]
    assert list(df["variant"]) == [v for v in VARIANTS for s in SEEDS]
    assert not (root / "SHIP_DECISION_SUMMARY.md").exists()


def test_empty_root_fails_under_strict(tmp_path):
    root = tmp_path / "empty_s"
    root.mkdir()
    r = run(root, "--strict")
    assert r.returncode == 1, r.stdout


# --------------------------------------------------------------------------
# happy path shape
# --------------------------------------------------------------------------
def test_full_fixture_populates_every_gate_block(tmp_path):
    root = build_root(tmp_path / "full")
    r = run(root, "--strict")
    assert r.returncode == 0, r.stdout + r.stderr
    df = pd.read_csv(root / "SHIP_DECISION_TABLE.csv")
    assert len(df) == 6
    for col in ("gate_iou", "hamp_mean3_iou", "sds_rmse_mean",
                "sds_rmse_s8", "sds_rmse_s32", "sds_rmse_s112",
                "trend_s32", "trend_s32_hac_se", "trend_s112",
                "trend_s112_hac_se"):
        assert df[col].notna().all(), col
    # off-threshold SDS rows must not leak into the thr-0.5 means
    assert df["sds_rmse_mean"].max() < 20.0
    assert "0 unmatched" in r.stdout
    assert "!!" not in r.stdout
    assert (root / "SHIP_DECISION_SUMMARY.md").exists()
