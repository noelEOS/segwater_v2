"""Contract tests for ckptsel.py — checkpoint roles, pools, and the dual rule.

Filename rules need no torch: seed dirs are fabricated from empty files. The
golden-parity test against ensure_best_ckpts (which imports torch at module
level) is skipped when torch is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "evaluation" / "vm"))

from ckptsel import (  # noqa: E402
    CkptSelError,
    assert_distinct_weights,
    best_pool,
    is_best_candidate,
    miou_of,
    pick_best_filename_rule,
    pick_best_float_rule,
    resolve_best,
    resolve_last,
    require_seed_token,
    role_of,
    sha256,
    step_of,
    step_of_checkpoint,
)

# Realistic seed-dir contents (historical `miou` naming, which the filename
# rule was designed for; three retained top-k files per seed dir is the real
# hazard that forbids glob-based best resolution).
S42_NAMES = [
    "swinb_s42_step36000_miou0.9481.pth",
    "swinb_s42_step38400_miou0.9506.pth",
    "swinb_s42_step39600_miou0.9506.pth",
    "swinb_s42_step40320_last.pth",
    "swinb_s42_step31200_snap_miou0.9460.pth",
    "swinb_s42_swa5_step31200-40320.pth",
]


def make_seed_dir(root: Path, names=S42_NAMES, best_target=None) -> Path:
    d = root / "s42"
    d.mkdir(parents=True)
    for n in names:
        (d / n).touch()
    if best_target is not None:
        (d / "best.pth").symlink_to(best_target)
    return d


# --------------------------------------------------------------------------- #
# parsing + roles
# --------------------------------------------------------------------------- #
def test_step_and_miou_parsing():
    p = "swinb_s42_step38400_miou0.9506.pth"
    assert step_of(p) == 38400
    assert miou_of(p) == pytest.approx(0.9506)
    assert step_of("no_tokens_here.pth") == -1
    assert miou_of("no_tokens_here.pth") == -1.0


def test_pmwiou_names_carry_no_parseable_miou():
    """Regression pin, not a bug: 'pmwiou' does NOT contain 'miou' (p-m-w-iou),
    so pmwiou-lineage filenames parse as score-less. The filename rule then
    degrades to highest-step; best for those lineages comes from the trainer's
    best.pth symlink, never from filename parsing."""
    p = "swinb_mx630s2_s42_step38400_pmwiou0.9506.pth"
    assert miou_of(p) == -1.0
    assert role_of(p) == "other"   # ladder classification: not a `_miou` top-k
    assert step_of(p) == 38400     # step still parses


def test_role_of_order_matters():
    assert role_of("m_step40320_last.pth") == "last"
    # A snapshot name also contains "miou"; _snap_ must win.
    assert role_of("m_step31200_snap_miou0.9460.pth") == "snap"
    assert role_of("m_step38400_miou0.9506.pth") == "best"
    assert role_of("random.pth") == "other"


def test_step_of_checkpoint_prefers_payload():
    p = "m_step100_miou0.9.pth"
    assert step_of_checkpoint(p, {"step": 200}) == 200   # payload wins
    assert step_of_checkpoint(p, {}) == 100              # filename fallback
    assert step_of_checkpoint("m.pth", {}) == -1


# --------------------------------------------------------------------------- #
# the candidate pool
# --------------------------------------------------------------------------- #
def test_pool_excludes_best_snap_swa_last(tmp_path):
    d = make_seed_dir(tmp_path, best_target=S42_NAMES[1])
    pool = best_pool(d)
    names = [p.name for p in pool]
    assert names == sorted(S42_NAMES[:3])  # only the three pmwiou step ckpts
    assert not is_best_candidate("best.pth")
    assert not is_best_candidate("m_step31200_snap_miou0.9460.pth")
    assert not is_best_candidate("m_swa5_step31200-40320.pth")
    assert not is_best_candidate("m_step40320_last.pth")


def test_empty_pool_raises(tmp_path):
    d = tmp_path / "s19"
    d.mkdir()
    (d / "m_step1_last.pth").touch()  # only exclusions present
    with pytest.raises(CkptSelError):
        best_pool(d)


# --------------------------------------------------------------------------- #
# the dual rule
# --------------------------------------------------------------------------- #
def test_filename_rule_tiebreak_is_the_s42_case(tmp_path):
    # step38400 and step39600 tie at 4-dp mIoU; the higher step ships.
    d = make_seed_dir(tmp_path)
    pick = pick_best_filename_rule(best_pool(d))
    assert pick.name == "swinb_s42_step39600_miou0.9506.pth"


def test_filename_rule_matches_ensure_best_ckpts(tmp_path):
    """Golden parity with the implementation being replaced."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
    import ensure_best_ckpts  # noqa: E402

    d = make_seed_dir(tmp_path)
    pool = best_pool(d)
    assert pick_best_filename_rule(pool) == ensure_best_ckpts.pick_best_filename_rule(pool)


def test_float_rule_argmax_and_ties():
    a, b, c = Path("a.pth"), Path("b.pth"), Path("c.pth")
    assert pick_best_float_rule([(0.90, 1, a), (0.95, 2, b), (0.93, 3, c)]) == b
    # Ties on (miou, step): last in input order wins (stable sort, preserved
    # from both originals).
    assert pick_best_float_rule([(0.95, 2, a), (0.95, 2, c)]) == c
    # Ties on miou alone: later step wins (the trainer rule).
    assert pick_best_float_rule([(0.95, 3, a), (0.95, 2, c)]) == a
    with pytest.raises(CkptSelError):
        pick_best_float_rule([])


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def test_resolve_best_via_symlink_only(tmp_path):
    d = make_seed_dir(tmp_path, best_target=S42_NAMES[1])
    assert resolve_best(d).name == S42_NAMES[1]


def test_resolve_best_missing_symlink_raises(tmp_path):
    d = make_seed_dir(tmp_path)
    with pytest.raises(CkptSelError) as e:
        resolve_best(d)
    assert "pmwiou" in str(e.value)  # the error teaches the glob hazard


def test_resolve_best_pointing_at_last_raises(tmp_path):
    d = make_seed_dir(tmp_path, best_target="swinb_s42_step40320_last.pth")
    with pytest.raises(CkptSelError):
        resolve_best(d)


def test_resolve_last_exactly_one(tmp_path):
    d = make_seed_dir(tmp_path)
    assert resolve_last(d).name == "swinb_s42_step40320_last.pth"
    (d / "swinb_s42_step40000_last.pth").touch()
    with pytest.raises(CkptSelError) as e:
        resolve_last(d)
    assert "found 2" in str(e.value)
    empty = tmp_path / "s19"
    empty.mkdir()
    with pytest.raises(CkptSelError):
        resolve_last(empty)


def test_require_seed_token():
    require_seed_token("swinb_mx630s2_s42_step38400_pmwiou0.9506.pth", "s42")
    with pytest.raises(CkptSelError) as e:
        require_seed_token("swinb_mx630s2_s19_step38400_pmwiou0.9506.pth", "s42")
    assert "mis-copied" in str(e.value)


# --------------------------------------------------------------------------- #
# weight distinctness
# --------------------------------------------------------------------------- #
def test_distinct_weights_pass_and_collide(tmp_path):
    p1, p2, p3 = (tmp_path / n for n in ("a.pth", "b.pth", "c.pth"))
    p1.write_bytes(b"weights-A")
    p2.write_bytes(b"weights-B")
    p3.write_bytes(b"weights-A")  # duplicate of p1
    digests = assert_distinct_weights({"s19/last": p1, "s42/last": p2})
    assert digests["s19/last"] == sha256(p1)
    with pytest.raises(CkptSelError) as e:
        assert_distinct_weights({"s19/last": p1, "s42/last": p2, "s58/last": p3})
    assert "DUPLICATE WEIGHTS" in str(e.value)
