"""Contract tests for naming.py: prefix-collision detection + atomic writes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "evaluation" / "vm"))

from naming import (  # noqa: E402
    NameCollisionError,
    atomic_to_csv,
    atomic_write,
    atomic_write_text,
    find_prefix_collisions,
    require_no_prefix_collisions,
)


# --------------------------------------------------------------------------- #
# prefix collisions
# --------------------------------------------------------------------------- #
def test_real_incident_pair_collides():
    # The 2026-07-29 hazard: mx630k is a prefix of mx630k_best.
    names = ["demak_full_mx630k", "demak_full_mx630k_best"]
    assert find_prefix_collisions(names) == [
        ("demak_full_mx630k", "demak_full_mx630k_best")
    ]


def test_shared_token_prefix_is_not_a_collision():
    # The extension must be underscore-separated; these are distinct sweeps.
    assert find_prefix_collisions(["unet_resnet50", "unetplusplus_resnet50"]) == []


def test_chain_of_extensions():
    got = find_prefix_collisions(["a", "a_b", "a_bc"])
    assert got == [("a", "a_b"), ("a", "a_bc")]  # a_b vs a_bc do NOT collide


def test_clean_set_passes():
    require_no_prefix_collisions(
        ["gate_best_demak_s19", "gate_last_demak_s19", "gate_swa5_demak_s19"]
    )


def test_require_raises_listing_every_pair():
    with pytest.raises(NameCollisionError) as e:
        require_no_prefix_collisions(["x", "x_1", "y", "y_2"], what="config name")
    msg = str(e.value)
    assert "config name" in msg
    assert "'x' is a prefix of 'x_1'" in msg
    assert "'y' is a prefix of 'y_2'" in msg


def test_matches_gen_ship_configs_inline_check():
    """Golden parity with the expression gen_ship_configs.py shipped with."""
    names = [
        "demak_gate_ship_s19_best", "demak_gate_ship_s19_last",
        "narrabeen_ship_s42_last", "narrabeen_ship_s42_last_s112",
        "hampyeong_ship_s58_best",
    ]
    inline = sorted(set(
        (x, y) for x in names for y in names if x != y and y.startswith(x + "_")
    ))
    assert find_prefix_collisions(names) == inline
    assert inline  # the s112-suffixed name IS a collision; the check must see it


# --------------------------------------------------------------------------- #
# atomic writes
# --------------------------------------------------------------------------- #
def test_atomic_write_success_no_tmp_left(tmp_path):
    dest = tmp_path / "out.csv"
    with atomic_write(dest, newline="") as f:
        f.write("a,b\n1,2\n")
    assert dest.read_text() == "a,b\n1,2\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_exception_leaves_original_intact(tmp_path):
    dest = tmp_path / "out.csv"
    dest.write_text("original\n")
    with pytest.raises(RuntimeError):
        with atomic_write(dest) as f:
            f.write("partial garbage")
            raise RuntimeError("killed mid-write")
    assert dest.read_text() == "original\n"      # never clobbered
    assert not list(tmp_path.glob("*.tmp"))       # no stub left behind


def test_atomic_write_exception_with_no_prior_file(tmp_path):
    dest = tmp_path / "new.csv"
    with pytest.raises(RuntimeError):
        with atomic_write(dest) as f:
            f.write("partial")
            raise RuntimeError
    assert not dest.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_keep_bak(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_text("v1")
    atomic_write_text(dest, "v2", keep_bak=True)
    assert dest.read_text() == "v2"
    assert (tmp_path / "out.txt.bak").read_text() == "v1"


def test_atomic_to_csv_duck_typed(tmp_path):
    class Fake:
        def to_csv(self, f, **kw):
            f.write("x,y\n3,4\n")

    dest = tmp_path / "df.csv"
    atomic_to_csv(Fake(), dest)
    assert dest.read_text() == "x,y\n3,4\n"
    assert not list(tmp_path.glob("*.tmp"))
