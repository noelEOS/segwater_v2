"""Unit tests for the ship-campaign provenance guards.

``build_ship_manifest.check_run_dir`` is pure -- it takes a parsed
``run_config.yaml`` dict plus the counts main() measured and returns problem
strings -- so every guard is exercisable here with no run dir, no VM, no torch.
Also pins the stamp fix: the old ``name.split("_")[-6][:16]`` positional hack
against ``runsel.TIMESTAMP_PATTERN.search``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VM = REPO / "scripts" / "evaluation" / "vm"
sys.path.insert(0, str(VM))
sys.path.insert(0, str(VM / "ship"))

from build_ship_manifest import (  # noqa: E402
    ENCODER,
    check_run_dir,
    run_dir_stamp,
)
from runsel import TIMESTAMP_PATTERN  # noqa: E402

# A real-shaped dir name from the campaign. `run_inference_sweep.py` builds
# `<sweep>__<stamp>__<ckpt-name>__<preset>` and sanitizes `__` to `_`, and the
# ship campaign's ckpt/preset names are `mx630s2_<seed>_<variant>` and
# `native224_weighted_224_b0_s<stride>` -- an 8-token tail after the stamp.
GOOD_DIR = (
    "demak_full_ship_s42_last_s32_20260729T020404Z"
    "_mx630s2_s42_last_native224_weighted_224_b0_s32"
)
GOOD_CKPT = "outputs/mx630_stage2/s42/mx630s2_s42_last.pth"
BEST_CKPT = "outputs/mx630_stage2/s42/mx630s2_s42_pmwiou0p8123.pth"


def cfg_for(ckpt: str = GOOD_CKPT, stride: int = 32) -> dict:
    """A fully-conformant config; tests mutate one key at a time."""
    return {
        "model": {"encoder_name": ENCODER},
        "inference": {
            "checkpoint_path": ckpt,
            "data": {"stride": stride},
            "compute": {"amp_dtype": "bfloat16", "tf32": True},
            "stitching": {"accumulate_on_device": True},
        },
    }


def run(cfg, *, run_dir_name=GOOD_DIR, variant="last", n_tif=213,
        strides=(32,), expected=213, gate="demak_full_s32", seed="s42"):
    return check_run_dir(gate, seed, variant, run_dir_name, cfg, n_tif,
                         list(strides), expected)


def test_good_arm_has_no_problems():
    assert run(cfg_for()) == []


def test_good_best_arm_has_no_problems():
    assert run(cfg_for(BEST_CKPT), variant="best") == []


@pytest.mark.parametrize("mutate, substring", [
    # encoder
    (lambda c: c["model"].__setitem__("encoder_name", "tu-convnextv2_large"),
     "encoder tu-convnextv2_large"),
    # ckpt outside the seed's mx630_stage2 dir
    (lambda c: c["inference"].__setitem__(
        "checkpoint_path", "outputs/mx630_stage2/s19/mx630s2_s42_last.pth"),
     "ckpt not under mx630_stage2/s42"),
    # ckpt filename lacks the seed token (right dir, wrong file copied in)
    (lambda c: c["inference"].__setitem__(
        "checkpoint_path", "outputs/mx630_stage2/s42/mx630s2_last.pth"),
     "ckpt filename lacks _s42_"),
    # bf16 compute flags
    (lambda c: c["inference"]["compute"].__setitem__("amp_dtype", "float16"),
     "amp_dtype='float16'"),
    (lambda c: c["inference"]["compute"].__setitem__("tf32", False),
     "tf32=False"),
    (lambda c: c["inference"]["stitching"].__setitem__(
        "accumulate_on_device", None),
     "accumulate_on_device=None"),
])
def test_single_guard_fires(mutate, substring):
    cfg = cfg_for()
    mutate(cfg)
    problems = run(cfg)
    assert any(substring in p for p in problems), problems


def test_missing_compute_section_is_caught():
    """A config key that never reached the code leaves the run silently fp32."""
    cfg = cfg_for()
    del cfg["inference"]["compute"]
    problems = run(cfg)
    assert any("amp_dtype=None" in p for p in problems), problems
    assert any("tf32=None" in p for p in problems), problems


def test_variant_last_rejects_pmwiou_ckpt():
    problems = run(cfg_for(BEST_CKPT), variant="last")
    assert any("variant=last but ckpt is mx630s2_s42_pmwiou0p8123.pth" in p
               for p in problems), problems


def test_variant_best_rejects_last_ckpt():
    problems = run(cfg_for(GOOD_CKPT), variant="best")
    assert any("variant=best but ckpt is mx630s2_s42_last.pth" in p
               for p in problems), problems


def test_config_dir_stride_mismatch():
    """Config says 112, the dir name (…_b0_s32) says 32."""
    problems = run(cfg_for(stride=112), strides=(32, 112))
    assert any("config stride 112 != dir stride 32" in p for p in problems), problems


def test_stride_not_in_gate_list():
    problems = run(cfg_for(), strides=(112,))
    assert any("unexpected stride 32" in p for p in problems), problems


def test_stride_absent_from_dir_name():
    problems = run(cfg_for(), run_dir_name="demak_full_ship_s42_last_20260729T020404Z")
    assert any("config stride 32 != dir stride -1" in p for p in problems), problems


def test_raster_count_short():
    problems = run(cfg_for(), n_tif=206, expected=213)
    assert any("206 probability tifs, expected 213" in p for p in problems), problems


def test_missing_stamp_is_a_problem():
    bad = "demak_full_ship_s42_last_s32_notastamp_mx630s2_s42_last_224_b0_s32"
    problems = run(cfg_for(), run_dir_name=bad)
    assert any("no UTC stamp in run dir name" in p for p in problems), problems
    assert run_dir_stamp(bad) == ""


def test_tag_carries_gate_seed_variant_stride():
    problems = run(cfg_for(), n_tif=1, gate="narrabeen", seed="s58",
                   variant="last", strides=(32,), expected=87)
    assert problems[0].startswith("narrabeen/s58/last/s32: "), problems


# --- stamp extraction: regex vs the old positional [-6] hack -----------------

def old_stamp(name: str) -> str:
    """The pre-fix expression, kept here to demonstrate the divergence."""
    return name.split("_")[-6][:16]


@pytest.mark.parametrize("name, stamp, old_value", [
    # The real ship shape: 8 tokens follow the stamp, so [-6] lands on the
    # VARIANT token, not the stamp. The old column was wrong for every ship row.
    (GOOD_DIR, "20260729T020404Z", "last"),
    # A shorter tail (no `native224_weighted` preset words) shifts it again.
    ("demak_gate_ship_s42_last_20260729T020404Z_mx630s2_s42_last_224_b0_s32",
     "20260729T020404Z", "mx630s2"),
    # A longer tail lands inside the preset.
    ("narrabeen_ship_s19_best_20260729T113355Z_mx630s2_s19_pmwiou_native224"
     "_weighted_extra_224_b0_s8", "20260729T113355Z", "native224"),
    # A non-ship sweep from the same runs root: wrong there too, differently.
    ("demak_full_mx630k_20260729T020404Z_mx630k_native224_weighted_224_b0_s32",
     "20260729T020404Z", "mx630k"),
    # The one shape [-6] does hit: exactly 6 tokens from the end (single-token
    # ckpt name + 4-token preset). Nothing in this campaign produces it.
    ("demak_full_x_20260729T020404Z_a_weighted_224_b0_s32",
     "20260729T020404Z", "20260729T020404Z"),
])
def test_timestamp_search_beats_positional_index(name, stamp, old_value):
    m = TIMESTAMP_PATTERN.search(name)
    assert m is not None and m.group(0) == stamp
    assert run_dir_stamp(name) == stamp
    # The old expression, computed here rather than trusted, to pin the divergence.
    assert old_stamp(name) == old_value
    if old_value != stamp:
        assert TIMESTAMP_PATTERN.fullmatch(old_value) is None


def test_check_run_dir_does_not_mutate_its_config():
    cfg = cfg_for()
    snapshot = copy.deepcopy(cfg)
    run(cfg)
    assert cfg == snapshot


# --------------------------------------------------------------------------
# Lineage parameterisation (added for the ConvNeXtV2-Base `cnxb` campaign).
#
# `encoder` and `ckpt_subdir` used to be module constants pinned to Swin-B under
# `outputs/mx630_stage2/<seed>/`. A second lineage sits one level deeper
# (`mx630_stage2/upernet_tu-convnextv2_base/<seed>/`) with an encoder of its own,
# and -- critically -- with checkpoint FILENAMES identical to Swin-B's
# (`step23930_last.pth` exists in both, different weights). The path is
# therefore the only thing distinguishing them, so the subdir guard has to be
# exact rather than a substring of a shared ancestor.
# --------------------------------------------------------------------------

CNX_ENCODER = "tu-convnextv2_base"
CNX_SUBDIR = "mx630_stage2/upernet_tu-convnextv2_base"
CNX_CKPT = (
    "outputs/mx630_stage2/upernet_tu-convnextv2_base/s42/"
    "upernet_tu-convnextv2_base_s42_step23930_last.pth"
)
CNX_DIR = (
    "demak_full_cnxb_s42_last_s32_20260730T020404Z"
    "_cnxb_s42_last_native224_weighted_224_b0_s32"
)


def cnx_run(cfg, **kw):
    kw.setdefault("run_dir_name", CNX_DIR)
    return check_run_dir(
        kw.pop("gate", "demak_full_s32"), kw.pop("seed", "s42"),
        kw.pop("variant", "last"), kw.pop("run_dir_name"), cfg,
        kw.pop("n_tif", 213), list(kw.pop("strides", (32,))),
        kw.pop("expected", 213),
        encoder=kw.pop("encoder", CNX_ENCODER),
        ckpt_subdir=kw.pop("ckpt_subdir", CNX_SUBDIR),
    )


def test_convnextv2_lineage_passes_every_guard():
    cfg = cfg_for(ckpt=CNX_CKPT)
    cfg["model"]["encoder_name"] = CNX_ENCODER
    assert cnx_run(cfg) == []


def test_defaults_still_pin_swinb():
    """Omitting the new kwargs must reproduce the original Swin-B behaviour."""
    swin = cfg_for()                      # Swin-B encoder + mx630_stage2/s42 path
    assert run(swin) == []
    # ...and the ConvNeXtV2 config must FAIL under those defaults, on both axes.
    cnx = cfg_for(ckpt=CNX_CKPT)
    cnx["model"]["encoder_name"] = CNX_ENCODER
    problems = run(cnx)
    assert any("encoder" in p for p in problems)
    assert any("ckpt not under mx630_stage2/s42" in p for p in problems)


def test_swinb_checkpoint_rejected_under_convnextv2_lineage():
    """The cross-lineage confusion this guard exists for.

    A Swin-B checkpoint path sits under `mx630_stage2/s42/`, which is a PREFIX
    of the ConvNeXtV2 lineage root's ancestor -- so a sloppy substring check
    could accept it. It must be rejected.
    """
    cfg = cfg_for()                       # Swin-B path AND Swin-B encoder
    problems = cnx_run(cfg)
    assert any("ckpt not under %s/s42" % CNX_SUBDIR in p for p in problems)
    assert any("encoder" in p for p in problems)


def test_identical_last_filename_across_lineages_is_distinguished_by_path():
    """`step23930_last.pth` exists in both lineages with different weights.

    Same basename, same variant, same seed -- only the directory differs. Each
    must pass under its own lineage and fail under the other's.
    """
    base = "upernet_%s_s42_step23930_last.pth"
    swin_ck = "outputs/mx630_stage2/s42/" + base % "tu-swin_base_patch4_window7_224"
    cnx_ck = ("outputs/mx630_stage2/upernet_tu-convnextv2_base/s42/"
              + base % "tu-convnextv2_base")

    swin_cfg = cfg_for(ckpt=swin_ck)
    cnx_cfg = cfg_for(ckpt=cnx_ck)
    cnx_cfg["model"]["encoder_name"] = CNX_ENCODER

    assert run(swin_cfg) == []            # Swin-B under Swin-B defaults
    assert cnx_run(cnx_cfg) == []         # ConvNeXtV2 under ConvNeXtV2 kwargs
    assert any("ckpt not under" in p for p in run(cnx_cfg))
    assert any("ckpt not under" in p for p in cnx_run(swin_cfg))


def test_gates_for_tag_places_tag_in_the_middle():
    """The tag must sit between gate and seed, and no name may prefix another."""
    from build_ship_manifest import gates_for_tag
    from naming import require_no_prefix_collisions

    g = gates_for_tag("cnxb")
    assert g["demak_gate"][0] % ("s42", "last") == "demak_gate_cnxb_s42_last"
    assert g["demak_full_s112"][0] % ("s42", "last") == "demak_full_cnxb_s42_last_s112"
    # Scene-count keys must be inherited from completion.py, not re-inlined.
    assert g["narrabeen"][1] == "narrabeen"
    assert g["demak_full_s112"][1] == g["demak_full_s32"][1] == "demak_full"

    names = [tmpl % (s, v)
             for tmpl, _, _ in g.values()
             for s in ("s19", "s42", "s58") for v in ("best", "last")]
    require_no_prefix_collisions(names, what="sweep name")   # raises on collision

    # Two tags must never collide with each other either.
    ship = gates_for_tag("ship")
    both = names + [tmpl % (s, v)
                    for tmpl, _, _ in ship.values()
                    for s in ("s19", "s42", "s58") for v in ("best", "last")]
    require_no_prefix_collisions(both, what="sweep name")
