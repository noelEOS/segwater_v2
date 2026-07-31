"""Make the registered Demak frequency-map analysis read SHIP-campaign rasters.

The analysis (`optical_validation_comparisson/scripts/11_s2_frequency_maps.py`)
resolves S1 probability rasters through `sem_core.scene_index(seed)`, which globs
`experiments/demak_semarang/inference/` for a `SEM_MODEL` token such as
`_upernet_swin_base_224_`. The ship campaign's rasters live somewhere else, under
different names, and carry no such token:

    ~/segwater_v2/outputs/inference/runs/
      demak_full_ship_<seed>_<variant>_s<stride>_<UTCstamp>Z_mx630s2_..._b0_s<stride>/
        <scene_id>/<scene_id>_probability_water.tif

So this module supplies a drop-in `scene_index(seed)` over campaign run dirs and
leaves everything else — `load_aoi`, `read_prob`, the AOI grid, and every
estimand-defining constant in the analysis (THR 0.5, GATE 0.9, MIN_OBS 10/5,
epochs 2017-2019 / 2022-2024, the 3-seed median) — untouched.

⚠️ The resulting numbers are for the **mx630_stage2** lineage. The 20 registered
CITABLE rows were computed on a **chip-based** S1 model. These are a comparison,
NOT a like-for-like replacement; label the lineage on every row.

Usage:
    import ship_core_shim as shim
    shim.configure(variant="last", stride=32)
    idx = shim.scene_index("s19")      # -> DataFrame[scene_id, datetime, tif]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runsel import resolve_run_dir  # noqa: E402

RUNS = Path.home() / "segwater_v2/outputs/inference/runs"
SEEDS = ("s19", "s42", "s58")
SCENE_RE = re.compile(r"^S1_(\d{8})_(\d{6})_(.+)$")

# The orbit-127 scene excluded a priori by every registered Demak product.
A_PRIORI_EXCLUSIONS = {"S1_20250730_105715_127_1_1"}

_CFG: dict = {"variant": None, "stride": None, "tag": "ship"}


def configure(variant: str, stride: int, tag: str = "ship") -> None:
    """Select the (variant, stride) cell and the CAMPAIGN whose run dirs to read.

    ``tag`` defaults to ``ship`` (the Swin-B campaign) so existing callers are
    unaffected. It is the middle token of the sweep name, and is the only thing
    separating one lineage's run dirs from another's on a shared runs root --
    ``demak_full_ship_s42_last_s32`` vs ``demak_full_cnxb_s42_last_s32`` vs
    ``..._cnxt_...``. Checkpoint FILENAMES collide across all three, so the tag
    cannot be inferred from the checkpoint.
    """
    if variant not in ("best", "last"):
        raise ValueError("variant must be best|last, got %r" % variant)
    if stride not in (32, 112):
        raise ValueError("stride must be 32|112, got %r" % stride)
    if not tag:
        raise ValueError("tag must be a non-empty campaign tag")
    _CFG.update(variant=variant, stride=stride, tag=tag)


def run_dir(seed: str) -> Path:
    """The one campaign run dir for (seed, configured variant+stride).

    Resolved through runsel, which anchors on the UTC stamp -- a bare prefix glob
    would match `..._s32` when asked for `..._s3`, and more importantly would not
    distinguish a re-run of the same sweep.
    """
    if _CFG["variant"] is None:
        raise RuntimeError("call configure(variant, stride) first")
    name = "demak_full_%s_%s_%s_s%d" % (
        _CFG["tag"], seed, _CFG["variant"], _CFG["stride"])
    return resolve_run_dir(RUNS, name)


def scene_index(seed: str) -> pd.DataFrame:
    """Drop-in for sem_core.scene_index: scene_id, datetime, tif.

    Returns ALL orbit-76 scenes with no QC exclusion (the registered convention),
    minus the a-priori orbit-127 scene. Asserts 213 scenes so a half-written run
    dir cannot silently shrink the series.
    """
    base = run_dir(seed)
    rows = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name in A_PRIORI_EXCLUSIONS:
            continue
        m = SCENE_RE.match(d.name)
        if m is None:
            continue
        tifs = sorted(d.glob("*_probability_water.tif"))
        if len(tifs) != 1:
            raise RuntimeError("%s: %d probability tifs" % (d, len(tifs)))
        rows.append(dict(
            scene_id=d.name,
            datetime=pd.to_datetime(m.group(1) + m.group(2),
                                    format="%Y%m%d%H%M%S", utc=True),
            tif=str(tifs[0])))
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    if len(df) != 213:
        raise RuntimeError("%s/%s s%d: %d scenes != 213"
                           % (seed, _CFG["variant"], _CFG["stride"], len(df)))
    return df


def audit() -> pd.DataFrame:
    """One row per seed: run dir, checkpoint, encoder, precision flags.

    Read back from each run's run_config.yaml rather than assumed, so the
    frequency maps carry their own provenance.
    """
    import yaml
    out = []
    for seed in SEEDS:
        d = run_dir(seed)
        cfg = yaml.safe_load((d / "run_config.yaml").read_text())
        inf = cfg["inference"]
        out.append(dict(
            seed=seed, variant=_CFG["variant"], stride=_CFG["stride"],
            run_dir=d.name, checkpoint=inf["checkpoint_path"],
            encoder=cfg["model"]["encoder_name"],
            amp_dtype=(inf.get("compute") or {}).get("amp_dtype"),
            accumulate_on_device=(inf.get("stitching") or {}).get("accumulate_on_device"),
            config_stride=inf["data"]["stride"]))
    df = pd.DataFrame(out)
    if df.checkpoint.nunique() != 3:
        raise RuntimeError("expected 3 distinct checkpoints, got %d" % df.checkpoint.nunique())
    if (df.config_stride != _CFG["stride"]).any():
        raise RuntimeError("stride mismatch vs config: %s" % df.config_stride.tolist())
    return df
