"""Interactive threshold-slider visualizer for the Hampyeong Bay DEM flood ground truth.

Hampyeong's validation masks are built by flooding a TanDEM-X coastal DEM at the
tide-gauge reading of each Sentinel-1 overpass: water = DEM <= reading (elevations
are with respect to the WGS84 ellipsoid, so the readings sit at ~20-26 m). This
script emits a single self-contained HTML page whose slider sweeps that threshold
across the DEM's coastal band, with snap buttons at the five validation dates, so
the sensitivity of the ground truth to the water level can be inspected directly.

The DEM is quantized to 255 uint8 bins (250 uniform edges over the coastal window
plus one forced edge per date reading) and embedded, together with a 16-level
hillshade and a bit-packed validity raster, as three hand-written grayscale PNGs.
The page thresholds the codes per pixel on a canvas, so no server or library is
needed. Before writing anything the script hard-asserts that both the float DEM
comparison and the quantized-code comparison reproduce all five staged flood
masks with zero mismatching pixels.

Everything is rendered on the FULL-BAY grid (1538x2092); the analysis extent is
selectable in the page between three valid masks:
  * reported     -- the original hand-delineated VAL_AOI mask (1538x1672, the
                    northern subwindow of the full-bay grid) that every reported
                    Hampyeong number uses;
  * full_bay     -- that mask extended south to the DEM's own extent;
  * bridge_south -- the extended mask clipped to the bay interior south of the
                    Hampyeong bridge (a strict subset of full_bay).
The three are packed one-bit-each into a single small PNG (bit0/bit1/bit2), so
toggling in the page is a mask-bit swap with no reload and no layout change.

The hypsometric curve carries an optional (default-off) overlay of the 24
gauge-pairable descending scenes of the reported frame, read from the tidal
conditions CSV: one dot per scene at its own overpass threshold, sitting on the
active extent's curve. The five validation dates are asserted to appear there
with exactly the TIDE_READINGS thresholds and bin codes.

The DEM lives on the NAS; the valid mask and staged flood masks may be read from
a local mirror. Pass --nas-root / --mask-root accordingly.

Writes experiments/hampyeong/dem_threshold_visualizer/hampyeong_dem_threshold_visualizer.html.

Run with an interpreter that has rasterio, e.g.
    /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import rasterio

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluation.hampyeong_model_comparison import (  # noqa: E402
    DEFAULT_NAS_ROOT,
    EXPECTED_VALID_PIXELS,
    gt_path,
    valid_mask_path,
)

DEFAULT_MASK_ROOT = "ancillary_local/hampyeong/nas_root"
DEFAULT_EXTENT_ROOT = "experiments/hampyeong/dem_valid_mask_extension"
DEFAULT_OUT = (
    "experiments/hampyeong/dem_threshold_visualizer/hampyeong_dem_threshold_visualizer.html"
)
DEFAULT_TIDAL_CSV = "experiments/hampyeong/tidal_conditions_analysis/tidal_conditions.csv"

# The 24 gauge-pairable descending scenes of the reported frame; see that
# directory's README for the datum derivation and its four gates.
N_SCENES = 24

# Mean sea level on the gauge scale (341 cm above DL) carried into the DEM's
# vertical frame by the same constant as dem_threshold_m: 341/100 + 20.326.
MSL_DEM_M = 23.736

# Rows of the full-bay grid covered by the original VAL_AOI mask/DEM. The VAL_AOI
# raster is the northern subwindow of the full-bay raster (same origin, same 10 m
# EPSG:32652 grid); asserted byte-identical below.
VAL_AOI_ROWS = 1672

# The three selectable extents, in bit order (bit i of the packed mask PNG).
# 'reported' is the mask every reported Hampyeong number is computed over.
EXTENT_KEYS = ("reported", "full_bay", "bridge_south")
EXTENT_LABELS = {
    "reported": "reported",
    "full_bay": "full bay",
    "bridge_south": "bridge south",
}
EXTENT_BLURBS = {
    "reported": "the hand-delineated VAL_AOI mask used for every reported Hampyeong number",
    "full_bay": "that mask extended south to the DEM's own full-bay extent",
    "bridge_south": "the extended mask clipped to the bay interior south of the bridge",
}
EXTENDED_MASK_FILES = {
    "full_bay": "DEM_VALID_MASK_aoi_extended_full_bay.tif",
    "bridge_south": "DEM_VALID_MASK_aoi_extended_bridge_south.tif",
}

# Tide-gauge readings (m, WGS84 ellipsoidal datum) at each descending overpass.
TIDE_READINGS = {
    "20210305": 24.906,
    "20210422": 22.356,
    "20210504": 23.246,
    "20210621": 21.756,
    "20210913": 25.166,
}

# The three dates the manuscript scores; the other two exist but are not in Table 2.
SCORED_DATES = {"20210305", "20210422", "20210621"}

# Quantization window: the coastal band that the 24 descending readings span
# (20.886-25.976 m), padded. 249 uniform edges + 5 forced date edges = 254 edges
# = 255 bins, so codes 0-254 fit uint8. Validity now travels in its own packed
# mask PNG, so 255 is never emitted; N_CODES is the slider/curve length.
WINDOW_LO = 19.5
WINDOW_HI = 27.0
N_UNIFORM = 249
N_CODES = 255

PIXEL_AREA_KM2 = 10.0 * 10.0 / 1e6

_DEM_DIR = "Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM"


def dem_path(nas_root: Path) -> Path:
    """The full-bay DEM (1538x2092).

    The VAL_AOI DEM this visualizer used to read is exactly this raster's northern
    VAL_AOI_ROWS-row subwindow (same origin, same 10 m EPSG:32652 grid, verified
    byte-identical in float32). It is no longer opened: reproducing the five staged
    flood masks from these rows -- which verify_against_flood_masks asserts bit-
    exactly -- is a strictly stronger check than comparing the two DEMs.
    """
    return nas_root / _DEM_DIR / "Hamp_bay_TDX_UTM_10m_edited_clipped_full_bay.tif"


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def reading_f32(reading: float) -> float:
    """The reading as the staged masks saw it: float32-rounded, widened to float64.

    FLOAT DISCIPLINE (do not "simplify" this). The staged flood masks were
    produced by a float32 comparison, `float32_dem <= float32(reading)`, so
    float32(reading) -- not the decimal literal -- is the true cut value. The two
    differ in both directions and each direction costs exactly one pixel:
      * float32(25.166) = 25.166000366... > 25.166, so comparing the float64 DEM
        against the literal drops one pixel at 20210913;
      * float32(21.756) < 21.756, so an edge built from the literal keeps one
        pixel too many at 20210621.
    Anchor on this value, then do the nextafter step and the searchsorted in
    float64 -- np.nextafter on a float32 overshoots a whole float32 ULP, which
    re-breaks 20210621.
    """
    return float(np.float32(reading))


def build_edges(readings: dict[str, float]) -> np.ndarray:
    """Bin edges: uniform interior plus one forced edge just above each reading.

    A forced edge at nextafter(float32(r), +inf) makes `code <= code_for(r)`
    select exactly the pixels of `dem <= float32(r)`: searchsorted(side="left")
    puts a value equal to the cut strictly below that edge, and anything above
    it at or above it. See reading_f32 for the float rules.
    """
    uniform = np.linspace(WINDOW_LO, WINDOW_HI, N_UNIFORM, dtype=np.float64)
    forced = [math.nextafter(reading_f32(r), math.inf) for r in readings.values()]
    edges = np.unique(np.concatenate([uniform, np.asarray(forced, dtype=np.float64)]))
    if edges.size > N_CODES - 1:
        raise ValueError(f"{edges.size} edges exceeds the {N_CODES - 1}-edge uint8 budget")
    return edges


def quantize(dem_f64: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin code per pixel; water at threshold t is `codes <= code_for(t, edges)`."""
    codes = np.searchsorted(edges, dem_f64, side="left")
    if codes.max() > N_CODES - 1:
        raise ValueError(f"code {codes.max()} exceeds the {N_CODES - 1} code budget")
    return codes.astype(np.uint8)


def code_for(threshold: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, reading_f32(threshold), side="left"))


def load_scene_conditions(csv_path: Path, edges: np.ndarray) -> list[dict]:
    """The 24 pairable scenes with their overpass tide state and DEM-frame threshold.

    Each scene's `dem_threshold_m` is the same kind of quantity as TIDE_READINGS
    -- a cut applied as `dem <= float32(threshold)` -- so it is mapped to a bin
    code through the very same code_for/reading_f32 machinery, keeping the
    overlay's x positions on exactly the axis the slider sweeps.

    Two invariants are asserted: the file holds exactly N_SCENES rows, and the
    five validation dates appear among them with thresholds and bin codes equal
    to the staged-mask readings this script verifies bit-exactly. Nothing in the
    CSV flags those five (`reported_scene` marks only the three manuscript
    scenes), so membership is decided by the TIDE_READINGS keys.
    """
    with _require(csv_path, "tidal conditions CSV").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == N_SCENES, f"tidal CSV has {len(rows)} rows, expected {N_SCENES}"

    scenes = []
    for row in rows:
        date = row["acq_time_utc"][:10].replace("-", "")
        threshold = float(row["dem_threshold_m"])
        scenes.append(
            {
                "date": date,
                "thr": threshold,
                "code": code_for(threshold, edges),
                "level_cm": float(row["tide_level_cm"]),
                "msl_m": float(row["height_msl_m"]),
                "phase": row["phase"],
                "limb_frac": float(row["limb_fraction"]),
                "pctile": float(row["springneap_pctile_2021"]),
                "is_mask_date": date in TIDE_READINGS,
                "is_scored": row["reported_scene"] == "True",
            }
        )

    dates = [s["date"] for s in scenes]
    assert len(set(dates)) == N_SCENES, "duplicate acquisition dates in the tidal CSV"
    by_date = {s["date"]: s for s in scenes}
    for date, reading in TIDE_READINGS.items():
        assert date in by_date, f"validation date {date} missing from the tidal CSV"
        scene = by_date[date]
        assert scene["thr"] == reading, (
            f"{date}: CSV threshold {scene['thr']} != staged reading {reading}"
        )
        assert scene["code"] == code_for(reading, edges), (
            f"{date}: CSV bin code {scene['code']} != reading code {code_for(reading, edges)}"
        )
    n_mask = sum(s["is_mask_date"] for s in scenes)
    n_scored = sum(s["is_scored"] for s in scenes)
    assert n_mask == len(TIDE_READINGS), f"{n_mask} mask dates matched, expected {len(TIDE_READINGS)}"
    assert n_scored == len(SCORED_DATES), f"{n_scored} scored scenes, expected {len(SCORED_DATES)}"
    assert {s["date"] for s in scenes if s["is_scored"]} == SCORED_DATES, (
        "reported_scene flags disagree with SCORED_DATES"
    )
    return scenes


def verify_against_flood_masks(
    dem_f64: np.ndarray,
    codes: np.ndarray,
    edges: np.ndarray,
    mask_root: Path,
    reported: np.ndarray,
) -> dict[str, float]:
    """Hard gate: float and quantized thresholding must both reproduce the staged masks.

    The staged masks only exist on the VAL_AOI grid, so both comparisons are made
    on the northern VAL_AOI_ROWS rows of the full-bay rasters. Passing this also
    proves the full-bay DEM's north subwindow is the VAL_AOI DEM the masks were
    built from -- no separate VAL_AOI read is needed.
    """
    water_fractions: dict[str, float] = {}
    dem_north = dem_f64[:VAL_AOI_ROWS]
    codes_north = codes[:VAL_AOI_ROWS]
    reported_north = reported[:VAL_AOI_ROWS]
    n_valid = int(reported.sum())
    for date, reading in TIDE_READINGS.items():
        path = _require(gt_path(mask_root, date), f"flood mask {date}")
        with rasterio.open(path) as src:
            staged = src.read(1) > 0.5
        if staged.shape != dem_north.shape:
            raise ValueError(f"{date}: staged mask shape {staged.shape} != {dem_north.shape}")

        float_water = dem_north <= reading_f32(reading)
        n_float = int(np.count_nonzero(float_water != staged))
        assert n_float == 0, f"{date}: float thresholding mismatches staged mask on {n_float} px"

        quant_water = codes_north <= code_for(reading, edges)
        n_quant = int(np.count_nonzero(quant_water != staged))
        assert n_quant == 0, f"{date}: quantized thresholding mismatches staged mask on {n_quant} px"

        water_fractions[date] = float(np.count_nonzero(staged & reported_north) / n_valid)
        print(
            f"  {date}  reading {reading:7.3f} m  code {code_for(reading, edges):3d}  "
            f"float 0 mismatch  quantized 0 mismatch  water fraction {water_fractions[date]:.5f}"
        )
    return water_fractions


def hillshade(dem_f64: np.ndarray, res: float = 10.0, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    """Standard Horn hillshade, quantized to 16 levels (keeps the PNG small)."""
    dy, dx = np.gradient(dem_f64, res, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az_rad = math.radians(360.0 - az + 90.0)
    alt_rad = math.radians(alt)
    shaded = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    shaded = np.clip(shaded, 0.0, 1.0)
    return (np.round(shaded * 15.0) * 17.0).astype(np.uint8)


def png_gray8(arr: np.ndarray) -> bytes:
    """Minimal 8-bit grayscale PNG (colour type 0, filter 0) built from stdlib only."""
    if arr.dtype != np.uint8 or arr.ndim != 2:
        raise ValueError("png_gray8 expects a 2-D uint8 array")
    height, width = arr.shape
    raw = np.zeros((height, width + 1), dtype=np.uint8)
    raw[:, 1:] = arr
    compressed = zlib.compress(raw.tobytes(), 9)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def hypsometric_curve(codes: np.ndarray, valid: np.ndarray) -> list[float]:
    """Water fraction of the valid area at every slider position (255 values)."""
    counts = np.bincount(codes[valid].astype(np.int64), minlength=N_CODES)[:N_CODES]
    return [round(v, 6) for v in (np.cumsum(counts) / valid.sum()).tolist()]


def load_extent_masks(mask_root: Path, extent_root: Path, shape: tuple[int, int]) -> dict:
    """The three selectable valid masks on the full-bay grid, with their invariants asserted."""
    with rasterio.open(_require(valid_mask_path(mask_root), "valid mask")) as src:
        original = src.read(1) > 0.5
    if original.shape != (VAL_AOI_ROWS, shape[1]):
        raise ValueError(f"valid mask shape {original.shape} != VAL_AOI grid")

    masks = {"reported": np.zeros(shape, dtype=bool)}
    masks["reported"][:VAL_AOI_ROWS] = original

    for key, name in EXTENDED_MASK_FILES.items():
        with rasterio.open(_require(extent_root / name, f"{key} mask")) as src:
            arr = src.read(1) > 0.5
        if arr.shape != shape:
            raise ValueError(f"{key} mask shape {arr.shape} != full-bay grid {shape}")
        masks[key] = arr

    # README claims the extension keeps the original mask byte-identical over its
    # 1672 rows, and that bridge_south is a strict subset of full_bay. Assert both.
    assert np.array_equal(masks["full_bay"][:VAL_AOI_ROWS], original), (
        "full_bay mask's north window differs from the original NAS valid mask"
    )
    assert not (masks["bridge_south"] & ~masks["full_bay"]).any(), (
        "bridge_south is not a subset of full_bay"
    )

    n_reported = int(masks["reported"].sum())
    assert n_reported == EXPECTED_VALID_PIXELS, f"valid px {n_reported} != {EXPECTED_VALID_PIXELS}"
    return masks


def pack_masks(masks: dict) -> np.ndarray:
    """One uint8 raster with extent i in bit i (values 0-7), so the page needs one PNG."""
    packed = np.zeros(masks["reported"].shape, dtype=np.uint8)
    for bit, key in enumerate(EXTENT_KEYS):
        packed |= (masks[key].astype(np.uint8) << bit)
    assert packed.max() <= 7
    return packed


def build_html(payload: dict) -> str:
    """Self-contained page: theme CSS copied from hampyeong_bootstrap_chart.html."""
    data = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("__PAYLOAD__", data)


_HTML_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>Hampyeong: DEM flood threshold explorer</title>
<style>
  :root{
    --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984;
    --grid:#e7e6e2; --zero:#8a8984; --band:#f4f3ef;
    --accent:#0b6fb8; --scored:#0ca30c;
    --water:#2f7fc4; --land:#b8ae9a; --invalid:#d9d8d3; --context:#c9c8c3;
    --scene:#8a8984; --sceneline:#c2c1bb;
  }
  :root[data-theme="dark"]{
    --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#8f8e86;
    --grid:#333330; --zero:#8f8e86; --band:#242422;
    --accent:#5cb3f0; --scored:#0ca30c;
    --water:#2b6ea8; --land:#6d6552; --invalid:#2e2e2b; --context:#3a3a36;
    --scene:#9d9c93; --sceneline:#54534e;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#8f8e86;
    --grid:#333330; --zero:#8f8e86; --band:#242422;
    --accent:#5cb3f0; --water:#2b6ea8; --land:#6d6552; --invalid:#2e2e2b;
    --context:#3a3a36; --scene:#9d9c93; --sceneline:#54534e;
  }}
  *{box-sizing:border-box}
  body{background:var(--surface);color:var(--ink);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    margin:0;padding:28px 24px 44px;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1140px;margin:0 auto}
  h1{font-size:19px;font-weight:650;margin:0 0 4px}
  .sub{color:var(--ink2);font-size:13.5px;margin:0 0 6px;max-width:860px}
  .note{color:var(--muted);font-size:12px;margin:0 0 18px;max-width:860px}
  .cols{display:grid;grid-template-columns:1fr;gap:24px}
  @media(min-width:900px){.cols{grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:26px}}
  .panel{min-width:0}
  canvas{width:100%;height:auto;display:block;border:1px solid var(--grid);
    border-radius:4px;background:var(--band);image-rendering:pixelated}
  .ctl{margin:16px 0 4px}
  input[type=range]{width:100%;accent-color:var(--accent);margin:0}
  .scale{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:2px}
  .readout{display:flex;gap:26px;align-items:baseline;flex-wrap:wrap;margin:12px 0 2px}
  .big{font-size:26px;font-weight:640;font-variant-numeric:tabular-nums;line-height:1.1}
  .biglab{font-size:11.5px;color:var(--muted);display:block;font-weight:400}
  .snaps{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 0}
  button{font:inherit;font-size:12px;padding:5px 10px;border-radius:6px;cursor:pointer;
    color:var(--ink2);background:var(--surface);border:1px solid var(--grid);
    display:inline-flex;flex-direction:column;align-items:flex-start;line-height:1.3}
  button:hover{border-color:var(--accent);color:var(--ink)}
  button.on{border-color:var(--accent);color:var(--ink);background:var(--band)}
  button b{font-weight:620}
  button i{font-style:normal;font-size:10.5px;color:var(--muted)}
  button .dot{display:inline-block;width:5px;height:5px;border-radius:50%;
    background:var(--scored);margin-left:5px;vertical-align:middle}
  /* Extent picker: a slim pill row directly above the map. Same visual language
     as the snap buttons, one size down, so toggling never reflows the layout. */
  .extentbar{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:0 0 8px}
  .extentlab{font-size:11.5px;color:var(--muted)}
  .seg{display:inline-flex;border:1px solid var(--grid);border-radius:6px;overflow:hidden}
  .seg button{font-size:11.5px;padding:3px 9px;border:0;border-radius:0;
    display:inline-block;line-height:1.5;border-left:1px solid var(--grid)}
  .seg button:first-child{border-left:0}
  .seg button:hover{color:var(--ink);background:var(--band)}
  .seg button.on{color:var(--ink);background:var(--band);font-weight:620;
    box-shadow:inset 0 -2px 0 var(--accent)}
  .ptitle{font-size:13px;font-weight:620;margin:0 0 1px}
  .ptide{font-size:11.5px;color:var(--muted);margin:0 0 8px}
  /* Curve header: title on the left, the scene-overlay pill on the right. The
     pill is the .seg language one size down; it is laid out unconditionally and
     the note below it only toggles visibility, so nothing ever reflows. */
  .chead{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:22px 0 0}
  .pill{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:var(--ink2);
    border:1px solid var(--grid);border-radius:6px;padding:3px 8px;cursor:pointer;
    background:var(--surface);line-height:1.5;white-space:nowrap;user-select:none}
  .pill:hover{color:var(--ink);background:var(--band)}
  .pill:focus-within{border-color:var(--accent);outline:2px solid transparent}
  .pill input{accent-color:var(--accent);margin:0;width:12px;height:12px;cursor:pointer}
  .pill.on{color:var(--ink);background:var(--band);border-color:var(--accent)}
  .scnote{font-size:11px;color:var(--muted);margin:0 0 6px;visibility:hidden;min-height:1.5em}
  .scnote.on{visibility:visible}
  .tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);
    padding:6px 9px;border-radius:6px;font-size:11.5px;white-space:nowrap;opacity:0;
    transition:opacity .1s;z-index:9;box-shadow:0 2px 8px rgba(0,0,0,.25)}
  .mk{cursor:pointer}
  .legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:12px;
    color:var(--ink2);margin:10px 0 0}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .sw{width:12px;height:12px;border-radius:3px;border:1px solid var(--grid)}
  text{fill:var(--ink2)}
  .axlab{fill:var(--muted);font-size:10px}
  table{border-collapse:collapse;margin-top:12px;font-size:11.5px}
  th,td{padding:3px 12px 3px 0;text-align:right;border-bottom:1px solid var(--grid)}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--ink2);font-weight:600}
  details summary{cursor:pointer;color:var(--ink2);font-size:12.5px;margin-top:20px}
</style>

<div class="wrap">
  <h1>Hampyeong Bay — how the validation ground truth moves with the water level</h1>
  <p class="sub">The validation masks are made by flooding a TanDEM-X DEM at each overpass's tide-gauge
  reading: <b>water = DEM &le; threshold</b>. Drag the slider to sweep that threshold, or snap to the
  five Sentinel-1 descending dates. Elevations are with respect to the WGS84 ellipsoid, so the
  readings sit near 20–26 m rather than near 0.</p>
  <p class="note">Water fraction is over the <span id="vpx"></span> valid pixels of the active
  extent (<span id="extdesc"></span>). A large share of every extent is permanent sea, encoded as
  exactly 0 m in the DEM, so it floods at any threshold and the water fraction has an
  extent-dependent floor — here <span id="floorval"></span>. Dates marked <span class="dot"></span>
  are the three the manuscript scores.</p>

  <div class="cols">
    <div class="panel">
      <div class="extentbar">
        <span class="extentlab">Extent:</span>
        <span class="seg" id="extents"></span>
      </div>
      <canvas id="map"></canvas>
      <div class="legend">
        <span><span class="sw" style="background:var(--water)"></span>water (DEM ≤ threshold)</span>
        <span><span class="sw" style="background:var(--land)"></span>land</span>
        <span><span class="sw" style="background:var(--context)"></span>outside active extent</span>
      </div>
    </div>

    <div class="panel">
      <div class="readout">
        <div><span class="big" id="mval">–</span><span class="biglab">threshold (m, WGS84)</span></div>
        <div><span class="big" id="wfval">–</span><span class="biglab">water fraction of valid area</span></div>
        <div><span class="big" id="kmval">–</span><span class="biglab">water area (km²)</span></div>
      </div>

      <div class="ctl">
        <input type="range" id="slider" min="0" max="254" step="1">
        <div class="scale"><span id="slo"></span><span id="shi"></span></div>
      </div>

      <div class="ptitle">Snap to a validation date</div>
      <div class="ptide">tide-gauge reading at the Sentinel-1 overpass</div>
      <div class="snaps" id="snaps"></div>

      <div class="chead">
        <div class="ptitle">Hypsometric curve</div>
        <label class="pill" id="scpill" title="Show one point per pairable Sentinel-1 scene of the reported frame, at its overpass tide level">
          <input type="checkbox" id="sctoggle"><span>scene samplings (<span id="scn"></span>)</span>
        </label>
      </div>
      <div class="ptide">flooded fraction of the valid area vs threshold</div>
      <div class="scnote" id="scnote"></div>
      <svg id="curve" width="100%" role="img"></svg>
    </div>
  </div>

  <details><summary>Show date table</summary><div id="table"></div></details>
</div>
<div class="tip" id="tip"></div>

<script>
const D=__PAYLOAD__;
const W=D.w,H=D.h,EDGES=D.edges,DATES=D.dates,EXTENTS=D.extents;
const PX_KM2=D.px_km2,SCENES=D.scenes,MSL=D.msl_m;

// Active extent: index into EXTENTS, which is also the bit index in the packed
// mask raster (bit0 reported, bit1 full_bay, bit2 bridge_south).
let ext=0;
const EX=()=>EXTENTS[ext];
const CURVE=()=>EXTENTS[ext].curve;

const fmtDate=d=>`${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)}`;
// A bin code c covers [EDGES[c-1], EDGES[c]); report its upper edge, which is the
// highest elevation the code floods (code 254 is the open-ended top bin).
const metres=c=>c>=EDGES.length?EDGES[EDGES.length-1]:EDGES[c];
// Every per-date water fraction is read off the ACTIVE extent's curve at that
// date's bin code -- never a baked-in literal, so the three extents stay honest.
const wfAt=code=>CURVE()[code];

document.getElementById('slo').textContent=metres(0).toFixed(2)+' m';
document.getElementById('shi').textContent='≥ '+metres(EDGES.length-1).toFixed(2)+' m';

const cv=document.getElementById('map');
cv.width=W; cv.height=H;
const ctx=cv.getContext('2d',{alpha:false});
const img=ctx.createImageData(W,H);
const buf=new Uint32Array(img.data.buffer);

let codes=null,shade=null,vmask=null,pending=null,lutWater=null,lutLand=null,lutContext=null;

function decode(dataUri){
  return new Promise((res,rej)=>{
    const im=new Image();
    im.onload=()=>{
      const c=document.createElement('canvas'); c.width=W; c.height=H;
      const cx=c.getContext('2d',{willReadFrequently:true});
      cx.drawImage(im,0,0);
      const d=cx.getImageData(0,0,W,H).data;
      const out=new Uint8Array(W*H);
      for(let i=0,j=0;i<out.length;i++,j+=4) out[i]=d[j];
      res(out);
    };
    im.onerror=rej;
    im.src=dataUri;
  });
}

function cssRGB(name){
  const v=getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const c=document.createElement('canvas'); c.width=c.height=1;
  const cx=c.getContext('2d'); cx.fillStyle=v; cx.fillRect(0,0,1,1);
  const p=cx.getImageData(0,0,1,1).data;
  return [p[0],p[1],p[2]];
}
// Pack RGBA little-endian for the Uint32 view (canvas byte order is R,G,B,A).
const pack=(r,g,b)=>(255<<24)|(b<<16)|(g<<8)|r;

function buildLUTs(){
  const w=cssRGB('--water'),l=cssRGB('--land'),v=cssRGB('--context');
  lutWater=new Uint32Array(256); lutLand=new Uint32Array(256); lutContext=new Uint32Array(256);
  for(let s=0;s<256;s++){
    // Hillshade modulates each base colour between 55% and 130% brightness.
    const f=0.55+(s/255)*0.75;
    lutWater[s]=pack(Math.min(255,w[0]*f|0),Math.min(255,w[1]*f|0),Math.min(255,w[2]*f|0));
    lutLand[s] =pack(Math.min(255,l[0]*f|0),Math.min(255,l[1]*f|0),Math.min(255,l[2]*f|0));
    // Out-of-extent context: keep the hillshade relief legible but pull it far
    // toward the flat --context tone (25% shading, 75% flat) so it reads as
    // background geography, not as data.
    const g=0.80+(s/255)*0.28, k=0.25;
    lutContext[s]=pack(
      Math.min(255,(v[0]*(1-k)+v[0]*g*k)|0),
      Math.min(255,(v[1]*(1-k)+v[1]*g*k)|0),
      Math.min(255,(v[2]*(1-k)+v[2]*g*k)|0));
  }
}

function render(code){
  if(!codes) return;
  const bit=1<<ext;
  for(let i=0;i<buf.length;i++){
    buf[i]= (vmask[i]&bit) ? (codes[i]<=code ? lutWater[shade[i]] : lutLand[shade[i]])
                           : lutContext[shade[i]];
  }
  ctx.putImageData(img,0,0);
}

const slider=document.getElementById('slider');
function update(){
  const code=+slider.value;
  document.getElementById('mval').textContent=metres(code).toFixed(3);
  const wf=wfAt(code);
  document.getElementById('wfval').textContent=wf.toFixed(5);
  document.getElementById('kmval').textContent=(wf*EX().valid_px*PX_KM2).toFixed(1);
  document.querySelectorAll('#snaps button').forEach(b=>
    b.classList.toggle('on',+b.dataset.code===code));
  moveCursor(code);
  if(pending===null) pending=requestAnimationFrame(()=>{pending=null;render(+slider.value);});
}
slider.addEventListener('input',update);

// snap buttons -- the wf annotation is re-derived from the active curve
function drawSnaps(){
  document.getElementById('snaps').innerHTML=DATES.map(d=>
    `<button data-code="${d.code}"><b>${fmtDate(d.date)}${d.scored?'<span class="dot"></span>':''}</b>`+
    `<i>${d.reading.toFixed(3)} m · wf ${wfAt(d.code).toFixed(3)}</i></button>`).join('');
  document.querySelectorAll('#snaps button').forEach(b=>
    b.addEventListener('click',()=>{slider.value=b.dataset.code;update();}));
}
drawSnaps();

// extent picker: swap the active mask bit, curve and readouts; the canvas keeps
// the full-bay size for all three, so nothing in the layout moves.
document.getElementById('extents').innerHTML=EXTENTS.map((e,i)=>
  `<button data-ext="${i}" title="${e.blurb}">${e.label}</button>`).join('');
function setExtent(i){
  ext=i;
  document.querySelectorAll('#extents button').forEach(b=>
    b.classList.toggle('on',+b.dataset.ext===ext));
  document.getElementById('vpx').textContent=EX().valid_px.toLocaleString();
  document.getElementById('extdesc').textContent=EX().blurb;
  document.getElementById('floorval').textContent=CURVE()[0].toFixed(3);
  drawCurve(); drawSnaps(); drawTable(); update();
}
document.querySelectorAll('#extents button').forEach(b=>
  b.addEventListener('click',()=>setExtent(+b.dataset.ext)));

// hypsometric curve. Only the polyline depends on the extent -- the axes, the
// date ticks and the code->metres mapping are shared, so switching extents swaps
// the curve in place without moving anything.
const CW=420,CH=190,cpL=44,cpR=10,cpT=8,cpB=26,NC=EDGES.length+1;
const cx_=c=>cpL+(c/(NC-1))*(CW-cpL-cpR);
const cy_=f=>cpT+(1-f)*(CH-cpT-cpB);
let cur=null,curdot=null,curpoly=null;

// scene-sampling overlay: one dot per pairable scene of the reported frame, at
// its own overpass threshold. Default off; drawn inside the curve SVG so the
// dots sit ON the active extent's curve and follow it when the extent changes.
const tip=document.getElementById('tip');
const showTip=(e,h)=>{tip.innerHTML=h;tip.style.opacity=1;
  tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8)+'px';
  tip.style.top=(e.clientY-tip.offsetHeight-10)+'px';};
const hideTip=()=>tip.style.opacity=0;
const sctoggle=document.getElementById('sctoggle');
let scenesOn=false;
document.getElementById('scn').textContent=SCENES.length;

function sceneTip(s){
  const wf=wfAt(s.code);
  const msl=(s.msl_m>=0?'+':'')+s.msl_m.toFixed(2);
  const kind=s.is_scored?' · scored date':(s.is_mask_date?' · validation date':'');
  return `<b>${fmtDate(s.date)}</b>${kind}<br>`+
    `${s.level_cm.toFixed(1)} cm gauge (${msl} m MSL) · threshold ${s.thr.toFixed(3)} m<br>`+
    `water fraction ${wf.toFixed(5)} · ${s.phase} (limb ${s.limb_frac.toFixed(2)})`+
    ` · spring/neap p${s.pctile.toFixed(0)}`;
}
// MSL reference: the same code axis, so it lands between the same bin edges as
// any other threshold in this frame.
const MSL_CODE=(()=>{let i=0; while(i<EDGES.length&&EDGES[i]<=MSL) i++; return i;})();

function sceneOverlaySVG(){
  if(!scenesOn) return '';
  let s=`<g id="scdots">`;
  const xm=cx_(MSL_CODE);
  s+=`<line x1="${xm.toFixed(1)}" y1="${cpT}" x2="${xm.toFixed(1)}" y2="${CH-cpB}" `+
     `stroke="var(--sceneline)" stroke-width="1" stroke-dasharray="1 3"/>`;
  s+=`<text class="axlab" x="${(xm+3).toFixed(1)}" y="${cpT+9}">MSL</text>`;
  SCENES.forEach((sc,i)=>{
    const x=cx_(sc.code).toFixed(1),y=cy_(wfAt(sc.code)).toFixed(1);
    const col=sc.is_scored?'var(--scored)':(sc.is_mask_date?'var(--accent)':'var(--scene)');
    // Whisker only on hover (a <g> hover rule would need CSS in the SVG; it is
    // toggled from JS instead), so 24 points on a monotone curve stay readable.
    s+=`<line class="whisk" x1="${x}" y1="${CH-cpB}" x2="${x}" y2="${y}" `+
       `stroke="${col}" stroke-width="0.9" opacity="0"/>`;
    s+=`<circle class="mk scdot" data-i="${i}" cx="${x}" cy="${y}" r="3" `+
       (sc.is_mask_date
         ? `fill="var(--surface)" stroke="${col}" stroke-width="1.8"`
         : `fill="${col}" fill-opacity="0.75" stroke="var(--surface)" stroke-width="0.7"`)+
       `/>`;
  });
  return s+`</g>`;
}

function drawCurve(){
  let svg=`<svg id="curve" viewBox="0 0 ${CW} ${CH}" width="100%" role="img">`;
  for(let k=0;k<=4;k++){const f=k/4;
    svg+=`<line x1="${cpL}" y1="${cy_(f)}" x2="${CW-cpR}" y2="${cy_(f)}" stroke="var(--grid)" stroke-width="1"/>`;
    svg+=`<text class="axlab" x="${cpL-6}" y="${cy_(f)+3.4}" text-anchor="end">${f.toFixed(2)}</text>`;}
  DATES.forEach(d=>{
    svg+=`<line x1="${cx_(d.code)}" y1="${cpT}" x2="${cx_(d.code)}" y2="${CH-cpB}" `+
         `stroke="${d.scored?'var(--scored)':'var(--muted)'}" stroke-width="1" stroke-dasharray="2 3" opacity="0.7"/>`;});
  svg+=`<polyline id="curpoly" fill="none" stroke="var(--accent)" stroke-width="1.8" points="`+
    CURVE().map((f,c)=>`${cx_(c).toFixed(1)},${cy_(f).toFixed(1)}`).join(' ')+`"/>`;
  svg+=sceneOverlaySVG();
  svg+=`<line id="cur" x1="0" y1="${cpT}" x2="0" y2="${CH-cpB}" stroke="var(--ink)" stroke-width="1.4"/>`;
  svg+=`<circle id="curdot" r="3.2" fill="var(--ink)"/>`;
  for(let k=0;k<=4;k++){const c=Math.round(k*(NC-1)/4);
    svg+=`<text class="axlab" x="${cx_(c)}" y="${CH-cpB+14}" text-anchor="middle">${metres(c).toFixed(1)}</text>`;}
  svg+=`<text class="axlab" x="${CW/2}" y="${CH-3}" text-anchor="middle">threshold (m, WGS84 ellipsoid)</text>`;
  svg+='</svg>';
  document.getElementById('curve').outerHTML=svg;
  cur=document.getElementById('cur'); curdot=document.getElementById('curdot');
  bindSceneDots();
}
function bindSceneDots(){
  document.querySelectorAll('.scdot').forEach(m=>{
    const whisk=m.previousElementSibling;
    m.addEventListener('mousemove',e=>showTip(e,sceneTip(SCENES[+m.dataset.i])));
    m.addEventListener('mouseenter',()=>{whisk.setAttribute('opacity','0.45');m.setAttribute('r','4.2');});
    m.addEventListener('mouseleave',()=>{whisk.setAttribute('opacity','0');m.setAttribute('r','3');hideTip();});
  });
}
function setScenes(on){
  scenesOn=on;
  sctoggle.checked=on;
  document.getElementById('scpill').classList.toggle('on',on);
  document.getElementById('scnote').classList.toggle('on',on);
  hideTip();
  drawCurve(); moveCursor(+slider.value);
}
document.getElementById('scnote').textContent=
  `Points: the ${SCENES.length} pairable Sentinel-1 scenes of the reported frame, each at its `+
  `overpass tide level; rings mark the ${DATES.length} validation dates.`;
sctoggle.addEventListener('change',()=>setScenes(sctoggle.checked));
drawCurve();
function moveCursor(code){
  const x=cx_(code); cur.setAttribute('x1',x); cur.setAttribute('x2',x);
  curdot.setAttribute('cx',x); curdot.setAttribute('cy',cy_(wfAt(code)));
}

// date table
function drawTable(){
  document.getElementById('table').innerHTML=
    '<table><thead><tr><th>Date</th><th>Reading (m)</th><th>Bin code</th>'+
    '<th>Water fraction</th><th>Water area (km²)</th><th>Scored</th></tr></thead><tbody>'+
    DATES.map(d=>`<tr><td>${fmtDate(d.date)}</td><td>${d.reading.toFixed(3)}</td><td>${d.code}</td>`+
      `<td>${wfAt(d.code).toFixed(5)}</td><td>${(wfAt(d.code)*EX().valid_px*PX_KM2).toFixed(1)}</td>`+
      `<td>${d.scored?'yes':'no'}</td></tr>`).join('')+
    '</tbody></table>';
}
drawTable();

// theme: rebuild the colour LUTs whenever the resolved theme changes
function onTheme(){buildLUTs();render(+slider.value);}
new MutationObserver(onTheme).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme:dark)').addEventListener('change',onTheme);

Promise.all([decode(D.elev_png),decode(D.hill_png),decode(D.mask_png)]).then(([c,s,m])=>{
  codes=c; shade=s; vmask=m; buildLUTs();
  slider.value=DATES.find(d=>d.date==='20210422').code;
  setExtent(0);
});
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nas-root", default=DEFAULT_NAS_ROOT, help="root holding the DEM (NAS)")
    ap.add_argument(
        "--mask-root",
        default=DEFAULT_MASK_ROOT,
        help="root holding the valid mask and staged flood masks (local mirror by default)",
    )
    ap.add_argument(
        "--extent-root",
        default=DEFAULT_EXTENT_ROOT,
        help="dir holding the two extended valid masks",
    )
    ap.add_argument(
        "--tidal-csv",
        default=DEFAULT_TIDAL_CSV,
        help="per-scene tidal conditions CSV for the scene-sampling overlay",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="output HTML path")
    args = ap.parse_args()

    nas_root = Path(args.nas_root)
    mask_root = Path(args.mask_root)
    extent_root = Path(args.extent_root)
    out = Path(args.out)

    t0 = time.time()
    dem_file = _require(dem_path(nas_root), "full-bay DEM")
    with rasterio.open(dem_file) as src:
        # Widen to float64 ONCE, here. Widening is exact, so it changes no
        # comparison as long as the cut value is float32-anchored: see
        # reading_f32 for why the raw decimal literal is not the cut value.
        dem = src.read(1).astype(np.float64)
    masks = load_extent_masks(mask_root, extent_root, dem.shape)
    print(f"read full-bay DEM {dem.shape} + 3 extent masks in {time.time() - t0:.2f} s")

    t0 = time.time()
    edges = build_edges(TIDE_READINGS)
    codes = quantize(dem, edges)
    print(f"quantized to {edges.size} edges / {edges.size + 1} bins in {time.time() - t0:.2f} s")

    scenes = load_scene_conditions(Path(args.tidal_csv), edges)
    print(
        f"loaded {len(scenes)} pairable scenes "
        f"({min(s['thr'] for s in scenes):.3f}-{max(s['thr'] for s in scenes):.3f} m); "
        f"5 validation thresholds and bin codes match TIDE_READINGS"
    )

    t0 = time.time()
    print("verifying against staged flood masks (northern VAL_AOI window):")
    water_fractions = verify_against_flood_masks(dem, codes, edges, mask_root, masks["reported"])
    print(f"all 5 dates reproduce bit-exactly ({time.time() - t0:.2f} s)")

    curves = {key: hypsometric_curve(codes, masks[key]) for key in EXTENT_KEYS}
    print("extents:")
    for key in EXTENT_KEYS:
        n = int(masks[key].sum())
        print(
            f"  {key:<13} {n:>9,} valid px  "
            f"({n * PIXEL_AREA_KM2:7.1f} km²)  water-fraction floor {curves[key][0]:.5f}"
        )
    # The reported extent's curve must agree with the directly-counted staged
    # water fractions -- the page reads every date annotation off the curve.
    for date, wf in water_fractions.items():
        c = code_for(TIDE_READINGS[date], edges)
        assert abs(curves["reported"][c] - wf) < 1e-6, f"{date}: curve {curves['reported'][c]} != {wf}"

    t0 = time.time()
    shade = hillshade(dem)
    packed = pack_masks(masks)
    elev_png = png_gray8(codes)
    hill_png = png_gray8(shade)
    mask_png = png_gray8(packed)
    print(
        f"encoded PNGs in {time.time() - t0:.2f} s: "
        f"elevation {len(elev_png) / 1024:.0f} KB, hillshade {len(hill_png) / 1024:.0f} KB, "
        f"extent mask {len(mask_png) / 1024:.0f} KB"
    )

    payload = {
        "w": int(dem.shape[1]),
        "h": int(dem.shape[0]),
        "px_km2": PIXEL_AREA_KM2,
        "edges": [float(e) for e in edges],
        "extents": [
            {
                "key": key,
                "label": EXTENT_LABELS[key],
                "blurb": EXTENT_BLURBS[key],
                "valid_px": int(masks[key].sum()),
                "curve": curves[key],
            }
            for key in EXTENT_KEYS
        ],
        "dates": [
            {
                "date": date,
                "reading": reading,
                "code": code_for(reading, edges),
                "scored": date in SCORED_DATES,
            }
            for date, reading in sorted(TIDE_READINGS.items())
        ],
        "scenes": scenes,
        "msl_m": MSL_DEM_M,
        "elev_png": "data:image/png;base64," + base64.b64encode(elev_png).decode("ascii"),
        "hill_png": "data:image/png;base64," + base64.b64encode(hill_png).decode("ascii"),
        "mask_png": "data:image/png;base64," + base64.b64encode(mask_png).decode("ascii"),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(payload), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
