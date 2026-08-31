"""hamp24 A4 — per-pixel error aggregates on the VM (one pass, full-bay grid).

For each of the 24 dates: read the 21 arms' probability rasters windowed to
the full-bay grid, threshold at 0.5, and against the full-bay GT compute
  * per-pixel cross-model error count `n_wrong` (0..21, SUPP03 construction),
  * per-model / per-extent elevation-binned error histograms, where the bin
    variable is dz = DEM - float32(threshold_date) (elevation above the
    instantaneous waterline), 10 cm bins over [-3, +3] m plus open ends,
  * consensus rows (n_wrong >= 18 of 21 = at least 6 of 7 architectures).

The three extents reuse the SAME arrays: nominal = top-window mask (GT
identity gated G3), bridge_south = full_bay AND the bridge AOI (gated G4).

Outputs (small, authorized for egress): per-date `error_frequency_<date>.npz`
(uint8 n_wrong + bool reference/valid + bounds, SUPP03 schema),
`persistence_consensus18.npz` (per-pixel count of dates with consensus
error), `elev_error_histograms.csv`.

Run:
  python scripts/evaluation/vm/hamp24_pixel_error_aggregates.py \
      --spec scripts/evaluation/vm/specs/hamp24_full_bay.yaml \
      --tidal-csv ~/hampyeong_ron_134_ts_16_sn_15_all24/tidal_conditions.csv \
      --dem ~/ancillary/hampyeong/dem/Hamp_bay_TDX_UTM_10m_edited_clipped_full_bay.tif \
      --nominal-mask ~/ancillary/hampyeong/nas_root/Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM/DEM_VALID_MASK_aoi.tif \
      --bridge-mask ~/ancillary/hampyeong/valid_masks_extents/DEM_VALID_MASK_aoi_extended_bridge_south.tif \
      --out-dir ~/hamp24_pixel_aggregates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_pairbased_hampyeong import audit, require  # noqa: E402

CONSENSUS_MIN = 18          # of 21 arms ~= 6 of 7 architectures
BIN_STEP = 0.10
BIN_LO, BIN_HI = -3.0, 3.0


def read1(path: Path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile, src.bounds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--tidal-csv", required=True, type=Path)
    ap.add_argument("--dem", required=True, type=Path)
    ap.add_argument("--nominal-mask", required=True, type=Path)
    ap.add_argument("--bridge-mask", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    spec.setdefault("repo", "/home/noel/segwater_v2")
    spec.setdefault("pair_root", "outputs/inference")
    spec.setdefault("variant_column", False)
    repo = Path(spec["repo"])
    pair_root = repo / spec["pair_root"]
    gt_dir = Path(spec["gt_dir"])
    fullbay_mask_path = Path(spec["valid_mask"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hist_out = args.out_dir / "elev_error_histograms.csv"
    require(not hist_out.exists(), f"refusing to clobber {hist_out}")

    dem, dem_prof, dem_bounds = read1(args.dem)
    fb_mask, fb_prof, _ = read1(fullbay_mask_path)
    fb_mask = fb_mask == 1
    require(fb_prof["transform"] == dem_prof["transform"], "full-bay mask grid != DEM grid")
    nom, nom_prof, _ = read1(args.nominal_mask)
    nominal = np.zeros(dem.shape, bool)
    nominal[: nom.shape[0]] = nom == 1
    bs, bs_prof, _ = read1(args.bridge_mask)
    require(bs_prof["transform"] == dem_prof["transform"], "bridge mask grid != DEM grid")
    bridge = bs == 1
    extents = {"nominal": nominal, "full_bay": fb_mask, "bridge_south": bridge}

    tidal = pd.read_csv(args.tidal_csv)
    tidal["date_utc"] = tidal.scene_id.str.slice(17, 25)
    thr = dict(zip(tidal.date_utc, tidal.dem_threshold_m))
    scenes = {str(s)[17:25]: f"{s}_Clipped" for s in tidal.scene_id}
    dates = sorted(scenes)
    require(len(dates) == 24, f"expected 24 dates, got {len(dates)}")

    entries = audit(spec, repo, pair_root)
    n_models = len(entries)

    # bin edges: [-inf, -3.0, -2.9, ..., +3.0, +inf]
    inner = np.round(np.arange(BIN_LO, BIN_HI + BIN_STEP / 2, BIN_STEP), 10)
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    n_bins = len(edges) - 1

    persistence = np.zeros(dem.shape, np.uint8)
    rows = []
    for date in dates:
        gt, gt_prof, gt_bounds = read1(gt_dir / f"DEM_FLOOD_S1_DESC_{date}_VAL_AOI.tif")
        require(gt_prof["transform"] == dem_prof["transform"], f"{date}: GT grid != DEM grid")
        gt = gt == 1
        cut = float(np.float32(thr[date]))
        dz = dem - cut
        bin_idx = np.digitize(dz, edges[1:-1])          # 0..n_bins-1
        scene = scenes[date]

        n_wrong = np.zeros(dem.shape, np.uint8)
        # per-extent per-bin valid-pixel counts (model-independent)
        for ext, m in extents.items():
            nv = np.bincount(bin_idx[m], minlength=n_bins)
            for b in range(n_bins):
                if nv[b]:
                    rows.append((ext, "__N_VALID__", date, b, int(nv[b]), 0, 0))

        preds = {}
        for e in entries:
            pred_path = (pair_root / e["resolved_run_dir"] / scene
                         / f"{scene}_probability_water.tif")
            with rasterio.open(pred_path) as src:
                win = from_bounds(*gt_bounds, transform=src.transform)
                prob = src.read(1, window=win, boundless=False)
            require(prob.shape == dem.shape, f"{date} {e['label']}: window shape {prob.shape}")
            pred = prob >= 0.5
            err = pred != gt
            n_wrong += err.astype(np.uint8)
            preds[(e["label"], e["seed"])] = pred

            fp = pred & ~gt
            fn = ~pred & gt
            for ext, m in extents.items():
                c_fp = np.bincount(bin_idx[m & fp], minlength=n_bins)
                c_fn = np.bincount(bin_idx[m & fn], minlength=n_bins)
                for b in range(n_bins):
                    if c_fp[b] or c_fn[b]:
                        rows.append((ext, f"{e['label']}|s{e['seed']}", date, b,
                                     0, int(c_fp[b]), int(c_fn[b])))

        consensus = n_wrong >= CONSENSUS_MIN
        persistence += consensus.astype(np.uint8)
        for ext, m in extents.items():
            c_err = np.bincount(bin_idx[m & consensus], minlength=n_bins)
            for b in range(n_bins):
                if c_err[b]:
                    rows.append((ext, "__CONSENSUS18__", date, b, 0, int(c_err[b]), 0))

        np.savez_compressed(
            args.out_dir / f"error_frequency_{date}.npz",
            n_wrong=n_wrong, reference_water=gt, valid=fb_mask,
            bounds=np.array(dem_bounds, dtype=np.float64),
            n_models=np.array([n_models]),
            threshold_m=np.array([cut]),
        )
        print(f"{date}: done (consensus px {int(consensus[fb_mask].sum()):,})", flush=True)

    np.savez_compressed(args.out_dir / "persistence_consensus18.npz",
                        n_dates_consensus=persistence, valid=fb_mask, dem=dem.astype(np.float32),
                        bounds=np.array(dem_bounds, dtype=np.float64),
                        n_dates=np.array([len(dates)]), consensus_min=np.array([CONSENSUS_MIN]))

    df = pd.DataFrame(rows, columns=["extent", "model", "date", "bin", "n_valid", "n_fp", "n_fn"])
    df["bin_lo"] = [float(edges[b]) for b in df.bin]
    df["bin_hi"] = [float(edges[b + 1]) for b in df.bin]
    df.to_csv(hist_out, index=False)
    print(f"wrote {hist_out} ({len(df)} rows) + 24 npz + persistence map")


if __name__ == "__main__":
    main()
