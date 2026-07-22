"""Apply a fitted Platt transform p' = sigmoid(a*logit(p) + b) to probability rasters.

Phase 3 of Option 2 (docs/threshold_calibration/PLAN_option2_platt_scaling.md).
No re-inference: the transform is applied to EXISTING *_probability_water.tif
files. p is clamped to [eps, 1-eps] (eps = 1e-6) before the logit; saturated
(quantized) pixels stay saturated and the binary decision is unaffected by the
clamp (the transform is monotone). Originals are never overwritten: calibrated
rasters are written under a parallel subfolder, same filename, float32.

(a, b) come either from --platt-params (platt_params.csv written by
fit_platt_scaling.py, row selected by --name) or explicitly via --a/--b.

Usage:
    python scripts/evaluation/apply_platt.py \
        --platt-params outputs/evaluation/val_split_calibration/platt_params.csv \
        --name upernet_tu-swin_base_patch4_window7_224_ensemble \
        --out-subdir platt_calibrated \
        raster1_probability_water.tif [raster2.tif ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

EPS = 1e-6


def platt_transform(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p64 = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    z = a * (np.log(p64) - np.log1p(-p64)) + b
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rasters", nargs="+", type=Path)
    ap.add_argument("--platt-params", type=Path, default=None,
                    help="platt_params.csv from fit_platt_scaling.py")
    ap.add_argument("--name", default=None,
                    help="Row (name column) in --platt-params to use")
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--out-subdir", default="platt_calibrated",
                    help="Subfolder (next to each input) for calibrated rasters")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.platt_params is not None:
        if args.name is None:
            raise SystemExit("--name is required with --platt-params")
        if args.a is not None or args.b is not None:
            raise SystemExit("--platt-params and explicit --a/--b are mutually exclusive")
        table = pd.read_csv(args.platt_params).set_index("name")
        if args.name not in table.index:
            raise SystemExit(f"No row named {args.name} in {args.platt_params}")
        row = table.loc[args.name]
        a, b = float(row["a"]), float(row["b"])
        source = f"{args.platt_params}::{args.name}"
    elif args.a is not None and args.b is not None:
        a, b = args.a, args.b
        source = "explicit --a/--b"
    else:
        raise SystemExit("Provide --platt-params/--name or both --a and --b")
    if a <= 0:
        raise SystemExit(f"a = {a} <= 0 is not a valid Platt slope")

    tau_eq = 1.0 / (1.0 + np.exp(b / a))
    print(f"Applying p' = sigmoid({a:.6f}*logit(p) {b:+.6f})  "
          f"[tau_eq = {tau_eq:.4f}; decisions at p'>=0.5 == p>=tau_eq]")

    for src_path in args.rasters:
        if not src_path.exists():
            raise SystemExit(f"Missing raster: {src_path}")
        out_dir = src_path.parent / args.out_subdir
        out_path = out_dir / src_path.name
        if out_path.exists() and not args.overwrite:
            print(f"  skip (exists): {out_path}")
            continue
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            data = src.read(1)
        profile.update(dtype="float32", count=1)
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(platt_transform(data, a, b), 1)
        prov = {"source_raster": str(src_path), "a": a, "b": b,
                "tau_eq": float(tau_eq), "eps_clamp": EPS, "params_from": source}
        (out_dir / (src_path.stem + "_platt_provenance.json")).write_text(
            json.dumps(prov, indent=2))
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
