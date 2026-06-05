"""
Collate bootstrap_summary.csv files from all subdirectories of a target directory.

For each subdirectory that contains a bootstrap_summary.csv the script:
  - extracts run_name by stripping the trailing __<timestamp> suffix
  - prepends run_name and run_dir columns to each row
  - concatenates everything into a single CSV

Usage
-----
    python scripts/evaluation/collate_bootstrap_summaries.py \\
        [target_dir] \\
        [--output path/to/collated.csv]

Defaults
--------
    target_dir : outputs/evaluation/indonesia_inference_run_bootstrap
    output     : <target_dir>/collated_bootstrap_summary.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

TIMESTAMP_SUFFIX_RE = re.compile(r"__\d{8}T\d{6}Z$")


def run_name_from_dirname(dirname: str) -> str:
    return TIMESTAMP_SUFFIX_RE.sub("", dirname)


def collect(target_dir: Path) -> pd.DataFrame:
    frames = []
    missing = []

    for child in sorted(target_dir.iterdir()):
        if not child.is_dir():
            continue
        csv_path = child / "bootstrap_summary.csv"
        if not csv_path.exists():
            missing.append(child.name)
            continue

        df = pd.read_csv(csv_path)
        df.insert(0, "run_name", run_name_from_dirname(child.name))
        df.insert(1, "run_dir", child.name)
        frames.append(df)

    if missing:
        print(f"Skipped (no bootstrap_summary.csv): {missing}", file=sys.stderr)

    if not frames:
        raise RuntimeError(f"No bootstrap_summary.csv files found under {target_dir}")

    return pd.concat(frames, ignore_index=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collate bootstrap_summary.csv files into one CSV.")
    p.add_argument(
        "target_dir",
        nargs="?",
        default="outputs/evaluation/indonesia_inference_run_bootstrap",
        help="Root directory containing per-run subdirectories (default: outputs/evaluation/indonesia_inference_run_bootstrap)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <target_dir>/collated_bootstrap_summary.csv)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    target_dir = Path(args.target_dir)

    if not target_dir.is_dir():
        print(f"ERROR: target directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else target_dir / "collated_bootstrap_summary.csv"

    print(f"Scanning: {target_dir}")
    df = collect(target_dir)
    df.to_csv(output_path, index=False)

    print(f"Collated {df['run_dir'].nunique()} runs, {len(df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
