"""Export W&B run curves (train/val IoU, per-term losses, LR) to local CSV/JSON.

Training metrics are logged only to W&B (see src/engine/trainer.py); nothing is
written to disk alongside the checkpoints. This module pulls a run's full logged
history back down via the public API so the curves can be archived next to the
weights or replotted offline without hitting the W&B server again.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb


def _resolve_run(api: wandb.Api, run_ref: str, project: str | None) -> "wandb.apis.public.Run":
    """Return a W&B Run from an 'entity/project/run_id' path, a run URL, or a bare id+project."""
    ref = run_ref.strip()

    # A full run URL: https://wandb.ai/<entity>/<project>/runs/<id>?...
    if ref.startswith("http"):
        parts = ref.split("?")[0].rstrip("/").split("/")
        entity, project_name, run_id = parts[-4], parts[-3], parts[-1]
        return api.run(f"{entity}/{project_name}/{run_id}")

    # An 'entity/project/run_id' or 'project/run_id' path is accepted directly.
    if ref.count("/") >= 1:
        return api.run(ref)

    # A bare run id needs the project to disambiguate.
    if project is None:
        raise ValueError(
            f"Run reference '{run_ref}' is a bare id; pass --project (and optionally an entity prefix)."
        )
    return api.run(f"{project}/{ref}")


def _fetch_history(run: "wandb.apis.public.Run") -> pd.DataFrame:
    """Return a run's full logged history as a DataFrame.

    Prefers run.history(), which uses W&B's pure-Python/GraphQL path. scan_history()
    would stream every row exactly, but some wandb builds route it through a native
    parquet lib that fails to load on Apple Silicon, so it is only a fallback. The
    large `samples` cap keeps history() effectively complete rather than downsampled.
    """
    try:
        return run.history(samples=1_000_000, pandas=True)
    except Exception:
        return pd.DataFrame(list(run.scan_history()))


def export_run_history(
    run_ref: str,
    output_dir: str | Path,
    project: str | None = None,
    history_name: str = "history.csv",
    summary_name: str = "summary.json",
    config_name: str = "config.json",
) -> dict[str, Path]:
    """Fetch a W&B run's full metric history and write it locally as CSV/JSON.

    Args:
        run_ref: 'entity/project/run_id' path, a run URL, or a bare run id (needs project).
        output_dir: Directory to write the files into (created if missing). Use the run's
            local output_dir (e.g. outputs/stage2/.../<timestamp>_s42) to colocate with weights.
        project: 'entity/project' or 'project', required only when run_ref is a bare id.
        history_name / summary_name / config_name: Output filenames.

    Returns:
        Mapping of {"history": path, "summary": path, "config": path} for the files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = _resolve_run(api, run_ref, project)

    history = _fetch_history(run)
    if "global_step" in history.columns:
        history = history.sort_values("global_step").reset_index(drop=True)

    history_path = output_dir / history_name
    history.to_csv(history_path, index=False)

    summary_path = output_dir / summary_name
    summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    config_path = output_dir / config_name
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    config_path.write_text(json.dumps(config, indent=2, default=str))

    print(
        f"Exported {len(history)} logged steps from '{run.entity}/{run.project}/{run.id}' "
        f"({run.name}) -> {output_dir}"
    )
    return {"history": history_path, "summary": summary_path, "config": config_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_ref",
        help="W&B run: 'entity/project/run_id' path, a run URL, or a bare run id (with --project).",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write history.csv / summary.json / config.json into.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="'entity/project' or 'project', required only when run_ref is a bare run id.",
    )
    args = parser.parse_args()

    export_run_history(args.run_ref, args.output_dir, project=args.project)


if __name__ == "__main__":
    main()
