"""Fetch best Optuna configs from the Neon HPO database.

Select studies by exact name, glob pattern (family), or take everything.
Writes a plain-text report with the Hydra override line for each study.

Examples
--------
  # one study
  python scripts/analysis/get_best_hpo_configs.py --study hpo_pp_upernet_convnextv2_base

  # a family (glob)
  python scripts/analysis/get_best_hpo_configs.py --pattern '*_pp_*'

  # everything
  python scripts/analysis/get_best_hpo_configs.py --all
"""

import argparse
import fnmatch
import os
from datetime import datetime

import optuna
from dotenv import load_dotenv
from optuna.storages import RDBStorage
from optuna.trial import TrialState

PARAM_TO_HYDRA = {
    "base_learning_rate": "trainer.base_learning_rate",
    "weight_decay": "trainer.weight_decay",
    "label_smoothing": "model.label_smoothing",
    "dice_weight": "model.dice_weight",
}


def make_storage(db_url: str) -> RDBStorage:
    return RDBStorage(
        url=db_url,
        engine_kwargs={
            "pool_size": 10,
            "pool_recycle": 300,
            "pool_pre_ping": True,
            "connect_args": {"connect_timeout": 10},
        },
    )


def hydra_overrides_from_params(params: dict) -> list[str]:
    overrides = []

    for optuna_name, hydra_name in PARAM_TO_HYDRA.items():
        if optuna_name in params:
            overrides.append(f"{hydra_name}={params[optuna_name]}")

    # optimize.py derives ce_weight from dice_weight
    if "dice_weight" in params:
        overrides.append(f"model.ce_weight={1.0 - params['dice_weight']}")

    return overrides


def select_studies(all_names: list[str], args) -> list[str]:
    if args.all:
        return sorted(all_names)

    selected = []
    missing = []

    for name in args.study:
        if name in all_names:
            selected.append(name)
        else:
            missing.append(name)

    if missing:
        raise SystemExit(
            "Study not found in database: "
            + ", ".join(missing)
            + "\nAvailable studies:\n  "
            + "\n  ".join(sorted(all_names))
        )

    for pattern in args.pattern:
        matches = fnmatch.filter(all_names, pattern)
        if not matches:
            raise SystemExit(
                f"Pattern {pattern!r} matched no studies.\nAvailable studies:\n  "
                + "\n  ".join(sorted(all_names))
            )
        selected.extend(matches)

    return sorted(set(selected))


def report_study(study_name: str, storage: RDBStorage) -> list[str]:
    """Return the report lines for one study."""
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == TrialState.COMPLETE]

    lines = ["=" * 80, f"Study: {study_name}", "=" * 80]

    if not complete:
        lines.append("Status: NO_COMPLETE_TRIALS")
        lines.append(f"Trials: {len(trials)} total, 0 complete")
        lines.append("")
        return lines

    best = study.best_trial
    overrides = hydra_overrides_from_params(dict(best.params))

    counts = {
        state.name: sum(1 for t in trials if t.state == state)
        for state in (
            TrialState.COMPLETE,
            TrialState.PRUNED,
            TrialState.FAIL,
            TrialState.RUNNING,
        )
    }

    lines.append("Status: OK")
    lines.append(
        "Trials: {} total ({})".format(
            len(trials),
            ", ".join(f"{k.lower()}={v}" for k, v in counts.items()),
        )
    )
    lines.append(f"Best trial: {best.number}")
    lines.append(f"Best value: {best.value:.6f}")
    lines.append("")
    lines.append("Best params:")
    for key, value in sorted(best.params.items()):
        lines.append(f"  {key} = {value}")
    lines.append("")
    lines.append("Hydra overrides:")
    for override in overrides:
        lines.append(f"  {override}")
    lines.append("")
    lines.append("One-line:")
    lines.append("  " + " ".join(overrides))
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Write best Optuna configs for selected studies to a .txt file.",
    )
    parser.add_argument(
        "--study",
        action="append",
        default=[],
        metavar="NAME",
        help="Exact study name (repeatable).",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        metavar="GLOB",
        help="Glob over study names, e.g. '*_pp_*' (repeatable). Quote it so the "
        "shell does not expand it.",
    )
    parser.add_argument("--all", action="store_true", help="Select every study.")
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output .txt path (default: outputs/optuna_best_configs/"
        "best_configs_<timestamp>.txt).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list study names in the database and exit.",
    )
    args = parser.parse_args()

    load_dotenv()
    db_url = os.getenv("NEON_DB_URL")
    if not db_url:
        raise SystemExit("NEON_DB_URL not found. Put it in .env or export it.")

    storage = make_storage(db_url)
    all_names = [s.study_name for s in optuna.get_all_study_summaries(storage)]

    if args.list:
        for name in sorted(all_names):
            print(name)
        return

    if not (args.study or args.pattern or args.all):
        parser.error("Give at least one of --study, --pattern, or --all.")

    selected = select_studies(all_names, args)

    out_path = args.out
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = os.path.join(
            "outputs", "optuna_best_configs", f"best_configs_{stamp}.txt"
        )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    selector = []
    if args.all:
        selector.append("--all")
    selector += [f"--study {s}" for s in args.study]
    selector += [f"--pattern {p}" for p in args.pattern]

    header = [
        "Optuna best configs",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Selection: {' '.join(selector)}",
        f"Studies matched: {len(selected)}",
        "",
    ]

    body = []
    for name in selected:
        print(f"Fetching {name} ...")
        body.extend(report_study(name, storage))

    text = "\n".join(header + body)

    with open(out_path, "w") as f:
        f.write(text)

    print()
    print(text)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
