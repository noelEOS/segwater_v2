"""
Retrieve and print the best trial config from the Neon DB Optuna storage.

Usage:
    python scripts/get_best_config.py                    # best across all studies
    python scripts/get_best_config.py --study my_study   # best for one study
    python scripts/get_best_config.py --json             # output as JSON
"""
import argparse
import json
import os
import sys

import optuna
from dotenv import load_dotenv
from optuna.storages import RDBStorage


def build_storage(db_url: str) -> RDBStorage:
    return RDBStorage(
        url=db_url,
        engine_kwargs={
            "pool_size": 10,
            "pool_recycle": 300,
            "pool_pre_ping": True,
            "connect_args": {"connect_timeout": 10},
        },
    )


def best_trial_for_study(study_name: str, storage: RDBStorage):
    study = optuna.load_study(study_name=study_name, storage=storage)
    try:
        return study.best_trial, study_name
    except ValueError:
        return None, study_name


def main():
    parser = argparse.ArgumentParser(description="Get best Optuna trial config from Neon DB")
    parser.add_argument("--study", default=None, help="Study name (default: search all studies)")
    parser.add_argument("--json", action="store_true", help="Output params as JSON only")
    args = parser.parse_args()

    load_dotenv()
    db_url = os.getenv("NEON_DB_URL")
    if not db_url:
        print("Error: NEON_DB_URL not set in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    storage = build_storage(db_url)

    if args.study:
        study_names = [args.study]
    else:
        summaries = optuna.get_all_study_summaries(storage)
        if not summaries:
            print("No studies found in the database.", file=sys.stderr)
            sys.exit(1)
        study_names = [s.study_name for s in summaries]

    best_trial = None
    best_study = None

    for name in study_names:
        trial, study_name = best_trial_for_study(name, storage)
        if trial is None:
            continue
        if best_trial is None or trial.value > best_trial.value:
            best_trial = trial
            best_study = study_name

    if best_trial is None:
        print("No completed trials found.", file=sys.stderr)
        sys.exit(1)

    params = dict(best_trial.params)
    dice_key = next((k for k in params if "dice_weight" in k), None)
    if dice_key is not None:
        ce_key = dice_key.replace("dice_weight", "ce_weight")
        params[ce_key] = 1 - params[dice_key]

    if args.json:
        print(json.dumps(params, indent=2))
        return

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  Study      : {best_study}")
    print(f"  Trial #    : {best_trial.number}")
    print(f"  mIoU       : {best_trial.value:.4f}")
    print(f"  Datetime   : {best_trial.datetime_complete}")
    print("-" * 60)
    print("  PARAMS:")
    for k, v in sorted(params.items()):
        print(f"    {k:<35} {v}")
    if best_trial.user_attrs:
        print("-" * 60)
        print("  USER ATTRS:")
        for k, v in sorted(best_trial.user_attrs.items()):
            print(f"    {k:<35} {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
