import os
import json
import csv
import argparse
import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from dotenv import load_dotenv


PARAM_TO_HYDRA = {
    "base_learning_rate": "trainer.base_learning_rate",
    "weight_decay": "trainer.weight_decay",
    "label_smoothing": "model.label_smoothing",
    "dice_weight": "model.dice_weight",
}


def make_storage(db_url: str):
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

    # Your optimize.py derives ce_weight from dice_weight
    if "dice_weight" in params:
        ce_weight = 1.0 - params["dice_weight"]
        overrides.append(f"model.ce_weight={ce_weight}")

    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/optuna_best_configs")
    args = parser.parse_args()

    load_dotenv()
    db_url = os.getenv("NEON_DB_URL")

    if not db_url:
        raise RuntimeError("NEON_DB_URL not found. Put it in .env or export it.")

    os.makedirs(args.out_dir, exist_ok=True)

    storage = make_storage(db_url)
    summaries = optuna.get_all_study_summaries(storage)

    rows = []

    for summary in summaries:
        study_name = summary.study_name
        study = optuna.load_study(study_name=study_name, storage=storage)

        complete_trials = study.get_trials(
            deepcopy=False,
            states=(TrialState.COMPLETE,),
        )

        if not complete_trials:
            rows.append({
                "study_name": study_name,
                "status": "NO_COMPLETE_TRIALS",
                "best_trial_number": None,
                "best_value": None,
                "best_params": {},
                "hydra_overrides": [],
                "hydra_overrides_one_line": "",
            })
            continue

        best = study.best_trial
        params = dict(best.params)
        hydra_overrides = hydra_overrides_from_params(params)

        rows.append({
            "study_name": study_name,
            "status": "OK",
            "best_trial_number": best.number,
            "best_value": best.value,
            "best_params": params,
            "hydra_overrides": hydra_overrides,
            "hydra_overrides_one_line": " ".join(hydra_overrides),
        })

    json_path = os.path.join(args.out_dir, "best_configs.json")
    csv_path = os.path.join(args.out_dir, "best_configs.csv")

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "study_name",
                "status",
                "best_trial_number",
                "best_value",
                "hydra_overrides_one_line",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "study_name": row["study_name"],
                "status": row["status"],
                "best_trial_number": row["best_trial_number"],
                "best_value": row["best_value"],
                "hydra_overrides_one_line": row["hydra_overrides_one_line"],
            })

    print("\nBest configs by study\n" + "=" * 80)

    for row in rows:
        print(f"\nStudy: {row['study_name']}")
        print(f"Status: {row['status']}")

        if row["status"] != "OK":
            continue

        print(f"Best trial: {row['best_trial_number']}")
        print(f"Best value: {row['best_value']:.6f}")
        print("Hydra overrides:")
        for override in row["hydra_overrides"]:
            print(f"  {override}")

        print("\nOne-line:")
        print(row["hydra_overrides_one_line"])

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()