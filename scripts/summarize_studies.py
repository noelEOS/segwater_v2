import os
import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from dotenv import load_dotenv
from tabulate import tabulate

def summarize_database():
    load_dotenv()
    db_url = os.getenv("NEON_DB_URL")

    if not db_url:
        print("Error: NEON_DB_URL not found in .env file.")
        return

    # Using the "Senior Engineer" storage config to ensure stable connection
    storage = RDBStorage(
        url=db_url,
        engine_kwargs={
            "pool_size": 10,
            "pool_recycle": 300,
            "pool_pre_ping": True,
            "connect_args": {"connect_timeout": 10}
        }
    )

    summaries = optuna.get_all_study_summaries(storage)
    
    if not summaries:
        print("No studies found in the database.")
        return

    table_data = []
    
    for s in summaries:
        study = optuna.load_study(study_name=s.study_name, storage=storage)
        trials = study.trials
        
        counts = {
            "TOTAL": len(trials),
            "COMPLETE": len([t for t in trials if t.state == TrialState.COMPLETE]),
            "PRUNED": len([t for t in trials if t.state == TrialState.PRUNED]),
            "FAIL": len([t for t in trials if t.state == TrialState.FAIL]),
            "RUNNING": len([t for t in trials if t.state == TrialState.RUNNING])
        }

        # Get best mIoU if available
        try:
            best_val = f"{study.best_value:.4f}"
        except ValueError:
            best_val = "N/A"

        table_data.append([
            s.study_name,
            counts["TOTAL"],
            counts["COMPLETE"],
            counts["PRUNED"],
            counts["FAIL"],
            counts["RUNNING"],
            best_val
        ])

    headers = ["Study Name", "Total", "Comp.", "Pruned", "Fail", "Run.", "Best mIoU"]
    print("\n" + "="*80)
    print("OPTUNA HPO DATABASE SUMMARY")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80 + "\n")

if __name__ == "__main__":
    summarize_database()