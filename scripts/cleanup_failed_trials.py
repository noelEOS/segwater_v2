import os
import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from dotenv import load_dotenv

def hunt_zombies():
    load_dotenv()
    db_url = os.getenv("NEON_DB_URL")
    
    storage = RDBStorage(
        url=db_url,
        engine_kwargs={"pool_pre_ping": True}
    )

    # Get all studies to audit the whole database
    summaries = optuna.get_all_study_summaries(storage)
    
    for summary in summaries:
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        running_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        
        if not running_trials:
            continue

        print(f"\n--- Study: {summary.study_name} ---")
        print(f"Found {len(running_trials)} trials marked as RUNNING.")
        
        for trial in running_trials:
            # SAFETY CHECK: Confirm before killing
            confirm = input(f"  > Set Trial {trial.number} to FAIL? (y/n/skip): ")
            if confirm.lower() == 'y':
                # The Senior Engineer way: uses the high-level 'tell' API
                study.tell(trial.number, TrialState.FAIL)
                print(f"    Trial {trial.number} set to FAIL.")
            else:
                print(f"    Trial {trial.number} kept as RUNNING.")

if __name__ == "__main__":
    hunt_zombies()