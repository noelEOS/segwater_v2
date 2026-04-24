import os
import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from dotenv import load_dotenv

def cleanup_zombies(study_name: str):
    load_dotenv()
    db_url = os.getenv("NEON_DB_URL")
    
    storage = RDBStorage(
        url=db_url,
        engine_kwargs={"pool_pre_ping": True}
    )

    study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Identify trials that are stuck in RUNNING
    running_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    
    if not running_trials:
        print(f"No running trials found for study: {study_name}")
        return

    print(f"Found {len(running_trials)} trials marked as RUNNING.")
    
    for trial in running_trials:
        # CRITICAL: Do not kill the one you know is actually active!
        # You can check trial.number or just be careful if running this while training.
        confirm = input(f"Set Trial {trial.number} to FAIL? (y/n): ")
        if confirm.lower() == 'y':
            storage.set_trial_state(trial._trial_id, TrialState.FAIL)
            print(f"Trial {trial.number} has been set to FAIL.")

if __name__ == "__main__":
    # Replace with the specific study name you identified
    cleanup_zombies("hpo_unetplusplus_resnet50")