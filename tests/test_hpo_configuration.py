import optuna
import pytest

from src.utils.hpo import (
    build_pruner,
    resolve_hpo_schedule,
    suggest_hyperparameters,
)


class _RecordingTrial:
    def __init__(self):
        self.calls = {}

    def suggest_float(self, name, low, high, log):
        self.calls[name] = {"low": low, "high": high, "log": log}
        return (low + high) / 2


def test_mixed_forward_search_space_bounds():
    trial = _RecordingTrial()
    search_space = {
        "base_learning_rate": {"low": 3e-5, "high": 1e-3, "log": True},
        "weight_decay": {"low": 1e-7, "high": 1e-1, "log": True},
        "label_smoothing": {"low": 0.0, "high": 0.12, "log": False},
        "dice_weight": {"low": 0.0, "high": 0.5, "log": False},
    }

    suggest_hyperparameters(trial, search_space)

    assert trial.calls == search_space


def test_successive_halving_rungs_are_two_four_eight_epochs():
    pruner = build_pruner({
        "type": "successive_halving",
        "min_resource": 4786,
        "reduction_factor": 2,
        "min_early_stopping_rate": 0,
        "bootstrap_count": 0,
    })

    assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)
    assert pruner._min_resource == 4786
    assert pruner._reduction_factor == 2
    assert [4786 * 2**i for i in range(3)] == [4786, 9572, 19144]


def test_hpo_schedule_keeps_full_cosine_horizon():
    schedule = resolve_hpo_schedule(
        {
            "max_steps": 23930,
            "scheduler_total_steps": 23930,
            "warmup_steps": 500,
            "val_check_interval": 4786,
        },
        {
            "max_steps": 1500,
            "warmup_steps": 500,
            "val_check_interval": 400,
        },
    )

    assert schedule == {
        "max_steps": 23930,
        "scheduler_total_steps": 23930,
        "warmup_steps": 500,
        "val_check_interval": 4786,
    }


def test_scheduler_horizon_cannot_end_before_trial():
    with pytest.raises(ValueError, match="scheduler_total_steps"):
        resolve_hpo_schedule(
            {
                "max_steps": 23930,
                "scheduler_total_steps": 19144,
                "warmup_steps": 500,
                "val_check_interval": 4786,
            },
            {},
        )
