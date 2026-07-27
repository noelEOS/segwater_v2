"""Small, testable helpers for the Optuna HPO protocol."""

from collections.abc import Mapping

import optuna


DEFAULT_SEARCH_SPACE = {
    "base_learning_rate": {"low": 1e-5, "high": 1e-2, "log": True},
    "weight_decay": {"low": 1e-5, "high": 1e-1, "log": True},
    "label_smoothing": {"low": 0.0, "high": 0.2, "log": False},
    "dice_weight": {"low": 0.0, "high": 1.0, "log": False},
}


def suggest_hyperparameters(trial, search_space: Mapping | None = None) -> dict:
    """Sample the four optimizer/loss parameters from config-defined ranges."""
    search_space = search_space or {}
    values = {}
    for name, defaults in DEFAULT_SEARCH_SPACE.items():
        configured = search_space.get(name, {})
        low = float(configured.get("low", defaults["low"]))
        high = float(configured.get("high", defaults["high"]))
        log = bool(configured.get("log", defaults["log"]))
        values[name] = trial.suggest_float(name, low, high, log=log)
    return values


def resolve_hpo_schedule(hpo_cfg: Mapping | None, trainer_cfg: Mapping) -> dict:
    """Resolve trial stop/report steps separately from the scheduler horizon."""
    hpo_cfg = hpo_cfg or {}
    max_steps = int(
        hpo_cfg["max_steps"]
        if "max_steps" in hpo_cfg
        else trainer_cfg["max_steps"]
    )
    scheduler_total_steps = int(
        hpo_cfg.get("scheduler_total_steps", max_steps)
    )
    warmup_steps = int(
        hpo_cfg["warmup_steps"]
        if "warmup_steps" in hpo_cfg
        else trainer_cfg["warmup_steps"]
    )
    val_check_interval = int(
        hpo_cfg["val_check_interval"]
        if "val_check_interval" in hpo_cfg
        else trainer_cfg["val_check_interval"]
    )

    if max_steps <= 0:
        raise ValueError(f"hpo.max_steps must be > 0, got {max_steps}")
    if scheduler_total_steps < max_steps:
        raise ValueError(
            "hpo.scheduler_total_steps must be >= hpo.max_steps so pruned and "
            "completed trials share one full-horizon LR schedule"
        )
    if not 0 < warmup_steps < scheduler_total_steps:
        raise ValueError(
            "hpo.warmup_steps must be between 0 and scheduler_total_steps, got "
            f"{warmup_steps} and {scheduler_total_steps}"
        )
    if val_check_interval <= 0:
        raise ValueError(
            f"hpo.val_check_interval must be > 0, got {val_check_interval}"
        )

    return {
        "max_steps": max_steps,
        "scheduler_total_steps": scheduler_total_steps,
        "warmup_steps": warmup_steps,
        "val_check_interval": val_check_interval,
    }


def build_pruner(pruner_cfg: Mapping | None, legacy_min_resource: int = 800):
    """Construct the configured pruner while retaining legacy fallback support."""
    if not pruner_cfg:
        return optuna.pruners.HyperbandPruner(
            min_resource=int(legacy_min_resource)
        )

    pruner_type = str(pruner_cfg.get("type", "successive_halving")).lower()
    if pruner_type in {"successive_halving", "sha"}:
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=int(pruner_cfg.get("min_resource", 1)),
            reduction_factor=int(pruner_cfg.get("reduction_factor", 4)),
            min_early_stopping_rate=int(
                pruner_cfg.get("min_early_stopping_rate", 0)
            ),
            bootstrap_count=int(pruner_cfg.get("bootstrap_count", 0)),
        )
    if pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(
                pruner_cfg.get("min_resource", legacy_min_resource)
            ),
            reduction_factor=int(pruner_cfg.get("reduction_factor", 3)),
        )
    if pruner_type in {"none", "nop"}:
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unsupported pruner.type: {pruner_type}")
