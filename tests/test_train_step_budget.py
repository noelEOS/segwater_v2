"""Tests for the stage-2 (train.py) step-budget computation under gradient
accumulation, validating the two pipeline invariants:

  1. NO-ACCUM BEHAVIOUR UNCHANGED: with accumulate_grad_batches=1 the budget is
     exactly the previous epochs * len(train_dl).

  2. ACCUM == EQUIVALENT LARGE-BATCH: a run with (accum=k, bs=B) behaves like a
     run with (accum=1, bs=k*B) at the same epochs -- same optimizer-step
     budget (hence same LR schedule), provided the loader length divides
     cleanly. With drop_last=False a non-divisible loader length may differ by
     up to ~epochs steps; that bound is asserted explicitly.

These import the real helper used by scripts/train.py so the shipped arithmetic
is under test, not a copy of it.
"""

import math

import torch

from src.engine.trainer import compute_total_steps


# --- Invariant 1: accum=1 reproduces the old budget exactly -----------------

def test_accum_one_budget_is_unchanged():
    for n in (1, 7, 1000, 4100, 99999):
        assert compute_total_steps(n, epochs=3, accumulate_grad_batches=1) == 3 * n


def test_accum_one_matches_prefix_formula_over_many_sizes():
    # Spot-check across epochs too; accum=1 must always equal epochs * len.
    for epochs in (1, 10, 25):
        for n in (1234, 4100, 50000):
            assert compute_total_steps(n, epochs, 1) == epochs * n


# --- Invariant 2: accum=k,bs=B equivalent to accum=1,bs=k*B ------------------

def _len_train_dl(n_samples, batch_size, drop_last=False):
    # Mirrors torch DataLoader length with the project's drop_last=False.
    if drop_last:
        return n_samples // batch_size
    return math.ceil(n_samples / batch_size)


def test_equivalent_configs_match_when_divisible():
    # N chosen so both loaders divide cleanly: budgets must be identical.
    N, epochs = 256_000, 10
    a = compute_total_steps(_len_train_dl(N, 128), epochs, accumulate_grad_batches=2)
    b = compute_total_steps(_len_train_dl(N, 256), epochs, accumulate_grad_batches=1)
    assert a == b == 10_000


def test_equivalent_configs_drift_bounded_by_epochs():
    # With drop_last=False and non-divisible N, the two budgets may differ, but
    # never by more than ~epochs steps (one steps_per_epoch rounding * epochs).
    epochs = 10
    for N in (256_001, 255_999, 199_999, 123_457, 300_017, 1):
        a = compute_total_steps(_len_train_dl(N, 128), epochs, 2)
        b = compute_total_steps(_len_train_dl(N, 256), epochs, 1)
        assert abs(a - b) <= epochs, f"N={N}: drift {abs(a-b)} exceeds {epochs}"


def test_higher_accum_equivalence_and_drift():
    # Generalise to accum=4,bs=64 vs accum=1,bs=256.
    epochs = 10
    N = 256_000
    a = compute_total_steps(_len_train_dl(N, 64), epochs, 4)
    b = compute_total_steps(_len_train_dl(N, 256), epochs, 1)
    assert a == b


# --- Scheduler-level consequence: the budget makes the cosine complete -------

def test_cosine_completes_decay_with_accum_budget():
    # The whole point of the fix: a warmup+cosine sized with the accum-aware
    # budget, stepped once per optimizer step, lands on eta_min by the end.
    peak, eta_min, warmup = 4.5593328470445344e-05, 1e-6, 1000
    total_steps = compute_total_steps(_len_train_dl(256_000, 128), epochs=10, accumulate_grad_batches=2)

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=peak)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup)
    decay_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup), eta_min=eta_min)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, decay_sched], milestones=[warmup])

    for _ in range(total_steps):
        opt.step()
        sched.step()

    final_lr = opt.param_groups[0]["lr"]
    assert math.isclose(final_lr, eta_min, rel_tol=1e-6, abs_tol=1e-9), f"final LR {final_lr:.3e} != eta_min"


def test_miscounted_budget_leaves_lr_high():
    # Negative control: the OLD micro-batch budget (no // accum) leaves the LR
    # far above eta_min at the end of the optimizer steps actually run.
    peak, eta_min, warmup = 4.5593328470445344e-05, 1e-6, 1000
    epochs, accum = 10, 2
    buggy_total = epochs * _len_train_dl(256_000, 128)   # micro-batch count
    opt_steps_run = buggy_total // accum                  # what the loop runs

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=peak)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup)
    decay_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, buggy_total - warmup), eta_min=eta_min)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, decay_sched], milestones=[warmup])
    for _ in range(opt_steps_run):
        opt.step()
        sched.step()

    assert opt.param_groups[0]["lr"] > 0.4 * peak, "expected buggy budget to leave LR high"
