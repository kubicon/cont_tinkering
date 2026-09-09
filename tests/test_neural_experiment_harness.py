"""Checks for `experiments/one_shot_neural/`: the budget arithmetic and the run/score loop.

The harness's job is to make different methods comparable, and the two places that can
quietly destroy that are the cost model (if a method's payoff-evaluation count is wrong,
every plot is wrong) and the split between training and measurement (if scoring leaks into
the timed region, the wall-time plot is a plot of the metric). Both are pinned here, along
with an end-to-end cell -> checkpoints -> scores pass.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import jax
import numpy as np
import pytest

_X64_BEFORE = jax.config.jax_enable_x64

from baselines.common import load_checkpoints  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)

HARNESS = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "one_shot_neural"


def _load(name: str):
    """The harness scripts are scripts, not modules."""
    path = HARNESS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module      # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_cell():
    jax.config.update("jax_enable_x64", True)
    return _load("run_cell")


@pytest.fixture(scope="module")
def score():
    return _load("score")


def test_budget_buys_the_same_payoff_evaluations_in_every_method(run_cell):
    """The whole point of the harness: one budget, one currency. Each method's planned
    unit count times its per-unit cost must come back to (just under) the budget."""
    settings = run_cell.Settings(br_steps=10, br_epochs=10, payoff_samples=64, atoms=8)
    budget, batch = 2_000_000, 256
    for method in run_cell.METHODS:
        plan = run_cell.plan_units(method, budget, settings, batch)
        units = plan.get("iterations", plan.get("rounds"))
        spent = units * plan["evals_per_unit"]
        if method == "psro":     # plus the empirical matrix over the grown populations
            spent += (units + 1) ** 2 * plan["matrix_evals_per_pair"]
        assert spent <= budget, (method, spent)
        assert spent > 0.5 * budget, (method, spent)   # and it must actually spend it


def test_psro_budget_accounts_for_the_empirical_matrix(run_cell):
    """PSRO's payoff matrix grows quadratically with the population; a planner that
    ignored it would hand PSRO several times the budget of every other method."""
    settings = run_cell.Settings(br_steps=10, br_epochs=10, payoff_samples=1024)
    cheap = run_cell.plan_units("psro", 5_000_000, settings, 256)["rounds"]
    settings_free = run_cell.Settings(br_steps=10, br_epochs=10, payoff_samples=1)
    free = run_cell.plan_units("psro", 5_000_000, settings_free, 256)["rounds"]
    assert cheap < free


def test_chunk_alignment_avoids_a_second_compilation(run_cell):
    units, log_every = run_cell._aligned(1000, 7)
    assert units % log_every == 0
    assert units <= 1000 and units >= 1000 - log_every


def test_cell_runs_scores_and_is_skipped_on_a_rerun(run_cell, score, tmp_path):
    settings = run_cell.Settings(checkpoints=2, grid=101, samples=256, bins=21)
    row = run_cell.run_cell("configs/two_point.yaml", "mmd_discrete", 0, 4000, settings,
                            tmp_path, score=False)
    assert row["status"] == "ok"
    directory = tmp_path / "two_point" / "mmd_discrete" / "seed0"

    history = json.loads((directory / "history.json").read_text())
    # Fast path: costs are recorded, exploitability is not (that is score.py's job).
    assert all("wall_time" in h and "payoff_evals" in h for h in history)
    assert not any("expl" in h for h in history)
    assert history[-1]["payoff_evals"] <= 4000
    assert history[-1]["wall_time"] > 0.0

    checkpoints = load_checkpoints(directory / "checkpoints")
    assert len(checkpoints) == len(history)
    assert np.allclose(checkpoints[-1].weights_0.sum(), 1.0)

    from baselines.common import GridOracle, load_game
    oracle = GridOracle(load_game("configs/two_point.yaml")[0], points=101)
    scores = score.score_run(directory, oracle)
    assert len(scores) == len(checkpoints)
    assert all(s["expl"] >= 0.0 for s in scores)
    # The cost columns survive the join with the run's own history.
    assert [s["payoff_evals"] for s in scores] == [h["payoff_evals"] for h in history]
    assert "target_expl" in scores[-1]     # the discretized policy's averaged iterate

    stamp = (directory / "meta.json").stat().st_mtime_ns
    again = run_cell.run_cell("configs/two_point.yaml", "mmd_discrete", 0, 4000, settings,
                              tmp_path, score=False)
    assert again["status"] == "ok"
    assert (directory / "meta.json").stat().st_mtime_ns == stamp    # skipped, not rerun


def test_failed_cell_is_recorded_not_raised(run_cell, tmp_path):
    settings = run_cell.Settings(checkpoints=1, grid=101)
    row = run_cell.run_cell("configs/blotto.yaml", "mmd_discrete", 0, 1000, settings, tmp_path)
    assert row["status"] == "failed"
    assert (tmp_path / "blotto" / "mmd_discrete" / "seed0" / "error.txt").exists()


def test_compile_warning_fires_on_an_underbudgeted_best_response(run_cell):
    settings = run_cell.Settings(br_steps=2, br_epochs=5)
    assert run_cell.compile_warning("nfsp", {"rounds": 20}, settings) is not None
    assert run_cell.compile_warning("psro", {"rounds": 20}, settings) is not None
    # A properly budgeted best response, and a method that compiles once, are both fine.
    assert run_cell.compile_warning("nfsp", {"rounds": 5},
                                    run_cell.Settings(br_steps=50, br_epochs=20)) is None
    assert run_cell.compile_warning("mmd_discrete", {"iterations": 10}, settings) is None


def test_budget_warning_fires_when_one_unit_overspends(run_cell):
    """PSRO pays for its whole empirical payoff matrix before any best response, so a
    small budget cannot buy even one round -- and the unit count is floored at 1."""
    settings = run_cell.Settings(br_steps=2, br_epochs=5, payoff_samples=256)
    plan = run_cell.plan_units("psro", 60_000, settings, 256)
    assert run_cell.budget_warning("psro", plan, settings, 60_000) is not None
    big = run_cell.plan_units("psro", 20_000_000, settings, 256)
    assert run_cell.budget_warning("psro", big, settings, 20_000_000) is None


def test_access_model_is_recorded_for_every_method(run_cell):
    """A comparison that hides which methods get exact gradients is not a fair one.

    Keyed on `ALL_METHODS`, not the default grid: `OPTIONAL_METHODS` are still runnable
    by name, so a run of one still has to carry its access model into `meta.json`.
    """
    assert set(run_cell.ACCESS_MODEL) == set(run_cell.ALL_METHODS)
    assert "exact" in run_cell.ACCESS_MODEL["sisa"]
    assert "zeroth order" in run_cell.ACCESS_MODEL["jpspg"]
    assert "pathwise" in run_cell.ACCESS_MODEL["rpn_pathwise"]


def test_every_runnable_method_is_wired_end_to_end(run_cell):
    """The grid and the opt-in pair must both be priceable and runnable -- a method in
    `ALL_METHODS` with no runner or no cost formula fails only once a cell is launched."""
    assert set(run_cell.RUNNERS) == set(run_cell.ALL_METHODS)
    assert not set(run_cell.METHODS) & set(run_cell.OPTIONAL_METHODS)
    settings = run_cell.Settings()
    for method in run_cell.ALL_METHODS:
        plan = run_cell.plan_units(method, 20_000_000, settings, 256)
        assert plan["evals_per_unit"] > 0, method
        assert plan.get("iterations", plan.get("rounds")) >= 1, method
