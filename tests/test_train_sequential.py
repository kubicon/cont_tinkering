"""Checks for `train_sequential.py` and the run accounting behind it.

The script's whole reason to exist is that three algorithms with nothing in
common structurally can be compared, so what is pinned here is the part that
makes a comparison possible rather than anything about the training:

  * every solver writes the *same* cost columns, and they are the units a
    comparison is plotted against -- wall time, episodes, environment steps;
  * the environment-step count is exact where it claims to be (a batch of
    `num_envs` episodes of known length) and estimated only where it says so;
  * every solver's checkpoint is loadable, by the repo's own loader, under the
    entry names that make `best_response.py` able to read it blind;
  * the config schema carries each algorithm's own schedule, and rejects a
    solver it does not have.
"""

from __future__ import annotations

import dataclasses
import json

import jax
import numpy as np
import pytest

_X64_BEFORE = jax.config.jax_enable_x64

import train_sequential as ts  # noqa: E402
from baselines.neural.sequential_run import (  # noqa: E402
    Budget,
    SequentialRunLog,
    Stopwatch,
    mean_columns,
)
from baselines.neural.sequential_scoring import build_scorer  # noqa: E402
from games.discretized import DiscretizedSequentialGame  # noqa: E402
from baselines.neural import sequential_oracle as so  # noqa: E402
from training.checkpoint import load_checkpoint_step_multi  # noqa: E402
from training.config import MixturePPOHyperparams  # noqa: E402
from training.run_config import load_run_config, run_config_from_dict  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)


@dataclasses.dataclass
class _Args:
    """The CLI flags `apply_overrides` folds in; `None` means "not given"."""

    solver: str | None = None
    checkpoint_dir: str | None = None
    seed: int | None = None
    score_every: int | None = None


# --------------------------------------------------------------------------- config

def test_the_solver_sections_parse_and_reach_the_config():
    config = run_config_from_dict({
        "game": {"name": "kuhn", "num_cards": 3},
        "train": {"solver": "psro", "seed": 3},
        "discrete": {"bins": 8},
        "rpn": {"iterations": 11, "estimator": "joint"},
        "nfsp": {"rounds": 7, "eta": 0.25},
        "psro": {"rounds": 5, "meta_solver": "uniform"},
        "scoring": {"score_every": 4},
    })
    assert config.train.solver == "psro"
    assert (config.nfsp.rounds, config.nfsp.eta) == (7, 0.25)
    assert (config.psro.rounds, config.psro.meta_solver) == (5, "uniform")
    assert config.scoring.score_every == 4
    assert config.discrete.bins == 8
    assert (config.rpn.iterations, config.rpn.estimator) == (11, "joint")


def test_an_unknown_solver_is_rejected():
    with pytest.raises(ValueError, match="train.solver"):
        run_config_from_dict({"game": {"name": "kuhn"}, "train": {"solver": "cfr"}})


def test_the_existing_configs_still_load_and_default_to_self_play():
    for path in ("configs/kuhn.yaml", "configs/leduc.yaml", "configs/sequential_blotto.yaml",
                 "configs/multi_point.yaml"):
        assert load_run_config(path).train.solver == "self_play"
    # ... and the shipped comparison configs carry all three schedules.
    for path in ("configs/kuhn_solvers.yaml", "configs/leduc_solvers.yaml"):
        config = load_run_config(path)
        assert config.nfsp.rounds > 0 and config.psro.rounds > 0
        assert config.discrete.bins > 1
        assert config.rpn.iterations > 0


def test_cli_overrides_win_over_the_file():
    config = load_run_config("configs/kuhn_solvers.yaml")
    overridden = ts.apply_overrides(config, _Args(solver="nfsp", seed=0, score_every=0))
    assert overridden.train.solver == "nfsp"
    assert overridden.train.seed == 0            # 0 is a value, not "unset"
    assert overridden.scoring.score_every == 0
    assert overridden.train.checkpoint_dir == config.train.checkpoint_dir


# --------------------------------------------------------------------------- accounting

def test_training_steps_are_counted_exactly():
    """`num_envs` episodes per iteration, each of the length that iteration
    measured -- no estimate anywhere in this path."""
    budget = Budget(default_episode_length=3.0)
    rows = [{"episode_length": 2.0}, {"episode_length": 4.0}]
    budget.add_training(rows, num_envs=10)
    assert budget.iterations == 2
    assert budget.episodes == 20
    assert budget.env_steps == pytest.approx(10 * (2.0 + 4.0))
    assert budget.mean_episode_length == pytest.approx(3.0)


def test_evaluation_episodes_are_priced_at_the_measured_mean():
    budget = Budget(default_episode_length=8.0)
    # Before any training there is nothing to average, so the default (the game's
    # `max_steps`, an upper bound) stands in.
    budget.add_episodes(100)
    assert budget.env_steps == pytest.approx(800.0)
    budget.add_training([{"episode_length": 2.0}], num_envs=50)
    budget.add_episodes(100)
    assert budget.env_steps == pytest.approx(800.0 + 100.0 + 200.0)
    assert budget.episodes == 250
    assert budget.row()["iterations"] == 1


def test_mean_columns_averages_the_interval_and_skips_bookkeeping():
    rows = [{"iteration": 1, "loss": 1.0, "grad_norm": 4.0},
            {"iteration": 2, "loss": 3.0, "grad_norm": 6.0}]
    assert mean_columns(rows) == {"loss": 2.0, "grad_norm": 5.0}
    assert mean_columns(rows, prefix="br0_") == {"br0_loss": 2.0, "br0_grad_norm": 5.0}
    assert mean_columns([]) == {}


def test_the_stopwatch_leaves_out_what_it_is_stopped_for():
    clock = Stopwatch(running=True)
    clock.stop()
    paused = clock.seconds
    assert clock.seconds == paused        # stopped time does not accumulate
    clock.start()
    assert clock.seconds >= paused


def test_grad_norm_reaches_the_history():
    """`ppo_update` reports the pre-clipping gradient norm, so every solver's rows
    carry it without any of them asking for it."""
    game, _, config = _tiny_kuhn()
    hyperparams = so.br_hyperparams(game, 0, config)
    opponent = so.initial_policy(game, 1, hyperparams, 1)
    response = so.train_sequential_best_response(
        game, 0, opponent, hyperparams, steps=1, epochs=2, seed=0)
    assert all("grad_norm" in row and np.isfinite(row["grad_norm"]) for row in response.history)


# --------------------------------------------------------------------------- run log

def test_the_run_log_streams_rows_and_writes_loadable_checkpoints(tmp_path):
    game, _, config = _tiny_kuhn()
    hyperparams = so.br_hyperparams(game, 0, config)
    policy = so.initial_policy(game, 0, hyperparams, 0)

    log = SequentialRunLog(tmp_path / "run", {"solver": "test"})
    log.record({"t": 1, "wall_time": 0.5, "expl": np.float32(0.25)})
    log.checkpoint(1, {"player_0": (hyperparams, policy.params[0])},
                   {"meta_weights_0": np.ones(2)})
    log.finish({"totals": {"rows": 1}})

    streamed = [json.loads(line) for line in
                (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert len(streamed) == 1 and streamed[0]["t"] == 1
    # Every row carries the total clock beside whatever the solver put in it.
    assert "total_wall_time" in streamed[0]
    assert json.loads((tmp_path / "run" / "history.json").read_text()) == streamed
    assert json.loads((tmp_path / "run" / "meta.json").read_text())["totals"]["rows"] == 1

    entries = load_checkpoint_step_multi(tmp_path / "run" / "checkpoints", 1,
                                         hyperparams_cls=MixturePPOHyperparams)
    assert "player_0" in entries
    assert np.load(tmp_path / "run" / "checkpoints" / "1.npz")["meta_weights_0"].shape == (2,)


def test_a_run_log_without_a_directory_is_inert():
    log = SequentialRunLog(None, {"solver": "test"})
    log.record({"t": 1})
    log.checkpoint(1, {})
    assert log.finish()["solver"] == "test"
    assert len(log.history) == 1


# --------------------------------------------------------------------------- end to end

def _tiny_kuhn():
    """Classic Kuhn (exact metric) on the smallest schedule that exercises everything."""
    config = load_run_config("configs/kuhn_classic.yaml")
    config = dataclasses.replace(
        config,
        ppo=dataclasses.replace(config.ppo, batch_size=32),
        train=dataclasses.replace(config.train, steps=2, epochs=2, seed=0),
        nfsp=dataclasses.replace(config.nfsp, rounds=1, br_steps=1, br_epochs=2,
                                 reservoir_episodes=256, sl_steps=5, sl_batch=8),
        psro=dataclasses.replace(config.psro, rounds=1, br_steps=1, br_epochs=2,
                                 payoff_episodes=200),
        scoring=dataclasses.replace(config.scoring, score_every=0, episodes=200),
    )
    return config.game.build(), config.game, config


COST_COLUMNS = ("wall_time", "total_wall_time", "iterations", "episodes", "env_steps")


@pytest.mark.parametrize("solver", ["self_play", "discrete_mmd", "nfsp", "psro"])
def test_every_solver_writes_the_same_cost_columns(solver, tmp_path):
    game, _, config = _tiny_kuhn()
    config = dataclasses.replace(
        config, train=dataclasses.replace(config.train, solver=solver,
                                          checkpoint_dir=str(tmp_path / solver)))
    game = ts.prepare_game(game, config)
    log = SequentialRunLog(config.train.checkpoint_dir, {"solver": solver})
    ts.SOLVER_RUNNERS[solver](game, config, log, ts.build_run_scorer(game, config))
    log.finish()

    assert log.history
    for row in log.history:
        assert all(column in row for column in COST_COLUMNS), row
        # Kuhn is exactly scorable, so every solver reports the same headline.
        assert np.isfinite(row["expl"])
    # Cost is cumulative, and nothing is free.
    assert [row["episodes"] for row in log.history] == sorted(row["episodes"] for row in log.history)
    assert log.history[-1]["env_steps"] > 0
    assert log.history[-1]["iterations"] > 0
    # The loss and the gradient norm are there to be plotted beside the cost.
    keys = set(log.history[-1])
    assert any("loss" in k for k in keys)
    assert any("grad_norm" in k for k in keys)


@pytest.mark.parametrize("solver,entry", [("self_play", "player_0"),
                                          ("discrete_mmd", "player_0"),
                                          ("nfsp", "player_0"),
                                          ("psro", "player0_policy0")])
def test_every_solver_checkpoints_weights_that_load_back(solver, entry, tmp_path):
    game, _, config = _tiny_kuhn()
    config = dataclasses.replace(
        config, train=dataclasses.replace(config.train, solver=solver,
                                          checkpoint_dir=str(tmp_path / solver)))
    game = ts.prepare_game(game, config)
    log = SequentialRunLog(config.train.checkpoint_dir, {"solver": solver})
    ts.SOLVER_RUNNERS[solver](game, config, log, ts.build_run_scorer(game, config))
    log.finish()

    step = log.history[-1]["t"]
    entries = load_checkpoint_step_multi(tmp_path / solver / "checkpoints", step,
                                         hyperparams_cls=MixturePPOHyperparams)
    assert entry in entries, sorted(entries)
    # PSRO's strategy is a population, so its meta-weights ride beside the params;
    # the other two are a plain pair of policies and need nothing extra.
    if solver == "psro":
        weights = np.load(tmp_path / solver / "checkpoints" / f"{step}.npz")["meta_weights_0"]
        assert np.isclose(weights.sum(), 1.0)


def test_the_script_refuses_a_one_shot_game(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["train_sequential.py", "configs/multi_point.yaml",
                                     "--checkpoint-dir", str(tmp_path / "x")])
    # `main` sets the process-wide x64 flag (it runs in `train.py`'s precision,
    # not the one-shot baselines'), so it is put back for whatever runs next.
    before = jax.config.jax_enable_x64
    try:
        with pytest.raises(ValueError, match="one-shot game"):
            ts.main()
    finally:
        jax.config.update("jax_enable_x64", before)


def test_a_second_solver_will_not_overwrite_the_first_ones_directory(tmp_path):
    config = load_run_config("configs/kuhn_solvers.yaml")
    config = dataclasses.replace(config, train=dataclasses.replace(
        config.train, checkpoint_dir=str(tmp_path / "run"), solver="nfsp"))
    SequentialRunLog(config.train.checkpoint_dir, {"solver": "nfsp"})   # writes meta.json

    ts.check_directory_is_free(config)          # the same solver again is a re-run
    with pytest.raises(ValueError, match="already holds"):
        ts.check_directory_is_free(dataclasses.replace(
            config, train=dataclasses.replace(config.train, solver="psro")))


def test_only_discrete_mmd_gets_a_discretized_game():
    """`prepare_game` is the entire difference between `discrete_mmd` and
    `self_play`; the hyperparameters and the metric then follow from the game."""
    game, _, config = _tiny_kuhn()
    for solver in ("self_play", "nfsp", "psro", "rpn"):
        assert ts.prepare_game(game, dataclasses.replace(
            config, train=dataclasses.replace(config.train, solver=solver))) is game

    config = dataclasses.replace(
        config, train=dataclasses.replace(config.train, solver="discrete_mmd"),
        discrete=dataclasses.replace(config.discrete, bins=8))
    discrete = ts.prepare_game(game, config)
    assert isinstance(discrete, DiscretizedSequentialGame)
    assert discrete.action_space(0).num_atoms == game.action_space(0).num_atoms + 8
    # The dead Gaussian head must not reach the torso through its box penalty.
    assert ts.solver_hyperparams(discrete, 0, config).mean_box_penalty_coef == 0.0
    assert ts.solver_hyperparams(game, 0, config).mean_box_penalty_coef != 0.0


def test_describe_names_the_schedule_and_the_metric():
    game, _, config = _tiny_kuhn()
    for solver in ("self_play", "nfsp", "psro"):
        text = ts.describe(game, dataclasses.replace(
            config, train=dataclasses.replace(config.train, solver=solver)))
        assert solver in text and "exact" in text
    discrete = dataclasses.replace(
        config, train=dataclasses.replace(config.train, solver="discrete_mmd"))
    text = ts.describe(ts.prepare_game(game, discrete), discrete)
    assert "actions" in text and "bins" in text

    leduc = load_run_config("configs/leduc_solvers.yaml")
    assert "expl_lb" in ts.describe(leduc.game.build(), leduc)


def test_the_scorer_is_the_same_one_for_every_solver():
    """A comparison is only worth as much as its metric being one metric."""
    game, _, config = _tiny_kuhn()
    hyperparams = tuple(so.br_hyperparams(game, player, config) for player in (0, 1))
    reference = build_scorer(game, hyperparams, exact_grid=config.scoring.exact_grid,
                             episodes=config.scoring.episodes, seed=config.train.seed)
    policies = tuple(so.initial_policy(game, player, hyperparams[player], player)
                     for player in (0, 1))
    ours = ts.build_run_scorer(game, config)(policies, 1, full=False)
    assert ours["expl"] == pytest.approx(reference(policies, 1, full=False)["expl"])
