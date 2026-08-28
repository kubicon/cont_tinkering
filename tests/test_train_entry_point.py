"""Checks that `train.py` routes both families of game off one config schema."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest
import yaml

import train
from games.base import ZeroSumGame
from games.configs import GAME_CONFIGS
from games.sequential import TERMINAL, SequentialZeroSumGame
from games.sequential_examples import ContinuousKuhnPoker
from games.spaces import hybrid
from training.run_config import load_run_config, run_config_from_dict
from training.sequential_trainer import SequentialSelfPlayPPOTrainer


def _config(path: str = "configs/kuhn_classic.yaml", **train_overrides):
    raw = yaml.safe_load(open(path))
    raw["train"].update(train_overrides)
    return run_config_from_dict(raw)


@pytest.mark.parametrize(
    "path",
    [
        "configs/kuhn.yaml",
        "configs/kuhn_classic.yaml",
        "configs/leduc.yaml",
        "configs/leduc_classic.yaml",
    ],
)
def test_the_shipped_sequential_configs_load_and_build(path):
    config = load_run_config(path)
    assert isinstance(config.game.build(), SequentialZeroSumGame)


def test_the_trees_are_registered_alongside_the_one_shot_games():
    """One registry, two kinds of game -- `train.py` tells them apart by type."""
    assert {"kuhn", "leduc"} <= set(GAME_CONFIGS)
    built = {name: cls().build() for name, cls in GAME_CONFIGS.items()}
    sequential = {n for n, g in built.items() if isinstance(g, SequentialZeroSumGame)}
    one_shot = {n for n, g in built.items() if isinstance(g, ZeroSumGame)}
    assert sequential == {"kuhn", "leduc"}
    assert one_shot == set(GAME_CONFIGS) - sequential
    assert not (sequential & one_shot)  # nothing is both


def test_hyperparameters_come_off_the_action_space_for_a_sequential_game():
    """`build_hyperparams` is shared: a `HybridSpace` supplies bounds *and* atoms."""
    config = _config()
    game = config.game.build()
    hyperparams = train.build_hyperparams(game, 0, config)

    assert hyperparams.num_atoms == game.action_space(0).num_atoms == 2
    assert hyperparams.low == (game.min_bet,)
    assert hyperparams.high == (game.max_bet,)
    assert hyperparams.action_dim == 1


def test_a_one_shot_game_still_reports_no_atoms():
    config = run_config_from_dict(yaml.safe_load(open("configs/quadratic.yaml")))
    game = config.game.build()
    assert train.build_hyperparams(game, 0, config).num_atoms == 0


def test_fixed_opponent_is_rejected_for_a_sequential_game():
    """There is no fixed-opponent tree rollout; failing loudly beats failing deep."""
    config = _config(mode="fixed_opponent")
    with pytest.raises(ValueError, match="sequential"):
        train.run_sequential(config.game.build(), config)


def test_kuhn_gets_exploitability_and_strategy_hooks():
    config = _config()
    hooks = train.build_sequential_hooks(config.game.build(), config)
    assert set(hooks) == {"metric_fn", "strategy_log_fn"}


def test_leduc_trains_without_an_exploitability_hook():
    """Its public state carries real bet sizes, so there is no tabular best response."""
    config = _config("configs/leduc_classic.yaml")
    assert train.build_sequential_hooks(config.game.build(), config) == {}


class _CoinFlip(SequentialZeroSumGame):
    """The smallest possible sequential game: player 0 picks a number, then player 1 does.

    Exists to check two things Kuhn cannot -- that `SequentialZeroSumGame` is
    implementable by something that is not Kuhn, and that a sequential game with
    no exact best response still trains, just without an exploitability number.
    """

    def __init__(self):
        self._space = hybrid(1, [-1.0], [1.0])

    @property
    def max_steps(self) -> int:
        return 2

    def action_space(self, player: int):
        return self._space

    def obs_dim(self, player: int) -> int:
        return 2

    def initial_state(self, key):
        del key
        return {"turn": jnp.zeros((), jnp.int32), "first": jnp.zeros((), jnp.float32)}

    def current_player(self, state):
        return jnp.where(state["turn"] < 2, state["turn"], TERMINAL).astype(jnp.int32)

    def observation(self, player: int, state):
        return jnp.stack([state["turn"].astype(jnp.float32), state["first"]])

    def action_mask(self, player: int, state):
        return jnp.ones(2, dtype=bool)

    def payoff(self, state):
        return state["first"]

    def _step(self, state, action, key):
        del key
        value = jnp.squeeze(action.value, axis=-1)
        return {
            "turn": state["turn"] + 1,
            "first": jnp.where(state["turn"] == 0, value, state["first"]),
        }


def test_a_sequential_game_without_a_best_response_trains_without_hooks():
    """The hooks are optional: an unfamiliar tree simply reports no exploitability."""
    assert train.build_sequential_hooks(_CoinFlip(), _config()) == {}


def test_an_unfamiliar_sequential_game_still_trains():
    game = _CoinFlip()
    config = _config(steps=1, epochs=2, checkpoint_dir=None)
    hyperparams = train.build_hyperparams(game, 0, config)
    assert hyperparams.num_atoms == 1
    SequentialSelfPlayPPOTrainer(game, hyperparams, hyperparams, seed=0).train(1, epochs=2)


@pytest.mark.parametrize("path", ["configs/kuhn_classic.yaml", "configs/leduc_classic.yaml"])
def test_a_short_run_completes_through_the_entry_point(path):
    config = _config(path, steps=1, epochs=2, checkpoint_dir=None)
    train.run_sequential(config.game.build(), config)
