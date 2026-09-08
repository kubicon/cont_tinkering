"""Checks for `games/discretized.py` and the discretized Kuhn strategy reader.

The wrapper's claim is a strong one -- that a discretized run is the *same*
trainer, rollout and loss on a different action set -- so what is pinned here is
exactly that claim:

  * a policy on a wrapped game can only ever play a grid point, checked by
    playing hands and looking at what reached the underlying game's state;
  * it plays through its categorical head alone (`atom_frac` is exactly 1);
  * the tree underneath is untouched -- same horizon, same observations, same
    payoffs;
  * and the strategy read back out of such a policy is the strategy it actually
    plays, checked by comparing the exact value computed from the tables against
    the value of playing them out.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_X64_BEFORE = jax.config.jax_enable_x64

from baselines.neural import sequential_oracle as so  # noqa: E402
from games.discretized import (  # noqa: E402
    DiscretizedSequentialGame,
    base_game,
    discretize,
    linear_action_grid,
)
from games.kuhn_best_response import bet_grid, game_value  # noqa: E402
from games.sequential_examples import KIND_BET, KIND_PASSIVE, ContinuousKuhnPoker  # noqa: E402
from games.spaces import HybridAction  # noqa: E402
from training.best_response import PairEvaluator  # noqa: E402
from training.hyperparams import build_hyperparams  # noqa: E402
from training.kuhn_evaluation import strategy_from_policy  # noqa: E402
from training.mixture import build_mixture_network  # noqa: E402
from training.run_config import load_run_config  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)

BINS = 4


@pytest.fixture(scope="module")
def kuhn():
    """Continuous-bet Kuhn, and the same game on a four-size grid."""
    game = ContinuousKuhnPoker(num_cards=3, min_bet=0.25, max_bet=2.0)
    return game, DiscretizedSequentialGame(game, BINS)


def _policies(game, seed: int = 0):
    """A pair of untrained policies for `game`, and the network they share."""
    config = load_run_config("configs/kuhn_solvers.yaml")
    hyperparams = [build_hyperparams(game, player, config) for player in (0, 1)]
    network = build_mixture_network(hyperparams[0])
    state = game.initial_state(jax.random.PRNGKey(seed))
    params = [network.init(jax.random.PRNGKey(seed + player),
                           game.observation(player, state))
              for player in (0, 1)]
    return network, params, hyperparams


# --------------------------------------------------------------------------- the grid

def test_the_grid_is_linear_and_includes_both_endpoints():
    space = ContinuousKuhnPoker(min_bet=0.5, max_bet=2.0).action_space(0)
    grid = linear_action_grid(space, 4)
    assert grid.shape == (4, 1)
    assert np.allclose(grid[:, 0], [0.5, 1.0, 1.5, 2.0])


def test_a_degenerate_box_collapses_to_its_single_size():
    space = ContinuousKuhnPoker(min_bet=1.0, max_bet=1.0).action_space(0)
    assert np.allclose(linear_action_grid(space, 5), 1.0)


def test_the_grid_refuses_to_get_too_wide():
    game = ContinuousKuhnPoker()
    with pytest.raises(ValueError, match="max_actions"):
        DiscretizedSequentialGame(game, bins=64, max_actions=16)


def test_discretize_is_a_no_op_without_bins():
    game = ContinuousKuhnPoker()
    assert discretize(game, 0) is game
    assert discretize(game, None) is game
    assert base_game(discretize(game, 4)) is game
    assert base_game(game) is game


# --------------------------------------------------------------------------- the wrapper

def test_the_continuous_branch_is_illegal_everywhere(kuhn):
    game, discrete = kuhn
    space = discrete.action_space(0)
    assert space.num_atoms == 2 + BINS      # check/fold, call, and one kind per size
    state = game.initial_state(jax.random.PRNGKey(0))
    for player in (0, 1):
        mask = np.asarray(discrete.action_mask(player, state))
        assert mask.shape == (space.num_atoms + 1,)
        assert not mask[-1]                 # the continuous branch
        assert mask.any()                   # ... and something is always legal
        # Betting is legal at the opening node, so every size is.
        assert mask[2:2 + BINS].all()


def test_the_tree_underneath_is_untouched(kuhn):
    game, discrete = kuhn
    assert discrete.max_steps == game.max_steps
    assert discrete.obs_dim(0) == game.obs_dim(0)
    state = game.initial_state(jax.random.PRNGKey(3))
    assert int(discrete.current_player(state)) == int(game.current_player(state))
    assert np.allclose(discrete.observation(1, state), game.observation(1, state))
    assert float(discrete.payoff(state)) == float(game.payoff(state))


def test_a_grid_kind_becomes_the_size_it_names(kuhn):
    _, discrete = kuhn
    state = discrete.initial_state(jax.random.PRNGKey(0))   # player 0 to act
    sizes = np.asarray(discrete.grid(0))[:, 0]
    for index, size in enumerate(sizes):
        action = HybridAction(kind=jnp.asarray(2 + index, dtype=jnp.int32),
                              value=jnp.asarray([99.0]))    # ignored: the kind names the size
        base = discrete.base_action(state, action)
        assert int(base.kind) == KIND_BET
        assert float(base.value[0]) == pytest.approx(float(size))
    # The original atoms keep their meaning and their index.
    passive = discrete.base_action(
        state, HybridAction(kind=jnp.asarray(KIND_PASSIVE, dtype=jnp.int32),
                            value=jnp.asarray([0.0])))
    assert int(passive.kind) == KIND_PASSIVE


def test_only_grid_sizes_ever_reach_the_game(kuhn):
    """The property the whole wrapper exists for, checked by playing hands."""
    _, discrete = kuhn
    action_fns = (discrete.random_action_fn(0), discrete.random_action_fn(1))
    states, _ = jax.vmap(lambda key: discrete.play_episode(action_fns, key))(
        jax.random.split(jax.random.PRNGKey(0), 512))
    bets = np.asarray(states.bet)
    played = bets[bets > 0]
    assert played.size > 0                                   # the test would be vacuous otherwise
    sizes = np.asarray(discrete.grid(0))[:, 0]
    assert np.all(np.min(np.abs(played[:, None] - sizes[None, :]), axis=1) < 1e-6)


def test_a_discretized_run_uses_the_categorical_head_alone(kuhn):
    """`atom_frac` is the share of sampled actions that were atoms. On a wrapped
    game it must be exactly 1: the Gaussian branch is masked out of every state,
    so the loss reduces to its categorical factor."""
    _, discrete = kuhn
    config = load_run_config("configs/kuhn_solvers.yaml")
    hyperparams = so.br_hyperparams(discrete, 0, config)
    assert hyperparams.num_atoms == 2 + BINS
    opponent = so.initial_policy(discrete, 1, hyperparams, 1)
    response = so.train_sequential_best_response(
        discrete, 0, opponent, hyperparams, steps=1, epochs=2, seed=0)
    assert all(row["atom_frac"] == pytest.approx(1.0) for row in response.history)


# --------------------------------------------------------------------------- reading it back

def test_the_discrete_reader_returns_a_valid_strategy(kuhn):
    game, discrete = kuhn
    network, params, _ = _policies(discrete)
    grid = bet_grid(game, 13)          # 13 points: the 4 bins land exactly on 0, 4, 8, 12
    strategy = strategy_from_policy(discrete, network, params[0], 0, grid)
    strategy.validate()

    sizes = np.asarray(discrete.grid(0))[:, 0]
    on_grid = np.isclose(np.asarray(grid)[None, :], sizes[:, None], atol=1e-6).any(axis=0)
    assert np.allclose(np.asarray(strategy.open_bet)[:, ~on_grid], 0.0)
    assert np.asarray(strategy.open_bet)[:, on_grid].sum() > 0.0


def test_the_strategy_read_back_is_the_one_that_gets_played(kuhn):
    """The end-to-end check on both the wrapper and the reader: the value computed
    exactly from the tables must be the value of actually playing the policies."""
    game, discrete = kuhn
    network, params, _ = _policies(discrete, seed=5)
    grid = bet_grid(game, 13)
    strategies = [strategy_from_policy(discrete, network, params[player], player, grid)
                  for player in (0, 1)]
    exact = float(game_value(game, grid, strategies[0], strategies[1]))

    played = PairEvaluator(discrete, (network, network)).evaluate(
        (params[0], params[1]), 0, jax.random.PRNGKey(0), num_episodes=40_000)
    assert exact == pytest.approx(played.value, abs=5 * played.stderr + 1e-3)
