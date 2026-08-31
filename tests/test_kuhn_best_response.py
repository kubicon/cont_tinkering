"""Checks for the exact Kuhn best response and the policy tables it consumes.

Three independent things are being pinned, deliberately by three different
routes -- the risk with a piece of arithmetic this dense is that it is checked
only against itself:

  * against a **closed form**: the analytic Kuhn equilibrium must come out at
    value `-1/18` with exploitability zero,
  * against **actual gameplay**: `game_value` must agree with a Monte-Carlo
    rollout of the same tables through the real `step` function, which also
    validates the observation layout the tables are read through,
  * against **hand computation**: a best response to a strategy that always
    folds must be worth exactly `+1`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.kuhn_best_response import (
    KuhnStrategy,
    analytic_equilibrium,
    best_response_value_first,
    best_response_value_second,
    bet_grid,
    exploitability,
    game_value,
)
from games.sequential_examples import (
    KIND_BET,
    KIND_CALL,
    KIND_PASSIVE,
    NUM_NODES,
    ContinuousKuhnPoker,
)
from games.spaces import HybridAction
from training.config import MixturePPOHyperparams
from training.kuhn_evaluation import (
    clipped_mixture_grid_probs,
    evaluate_networks,
    strategy_from_network,
)
from training.mixture import build_mixture_network

KUHN_VALUE = -1.0 / 18.0


def _random_strategy(key, num_cards: int, num_grid: int) -> KuhnStrategy:
    open_key, call_key = jax.random.split(key)
    raw = jax.random.uniform(open_key, (num_cards, num_grid + 1))
    normalized = raw / jnp.sum(raw, axis=-1, keepdims=True)
    return KuhnStrategy(
        open_check=normalized[:, 0],
        open_bet=normalized[:, 1:],
        call=jax.random.uniform(call_key, (num_cards, num_grid)),
    )


def _networks(game: ContinuousKuhnPoker, seed: int = 0, num_components: int = 2):
    space = game.action_space(0)
    hyperparams = MixturePPOHyperparams(
        action_dim=1,
        hidden_dims=(16,),
        num_components=num_components,
        num_atoms=space.num_atoms,
        low=(float(space.low[0]),),
        high=(float(space.high[0]),),
    )
    networks = (build_mixture_network(hyperparams), build_mixture_network(hyperparams))
    keys = jax.random.split(jax.random.PRNGKey(seed))
    state = game.initial_state(jax.random.PRNGKey(99))
    params = tuple(
        net.init(k, game.observation(p, state)) for p, (net, k) in enumerate(zip(networks, keys))
    )
    return networks, params


# ---- against the closed form --------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0])
def test_the_analytic_equilibrium_has_value_minus_one_eighteenth(alpha):
    game = ContinuousKuhnPoker(min_bet=1.0, max_bet=1.0)
    grid = bet_grid(game)
    first, second = analytic_equilibrium(game, alpha=alpha)
    first.validate()
    second.validate()
    assert float(game_value(game, grid, first, second)) == pytest.approx(KUHN_VALUE, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0])
def test_the_analytic_equilibrium_is_unexploitable(alpha):
    """The whole one-parameter family, not just one member of it."""
    game = ContinuousKuhnPoker(min_bet=1.0, max_bet=1.0)
    grid = bet_grid(game)
    first, second = analytic_equilibrium(game, alpha=alpha)
    assert float(exploitability(game, grid, first, second)) == pytest.approx(0.0, abs=1e-6)


def test_neither_best_response_beats_the_game_value_at_equilibrium():
    """`BR_0 >= v` and `BR_1 >= -v` always; at equilibrium both are tight."""
    game = ContinuousKuhnPoker(min_bet=1.0, max_bet=1.0)
    grid = bet_grid(game)
    first, second = analytic_equilibrium(game)
    assert float(best_response_value_first(game, grid, second)) == pytest.approx(KUHN_VALUE, abs=1e-6)
    assert float(best_response_value_second(game, grid, first)) == pytest.approx(-KUHN_VALUE, abs=1e-6)


# ---- against gameplay ----------------------------------------------------


def _table_action_fn(game: ContinuousKuhnPoker, strategy: KuhnStrategy, grid, player: int):
    """Play a `KuhnStrategy` through the real game, decoding the observation.

    Deliberately reads the infoset back out of `obs` rather than out of the
    state: if the tables were read through a layout that does not match what the
    game emits, this is where it shows up.
    """
    num_cards = game.num_cards
    open_node, faced_node = game.decision_nodes(player)

    def action_fn(obs, mask, key):
        del mask
        card = jnp.argmax(obs[:num_cards])
        node = jnp.argmax(obs[num_cards : num_cards + NUM_NODES])
        open_key, call_key = jax.random.split(key)

        weights = jnp.concatenate([strategy.open_check[card][None], strategy.open_bet[card]])
        choice = jax.random.categorical(open_key, jnp.log(weights + 1e-30))
        open_kind = jnp.where(choice == 0, KIND_PASSIVE, KIND_BET)
        open_value = grid[jnp.maximum(choice - 1, 0)]

        index = jnp.argmin(jnp.abs(grid - obs[-1] * game.max_bet))
        calls = jax.random.uniform(call_key) < strategy.call[card, index]
        faced_kind = jnp.where(calls, KIND_CALL, KIND_PASSIVE)

        at_open = node == open_node
        return HybridAction(
            kind=jnp.where(at_open, open_kind, faced_kind).astype(jnp.int32),
            value=jnp.where(at_open, open_value, grid[0])[None],
        )

    return action_fn


@pytest.mark.parametrize("min_bet, max_bet, num_points", [(1.0, 1.0, 1), (0.5, 2.0, 9)])
def test_game_value_agrees_with_monte_carlo_gameplay(min_bet, max_bet, num_points):
    """The tensor arithmetic, checked against the game actually being played."""
    game = ContinuousKuhnPoker(min_bet=min_bet, max_bet=max_bet)
    grid = bet_grid(game, num_points)
    key_0, key_1, play_key = jax.random.split(jax.random.PRNGKey(7), 3)
    strategy_0 = _random_strategy(key_0, game.num_cards, grid.shape[0])
    strategy_1 = _random_strategy(key_1, game.num_cards, grid.shape[0])

    action_fns = (
        _table_action_fn(game, strategy_0, grid, 0),
        _table_action_fn(game, strategy_1, grid, 1),
    )
    play = jax.jit(jax.vmap(lambda k: game.play_episode(action_fns, k)))
    _, payoff = play(jax.random.split(play_key, 400_000))

    standard_error = float(jnp.std(payoff)) / jnp.sqrt(payoff.shape[0])
    assert float(jnp.mean(payoff)) == pytest.approx(
        float(game_value(game, grid, strategy_0, strategy_1)), abs=4 * standard_error
    )


# ---- against hand computation -------------------------------------------


def test_best_response_to_a_pushover_is_worth_exactly_one():
    """An opponent who never bets and always folds hands over an ante every deal."""
    game = ContinuousKuhnPoker(min_bet=1.0, max_bet=1.0)
    grid = bet_grid(game)
    pushover = KuhnStrategy(
        open_check=jnp.ones(game.num_cards),
        open_bet=jnp.zeros((game.num_cards, 1)),
        call=jnp.zeros((game.num_cards, 1)),
    )
    assert float(best_response_value_first(game, grid, pushover)) == pytest.approx(1.0, abs=1e-6)


def test_exploitability_is_non_negative_for_arbitrary_strategies():
    game = ContinuousKuhnPoker(min_bet=0.5, max_bet=2.0)
    grid = bet_grid(game, 65)
    for seed in range(12):
        key_0, key_1 = jax.random.split(jax.random.PRNGKey(seed))
        value = float(
            exploitability(
                game,
                grid,
                _random_strategy(key_0, game.num_cards, grid.shape[0]),
                _random_strategy(key_1, game.num_cards, grid.shape[0]),
            )
        )
        assert value >= -1e-6, f"seed {seed} gave {value}"


# ---- the clipped mixture -------------------------------------------------


def test_clipped_mixture_probabilities_sum_to_the_component_weights():
    grid = jnp.linspace(0.5, 2.0, 129)
    weights = jnp.array([0.3, 0.2])
    probs = clipped_mixture_grid_probs(
        weights, jnp.array([0.8, 1.6]), jnp.array([0.2, 0.4]), grid
    )
    assert float(jnp.sum(probs)) == pytest.approx(float(jnp.sum(weights)), rel=1e-5)


@pytest.mark.parametrize("mean, expect_low, expect_high", [(-5.0, 1.0, 0.0), (9.0, 0.0, 1.0)])
def test_clipping_mass_lands_on_the_endpoints(mean, expect_low, expect_high):
    """Means outside the box are the common case early in training, not an edge case."""
    grid = jnp.linspace(0.5, 2.0, 129)
    probs = clipped_mixture_grid_probs(
        jnp.array([1.0]), jnp.array([mean]), jnp.array([0.5]), grid
    )
    assert float(probs[0]) == pytest.approx(expect_low, abs=1e-6)
    assert float(probs[-1]) == pytest.approx(expect_high, abs=1e-6)


def test_clipped_mixture_matches_sampling_the_same_clipped_mixture():
    grid = jnp.linspace(0.5, 2.0, 33)
    weights, means, stds = (
        jnp.array([0.6, 0.4]),
        jnp.array([0.7, 1.7]),
        jnp.array([0.3, 0.25]),
    )
    probs = clipped_mixture_grid_probs(weights, means, stds, grid)

    key = jax.random.PRNGKey(0)
    component_key, noise_key = jax.random.split(key)
    which = jax.random.categorical(component_key, jnp.log(weights), shape=(200_000,))
    raw = means[which] + stds[which] * jax.random.normal(noise_key, (200_000,))
    clipped = jnp.clip(raw, grid[0], grid[-1])
    cell = jnp.argmin(jnp.abs(clipped[:, None] - grid[None, :]), axis=-1)
    empirical = jnp.bincount(cell, length=grid.shape[0]) / cell.shape[0]

    np.testing.assert_allclose(probs, empirical, atol=5e-3)


# ---- reading a policy out of a network ----------------------------------


def test_strategy_from_network_is_a_valid_distribution():
    game = ContinuousKuhnPoker(min_bet=0.5, max_bet=2.0)
    grid = bet_grid(game, 129)
    networks, params = _networks(game)
    for player in (0, 1):
        strategy_from_network(game, networks[player], params[player], player, grid).validate()


def test_a_freshly_initialized_pair_is_exploitable_but_finite():
    game = ContinuousKuhnPoker(min_bet=0.5, max_bet=2.0)
    networks, params = _networks(game)
    metrics = evaluate_networks(game, networks, params)
    assert float(metrics["exploitability"]) > 0.0
    assert all(jnp.isfinite(v) for v in metrics.values())
    assert float(metrics["exploitability"]) == pytest.approx(
        float(metrics["br_first"] + metrics["br_second"]), rel=1e-6
    )


def test_refining_the_grid_barely_moves_the_answer():
    """The grid is a lower bound; doubling it is how you check it is fine enough."""
    game = ContinuousKuhnPoker(min_bet=0.5, max_bet=2.0)
    networks, params = _networks(game)
    coarse = float(evaluate_networks(game, networks, params, bet_grid(game, 257))["exploitability"])
    fine = float(evaluate_networks(game, networks, params, bet_grid(game, 1025))["exploitability"])
    assert fine >= coarse - 1e-6  # refining can only ever help the responder
    assert fine == pytest.approx(coarse, abs=5e-3)


def test_a_fixed_bet_size_collapses_the_grid_to_one_exact_point():
    game = ContinuousKuhnPoker(min_bet=1.0, max_bet=1.0)
    grid = bet_grid(game, 1025)
    assert grid.shape == (1,)
    assert float(grid[0]) == 1.0
