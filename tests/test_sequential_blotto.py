"""Rules-level checks for `games.sequential_blotto.ContinuousSequentialBlotto`.

Two halves, as in `test_leduc.py`: the `games.sequential` contract a batched,
`jit`ed rollout depends on (fixed shapes, a bounded horizon, absorbing terminal
states, a mask that always admits something), and the Blotto rules themselves --
that the budget really binds, that a front goes to the bigger bid, and above all
that a player is never shown the opponent's commitment, which is the single
assumption the whole game rests on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.sequential import TERMINAL
from games.sequential_blotto import BlottoState, ContinuousSequentialBlotto
from games.spaces import HybridAction

BATCH = 256


def _game(**kwargs) -> ContinuousSequentialBlotto:
    return ContinuousSequentialBlotto(**kwargs)


def _bid(fraction: float) -> HybridAction:
    return HybridAction(
        kind=jnp.zeros((), dtype=jnp.int32),
        value=jnp.asarray([fraction], dtype=jnp.float32),
    )


def _play(game: ContinuousSequentialBlotto, fractions, seed: int = 0) -> BlottoState:
    """Walk a line: `fractions` alternate player 0, player 1, player 0, ...

    `seed` keys the front resolutions, which are chance moves whenever the
    contest is soft or the bids tie.
    """
    state = game.initial_state(jax.random.PRNGKey(seed))
    keys = jax.random.split(jax.random.PRNGKey(seed), max(len(fractions), 1))
    for fraction, key in zip(fractions, keys):
        state = game.step(state, _bid(fraction), key)
    return state


# ---- the sequential-game contract ------------------------------------------


def test_random_play_always_terminates_within_max_steps():
    game = _game(num_fields=4)
    _, payoffs = jax.vmap(
        lambda key: game.play_episode(
            (game.random_action_fn(0), game.random_action_fn(1)), key
        )
    )(jax.random.split(jax.random.PRNGKey(0), BATCH))
    assert np.all(np.isfinite(np.asarray(payoffs)))

    final, _ = game.play_episode(
        (game.random_action_fn(0), game.random_action_fn(1)), jax.random.PRNGKey(0)
    )
    assert int(game.current_player(final)) == TERMINAL


def test_stepping_a_terminal_state_is_a_noop():
    game = _game(num_fields=2)
    terminal = _play(game, [0.5, 0.5, 1.0, 1.0])
    assert int(game.current_player(terminal)) == TERMINAL

    stepped = game.step(terminal, _bid(1.0), jax.random.PRNGKey(7))
    for before, after in zip(jax.tree_util.tree_leaves(terminal), jax.tree_util.tree_leaves(stepped)):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_state_shapes_and_dtypes_survive_a_step():
    game = _game(num_fields=3)
    state = game.initial_state(jax.random.PRNGKey(0))
    stepped = game.step(state, _bid(0.4), jax.random.PRNGKey(1))
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        assert before.shape == after.shape
        assert before.dtype == after.dtype


def test_the_players_alternate_front_by_front():
    game = _game(num_fields=2)
    state = game.initial_state(jax.random.PRNGKey(0))
    seen = []
    for i in range(game.max_steps):
        seen.append(int(game.current_player(state)))
        state = game.step(state, _bid(0.5), jax.random.PRNGKey(i))
    assert seen == [0, 1, 0, 1]
    assert int(game.current_player(state)) == TERMINAL


def test_the_mask_always_admits_the_single_bidding_kind():
    game = _game(num_fields=2)
    state = game.initial_state(jax.random.PRNGKey(0))
    for i in range(game.max_steps + 1):
        for player in (0, 1):
            mask = game.action_mask(player, state)
            assert mask.shape == (game.num_kinds(player),) == (1,)
            assert bool(jnp.all(mask))
        state = game.step(state, _bid(0.5), jax.random.PRNGKey(i))


def test_observation_width_matches_obs_dim_at_every_step():
    game = _game(num_fields=3)
    state = game.initial_state(jax.random.PRNGKey(0))
    for i in range(game.max_steps + 1):
        for player in (0, 1):
            assert game.observation(player, state).shape == (game.obs_dim(player),)
        state = game.step(state, _bid(0.3), jax.random.PRNGKey(i))


# ---- the information structure ---------------------------------------------


def test_observation_hides_the_opponents_bid_on_the_live_front():
    """Player 1 answering a front must not be able to tell what player 0 committed.

    This is the property that keeps the front a simultaneous move; without it
    player 1 best-responds with "that plus epsilon" and the game has no
    equilibrium at all.
    """
    game = _game(num_fields=2)
    small = _play(game, [0.1])
    large = _play(game, [0.9])
    np.testing.assert_allclose(
        np.asarray(game.observation(1, small)), np.asarray(game.observation(1, large))
    )


def test_observation_hides_the_opponents_remaining_budget():
    """A finished front tells you *who* won it, never how dearly it was bought."""
    game = _game(num_fields=3, sharpness=None)
    # Two lines where player 0 wins the first front, spending very differently.
    cheap = _play(game, [0.2, 0.1])
    dear = _play(game, [0.9, 0.1])
    np.testing.assert_array_equal(np.asarray(cheap.results), np.asarray(dear.results))
    np.testing.assert_allclose(
        np.asarray(game.observation(1, cheap)), np.asarray(game.observation(1, dear))
    )
    # ... while player 0 does of course remember their own spending.
    assert not np.allclose(
        np.asarray(game.observation(0, cheap)), np.asarray(game.observation(0, dear))
    )


def test_observation_reveals_own_spends_results_and_budget():
    game = _game(num_fields=2, budget=1.0, sharpness=None)
    state = _play(game, [0.5, 0.25])  # player 0 commits 0.5, player 1 commits 0.25
    obs_0 = np.asarray(game.observation(0, state))
    own_spends, results, remaining, front = obs_0[:2], obs_0[2:4], obs_0[4], obs_0[5:]

    np.testing.assert_allclose(own_spends, [0.5, 0.0])
    np.testing.assert_allclose(results, [1.0, 0.0])  # the bigger bid took front 0
    np.testing.assert_allclose(remaining, 0.5)
    np.testing.assert_allclose(front, [0.0, 1.0, 0.0])  # front 1 is live


def test_results_are_signed_from_each_players_own_point_of_view():
    game = _game(num_fields=2, sharpness=None)
    state = _play(game, [0.5, 0.25])
    obs_0 = np.asarray(game.observation(0, state))
    obs_1 = np.asarray(game.observation(1, state))
    np.testing.assert_allclose(obs_0[2:4], -obs_1[2:4])
    np.testing.assert_allclose(obs_0[2:4], [1.0, 0.0])


def test_own_spends_give_perfect_recall_of_where_the_budget_went():
    """Two lines with the same budget left but different histories differ in the obs."""
    game = _game(num_fields=3, budget=1.0, sharpness=None)
    early = _play(game, [0.5, 0.0, 0.0, 0.0])  # player 0 spent 0.5 on front 0, 0 on front 1
    late = _play(game, [0.0, 0.0, 0.5, 0.0])  # ... and the reverse
    np.testing.assert_allclose(
        float(game.remaining_budget(0, early)), float(game.remaining_budget(0, late))
    )
    assert not np.allclose(
        np.asarray(game.observation(0, early)), np.asarray(game.observation(0, late))
    )


# ---- the Blotto rules -------------------------------------------------------


def test_a_bid_is_a_fraction_of_what_is_left():
    game = _game(num_fields=3, budget=1.0)
    # Player 0 commits half of 1.0, then half of the remaining 0.5.
    state = _play(game, [0.5, 0.0, 0.5, 0.0])
    np.testing.assert_allclose(np.asarray(state.spends[0]), [0.5, 0.25, 0.0], atol=1e-6)
    np.testing.assert_allclose(float(game.remaining_budget(0, state)), 0.25, atol=1e-6)


def test_the_budget_can_never_be_overspent():
    game = _game(num_fields=4, budget=2.0)
    state = _play(game, [1.0] * 8)  # everyone shoves everything at every front
    spent = np.asarray(state.spends).sum(axis=1)
    assert np.all(spent <= 2.0 + 1e-6)
    # An all-in first front leaves nothing for the rest.
    np.testing.assert_allclose(np.asarray(state.spends[0]), [2.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_the_bid_fraction_is_clipped_to_the_action_box():
    game = _game(num_fields=1, budget=1.0)
    over = _play(game, [3.0])
    under = _play(game, [-2.0])
    np.testing.assert_allclose(float(over.spends[0, 0]), 1.0, atol=1e-6)
    np.testing.assert_allclose(float(under.spends[0, 0]), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    "fractions, expected",
    [
        ([0.7, 0.2], 1.0),  # player 0 outbids
        ([0.2, 0.7], -1.0),  # player 1 outbids
    ],
)
def test_a_hard_front_goes_to_the_bigger_bid(fractions, expected):
    game = _game(num_fields=1, sharpness=None)
    state = _play(game, fractions)
    np.testing.assert_allclose(float(game.payoff(state)), expected)


def test_a_hard_tie_is_a_coin_flip():
    game = _game(num_fields=1, sharpness=None)
    payoffs = np.asarray(
        [
            float(game.payoff(_play(game, [0.4, 0.4], seed=seed)))
            for seed in range(BATCH)
        ]
    )
    assert set(np.unique(payoffs)) == {-1.0, 1.0}
    assert abs(payoffs.mean()) < 0.2


def test_a_soft_front_is_a_logistic_contest_of_the_bid_gap():
    game = _game(num_fields=1, sharpness=8.0)
    # Player 0 bids 0.75, player 1 bids 0.25: a gap of 0.5.
    wins = np.asarray(
        [float(game.payoff(_play(game, [0.75, 0.25], seed=seed))) for seed in range(1024)]
    )
    expected = 2.0 * float(jax.nn.sigmoid(8.0 * 0.5)) - 1.0
    assert abs(wins.mean() - expected) < 0.05


def test_losing_a_front_still_costs_the_bid():
    """All-pay: the loser's commitment is gone too, which is what makes budget scarce."""
    game = _game(num_fields=2, sharpness=None)
    state = _play(game, [0.5, 1.0])  # player 1 takes front 0 by shoving everything
    np.testing.assert_allclose(float(game.payoff(state)), -1.0)
    np.testing.assert_allclose(float(game.remaining_budget(0, state)), 0.5, atol=1e-6)
    np.testing.assert_allclose(float(game.remaining_budget(1, state)), 0.0, atol=1e-6)


def test_field_values_weight_the_fronts():
    game = _game(num_fields=2, field_values=(1.0, 4.0), sharpness=None)
    # Player 0 takes the cheap front, player 1 the valuable one.
    state = _play(game, [0.9, 0.1, 1.0, 1.0])
    np.testing.assert_allclose(float(game.payoff(state)), 1.0 - 4.0)


def test_payoff_is_zero_sum_under_swapping_the_two_lines():
    game = _game(num_fields=3, field_values=(1.0, 2.0, 3.0), sharpness=None)
    line = [0.6, 0.3, 0.5, 0.8, 1.0, 1.0]
    swapped = [line[i ^ 1] for i in range(len(line))]
    np.testing.assert_allclose(
        float(game.payoff(_play(game, line))), -float(game.payoff(_play(game, swapped)))
    )


def test_uniform_random_play_is_a_fair_game():
    """Nothing in the rules favours either colonel, so random play is worth ~0."""
    game = _game(num_fields=3)
    _, payoffs = jax.vmap(
        lambda key: game.play_episode(
            (game.random_action_fn(0), game.random_action_fn(1)), key
        )
    )(jax.random.split(jax.random.PRNGKey(0), 4096))
    assert abs(float(jnp.mean(payoffs))) < 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_fields=0),
        dict(budget=0.0),
        dict(sharpness=-1.0),
        dict(num_fields=2, field_values=(1.0, 2.0, 3.0)),
        dict(num_fields=2, field_values=(1.0, 0.0)),
    ],
)
def test_the_constructor_rejects_a_malformed_game(kwargs):
    with pytest.raises(ValueError):
        _game(**kwargs)
