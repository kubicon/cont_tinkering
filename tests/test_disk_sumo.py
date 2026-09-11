"""Rules-level checks for `games.disk_sumo.DiskSumo`.

Two halves, as in `test_sequential_blotto.py`: the `games.sequential` contract a
batched, `jit`ed rollout depends on (fixed shapes, a bounded horizon, absorbing
terminal states), and the sumo rules themselves -- that pushing an idle opponent
wins, that the two seats are mirror images, that contact conserves momentum, and
above all that player 1 never sees player 0's force for the live control step,
which is what makes the two decisions one simultaneous move.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.disk_sumo import DiskSumo, DiskSumoState
from games.sequential import TERMINAL
from games.spaces import HybridAction

BATCH = 256


def _game(**kwargs) -> DiskSumo:
    return DiskSumo(**kwargs)


def _force(x: float, y: float) -> HybridAction:
    return HybridAction(
        kind=jnp.zeros((), dtype=jnp.int32), value=jnp.asarray([x, y], dtype=jnp.float32)
    )


def _constant(x: float, y: float):
    action = _force(x, y)
    return lambda obs, mask, key: action


def _deterministic(**kwargs) -> DiskSumo:
    """Fixed facing axis and no jitter: the start is exactly symmetric."""
    return _game(random_orientation=False, start_jitter=0.0, **kwargs)


# ---- the sequential-game contract ------------------------------------------


def test_random_play_terminates_within_max_steps_with_bounded_payoffs():
    game = _game(horizon=20, margin_weight=0.5)
    finals, payoffs = jax.vmap(
        lambda key: game.play_episode(
            (game.random_action_fn(0), game.random_action_fn(1)), key
        )
    )(jax.random.split(jax.random.PRNGKey(0), BATCH))
    payoffs = np.asarray(payoffs)
    assert np.all(np.isfinite(payoffs))
    assert np.all(np.abs(payoffs) <= 1.0)
    players = jax.vmap(game.current_player)(finals)
    assert np.all(np.asarray(players) == TERMINAL)


def test_stepping_a_terminal_state_is_a_noop():
    game = _deterministic(horizon=2)
    state = game.initial_state(jax.random.PRNGKey(0))
    for _ in range(game.max_steps):
        state = game.step(state, _force(1.0, 0.0), jax.random.PRNGKey(1))
    assert int(game.current_player(state)) == TERMINAL

    stepped = game.step(state, _force(-1.0, 1.0), jax.random.PRNGKey(7))
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_state_shapes_and_dtypes_survive_a_step():
    game = _game()
    state = game.initial_state(jax.random.PRNGKey(0))
    stepped = game.step(state, _force(0.3, -0.2), jax.random.PRNGKey(1))
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        assert before.shape == after.shape and before.dtype == after.dtype
    for player in (0, 1):
        assert game.observation(player, state).shape == (game.obs_dim(player),)
        assert game.action_mask(player, state).shape == (game.num_kinds(player),)


def test_players_alternate_and_the_physics_waits_for_player_one():
    game = _game()
    state = game.initial_state(jax.random.PRNGKey(0))
    assert int(game.current_player(state)) == 0
    after_0 = game.step(state, _force(1.0, 0.0), jax.random.PRNGKey(1))
    assert int(game.current_player(after_0)) == 1
    np.testing.assert_array_equal(np.asarray(after_0.pos), np.asarray(state.pos))
    after_1 = game.step(after_0, _force(1.0, 0.0), jax.random.PRNGKey(2))
    assert int(game.current_player(after_1)) == 0
    assert not np.allclose(np.asarray(after_1.pos), np.asarray(state.pos))


# ---- information ------------------------------------------------------------


def test_player_one_never_sees_player_zeros_live_force():
    game = _game()
    state = game.initial_state(jax.random.PRNGKey(3))
    parked = [game.step(state, _force(x, y), jax.random.PRNGKey(4))
              for x, y in ((1.0, 0.0), (-1.0, 0.5), (0.0, -1.0))]
    assert not np.allclose(np.asarray(parked[0].pending), np.asarray(parked[1].pending))
    observations = [np.asarray(game.observation(1, s)) for s in parked]
    for obs in observations[1:]:
        np.testing.assert_array_equal(obs, observations[0])


def test_a_symmetric_start_looks_identical_from_both_seats():
    """The egocentric frames are mirror images, whatever the facing axis."""
    game = _game(start_jitter=0.0)
    for seed in range(5):
        state = game.initial_state(jax.random.PRNGKey(seed))
        np.testing.assert_allclose(
            np.asarray(game.observation(0, state)), np.asarray(game.observation(1, state)), atol=1e-6
        )


# ---- rules ------------------------------------------------------------------


def test_charging_an_idle_opponent_pushes_them_out():
    game = _deterministic()
    final, payoff = game.play_episode((_constant(1.0, 0.0), _constant(0.0, 0.0)), jax.random.PRNGKey(0))
    assert bool(final.done) and float(payoff) == 1.0
    final, payoff = game.play_episode((_constant(0.0, 0.0), _constant(1.0, 0.0)), jax.random.PRNGKey(0))
    assert bool(final.done) and float(payoff) == -1.0


def test_equal_head_on_charges_stalemate_at_zero():
    game = _deterministic(margin_weight=0.5)
    final, payoff = game.play_episode((_constant(1.0, 0.0), _constant(1.0, 0.0)), jax.random.PRNGKey(0))
    assert not bool(final.done)
    assert abs(float(payoff)) < 1e-5


def test_the_timeout_margin_rewards_the_disk_nearer_the_centre():
    game = _deterministic(margin_weight=0.5)
    state = game.initial_state(jax.random.PRNGKey(0)).replace(
        pos=jnp.asarray([[0.1, 0.0], [0.7, 0.0]], dtype=jnp.float32),
        turn=jnp.asarray(game.max_steps, dtype=jnp.int32),
    )
    assert int(game.current_player(state)) == TERMINAL
    assert float(game.payoff(state)) == pytest.approx(0.5 * 0.6)
    assert float(_deterministic(margin_weight=0.0).payoff(state)) == 0.0


def test_diagonal_forces_are_no_stronger_than_straight_ones():
    game = _deterministic(drag=0.0)
    state = game.initial_state(jax.random.PRNGKey(0))
    straight = game.step(game.step(state, _force(0.0, 1.0), jax.random.PRNGKey(1)),
                         _force(0.0, 0.0), jax.random.PRNGKey(2))
    diagonal = game.step(game.step(state, _force(1.0, 1.0), jax.random.PRNGKey(1)),
                         _force(0.0, 0.0), jax.random.PRNGKey(2))
    assert np.linalg.norm(np.asarray(diagonal.vel[0])) == pytest.approx(
        np.linalg.norm(np.asarray(straight.vel[0])), rel=1e-5
    )


def test_contact_conserves_momentum_without_drag_or_forces():
    game = _deterministic(drag=0.0, contact_damping=0.0)
    state = DiskSumoState(
        pos=jnp.asarray([[-0.2, 0.0], [0.2, 0.05]], dtype=jnp.float32),
        vel=jnp.asarray([[1.0, 0.0], [-0.5, 0.0]], dtype=jnp.float32),
        pending=jnp.zeros((2,), dtype=jnp.float32),
        result=jnp.zeros((), dtype=jnp.float32),
        done=jnp.zeros((), dtype=bool),
        turn=jnp.asarray(1, dtype=jnp.int32),  # player 1 to act: the next step resolves
    )
    stepped = game.step(state, _force(0.0, 0.0), jax.random.PRNGKey(0))
    # They collided (the velocities changed) ...
    assert not np.allclose(np.asarray(stepped.vel), np.asarray(state.vel))
    # ... and total momentum did not.
    np.testing.assert_allclose(
        np.asarray(stepped.vel).sum(axis=0), np.asarray(state.vel).sum(axis=0), atol=1e-5
    )


def test_the_disk_that_leaves_first_loses_a_simultaneous_exit():
    game = _deterministic(drag=0.0)
    state = DiskSumoState(
        pos=jnp.asarray([[-0.95, 0.0], [0.92, 0.0]], dtype=jnp.float32),
        vel=jnp.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        pending=jnp.zeros((2,), dtype=jnp.float32),
        result=jnp.zeros((), dtype=jnp.float32),
        done=jnp.zeros((), dtype=bool),
        turn=jnp.asarray(1, dtype=jnp.int32),
    )
    stepped = game.step(state, _force(0.0, 0.0), jax.random.PRNGKey(0))
    assert bool(stepped.done)
    assert np.all(np.linalg.norm(np.asarray(stepped.pos), axis=-1) > game.ring_radius)
    assert float(game.payoff(stepped)) == -1.0  # player 0 was nearer the edge


def test_random_play_is_fair_on_average():
    game = _game(horizon=30, margin_weight=0.5)
    _, payoffs = jax.vmap(
        lambda key: game.play_episode(
            (game.random_action_fn(0), game.random_action_fn(1)), key
        )
    )(jax.random.split(jax.random.PRNGKey(1), 2000))
    assert abs(float(np.mean(np.asarray(payoffs)))) < 0.05


@pytest.mark.parametrize("kwargs", [
    dict(horizon=0), dict(substeps=0), dict(ring_radius=0.0), dict(start_distance=0.1),
    dict(start_distance=1.9), dict(margin_weight=1.5), dict(drag=-1.0),
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _game(**kwargs)
