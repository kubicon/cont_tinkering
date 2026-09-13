"""Rules-level checks for `games.mjx_sumo.MjxSumo`.

Deliberately the same suite as `test_disk_sumo.py`, because `MjxSumo` is meant
to be the same game on a real solver: the `games.sequential` contract a batched,
`jit`ed rollout depends on (fixed shapes, a bounded horizon, absorbing terminal
states), then the sumo rules themselves -- pushing an idle opponent wins, the
two seats are mirror images, contact conserves momentum, and above all player 1
never sees player 0's force for the live control step.

On top of that there are the checks that only make sense here: that the state
carrying an `mjx.Data` still survives the terminal-state guard leafwise, that it
`vmap`s, and that MuJoCo's physics reproduces the drag law the observation's
velocity scale assumes.

MJX on CPU is slow, so the horizons and batches here are the smallest ones that
still make each assertion meaningful.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.mjx_sumo import MjxSumo, MjxSumoState
from games.sequential import TERMINAL
from games.spaces import HybridAction

BATCH = 32


def _game(**kwargs) -> MjxSumo:
    kwargs.setdefault("horizon", 20)
    kwargs.setdefault("substeps", 4)
    return MjxSumo(**kwargs)


def _deterministic(**kwargs) -> MjxSumo:
    """Fixed facing axis and no jitter: the start is exactly symmetric."""
    return _game(random_orientation=False, start_jitter=0.0, **kwargs)


def _force(x: float, y: float) -> HybridAction:
    return HybridAction(
        kind=jnp.zeros((), dtype=jnp.int32), value=jnp.asarray([x, y], dtype=jnp.float32)
    )


def _constant(x: float, y: float):
    action = _force(x, y)
    return lambda obs, mask, key: action


def _state(game: MjxSumo, pos, vel, turn: int) -> MjxSumoState:
    """A hand-built state: `pos`/`vel` are `(2, 2)`, laid into `qpos`/`qvel`."""
    state = game.initial_state(jax.random.PRNGKey(0))
    return state.replace(
        data=state.data.replace(
            qpos=jnp.asarray(pos, dtype=jnp.float32).reshape(-1),
            qvel=jnp.asarray(vel, dtype=jnp.float32).reshape(-1),
        ),
        turn=jnp.asarray(turn, dtype=jnp.int32),
    )


# ---- the sequential-game contract ------------------------------------------


def test_random_play_terminates_within_max_steps_with_bounded_payoffs():
    game = _game(margin_weight=0.5)
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
    """The guard in `games.sequential.step` has to hold across every `mjx.Data` leaf."""
    game = _deterministic(horizon=2)
    state = game.initial_state(jax.random.PRNGKey(0))
    for _ in range(game.max_steps):
        state = game.step(state, _force(1.0, 0.0), jax.random.PRNGKey(1))
    assert int(game.current_player(state)) == TERMINAL

    stepped = game.step(state, _force(-1.0, 1.0), jax.random.PRNGKey(7))
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) > 10  # the mjx.Data arrays, not just the bookkeeping fields
    for before, after in zip(leaves, jax.tree_util.tree_leaves(stepped)):
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
    np.testing.assert_array_equal(
        np.asarray(after_0.data.qpos), np.asarray(state.data.qpos)
    )
    after_1 = game.step(after_0, _force(1.0, 0.0), jax.random.PRNGKey(2))
    assert int(game.current_player(after_1)) == 0
    assert not np.allclose(np.asarray(after_1.data.qpos), np.asarray(state.data.qpos))


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
    game = _deterministic(horizon=50)
    final, payoff = game.play_episode((_constant(1.0, 0.0), _constant(0.0, 0.0)), jax.random.PRNGKey(0))
    assert bool(final.done) and float(payoff) == 1.0
    final, payoff = game.play_episode((_constant(0.0, 0.0), _constant(1.0, 0.0)), jax.random.PRNGKey(0))
    assert bool(final.done) and float(payoff) == -1.0


def test_equal_head_on_charges_stalemate_at_zero():
    game = _deterministic(horizon=50, margin_weight=0.5)
    final, payoff = game.play_episode((_constant(1.0, 0.0), _constant(1.0, 0.0)), jax.random.PRNGKey(0))
    assert not bool(final.done)
    assert abs(float(payoff)) < 1e-5


def test_the_timeout_margin_rewards_the_disk_nearer_the_centre():
    game = _deterministic(margin_weight=0.5)
    state = _state(game, [[0.1, 0.0], [0.7, 0.0]], jnp.zeros((2, 2)), turn=game.max_steps)
    assert int(game.current_player(state)) == TERMINAL
    assert float(game.payoff(state)) == pytest.approx(0.5 * 0.6, abs=1e-6)
    assert float(_deterministic(margin_weight=0.0).payoff(state)) == 0.0


def test_diagonal_forces_are_no_stronger_than_straight_ones():
    """The unit-disk projection, not the `[-1, 1]^2` box, is what bounds a push."""
    game = _deterministic(drag=0.0)
    state = game.initial_state(jax.random.PRNGKey(0))
    straight = game.step(game.step(state, _force(0.0, 1.0), jax.random.PRNGKey(1)),
                         _force(0.0, 0.0), jax.random.PRNGKey(2))
    diagonal = game.step(game.step(state, _force(1.0, 1.0), jax.random.PRNGKey(1)),
                         _force(0.0, 0.0), jax.random.PRNGKey(2))
    speed = lambda s: np.linalg.norm(np.asarray(game._velocities(s)[0]))
    assert speed(diagonal) == pytest.approx(speed(straight), rel=1e-5)


def test_contact_conserves_momentum_without_drag_or_forces():
    game = _deterministic(drag=0.0)
    state = _state(
        game, [[-0.2, 0.0], [0.2, 0.05]], [[1.0, 0.0], [-0.5, 0.0]],
        turn=1,  # player 1 to act: the next step resolves
    )
    stepped = game.step(state, _force(0.0, 0.0), jax.random.PRNGKey(0))
    # They collided (the velocities changed) ...
    assert not np.allclose(np.asarray(stepped.data.qvel), np.asarray(state.data.qvel))
    # ... and total momentum did not.
    np.testing.assert_allclose(
        np.asarray(game._velocities(stepped)).sum(axis=0),
        np.asarray(game._velocities(state)).sum(axis=0),
        atol=1e-5,
    )


def test_the_disk_that_leaves_first_loses_a_simultaneous_exit():
    game = _deterministic(drag=0.0)
    state = _state(game, [[-0.95, 0.0], [0.92, 0.0]], [[-1.0, 0.0], [1.0, 0.0]], turn=1)
    stepped = game.step(state, _force(0.0, 0.0), jax.random.PRNGKey(0))
    assert bool(stepped.done)
    assert np.all(np.linalg.norm(np.asarray(game._positions(stepped)), axis=-1) > game.ring_radius)
    assert float(game.payoff(stepped)) == -1.0  # player 0 was nearer the edge


def test_random_play_is_fair_on_average():
    game = _game(margin_weight=0.5)
    _, payoffs = jax.vmap(
        lambda key: game.play_episode(
            (game.random_action_fn(0), game.random_action_fn(1)), key
        )
    )(jax.random.split(jax.random.PRNGKey(1), 128))
    assert abs(float(np.mean(np.asarray(payoffs)))) < 0.1


# ---- the physics the observation assumes ------------------------------------


def test_a_lone_puck_settles_at_the_speed_the_observation_normalizes_by():
    """`_speed_scale` is `max_force / drag`; MuJoCo's joint damping has to agree.

    The observation divides velocities by it, so if the two disagree the network
    sees an input whose scale is silently wrong -- which no rules test catches.

    Player 0 pushes *away* from the opponent (egocentric x points at them, so
    `(-1, 0)` retreats) in an oversized ring: no contact to share the drag with,
    and no exit before the clock runs out, which is what makes this the one-body
    drag law rather than a two-body one.
    """
    game = _deterministic(horizon=40, substeps=10, ring_radius=5.0, max_force=1.0, drag=2.0)
    final, _ = game.play_episode((_constant(-1.0, 0.0), _constant(0.0, 0.0)), jax.random.PRNGKey(0))
    assert not bool(final.done)  # nobody left the ring, so the puck ran free throughout
    positions = np.asarray(game._positions(final))
    assert np.linalg.norm(positions[0] - positions[1]) > 2 * game.disk_radius  # never touched
    assert game._speed_scale == pytest.approx(0.5)
    speed = np.linalg.norm(np.asarray(game._velocities(final)[0]))
    assert speed == pytest.approx(game._speed_scale, rel=0.02)


@pytest.mark.parametrize("kwargs", [
    dict(horizon=0), dict(substeps=0), dict(ring_radius=0.0), dict(start_distance=0.1),
    dict(start_distance=1.9), dict(margin_weight=1.5), dict(drag=-1.0),
    dict(solref_timeconst=0.0), dict(solver_iterations=0),
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MjxSumo(**kwargs)
