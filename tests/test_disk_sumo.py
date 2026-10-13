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


# ---- observation delay ------------------------------------------------------


def _control_step(game: DiskSumo, state: DiskSumoState, force_0, force_1) -> DiskSumoState:
    key = jax.random.PRNGKey(0)
    return game.step(game.step(state, _force(*force_0), key), _force(*force_1), key)


def _trajectory(game: DiskSumo, moves) -> list[DiskSumoState]:
    """The states at the start of every control step of a bout playing `moves`."""
    states = [game.initial_state(jax.random.PRNGKey(0))]
    for force_0, force_1 in moves:
        states.append(_control_step(game, states[-1], force_0, force_1))
    return states


# Distinct every step and for each seat, so no two control steps look alike.
_MOVES = [((0.9, 0.3), (0.2, -0.8)), ((0.1, -0.6), (1.0, 0.0)), ((-0.4, 0.7), (0.5, 0.5)),
          ((1.0, 0.0), (-0.3, 0.2)), ((0.6, -0.6), (0.0, 1.0)), ((0.0, 0.4), (0.8, -0.1))]


@pytest.mark.parametrize("delay", [1, 3])
def test_a_delayed_opponent_is_seen_as_it_was(delay):
    game = _deterministic(egocentric=False, observation_delay=delay)
    states = _trajectory(game, _MOVES)
    for t, state in enumerate(states):
        seen = states[max(t - delay, 0)]
        for own, opp in ((0, 1), (1, 0)):
            obs = np.asarray(game.observation(own, state))
            np.testing.assert_allclose(obs[0:2], np.asarray(state.pos[own]), atol=1e-6)
            np.testing.assert_allclose(obs[2:4], np.asarray(state.vel[own]) / game._speed_scale, atol=1e-6)
            np.testing.assert_allclose(obs[4:6], np.asarray(seen.pos[opp]), atol=1e-6)
            np.testing.assert_allclose(obs[6:8], np.asarray(seen.vel[opp]) / game._speed_scale, atol=1e-6)
            assert obs[8] == pytest.approx(1.0 - np.linalg.norm(np.asarray(state.pos[own])), abs=1e-6)
            assert obs[9] == pytest.approx(1.0 - np.linalg.norm(np.asarray(seen.pos[opp])), abs=1e-6)
            assert obs[10] == pytest.approx(1.0 - t / game.horizon)


def test_the_delay_changes_what_is_seen_but_not_the_physics():
    live = _trajectory(_deterministic(egocentric=False), _MOVES)
    delayed = _trajectory(_deterministic(egocentric=False, observation_delay=3), _MOVES)
    for a, b in zip(live, delayed):
        np.testing.assert_array_equal(np.asarray(a.pos), np.asarray(b.pos))
        np.testing.assert_array_equal(np.asarray(a.vel), np.asarray(b.vel))
    # The start is all there is to see, so the first observation is the live one...
    live_game, delayed_game = _deterministic(), _deterministic(observation_delay=3)
    np.testing.assert_array_equal(
        np.asarray(live_game.observation(0, live_game.initial_state(jax.random.PRNGKey(0)))),
        np.asarray(delayed_game.observation(0, delayed_game.initial_state(jax.random.PRNGKey(0)))),
    )
    # ...and a later one is not.
    assert not np.allclose(
        np.asarray(_deterministic(egocentric=False).observation(0, live[-1])),
        np.asarray(_deterministic(egocentric=False, observation_delay=3).observation(0, delayed[-1])),
    )


def test_the_egocentric_frame_points_at_the_delayed_opponent():
    """`(1, 0)` charges where the opponent was seen, in observation and action alike."""
    delay = 2
    game = _deterministic(drag=0.5, observation_delay=delay)
    # Both sidestep hard, so the line of centres turns between sighting and now.
    states = _trajectory(game, [((0.0, 1.0), (0.0, 1.0))] * 6)
    state, seen = states[-1], states[-1 - delay]
    for own, opp in ((0, 1), (1, 0)):
        towards = np.asarray(seen.pos[opp] - state.pos[own])
        towards = towards / np.linalg.norm(towards)
        truly = np.asarray(state.pos[opp] - state.pos[own])
        assert not np.allclose(towards, truly / np.linalg.norm(truly), atol=1e-3)

        obs = np.asarray(game.observation(own, state))
        # The seen opponent sits dead ahead: relative position along +x only.
        np.testing.assert_allclose((obs[4:6] - obs[0:2])[1], 0.0, atol=1e-6)
        assert (obs[4:6] - obs[0:2])[0] > 0.0

        force, effort = game._world_force(state, own, jnp.asarray([1.0, 0.0]))
        np.testing.assert_allclose(np.asarray(force), game.max_force * towards, atol=1e-6)
        assert float(effort) == pytest.approx(1.0)


def test_the_opponents_stamina_is_delayed_and_ones_own_is_not():
    delay = 2
    game = _deterministic(stamina_enabled=True, observation_delay=delay)
    # Player 0 charges throughout and player 1 rests, so the two levels part.
    states = _trajectory(game, [((1.0, 0.0), (1.0, 0.0))] * 2 + [((1.0, 0.0), (0.0, 0.0))] * 3)
    state, seen = states[-1], states[-1 - delay]
    assert float(state.stamina[0]) != pytest.approx(float(seen.stamina[0]))
    assert float(state.stamina[1]) != pytest.approx(float(seen.stamina[1]))
    np.testing.assert_allclose(
        np.asarray(game.observation(0, state)[-2:]), [state.stamina[0], seen.stamina[1]], atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(game.observation(1, state)[-2:]), [state.stamina[1], seen.stamina[0]], atol=1e-6
    )


@pytest.mark.parametrize("delay, frames", [(3, 1), (2, 3), (0, 3)])
def test_older_sightings_follow_the_observation_newest_first(delay, frames):
    game = _deterministic(egocentric=False, observation_delay=delay, observation_frames=frames)
    plain = _deterministic(egocentric=False, observation_delay=delay)
    assert game.obs_dim(0) == game.obs_dim(1) == 11 + 4 * (frames - 1)
    states, plain_states = _trajectory(game, _MOVES), _trajectory(plain, _MOVES)
    for t, state in enumerate(states):
        for own, opp in ((0, 1), (1, 0)):
            obs = np.asarray(game.observation(own, state))
            # The memory is appended: what one sighting showed is where it was.
            np.testing.assert_array_equal(obs[:11], np.asarray(plain.observation(own, plain_states[t])))
            for frame in range(1, frames):
                seen = states[max(t - delay - frame, 0)]
                sighting = obs[11 + 4 * (frame - 1): 11 + 4 * frame]
                np.testing.assert_allclose(sighting[0:2], np.asarray(seen.pos[opp]), atol=1e-6)
                np.testing.assert_allclose(
                    sighting[2:4], np.asarray(seen.vel[opp]) / game._speed_scale, atol=1e-6
                )


def test_older_sightings_are_drawn_in_the_current_frame():
    delay, frames = 1, 3
    game = _deterministic(drag=0.5, observation_delay=delay, observation_frames=frames)
    states = _trajectory(game, [((0.0, 1.0), (0.0, 1.0))] * 6)
    state = states[-1]
    for own, opp in ((0, 1), (1, 0)):
        obs = np.asarray(game.observation(own, state))
        rotation = np.asarray(game._frame(state.pos[own], states[-1 - delay].pos[opp]))
        for frame in range(1, frames):
            seen = states[-1 - delay - frame]
            sighting = obs[11 + 4 * (frame - 1): 11 + 4 * frame]
            np.testing.assert_allclose(sighting[0:2], rotation @ np.asarray(seen.pos[opp]), atol=1e-6)
            np.testing.assert_allclose(
                sighting[2:4], rotation @ np.asarray(seen.vel[opp]) / game._speed_scale, atol=1e-6
            )
        # The frame turned between sightings, so an old one is no longer dead ahead.
        assert abs((obs[11:13] - obs[0:2])[1]) > 1e-3


def test_older_sightings_carry_the_opponents_stamina():
    delay, frames = 1, 3
    game = _deterministic(stamina_enabled=True, observation_delay=delay, observation_frames=frames)
    assert game.obs_dim(0) == 13 + 5 * (frames - 1)
    states = _trajectory(game, [((1.0, 0.0), (1.0, 0.0))] * 2 + [((1.0, 0.0), (0.0, 0.0))] * 3)
    state = states[-1]
    obs = np.asarray(game.observation(0, state))
    np.testing.assert_allclose(obs[11:13], [state.stamina[0], states[-1 - delay].stamina[1]], atol=1e-6)
    older = [float(obs[13 + 5 * (frame - 1) + 4]) for frame in range(1, frames)]
    np.testing.assert_allclose(
        older, [float(states[-1 - delay - frame].stamina[1]) for frame in range(1, frames)], atol=1e-6
    )
    assert older[0] != pytest.approx(older[1])


def test_a_delayed_bout_keeps_its_shapes_and_plays_to_the_end():
    game = _game(horizon=20, observation_delay=3, observation_frames=4)
    state = game.initial_state(jax.random.PRNGKey(0))
    # Three steps of delay and three older sightings behind the newest.
    assert state.past_pos.shape == (6, 2, 2) and state.past_stamina.shape == (6, 2)
    assert _game(observation_delay=3).initial_state(jax.random.PRNGKey(0)).past_pos.shape == (3, 2, 2)
    assert _game().initial_state(jax.random.PRNGKey(0)).past_pos.shape == (0, 2, 2)
    stepped = _control_step(game, state, (0.3, -0.2), (1.0, 0.0))
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        assert before.shape == after.shape and before.dtype == after.dtype
    assert game.observation(0, stepped).shape == (game.obs_dim(0),) == (23,)

    final, payoffs = jax.vmap(
        lambda key: game.play_episode(
            (game.random_action_fn(0), game.random_action_fn(1)), key
        )
    )(jax.random.split(jax.random.PRNGKey(1), BATCH))
    assert np.all(np.isfinite(np.asarray(payoffs)))
    assert np.all(np.abs(np.asarray(payoffs)) <= 1.0)


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
    state = game.initial_state(jax.random.PRNGKey(0)).replace(
        pos=jnp.asarray([[-0.2, 0.0], [0.2, 0.05]], dtype=jnp.float32),
        vel=jnp.asarray([[1.0, 0.0], [-0.5, 0.0]], dtype=jnp.float32),
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
    state = game.initial_state(jax.random.PRNGKey(0)).replace(
        pos=jnp.asarray([[-0.95, 0.0], [0.92, 0.0]], dtype=jnp.float32),
        vel=jnp.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        turn=jnp.asarray(1, dtype=jnp.int32),
    )
    stepped = game.step(state, _force(0.0, 0.0), jax.random.PRNGKey(0))
    assert bool(stepped.done)
    assert np.all(np.linalg.norm(np.asarray(stepped.pos), axis=-1) > game.ring_radius)
    assert float(game.payoff(stepped)) == -1.0  # player 0 was nearer the edge


def test_random_start_places_both_disks_legally():
    """Inside the ring, whole disk in, and never already in contact."""
    game = _game(random_start=True)
    pos = np.asarray(jax.vmap(game.initial_state)(jax.random.split(jax.random.PRNGKey(0), BATCH)).pos)
    radii = np.linalg.norm(pos, axis=-1)
    gaps = np.linalg.norm(pos[:, 1] - pos[:, 0], axis=-1)
    assert np.all(radii <= game.ring_radius - game.disk_radius + 1e-5)
    assert np.all(gaps >= 2.0 * game.disk_radius)


def test_random_start_reaches_both_the_middle_and_the_rim():
    """The point of it: a bout can open anywhere, not just where the facing start is."""
    game = _game(random_start=True)
    pos = np.asarray(jax.vmap(game.initial_state)(jax.random.split(jax.random.PRNGKey(1), BATCH)).pos)
    radii = np.linalg.norm(pos, axis=-1)
    assert radii.min() < 0.2 and radii.max() > 0.8
    # Uniform on the area of a disk of radius r has mean radius 2r/3.
    assert abs(radii.mean() - 2.0 / 3.0 * (game.ring_radius - game.disk_radius)) < 0.05


def test_random_start_is_off_by_default():
    """The facing start is untouched: mirrored across the centre, `start_distance` apart."""
    game = _deterministic()
    pos = np.asarray(game.initial_state(jax.random.PRNGKey(0)).pos)
    np.testing.assert_allclose(pos[0], -pos[1], atol=1e-6)
    assert float(np.linalg.norm(pos[1] - pos[0])) == pytest.approx(game.start_distance, abs=1e-5)


def test_random_start_stays_fair_on_average():
    """Both seats are drawn from the same distribution, so neither gains from it."""
    game = _game(horizon=30, margin_weight=0.5, random_start=True)
    _, payoffs = jax.vmap(
        lambda key: game.play_episode(
            (game.random_action_fn(0), game.random_action_fn(1)), key
        )
    )(jax.random.split(jax.random.PRNGKey(2), 2000))
    assert abs(float(np.mean(np.asarray(payoffs)))) < 0.05


def test_force_asymmetry_randomizes_an_observable_strong_role():
    game = _game(force_asymmetry=0.2)
    states = jax.vmap(game.initial_state)(jax.random.split(jax.random.PRNGKey(9), BATCH))
    multipliers = np.asarray(states.force_multiplier)

    np.testing.assert_allclose(
        np.sort(multipliers, axis=-1),
        np.broadcast_to([0.8, 1.2], multipliers.shape),
        atol=1e-6,
    )
    assert np.any(multipliers[:, 0] > multipliers[:, 1])
    assert np.any(multipliers[:, 1] > multipliers[:, 0])
    assert game.obs_dim(0) == game.obs_dim(1) == 13

    state = jax.tree_util.tree_map(lambda x: x[0], states)
    for player in (0, 1):
        obs = np.asarray(game.observation(player, state))
        assert obs[-2] == pytest.approx(multipliers[0, player])
        assert obs[-1] == pytest.approx(multipliers[0, 1 - player])


def test_force_asymmetry_scales_the_applied_force():
    game = _deterministic(force_asymmetry=0.2)
    state = game.initial_state(jax.random.PRNGKey(0)).replace(
        force_multiplier=jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    )
    after_0 = game.step(state, _force(1.0, 0.0), jax.random.PRNGKey(1))
    assert np.linalg.norm(np.asarray(after_0.pending)) == pytest.approx(1.2)


def test_stamina_is_observable_but_pending_effort_is_hidden():
    game = _deterministic(stamina_enabled=True)
    state = game.initial_state(jax.random.PRNGKey(0)).replace(
        stamina=jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    )
    assert game.obs_dim(0) == game.obs_dim(1) == 13
    np.testing.assert_allclose(np.asarray(game.observation(0, state)[-2:]), [0.25, 0.75])
    np.testing.assert_allclose(np.asarray(game.observation(1, state)[-2:]), [0.75, 0.25])

    parked = [game.step(state, _force(x, y), jax.random.PRNGKey(1))
              for x, y in ((1.0, 0.0), (0.0, 0.0))]
    assert float(parked[0].pending_effort) != float(parked[1].pending_effort)
    np.testing.assert_array_equal(
        np.asarray(game.observation(1, parked[0])), np.asarray(game.observation(1, parked[1]))
    )


def test_full_thrust_drains_stamina_and_rest_recovers_it():
    game = _deterministic(
        stamina_enabled=True, stamina_drain_rate=0.5, stamina_recovery_rate=0.25
    )
    state = game.initial_state(jax.random.PRNGKey(0))
    charged = game.step(
        game.step(state, _force(1.0, 0.0), jax.random.PRNGKey(1)),
        _force(1.0, 0.0), jax.random.PRNGKey(2),
    )
    np.testing.assert_allclose(np.asarray(charged.stamina), [0.95, 0.95], atol=1e-6)

    rested = game.step(
        game.step(charged, _force(0.0, 0.0), jax.random.PRNGKey(3)),
        _force(0.0, 0.0), jax.random.PRNGKey(4),
    )
    np.testing.assert_allclose(np.asarray(rested.stamina), [0.975, 0.975], atol=1e-6)


def test_low_stamina_reduces_available_thrust():
    game = _deterministic(stamina_enabled=True, stamina_min_force=0.25)
    state = game.initial_state(jax.random.PRNGKey(0)).replace(
        stamina=jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    )
    after_0 = game.step(state, _force(1.0, 0.0), jax.random.PRNGKey(1))
    assert np.linalg.norm(np.asarray(after_0.pending)) == pytest.approx(0.25)


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
    dict(force_asymmetry=-0.1), dict(force_asymmetry=1.0),
    dict(stamina_drain_rate=-0.1), dict(stamina_recovery_rate=-0.1),
    dict(stamina_min_force=-0.1), dict(stamina_min_force=1.1),
    dict(observation_delay=-1), dict(observation_delay=1.5),
    dict(observation_frames=0), dict(observation_frames=2.5),
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _game(**kwargs)
