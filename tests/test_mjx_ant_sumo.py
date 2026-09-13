"""Rules-level checks for `games.mjx_sumo.MjxAntSumo`.

The same shape of suite as `test_mjx_sumo.py` -- the `games.sequential` contract,
then the information structure, then the rules -- plus the things only an
articulated body brings: that the model assembles with both ants' joints and
actuators laid out where the code indexes them, and that an ant can lose by
being knocked onto its back as well as by leaving the ring.

Two ants on MJX are expensive on CPU, so every game here runs the shortest
horizon and fewest substeps that still make the assertion, and the losing
conditions are tested by *constructing* the state that triggers them rather than
by playing until one happens by chance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.mjx_sumo import MjxAntSumo, MjxSumoState
from games.sequential import TERMINAL
from games.spaces import HybridAction

NU = 8  # joint torques per ant


def _game(**kwargs) -> MjxAntSumo:
    """Cheapest bout that still steps real physics: timestep stays the stable 0.01."""
    kwargs.setdefault("horizon", 3)
    kwargs.setdefault("dt", 0.02)
    kwargs.setdefault("substeps", 2)
    return MjxAntSumo(**kwargs)


def _deterministic(**kwargs) -> MjxAntSumo:
    return _game(random_orientation=False, start_jitter=0.0, **kwargs)


def _torque(*values: float) -> HybridAction:
    value = jnp.asarray(values if values else (0.0,) * NU, dtype=jnp.float32)
    return HybridAction(kind=jnp.zeros((), dtype=jnp.int32), value=value)


def _idle(obs, mask, key):
    return _torque()


def _place(game: MjxAntSumo, state: MjxSumoState, player: int, xy=None, height=None) -> MjxSumoState:
    """Move one torso, leaving the rest of the configuration alone."""
    root = game._root_qpos[player]
    qpos = state.data.qpos
    if xy is not None:
        qpos = qpos.at[root:root + 2].set(jnp.asarray(xy, dtype=qpos.dtype))
    if height is not None:
        qpos = qpos.at[root + 2].set(jnp.asarray(height, dtype=qpos.dtype))
    return state.replace(data=state.data.replace(qpos=qpos))


# ---- the model the code indexes into ---------------------------------------


def test_the_assembled_model_has_both_ants_where_the_code_looks_for_them():
    game = _game()
    model = game._mj_model
    assert model.nu == 2 * NU
    assert game.action_space(0).box.shape == (NU,)
    assert game.obs_dim(0) == game.obs_dim(1) == MjxAntSumo.OBS_DIM == 65
    # Disjoint, non-overlapping slices per ant -- an index error here would read
    # the wrong agent's state and no rules test would notice.
    assert game._root_qpos[0] != game._root_qpos[1]
    indices = np.concatenate([np.asarray(game._hinge_qpos[p]) for p in (0, 1)])
    assert len(set(indices.tolist())) == 2 * NU


def test_each_ant_collides_with_the_floor_and_the_other_but_not_itself():
    """The stock ant switches agent-agent collision off; sumo needs it back on."""
    game = _game()
    model = game._mj_model
    contype, conaffinity = {}, {}
    for geom in range(model.ngeom):
        name = model.geom(geom).name
        owner = "floor" if name == "floor" else name[:2]
        contype.setdefault(owner, model.geom_contype[geom])
        conaffinity.setdefault(owner, model.geom_conaffinity[geom])
    collides = lambda a, b: bool(contype[a] & conaffinity[b] or contype[b] & conaffinity[a])
    assert collides("p0", "p1")
    assert collides("p0", "floor") and collides("p1", "floor")
    assert not collides("p0", "p0") and not collides("p1", "p1")


# ---- the sequential-game contract ------------------------------------------


def test_state_shapes_and_dtypes_survive_a_step():
    game = _game()
    state = game.initial_state(jax.random.PRNGKey(0))
    stepped = game.step(state, _torque(*([0.3] * NU)), jax.random.PRNGKey(1))
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        assert before.shape == after.shape and before.dtype == after.dtype
    for player in (0, 1):
        assert game.observation(player, state).shape == (game.obs_dim(player),)
        assert game.action_mask(player, state).shape == (game.num_kinds(player),)
        assert np.all(np.isfinite(np.asarray(game.observation(player, state))))


def test_stepping_a_terminal_state_is_a_noop():
    """The guard in `games.sequential.step` has to hold across every `mjx.Data` leaf."""
    game = _deterministic(horizon=1)
    state = game.initial_state(jax.random.PRNGKey(0))
    for _ in range(game.max_steps):
        state = game.step(state, _torque(), jax.random.PRNGKey(1))
    assert int(game.current_player(state)) == TERMINAL

    stepped = game.step(state, _torque(*([1.0] * NU)), jax.random.PRNGKey(7))
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_players_alternate_and_the_physics_waits_for_player_one():
    game = _game()
    state = game.initial_state(jax.random.PRNGKey(0))
    assert int(game.current_player(state)) == 0
    after_0 = game.step(state, _torque(*([1.0] * NU)), jax.random.PRNGKey(1))
    assert int(game.current_player(after_0)) == 1
    np.testing.assert_array_equal(np.asarray(after_0.data.qpos), np.asarray(state.data.qpos))
    after_1 = game.step(after_0, _torque(*([1.0] * NU)), jax.random.PRNGKey(2))
    assert int(game.current_player(after_1)) == 0
    assert not np.allclose(np.asarray(after_1.data.qpos), np.asarray(state.data.qpos))


def test_random_play_terminates_within_max_steps_with_bounded_payoffs():
    game = _game(margin_weight=0.5)
    finals, payoffs = jax.vmap(
        lambda key: game.play_episode(
            (game.random_action_fn(0), game.random_action_fn(1)), key
        )
    )(jax.random.split(jax.random.PRNGKey(0), 8))
    payoffs = np.asarray(payoffs)
    assert np.all(np.isfinite(payoffs))
    assert np.all(np.abs(payoffs) <= 1.0)
    assert np.all(np.asarray(jax.vmap(game.current_player)(finals)) == TERMINAL)


# ---- information ------------------------------------------------------------


def test_player_one_never_sees_player_zeros_live_torque():
    game = _game()
    state = game.initial_state(jax.random.PRNGKey(3))
    parked = [game.step(state, _torque(*torques), jax.random.PRNGKey(4)) for torques in (
        [1.0] * NU, [-1.0] * NU, [0.5, -0.5] * (NU // 2),
    )]
    assert not np.allclose(np.asarray(parked[0].pending), np.asarray(parked[1].pending))
    observations = [np.asarray(game.observation(1, s)) for s in parked]
    for obs in observations[1:]:
        np.testing.assert_array_equal(obs, observations[0])


def test_a_symmetric_start_looks_identical_from_both_seats():
    """Both ants stand in the same pose, and their egocentric frames are mirror images."""
    game = _game(start_jitter=0.0)
    for seed in range(3):
        state = game.initial_state(jax.random.PRNGKey(seed))
        np.testing.assert_allclose(
            np.asarray(game.observation(0, state)), np.asarray(game.observation(1, state)), atol=1e-5
        )


def test_the_action_is_not_rotated_into_the_egocentric_frame():
    """A joint torque is already in the body's own frame; only the observation rotates."""
    game = _game()
    state = game.initial_state(jax.random.PRNGKey(0))
    torques = [0.7, -0.2, 0.1, 0.4, -0.9, 0.3, 0.0, 1.0]
    parked = game.step(state, _torque(*torques), jax.random.PRNGKey(1))
    np.testing.assert_allclose(np.asarray(parked.pending), np.asarray(torques), atol=1e-6)


# ---- rules ------------------------------------------------------------------


def test_an_ant_outside_the_ring_has_lost():
    game = _deterministic()
    state = game.initial_state(jax.random.PRNGKey(0))
    state = _place(game, state, 0, xy=(-(game.ring_radius + 0.5), 0.0))
    state = state.replace(turn=jnp.asarray(1, dtype=jnp.int32))  # player 1 to act: resolves
    stepped = game.step(state, _torque(), jax.random.PRNGKey(1))
    assert bool(stepped.done)
    assert float(game.payoff(stepped)) == -1.0


def test_an_ant_on_its_back_has_lost_even_inside_the_ring():
    """The second losing condition of Bansal et al.: down is out."""
    game = _deterministic()
    state = game.initial_state(jax.random.PRNGKey(0))
    state = _place(game, state, 1, xy=(0.5, 0.0), height=game.knockdown_height - 0.1)
    state = state.replace(turn=jnp.asarray(1, dtype=jnp.int32))
    stepped = game.step(state, _torque(), jax.random.PRNGKey(1))
    assert bool(stepped.done)
    assert float(game.payoff(stepped)) == 1.0  # player 1 went down, so player 0 wins
    # ... and it was the knockdown, not the ring, that did it.
    positions = np.asarray(game._positions(stepped))
    assert np.linalg.norm(positions[1]) < game.ring_radius


def test_the_ant_that_goes_first_loses_a_simultaneous_exit():
    game = _deterministic()
    state = game.initial_state(jax.random.PRNGKey(0))
    # Player 0 starts outside; player 1 is only knocked down once the step runs.
    state = _place(game, state, 0, xy=(-(game.ring_radius + 0.5), 0.0))
    state = _place(game, state, 1, height=game.knockdown_height + 0.02)
    state = state.replace(turn=jnp.asarray(1, dtype=jnp.int32))
    stepped = game.step(state, _torque(), jax.random.PRNGKey(1))
    assert bool(stepped.done)
    assert float(game.payoff(stepped)) == -1.0  # player 0 was already out at substep 0


def test_an_idle_bout_is_a_draw_and_nobody_falls_over():
    """Two ants left alone stand still; pure sumo pays exactly nothing for that."""
    game = _deterministic(horizon=6, margin_weight=0.5)
    final, payoff = game.play_episode((_idle, _idle), jax.random.PRNGKey(0))
    assert not bool(final.done)
    heights = np.asarray(game._heights(final.data))
    assert np.all(heights > game.knockdown_height)
    assert abs(float(payoff)) < 1e-5  # symmetric start, symmetric idling


def test_the_timeout_margin_rewards_the_ant_nearer_the_centre():
    game = _deterministic(margin_weight=0.5)
    state = game.initial_state(jax.random.PRNGKey(0))
    state = _place(game, state, 0, xy=(0.3, 0.0))
    state = _place(game, state, 1, xy=(2.1, 0.0))
    state = state.replace(turn=jnp.asarray(game.max_steps, dtype=jnp.int32))
    assert int(game.current_player(state)) == TERMINAL
    expected = 0.5 * (2.1 - 0.3) / game.ring_radius
    assert float(game.payoff(state)) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("kwargs", [
    dict(horizon=0), dict(substeps=0), dict(ring_radius=0.0), dict(gear=0.0),
    dict(margin_weight=1.5), dict(start_jitter=-1.0),
    dict(knockdown_height=0.9),   # at or above start_height: over before it begins
    dict(start_distance=10.0),    # wider than the ring
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _game(**kwargs)
