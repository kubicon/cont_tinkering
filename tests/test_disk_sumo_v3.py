"""Rules-level checks for `games.disk_sumo_v3.DiskSumoV3`.

The v2 machinery it inherits (archetype dealing, hidden types, stamina) is
covered by `tests/test_disk_sumo_v2.py`; this re-checks the contract on the new
integrator and then the two things v3 adds: the grip cut-out and the impact
brace.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.configs import GAME_CONFIGS
from games.disk_sumo_v2 import DiskSumoV2
from games.disk_sumo_v3 import DEFAULT_ARCHETYPES, DiskSumoV3
from games.sequential import TERMINAL
from games.spaces import HybridAction

BATCH = 256


def _force(x: float, y: float) -> HybridAction:
    return HybridAction(
        kind=jnp.zeros((), dtype=jnp.int32), value=jnp.asarray([x, y], dtype=jnp.float32)
    )


def _deterministic(**kwargs) -> DiskSumoV3:
    return DiskSumoV3(random_orientation=False, start_jitter=0.0, **kwargs)


def _with(game: DiskSumoV3, state, a: str, b: str):
    names = game.archetype_names
    return state.replace(archetype=jnp.asarray([names.index(a), names.index(b)], dtype=jnp.int32))


def _control_step(game, state, force0, force1):
    state = game.step(state, _force(*force0), jax.random.PRNGKey(1))
    return game.step(state, _force(*force1), jax.random.PRNGKey(2))


def _apart(game, state):
    """Disks far apart on the x-axis, so neither touches the other."""
    return state.replace(pos=jnp.asarray([[-0.5, 0.0], [0.5, 0.0]]))


# ---- the sequential-game contract ------------------------------------------


def test_random_play_terminates_with_bounded_payoffs():
    game = DiskSumoV3(horizon=20, margin_weight=0.5)
    finals, payoffs = jax.vmap(
        lambda key: game.play_episode((game.random_action_fn(0), game.random_action_fn(1)), key)
    )(jax.random.split(jax.random.PRNGKey(0), BATCH))
    payoffs = np.asarray(payoffs)
    assert np.all(np.isfinite(payoffs)) and np.all(np.abs(payoffs) <= 1.0)
    assert np.all(np.asarray(jax.vmap(game.current_player)(finals)) == TERMINAL)


def test_shapes_and_dtypes_survive_a_step():
    game = DiskSumoV3()
    state = game.initial_state(jax.random.PRNGKey(0))
    stepped = _control_step(game, state, (0.3, -0.2), (1.0, 0.0))
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        assert before.shape == after.shape and before.dtype == after.dtype
    assert game.obs_dim(0) == 11 + 2 * 3 + 2
    for player in (0, 1):
        assert game.observation(player, stepped).shape == (game.obs_dim(player),)


def test_config_builds_the_game_and_leaves_v2_alone():
    game = GAME_CONFIGS["disk_sumo_v3"](grip_width=0.1).build()
    assert isinstance(game, DiskSumoV3) and game.grip_width == 0.1
    assert game.archetype_names == tuple(DEFAULT_ARCHETYPES) == ("rammer", "grinder", "anchor")
    v2 = GAME_CONFIGS["disk_sumo_v2"]().build()
    assert not isinstance(v2, DiskSumoV3) and len(v2.archetype_names) == 7
    assert "top_speed" not in v2.archetype_table["quick"]


def test_random_play_is_fair_on_average():
    game = DiskSumoV3(horizon=30, margin_weight=0.5)
    _, payoffs = jax.vmap(
        lambda key: game.play_episode((game.random_action_fn(0), game.random_action_fn(1)), key)
    )(jax.random.split(jax.random.PRNGKey(1), 2000))
    assert abs(float(np.mean(np.asarray(payoffs)))) < 0.05


# ---- the new mechanics ------------------------------------------------------


def test_without_new_traits_the_physics_is_v2s():
    """Neutral `top_speed`, `impact_brace` and `brace_speed` reproduce v2 exactly."""
    traits = {"a": {"force": 1.2, "mass": 1.3, "brace": 2.0}, "b": {"drag": 0.7, "drain": 1.4}}
    kwargs = dict(archetypes=["a", "b"], archetype_traits=traits, horizon=15)
    v2, v3 = DiskSumoV2(**kwargs), DiskSumoV3(**kwargs)
    s2 = v2.initial_state(jax.random.PRNGKey(5))
    s3 = v3.initial_state(jax.random.PRNGKey(5))
    for force0, force1 in [((1.0, 0.0), (1.0, 0.2)), ((0.0, 0.0), (1.0, 0.0))] * 5:
        s2, s3 = _control_step(v2, s2, force0, force1), _control_step(v3, s3, force0, force1)
    np.testing.assert_allclose(np.asarray(s3.pos), np.asarray(s2.pos), atol=1e-5)
    np.testing.assert_allclose(np.asarray(s3.vel), np.asarray(s2.vel), atol=1e-5)
    np.testing.assert_allclose(np.asarray(v3.observation(0, s3)), np.asarray(v2.observation(0, s2)), atol=1e-5)


def test_grip_is_full_below_top_speed_and_cuts_out_at_it():
    traits = {"slow": {"top_speed": 0.25}, "free": {}}
    game = _deterministic(archetypes=["slow", "free"], archetype_traits=traits, egocentric=False,
                          ring_radius=100.0, stamina_drain_rate=0.0)
    state = _with(game, _apart(game, game.initial_state(jax.random.PRNGKey(0))), "slow", "free")
    early = _control_step(game, state, (0.0, 1.0), (0.0, 1.0))
    # From rest both disks accelerate identically: no grip is lost well below top_speed.
    np.testing.assert_allclose(np.asarray(early.vel[0]), np.asarray(early.vel[1]), rtol=1e-4)
    for _ in range(30):
        state = _control_step(game, state, (0.0, 1.0), (0.0, 1.0))
    speed = np.linalg.norm(np.asarray(state.vel), axis=-1)
    cap = 0.25 * game._speed_scale
    # Thrust fades over a band `grip_width` wide, so the disk settles just past top_speed.
    assert 0.95 * cap < speed[0] < (1.0 + 2.0 * game.grip_width) * cap
    assert speed[1] > 0.75 * game._speed_scale


def test_a_disk_knocked_back_past_top_speed_cannot_brake():
    traits = {"slow": {"top_speed": 0.1}}
    game = _deterministic(archetypes=["slow"], archetype_traits=traits, egocentric=False)
    state = _apart(game, game.initial_state(jax.random.PRNGKey(0)))
    sliding = state.replace(vel=jnp.asarray([[0.0, -1.0], [0.0, 0.0]]))
    braking = _control_step(game, sliding, (0.0, 1.0), (0.0, 0.0))
    coasting = _control_step(game, sliding, (0.0, 0.0), (0.0, 0.0))
    # Far above top_speed the thrust has no grip: pushing against the slide changes nothing.
    np.testing.assert_allclose(np.asarray(braking.vel), np.asarray(coasting.vel), atol=1e-6)
    still = _control_step(game, state.replace(vel=jnp.zeros((2, 2))), (0.0, 1.0), (0.0, 0.0))
    assert float(still.vel[0, 1]) > 0.0


def test_impact_brace_ignores_slow_motion_and_soaks_up_fast_motion():
    traits = {"anchor": {"impact_brace": 8.0, "brace_speed": 0.5}, "plain": {}}
    game = _deterministic(archetypes=["anchor", "plain"], archetype_traits=traits, egocentric=False)
    state = _apart(game, game.initial_state(jax.random.PRNGKey(0)))

    def decay(name, speed, effort):
        s = _with(game, state, name, name).replace(vel=jnp.asarray([[speed, 0.0], [0.0, 0.0]]))
        stepped = _control_step(game, s, (0.0, effort), (0.0, 0.0))
        return float(stepped.vel[0, 0]) / speed

    slow, fast = 0.4 * game._speed_scale, 2.0 * game._speed_scale
    # Below brace_speed the anchor is an ordinary disk; above it an idle anchor stops hard.
    assert decay("anchor", slow, 0.0) == pytest.approx(decay("plain", slow, 0.0), abs=1e-6)
    assert decay("anchor", fast, 0.0) < 0.5 * decay("plain", fast, 0.0)
    # Thrusting switches the brace off.
    assert decay("anchor", fast, 1.0) == pytest.approx(decay("plain", fast, 1.0), abs=1e-5)


@pytest.mark.parametrize("kwargs", [
    dict(grip_width=0.0),
    dict(archetype_traits={"rammer": {"top_speed": 0.0}}),
    dict(archetype_traits={"anchor": {"impact_brace": -1.0}}),
    dict(archetype_traits={"anchor": {"brace_speed": -0.1}}),
    dict(archetype_traits={"anchor": {"radius": 1.0}}),
    dict(archetypes=["quick"]),
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DiskSumoV3(**kwargs)
