"""Rules-level checks for `games.disk_sumo_v2.DiskSumoV2`.

The `games.sequential` contract and the information structure are re-checked
(the step and observation are overridden), then the archetype mechanics: that
each seat's archetype is drawn independently, that every trait acts on the
physics the way its name says, and that a hidden archetype really is hidden.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.configs import GAME_CONFIGS
from games.disk_sumo_v2 import DEFAULT_ARCHETYPES, DiskSumoV2
from games.sequential import TERMINAL
from games.spaces import HybridAction

BATCH = 256


def _force(x: float, y: float) -> HybridAction:
    return HybridAction(
        kind=jnp.zeros((), dtype=jnp.int32), value=jnp.asarray([x, y], dtype=jnp.float32)
    )


def _deterministic(**kwargs) -> DiskSumoV2:
    return DiskSumoV2(random_orientation=False, start_jitter=0.0, **kwargs)


def _with(game: DiskSumoV2, state, a: str, b: str):
    names = game.archetype_names
    return state.replace(archetype=jnp.asarray([names.index(a), names.index(b)], dtype=jnp.int32))


def _control_step(game, state, force0, force1):
    state = game.step(state, _force(*force0), jax.random.PRNGKey(1))
    return game.step(state, _force(*force1), jax.random.PRNGKey(2))


# ---- the sequential-game contract ------------------------------------------


def test_random_play_terminates_with_bounded_payoffs():
    game = DiskSumoV2(horizon=20, margin_weight=0.5)
    finals, payoffs = jax.vmap(
        lambda key: game.play_episode((game.random_action_fn(0), game.random_action_fn(1)), key)
    )(jax.random.split(jax.random.PRNGKey(0), BATCH))
    payoffs = np.asarray(payoffs)
    assert np.all(np.isfinite(payoffs)) and np.all(np.abs(payoffs) <= 1.0)
    assert np.all(np.asarray(jax.vmap(game.current_player)(finals)) == TERMINAL)


@pytest.mark.parametrize("observe_velocity_change", [False, True])
def test_shapes_and_dtypes_survive_a_step(observe_velocity_change):
    game = DiskSumoV2(observe_velocity_change=observe_velocity_change)
    state = game.initial_state(jax.random.PRNGKey(0))
    stepped = _control_step(game, state, (0.3, -0.2), (1.0, 0.0))
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        assert before.shape == after.shape and before.dtype == after.dtype
    assert game.obs_dim(0) == 11 + 2 * 7 + 2 + 4 * observe_velocity_change
    for player in (0, 1):
        assert game.observation(player, stepped).shape == (game.obs_dim(player),)


def test_config_builds_the_game():
    game = GAME_CONFIGS["disk_sumo_v2"](archetypes=["quick", "heavy"]).build()
    assert isinstance(game, DiskSumoV2) and game.archetype_names == ("quick", "heavy")
    assert game.obs_dim(0) == 11 + 4 + 2


# ---- archetype assignment and information ---------------------------------


def test_archetypes_are_drawn_independently_per_seat():
    game = DiskSumoV2()
    archetypes = np.asarray(
        jax.vmap(game.initial_state)(jax.random.split(jax.random.PRNGKey(0), 4000)).archetype
    )
    for seat in (0, 1):
        counts = np.bincount(archetypes[:, seat], minlength=game.num_archetypes)
        assert counts.min() > 0.7 * 4000 / game.num_archetypes
    assert np.mean(archetypes[:, 0] == archetypes[:, 1]) == pytest.approx(1 / 7, abs=0.03)


def test_observed_archetypes_are_one_hot_own_first():
    game = DiskSumoV2()
    state = _with(game, game.initial_state(jax.random.PRNGKey(0)), "strong", "brace")
    k = game.num_archetypes
    obs = np.asarray(game.observation(1, state))
    np.testing.assert_array_equal(obs[11:11 + k], np.eye(k)[game.archetype_names.index("brace")])
    np.testing.assert_array_equal(obs[11 + k:11 + 2 * k], np.eye(k)[game.archetype_names.index("strong")])


def test_hidden_archetype_and_stamina_are_invisible_to_the_opponent():
    game = DiskSumoV2(observe_opponent_archetype=False)
    base = game.initial_state(jax.random.PRNGKey(0))
    a = _with(game, base, "quick", "strong").replace(stamina=jnp.asarray([1.0, 0.2]))
    b = _with(game, base, "burst", "strong").replace(stamina=jnp.asarray([0.4, 0.2]))
    np.testing.assert_array_equal(np.asarray(game.observation(1, a)), np.asarray(game.observation(1, b)))
    assert not np.array_equal(np.asarray(game.observation(0, a)), np.asarray(game.observation(0, b)))


def test_opponent_stamina_can_stay_visible_with_a_hidden_archetype():
    game = DiskSumoV2(observe_opponent_archetype=False, observe_opponent_stamina=True)
    state = game.initial_state(jax.random.PRNGKey(0)).replace(stamina=jnp.asarray([0.3, 0.8]))
    np.testing.assert_allclose(np.asarray(game.observation(0, state))[-2:], [0.3, 0.8])


def test_player_one_never_sees_player_zeros_live_force():
    game = DiskSumoV2(observe_velocity_change=True)
    state = game.initial_state(jax.random.PRNGKey(3))
    parked = [game.step(state, _force(x, y), jax.random.PRNGKey(4)) for x, y in ((1.0, 0.0), (0.0, -1.0))]
    np.testing.assert_array_equal(
        np.asarray(game.observation(1, parked[0])), np.asarray(game.observation(1, parked[1]))
    )


def test_velocity_change_observation_reports_the_last_control_step():
    game = _deterministic(observe_velocity_change=True, egocentric=False)
    state = game.initial_state(jax.random.PRNGKey(0))
    stepped = _control_step(game, state, (0.0, 1.0), (0.0, 0.0))
    obs = np.asarray(game.observation(0, stepped))
    expected = np.asarray(stepped.vel - stepped.prev_vel) / game._speed_scale
    np.testing.assert_allclose(obs[-4:], expected.reshape(-1), atol=1e-6)
    assert obs[-3] > 0.0  # player 0 accelerated along +y


# ---- traits act on the physics --------------------------------------------


def test_strong_wins_a_head_on_shove_against_quick():
    game = _deterministic(margin_weight=0.5)
    state = _with(game, game.initial_state(jax.random.PRNGKey(0)), "strong", "quick")
    state = state.replace(stamina=jnp.ones(2))
    for _ in range(15):
        state = _control_step(game, state, (1.0, 0.0), (1.0, 0.0))
    radii = np.linalg.norm(np.asarray(state.pos), axis=-1)
    assert radii[1] > radii[0]


def test_quick_has_the_highest_top_speed_and_nimble_the_fastest_response():
    game = _deterministic(drag=0.5)
    traits = {n: {k: float(v) for k, v in game.traits(jnp.asarray(i)).items()}
              for i, n in enumerate(game.archetype_names)}
    top = {n: t["force"] / t["drag"] for n, t in traits.items()}
    response = {n: t["mass"] / t["drag"] for n, t in traits.items()}
    assert max(top, key=top.get) == "quick"
    assert min(response, key=response.get) == "nimble"


def test_heavy_is_knocked_back_less_than_nimble():
    game = _deterministic(drag=0.0, contact_damping=0.0)
    state = game.initial_state(jax.random.PRNGKey(0)).replace(
        pos=jnp.asarray([[-0.2, 0.0], [0.2, 0.0]]),
        vel=jnp.asarray([[1.5, 0.0], [0.0, 0.0]]),
        turn=jnp.asarray(1, dtype=jnp.int32),
    )
    kicks = {}
    for target in ("heavy", "nimble"):
        stepped = game.step(_with(game, state, "strong", target), _force(0.0, 0.0), jax.random.PRNGKey(0))
        kicks[target] = float(stepped.vel[1, 0])
    assert 0.0 < kicks["heavy"] < kicks["nimble"]


def test_brace_resists_being_pushed_only_while_idle():
    game = _deterministic()
    state = _with(game, game.initial_state(jax.random.PRNGKey(0)), "strong", "brace").replace(
        vel=jnp.asarray([[0.0, 0.0], [1.0, 0.0]]), turn=jnp.asarray(1, dtype=jnp.int32)
    )
    idle = game.step(state, _force(0.0, 0.0), jax.random.PRNGKey(0))
    sideways = game.step(state, _force(0.0, 1.0), jax.random.PRNGKey(0))  # full effort, no x-force
    assert float(idle.vel[1, 0]) < float(sideways.vel[1, 0])


def test_endurance_drains_slower_and_burst_faster():
    game = _deterministic()
    state = game.initial_state(jax.random.PRNGKey(0))
    drained = {}
    for name in ("endurance", "burst"):
        s = _with(game, state, name, name)
        for _ in range(5):
            s = _control_step(game, s, (1.0, 0.0), (1.0, 0.0))
        drained[name] = 1.0 - float(s.stamina[0])
    assert drained["endurance"] < 0.25 < drained["burst"]


def test_contact_conserves_momentum_with_unequal_masses():
    game = _deterministic(drag=0.0, contact_damping=0.0)
    state = _with(game, game.initial_state(jax.random.PRNGKey(0)), "heavy", "nimble").replace(
        pos=jnp.asarray([[-0.2, 0.0], [0.2, 0.05]]),
        vel=jnp.asarray([[1.0, 0.0], [-0.5, 0.0]]),
        turn=jnp.asarray(1, dtype=jnp.int32),
    )
    stepped = game.step(state, _force(0.0, 0.0), jax.random.PRNGKey(0))
    mass = np.asarray(game.traits(state.archetype)["mass"])[:, None]
    assert not np.allclose(np.asarray(stepped.vel), np.asarray(state.vel))
    np.testing.assert_allclose(
        (mass * np.asarray(stepped.vel)).sum(0), (mass * np.asarray(state.vel)).sum(0), atol=1e-4
    )


def test_trait_overrides_and_new_archetypes():
    game = DiskSumoV2(archetypes=["quick", "tank"],
                      archetype_traits={"quick": {"force": 0.5}, "tank": {"mass": 3.0}})
    assert game.archetype_table["quick"]["force"] == 0.5
    assert game.archetype_table["quick"]["drag"] == DEFAULT_ARCHETYPES["quick"]["drag"]
    assert game.archetype_table["tank"] == {"force": 1.0, "mass": 3.0, "drag": 1.0, "drain": 1.0,
                                             "recovery": 1.0, "min_force": 1.0, "brace": 0.0}


def test_random_play_is_fair_on_average():
    game = DiskSumoV2(horizon=30, margin_weight=0.5)
    _, payoffs = jax.vmap(
        lambda key: game.play_episode((game.random_action_fn(0), game.random_action_fn(1)), key)
    )(jax.random.split(jax.random.PRNGKey(1), 2000))
    assert abs(float(np.mean(np.asarray(payoffs)))) < 0.05


@pytest.mark.parametrize("kwargs", [
    dict(archetypes=[]), dict(archetypes=["quick", "quick"]), dict(archetypes=["giant"]),
    dict(archetype_traits={"quick": {"radius": 2.0}}),
    dict(archetype_traits={"quick": {"mass": 0.0}}),
    dict(archetype_traits={"brace": {"brace": -1.0}}),
    dict(archetype_traits={"burst": {"min_force": 5.0}}),
    dict(horizon=0),
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DiskSumoV2(**kwargs)
