"""The leg-count variants of `games.mjx_sumo.MjxLeggedSumo`: ant, bug and spider.

`test_mjx_ant_sumo.py` is the deep suite -- it tests the sumo *rules* once, on
the ant, and those rules live in shared code that no body overrides. What is
left to check is what actually varies with the leg count, and that is what this
file does, for every body in the family: that the model assembles with the joints
and actuators the code indexes by name, that the widths derived from
`LEG_ANGLES` match the compiled model, that the body stands up, and that a bout
still looks identical from both seats.

Two walkers on MJX are expensive on CPU, so every game here is the shortest,
coarsest bout that still steps real physics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.mjx_sumo import MjxAntSumo, MjxBugSumo, MjxLeggedSumo, MjxSpiderSumo
from games.spaces import HybridAction

BODIES = (MjxAntSumo, MjxBugSumo, MjxSpiderSumo)


def _game(cls, **kwargs) -> MjxLeggedSumo:
    kwargs.setdefault("horizon", 2)
    kwargs.setdefault("dt", 0.02)
    kwargs.setdefault("substeps", 2)
    kwargs.setdefault("random_orientation", False)
    kwargs.setdefault("start_jitter", 0.0)
    return cls(**kwargs)


@pytest.mark.parametrize("cls", BODIES, ids=lambda c: c.__name__)
def test_the_widths_on_the_class_match_the_compiled_model(cls):
    """`LEG_ANGLES` is the only thing a body declares; everything else follows from it."""
    game = _game(cls)
    legs = len(cls.LEG_ANGLES)
    torques = 2 * legs  # a hip and an ankle each

    assert len(cls.HINGES) == torques
    assert game._mj_model.nu == 2 * torques  # both agents
    assert game.action_space(0).box.shape == (torques,)
    assert cls.AGENT_OBS_DIM == cls.TORSO_OBS_DIM + 2 * torques
    assert cls.OBS_DIM == 2 * cls.AGENT_OBS_DIM + 3
    assert game.obs_dim(0) == game.obs_dim(1) == cls.OBS_DIM

    state = game.initial_state(jax.random.PRNGKey(0))
    assert game.observation(0, state).shape == (cls.OBS_DIM,)


@pytest.mark.parametrize("cls", BODIES, ids=lambda c: c.__name__)
def test_every_leg_is_indexed_once_and_stands_the_same_way(cls):
    game = _game(cls)
    indices = np.concatenate([np.asarray(game._hinge_qpos[p]) for p in (0, 1)])
    assert len(set(indices.tolist())) == 2 * len(cls.HINGES)  # disjoint per agent
    assert game._root_qpos[0] != game._root_qpos[1]

    # Hips centred, every ankle bent the same way: one positive range bends them
    # all downwards, whatever the leg count (see assets/mjx_legged_leg.xml).
    stance = np.asarray(game._stance[0])
    np.testing.assert_allclose(stance[0::2], 0.0, atol=1e-6)
    np.testing.assert_allclose(stance[1::2], cls.ANKLE_STANCE, atol=1e-6)


@pytest.mark.parametrize("cls", BODIES, ids=lambda c: c.__name__)
def test_the_body_stands_up_and_stays_put_when_left_alone(cls):
    """A body that collapses under its own weight would end every bout at step one."""
    game = _game(cls, horizon=4)
    idle = lambda obs, mask, key: HybridAction(
        kind=jnp.zeros((), dtype=jnp.int32),
        value=jnp.zeros(len(cls.HINGES), dtype=jnp.float32),
    )
    final, payoff = game.play_episode((idle, idle), jax.random.PRNGKey(0))
    heights = np.asarray(game._heights(final.data))
    assert np.all(heights > game.knockdown_height)
    assert not bool(final.done)
    assert abs(float(payoff)) < 1e-5  # symmetric start, symmetric idling


@pytest.mark.parametrize("cls", BODIES, ids=lambda c: c.__name__)
def test_a_symmetric_start_looks_identical_from_both_seats(cls):
    game = _game(cls)
    state = game.initial_state(jax.random.PRNGKey(0))
    np.testing.assert_allclose(
        np.asarray(game.observation(0, state)), np.asarray(game.observation(1, state)), atol=1e-5
    )


@pytest.mark.parametrize("cls", BODIES, ids=lambda c: c.__name__)
def test_each_body_collides_with_the_floor_and_the_other_but_not_itself(cls):
    game = _game(cls)
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


def test_the_bodies_are_distinct_sizes():
    """Guards against a copy-paste subclass that silently reuses another's legs."""
    widths = {cls.__name__: (len(cls.LEG_ANGLES), cls.OBS_DIM) for cls in BODIES}
    assert len(set(widths.values())) == len(BODIES), widths
    assert widths["MjxSpiderSumo"] < widths["MjxAntSumo"] < widths["MjxBugSumo"]


def test_a_body_with_no_legs_declared_cannot_be_built():
    with pytest.raises(ValueError, match="LEG_ANGLES"):
        MjxLeggedSumo()
