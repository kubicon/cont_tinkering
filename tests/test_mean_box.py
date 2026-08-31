"""Checks for the straight-through mean projection and its box penalty.

`clip_means` used to be a plain `jnp.clip` on the mean head, which is
zero-gradient outside the box: a mean that ever left it stopped being trained
at that observation. What replaces it is a straight-through projection (the
policy uses the projected mean, the raw mean keeps receiving the gradient taken
at the boundary) plus `mean_box_penalty_coef * mean_box_excess` to supply the
restoring force the projection on its own has no way to provide.

Pinned here: the projected values are exactly the old clip's, the gradient
outside the box is no longer zero, the penalty points back into the box, and a
zero-width dimension -- the one case where a dead gradient is right -- stays
frozen.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from training.config import MixturePPOHyperparams
from training.mixture import (
    Episode,
    build_mixture_network,
    build_mixture_ppo_loss_fn,
    component_to_kind,
    expand_kind_mask,
    mean_box_excess,
    project_means_to_box,
    sample_mixture_component,
)

OBS_DIM = 4
LOW, HIGH = 0.0, 1.0


def _network(clip_means: bool, num_components: int = 3):
    hyperparams = MixturePPOHyperparams(
        action_dim=1,
        hidden_dims=(8,),
        num_components=num_components,
        num_atoms=0,
        low=(LOW,),
        high=(HIGH,),
        clip_means=clip_means,
    )
    network = build_mixture_network(hyperparams)
    params = network.init(jax.random.PRNGKey(0), jnp.zeros(OBS_DIM))
    return network, params


def _push_means_out_of_the_box(params, offset: float = 0.5):
    """Bias the mean head to `HIGH + offset`, i.e. every component outside the box."""
    head = params["params"]["means_head"]
    return {
        "params": {
            **params["params"],
            "means_head": {
                "kernel": jnp.zeros_like(head["kernel"]),
                "bias": jnp.full_like(head["bias"], HIGH + offset),
            },
        }
    }


def _episode(network, params, batch: int = 32, shared_obs: bool = False) -> Episode:
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    mask = expand_kind_mask(jnp.ones((1,), dtype=bool), network.num_components)

    def one(key):
        obs_key, sample_key, reward_key = jax.random.split(key, 3)
        obs = jnp.zeros(OBS_DIM) if shared_obs else jax.random.normal(obs_key, (OBS_DIM,))
        logits, means, scale_trils, value = network.apply(params, obs)
        component, raw_action = sample_mixture_component(
            logits, means, scale_trils, mask, 0, sample_key
        )
        return Episode(
            actor=jnp.int32(0),
            obs=obs,
            action_mask=mask,
            logits=logits,
            means=means,
            scale_trils=scale_trils,
            magnet_logits=logits,
            magnet_means=means,
            magnet_scale_trils=scale_trils,
            component=component,
            raw_action=raw_action,
            action_kind=component_to_kind(component, 0),
            action_value=jnp.clip(raw_action, LOW, HIGH),
            value=value,
            reward=jax.random.normal(reward_key, ()),
        )

    return jax.vmap(one)(keys)


def _loss_grad(network, params, episode, mean_box_penalty_coef, shared_obs=False):
    loss_fn = build_mixture_ppo_loss_fn(
        0, 0.1, 0.1, 0.05, 0.05, 0.2, 0.2,
        mean_box_penalty_coef=mean_box_penalty_coef,
        shared_obs=shared_obs,
    )
    return jax.grad(lambda p: loss_fn(p, network, episode, 0.1, 0.5, 0.0)[0])(params)


# ---- the projection itself ------------------------------------------------


def test_the_projected_value_is_the_clip_it_replaces():
    low, high = jnp.array([LOW]), jnp.array([HIGH])
    means = jnp.array([[-0.4], [0.5], [1.9]])
    assert jnp.allclose(project_means_to_box(means, low, high), jnp.clip(means, low, high))


def test_a_mean_outside_the_box_still_receives_a_gradient():
    """The whole point: `jnp.clip` gives 0 here, the straight-through projection gives 1."""
    low, high = jnp.array([LOW]), jnp.array([HIGH])
    outside = jnp.array([[-0.4], [1.9]])

    straight_through = jax.grad(lambda m: jnp.sum(project_means_to_box(m, low, high)))(outside)
    hard_clip = jax.grad(lambda m: jnp.sum(jnp.clip(m, low, high)))(outside)

    assert jnp.allclose(straight_through, jnp.ones_like(outside))
    assert jnp.allclose(hard_clip, jnp.zeros_like(outside))


def test_a_zero_width_dimension_stays_frozen():
    """`min_bet == max_bet`: the mean is a constant, so a dead gradient is the right answer."""
    low, high = jnp.array([LOW, 1.0]), jnp.array([HIGH, 1.0])
    means = jnp.array([[1.5, 1.5]])

    projected = project_means_to_box(means, low, high)
    grad = jax.grad(lambda m: jnp.sum(project_means_to_box(m, low, high)))(means)

    assert jnp.allclose(projected, jnp.array([[HIGH, 1.0]]))
    assert jnp.allclose(grad, jnp.array([[1.0, 0.0]]))
    # And nothing to pull back either, however far out the raw mean sits.
    assert mean_box_excess(means, low, high) == pytest.approx(0.25)


# ---- the penalty ----------------------------------------------------------


def test_the_penalty_is_zero_inside_the_box_and_points_back_in_outside_it():
    low, high = jnp.array([LOW]), jnp.array([HIGH])
    inside = jnp.array([[0.2], [0.9]])
    outside = jnp.array([[-0.5], [1.5]])

    assert mean_box_excess(inside, low, high) == pytest.approx(0.0)
    assert jnp.allclose(jax.grad(mean_box_excess)(inside, low, high), jnp.zeros_like(inside))

    assert mean_box_excess(outside, low, high) == pytest.approx(0.5)
    grad = jax.grad(mean_box_excess)(outside, low, high)
    assert grad[0, 0] < 0.0  # below the box: pushed up
    assert grad[1, 0] > 0.0  # above the box: pushed down


@pytest.mark.parametrize("shared_obs", [False, True])
def test_the_loss_trains_an_escaped_mean_and_pulls_it_back(shared_obs):
    """End to end: the mean head is outside the box, and the loss does something about it."""
    network, params = _network(clip_means=True)
    params = _push_means_out_of_the_box(params)
    episode = _episode(network, params, shared_obs=shared_obs)

    unpenalized = _loss_grad(network, params, episode, 0.0, shared_obs)["params"]["means_head"]
    penalized = _loss_grad(network, params, episode, 2.0, shared_obs)["params"]["means_head"]

    # Under the old hard clip this was exactly zero -- the dead gradient.
    assert jnp.any(unpenalized["bias"] != 0.0)
    # The penalty's share is `2 * coef * excess > 0`, i.e. descent moves the bias down.
    assert jnp.all(penalized["bias"] - unpenalized["bias"] > 0.0)


def test_the_penalty_is_reported_and_is_inert_without_clip_means():
    network, params = _network(clip_means=False)
    params = _push_means_out_of_the_box(params)
    episode = _episode(network, params)

    loss_fn = build_mixture_ppo_loss_fn(0, 0.1, 0.1, 0.05, 0.05, 0.2, 0.2, mean_box_penalty_coef=2.0)
    _, metrics = loss_fn(params, network, episode, 0.1, 0.5, 0.0)
    assert metrics["mean_box_penalty"] == pytest.approx(0.0)

    network, params = _network(clip_means=True)
    params = _push_means_out_of_the_box(params)
    _, metrics = loss_fn(params, network, _episode(network, params), 0.1, 0.5, 0.0)
    assert metrics["mean_box_penalty"] == pytest.approx(2.0 * 3 * 0.25)
