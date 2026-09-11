"""Checks for `bucket_means`: one fixed bucket of the box per mixture component.

Pinned here: the buckets tile the box, each mean starts at its bucket's center
and cannot leave the bucket however far the raw head is pushed, exploration
draws inside the sampled component's bucket, and the loss scores an explored
sample against that same bucket's uniform density.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.config import MixturePPOHyperparams
from training.mixture import (
    behavior_gaussian_log_prob,
    bucket_bounds,
    build_mixture_network,
    component_boxes,
    gaussian_component_index,
    sample_mixture_component,
)

OBS_DIM = 4
LOW, HIGH = 0.25, 2.0
NUM_COMPONENTS = 4


def _network(bucket_means: bool = True, action_dim: int = 1, num_components: int = NUM_COMPONENTS):
    hyperparams = MixturePPOHyperparams(
        action_dim=action_dim,
        hidden_dims=(8,),
        num_components=num_components,
        num_atoms=0,
        low=(LOW,) * action_dim,
        high=(HIGH,) * action_dim,
        clip_means=True,
        bucket_means=bucket_means,
    )
    network = build_mixture_network(hyperparams)
    params = network.init(jax.random.PRNGKey(0), jnp.zeros(OBS_DIM))
    return network, params


def test_buckets_tile_the_box():
    lows, highs = bucket_bounds(jnp.array([LOW]), jnp.array([HIGH]), NUM_COMPONENTS)
    edges = np.linspace(LOW, HIGH, NUM_COMPONENTS + 1)
    np.testing.assert_allclose(lows[:, 0], edges[:-1], rtol=1e-6)
    np.testing.assert_allclose(highs[:, 0], edges[1:], rtol=1e-6)


def test_two_dimensional_buckets_form_a_grid():
    lows, highs = bucket_bounds(jnp.zeros(2), jnp.ones(2), 4)
    cells = {(float(lo[0]), float(lo[1])) for lo in lows}
    assert cells == {(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)}
    np.testing.assert_allclose(highs - lows, 0.5)
    with pytest.raises(ValueError, match="perfect"):
        bucket_bounds(jnp.zeros(2), jnp.ones(2), 3)


def test_means_start_at_bucket_centers():
    network, params = _network()
    _, means, _, _ = network.apply(params, jax.random.normal(jax.random.PRNGKey(1), (OBS_DIM,)))
    lows, highs = component_boxes(network)
    np.testing.assert_allclose(means, 0.5 * (lows + highs), rtol=1e-6)


@pytest.mark.parametrize("push", [-50.0, 50.0])
def test_means_stay_in_their_bucket(push):
    network, params = _network()
    head = params["params"]["means_head"]
    pushed = {"params": {**params["params"], "means_head": {
        "kernel": head["kernel"], "bias": jnp.full_like(head["bias"], push),
    }}}
    _, means, _, _ = network.apply(pushed, jnp.zeros(OBS_DIM))
    lows, highs = component_boxes(network)
    assert bool(jnp.all((means >= lows) & (means <= highs)))
    # Saturated at the edge the push points to.
    np.testing.assert_allclose(means, lows if push < 0 else highs, atol=1e-5)


def test_unbucketed_boxes_are_the_whole_box():
    network, _ = _network(bucket_means=False)
    lows, highs = component_boxes(network)
    np.testing.assert_allclose(lows, LOW)
    np.testing.assert_allclose(highs, HIGH)


def test_exploration_draws_inside_the_sampled_bucket():
    network, params = _network()
    logits, means, scale_trils, _ = network.apply(params, jnp.zeros(OBS_DIM))
    lows, highs = component_boxes(network)
    mask = jnp.ones_like(logits, dtype=bool)

    def one(key):
        # explore_eps = 1: every continuous action is the uniform draw (the
        # sampler itself accepts it; only the loss's density needs eps < 1).
        return sample_mixture_component(
            logits, means, scale_trils, mask, 0, key, explore_eps=jnp.float32(1.0),
            low=lows, high=highs,
        )

    components, actions = jax.vmap(one)(jax.random.split(jax.random.PRNGKey(2), 4096))
    index = gaussian_component_index(components, 0)
    assert bool(jnp.all((actions >= lows[index]) & (actions <= highs[index])))
    # Every bucket gets explored, and its draws spread across the bucket.
    for k in range(NUM_COMPONENTS):
        drawn = actions[index == k, 0]
        assert drawn.size > 0
        width = float(highs[k, 0] - lows[k, 0])
        assert float(drawn.max() - drawn.min()) > 0.9 * width


def test_behavior_density_uses_the_bucket_width():
    # A point inside a bucket of width w has uniform density 1/w there.
    x = jnp.array([0.5])
    log_gauss = jnp.float32(-jnp.inf)  # isolate the uniform part
    eps = jnp.float32(0.5)
    lp = behavior_gaussian_log_prob(log_gauss, x, eps, jnp.array([0.25]), jnp.array([0.6875]))
    np.testing.assert_allclose(lp, jnp.log(0.5) - jnp.log(0.4375), rtol=1e-5)
