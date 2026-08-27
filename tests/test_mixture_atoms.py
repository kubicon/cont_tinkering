"""Checks for the atom + legality-mask machinery in `training.mixture`.

Two things are being pinned here. First, that atoms and masks actually *do*
nothing when there are none (`num_atoms == 0`, all-`True` mask), so the one-shot
games in `games.examples` train through exactly the arithmetic they did before.
Second, that when they are present every term that has no business being defined
-- a Gaussian log-prob for a point mass, a probability for an illegal action --
is exactly zero rather than merely small, and carries no gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.spaces import MASKED_LOGIT
from training.actor_critic import (
    categorical_kl,
    gaussian_log_prob,
    masked_categorical_entropy,
    masked_log_softmax,
)
from training.config import MixturePPOHyperparams
from training.mixture import (
    Episode,
    build_mixture_network,
    build_mixture_ppo_loss_fn,
    component_to_kind,
    expand_kind_mask,
    gaussian_component_index,
    mixture_log_probs,
    mixture_marginal_log_prob,
    sample_mixture_component,
)

OBS_DIM = 4
ACTION_DIM = 1


def _network(num_atoms: int, num_components: int = 3):
    hyperparams = MixturePPOHyperparams(
        action_dim=ACTION_DIM,
        hidden_dims=(8,),
        num_components=num_components,
        num_atoms=num_atoms,
        low=(0.0,) * ACTION_DIM,
        high=(1.0,) * ACTION_DIM,
    )
    network = build_mixture_network(hyperparams)
    params = network.init(jax.random.PRNGKey(0), jnp.zeros(OBS_DIM))
    return network, params


def _episode(network, params, kind_mask, batch: int = 64, seed: int = 0) -> Episode:
    """A batch of samples drawn at random observations under a fixed kind mask."""
    num_atoms = network.num_atoms
    keys = jax.random.split(jax.random.PRNGKey(seed), batch)

    def one(key):
        obs_key, sample_key, reward_key = jax.random.split(key, 3)
        obs = jax.random.normal(obs_key, (OBS_DIM,))
        logits, means, log_stds, value = network.apply(params, obs)
        mask = expand_kind_mask(kind_mask, network.num_components)
        component, raw_action = sample_mixture_component(
            logits, means, log_stds, mask, num_atoms, sample_key
        )
        return Episode(
            obs=obs,
            action_mask=mask,
            logits=logits,
            means=means,
            log_stds=log_stds,
            magnet_logits=logits,
            magnet_means=means,
            magnet_log_stds=log_stds,
            component=component,
            raw_action=raw_action,
            action_kind=component_to_kind(component, num_atoms),
            action_value=jnp.clip(raw_action, 0.0, 1.0),
            value=value,
            reward=jax.random.normal(reward_key, ()),
        )

    return jax.vmap(one)(keys)


def _loss(network, params, episode, **coefs):
    defaults = dict(
        category_entropy_coef=0.1,
        gaussian_entropy_coef=0.1,
        trpo_category_kl_coef=0.05,
        trpo_gaussian_kl_coef=0.05,
        magnet_category_kl_coef=0.2,
        magnet_gaussian_kl_coef=0.2,
    )
    loss_fn = build_mixture_ppo_loss_fn(**{**defaults, **coefs})
    return loss_fn(params, network, episode, 0.1, 0.5, 0.0)


# ---- the degenerate case must be the old arithmetic ----------------------


def test_masked_helpers_reduce_to_their_unmasked_forms():
    """With nothing masked, every masked helper is the plain formula it replaced."""
    logits_p = jnp.array([0.3, -1.2, 2.0, 0.1])
    logits_q = jnp.array([-0.4, 0.9, 0.2, -1.1])
    mask = jnp.ones_like(logits_p, dtype=bool)

    np.testing.assert_allclose(
        masked_log_softmax(logits_p, mask), jax.nn.log_softmax(logits_p), rtol=1e-6
    )

    log_p = jax.nn.log_softmax(logits_p)
    np.testing.assert_allclose(
        masked_categorical_entropy(logits_p, mask), -jnp.sum(jnp.exp(log_p) * log_p), rtol=1e-6
    )

    log_q = jax.nn.log_softmax(logits_q)
    np.testing.assert_allclose(
        categorical_kl(logits_p, logits_q, mask),
        jnp.sum(jnp.exp(log_p) * (log_p - log_q)),
        rtol=1e-6,
    )


def test_marginal_log_prob_reduces_to_the_plain_mixture_density_without_atoms():
    network, params = _network(num_atoms=0)
    logits, means, log_stds, _ = network.apply(params, jnp.zeros(OBS_DIM))
    mask = jnp.ones_like(logits, dtype=bool)
    action = jnp.array([0.4])

    per_component = jax.vmap(gaussian_log_prob, in_axes=(None, 0, 0))(action, means, log_stds)
    expected = jax.nn.logsumexp(jax.nn.log_softmax(logits) + per_component)

    np.testing.assert_allclose(
        mixture_marginal_log_prob(logits, means, log_stds, mask, action, 0), expected, rtol=1e-6
    )


def test_an_atom_free_policy_keeps_the_plain_categorical_width():
    network, params = _network(num_atoms=0, num_components=3)
    logits, means, log_stds, _ = network.apply(params, jnp.zeros(OBS_DIM))
    assert logits.shape == (3,)
    assert means.shape == log_stds.shape == (3, ACTION_DIM)


# ---- indexing ------------------------------------------------------------


def test_logits_head_widens_by_the_number_of_atoms():
    network, params = _network(num_atoms=2, num_components=3)
    logits, means, log_stds, _ = network.apply(params, jnp.zeros(OBS_DIM))
    assert logits.shape == (5,)
    assert means.shape == log_stds.shape == (3, ACTION_DIM)  # atoms have no mean or spread


def test_expand_kind_mask_replicates_the_continuous_bit():
    kind_mask = jnp.array([True, False, True])  # 2 atoms, continuous legal
    np.testing.assert_array_equal(
        expand_kind_mask(kind_mask, 3), jnp.array([True, False, True, True, True])
    )
    np.testing.assert_array_equal(
        expand_kind_mask(jnp.array([True, True, False]), 3),
        jnp.array([True, True, False, False, False]),
    )


def test_component_maps_to_kind_and_gaussian_row():
    components = jnp.arange(5)
    np.testing.assert_array_equal(component_to_kind(components, 2), jnp.array([0, 1, 2, 2, 2]))
    np.testing.assert_array_equal(
        gaussian_component_index(components, 2), jnp.array([0, 0, 0, 1, 2])
    )


# ---- masks bind at sampling time ----------------------------------------


@pytest.mark.parametrize(
    "kind_mask", [(True, False, True), (True, True, False), (False, True, True)]
)
def test_sampling_never_draws_a_masked_component(kind_mask):
    network, params = _network(num_atoms=2, num_components=3)
    logits, means, log_stds, _ = network.apply(params, jnp.zeros(OBS_DIM))
    mask = expand_kind_mask(jnp.asarray(kind_mask), network.num_components)

    components = jax.vmap(
        lambda k: sample_mixture_component(logits, means, log_stds, mask, 2, k)[0]
    )(jax.random.split(jax.random.PRNGKey(0), 4000))

    assert bool(jnp.all(mask[components]))


def test_a_single_legal_kind_has_zero_entropy_and_probability_one():
    logits = jnp.array([0.5, -2.0, 1.0, 3.0])
    mask = jnp.array([False, True, False, False])
    log_probs = masked_log_softmax(logits, mask)

    assert float(jnp.exp(log_probs[1])) == pytest.approx(1.0)
    assert float(masked_categorical_entropy(logits, mask)) == pytest.approx(0.0, abs=1e-6)


def test_masked_logits_receive_no_gradient():
    """An illegal action's logit must not be pushed around by the entropy bonus."""
    logits = jnp.array([0.5, -2.0, 1.0, 3.0])
    mask = jnp.array([True, False, True, False])
    grad = jax.grad(lambda x: masked_categorical_entropy(x, mask))(logits)
    np.testing.assert_array_equal(grad[~mask], jnp.zeros(2))


def test_masked_entries_do_not_perturb_the_legal_distribution():
    """Whatever the illegal logits hold, the legal ones renormalize the same way."""
    mask = jnp.array([True, False, True])
    base = jnp.array([0.5, -2.0, 1.0])
    wild = jnp.array([0.5, 1e3, 1.0])
    np.testing.assert_allclose(
        jnp.exp(masked_log_softmax(base, mask))[mask],
        jnp.exp(masked_log_softmax(wild, mask))[mask],
        rtol=1e-6,
    )


def test_marginal_density_ignores_the_atoms_probability():
    """The Gaussian entropy term is conditional, so shifting atom mass must not move it."""
    network, params = _network(num_atoms=2, num_components=3)
    logits, means, log_stds, _ = network.apply(params, jnp.zeros(OBS_DIM))
    mask = jnp.ones_like(logits, dtype=bool)
    action = jnp.array([0.4])

    shifted = logits.at[:2].add(5.0)  # make folding far more likely
    np.testing.assert_allclose(
        mixture_marginal_log_prob(logits, means, log_stds, mask, action, 2),
        mixture_marginal_log_prob(shifted, means, log_stds, mask, action, 2),
        rtol=1e-6,
    )


# ---- the loss ------------------------------------------------------------


def test_an_atom_sample_contributes_no_gaussian_log_prob():
    network, params = _network(num_atoms=2, num_components=3)
    logits, means, log_stds, _ = network.apply(params, jnp.zeros(OBS_DIM))
    mask = jnp.ones_like(logits, dtype=bool)
    action = jnp.array([0.4])

    for component in (0, 1):
        _, gaussian = mixture_log_probs(
            logits, means, log_stds, mask, jnp.asarray(component), action, 2
        )
        assert float(gaussian) == 0.0
    for component in (2, 3, 4):
        _, gaussian = mixture_log_probs(
            logits, means, log_stds, mask, jnp.asarray(component), action, 2
        )
        assert float(gaussian) != 0.0


def test_loss_zeroes_every_gaussian_term_when_only_atoms_are_legal():
    """Facing a bet in Kuhn: fold or call, no bet size to speak of."""
    network, params = _network(num_atoms=2, num_components=3)
    episode = _episode(network, params, jnp.array([True, True, False]))
    loss, metrics = _loss(network, params, episode)

    assert float(metrics["atom_frac"]) == 1.0
    for key in (
        "gaussian_policy_loss",
        "gaussian_entropy",
        "gaussian_approx_kl",
        "gaussian_clip_frac",
        "trpo_gaussian_kl",
        "magnet_gaussian_kl",
    ):
        assert float(metrics[key]) == 0.0, key
    assert jnp.isfinite(loss)


def test_loss_and_gradients_stay_finite_with_a_mix_of_atoms_and_components():
    network, params = _network(num_atoms=2, num_components=3)
    episode = _episode(network, params, jnp.array([True, False, True]))
    (loss, metrics), grads = jax.value_and_grad(
        lambda p: _loss(network, p, episode), has_aux=True
    )(params)

    assert 0.0 < float(metrics["atom_frac"]) < 1.0  # both branches are actually exercised
    assert jnp.isfinite(loss)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))


def test_gradients_stay_finite_when_only_atoms_are_legal():
    """The all-Gaussians-masked path is where a `0 * -inf` would surface as a NaN."""
    network, params = _network(num_atoms=2, num_components=3)
    episode = _episode(network, params, jnp.array([True, True, False]))
    grads = jax.grad(lambda p: _loss(network, p, episode)[0])(params)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))


def test_masked_logit_constant_is_finite():
    """`-inf` would make `p * log p` a NaN in the entropy; the sentinel must not be one."""
    assert jnp.isfinite(jnp.asarray(MASKED_LOGIT))
    assert float(jnp.exp(masked_log_softmax(jnp.zeros(3), jnp.array([True, True, False]))[2])) == 0.0
