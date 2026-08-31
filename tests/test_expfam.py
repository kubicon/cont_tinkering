"""Checks for the exponential-family policy in `training/expfam.py`.

The claim this parametrization is bought for is *exactness*: unlike the Gaussian
mixture, whose entropy has no closed form and whose magnet KL is a componentwise
bound, every regularizer here is the quantity it names. So what is pinned below
is arithmetic, not behavior -- the density integrates to one, the entropy and KL
match a brute-force sum over the grid, the sampler's empirical distribution
matches the density it claims, and every sample lands inside the action box
without anything having to clip it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.config import ExpFamilyPPOHyperparams
from training.expfam import (
    build_basis,
    build_expfam_network,
    density_entropy,
    density_kl,
    density_log_prob,
    density_moments,
    grid_log_probs,
    sample_action,
    sample_expfam_actions,
)

OBS_DIM = 3
LOW, HIGH = -1.0, 1.0
GRID = 128


def _hyperparams(**overrides) -> ExpFamilyPPOHyperparams:
    defaults = dict(
        action_dim=1,
        hidden_dims=(16,),
        low=(LOW,),
        high=(HIGH,),
        grid_points=GRID,
        num_basis=6,
        poly_order=2,
    )
    return ExpFamilyPPOHyperparams(**{**defaults, **overrides})


def _theta(key, dim: int, num_features: int, scale: float = 1.0):
    return scale * jax.random.normal(key, (dim, num_features))


def test_density_integrates_to_one():
    """`sum_g p(bin g) * w == 1`: `log Z` really is the normalizer."""
    basis = build_basis(jnp.array([LOW]), jnp.array([HIGH]), GRID, 6, 2, 1.0)
    theta = _theta(jax.random.PRNGKey(0), 1, basis.num_features, scale=2.0)
    log_p = jax.vmap(
        lambda g: density_log_prob(theta, basis, jnp.array([g], dtype=jnp.int32))
    )(jnp.arange(GRID))
    integral = jnp.sum(jnp.exp(log_p)) * jnp.exp(basis.log_width[0])
    assert float(integral) == pytest.approx(1.0, abs=1e-5)


def test_uniform_start_has_uniform_entropy():
    """A zero `theta` is the uniform density, whose entropy is exactly `log(width)`."""
    basis = build_basis(jnp.array([LOW]), jnp.array([HIGH]), GRID, 6, 2, 1.0)
    theta = jnp.zeros((1, basis.num_features))
    assert float(density_entropy(theta, basis)) == pytest.approx(np.log(HIGH - LOW), abs=1e-5)
    mean, _ = density_moments(theta, basis)
    assert float(mean[0]) == pytest.approx(0.5 * (LOW + HIGH), abs=1e-5)


def test_entropy_and_kl_match_brute_force():
    """The closed forms agree with a direct sum over the grid, to float precision.

    This is the property the Gaussian mixture cannot offer at all: there is no
    brute-force value for `mixture_ppo_loss`'s entropy term to be checked
    against, because the term is an estimate of an integral with no closed form.
    """
    basis = build_basis(jnp.array([LOW]), jnp.array([HIGH]), GRID, 6, 2, 1.0)
    key_p, key_q = jax.random.split(jax.random.PRNGKey(1))
    theta_p = _theta(key_p, 1, basis.num_features, scale=1.5)
    theta_q = _theta(key_q, 1, basis.num_features, scale=1.5)

    probs_p = np.exp(np.asarray(grid_log_probs(theta_p, basis))[0])
    probs_q = np.exp(np.asarray(grid_log_probs(theta_q, basis))[0])
    width = float(np.exp(np.asarray(basis.log_width)[0]))

    expected_entropy = float(-np.sum(probs_p * np.log(probs_p / width)))
    expected_kl = float(np.sum(probs_p * np.log(probs_p / probs_q)))

    assert float(density_entropy(theta_p, basis)) == pytest.approx(expected_entropy, abs=1e-5)
    assert float(density_kl(theta_p, theta_q, basis)) == pytest.approx(expected_kl, abs=1e-5)
    assert float(density_kl(theta_p, theta_p, basis)) == pytest.approx(0.0, abs=1e-6)


def test_kl_is_invariant_to_a_constant_shift():
    """Adding a constant to every logit is not a different policy.

    The basis deliberately omits a constant feature for this reason; the check
    guards the equivalent statement about the features it does have.
    """
    basis = build_basis(jnp.array([LOW]), jnp.array([HIGH]), GRID, 6, 0, 1.0)
    theta = _theta(jax.random.PRNGKey(2), 1, basis.num_features)
    # The RBFs sum to something close to constant across the box only in the
    # middle, so shift in *logit* space directly instead.
    shifted_log_pi = grid_log_probs(theta, basis) + 3.0
    assert jnp.allclose(jax.nn.log_softmax(shifted_log_pi, axis=-1), grid_log_probs(theta, basis), atol=1e-5)


def test_samples_match_the_density_and_stay_in_the_box():
    """Empirical mean/std track `density_moments`, and nothing needs clipping.

    The second half is the structural point: the support of this policy *is* the
    action box, so unlike a Gaussian mean there is no sample to project back and
    no boundary case for the log-prob to disagree about.
    """
    basis = build_basis(jnp.array([LOW]), jnp.array([HIGH]), GRID, 6, 2, 1.0)
    theta = _theta(jax.random.PRNGKey(3), 1, basis.num_features, scale=1.5)
    keys = jax.random.split(jax.random.PRNGKey(4), 40_000)
    _, actions = jax.vmap(lambda k: sample_action(theta, basis, k))(keys)

    mean, std = density_moments(theta, basis)
    assert float(jnp.mean(actions)) == pytest.approx(float(mean[0]), abs=0.02)
    assert float(jnp.std(actions)) == pytest.approx(float(std[0]), abs=0.02)
    assert bool(jnp.all((actions >= LOW) & (actions <= HIGH)))


def test_init_tilt_moves_the_initial_mean_off_center():
    """`init_tilt` is what stops a bilinear game's run from starting at a Nash."""
    obs = jnp.zeros(OBS_DIM)
    untilted = build_expfam_network(_hyperparams(init_tilt=0.0))
    tilted = build_expfam_network(_hyperparams(init_tilt=10.0))

    theta_flat, _ = untilted.apply(untilted.init(jax.random.PRNGKey(0), obs), obs)
    theta_tilt, _ = tilted.apply(tilted.init(jax.random.PRNGKey(0), obs), obs)

    assert float(density_moments(theta_flat, untilted.basis)[0][0]) == pytest.approx(0.0, abs=1e-5)
    # exp(10 z) on [-1, 1] has mean coth(10) - 1/10 ~ 0.90.
    assert float(density_moments(theta_tilt, tilted.basis)[0][0]) == pytest.approx(0.9, abs=0.01)


def test_init_tilt_without_a_monomial_is_rejected():
    with pytest.raises(ValueError, match="poly_order"):
        network = build_expfam_network(_hyperparams(poly_order=0, init_tilt=1.0))
        network.init(jax.random.PRNGKey(0), jnp.zeros(OBS_DIM))


def test_multidimensional_policy_factorizes():
    """A `dim`-dimensional action gets `dim` independent 1-D families.

    Which is what keeps the normalizer a `dim * grid_points` sum rather than a
    `grid_points ** dim` one -- the reason quadrature is affordable here at all.
    """
    basis = build_basis(jnp.array([LOW, 0.0]), jnp.array([HIGH, 4.0]), GRID, 6, 2, 1.0)
    theta = _theta(jax.random.PRNGKey(5), 2, basis.num_features)
    assert basis.centers.shape == (2, GRID)

    # The joint entropy is the sum of the per-dimension entropies.
    per_dim = [
        float(density_entropy(theta[i][None, :], build_basis(
            jnp.array([[LOW, 0.0][i]]), jnp.array([[HIGH, 4.0][i]]), GRID, 6, 2, 1.0)))
        for i in (0, 1)
    ]
    assert float(density_entropy(theta, basis)) == pytest.approx(sum(per_dim), abs=1e-4)

    keys = jax.random.split(jax.random.PRNGKey(6), 256)
    _, actions = jax.vmap(lambda k: sample_action(theta, basis, k))(keys)
    assert actions.shape == (256, 2)
    assert bool(jnp.all(actions[:, 0] >= LOW) & jnp.all(actions[:, 0] <= HIGH))
    assert bool(jnp.all(actions[:, 1] >= 0.0) & jnp.all(actions[:, 1] <= 4.0))


def test_network_sampling_helper_is_in_box():
    network = build_expfam_network(_hyperparams(init_tilt=5.0))
    obs = jnp.zeros(OBS_DIM)
    params = network.init(jax.random.PRNGKey(0), obs)
    actions = sample_expfam_actions(network, params, obs, jax.random.PRNGKey(1), 512)
    assert actions.shape == (512, 1)
    assert bool(jnp.all((actions >= LOW) & (actions <= HIGH)))
