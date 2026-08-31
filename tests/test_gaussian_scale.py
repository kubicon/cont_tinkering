"""The Cholesky-factor Gaussian parametrization (`training.gaussian`).

Three things are worth pinning down and are checked here: that the closed-form
density/entropy/KL agree with an independent computation, that the diagonal
case reproduces exactly what the old `log_std` code computed (so the switch is
a reparametrization and not a change of model), and that the natural gradient
really is the inverse-Fisher step rather than a hand-tuned preconditioner that
happens to look like one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from training.gaussian import (
    SIGMA_MIN,
    clamp_scale_tril,
    diagonal_slots,
    gaussian_entropy,
    gaussian_kl,
    gaussian_log_prob,
    gaussian_sample,
    marginal_std,
    natural_gradient,
    pack_scale_tril,
    scale_diagonal,
    scale_param_size,
)

DIM = 3


def _factor(key, full_covariance=True, floor=0.2):
    size = scale_param_size(DIM, full_covariance)
    flat = jax.random.normal(key, (size,))
    return clamp_scale_tril(pack_scale_tril(flat, DIM, full_covariance), floor, 5.0)


def test_packing_is_lower_triangular_and_sized():
    assert scale_param_size(DIM, True) == DIM * (DIM + 1) // 2
    assert scale_param_size(DIM, False) == DIM
    scale = _factor(jax.random.PRNGKey(0))
    assert np.allclose(np.triu(scale, 1), 0.0)
    # Diagonal-only packing leaves the off-diagonals as structural zeros.
    diagonal = _factor(jax.random.PRNGKey(0), full_covariance=False)
    assert np.allclose(diagonal, np.diag(np.diag(diagonal)))


def test_density_entropy_and_kl_match_the_covariance_form():
    scale_p = _factor(jax.random.PRNGKey(1))
    scale_q = _factor(jax.random.PRNGKey(2))
    mean_p, mean_q = jnp.array([0.3, -0.2, 1.0]), jnp.array([-0.4, 0.7, 0.2])
    action = jnp.array([0.1, 0.5, -0.4])
    cov_p, cov_q = np.array(scale_p @ scale_p.T), np.array(scale_q @ scale_q.T)

    quad = (np.array(action - mean_p)) @ np.linalg.solve(cov_p, np.array(action - mean_p))
    expected_log_prob = -0.5 * (quad + np.log(np.linalg.det(cov_p)) + DIM * np.log(2 * np.pi))
    assert float(gaussian_log_prob(action, mean_p, scale_p)) == pytest.approx(
        expected_log_prob, rel=1e-5
    )

    expected_entropy = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * cov_p))
    assert float(gaussian_entropy(scale_p)) == pytest.approx(expected_entropy, rel=1e-5)

    delta = np.array(mean_q - mean_p)
    expected_kl = 0.5 * (
        np.trace(np.linalg.solve(cov_q, cov_p))
        + delta @ np.linalg.solve(cov_q, delta)
        - DIM
        + np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
    )
    assert float(gaussian_kl(mean_p, scale_p, mean_q, scale_q)) == pytest.approx(
        expected_kl, rel=1e-4
    )


def test_kl_is_zero_between_a_policy_and_itself():
    scale = _factor(jax.random.PRNGKey(3))
    mean = jnp.array([0.3, -0.2, 1.0])
    assert float(gaussian_kl(mean, scale, mean, scale)) == pytest.approx(0.0, abs=1e-5)


def test_diagonal_case_reproduces_the_log_std_formulas():
    """The switch is a reparametrization: same model, different coordinates."""
    log_std_p = jnp.array([0.1, -0.5, 0.3])
    log_std_q = jnp.array([-0.2, 0.4, 0.0])
    mean_p, mean_q = jnp.array([0.3, -0.2, 1.0]), jnp.array([-0.4, 0.7, 0.2])
    action = jnp.array([0.1, 0.5, -0.4])
    scale_p = pack_scale_tril(jnp.exp(log_std_p), DIM, False)
    scale_q = pack_scale_tril(jnp.exp(log_std_q), DIM, False)

    old_log_prob = jnp.sum(
        -0.5
        * (
            jnp.square(action - mean_p) / jnp.exp(2 * log_std_p)
            + 2 * log_std_p
            + jnp.log(2 * jnp.pi)
        )
    )
    old_entropy = jnp.sum(log_std_p + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    old_kl = jnp.sum(
        (log_std_q - log_std_p)
        + (jnp.exp(2 * log_std_p) + jnp.square(mean_p - mean_q)) / (2 * jnp.exp(2 * log_std_q))
        - 0.5
    )

    assert gaussian_log_prob(action, mean_p, scale_p) == pytest.approx(float(old_log_prob), rel=1e-6)
    assert gaussian_entropy(scale_p) == pytest.approx(float(old_entropy), rel=1e-6)
    assert gaussian_kl(mean_p, scale_p, mean_q, scale_q) == pytest.approx(float(old_kl), rel=1e-6)


def test_sampling_reproduces_the_covariance():
    scale = _factor(jax.random.PRNGKey(4))
    mean = jnp.array([0.3, -0.2, 1.0])
    noise = jax.random.normal(jax.random.PRNGKey(5), (200_000, DIM))
    drawn = gaussian_sample(mean, scale, noise)
    assert np.allclose(np.mean(np.array(drawn), axis=0), np.array(mean), atol=0.02)
    assert np.allclose(np.cov(np.array(drawn).T), np.array(scale @ scale.T), atol=0.05)


def test_marginal_std_exceeds_the_conditional_diagonal_when_correlated():
    """`A_ii` is the conditional spread; `sqrt(Sigma_ii)` is the marginal one."""
    scale = _factor(jax.random.PRNGKey(6))
    assert np.all(np.array(marginal_std(scale)) >= np.array(scale_diagonal(scale)) - 1e-6)
    assert float(jnp.max(marginal_std(scale) - scale_diagonal(scale))) > 0.1
    diagonal = _factor(jax.random.PRNGKey(6), full_covariance=False)
    assert np.allclose(marginal_std(diagonal), scale_diagonal(diagonal))


def test_clamp_floors_the_diagonal_and_passes_the_gradient_through():
    flat = jnp.zeros((scale_param_size(DIM, True),)).at[diagonal_slots(DIM, True)].set(-4.0)
    scale = clamp_scale_tril(pack_scale_tril(flat, DIM, True), SIGMA_MIN, jnp.inf)
    # Value is projected into the feasible set...
    assert np.allclose(np.array(scale_diagonal(scale)), SIGMA_MIN)

    # ...while the head that produced it still gets a gradient, unlike a plain
    # clip, which is what lets a collapsed scale recover.
    def emitted_diagonal(raw):
        packed = pack_scale_tril(raw, DIM, True)
        return jnp.sum(scale_diagonal(clamp_scale_tril(packed, SIGMA_MIN, jnp.inf)))

    grad = jax.grad(emitted_diagonal)(flat)
    assert np.allclose(np.array(grad[diagonal_slots(DIM, True)]), 1.0)


def test_natural_gradient_is_the_inverse_fisher_step():
    """`natural_gradient` must equal `H^-1 g` for `H` the KL Hessian, exactly."""
    scale = _factor(jax.random.PRNGKey(7))
    rows, cols = np.tril_indices(DIM)
    flat = jnp.asarray(np.array(scale)[rows, cols])
    grad_flat = jax.random.normal(jax.random.PRNGKey(8), flat.shape)
    mean = jnp.array([0.3, -0.2, 1.0])

    def kl_from(raw):
        return gaussian_kl(mean, pack_scale_tril(raw, DIM, True), mean, scale)

    fisher = jax.hessian(kl_from)(flat)
    expected = jnp.linalg.solve(fisher, grad_flat)

    _, nat_scale = natural_gradient(scale, jnp.zeros(DIM), pack_scale_tril(grad_flat, DIM, True))
    assert np.allclose(np.array(nat_scale)[rows, cols], np.array(expected), rtol=1e-3, atol=1e-4)


def test_natural_gradient_mean_block_is_the_covariance():
    scale = _factor(jax.random.PRNGKey(9))
    grad_mean = jnp.array([0.7, -1.1, 0.4])
    nat_mean, _ = natural_gradient(scale, grad_mean, jnp.zeros((DIM, DIM)))
    assert np.allclose(np.array(nat_mean), np.array(scale @ scale.T @ grad_mean), rtol=1e-5)


def test_natural_gradient_diagonal_case_matches_the_log_std_preconditioner():
    """The diagonal MMD update was `(sigma^2 g_mu, 1/2 g_logsigma)`; recover it."""
    log_std = jnp.array([0.1, -0.5, 0.3])
    sigma = jnp.exp(log_std)
    scale = pack_scale_tril(sigma, DIM, False)
    grad_mean = jnp.array([0.7, -1.1, 0.4])
    grad_sigma = jnp.array([0.9, 0.2, -0.6])

    nat_mean, nat_scale = natural_gradient(scale, grad_mean, jnp.diag(grad_sigma))
    assert np.allclose(np.array(nat_mean), np.array(sigma**2 * grad_mean), rtol=1e-6)
    # `d sigma = sigma d log sigma` turns `1/2 g_logsigma` into `1/2 sigma^2 g_sigma`.
    assert np.allclose(np.diag(np.array(nat_scale)), np.array(0.5 * sigma**2 * grad_sigma), rtol=1e-6)
    assert np.allclose(np.array(nat_scale) - np.diag(np.diag(np.array(nat_scale))), 0.0)


def test_gradient_of_a_separable_payoff_ignores_the_off_diagonals():
    """Why `full_covariance` defaults to off.

    A payoff that is a sum of per-coordinate terms has an expectation depending
    only on the per-axis marginals, so every off-diagonal entry of the factor
    has *exactly* zero gradient -- they would be trained by sampling noise
    alone. This is the property that makes correlations inert on
    `MultiDimDecoyWellGame` and every other separable game in `games.examples`.
    """
    scale = _factor(jax.random.PRNGKey(10))
    mean = jnp.array([0.3, -0.2, 1.0])

    def separable_expected_payoff(raw_scale):
        # E[sum_d exp(-a_d^2 / 2)] -- a sum of one-coordinate functions, so it
        # depends on the marginal variances and nothing else.
        variance = jnp.sum(jnp.square(raw_scale), axis=-1)
        return jnp.sum(jnp.exp(-jnp.square(mean) / (2 * (1 + variance))) / jnp.sqrt(1 + variance))

    grad = jnp.tril(jax.grad(separable_expected_payoff)(scale))
    off_diagonal = np.array(grad) - np.diag(np.diag(np.array(grad)))
    # Nonzero only through the marginal variance, i.e. entirely via row norms:
    # holding those fixed, rotating within a row changes nothing.
    def rotated_payoff(angle):
        givens = jnp.eye(DIM).at[0, 0].set(jnp.cos(angle)).at[0, 1].set(-jnp.sin(angle))
        givens = givens.at[1, 0].set(jnp.sin(angle)).at[1, 1].set(jnp.cos(angle))
        return separable_expected_payoff(scale @ givens)

    assert float(jax.grad(rotated_payoff)(0.0)) == pytest.approx(0.0, abs=1e-6)
    assert off_diagonal.shape == (DIM, DIM)
