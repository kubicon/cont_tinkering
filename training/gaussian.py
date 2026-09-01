"""Multivariate Gaussian policies parametrized by a Cholesky factor.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np

# Floor on the diagonal of the scale factor. Bounds `det Sigma >= SIGMA_MIN^(2d)`
# and is what keeps the KL regularizer's Lipschitz constant finite; note it does
# *not* lower bound the smallest eigenvalue of `Sigma`, which a triangular factor
# with large off-diagonal entries can still drive towards zero. Nothing here needs
# it to.
SIGMA_MIN = 1e-3

# `SIGMA_MIN` in the coordinate `scale_parameterization: "log"` works in.
LOG_SIGMA_MIN = float(np.log(SIGMA_MIN))


def scale_param_size(action_dim: int, full_covariance: bool) -> int:
    """Number of free scale parameters per Gaussian component."""
    return action_dim * (action_dim + 1) // 2 if full_covariance else action_dim


def tril_positions(action_dim: int, full_covariance: bool):
    """`(rows, cols)` of the free scale entries, in the packed vector's order.

    Row-major over the lower triangle -- `(0,0), (1,0), (1,1), (2,0), ...` --
    matching `numpy.tril_indices`. Static (plain numpy): `action_dim` is a
    module attribute, never traced.
    """
    if full_covariance:
        return np.tril_indices(action_dim)
    idx = np.arange(action_dim)
    return idx, idx


def diagonal_slots(action_dim: int, full_covariance: bool) -> np.ndarray:
    """Indices *within the packed vector* that land on the factor's diagonal.

    What a bias initializer needs: the diagonal entries carry the initial
    standard deviation, every off-diagonal entry starts at zero (an
    uncorrelated policy).
    """
    rows, cols = tril_positions(action_dim, full_covariance)
    return np.flatnonzero(rows == cols)


def pack_scale_tril(flat: chex.Array, action_dim: int, full_covariance: bool) -> chex.Array:
    """`(..., scale_param_size)` of free entries -> `(..., d, d)` lower triangular.

    Everything above the diagonal is a structural zero rather than a parameter
    that happens to be zero: it is never allocated, so it can never be trained
    into a factor that is not lower triangular.
    """
    rows, cols = tril_positions(action_dim, full_covariance)
    empty = jnp.zeros(flat.shape[:-1] + (action_dim, action_dim), dtype=flat.dtype)
    return empty.at[..., rows, cols].set(flat)


def scale_diagonal(scale_tril: chex.Array) -> chex.Array:
    """`diag(A)` -- the per-dimension conditional standard deviations."""
    return jnp.diagonal(scale_tril, axis1=-2, axis2=-1)


def marginal_std(scale_tril: chex.Array) -> chex.Array:
    """`sqrt(diag(Sigma))` -- the per-dimension *marginal* standard deviations.

    Row norms of `A`, since `Sigma_ii = sum_j A_ij^2`. Distinct from
    `scale_diagonal` as soon as there are correlations: `A_ii` is the standard
    deviation of coordinate `i` *given* the coordinates before it, and is the
    smaller of the two. Any consumer reasoning about one coordinate on its own
    (a per-axis CDF, a printed policy summary) wants this one.
    """
    return jnp.sqrt(jnp.sum(jnp.square(scale_tril), axis=-1))


def straight_through_clip(x: chex.Array, lo: chex.Array, hi: chex.Array) -> chex.Array:
    """`clip(x, lo, hi)` by value, the identity by gradient.

    Value is exactly `clip(x, lo, hi)` (the second term is a bit-exact zero) and
    `d/dx` is exactly `1`, in range and out of it alike -- a plain `jnp.clip`
    would zero the gradient at the boundary, so a parameter that once saturated
    could never come back.

    The price is that the raw parameter is free to drift arbitrarily far outside
    `[lo, hi]` while the value stays pinned, and recovery then costs as many
    steps as the drift. That is a real hazard in `sigma` coordinates, where the
    loss offers no restoring force once the value is pinned; it is a much
    smaller one in `log sigma`, where the entropy bonus and both KL log-det
    terms each contribute a *constant* upward force on the raw parameter (their
    derivative w.r.t. `log sigma` does not vanish with `sigma`), so a saturated
    floor is actively pushed back into range.
    """
    clipped = jnp.clip(x, lo, hi)
    return jax.lax.stop_gradient(clipped) + (x - jax.lax.stop_gradient(x))


def clamp_scale_tril(
    scale_tril: chex.Array, sigma_min: chex.Array, sigma_max: chex.Array
) -> chex.Array:
    """Clamp `diag(A)` to `[sigma_min, sigma_max]`, straight-through.

    """
    adjusted = straight_through_clip(scale_diagonal(scale_tril), sigma_min, sigma_max)
    eye = jnp.eye(scale_tril.shape[-1], dtype=scale_tril.dtype)
    # Replace the diagonal rather than add a correction to it: adding would lose
    # precision whenever the raw diagonal sits far below the floor.
    return scale_tril * (1.0 - eye) + adjusted[..., :, None] * eye


def scale_tril_from_log_diag(
    flat: chex.Array,
    action_dim: int,
    full_covariance: bool,
    log_sigma_min: chex.Array,
    log_sigma_max: chex.Array,
    max_correlation: float = 0.0,
) -> chex.Array:
    """`pack_scale_tril`'s counterpart for `scale_parameterization: "log"`.

    Reads the packed head output as `(d, S)` rather than as the factor entries
    themselves, and builds

        A = diag(exp(d)) (I + S),    S strictly lower triangular,

    so that `A_ii = exp(d_i)` -- the same *conditional* standard deviation the
    linear parametrization puts on the diagonal, but positive by construction
    rather than by projection. Two things follow that the linear factor cannot
    express:

    * `log det A = sum_i d_i` exactly. Every log-determinant term in the loss
      (the Gaussian entropy bonus, both KL regularizers) therefore depends on
      `d` alone, with no cross-talk from the correlation structure: the spread
      and the correlations receive separate gradients.
    * `S` is dimensionless -- a correlation-like quantity, not one in action
      units -- so a single learning rate is coherent across the diagonal and the
      off-diagonal, which it is not when both are entries of `A`.

    `max_correlation > 0` squashes `S` through `c tanh(S/c)`, which bounds the
    condition number of `Sigma`. Worth knowing that flooring `diag(A)` alone
    does *not* bound the smallest eigenvalue of `Sigma` (see `SIGMA_MIN`); this
    is the knob that does. `0.0` (the default) leaves `S` unbounded.

    `log_sigma_max` bounds the conditional standard deviation, not the marginal
    one: with correlations present, row `i` of `A` has norm `exp(d_i) ||I + S||`
    along that row, which is the larger of the two.
    """
    packed = pack_scale_tril(flat, action_dim, full_covariance)
    eye = jnp.eye(action_dim, dtype=packed.dtype)
    log_diag = straight_through_clip(scale_diagonal(packed), log_sigma_min, log_sigma_max)
    # The upper triangle is already a structural zero out of `pack_scale_tril`,
    # so masking the diagonal is all it takes to be left with `S`.
    strict_lower = packed * (1.0 - eye)
    if full_covariance and max_correlation > 0.0:
        strict_lower = max_correlation * jnp.tanh(strict_lower / max_correlation)
    return jnp.exp(log_diag)[..., :, None] * (eye + strict_lower)


def _solve_lower(scale_tril: chex.Array, rhs: chex.Array) -> chex.Array:
    """`A^-1 rhs` for a lower-triangular `A`, batched over leading axes.

    `jax.lax.linalg.triangular_solve` broadcasts over leading batch axes on its
    own, so this works both per-sample and on a whole batch without a `vmap`;
    it also reads only the lower triangle, which is what makes the gradient
    w.r.t. the structural zeros above the diagonal exactly zero.
    """
    return jax.lax.linalg.triangular_solve(scale_tril, rhs, left_side=True, lower=True)


def _solve_vec(scale_tril: chex.Array, vec: chex.Array) -> chex.Array:
    return _solve_lower(scale_tril, vec[..., None])[..., 0]


def log_scale_det(scale_tril: chex.Array) -> chex.Array:
    """`sum_i log A_ii == 1/2 log det Sigma`."""
    return jnp.sum(jnp.log(scale_diagonal(scale_tril)), axis=-1)


def gaussian_log_prob(action: chex.Array, mean: chex.Array, scale_tril: chex.Array) -> chex.Array:
    """`log N(action | mean, A A^T)`, summed over the action dimensions.

    `-1/2 ||A^-1 (x - mu)||^2 - sum_i log A_ii - d/2 log(2 pi)`: one triangular
    solve, no covariance ever formed.
    """
    action_dim = scale_tril.shape[-1]
    z = _solve_vec(scale_tril, action - mean)
    return (
        -0.5 * jnp.sum(jnp.square(z), axis=-1)
        - log_scale_det(scale_tril)
        - 0.5 * action_dim * jnp.log(2.0 * jnp.pi)
    )


def gaussian_entropy(scale_tril: chex.Array) -> chex.Array:
    """Differential entropy `sum_i log A_ii + d/2 log(2 pi e)`."""
    action_dim = scale_tril.shape[-1]
    return log_scale_det(scale_tril) + 0.5 * action_dim * jnp.log(2.0 * jnp.pi * jnp.e)


def gaussian_kl(
    mean_p: chex.Array, scale_p: chex.Array, mean_q: chex.Array, scale_q: chex.Array
) -> chex.Array:
    """`KL(N_p || N_q)` between two multivariate Gaussians, exact.

    Unlike a *mixture* of Gaussians (generally intractable), a single Gaussian
    has a closed form; in factor coordinates it is

        1/2 [ ||A_q^-1 A_p||_F^2 + ||A_q^-1 (mu_p - mu_q)||^2 - d ]
            + sum_i log A_q,ii - sum_i log A_p,ii,

    i.e. `1/2 tr(Sigma_q^-1 Sigma_p)` and the Mahalanobis term each become one
    triangular solve. With both factors diagonal this is the usual per-dimension
    `log(s_q/s_p) + (s_p^2 + (mu_p - mu_q)^2) / (2 s_q^2) - 1/2`, summed.
    """
    action_dim = scale_p.shape[-1]
    ratio = _solve_lower(scale_q, scale_p)  # A_q^-1 A_p
    mahalanobis = _solve_vec(scale_q, mean_p - mean_q)
    trace_term = jnp.sum(jnp.square(ratio), axis=(-2, -1))
    quad_term = jnp.sum(jnp.square(mahalanobis), axis=-1)
    return (
        0.5 * (trace_term + quad_term - action_dim)
        + log_scale_det(scale_q)
        - log_scale_det(scale_p)
    )


def gaussian_sample(mean: chex.Array, scale_tril: chex.Array, noise: chex.Array) -> chex.Array:
    """`mu + A xi` -- the reparametrized draw, `xi` standard normal.

    The one place correlations enter the sampler: with `A` diagonal this is the
    elementwise `mu + sigma * xi`, and with a full factor it is the triangular
    matvec that correlates the coordinates.
    """
    return mean + jnp.einsum("...ij,...j->...i", scale_tril, noise)


def natural_gradient(
    scale_tril: chex.Array, grad_mean: chex.Array, grad_scale: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """`F^-1 grad` in `(mu, A)` coordinates, `F` the Gaussian Fisher metric.

    Second-order expansion of the KL between two nearby Gaussians in these
    coordinates gives, with `E = A^-1 dA`,

        2 KL = d mu^T Sigma^-1 d mu + 1/2 tr((E + E^T)^2),

    and maximizing `<G, dA>` against it yields `E = phi(A^T G)`, where `phi`
    keeps the lower triangle and halves the diagonal. Hence

        natural d mu = Sigma grad_mu = A (A^T grad_mu),
        natural dA   = A phi(A^T grad_A).

    With `A = diag(sigma)` this is `sigma^2 grad_mu` and `1/2 sigma^2 grad_sigma`
    -- exactly the preconditioner the diagonal MMD updates already use, written
    in `sigma` rather than `log sigma` coordinates (`d sigma = sigma d log
    sigma` turns the latter's `1/2 grad_logsigma` into the former's
    `1/2 sigma^2 grad_sigma`). Off the diagonal it is the correlation-aware
    generalization.

    `grad_scale` is lower-triangularized on the way in: the upper triangle holds
    structural zeros, not parameters, and must not steer the step.
    """
    grad_scale = jnp.tril(grad_scale)
    swapped = jnp.swapaxes(scale_tril, -2, -1)  # A^T
    nat_mean = jnp.einsum("...ij,...j->...i", scale_tril, jnp.einsum("...ij,...j->...i", swapped, grad_mean))
    inner = jnp.einsum("...ij,...jk->...ik", swapped, grad_scale)  # A^T G
    eye = jnp.eye(scale_tril.shape[-1], dtype=scale_tril.dtype)
    phi = jnp.tril(inner) - 0.5 * scale_diagonal(inner)[..., :, None] * eye
    nat_scale = jnp.einsum("...ij,...jk->...ik", scale_tril, phi)
    return nat_mean, nat_scale
