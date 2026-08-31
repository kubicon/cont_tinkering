"""Actor-critic network and the Gaussian policy built on top of it.

Reuses `nets.MLP` for both heads so the same activation/normalization
toggles from `nets` apply here. The policy's action distribution is a
Gaussian over the *unbounded* action produced by the actor head;
`ActionSpace.clip` is applied only when the action is actually played in the
game (the standard "clip at execution time, train on the unclipped
Gaussian" convention used by most continuous-control PPO implementations).

The scale is a Cholesky factor of the covariance, floored at
`gaussian.SIGMA_MIN` -- diagonal by default, full with `full_covariance`. See
`training.gaussian` for why the factor (rather than the variance, or a
`log`-parametrized standard deviation) is the parametrization that keeps the
KL regularizer uniformly strongly convex, and for the distribution formulas
themselves.
"""

from __future__ import annotations

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from games.spaces import MASKED_LOGIT
from nets import MLP

from .gaussian import (
    SIGMA_MIN,
    clamp_scale_tril,
    diagonal_slots,
    gaussian_entropy,
    gaussian_kl,
    gaussian_log_prob,
    gaussian_sample,
    pack_scale_tril,
    scale_param_size,
)

__all__ = [
    "ActorCritic",
    "categorical_kl",
    "gaussian_entropy",
    "gaussian_kl",
    "gaussian_log_prob",
    "gaussian_sample",
    "masked_categorical_entropy",
    "masked_log_softmax",
    "scale_init",
]


def scale_init(action_dim: int, full_covariance: bool):
    """Initializer for a packed scale vector: unit diagonal, zero off-diagonal.

    An uncorrelated unit-variance policy, i.e. what a `zeros` initializer used
    to mean back when the head emitted `log_std`. It cannot mean that any more:
    the head now emits the factor entries themselves, where zero is the
    degenerate scale rather than the unit one.
    """
    slots = diagonal_slots(action_dim, full_covariance)
    size = scale_param_size(action_dim, full_covariance)

    def init_fn(key: chex.PRNGKey, shape: tuple[int, ...], dtype=jnp.float32) -> chex.Array:
        del key
        row = jnp.zeros((size,), dtype=dtype).at[slots].set(1.0)
        return jnp.broadcast_to(row, shape).astype(dtype)

    return init_fn


class ActorCritic(nn.Module):
    """Shared-input, separate-head actor-critic: Gaussian policy + state-value baseline."""

    action_dim: int
    hidden_dims: tuple[int, ...]
    activation: str = "tanh"
    normalization: str = "none"
    # Correlate the action coordinates. Off by default: every separable payoff
    # (see `training.gaussian`) has an expected value depending only on the
    # per-axis marginals, so the off-diagonal entries would receive gradient
    # from the KL term alone.
    full_covariance: bool = False

    @nn.compact
    def __call__(self, obs: chex.Array, train: bool = False) -> tuple[chex.Array, chex.Array, chex.Array]:
        """`(mean, scale_tril, value)` -- `scale_tril` is `(action_dim, action_dim)`.

        The scale is state-independent (a bare parameter, not a head off the
        torso), as it was when it was a `log_std` vector.
        """
        mean = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.action_dim,
            activation=self.activation,
            normalization=self.normalization,
            output_activation="identity",
            name="actor",
        )(obs, train=train)
        scale_flat = self.param(
            "scale",
            scale_init(self.action_dim, self.full_covariance),
            (scale_param_size(self.action_dim, self.full_covariance),),
        )
        scale_tril = clamp_scale_tril(
            pack_scale_tril(scale_flat, self.action_dim, self.full_covariance),
            SIGMA_MIN,
            jnp.inf,
        )
        value = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            normalization=self.normalization,
            output_activation="identity",
            name="critic",
        )(obs, train=train)
        return mean, scale_tril, jnp.squeeze(value, axis=-1)


def masked_log_softmax(logits: chex.Array, mask: chex.Array) -> chex.Array:
    """`log_softmax` over the entries `mask` marks legal, the rest driven to ~`-inf`.

    Illegal entries are set to `MASKED_LOGIT` (a large *finite* negative) rather
    than `-inf`: their probability then underflows to exactly `0.0`, so the
    `p * log p` in an entropy or a KL is `0.0 * finite == 0.0` instead of the
    `0.0 * -inf == nan` that `-inf` would produce. `mask` must have at least one
    `True` entry, or the result is a uniform distribution over nothing.
    """
    return jax.nn.log_softmax(jnp.where(mask, logits, MASKED_LOGIT))


def masked_categorical_entropy(logits: chex.Array, mask: chex.Array) -> chex.Array:
    """Entropy of `masked_log_softmax(logits, mask)`, over the legal entries only."""
    log_p = masked_log_softmax(logits, mask)
    return -jnp.sum(jnp.where(mask, jnp.exp(log_p) * log_p, 0.0), axis=-1)


def categorical_kl(logits_p: chex.Array, logits_q: chex.Array, mask: chex.Array) -> chex.Array:
    """`KL(Cat_p || Cat_q)` over the entries `mask` marks legal, exact.

    Both distributions are conditioned on the *same* mask -- they are two
    policies evaluated at one state, and legality is a property of the state,
    not of the policy -- so the illegal entries cancel and contribute nothing.
    """
    log_p = masked_log_softmax(logits_p, mask)
    log_q = masked_log_softmax(logits_q, mask)
    return jnp.sum(jnp.where(mask, jnp.exp(log_p) * (log_p - log_q), 0.0), axis=-1)
