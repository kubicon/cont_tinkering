"""Actor-critic network and the diagonal Gaussian policy built on top of it.

Reuses `nets.MLP` for both heads so the same activation/normalization
toggles from `nets` apply here. The policy's action distribution is a
diagonal Gaussian over the *unbounded* action produced by the actor head;
`ActionSpace.clip` is applied only when the action is actually played in the
game (the standard "clip at execution time, train on the unclipped
Gaussian" convention used by most continuous-control PPO implementations).
"""

from __future__ import annotations

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from nets import MLP


class ActorCritic(nn.Module):
    """Shared-input, separate-head actor-critic: Gaussian policy + state-value baseline."""

    action_dim: int
    hidden_dims: tuple[int, ...]
    activation: str = "tanh"
    normalization: str = "none"

    @nn.compact
    def __call__(self, obs: chex.Array, train: bool = False) -> tuple[chex.Array, chex.Array, chex.Array]:
        mean = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.action_dim,
            activation=self.activation,
            normalization=self.normalization,
            output_activation="identity",
            name="actor",
        )(obs, train=train)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        value = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            normalization=self.normalization,
            output_activation="identity",
            name="critic",
        )(obs, train=train)
        return mean, log_std, jnp.squeeze(value, axis=-1)


def gaussian_log_prob(action: chex.Array, mean: chex.Array, log_std: chex.Array) -> chex.Array:
    """Log-density of a diagonal Gaussian, summed over the action dimensions."""
    var = jnp.exp(2 * log_std)
    per_dim = -0.5 * (jnp.square(action - mean) / var + 2 * log_std + jnp.log(2 * jnp.pi))
    return jnp.sum(per_dim, axis=-1)


def gaussian_entropy(log_std: chex.Array) -> chex.Array:
    """Entropy of a diagonal Gaussian, summed over the action dimensions."""
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


def gaussian_kl(mean_p: chex.Array, log_std_p: chex.Array, mean_q: chex.Array, log_std_q: chex.Array) -> chex.Array:
    """`KL(N_p || N_q)` between two diagonal Gaussians, summed over the action dimensions.

    Unlike a mixture of Gaussians (generally intractable), a single diagonal
    Gaussian has a closed form: `log(std_q/std_p) + (var_p + (mean_p -
    mean_q)^2) / (2 * var_q) - 1/2` per dimension.
    """
    var_p = jnp.exp(2 * log_std_p)
    var_q = jnp.exp(2 * log_std_q)
    per_dim = (log_std_q - log_std_p) + (var_p + jnp.square(mean_p - mean_q)) / (2 * var_q) - 0.5
    return jnp.sum(per_dim, axis=-1)


def categorical_kl(logits_p: chex.Array, logits_q: chex.Array) -> chex.Array:
    """`KL(Cat_p || Cat_q)`, exact -- both distributions' full probabilities are available from logits."""
    log_p = jax.nn.log_softmax(logits_p)
    log_q = jax.nn.log_softmax(logits_q)
    return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)
