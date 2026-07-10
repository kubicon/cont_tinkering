"""Continuous action spaces for players in a game.

Every space exposes the same small interface (`shape`, `sample`, `clip`,
`contains`), so `ZeroSumGame` and its solvers never need to know which kind
of space a player acts in. `clip` in particular is what lets a generic
projected-gradient best response work for any space: it projects an
unconstrained gradient step back onto the feasible set.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chex
import jax
import jax.numpy as jnp


@runtime_checkable
class ActionSpace(Protocol):
    """Structural interface every action space must satisfy."""

    shape: tuple[int, ...]

    def sample(self, key: chex.PRNGKey, batch_shape: tuple[int, ...] = ()) -> chex.Array:
        """Draw action(s) with leading shape `batch_shape` uniformly (or otherwise) from the space."""
        ...

    def clip(self, action: chex.Array) -> chex.Array:
        """Project `action` (last axes matching `shape`) back onto the space."""
        ...

    def contains(self, action: chex.Array) -> chex.Array:
        """Boolean array (batch shape only) indicating membership."""
        ...


@chex.dataclass(frozen=True)
class BoxSpace:
    """Axis-aligned box: low <= a <= high, elementwise."""

    low: chex.Array
    high: chex.Array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    def sample(self, key: chex.PRNGKey, batch_shape: tuple[int, ...] = ()) -> chex.Array:
        shape = batch_shape + self.shape
        return jax.random.uniform(key, shape, minval=self.low, maxval=self.high)

    def clip(self, action: chex.Array) -> chex.Array:
        return jnp.clip(action, self.low, self.high)

    def contains(self, action: chex.Array) -> chex.Array:
        return jnp.all((action >= self.low) & (action <= self.high), axis=-1)


def box(low, high) -> BoxSpace:
    low = jnp.asarray(low, dtype=jnp.float32)
    high = jnp.asarray(high, dtype=jnp.float32)
    chex.assert_equal_shape([low, high])
    return BoxSpace(low=low, high=high)


def _project_to_simplex(x: chex.Array, total: chex.Array) -> chex.Array:
    """Euclidean projection of a single vector `x` onto {a >= 0, sum(a) == total}."""
    n = x.shape[-1]
    u = jnp.sort(x)[::-1]
    css = jnp.cumsum(u) - total
    ks = jnp.arange(1, n + 1, dtype=x.dtype)
    cond = u - css / ks > 0
    k = jnp.sum(cond)
    tau = css[k - 1] / k
    return jnp.maximum(x - tau, 0.0)


@chex.dataclass(frozen=True)
class SimplexSpace:
    """Non-negative vectors summing to `total` (a scaled probability simplex).

    Useful for resource-allocation games (e.g. Colonel Blotto) where an
    action is "how much of a fixed budget to put on each front".
    """

    dim: int
    total: chex.Array

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dim,)

    def sample(self, key: chex.PRNGKey, batch_shape: tuple[int, ...] = ()) -> chex.Array:
        shape = batch_shape + self.shape
        raw = jax.random.dirichlet(key, alpha=jnp.ones(self.dim), shape=batch_shape)
        return raw.reshape(shape) * self.total

    def clip(self, action: chex.Array) -> chex.Array:
        flat = action.reshape((-1, self.dim))
        projected = jax.vmap(lambda a: _project_to_simplex(a, self.total))(flat)
        return projected.reshape(action.shape)

    def contains(self, action: chex.Array) -> chex.Array:
        non_negative = jnp.all(action >= -1e-6, axis=-1)
        sums_to_total = jnp.isclose(jnp.sum(action, axis=-1), self.total, atol=1e-4)
        return non_negative & sums_to_total


def simplex(dim: int, total: float = 1.0) -> SimplexSpace:
    return SimplexSpace(dim=dim, total=jnp.asarray(total, dtype=jnp.float32))
