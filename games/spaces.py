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


# Stand-in for `-inf` when masking an illegal action out of a set of logits.
# Deliberately finite: `softmax` underflows it to an exact 0.0, so the `p * log p`
# in an entropy or a KL is `0.0 * finite == 0.0` rather than `0.0 * -inf == nan`.
MASKED_LOGIT = -1e9


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


@chex.dataclass(frozen=True)
class HybridAction:
    """One action from a `HybridSpace`: a discrete kind, plus a continuous value.

    `kind` in `[0, num_atoms]`. The first `num_atoms` values name a *discrete*
    action carrying no parameter (fold, call, check); `kind == num_atoms` is the
    continuous branch, and only then is `value` read. `value` is still carried
    (as whatever the policy proposed) when an atom was chosen, so the pytree has
    one fixed shape -- a game must simply not look at it.
    """

    kind: chex.Array  # () int32 in [0, num_atoms]
    value: chex.Array  # (dim,) float32


@chex.dataclass(frozen=True)
class HybridSpace:
    """`num_atoms` parameterless discrete actions alongside one continuous `box`.

    This is what makes a genuinely mixed discrete/continuous decision
    representable. Encoding "fold" as a region of a continuous action (say
    `a <= 0`) instead has two costs a policy cannot work around: the
    probability of folding becomes an *integral* over the Gaussian mixture
    rather than a number the policy states directly, and the gradient that is
    supposed to teach it when to fold is the gradient of a payoff that is
    piecewise constant across that whole region -- i.e. noise. An atom gets a
    categorical log-prob, so it gets a categorical policy gradient.

    A `MixtureActorCritic` realizes this by widening its categorical head to
    `num_atoms + num_components` and letting the first `num_atoms` entries be
    point masses; see `training.mixture`.
    """

    num_atoms: int
    box: BoxSpace

    @property
    def num_kinds(self) -> int:
        """`num_atoms + 1` -- every atom, plus the continuous branch."""
        return self.num_atoms + 1

    @property
    def continuous_kind(self) -> int:
        """The `kind` value meaning "a continuous action, read `value`"."""
        return self.num_atoms

    @property
    def shape(self) -> tuple[int, ...]:
        return self.box.shape

    @property
    def low(self) -> chex.Array:
        return self.box.low

    @property
    def high(self) -> chex.Array:
        return self.box.high

    def sample(self, key: chex.PRNGKey, mask: chex.Array | None = None) -> HybridAction:
        """A uniform action: a uniform *legal* kind, and a uniform point of `box`.

        `mask` is a `(num_kinds,)` boolean of which kinds are legal here (see
        `SequentialZeroSumGame.action_mask`); `None` means all of them are.
        """
        kind_key, value_key = jax.random.split(key)
        logits = jnp.zeros(self.num_kinds)
        if mask is not None:
            logits = jnp.where(mask, logits, MASKED_LOGIT)
        return HybridAction(
            kind=jax.random.categorical(kind_key, logits).astype(jnp.int32),
            value=self.box.sample(value_key),
        )

    def clip(self, action: HybridAction) -> HybridAction:
        return HybridAction(
            kind=jnp.clip(action.kind, 0, self.num_atoms),
            value=self.box.clip(action.value),
        )

    def contains(self, action: HybridAction) -> chex.Array:
        in_range = (action.kind >= 0) & (action.kind <= self.num_atoms)
        return in_range & (self.box.contains(action.value) | (action.kind < self.num_atoms))


def hybrid(num_atoms: int, low, high) -> HybridSpace:
    if num_atoms < 0:
        raise ValueError(f"num_atoms must be non-negative, got {num_atoms}")
    return HybridSpace(num_atoms=num_atoms, box=box(low, high))
