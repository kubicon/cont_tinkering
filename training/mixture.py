"""A two-phase (discrete-then-continuous) actor-critic.
"""

from __future__ import annotations

import math
from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from games.base import ZeroSumGame
from games.spaces import MASKED_LOGIT
from nets import Activation, Normalization

from .actor_critic import (
    categorical_kl,
    gaussian_kl,
    gaussian_log_prob,
    gaussian_sample,
    masked_categorical_entropy,
    masked_log_softmax,
)
from .config import MixturePPOHyperparams
from .gaussian import (
    LOG_SIGMA_MIN,
    SIGMA_MIN,
    clamp_scale_tril,
    diagonal_slots,
    pack_scale_tril,
    scale_param_size,
    scale_tril_from_log_diag,
)
from .vtrace import vtrace

SCALE_PARAMETERIZATIONS = ("linear", "log")

OpponentActionFn = Callable[[chex.PRNGKey, int], chex.Array]

MIN_ACTION_WIDTH = SIGMA_MIN


def _action_width(low: chex.Array, high: chex.Array) -> chex.Array:
    """`high - low`, floored so a degenerate (zero-width) box stays representable."""
    return jnp.maximum(high - low, MIN_ACTION_WIDTH)


def _spread_bias_init(low: chex.Array, high: chex.Array, num_components: int) -> Callable:
    """Bias initializer spreading each component's mean evenly across `[low, high]`.
    """

    def init_fn(key: chex.PRNGKey, shape: tuple[int, ...], dtype=jnp.float32) -> chex.Array:
        del key
        fractions = (jnp.arange(num_components, dtype=dtype) + 0.5) / num_components
        values = low[None, :] + fractions[:, None] * (high - low)[None, :]
        return values.reshape(shape).astype(dtype)

    return init_fn


def bucket_bounds(
    low: chex.Array, high: chex.Array, num_components: int
) -> tuple[chex.Array, chex.Array]:
    """`(lows, highs)`, each `(num_components, d)`: the box cut into one bucket per component.

    The buckets are a regular grid, `n` equal slices per axis with `n ** d ==
    num_components`, components in row-major order over it -- so for the usual
    one-dimensional bet size, component `k` owns the `k`-th of `num_components`
    equal intervals of `[low, high]`, exactly the cells `_spread_bias_init`
    centers the unbucketed means in.
    """
    action_dim = low.shape[0]
    per_axis = round(num_components ** (1.0 / action_dim))
    if per_axis ** action_dim != num_components:
        raise ValueError(
            f"bucket_means needs num_components to be a perfect {action_dim}-th power "
            f"(one grid of buckets over the {action_dim}-d box), got {num_components}"
        )
    cells = np.stack(
        np.unravel_index(np.arange(num_components), (per_axis,) * action_dim), axis=-1
    )  # (num_components, d) grid index of each component's bucket
    width = (high - low) / per_axis
    lows = low[None, :] + cells * width[None, :]
    return lows, lows + width[None, :]


def component_boxes(network: "MixtureActorCritic") -> tuple[chex.Array, chex.Array]:
    """`(lows, highs)`, each `(num_components, d)`: where each component's mean lives.

    Its bucket under `bucket_means`, the whole action box for every component
    otherwise. Also what exploration draws from: see `sample_mixture_component`.
    """
    if network.bucket_means:
        return bucket_bounds(network.low, network.high, network.num_components)
    shape = (network.num_components, network.action_dim)
    return jnp.broadcast_to(network.low, shape), jnp.broadcast_to(network.high, shape)


def _sigma_bounds(
    low: chex.Array, high: chex.Array, sigma_min: float | None, sigma_max: float | None
) -> tuple[float, chex.Array | float]:
    """`(lo, hi)` bounds on a component's standard deviation, in sigma units.

    `None` falls back to the built-in bounds -- `SIGMA_MIN` below, the box width
    above -- and the default ceiling is raised to meet a custom floor, so a
    floor set above a narrow box's width is honored rather than inverted.
    """
    lo = SIGMA_MIN if sigma_min is None else sigma_min
    hi = jnp.maximum(_action_width(low, high), lo) if sigma_max is None else sigma_max
    return lo, hi


def _scale_bias_init(
    low: chex.Array,
    high: chex.Array,
    num_components: int,
    full_covariance: bool,
    scale_parameterization: str = "linear",
    sigma_min: float | None = None,
    sigma_max: float | None = None,
) -> Callable:
    """Bias initializer for `scale_head`, scaled to the action range.

    The same initial policy either way -- each component's conditional standard
    deviation is one `num_components`-th of the box's half width, uncorrelated --
    written in whichever coordinate the head emits. Under `"log"` the diagonal
    carries `log std` and the off-diagonal stays at zero, which is exactly the
    `S = 0` that makes `I + S` the identity. That std is clipped into
    `[sigma_min, sigma_max]`, so the raw head starts inside its own clamp.
    """
    action_dim = low.shape[0]
    slots = diagonal_slots(action_dim, full_covariance)
    size = scale_param_size(action_dim, full_covariance)
    lo, hi = _sigma_bounds(low, high, sigma_min, sigma_max)

    def init_fn(key: chex.PRNGKey, shape: tuple[int, ...], dtype=jnp.float32) -> chex.Array:
        del key
        std = jnp.maximum(_action_width(low, high) / (2 * num_components), MIN_ACTION_WIDTH)
        std = jnp.clip(std, lo, hi)
        diag = jnp.log(std) if scale_parameterization == "log" else std
        row = jnp.zeros((size,), dtype=dtype).at[slots].set(diag.astype(dtype))
        values = jnp.broadcast_to(row[None, :], (num_components, size))
        return values.reshape(shape).astype(dtype)

    return init_fn


def _has_width(low: chex.Array, high: chex.Array) -> chex.Array:
    """Per action dimension: is there room inside the box for a mean to move?"""
    return (high - low) > 0.0


def project_means_to_box(means: chex.Array, low: chex.Array, high: chex.Array) -> chex.Array:
    """`clip(means, low, high)`, straight-through: the raw mean still gets a gradient.

    """
    projected = jnp.clip(means, low, high)
    return jnp.where(
        _has_width(low, high),
        means + jax.lax.stop_gradient(projected - means),
        jax.lax.stop_gradient(projected),
    )


def mean_box_excess(means: chex.Array, low: chex.Array, high: chex.Array) -> chex.Array:
    """Summed squared distance of `means` from `[low, high]`; `0.0` inside the box.
    """
    excess = jnp.maximum(means - high, 0.0) + jnp.maximum(low - means, 0.0)
    # Frozen dimensions (see `project_means_to_box`) have nothing to pull back.
    return jnp.sum(jnp.where(_has_width(low, high), jnp.square(excess), 0.0))


class MixtureActorCritic(nn.Module):
    """Shared torso, four linear heads: component logits, means, scales, value.
    """

    action_dim: int
    num_components: int
    hidden_dims: tuple[int, ...]
    low: chex.Array
    high: chex.Array
    activation: str = "tanh"
    normalization: str = "none"
    clip_means: bool = False
    num_atoms: int = 0
    # Correlate the action coordinates within a component; see `training.gaussian`.
    full_covariance: bool = False
    # "linear" or "log"; see `training.config.PPOHyperparams.scale_parameterization`.
    scale_parameterization: str = "linear"
    max_correlation: float = 0.0
    # Std bounds in sigma units under either parameterization; `None` keeps the
    # built-in ones (see `_sigma_bounds`).
    sigma_min: float | None = None
    sigma_max: float | None = None
    # Give each component a fixed bucket of the box (see `bucket_bounds`) and
    # keep its mean inside it: `mean = lo_k + (hi_k - lo_k) * sigmoid(raw)`,
    # starting at the bucket's center. Moving bet mass between regions of the
    # box is then a change of component weights -- which the categorical head
    # makes non-locally -- rather than a Gaussian mean crossing the box.
    bucket_means: bool = False

    @nn.compact
    def __call__(
        self, obs: chex.Array, train: bool = False, project_means: bool = True
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """`(logits, means, scale_trils, value)`; `scale_trils` is `(K, d, d)`.

        """
        torso = obs
        for dim in self.hidden_dims:
            torso = nn.Dense(dim)(torso)
            torso = Normalization(kind=self.normalization)(torso, use_running_average=not train)
            torso = Activation(kind=self.activation)(torso)

        logits = nn.Dense(self.num_atoms + self.num_components, name="logits_head")(torso)

        means_flat = nn.Dense(
            self.num_components * self.action_dim,
            kernel_init=nn.initializers.zeros,
            # Bucketed: a zero raw output is `sigmoid(0) = 1/2`, the bucket's center.
            bias_init=(
                nn.initializers.zeros if self.bucket_means
                else _spread_bias_init(self.low, self.high, self.num_components)
            ),
            name="means_head",
        )(torso)
        means = means_flat.reshape(self.num_components, self.action_dim)
        if self.bucket_means:
            # Inside the box by construction, so `clip_means`' projection below
            # is the identity and its box penalty is exactly zero.
            lows, highs = bucket_bounds(self.low, self.high, self.num_components)
            means = lows + (highs - lows) * jax.nn.sigmoid(means)
        if self.clip_means and project_means:
            means = project_means_to_box(means, self.low, self.high)

        scale_size = scale_param_size(self.action_dim, self.full_covariance)
        if self.scale_parameterization not in SCALE_PARAMETERIZATIONS:
            raise ValueError(
                f"scale_parameterization must be one of {SCALE_PARAMETERIZATIONS}, "
                f"got {self.scale_parameterization!r}"
            )
        scale_flat = nn.Dense(
            self.num_components * scale_size,
            kernel_init=nn.initializers.zeros,
            bias_init=_scale_bias_init(
                self.low, self.high, self.num_components, self.full_covariance,
                self.scale_parameterization, self.sigma_min, self.sigma_max,
            ),
            name="scale_head",
        )(torso)
        scale_raw = scale_flat.reshape(self.num_components, scale_size)
        # Floor and ceiling on the conditional standard deviations (`sigma_min`
        # / `sigma_max`, defaulting to `SIGMA_MIN` / the box width), given in
        # sigma units and applied in whichever coordinate the head emits.
        # Straight-through either way, so a saturated component keeps
        # receiving gradient.
        sigma_lo, sigma_hi = _sigma_bounds(self.low, self.high, self.sigma_min, self.sigma_max)
        if self.scale_parameterization == "log":
            scale_tril = scale_tril_from_log_diag(
                scale_raw,
                self.action_dim,
                self.full_covariance,
                LOG_SIGMA_MIN if self.sigma_min is None else math.log(sigma_lo),
                jnp.log(sigma_hi),
                self.max_correlation,
            )
        else:
            scale_tril = pack_scale_tril(scale_raw, self.action_dim, self.full_covariance)
            # The projection onto the feasible set of factors.
            scale_tril = clamp_scale_tril(scale_tril, sigma_lo, sigma_hi)

        value = nn.Dense(1, name="value_head")(torso)

        return logits, means, scale_tril, jnp.squeeze(value, axis=-1)


def expand_kind_mask(kind_mask: chex.Array, num_components: int) -> chex.Array:
    """A game's `(num_atoms + 1,)` kind mask, widened to the categorical head's logits.
    """
    num_atoms = kind_mask.shape[-1] - 1
    return jnp.concatenate(
        [kind_mask[..., :num_atoms], jnp.repeat(kind_mask[..., num_atoms:], num_components, axis=-1)],
        axis=-1,
    )


def component_to_kind(component: chex.Array, num_atoms: int) -> chex.Array:
    """The `HybridAction.kind` a sampled categorical index plays.
    """
    return jnp.minimum(component, num_atoms).astype(jnp.int32)


def gaussian_component_index(component: chex.Array, num_atoms: int) -> chex.Array:
    """Row of `means`/`scale_trils` that `component` refers to, clamped to be in range.
    """
    return jnp.maximum(component - num_atoms, 0)


def mixture_log_probs(
    logits: chex.Array,
    means: chex.Array,
    scale_trils: chex.Array,
    mask: chex.Array,
    component: chex.Array,
    raw_action: chex.Array,
    num_atoms: int,
) -> tuple[chex.Array, chex.Array]:
    """`(category_log_prob, gaussian_log_prob)` for one sample -- kept as separate factors.

    """
    category_log_prob = masked_log_softmax(logits, mask)[component]
    index = gaussian_component_index(component, num_atoms)
    log_prob = gaussian_log_prob(raw_action, means[index], scale_trils[index])
    return category_log_prob, jnp.where(component >= num_atoms, log_prob, 0.0)


def mixture_marginal_log_prob(
    logits: chex.Array,
    means: chex.Array,
    scale_trils: chex.Array,
    mask: chex.Array,
    raw_action: chex.Array,
    num_atoms: int,
) -> chex.Array:
    """`log p(a | a is continuous)` of the marginal mixture density at a single `raw_action`.
    """
    log_weights = masked_log_softmax(logits[num_atoms:], mask[num_atoms:])  # (num_components,)
    per_component = jax.vmap(gaussian_log_prob, in_axes=(None, 0, 0))(raw_action, means, scale_trils)
    return jax.nn.logsumexp(log_weights + per_component)


@chex.dataclass
class Episode:
    """Everything recorded while sampling transitions from a `MixtureActorCritic`.

    One dataclass covers both game shapes. A field's *trailing* axes are the
    per-decision ones documented below; whatever leading axes sit in front of
    them are batch axes, and there can be any number of them:

      * one-shot (`ZeroSumGame`): `(num_envs, ...)` -- one decision per episode,
        so the env axis is the only one.
      * sequential (`SequentialZeroSumGame`): `(num_envs, max_steps, ...)` --
        a trajectory is padded out to the static horizon, so most rows of a
        batch are *not* decisions of the player being trained.

    `actor` is what makes the second case work and costs the first case one
    `int32` per sample: it names who owned each decision, and is `TERMINAL`
    (`-1`) on the padding steps of a finished episode. Every reduction in the
    loss weights by `actor == player` (see `player_weight`), so padding and the
    opponent's interleaved decisions contribute exactly zero rather than
    approximately zero. In a one-shot batch every row is the same player's, the
    weight is all-ones, and every masked reduction collapses to a plain mean.
    """

    actor: chex.Array  # () int32, who made this decision -- `TERMINAL` on a padding step
    obs: chex.Array
    action_mask: chex.Array  # (num_atoms + num_components,) bool, which logits were legal here
    logits: chex.Array  # (num_atoms + num_components,) categorical logits at sample time
    means: chex.Array  # (num_components, action_dim) each Gaussian component's mean at sample time
    scale_trils: chex.Array  # (num_components, action_dim, action_dim) each component's Cholesky scale factor at sample time
    magnet_logits: chex.Array  # (num_atoms + num_components,) logits under `magnet_params`, same obs
    magnet_means: chex.Array  # (num_components, action_dim) under `magnet_params`
    magnet_scale_trils: chex.Array  # (num_components, action_dim, action_dim) under `magnet_params`
    component: chex.Array  # which categorical entry was sampled: an atom, or a Gaussian component
    raw_action: chex.Array  # unclipped Gaussian sample; meaningless (but finite) when an atom was drawn
    action_kind: chex.Array  # `HybridAction.kind` actually played -- `component_to_kind(component)`
    action_value: chex.Array  # `raw_action` clipped to the action space; read by the game only on the continuous kind
    value: chex.Array
    reward: chex.Array
    # `()` float per decision: the `explore_eps` its continuous action was drawn
    # under (see `sample_mixture_component`). `None` -- the default, and what
    # every non-exploring sampler leaves -- marks an on-policy rollout, which the
    # loss then scores exactly as before; set, it switches the loss to its
    # importance-weighted form (see `behavior_gaussian_log_prob`).
    behavior_eps: chex.Array | None = None
    # `()` float per decision, sequential rollouts only: player 0's reward for
    # this row's transition, the terminal payoff included on the episode's last
    # decision (see `training.sequential_rollout`). What `training.vtrace`
    # bootstraps from; `reward` is its Monte Carlo sum. `None` on one-shot batches.
    step_reward: chex.Array | None = None



def player_weight(episode: Episode, player: int) -> chex.Array:
    """`1.0` on the rows where `player` really decided something, `0.0` on everything else.

    The one weight every reduction in the loss goes through: it is what makes an
    opponent's step and a padding step contribute exactly nothing. All-ones for a
    one-shot batch, where every row is `player`'s.
    """
    return (episode.actor == player).astype(jnp.float32)


def masked_mean(values: chex.Array, weight: chex.Array) -> chex.Array:
    """`sum(weight * values) / sum(weight)`, safe when nothing is selected."""
    return jnp.sum(weight * values) / jnp.maximum(jnp.sum(weight), 1.0)


def normalized_advantage(raw: chex.Array, weight: chex.Array) -> chex.Array:
    """Standardize `raw` using only the entries `weight` selects.

    With an all-ones `weight` this is exactly `(raw - mean) / (std + 1e-8)`.
    """
    mean = masked_mean(raw, weight)
    variance = masked_mean(jnp.square(raw - mean), weight)
    return (raw - mean) / (jnp.sqrt(variance) + 1e-8)


def flatten_batch_axes(episode: Episode) -> Episode:
    """Collapse an `Episode`'s leading batch axes into one, leaving per-sample shapes alone.

    `actor` carries exactly the batch axes and nothing else, so its rank says how
    many there are. A one-shot batch already has one, and this is then a no-op; a
    trajectory batch has `(num_envs, max_steps)`, and flattening lets the loss
    `vmap` once over every decision rather than nesting a `vmap` per axis.
    """
    lead = episode.actor.ndim
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[lead:]), episode)



def sample_mixture_component(
    logits: chex.Array,
    means: chex.Array,
    scale_trils: chex.Array,
    mask: chex.Array,
    num_atoms: int,
    key: chex.PRNGKey,
    explore_eps: chex.Array | None = None,
    low: chex.Array | None = None,
    high: chex.Array | None = None,
) -> tuple[chex.Array, chex.Array]:
    """Draw one `(component, raw_action)` from a masked hybrid mixture policy.

    With `explore_eps`, the *behavior* policy is sampled instead: the component
    is drawn exactly as the policy says, and then, with probability
    `explore_eps`, the drawn Gaussian's sample is replaced by a uniform draw on
    the box `[low, high]`. So for a Gaussian component `k` the continuous
    action has density `(1 - eps) N_k(x) + eps U(x)` while every categorical
    probability -- check vs bet, which component -- is untouched. The loss
    corrects for it from `Episode.behavior_eps`; see `behavior_gaussian_log_prob`.
    `None` (the default) is the plain policy, with the same rng stream as before.

    `low`/`high` may also be `(num_components, d)` -- one box per component, as
    `component_boxes` gives -- and the uniform is then drawn on the drawn
    component's own box. Under `bucket_means` that keeps exploration local to
    the bucket: an explored size arrives with the hand composition the policy
    bets into *that* region, rather than the policy's overall betting mix.
    """
    if explore_eps is None:
        component_key, noise_key = jax.random.split(key)
    else:
        component_key, noise_key, coin_key, uniform_key = jax.random.split(key, 4)
    component = jax.random.categorical(component_key, jnp.where(mask, logits, MASKED_LOGIT))
    index = gaussian_component_index(component, num_atoms)
    mean = means[index]
    # `dtype=mean.dtype` rather than the default: under `jax_enable_x64` (which
    # `baselines.common` turns on process-wide) a bare `normal` draws float64, and
    # a policy whose observation is float32 -- Leduc's is, Kuhn's is not -- would
    # then produce a float64 action against float32 parameters, which the loss
    # rejects when it solves against `scale_tril`.
    raw_action = gaussian_sample(
        mean, scale_trils[index], jax.random.normal(noise_key, mean.shape, dtype=mean.dtype)
    )
    if explore_eps is not None:
        low, high = component_box(low, high, index)
        uniform = jax.random.uniform(
            uniform_key, mean.shape, dtype=mean.dtype,
            minval=low.astype(mean.dtype), maxval=high.astype(mean.dtype),
        )
        explore = jax.random.uniform(coin_key) < explore_eps
        raw_action = jnp.where(explore, uniform, raw_action)
    return component, raw_action


def component_box(
    low: chex.Array, high: chex.Array, index: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """The `(d,)` box of Gaussian component `index`: row `index` of a per-component
    `(num_components, d)` box, or a shared `(d,)` box as it is."""
    if low.ndim == 2:
        return low[index], high[index]
    return low, high


def behavior_gaussian_log_prob(
    gaussian_log_prob: chex.Array,
    raw_action: chex.Array,
    explore_eps: chex.Array,
    low: chex.Array,
    high: chex.Array,
) -> chex.Array:
    """`log((1 - eps) N_k(x) + eps U(x))`: the exploring sampler's density of `raw_action`.

    `gaussian_log_prob` is `log N_k(x)` under the sampling-time policy, for the
    component that was drawn. `U` is the uniform density on the box, zero
    outside it (a Gaussian draw can land there; a uniform one cannot). Since the
    categorical draw is unchanged, `N_k(x) / this` is the full importance
    ratio of the sample, and it is bounded by `1 / (1 - eps)`.
    """
    inside = jnp.all((raw_action >= low) & (raw_action <= high))
    log_uniform = jnp.where(inside, -jnp.sum(jnp.log(high - low)), -jnp.inf)
    return jnp.logaddexp(
        jnp.log1p(-explore_eps) + gaussian_log_prob, jnp.log(explore_eps) + log_uniform
    )


def hybrid_action_log_prob(
    logits: chex.Array,
    means: chex.Array,
    scale_trils: chex.Array,
    mask: chex.Array,
    component: chex.Array,
    raw_action: chex.Array,
    num_atoms: int,
    explore_eps: chex.Array | None = None,
    low: chex.Array | None = None,
    high: chex.Array | None = None,
) -> chex.Array:
    """`log p(a)` of the hybrid action played, marginalized over the Gaussian that drew it.

    The game only ever sees the action: an atom, or a continuous value. Which
    Gaussian component produced a continuous value is the sampler's internal
    coin, so the action's probability sums over it --
    `sum_k P(k) N_k(raw_action)` over the Gaussian entries -- and a ratio of
    two such densities is the lower-variance importance weight of the two
    (`mixture_log_probs` keeps the component-level factors the PPO heads use).

    With `explore_eps` this is the *exploring* behavior's density instead: the
    component is drawn as the policy says, then its sample is replaced by a
    uniform draw on `[low, high]` with probability `eps` (see
    `sample_mixture_component`), so the continuous density becomes
    `(1 - eps) sum_k P(k) N_k(x) + eps sum_k P(k) U_k(x)`, where `U_k` is the
    uniform on component `k`'s box -- one shared box (`(d,)`) or its own
    bucket (`(num_components, d)`, as `component_boxes` gives). With a shared
    box the second sum is `eps P(continuous) U(x)`. Atoms are never touched by
    exploration.
    """
    log_probs = masked_log_softmax(logits, mask)
    per_component = jax.vmap(gaussian_log_prob, in_axes=(None, 0, 0))(raw_action, means, scale_trils)
    continuous = jax.nn.logsumexp(log_probs[num_atoms:] + per_component)
    if explore_eps is not None:
        lows, highs = jnp.broadcast_to(low, means.shape), jnp.broadcast_to(high, means.shape)
        inside = jnp.all((raw_action >= lows) & (raw_action <= highs), axis=-1)
        log_uniform = jnp.where(inside, -jnp.sum(jnp.log(highs - lows), axis=-1), -jnp.inf)
        continuous = jnp.logaddexp(
            jnp.log1p(-explore_eps) + continuous,
            jnp.log(explore_eps) + jax.nn.logsumexp(log_probs[num_atoms:] + log_uniform),
        )
    return jnp.where(component >= num_atoms, continuous, log_probs[component])


def _sample_mixture_one(
    game: ZeroSumGame, player: int, network: MixtureActorCritic, params, magnet_params, key: chex.PRNGKey
) -> Episode:
    """One (unbatched) sample as an `Episode` with `reward` still zero -- the caller fills
    it in once it knows the opponent's action. Draws `obs`, samples a component/action,
    and evaluates both `params` and `magnet_params` at that `obs`. Everything is keyed off
    the single `key` passed in -- `vmap` this over `num_envs` keys to draw a whole batch
    (see `collect_mixture_episode`/`collect_mixture_self_play_episode`).
    """
    obs_key, sample_key = jax.random.split(key)
    space = game.action_space(player)
    obs = game.observation(player, obs_key)

    logits, means, scale_trils, value = network.apply(params, obs)
    magnet_logits, magnet_means, magnet_scale_trils, _ = network.apply(magnet_params, obs)

    # A one-shot `ZeroSumGame` has a single unconstrained continuous action: no
    # atoms, nothing illegal, so the mask is all-`True` and every masked term in
    # the loss reduces to its unmasked form.
    action_mask = jnp.ones_like(logits, dtype=bool)
    component, raw_action = sample_mixture_component(
        logits, means, scale_trils, action_mask, network.num_atoms, sample_key
    )

    return Episode(
        # Every row of a one-shot batch is the same player's single decision, so
        # `actor` is constant and `player_weight` comes out all-ones.
        actor=jnp.int32(player),
        obs=obs, action_mask=action_mask, logits=logits, means=means, scale_trils=scale_trils,
        magnet_logits=magnet_logits, magnet_means=magnet_means, magnet_scale_trils=magnet_scale_trils,
        component=component, raw_action=raw_action,
        action_kind=component_to_kind(component, network.num_atoms),
        action_value=space.clip(raw_action), value=value, reward=jnp.zeros(()),
    )


def sample_mixture_actions(
    network: MixtureActorCritic,
    params,
    obs: chex.Array,
    space,
    key: chex.PRNGKey,
    num_samples: int,
) -> chex.Array:
    """Draw `num_samples` clipped actions from the mixture policy at a single `obs`.
    """
    if network.num_atoms != 0:
        raise ValueError(
            "sample_mixture_actions returns plain continuous actions and so is only "
            f"meaningful for an atom-free policy, got num_atoms={network.num_atoms}. "
            "A game with atoms needs a tree-aware exploitability instead."
        )
    logits, means, scale_trils, _ = network.apply(params, obs)
    action_mask = jnp.ones_like(logits, dtype=bool)

    def one(k: chex.PRNGKey) -> chex.Array:
        _, raw_action = sample_mixture_component(logits, means, scale_trils, action_mask, 0, k)
        return space.clip(raw_action)

    return jax.vmap(one)(jax.random.split(key, num_samples))


def collect_mixture_episode(
    game: ZeroSumGame,
    network: MixtureActorCritic,
    params,
    magnet_params,
    opponent_action_fn: OpponentActionFn,
    key: chex.PRNGKey,
    num_envs: int,
    perspective: int = 0,
) -> Episode:
    """Like `rollout.collect_episode`, but for a `MixtureActorCritic`.

    The own-side sampling is `vmap`ped over `num_envs` independent rng keys
    (see `_sample_mixture_one`); the opponent's batch of actions still comes
    from a single call to `opponent_action_fn(key, num_envs)`, since that's
    its existing contract.
    """
    if perspective not in (0, 1):
        raise ValueError(f"perspective must be 0 or 1, got {perspective}")

    own_key, opponent_key = jax.random.split(key)
    keys = jax.random.split(own_key, num_envs)
    episode = jax.vmap(_sample_mixture_one, in_axes=(None, None, None, None, None, 0))(
        game, perspective, network, params, magnet_params, keys
    )

    opponent_action = opponent_action_fn(opponent_key, num_envs)
    if perspective == 0:
        reward = game.payoff_batch(episode.action_value, opponent_action)
    else:
        reward = -game.payoff_batch(opponent_action, episode.action_value)
    return episode.replace(reward=reward)


def _sample_self_play_episode_one(
    game: ZeroSumGame,
    network_1: MixtureActorCritic,
    params_1,
    magnet_params_1,
    network_2: MixtureActorCritic,
    params_2,
    magnet_params_2,
    key: chex.PRNGKey,
) -> tuple[Episode, Episode]:
    """One (unbatched) self-play sample -- both players' obs/sampling and the resulting
    reward, all keyed off the single `key` passed in.
    """
    key_1, key_2 = jax.random.split(key)
    episode_1 = _sample_mixture_one(game, 0, network_1, params_1, magnet_params_1, key_1)
    episode_2 = _sample_mixture_one(game, 1, network_2, params_2, magnet_params_2, key_2)

    reward = game.payoff(episode_1.action_value, episode_2.action_value)
    return episode_1.replace(reward=reward), episode_2.replace(reward=-reward)


def collect_mixture_self_play_episode(
    game: ZeroSumGame,
    network_1: MixtureActorCritic,
    params_1,
    magnet_params_1,
    network_2: MixtureActorCritic,
    params_2,
    magnet_params_2,
    key: chex.PRNGKey,
    num_envs: int,
) -> tuple[Episode, Episode]:
    """Like `rollout.collect_self_play_episode`, but for two `MixtureActorCritic`s.

    `vmap`ped purely over `num_envs` independent rng keys: everything else
    (both players' obs, sampling, and the resulting reward) happens inside
    `_sample_self_play_episode_one`, keyed off that one `key` per env.
    """
    keys = jax.random.split(key, num_envs)
    return jax.vmap(_sample_self_play_episode_one, in_axes=(None, None, None, None, None, None, None, 0))(
        game, network_1, params_1, magnet_params_1, network_2, params_2, magnet_params_2, keys
    )


def mixture_ppo_loss_from_outputs(
    logits: chex.Array,
    means: chex.Array,
    scale_trils: chex.Array,
    value_pred: chex.Array,
    num_atoms: int,
    episode: Episode,
    advantage: chex.Array,
    clip_eps: float,
    value_coef: float,
    category_entropy_coef: float,
    gaussian_entropy_coef: float,
    trpo_category_kl_coef: float,
    trpo_gaussian_kl_coef: float,
    magnet_category_kl_coef: float,
    magnet_gaussian_kl_coef: float,
    mean_box_penalty: chex.Array = 0.0,
    low: chex.Array | None = None,
    high: chex.Array | None = None,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    """Clipped-surrogate PPO loss plus KL penalties, for a single (unbatched) `Episode`,
    given the current policy's already-computed forward pass at `episode.obs`.

    `mixture_ppo_loss` is the usual entry point -- it runs that forward pass
    itself. Taking the outputs as arguments instead is what lets
    `build_mixture_ppo_loss_fn(shared_obs=True)` evaluate the network *once*
    for a whole batch that shares one observation, rather than once per sample.

    Computed as two *separate* factors (categorical, Gaussian) rather than
    summing the two heads' log-probs into one joint log-prob and clipping a
    single combined ratio (which only bounds their product, and lets one
    head's move be offset by the other's): each head gets its own PPO ratio
    and its own clipped surrogate against the shared advantage. The entropy
    bonus and both KL penalties are likewise split per head -- each head can
    need a very different weight, since e.g. collapsing the categorical
    component distribution is a qualitatively different failure from the
    Gaussian spread collapsing.

    Operates on one sample at a time -- `Episode` is a `chex.dataclass` (a
    registered pytree), so `vmap`ing this whole function over its leading
    (env) axis batches every field at once; see `build_mixture_ppo_loss_fn`.
    `advantage` is taken as a separate argument, already normalized, rather
    than computed from `episode.reward`/`episode.value` here, because
    normalization is a batch-wide statistic and so has to happen before
    vmapping.

    `mean_box_penalty` arrives already weighted by `mean_box_penalty_coef`
    (see `projected_means_and_penalty`): the projection that `means` went
    through has thrown away how far outside the box the raw mean head was, so
    the term has to be computed by the caller that still holds the raw means.
    It is a property of the observation, not of the sample, so it is identical
    for every sample sharing an `obs`.

    Both Gaussian KLs are summed over *every* component, weighted by that
    component's probability under the sampling-time policy, rather than being
    evaluated at the drawn component alone. Unlike the surrogate -- which needs
    the importance ratio at `raw_action` and so exists only for the component
    that was actually drawn -- a Gaussian KL is a closed form in the head
    outputs, with an exact gradient for all `num_components` of them at once.
    Indexing it by the draw would estimate a quantity that can simply be
    computed, at the cost of the draw's variance and of leaving a low-weight
    component's mean/scale rows un-anchored on the samples that missed it. The
    weights come from `episode.logits` (the policy the component was drawn
    from), not from the current `logits`, so the expectation matches the
    per-draw form exactly even after several PPO epochs have moved the policy.

    Atoms and legality masks both act by *zeroing* terms rather than by
    branching. For a sample that drew an atom, the Gaussian ratio, its clipped
    surrogate and the marginal-density entropy are all forced to `0.0`: an atom
    has no mean and no spread, so there is nothing there for those terms to say.
    The Gaussian KLs need no such factor -- their weights already carry it, and
    an illegal or atom-only state drives every Gaussian weight to exactly `0.0`.
    Illegal categorical entries are handled inside
    `masked_log_softmax`/`categorical_kl` via `episode.action_mask`, the mask
    recorded at sampling time -- re-applying exactly that mask is what keeps the
    PPO ratio a ratio of two densities over the same support.

    The current policy's outputs are the *only* ones evaluated for this
    update -- `episode.logits`/`means`/`scale_trils` (the sampling-time
    distribution) and `episode.magnet_logits`/`magnet_means`/`magnet_scale_trils`
    (the magnet snapshot's distribution, at the same `episode.obs`) come
    straight from the episode rather than being recomputed from
    `old_params`/`magnet_params` here; see `collect_mixture_episode`.
    `trpo_*_kl_coef * KL(old || current)` is a TRPO-style trust region
    against the policy this update's rollout was collected with;
    `magnet_*_kl_coef * KL(current || magnet)` pulls towards the
    periodically-snapshotted magnet policy.

    **Off-policy samples.** When `episode.behavior_eps` is set, the continuous
    action was drawn from the exploring behavior policy `mu` (see
    `sample_mixture_component`) and `low`/`high` must give the box it explored
    -- shared `(d,)`, or per component `(num_components, d)`.
    The categorical factor was sampled on-policy and is untouched. The Gaussian
    factor uses the decoupled PPO objective: the same clipped surrogate in
    `r = pi_new / pi_old`, weighted by `w = pi_old / mu` -- so the trust region
    still acts around the sampling-time policy, and `w <= 1 / (1 - eps)` keeps
    the weights bounded. The sampled-action entropy estimate `-log p(x)` gets
    the same `w`, turning it back into an estimate under `pi_old`; without it a
    uniform draw deep in a Gaussian's tail would dominate the bonus. The KLs are
    closed forms over the components and need no correction. Only this
    decision's own action is reweighted: the return still reflects the rest of
    the behavior trajectory -- in particular the *opponent's* exploration,
    which is exactly what the responder is meant to learn from.
    """
    mask = episode.action_mask

    # An atom is the whole action: it has no Gaussian factor to weigh in on.
    is_gaussian = (episode.component >= num_atoms).astype(jnp.float32)

    old_category_log_prob, old_gaussian_log_prob = mixture_log_probs(
        episode.logits, episode.means, episode.scale_trils, mask,
        episode.component, episode.raw_action, num_atoms,
    )
    new_category_log_prob, new_gaussian_log_prob = mixture_log_probs(
        logits, means, scale_trils, mask, episode.component, episode.raw_action, num_atoms
    )
    category_ratio = jnp.exp(new_category_log_prob - old_category_log_prob)
    gaussian_ratio = jnp.exp(new_gaussian_log_prob - old_gaussian_log_prob)

    def clipped_surrogate(ratio: chex.Array) -> chex.Array:
        unclipped = ratio * advantage
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        return jnp.minimum(unclipped, clipped)

    category_policy_loss = -clipped_surrogate(category_ratio)
    if episode.behavior_eps is None:
        # On-policy: the sample came from `pi_old` itself.
        is_weight = jnp.ones(())
        gaussian_surrogate = clipped_surrogate(gaussian_ratio)
    else:
        if low is None or high is None:
            raise ValueError("an exploring episode needs the policy's box (low, high) to score it")
        # The box the uniform was drawn on: the drawn component's own, when
        # `low`/`high` are per component (see `component_boxes`).
        explored_low, explored_high = component_box(
            low, high, gaussian_component_index(episode.component, num_atoms)
        )
        behavior_log_prob = behavior_gaussian_log_prob(
            old_gaussian_log_prob, episode.raw_action, episode.behavior_eps,
            explored_low, explored_high,
        )
        is_weight = jnp.exp(old_gaussian_log_prob - behavior_log_prob)
        # `w * min(r A, clip(r) A)`, with `w * r = pi_new / mu` formed directly and
        # the clip taken on the log-ratio: a uniform draw far into a narrow
        # Gaussian's tail has `pi_old(x)` underflowing, so `r` alone can overflow
        # (and `w * r` would be `0 * inf`) where the product is perfectly finite.
        log_ratio = new_gaussian_log_prob - old_gaussian_log_prob
        unclipped = jnp.exp(new_gaussian_log_prob - behavior_log_prob) * advantage
        clipped = is_weight * jnp.exp(
            jnp.clip(log_ratio, jnp.log1p(-clip_eps), jnp.log1p(clip_eps))
        ) * advantage
        gaussian_surrogate = jnp.minimum(unclipped, clipped)
    gaussian_policy_loss = -is_gaussian * gaussian_surrogate
    policy_loss = category_policy_loss + gaussian_policy_loss

    value_loss = jnp.square(value_pred - episode.reward)

    category_entropy = masked_categorical_entropy(logits, mask)
    action_entropy = -is_gaussian * is_weight * mixture_marginal_log_prob(
        logits, means, scale_trils, mask, episode.raw_action, num_atoms
    )
    entropy = category_entropy + action_entropy

    # Weight of each Gaussian component under the policy the component was drawn
    # from; sums to `P(drew a Gaussian)` rather than to one, which is what makes
    # the two sums below match the sampled-component estimator in expectation.
    component_weight = jnp.exp(masked_log_softmax(episode.logits, mask))[num_atoms:]

    trpo_category_kl = categorical_kl(episode.logits, logits, mask)
    trpo_gaussian_kl = jnp.sum(
        component_weight * gaussian_kl(episode.means, episode.scale_trils, means, scale_trils)
    )

    magnet_category_kl = categorical_kl(logits, episode.magnet_logits, mask)
    magnet_gaussian_kl = jnp.sum(
        component_weight
        * gaussian_kl(means, scale_trils, episode.magnet_means, episode.magnet_scale_trils)
    )

    loss = (
        policy_loss
        + value_coef * value_loss
        - category_entropy_coef * category_entropy
        - gaussian_entropy_coef * action_entropy
        + trpo_category_kl_coef * trpo_category_kl
        + trpo_gaussian_kl_coef * trpo_gaussian_kl
        + magnet_category_kl_coef * magnet_category_kl
        + magnet_gaussian_kl_coef * magnet_gaussian_kl
        + mean_box_penalty
    )

    # Both Gaussian diagnostics are expectations under `pi_old`, so they carry
    # the same importance weight (exactly 1 on an on-policy batch).
    category_approx_kl = old_category_log_prob - new_category_log_prob
    gaussian_approx_kl = is_gaussian * is_weight * (old_gaussian_log_prob - new_gaussian_log_prob)
    category_clip_frac = (jnp.abs(category_ratio - 1.0) > clip_eps).astype(jnp.float32)
    gaussian_clip_frac = is_gaussian * is_weight * (
        jnp.abs(gaussian_ratio - 1.0) > clip_eps
    ).astype(jnp.float32)

    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "category_policy_loss": category_policy_loss,
        "gaussian_policy_loss": gaussian_policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "category_entropy": category_entropy,
        "gaussian_entropy": action_entropy,  # marginal mixture entropy estimate (weighted by gaussian_entropy_coef)
        "atom_frac": 1.0 - is_gaussian,  # share of samples that drew a discrete atom
        "approx_kl": category_approx_kl + gaussian_approx_kl,
        "category_approx_kl": category_approx_kl,
        "gaussian_approx_kl": gaussian_approx_kl,
        "clip_frac": 0.5 * (category_clip_frac + gaussian_clip_frac),
        "category_clip_frac": category_clip_frac,
        "gaussian_clip_frac": gaussian_clip_frac,
        "trpo_kl": trpo_category_kl + trpo_gaussian_kl,
        "trpo_category_kl": trpo_category_kl,
        "trpo_gaussian_kl": trpo_gaussian_kl,
        "magnet_kl": magnet_category_kl + magnet_gaussian_kl,
        "magnet_category_kl": magnet_category_kl,
        "magnet_gaussian_kl": magnet_gaussian_kl,
        "mean_box_penalty": jnp.asarray(mean_box_penalty, dtype=jnp.float32),
    }
    return loss, metrics


def projected_means_and_penalty(
    network: MixtureActorCritic,
    raw_means: chex.Array,
    mean_box_penalty_coef: float,
) -> tuple[chex.Array, chex.Array]:
    """The means the loss should score, and the box penalty to add to that loss.

    Splits what `MixtureActorCritic.__call__` does in one step when it is only
    asked to act: the loss runs the forward pass with `project_means=False` and
    redoes the projection here, so that the raw means are still in hand for
    `mean_box_excess`. With `clip_means` off there is no box to speak of --
    means pass through and the penalty is exactly zero.
    """
    if not network.clip_means:
        return raw_means, jnp.zeros(())
    means = project_means_to_box(raw_means, network.low, network.high)
    penalty = mean_box_penalty_coef * mean_box_excess(raw_means, network.low, network.high)
    return means, penalty


def mixture_ppo_loss(
    params,
    network: MixtureActorCritic,
    episode: Episode,
    advantage: chex.Array,
    clip_eps: float,
    value_coef: float,
    category_entropy_coef: float,
    gaussian_entropy_coef: float,
    trpo_category_kl_coef: float,
    trpo_gaussian_kl_coef: float,
    magnet_category_kl_coef: float,
    magnet_gaussian_kl_coef: float,
    mean_box_penalty_coef: float = 0.0,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    """`mixture_ppo_loss_from_outputs`, forward-passing `params` at `episode.obs` first."""
    logits, raw_means, scale_trils, value_pred = network.apply(
        params, episode.obs, project_means=False
    )
    means, mean_box_penalty = projected_means_and_penalty(
        network, raw_means, mean_box_penalty_coef
    )
    lows, highs = component_boxes(network)
    return mixture_ppo_loss_from_outputs(
        logits, means, scale_trils, value_pred, network.num_atoms, episode, advantage,
        clip_eps, value_coef, category_entropy_coef, gaussian_entropy_coef,
        trpo_category_kl_coef, trpo_gaussian_kl_coef,
        magnet_category_kl_coef, magnet_gaussian_kl_coef, mean_box_penalty,
        low=lows, high=highs,
    )


def build_mixture_ppo_loss_fn(
    player: int,
    category_entropy_coef: float,
    gaussian_entropy_coef: float,
    trpo_category_kl_coef: float,
    trpo_gaussian_kl_coef: float,
    magnet_category_kl_coef: float,
    magnet_gaussian_kl_coef: float,
    mean_box_penalty_coef: float = 0.0,
    shared_obs: bool = False,
    advantage_estimator: str = "monte_carlo",
    gamma: float = 1.0,
    vtrace_lambda: float = 0.95,
    vtrace_rho_bar: float = 1.0,
    vtrace_c_bar: float = 1.0,
):
    """`player`'s PPO loss over a whole `Episode` batch, one-shot or sequential alike.

    Binds the six per-head entropy/KL coefficients and the mean-box penalty
    coefficient (all constant for one
    `ppo_update` call) and returns a function matching `ppo_update`'s
    `loss_fn` contract: `(params, network, batch, clip_eps, value_coef,
    entropy_coef) -> (scalar_loss, dict_of_scalar_metrics)`. `entropy_coef`
    (the generic `PPOHyperparams` field `ppo_update` always passes) is
    accepted but unused -- the mixture loss is weighted by the two bound
    per-head coefficients instead.

    The batch's leading axes are flattened to one and `mixture_ppo_loss` is
    `vmap`ed over that, then every reduction -- the advantage normalization
    included -- is weighted by `player_weight`. A one-shot batch is all
    `player`'s, so the weight is all-ones and each reduction is a plain mean; a
    trajectory batch holds both players' interleaved decisions plus the padding
    of finished episodes, and the same weighted reductions drop those to zero.
    `decisions_per_episode` is reported only when there *is* a time axis.

    `shared_obs` says every sample in the batch carries the *same*
    observation -- true of the one-shot games in `games.examples`, whose
    `ZeroSumGame.observation` is a constant (see
    `ZeroSumGame.constant_observation`, which is what the trainers pass
    here). The forward pass is then lifted out of the `vmap` and run once on
    `batch.obs[0]` instead of once per sample, which is the whole cost of the
    update for a small torso and a large `num_envs`; only the cheap per-sample
    log-prob/KL arithmetic stays batched. Mathematically identical -- the
    lifted outputs are exactly what the per-sample apply would have returned
    -- so it is purely a saving, but it is silently *wrong* if the
    observations actually differ, hence off by default (and never right for a
    sequential game, whose whole point is a per-infoset observation).

    `advantage_estimator` picks where the advantage and the value target come from:

      * `"monte_carlo"` -- the recorded return, `reward - value`, with the value
        head regressed on `reward`. No bootstrapping; what every run did before.
      * `"vtrace"` -- V-trace over the player's own decisions (`training.vtrace`),
        recomputed at every epoch from the *current* parameters: `V` is the
        current value head and `rho = pi / mu` is the current policy's marginal
        action probability over the behavior policy's (the recorded sampling-time
        policy, with its `explore_eps` exploration if any; see
        `hybrid_action_log_prob`). The PPO surrogate gets the V-trace advantage
        and the value head regresses on `v_s`. Sequential batches only -- a
        one-shot batch has no time axis to bootstrap along.
    """
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")
    if advantage_estimator not in ("monte_carlo", "vtrace"):
        raise ValueError(
            f"advantage_estimator must be 'monte_carlo' or 'vtrace', got {advantage_estimator!r}"
        )
    if advantage_estimator == "vtrace" and shared_obs:
        raise ValueError("advantage_estimator='vtrace' needs per-decision observations; shared_obs is one-shot only")

    coefs = (
        category_entropy_coef, gaussian_entropy_coef,
        trpo_category_kl_coef, trpo_gaussian_kl_coef,
        magnet_category_kl_coef, magnet_gaussian_kl_coef,
    )

    def loss_fn(
        params,
        network: MixtureActorCritic,
        batch: Episode,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        del entropy_coef

        if advantage_estimator == "vtrace":
            return vtrace_loss(params, network, batch, clip_eps, value_coef)

        weight = player_weight(batch, player)
        # Normalized over the player's own decisions, across the whole batch:
        # a batch statistic, so it has to be computed before the per-sample vmap.
        advantage = normalized_advantage(batch.reward - batch.value, weight)

        flat = flatten_batch_axes(batch)
        flat_weight, flat_advantage = weight.reshape(-1), advantage.reshape(-1)

        if shared_obs:
            logits, raw_means, scale_trils, value_pred = network.apply(
                params, flat.obs[0], project_means=False
            )
            means, mean_box_penalty = projected_means_and_penalty(
                network, raw_means, mean_box_penalty_coef
            )
            per_sample_loss, metrics = jax.vmap(
                mixture_ppo_loss_from_outputs,
                in_axes=(None, None, None, None, None, 0, 0, None, None, None, None, None, None, None, None, None),
            )(
                logits, means, scale_trils, value_pred, network.num_atoms, flat, flat_advantage,
                clip_eps, value_coef, *coefs, mean_box_penalty,
            )
        else:
            per_sample_loss, metrics = jax.vmap(
                mixture_ppo_loss,
                in_axes=(None, None, 0, 0, None, None, None, None, None, None, None, None, None),
            )(
                params, network, flat, flat_advantage, clip_eps, value_coef, *coefs,
                mean_box_penalty_coef,
            )

        loss = masked_mean(per_sample_loss, flat_weight)
        metrics = jax.tree_util.tree_map(lambda m: masked_mean(m, flat_weight), metrics)
        if batch.actor.ndim > 1:
            metrics["decisions_per_episode"] = jnp.mean(jnp.sum(weight, axis=-1))
        return loss, metrics

    def vtrace_loss(params, network, batch, clip_eps, value_coef):
        if batch.actor.ndim != 2 or batch.step_reward is None:
            raise ValueError(
                "advantage_estimator='vtrace' needs a sequential (num_envs, max_steps) batch with "
                "step_reward recorded; see training.sequential_rollout"
            )
        weight = player_weight(batch, player)
        flat = flatten_batch_axes(batch)
        flat_weight = weight.reshape(-1)
        # Per component: under `bucket_means` exploration is bucket-local.
        low, high = component_boxes(network)

        # One forward pass per decision; the V-trace targets and the loss both read it.
        logits, raw_means, scale_trils, value_pred = jax.vmap(
            lambda obs: network.apply(params, obs, project_means=False)
        )(flat.obs)
        means, mean_box_penalty = jax.vmap(
            lambda raw: projected_means_and_penalty(network, raw, mean_box_penalty_coef)
        )(raw_means)

        # `log pi(a) - log mu(a)`: the current policy against the one that sampled.
        target_log_prob = jax.vmap(
            hybrid_action_log_prob, in_axes=(0, 0, 0, 0, 0, 0, None)
        )(logits, means, scale_trils, flat.action_mask, flat.component, flat.raw_action,
          network.num_atoms)
        if flat.behavior_eps is None:
            behavior_log_prob = jax.vmap(
                hybrid_action_log_prob, in_axes=(0, 0, 0, 0, 0, 0, None)
            )(flat.logits, flat.means, flat.scale_trils, flat.action_mask, flat.component,
              flat.raw_action, network.num_atoms)
        else:
            behavior_log_prob = jax.vmap(
                hybrid_action_log_prob, in_axes=(0, 0, 0, 0, 0, 0, None, 0, None, None)
            )(flat.logits, flat.means, flat.scale_trils, flat.action_mask, flat.component,
              flat.raw_action, network.num_atoms, flat.behavior_eps, low, high)
        log_rhos = (target_log_prob - behavior_log_prob).reshape(weight.shape)

        sign = 1.0 if player == 0 else -1.0
        targets = vtrace(
            own=weight > 0,
            rewards=sign * batch.step_reward,
            values=value_pred.reshape(weight.shape),
            log_rhos=log_rhos,
            gamma=gamma,
            lambda_=vtrace_lambda,
            rho_bar=vtrace_rho_bar,
            c_bar=vtrace_c_bar,
        )
        targets = jax.tree_util.tree_map(jax.lax.stop_gradient, targets)
        flat_advantage = normalized_advantage(targets.pg_advantage, weight).reshape(-1)
        # The value head regresses on `reward`; here that is the V-trace target.
        scored = flat.replace(reward=targets.vs.reshape(-1))

        def per_sample(logits, means, scale_trils, value_pred, episode, adv, penalty):
            return mixture_ppo_loss_from_outputs(
                logits, means, scale_trils, value_pred, network.num_atoms, episode, adv,
                clip_eps, value_coef, *coefs, penalty, low=low, high=high,
            )

        per_sample_loss, metrics = jax.vmap(per_sample)(
            logits, means, scale_trils, value_pred, scored, flat_advantage, mean_box_penalty
        )
        loss = masked_mean(per_sample_loss, flat_weight)
        metrics = jax.tree_util.tree_map(lambda m: masked_mean(m, flat_weight), metrics)
        metrics["decisions_per_episode"] = jnp.mean(jnp.sum(weight, axis=-1))
        # How off-policy the batch is: the untruncated ratio and how often it is cut.
        raw_rho = jnp.exp(jnp.minimum(log_rhos, 20.0)).reshape(-1)
        metrics["vtrace_rho"] = masked_mean(raw_rho, flat_weight)
        metrics["vtrace_rho_clip_frac"] = masked_mean(
            (raw_rho > vtrace_rho_bar).astype(jnp.float32), flat_weight
        )
        metrics["vtrace_target"] = masked_mean(targets.vs.reshape(-1), flat_weight)
        return loss, metrics

    return loss_fn


def build_mixture_network(hyperparams: MixturePPOHyperparams) -> MixtureActorCritic:
    return MixtureActorCritic(
        action_dim=hyperparams.action_dim,
        num_components=hyperparams.num_components,
        hidden_dims=hyperparams.hidden_dims,
        low=jnp.asarray(hyperparams.low, dtype=jnp.float32),
        high=jnp.asarray(hyperparams.high, dtype=jnp.float32),
        activation=hyperparams.activation,
        normalization=hyperparams.normalization,
        clip_means=hyperparams.clip_means,
        num_atoms=hyperparams.num_atoms,
        full_covariance=hyperparams.full_covariance,
        scale_parameterization=hyperparams.scale_parameterization,
        max_correlation=hyperparams.max_correlation,
        sigma_min=hyperparams.sigma_min,
        sigma_max=hyperparams.sigma_max,
        bucket_means=hyperparams.bucket_means,
    )
