"""A two-phase (discrete-then-continuous) actor-critic.
"""

from __future__ import annotations

from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

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


def _scale_bias_init(
    low: chex.Array,
    high: chex.Array,
    num_components: int,
    full_covariance: bool,
    scale_parameterization: str = "linear",
) -> Callable:
    """Bias initializer for `scale_head`, scaled to the action range.

    The same initial policy either way -- each component's conditional standard
    deviation is one `num_components`-th of the box's half width, uncorrelated --
    written in whichever coordinate the head emits. Under `"log"` the diagonal
    carries `log std` and the off-diagonal stays at zero, which is exactly the
    `S = 0` that makes `I + S` the identity.
    """
    action_dim = low.shape[0]
    slots = diagonal_slots(action_dim, full_covariance)
    size = scale_param_size(action_dim, full_covariance)

    def init_fn(key: chex.PRNGKey, shape: tuple[int, ...], dtype=jnp.float32) -> chex.Array:
        del key
        std = jnp.maximum(_action_width(low, high) / (2 * num_components), MIN_ACTION_WIDTH)
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
            bias_init=_spread_bias_init(self.low, self.high, self.num_components),
            name="means_head",
        )(torso)
        means = means_flat.reshape(self.num_components, self.action_dim)
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
                self.scale_parameterization,
            ),
            name="scale_head",
        )(torso)
        scale_raw = scale_flat.reshape(self.num_components, scale_size)
        # Floor on the conditional standard deviations, ceiling at the box width,
        # in whichever coordinate the head emits. Straight-through either way, so
        # a saturated component keeps receiving gradient.
        if self.scale_parameterization == "log":
            scale_tril = scale_tril_from_log_diag(
                scale_raw,
                self.action_dim,
                self.full_covariance,
                LOG_SIGMA_MIN,
                jnp.log(_action_width(self.low, self.high)),
                self.max_correlation,
            )
        else:
            scale_tril = pack_scale_tril(scale_raw, self.action_dim, self.full_covariance)
            # The projection onto the feasible set of factors.
            scale_tril = clamp_scale_tril(
                scale_tril, SIGMA_MIN, _action_width(self.low, self.high)
            )

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
) -> tuple[chex.Array, chex.Array]:
    """Draw one `(component, raw_action)` from a masked hybrid mixture policy.
    """
    component_key, noise_key = jax.random.split(key)
    component = jax.random.categorical(component_key, jnp.where(mask, logits, MASKED_LOGIT))
    index = gaussian_component_index(component, num_atoms)
    mean = means[index]
    raw_action = gaussian_sample(mean, scale_trils[index], jax.random.normal(noise_key, mean.shape))
    return component, raw_action


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

    Atoms and legality masks both act by *zeroing* terms rather than by
    branching. For a sample that drew an atom, the Gaussian ratio, its clipped
    surrogate, both Gaussian KLs and the marginal-density entropy are all forced
    to `0.0`: an atom has no mean and no spread, so there is nothing there for
    those terms to say. Illegal categorical entries are handled inside
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
    gaussian_policy_loss = -is_gaussian * clipped_surrogate(gaussian_ratio)
    policy_loss = category_policy_loss + gaussian_policy_loss

    value_loss = jnp.square(value_pred - episode.reward)

    category_entropy = masked_categorical_entropy(logits, mask)
    action_entropy = -is_gaussian * mixture_marginal_log_prob(
        logits, means, scale_trils, mask, episode.raw_action, num_atoms
    )
    entropy = category_entropy + action_entropy

    index = gaussian_component_index(episode.component, num_atoms)
    mean = means[index]
    scale_tril = scale_trils[index]

    old_mean = episode.means[index]
    old_scale_tril = episode.scale_trils[index]
    trpo_category_kl = categorical_kl(episode.logits, logits, mask)
    trpo_gaussian_kl = is_gaussian * gaussian_kl(old_mean, old_scale_tril, mean, scale_tril)

    magnet_mean = episode.magnet_means[index]
    magnet_scale_tril = episode.magnet_scale_trils[index]
    magnet_category_kl = categorical_kl(logits, episode.magnet_logits, mask)
    magnet_gaussian_kl = is_gaussian * gaussian_kl(mean, scale_tril, magnet_mean, magnet_scale_tril)

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

    category_approx_kl = old_category_log_prob - new_category_log_prob
    gaussian_approx_kl = is_gaussian * (old_gaussian_log_prob - new_gaussian_log_prob)
    category_clip_frac = (jnp.abs(category_ratio - 1.0) > clip_eps).astype(jnp.float32)
    gaussian_clip_frac = is_gaussian * (jnp.abs(gaussian_ratio - 1.0) > clip_eps).astype(jnp.float32)

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
    return mixture_ppo_loss_from_outputs(
        logits, means, scale_trils, value_pred, network.num_atoms, episode, advantage,
        clip_eps, value_coef, category_entropy_coef, gaussian_entropy_coef,
        trpo_category_kl_coef, trpo_gaussian_kl_coef,
        magnet_category_kl_coef, magnet_gaussian_kl_coef, mean_box_penalty,
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
    """
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")

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
    )
