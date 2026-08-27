"""A two-phase (discrete-then-continuous) actor-critic.

`ActorCritic` (see `actor_critic.py`) is a single diagonal Gaussian: it can
represent one continuous mode. Some games (see
`games.examples.ContinuousMatchingPennies`) have Nash equilibria that are
genuinely multi-modal — a mix over a handful of distinct points in an
otherwise continuous action space — which a unimodal Gaussian cannot
represent no matter how it's trained.

`MixtureActorCritic` fixes that by picking, per action, one of
`num_components` categorical "phases" and then sampling a Gaussian centered
on that phase's own mean. Concretely: sample a component `k ~ Categorical
(logits)`, then `action ~ Normal(means[k], std)`. Training needs its own
rollout/loss functions because the policy now has a discrete part.

**Atoms.** The categorical head is widened to `num_atoms + num_components`,
and the first `num_atoms` entries are *point masses* -- discrete actions
carrying no continuous parameter at all (fold, check, call in a poker-like
game; see `games.spaces.HybridSpace`). Drawing one of those is the whole
action: no Gaussian is consulted, so its log-prob, its KL and its entropy
contribution are all masked out of the loss for that sample. This is what
makes the probability of a discrete choice a *number the policy states*,
rather than an integral of the Gaussian mixture over some region of the
action space -- and hence what gives that choice a real categorical policy
gradient instead of the gradient of a piecewise-constant payoff.

**Legality masks.** A game may forbid some kinds at some nodes (calling with
no bet outstanding). `action_mask` on the `Episode` records, per sample,
which categorical logits were legal *when it was sampled*; every softmax,
entropy and KL in the loss re-applies it. Skipping that would leave the PPO
importance ratios comparing against a distribution that was never sampled
from. The mask is over logits, not kinds: `expand_kind_mask` replicates the
continuous kind's bit across all `num_components` Gaussian components.

With `num_atoms == 0` and an all-`True` mask, every one of those terms
collapses back to the plain mixture policy, which is what the one-shot games
in `games.examples` use.
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
    masked_categorical_entropy,
    masked_log_softmax,
)
from .config import MixturePPOHyperparams

OpponentActionFn = Callable[[chex.PRNGKey, int], chex.Array]

LOG_STD_MIN = -6.907755  # log(1e-3)
MIN_ACTION_WIDTH = 1e-3  # == exp(LOG_STD_MIN)


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


def _std_bias_init(low: chex.Array, high: chex.Array, num_components: int) -> Callable:
    """Bias initializer for `log_std_head`, scaled to the action range.
    """

    def init_fn(key: chex.PRNGKey, shape: tuple[int, ...], dtype=jnp.float32) -> chex.Array:
        del key
        std = jnp.maximum(_action_width(low, high) / (2 * num_components), MIN_ACTION_WIDTH)
        log_std = jnp.log(std).astype(dtype)  # (action_dim,)
        values = jnp.broadcast_to(log_std[None, :], (num_components, log_std.shape[0]))
        return values.reshape(shape).astype(dtype)

    return init_fn


class MixtureActorCritic(nn.Module):
    """Shared torso, four linear heads: component logits, means, log-stds, value.
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

    @nn.compact
    def __call__(
        self, obs: chex.Array, train: bool = False
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
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
        if self.clip_means:
            means = jnp.clip(means, self.low, self.high)

        log_std_flat = nn.Dense(
            self.num_components * self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=_std_bias_init(self.low, self.high, self.num_components),
            name="log_std_head",
        )(torso)
        log_std = log_std_flat.reshape(self.num_components, self.action_dim)
        log_std = jnp.clip(log_std, LOG_STD_MIN, jnp.log(_action_width(self.low, self.high)))

        value = nn.Dense(1, name="value_head")(torso)

        return logits, means, log_std, jnp.squeeze(value, axis=-1)


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
    """Row of `means`/`log_stds` that `component` refers to, clamped to be in range.
    """
    return jnp.maximum(component - num_atoms, 0)


def mixture_log_probs(
    logits: chex.Array,
    means: chex.Array,
    log_stds: chex.Array,
    mask: chex.Array,
    component: chex.Array,
    raw_action: chex.Array,
    num_atoms: int,
) -> tuple[chex.Array, chex.Array]:
    """`(category_log_prob, gaussian_log_prob)` for one sample -- kept as separate factors.

    """
    category_log_prob = masked_log_softmax(logits, mask)[component]
    index = gaussian_component_index(component, num_atoms)
    log_prob = gaussian_log_prob(raw_action, means[index], log_stds[index])
    return category_log_prob, jnp.where(component >= num_atoms, log_prob, 0.0)


def mixture_marginal_log_prob(
    logits: chex.Array,
    means: chex.Array,
    log_stds: chex.Array,
    mask: chex.Array,
    raw_action: chex.Array,
    num_atoms: int,
) -> chex.Array:
    """`log p(a | a is continuous)` of the marginal mixture density at a single `raw_action`.
    """
    log_weights = masked_log_softmax(logits[num_atoms:], mask[num_atoms:])  # (num_components,)
    per_component = jax.vmap(gaussian_log_prob, in_axes=(None, 0, 0))(raw_action, means, log_stds)
    return jax.nn.logsumexp(log_weights + per_component)


@chex.dataclass
class Episode:
    """Everything recorded while sampling one batch of transitions from a `MixtureActorCritic`.
 
    """

    obs: chex.Array
    action_mask: chex.Array  # (num_atoms + num_components,) bool, which logits were legal here
    logits: chex.Array  # (num_atoms + num_components,) categorical logits at sample time
    means: chex.Array  # (num_components, action_dim) each Gaussian component's mean at sample time
    log_stds: chex.Array  # (num_components, action_dim) each component's log-std at sample time
    magnet_logits: chex.Array  # (num_atoms + num_components,) logits under `magnet_params`, same obs
    magnet_means: chex.Array  # (num_components, action_dim) under `magnet_params`
    magnet_log_stds: chex.Array  # (num_components, action_dim) under `magnet_params`
    component: chex.Array  # which categorical entry was sampled: an atom, or a Gaussian component
    raw_action: chex.Array  # unclipped Gaussian sample; meaningless (but finite) when an atom was drawn
    action_kind: chex.Array  # `HybridAction.kind` actually played -- `component_to_kind(component)`
    action_value: chex.Array  # `raw_action` clipped to the action space; read by the game only on the continuous kind
    value: chex.Array
    reward: chex.Array


def sample_mixture_component(
    logits: chex.Array,
    means: chex.Array,
    log_stds: chex.Array,
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
    raw_action = mean + jnp.exp(log_stds[index]) * jax.random.normal(noise_key, mean.shape)
    return component, raw_action


def _sample_mixture_one(
    game: ZeroSumGame, player: int, network: MixtureActorCritic, params, magnet_params, key: chex.PRNGKey
):
    """One (unbatched) sample: draws `obs`, samples a component/action, and evaluates both
    `params` and `magnet_params` at that `obs`. Everything is keyed off the single `key`
    passed in -- `vmap` this over `num_envs` keys to draw a whole batch (see
    `collect_mixture_episode`/`collect_mixture_self_play_episode`).
    """
    obs_key, sample_key = jax.random.split(key)
    space = game.action_space(player)
    obs = game.observation(player, obs_key)

    logits, means, log_stds, value = network.apply(params, obs)
    magnet_logits, magnet_means, magnet_log_stds, _ = network.apply(magnet_params, obs)

    # A one-shot `ZeroSumGame` has a single unconstrained continuous action: no
    # atoms, nothing illegal, so the mask is all-`True` and every masked term in
    # the loss reduces to its unmasked form.
    action_mask = jnp.ones_like(logits, dtype=bool)
    component, raw_action = sample_mixture_component(
        logits, means, log_stds, action_mask, network.num_atoms, sample_key
    )
    action_kind = component_to_kind(component, network.num_atoms)
    action_value = space.clip(raw_action)

    return (
        obs, action_mask, logits, means, log_stds, magnet_logits, magnet_means, magnet_log_stds,
        component, raw_action, action_kind, action_value, value,
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
    logits, means, log_stds, _ = network.apply(params, obs)
    action_mask = jnp.ones_like(logits, dtype=bool)

    def one(k: chex.PRNGKey) -> chex.Array:
        _, raw_action = sample_mixture_component(logits, means, log_stds, action_mask, 0, k)
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
    (
        obs, action_mask, logits, means, log_stds, magnet_logits, magnet_means, magnet_log_stds,
        component, raw_action, action_kind, action_value, value,
    ) = jax.vmap(_sample_mixture_one, in_axes=(None, None, None, None, None, 0))(
        game, perspective, network, params, magnet_params, keys
    )

    opponent_action = opponent_action_fn(opponent_key, num_envs)
    if perspective == 0:
        reward = game.payoff_batch(action_value, opponent_action)
    else:
        reward = -game.payoff_batch(opponent_action, action_value)

    return Episode(
        obs=obs, action_mask=action_mask, logits=logits, means=means, log_stds=log_stds,
        magnet_logits=magnet_logits, magnet_means=magnet_means, magnet_log_stds=magnet_log_stds,
        component=component, raw_action=raw_action, action_kind=action_kind,
        action_value=action_value, value=value, reward=reward,
    )


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
    (
        obs_1, action_mask_1, logits_1, means_1, log_stds_1,
        magnet_logits_1, magnet_means_1, magnet_log_stds_1,
        component_1, raw_action_1, action_kind_1, action_value_1, value_1,
    ) = _sample_mixture_one(game, 0, network_1, params_1, magnet_params_1, key_1)
    (
        obs_2, action_mask_2, logits_2, means_2, log_stds_2,
        magnet_logits_2, magnet_means_2, magnet_log_stds_2,
        component_2, raw_action_2, action_kind_2, action_value_2, value_2,
    ) = _sample_mixture_one(game, 1, network_2, params_2, magnet_params_2, key_2)

    reward = game.payoff(action_value_1, action_value_2)

    episode_1 = Episode(
        obs=obs_1, action_mask=action_mask_1, logits=logits_1, means=means_1, log_stds=log_stds_1,
        magnet_logits=magnet_logits_1, magnet_means=magnet_means_1, magnet_log_stds=magnet_log_stds_1,
        component=component_1, raw_action=raw_action_1, action_kind=action_kind_1,
        action_value=action_value_1, value=value_1, reward=reward,
    )
    episode_2 = Episode(
        obs=obs_2, action_mask=action_mask_2, logits=logits_2, means=means_2, log_stds=log_stds_2,
        magnet_logits=magnet_logits_2, magnet_means=magnet_means_2, magnet_log_stds=magnet_log_stds_2,
        component=component_2, raw_action=raw_action_2, action_kind=action_kind_2,
        action_value=action_value_2, value=value_2, reward=-reward,
    )
    return episode_1, episode_2


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
    log_stds: chex.Array,
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

    Atoms and legality masks both act by *zeroing* terms rather than by
    branching. For a sample that drew an atom, the Gaussian ratio, its clipped
    surrogate, both Gaussian KLs and the marginal-density entropy are all forced
    to `0.0`: an atom has no mean and no spread, so there is nothing there for
    those terms to say. Illegal categorical entries are handled inside
    `masked_log_softmax`/`categorical_kl` via `episode.action_mask`, the mask
    recorded at sampling time -- re-applying exactly that mask is what keeps the
    PPO ratio a ratio of two densities over the same support.

    The current policy's outputs are the *only* ones evaluated for this
    update -- `episode.logits`/`means`/`log_stds` (the sampling-time
    distribution) and `episode.magnet_logits`/`magnet_means`/`magnet_log_stds`
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
        episode.logits, episode.means, episode.log_stds, mask,
        episode.component, episode.raw_action, num_atoms,
    )
    new_category_log_prob, new_gaussian_log_prob = mixture_log_probs(
        logits, means, log_stds, mask, episode.component, episode.raw_action, num_atoms
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
        logits, means, log_stds, mask, episode.raw_action, num_atoms
    )
    entropy = category_entropy + action_entropy

    index = gaussian_component_index(episode.component, num_atoms)
    mean = means[index]
    log_std = log_stds[index]

    old_mean = episode.means[index]
    old_log_std = episode.log_stds[index]
    trpo_category_kl = categorical_kl(episode.logits, logits, mask)
    trpo_gaussian_kl = is_gaussian * gaussian_kl(old_mean, old_log_std, mean, log_std)

    magnet_mean = episode.magnet_means[index]
    magnet_log_std = episode.magnet_log_stds[index]
    magnet_category_kl = categorical_kl(logits, episode.magnet_logits, mask)
    magnet_gaussian_kl = is_gaussian * gaussian_kl(mean, log_std, magnet_mean, magnet_log_std)

    loss = (
        policy_loss
        + value_coef * value_loss
        - category_entropy_coef * category_entropy
        - gaussian_entropy_coef * action_entropy
        + trpo_category_kl_coef * trpo_category_kl
        + trpo_gaussian_kl_coef * trpo_gaussian_kl
        + magnet_category_kl_coef * magnet_category_kl
        + magnet_gaussian_kl_coef * magnet_gaussian_kl
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
    }
    return loss, metrics


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
) -> tuple[chex.Array, dict[str, chex.Array]]:
    """`mixture_ppo_loss_from_outputs`, forward-passing `params` at `episode.obs` first."""
    logits, means, log_stds, value_pred = network.apply(params, episode.obs)
    return mixture_ppo_loss_from_outputs(
        logits, means, log_stds, value_pred, network.num_atoms, episode, advantage,
        clip_eps, value_coef, category_entropy_coef, gaussian_entropy_coef,
        trpo_category_kl_coef, trpo_gaussian_kl_coef,
        magnet_category_kl_coef, magnet_gaussian_kl_coef,
    )


def build_mixture_ppo_loss_fn(
    category_entropy_coef: float,
    gaussian_entropy_coef: float,
    trpo_category_kl_coef: float,
    trpo_gaussian_kl_coef: float,
    magnet_category_kl_coef: float,
    magnet_gaussian_kl_coef: float,
    shared_obs: bool = False,
):
    """Batches `mixture_ppo_loss` over an `Episode`'s leading (env) axis.

    Binds the six per-head entropy/KL coefficients (constant for one
    `ppo_update` call) and returns a function matching `ppo_update`'s
    `loss_fn` contract: `(params, network, batch, clip_eps, value_coef,
    entropy_coef) -> (scalar_loss, dict_of_scalar_metrics)`. `entropy_coef`
    (the generic `PPOHyperparams` field `ppo_update` always passes) is
    accepted but unused -- the mixture loss is weighted by the two bound
    per-head coefficients instead. The batch-wide advantage normalization
    happens here, before the per-sample `vmap`; `batch` (an `Episode` whose
    fields all carry the leading env axis) is `vmap`ed as a single pytree
    argument rather than unpacked field by field.

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
    observations actually differ, hence off by default.
    """

    def loss_fn(
        params,
        network: MixtureActorCritic,
        batch: Episode,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        del entropy_coef

        advantage = batch.reward - batch.value
        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

        coefs = (
            category_entropy_coef, gaussian_entropy_coef,
            trpo_category_kl_coef, trpo_gaussian_kl_coef,
            magnet_category_kl_coef, magnet_gaussian_kl_coef,
        )
        if shared_obs:
            logits, means, log_stds, value_pred = network.apply(params, batch.obs[0])
            per_sample_loss, metrics = jax.vmap(
                mixture_ppo_loss_from_outputs,
                in_axes=(None, None, None, None, None, 0, 0, None, None, None, None, None, None, None, None),
            )(
                logits, means, log_stds, value_pred, network.num_atoms, batch, advantage,
                clip_eps, value_coef, *coefs,
            )
        else:
            per_sample_loss, metrics = jax.vmap(
                mixture_ppo_loss, in_axes=(None, None, 0, 0, None, None, None, None, None, None, None, None)
            )(params, network, batch, advantage, clip_eps, value_coef, *coefs)
        return jnp.mean(per_sample_loss), jax.tree_util.tree_map(jnp.mean, metrics)

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
    )
