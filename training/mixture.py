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
"""

from __future__ import annotations

from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from games.base import ZeroSumGame
from nets import Activation, Normalization

from .actor_critic import categorical_kl, gaussian_kl, gaussian_log_prob
from .config import MixturePPOHyperparams

OpponentActionFn = Callable[[chex.PRNGKey, int], chex.Array]

# Lower bound on each component's log-std, applied in `MixtureActorCritic.__call__`.
# The upper bound is `log(high - low)` (a std as wide as the whole action range),
# computed there since it is per-action-dimension.
#
# Both bounds are load-bearing, not defensive padding:
#   * ceiling -- the Gaussian entropy bonus in `mixture_ppo_loss` is
#     `-gaussian_entropy_coef * (-log p(a))` on the *marginal* mixture density, and
#     `-log p(a)` is unbounded above as std -> inf. So the bonus has no interior
#     optimum in the std: any `gaussian_entropy_coef` large enough to matter drives
#     log_std to +inf and the run NaNs (it does, reliably, above ~0.5).
#   * floor -- as std -> 0 the density (hence `gaussian_log_prob` and
#     `mixture_marginal_log_prob`) blows up, which NaNs the loss from the other side.
#     This mirrors the `[log 1e-3, log 1]` clip that `idealized_mmd.run` already applies.
LOG_STD_MIN = -6.907755  # log(1e-3)


def _spread_bias_init(low: chex.Array, high: chex.Array, num_components: int) -> Callable:
    """Bias initializer spreading each component's mean evenly across `[low, high]`.

    Component `k` (of `num_components`, 0-indexed) starts at fraction
    `(k + 0.5) / num_components` of the way from `low` to `high`. For
    `num_components=2` and `low=0, high=1` that's `[0.25, 0.75]` — distinct
    starting points so components don't collapse onto the same mean before
    training has a chance to separate them.
    """

    def init_fn(key: chex.PRNGKey, shape: tuple[int, ...], dtype=jnp.float32) -> chex.Array:
        del key
        fractions = (jnp.arange(num_components, dtype=dtype) + 0.5) / num_components
        values = low[None, :] + fractions[:, None] * (high - low)[None, :]
        return values.reshape(shape).astype(dtype)

    return init_fn


def _std_bias_init(low: chex.Array, high: chex.Array, num_components: int) -> Callable:
    """Bias initializer for `log_std_head`, scaled to the action range.

    A zero-initialized log-std means each component starts at `std = 1`,
    which is far too wide for a bounded action space (for `[0, 1]` almost
    all mass falls outside the bounds and clips to the endpoints, and the
    components' spread means -- see `_spread_bias_init` -- are swamped, so
    the modes are indistinguishable early and separate only slowly). This
    instead starts each component at `std = (high - low) / (2 *
    num_components)`, i.e. roughly half the spacing between adjacent
    component means, so the modes are distinct from the first step. Paired
    with a zero kernel on the head, the initial log-std is exactly this
    (obs-independent), matching how `means_head` is initialized.
    """

    def init_fn(key: chex.PRNGKey, shape: tuple[int, ...], dtype=jnp.float32) -> chex.Array:
        del key
        std = (high - low) / (2 * num_components)
        log_std = jnp.log(std).astype(dtype)  # (action_dim,)
        values = jnp.broadcast_to(log_std[None, :], (num_components, log_std.shape[0]))
        return values.reshape(shape).astype(dtype)

    return init_fn


class MixtureActorCritic(nn.Module):
    """Shared torso, four linear heads: component logits, means, log-stds, value.

    The torso (`hidden_dims` Dense -> normalization -> activation layers) is
    computed once and shared; `logits`, `means`, `log_std`, and `value` are
    each a single `nn.Dense` projection of that shared representation,
    rather than four separate networks — cheaper, and lets the heads share
    whatever features the torso learns.

    `__call__` returns `(logits, means, log_std, value)`:
      - `logits`: `(num_components,)`, the categorical distribution over components
      - `means`: `(num_components, action_dim)`, each component's Gaussian mean
      - `log_std`: `(num_components, action_dim)`, each component's own spread
      - `value`: scalar state-value estimate
    """

    action_dim: int
    num_components: int
    hidden_dims: tuple[int, ...]
    low: chex.Array
    high: chex.Array
    activation: str = "tanh"
    normalization: str = "none"

    @nn.compact
    def __call__(
        self, obs: chex.Array, train: bool = False
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        torso = obs
        for dim in self.hidden_dims:
            torso = nn.Dense(dim)(torso)
            torso = Normalization(kind=self.normalization)(torso, use_running_average=not train)
            torso = Activation(kind=self.activation)(torso)

        logits = nn.Dense(self.num_components, name="logits_head")(torso)

        means_flat = nn.Dense(
            self.num_components * self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=_spread_bias_init(self.low, self.high, self.num_components),
            name="means_head",
        )(torso)
        means = means_flat.reshape(self.num_components, self.action_dim)

        log_std_flat = nn.Dense(
            self.num_components * self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=_std_bias_init(self.low, self.high, self.num_components),
            name="log_std_head",
        )(torso)
        log_std = log_std_flat.reshape(self.num_components, self.action_dim)
        log_std = jnp.clip(log_std, LOG_STD_MIN, jnp.log(self.high - self.low))

        value = nn.Dense(1, name="value_head")(torso)

        return logits, means, log_std, jnp.squeeze(value, axis=-1)


def mixture_log_probs(
    logits: chex.Array, means: chex.Array, log_stds: chex.Array, component: chex.Array, raw_action: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """`(category_log_prob, gaussian_log_prob)` for one sample -- kept as separate factors.

    The component choice is treated as part of the stochastic action (the
    same way hybrid discrete+continuous action spaces are usually handled
    in PPO): the environment's payoff never sees `component`, but the
    log-probs of the actual sampling path (component, then Gaussian) are
    what make the PPO importance ratios well-defined. Returning them
    separately (rather than summed into one joint log-prob) lets each head
    get its own PPO ratio and clip in `mixture_ppo_loss`, instead of only
    the product of the two being bounded.
    """
    category_log_prob = jax.nn.log_softmax(logits)[component]
    mean = means[component]
    log_std = log_stds[component]
    return category_log_prob, gaussian_log_prob(raw_action, mean, log_std)


def _categorical_entropy(logits: chex.Array) -> chex.Array:
    log_p = jax.nn.log_softmax(logits)
    return -jnp.sum(jnp.exp(log_p) * log_p)


def mixture_marginal_log_prob(
    logits: chex.Array, means: chex.Array, log_stds: chex.Array, raw_action: chex.Array
) -> chex.Array:
    """`log p(a)` of the marginal mixture density at a single `raw_action`.

    Marginalizes the component out: `log sum_k softmax(logits)_k * N(a;
    means_k, std_k)`, computed stably with `logsumexp`. Used to estimate the
    action policy's entropy as `-log p(a)` (single-sample Monte Carlo).

    Unlike the average of the per-component Gaussian entropies (`E_k[H(a |
    component=k)]`), this depends on how far apart the component means are:
    when the modes collapse onto the same mean the marginal is a single
    Gaussian with the lowest possible entropy, and pulling them apart raises
    it. Maximizing it therefore rewards keeping the modes separated, whereas
    the joint entropy `H(component) + E_k[H(a | k)]` is maximized by a
    uniform categorical no matter where the means sit -- so components could
    (and, with permutation symmetry, would tend to) collapse together while
    the entropy metric still looked healthy.
    """
    log_weights = jax.nn.log_softmax(logits)  # (num_components,)
    per_component = jax.vmap(gaussian_log_prob, in_axes=(None, 0, 0))(raw_action, means, log_stds)
    return jax.nn.logsumexp(log_weights + per_component)


@chex.dataclass
class Episode:
    """Everything recorded while sampling one batch of transitions from a `MixtureActorCritic`.

    Storing the full sampling-time distribution (`logits`/`means`/`log_stds`,
    not just the log-prob of the action actually taken), and the magnet
    snapshot's distribution at that same `obs` (`magnet_logits`/
    `magnet_means`/`magnet_log_stds`), means both reference policies
    `mixture_ppo_loss` regularizes against are exactly what's in here
    already. `mixture_ppo_loss` only ever forward-passes `params` itself --
    no `old_params`/`magnet_params` need to be carried around or re-applied
    during the loss; everything else is data captured once, at rollout time.
    """

    obs: chex.Array
    logits: chex.Array  # (num_components,) categorical logits at sample time
    means: chex.Array  # (num_components, action_dim) each component's mean at sample time
    log_stds: chex.Array  # (num_components, action_dim) each component's log-std at sample time
    magnet_logits: chex.Array  # (num_components,) categorical logits under `magnet_params`, same obs
    magnet_means: chex.Array  # (num_components, action_dim) under `magnet_params`
    magnet_log_stds: chex.Array  # (num_components, action_dim) under `magnet_params`
    component: chex.Array  # which categorical component was sampled
    raw_action: chex.Array  # unclipped Gaussian sample from that component
    action: chex.Array  # `raw_action` clipped to the action space; used to play the game
    value: chex.Array
    reward: chex.Array


def _sample_mixture_one(
    game: ZeroSumGame, player: int, network: MixtureActorCritic, params, magnet_params, key: chex.PRNGKey
):
    """One (unbatched) sample: draws `obs`, samples a component/action, and evaluates both
    `params` and `magnet_params` at that `obs`. Everything is keyed off the single `key`
    passed in -- `vmap` this over `num_envs` keys to draw a whole batch (see
    `collect_mixture_episode`/`collect_mixture_self_play_episode`).
    """
    obs_key, component_key, noise_key = jax.random.split(key, 3)
    space = game.action_space(player)
    obs = game.observation(player, obs_key)

    logits, means, log_stds, value = network.apply(params, obs)
    magnet_logits, magnet_means, magnet_log_stds, _ = network.apply(magnet_params, obs)

    component = jax.random.categorical(component_key, logits)
    mean = means[component]
    log_std = log_stds[component]
    raw_action = mean + jnp.exp(log_std) * jax.random.normal(noise_key, mean.shape)
    action = space.clip(raw_action)

    return (
        obs, logits, means, log_stds, magnet_logits, magnet_means, magnet_log_stds, component, raw_action, action, value,
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

    A Monte-Carlo picture of the policy's mixed strategy (component draw
    then Gaussian, clipped to the space), used to estimate the mixture's
    exploitability -- see `ZeroSumGame.mixture_exploitability`.
    """
    logits, means, log_stds, _ = network.apply(params, obs)

    def one(k: chex.PRNGKey) -> chex.Array:
        component_key, noise_key = jax.random.split(k)
        component = jax.random.categorical(component_key, logits)
        mean = means[component]
        raw_action = mean + jnp.exp(log_stds[component]) * jax.random.normal(noise_key, mean.shape)
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
        obs, logits, means, log_stds, magnet_logits, magnet_means, magnet_log_stds, component, raw_action, action, value,
    ) = jax.vmap(_sample_mixture_one, in_axes=(None, None, None, None, None, 0))(
        game, perspective, network, params, magnet_params, keys
    )

    opponent_action = opponent_action_fn(opponent_key, num_envs)
    if perspective == 0:
        reward = game.payoff_batch(action, opponent_action)
    else:
        reward = -game.payoff_batch(opponent_action, action)

    return Episode(
        obs=obs, logits=logits, means=means, log_stds=log_stds,
        magnet_logits=magnet_logits, magnet_means=magnet_means, magnet_log_stds=magnet_log_stds,
        component=component, raw_action=raw_action, action=action, value=value, reward=reward,
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
        obs_1, logits_1, means_1, log_stds_1, magnet_logits_1, magnet_means_1, magnet_log_stds_1,
        component_1, raw_action_1, action_1, value_1,
    ) = _sample_mixture_one(game, 0, network_1, params_1, magnet_params_1, key_1)
    (
        obs_2, logits_2, means_2, log_stds_2, magnet_logits_2, magnet_means_2, magnet_log_stds_2,
        component_2, raw_action_2, action_2, value_2,
    ) = _sample_mixture_one(game, 1, network_2, params_2, magnet_params_2, key_2)

    reward = game.payoff(action_1, action_2)

    episode_1 = Episode(
        obs=obs_1, logits=logits_1, means=means_1, log_stds=log_stds_1,
        magnet_logits=magnet_logits_1, magnet_means=magnet_means_1, magnet_log_stds=magnet_log_stds_1,
        component=component_1, raw_action=raw_action_1, action=action_1, value=value_1, reward=reward,
    )
    episode_2 = Episode(
        obs=obs_2, logits=logits_2, means=means_2, log_stds=log_stds_2,
        magnet_logits=magnet_logits_2, magnet_means=magnet_means_2, magnet_log_stds=magnet_log_stds_2,
        component=component_2, raw_action=raw_action_2, action=action_2, value=value_2, reward=-reward,
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
    """Clipped-surrogate PPO loss plus KL penalties, for a single (unbatched) `Episode`.

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

    `params` is the *only* parameter set this function forward-passes
    through -- `episode.logits`/`means`/`log_stds` (the sampling-time
    distribution) and `episode.magnet_logits`/`magnet_means`/`magnet_log_stds`
    (the magnet snapshot's distribution, at the same `episode.obs`) come
    straight from the episode rather than being recomputed from
    `old_params`/`magnet_params` here; see `collect_mixture_episode`.
    `trpo_*_kl_coef * KL(old || current)` is a TRPO-style trust region
    against the policy this update's rollout was collected with;
    `magnet_*_kl_coef * KL(current || magnet)` pulls towards the
    periodically-snapshotted magnet policy.
    """
    logits, means, log_stds, value_pred = network.apply(params, episode.obs)

    old_category_log_prob, old_gaussian_log_prob = mixture_log_probs(
        episode.logits, episode.means, episode.log_stds, episode.component, episode.raw_action
    )
    new_category_log_prob, new_gaussian_log_prob = mixture_log_probs(
        logits, means, log_stds, episode.component, episode.raw_action
    )
    category_ratio = jnp.exp(new_category_log_prob - old_category_log_prob)
    gaussian_ratio = jnp.exp(new_gaussian_log_prob - old_gaussian_log_prob)

    def clipped_surrogate(ratio: chex.Array) -> chex.Array:
        unclipped = ratio * advantage
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        return jnp.minimum(unclipped, clipped)

    category_policy_loss = -clipped_surrogate(category_ratio)
    gaussian_policy_loss = -clipped_surrogate(gaussian_ratio)
    policy_loss = category_policy_loss + gaussian_policy_loss

    value_loss = jnp.square(value_pred - episode.reward)

    category_entropy = _categorical_entropy(logits)
    action_entropy = -mixture_marginal_log_prob(logits, means, log_stds, episode.raw_action)
    entropy = category_entropy + action_entropy

    mean = means[episode.component]
    log_std = log_stds[episode.component]

    old_mean = episode.means[episode.component]
    old_log_std = episode.log_stds[episode.component]
    trpo_category_kl = categorical_kl(episode.logits, logits)
    trpo_gaussian_kl = gaussian_kl(old_mean, old_log_std, mean, log_std)

    magnet_mean = episode.magnet_means[episode.component]
    magnet_log_std = episode.magnet_log_stds[episode.component]
    magnet_category_kl = categorical_kl(logits, episode.magnet_logits)
    magnet_gaussian_kl = gaussian_kl(mean, log_std, magnet_mean, magnet_log_std)

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
    gaussian_approx_kl = old_gaussian_log_prob - new_gaussian_log_prob
    category_clip_frac = (jnp.abs(category_ratio - 1.0) > clip_eps).astype(jnp.float32)
    gaussian_clip_frac = (jnp.abs(gaussian_ratio - 1.0) > clip_eps).astype(jnp.float32)

    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "category_policy_loss": category_policy_loss,
        "gaussian_policy_loss": gaussian_policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "category_entropy": category_entropy,
        "gaussian_entropy": action_entropy,  # marginal mixture entropy estimate (weighted by gaussian_entropy_coef)
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


def build_mixture_ppo_loss_fn(
    category_entropy_coef: float,
    gaussian_entropy_coef: float,
    trpo_category_kl_coef: float,
    trpo_gaussian_kl_coef: float,
    magnet_category_kl_coef: float,
    magnet_gaussian_kl_coef: float,
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

        per_sample_loss, metrics = jax.vmap(
            mixture_ppo_loss, in_axes=(None, None, 0, 0, None, None, None, None, None, None, None, None)
        )(
            params, network, batch, advantage, clip_eps, value_coef,
            category_entropy_coef, gaussian_entropy_coef,
            trpo_category_kl_coef, trpo_gaussian_kl_coef,
            magnet_category_kl_coef, magnet_gaussian_kl_coef,
        )
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
    )
