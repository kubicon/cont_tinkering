"""An exponential-family (log-linear) actor-critic, as an alternative to `mixture.py`.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from games.base import ZeroSumGame
from nets import Activation, Normalization

from .config import ExpFamilyPPOHyperparams

OpponentActionFn = Callable[[chex.PRNGKey, int], chex.Array]


# ---- the fixed basis ------------------------------------------------------
@dataclasses.dataclass(frozen=True, eq=False)
class Basis:
    """The precomputed feature grid: everything about `phi` the loss ever needs.

    Constant for a whole run (the basis is fixed, only `theta` is learned), so it
    is built once and closed over by the network -- `features` is a device array
    of shape `(dim, grid_points, num_features)`, i.e. `phi` evaluated at every
    bin center of every action dimension.
    """

    centers: chex.Array   # (dim, grid_points) bin centers
    features: chex.Array  # (dim, grid_points, num_features)
    log_width: chex.Array  # (dim,) log of the bin width, the `-log w` in log p(a)

    @property
    def dim(self) -> int:
        return self.centers.shape[0]

    @property
    def grid_points(self) -> int:
        return self.centers.shape[1]

    @property
    def num_features(self) -> int:
        return self.features.shape[-1]


def build_basis(
    low: chex.Array,
    high: chex.Array,
    grid_points: int,
    num_basis: int,
    poly_order: int,
    basis_width_scale: float,
) -> Basis:
    """`num_basis` evenly spaced RBF bumps plus `poly_order` monomials, per dimension.

    The monomials are in the box-normalized coordinate `z in [-1, 1]`, and the
    constant term is deliberately absent: it is unidentifiable under the softmax
    that `log Z` performs. `z^1` alone makes the family an exponential tilt of
    the uniform density, which is the exact entropy-regularized best response to
    a payoff linear in the action (`ContinuousMatchingPennies`); the RBFs are
    what let it be multi-modal against a bumpy one (`MultiPointGame`).

    `basis_width_scale` is in units of the RBF spacing: 1.0 gives bumps that
    overlap at roughly half height, which keeps `theta` well conditioned.
    Narrower bumps buy sharper modes at the cost of a spikier landscape.
    """
    if grid_points < 2:
        raise ValueError(f"grid_points must be at least 2, got {grid_points}")
    if num_basis < 0 or poly_order < 0:
        raise ValueError(f"num_basis and poly_order must be non-negative, got {num_basis}, {poly_order}")
    if num_basis + poly_order == 0:
        raise ValueError("an empty basis leaves the policy uniform and untrainable")

    low = jnp.atleast_1d(jnp.asarray(low, dtype=jnp.float32))
    high = jnp.atleast_1d(jnp.asarray(high, dtype=jnp.float32))
    dim = low.shape[0]
    width = jnp.maximum(high - low, 1e-6)

    # Bin centers: the midpoints of `grid_points` equal bins spanning the box.
    fractions = (jnp.arange(grid_points, dtype=jnp.float32) + 0.5) / grid_points  # (G,)
    centers = low[:, None] + fractions[None, :] * width[:, None]                  # (dim, G)
    z = 2.0 * fractions - 1.0                                                     # (G,) in [-1, 1]

    blocks = []
    if poly_order > 0:
        powers = jnp.arange(1, poly_order + 1, dtype=jnp.float32)                 # (P,)
        monomials = z[:, None] ** powers[None, :]                                 # (G, P)
        blocks.append(jnp.broadcast_to(monomials, (dim, grid_points, poly_order)))
    if num_basis > 0:
        rbf_fracs = (jnp.arange(num_basis, dtype=jnp.float32) + 0.5) / num_basis  # (B,)
        rbf_z = 2.0 * rbf_fracs - 1.0                                             # (B,) in [-1, 1]
        sigma = basis_width_scale * (2.0 / num_basis)
        bumps = jnp.exp(-0.5 * jnp.square((z[:, None] - rbf_z[None, :]) / sigma))  # (G, B)
        blocks.append(jnp.broadcast_to(bumps, (dim, grid_points, num_basis)))

    return Basis(
        centers=centers,
        features=jnp.concatenate(blocks, axis=-1),
        log_width=jnp.log(width / grid_points),
    )


def _tilt_bias_init(num_features: int, poly_order: int, init_tilt: float) -> Callable:
    """Bias initializer for the `theta` head: zeros, except an `init_tilt` on `z^1`.

    Zeros make the initial policy exactly uniform on the box -- the maximum
    entropy start, and the honest analogue of "no prior information". That is
    also, in a bilinear game, *already a Nash* (only the mean matters, and the
    uniform's mean is the box midpoint), so a run started there sits at a fixed
    point and reports convergence without having taken a step. `init_tilt`
    pushes the initial density off center by an exponential tilt `p ~ exp(t z)`
    -- the log-linear counterpart of the `idealized.init_means` the Gaussian
    experiments set for exactly this reason.

    The head's output is the flat `(dim * num_features,)` view of `theta`, so
    the `z^1` coefficient of dimension `i` sits at `i * num_features` (the
    monomial block is laid down first; see `build_basis`).
    """
    if init_tilt != 0.0 and poly_order < 1:
        raise ValueError("network.init_tilt needs network.poly_order >= 1 (it tilts the z^1 feature)")

    def init_fn(key: chex.PRNGKey, shape: tuple[int, ...], dtype=jnp.float32) -> chex.Array:
        del key
        flat = jnp.zeros(shape, dtype=dtype).reshape(-1)
        if init_tilt != 0.0:
            flat = flat.at[jnp.arange(flat.shape[0] // num_features) * num_features].set(init_tilt)
        return flat.reshape(shape)

    return init_fn


class ExpFamilyActorCritic(nn.Module):
    """Shared torso, two linear heads: the natural parameters `theta`, and the value.

    The `theta` head is initialized to zeros (kernel *and*, absent `init_tilt`,
    bias), so the policy starts uniform on the action box and every subsequent
    mode is something training put there -- as against `MixtureActorCritic`,
    whose components must be spread across the box by hand at init because a
    Gaussian mixture has no uniform-like starting point.
    """

    hidden_dims: tuple[int, ...]
    basis: Basis
    poly_order: int
    activation: str = "tanh"
    normalization: str = "none"
    init_tilt: float = 0.0

    @nn.compact
    def __call__(self, obs: chex.Array, train: bool = False) -> tuple[chex.Array, chex.Array]:
        """`(theta, value)` with `theta` of shape `(dim, num_features)`."""
        torso = obs
        for width in self.hidden_dims:
            torso = nn.Dense(width)(torso)
            torso = Normalization(kind=self.normalization)(torso, use_running_average=not train)
            torso = Activation(kind=self.activation)(torso)

        dim, num_features = self.basis.dim, self.basis.num_features
        theta_flat = nn.Dense(
            dim * num_features,
            kernel_init=nn.initializers.zeros,
            bias_init=_tilt_bias_init(num_features, self.poly_order, self.init_tilt),
            name="theta_head",
        )(torso)
        theta = theta_flat.reshape(dim, num_features)

        value = nn.Dense(1, name="value_head")(torso)
        return theta, jnp.squeeze(value, axis=-1)


# ---- the density, on the grid ---------------------------------------------
def grid_log_probs(theta: chex.Array, basis: Basis) -> chex.Array:
    """`(dim, grid_points)` log bin probabilities `log pi = log_softmax(theta . phi)`.

    Every other quantity in this module is a function of this one array, which
    is the whole reason the exponential family is cheap here: one
    `(dim, num_features) x (dim, num_features, grid_points)` contraction and a
    `log_softmax` gives the exact normalizer, and with it the exact log-prob,
    entropy and KL.
    """
    logits = jnp.einsum("df,dgf->dg", theta, basis.features)
    return jax.nn.log_softmax(logits, axis=-1)


def density_log_prob(theta: chex.Array, basis: Basis, bin_index: chex.Array) -> chex.Array:
    """`log p(a)` for the action that landed in `bin_index` -- a scalar.

    The bin index, not the action, is what is scored: the density is constant
    inside a bin, so `log p(a) = log pi_g - log w` exactly, and recording the
    index spares the loss any re-quantization of a stored float.
    """
    log_pi = grid_log_probs(theta, basis)
    picked = jnp.take_along_axis(log_pi, bin_index[:, None], axis=-1)[:, 0]  # (dim,)
    return jnp.sum(picked - basis.log_width)


def density_entropy(theta: chex.Array, basis: Basis) -> chex.Array:
    """Exact differential entropy of the piecewise-constant policy.

    `H = sum_i [ H(pi_i) + log w_i ]`. Exact, where `mixture.py` has to settle
    for `-log p_mix(a)` at the sampled action (an unbiased but noisy estimate)
    or the sum of component entropies (a bound).
    """
    log_pi = grid_log_probs(theta, basis)
    return jnp.sum(-jnp.sum(jnp.exp(log_pi) * log_pi, axis=-1) + basis.log_width)


def density_kl(theta_p: chex.Array, theta_q: chex.Array, basis: Basis) -> chex.Array:
    """Exact `KL(p || q)`. The bin widths cancel, so it is the discrete KL of the bins."""
    log_p = grid_log_probs(theta_p, basis)
    log_q = grid_log_probs(theta_q, basis)
    return jnp.sum(jnp.exp(log_p) * (log_p - log_q))


def density_moments(theta: chex.Array, basis: Basis) -> tuple[chex.Array, chex.Array]:
    """`(mean, std)` per action dimension -- for logging, not for the loss."""
    probs = jnp.exp(grid_log_probs(theta, basis))
    mean = jnp.sum(probs * basis.centers, axis=-1)
    var = jnp.sum(probs * jnp.square(basis.centers - mean[:, None]), axis=-1)
    return mean, jnp.sqrt(var)


def sample_action(theta: chex.Array, basis: Basis, key: chex.PRNGKey) -> tuple[chex.Array, chex.Array]:
    """Draw one action: a bin per dimension, then uniform inside it.

    Returns `(bin_index, action)`. The uniform jitter is what makes the sample a
    draw from the piecewise-constant *density* rather than from a lattice, and
    it costs nothing: the log-prob does not depend on where in the bin it fell.
    """
    bin_key, jitter_key = jax.random.split(key)
    log_pi = grid_log_probs(theta, basis)
    bin_index = jax.random.categorical(bin_key, log_pi, axis=-1).astype(jnp.int32)  # (dim,)
    center = jnp.take_along_axis(basis.centers, bin_index[:, None], axis=-1)[:, 0]
    half_width = 0.5 * jnp.exp(basis.log_width)
    jitter = jax.random.uniform(jitter_key, center.shape, minval=-1.0, maxval=1.0)
    return bin_index, center + half_width * jitter


def sample_expfam_actions(
    network: ExpFamilyActorCritic, params, obs: chex.Array, key: chex.PRNGKey, num_samples: int
) -> chex.Array:
    """`num_samples` actions from the policy at a single `obs` -- the exploitability input.

    No clipping: every bin center lies inside the box and the jitter stays
    within half a bin of it, so a sample is in the box by construction. That is
    the support mismatch the Gaussian version has to paper over with
    `space.clip`.
    """
    theta, _ = network.apply(params, obs)

    def one(k: chex.PRNGKey) -> chex.Array:
        _, action = sample_action(theta, network.basis, k)
        return action

    return jax.vmap(one)(jax.random.split(key, num_samples))


# ---- rollout ---------------------------------------------------------------
@chex.dataclass
class ExpFamilyEpisode:
    """One batch of one-shot decisions from an `ExpFamilyActorCritic`.

    Leaner than `mixture.Episode` because a one-shot game gives the policy
    nothing to keep track of: no `actor` (every row is the same player's single
    decision), no `action_mask` (nothing is illegal), no `component` (there is
    no latent). `theta` is the sampling-time policy and `magnet_theta` the
    magnet snapshot evaluated at the same `obs`, exactly as `Episode` records
    both for the mixture.
    """

    obs: chex.Array
    theta: chex.Array         # (dim, num_features) at sample time
    magnet_theta: chex.Array  # (dim, num_features) under `magnet_params`, same obs
    bin_index: chex.Array     # (dim,) int32, which bin was drawn
    action: chex.Array        # (dim,) the action played -- already inside the box
    value: chex.Array
    reward: chex.Array


def _sample_one(
    game: ZeroSumGame, player: int, network: ExpFamilyActorCritic, params, magnet_params, key: chex.PRNGKey
) -> ExpFamilyEpisode:
    """One (unbatched) sample, `reward` left zero for the caller to fill in."""
    obs_key, sample_key = jax.random.split(key)
    obs = game.observation(player, obs_key)

    theta, value = network.apply(params, obs)
    magnet_theta, _ = network.apply(magnet_params, obs)
    bin_index, action = sample_action(theta, network.basis, sample_key)

    return ExpFamilyEpisode(
        obs=obs, theta=theta, magnet_theta=magnet_theta,
        bin_index=bin_index, action=action, value=value, reward=jnp.zeros(()),
    )


def collect_expfam_episode(
    game: ZeroSumGame,
    network: ExpFamilyActorCritic,
    params,
    magnet_params,
    opponent_action_fn: OpponentActionFn,
    key: chex.PRNGKey,
    num_envs: int,
    perspective: int = 0,
) -> ExpFamilyEpisode:
    """`collect_mixture_episode`'s counterpart: `num_envs` samples against a fixed opponent."""
    if perspective not in (0, 1):
        raise ValueError(f"perspective must be 0 or 1, got {perspective}")

    own_key, opponent_key = jax.random.split(key)
    keys = jax.random.split(own_key, num_envs)
    episode = jax.vmap(_sample_one, in_axes=(None, None, None, None, None, 0))(
        game, perspective, network, params, magnet_params, keys
    )

    opponent_action = opponent_action_fn(opponent_key, num_envs)
    if perspective == 0:
        reward = game.payoff_batch(episode.action, opponent_action)
    else:
        reward = -game.payoff_batch(opponent_action, episode.action)
    return episode.replace(reward=reward)


def _sample_self_play_one(
    game: ZeroSumGame,
    network_1: ExpFamilyActorCritic, params_1, magnet_params_1,
    network_2: ExpFamilyActorCritic, params_2, magnet_params_2,
    key: chex.PRNGKey,
) -> tuple[ExpFamilyEpisode, ExpFamilyEpisode]:
    key_1, key_2 = jax.random.split(key)
    episode_1 = _sample_one(game, 0, network_1, params_1, magnet_params_1, key_1)
    episode_2 = _sample_one(game, 1, network_2, params_2, magnet_params_2, key_2)
    reward = game.payoff(episode_1.action, episode_2.action)
    return episode_1.replace(reward=reward), episode_2.replace(reward=-reward)


def collect_expfam_self_play_episode(
    game: ZeroSumGame,
    network_1: ExpFamilyActorCritic, params_1, magnet_params_1,
    network_2: ExpFamilyActorCritic, params_2, magnet_params_2,
    key: chex.PRNGKey,
    num_envs: int,
) -> tuple[ExpFamilyEpisode, ExpFamilyEpisode]:
    keys = jax.random.split(key, num_envs)
    return jax.vmap(_sample_self_play_one, in_axes=(None, None, None, None, None, None, None, 0))(
        game, network_1, params_1, magnet_params_1, network_2, params_2, magnet_params_2, keys
    )


# ---- loss ------------------------------------------------------------------
def expfam_ppo_loss_from_outputs(
    theta: chex.Array,
    value_pred: chex.Array,
    basis: Basis,
    episode: ExpFamilyEpisode,
    advantage: chex.Array,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    trpo_kl_coef: float,
    magnet_kl_coef: float,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    """Clipped-surrogate PPO loss plus the two KL penalties, for one sample.

    Structurally `mixture_ppo_loss_from_outputs` with the two-factor split
    collapsed: there is one density, so there is one ratio, one entropy term and
    one of each KL. The three regularizers are the exact quantities described in
    the module docstring rather than the mixture's estimates, which is the whole
    experimental point -- `magnet_kl` in particular is now genuinely
    `KL(current || magnet)` rather than a componentwise bound on it.

    `advantage` arrives already normalized over the batch (a batch statistic, so
    it cannot be computed inside the per-sample `vmap`), matching the mixture
    loss's contract.
    """
    old_log_prob = density_log_prob(episode.theta, basis, episode.bin_index)
    new_log_prob = density_log_prob(theta, basis, episode.bin_index)
    ratio = jnp.exp(new_log_prob - old_log_prob)

    unclipped = ratio * advantage
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy_loss = -jnp.minimum(unclipped, clipped)

    value_loss = jnp.square(value_pred - episode.reward)
    entropy = density_entropy(theta, basis)
    trpo_kl = density_kl(episode.theta, theta, basis)          # KL(old || current)
    magnet_kl = density_kl(theta, episode.magnet_theta, basis)  # KL(current || magnet)

    loss = (
        policy_loss
        + value_coef * value_loss
        - entropy_coef * entropy
        + trpo_kl_coef * trpo_kl
        + magnet_kl_coef * magnet_kl
    )

    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": old_log_prob - new_log_prob,
        "clip_frac": (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32),
        "trpo_kl": trpo_kl,
        "magnet_kl": magnet_kl,
    }
    return loss, metrics


def build_expfam_ppo_loss_fn(
    entropy_coef: float,
    trpo_kl_coef: float,
    magnet_kl_coef: float,
    shared_obs: bool = False,
):
    """The whole-batch loss, matching `ppo_update`'s `loss_fn` contract.

    `shared_obs` lifts the forward pass out of the `vmap` for a game whose
    observation is a constant (`ZeroSumGame.constant_observation`) -- the same
    saving, and the same silent wrongness if the observations really differ, as
    `build_mixture_ppo_loss_fn`'s flag of that name.

    The generic `entropy_coef` that `ppo_update` passes from `PPOHyperparams` is
    accepted and discarded; the bound coefficients above are used instead, so
    that the exponential-family run reads its weights from the same
    `density_*_coef` fields a config sets.
    """
    coefs = (entropy_coef, trpo_kl_coef, magnet_kl_coef)

    def loss_fn(
        params,
        network: ExpFamilyActorCritic,
        batch: ExpFamilyEpisode,
        clip_eps: float,
        value_coef: float,
        entropy_coef_unused: float,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        del entropy_coef_unused
        raw_advantage = batch.reward - batch.value
        advantage = (raw_advantage - jnp.mean(raw_advantage)) / (jnp.std(raw_advantage) + 1e-8)
        basis = network.basis

        if shared_obs:
            theta, value_pred = network.apply(params, batch.obs[0])
            per_sample_loss, metrics = jax.vmap(
                expfam_ppo_loss_from_outputs,
                in_axes=(None, None, None, 0, 0, None, None, None, None, None),
            )(theta, value_pred, basis, batch, advantage, clip_eps, value_coef, *coefs)
        else:
            def one(sample: ExpFamilyEpisode, adv: chex.Array):
                theta, value_pred = network.apply(params, sample.obs)
                return expfam_ppo_loss_from_outputs(
                    theta, value_pred, basis, sample, adv, clip_eps, value_coef, *coefs
                )

            per_sample_loss, metrics = jax.vmap(one)(batch, advantage)

        loss = jnp.mean(per_sample_loss)
        return loss, jax.tree_util.tree_map(jnp.mean, metrics)

    return loss_fn


def build_expfam_network(hyperparams: ExpFamilyPPOHyperparams) -> ExpFamilyActorCritic:
    basis = build_basis(
        low=jnp.asarray(hyperparams.low, dtype=jnp.float32),
        high=jnp.asarray(hyperparams.high, dtype=jnp.float32),
        grid_points=hyperparams.grid_points,
        num_basis=hyperparams.num_basis,
        poly_order=hyperparams.poly_order,
        basis_width_scale=hyperparams.basis_width_scale,
    )
    return ExpFamilyActorCritic(
        hidden_dims=hyperparams.hidden_dims,
        basis=basis,
        poly_order=hyperparams.poly_order,
        activation=hyperparams.activation,
        normalization=hyperparams.normalization,
        init_tilt=hyperparams.init_tilt,
    )
