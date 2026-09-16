"""Randomized policy networks trained by a zeroth-order simultaneous pseudo-gradient.

Martin & Sandholm's method for continuous-action games (IJCAI 2023,
arXiv:2211.15936). Two ideas, and they are separable:

**The representation.** A mixed strategy is an *implicit* generative policy
`a = f_theta(o, z)` with `z ~ N(0, I)`: feed noise in alongside the observation and read
an action out. It represents arbitrary continuous distributions with a fixed parameter
count -- no components to choose as in a Gaussian mixture, no cells as in a discretized
head, and no restriction to a family at all. For the box games here the output is
squashed into the action box with a scaled `tanh` (their Blotto and auction experiments
use a softmax and an absolute value respectively, for the same reason: put the support
where the game's actions live).

**The update.** Because the policy is implicit it has *no tractable density*, so PPO's
importance ratio and any KL-to-a-magnet term are unavailable -- there is no `log pi` to
evaluate. Their answer is to stop differentiating the policy at all and estimate the
gradient of each player's utility with respect to its own parameters by Gaussian
smoothing:

    grad_i u_i(theta)  ~  [u_i(theta_i + sigma z_i, theta_-i) - u_i(theta_i - sigma z_i, theta_-i)] z_i / (2 sigma)

the central-difference (antithetic) pseudo-gradient -- hence *simultaneous
pseudo-gradient*, and hence `2n` utility evaluations per perturbation for `n` players.
`jpspg.py` is the follow-up that makes that count constant.

**How many perturbations.** `perturbation_batch` of them, averaged, and the number is
not a detail. A single draw `z` is a *random direction* in a `d`-dimensional parameter
space, so it is nearly orthogonal to the gradient it is estimating: at the `d ~ 5e3` of a
64x64 policy, one antithetic pair has cosine `~0.02` with the true pseudo-gradient, and
the resulting step is ~99% noise. Both papers average -- IJCAI'25 runs `batch_size = 256`
-- and their published reference implementation's `pgrad(f, scale, batch_size=2)` default
is one pair, i.e. *not* the setting their experiments use. Running at 2 is the reason
this pair of baselines did not converge in `experiments/one_shot_neural/`; it is a
budget-allocation error, not a bug in the estimator. Note that the two papers pair `N`
with different optimizers, and the pairing is the point: IJCAI'23 takes `N = 1` and
answers it with plain SGD at `lr = 1e-6` for ~1e8 steps (the *trajectory* averages the
noise), while IJCAI'25 takes `batch = 256` and answers it with AdaBelief at `1e-4`.
`N = 1` *with* AdaBelief is the one combination neither paper runs: an adaptive optimizer
rescales each coordinate by its own believed deviation, so a pure-noise gradient does not
shrink -- it becomes a random walk at the full learning rate.

This is the honest counterweight to the mixture head: it gives up the density (and with
it the magnet, the trust region, and every regularizer this repo's method is built on)
and buys unrestricted expressiveness. Running both answers the question the paper cannot
ask itself -- whether the restriction we accept costs anything the freedom does not.

The dynamics wrapped around the estimator are the ones their appendix runs
(`--dynamics simultaneous|extragradient|optimistic`); they report simultaneous ascent
sufficing on their benchmarks, which is worth checking on the games in this repo where
it does not.

Usage:
    python -m baselines.neural.randomized_policy configs/two_point.yaml --iters 20000
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from games.base import ZeroSumGame
from nets.activations import Activation
from training.optimizers import build_optimizer

from ..common import ChunkRunner, GridOracle, StrategyPair, action_bounds
from . import br_oracle as bo
from .common import RunWriter, empirical_strategy, load_run, neural_parser, print_row, \
    report_final, strategy_row

PseudoGradientFn = Callable[..., tuple]


@dataclasses.dataclass(frozen=True)
class RandomizedPolicyHyperparams:
    """Everything needed to rebuild the policy and reproduce the run."""

    action_dim: int
    hidden_dims: tuple[int, ...]
    noise_dim: int = 8
    activation: str = "tanh"
    low: tuple[float, ...] = (0.0,)
    high: tuple[float, ...] = (1.0,)
    # `tanh` squashes into the box; `wrap` takes the output modulo the box, for periodic
    # games (`circle`), where a squash turns the box's faces into walls the cyclic dynamics
    # pile every sample against.
    squash: str = "tanh"
    # Optimization. The paper's settings: AdaBelief, alpha = 1e-4, sigma = 0.1.
    learning_rate: float = 1e-4
    optimizer: str = "adabelief"
    max_grad_norm: float = 0.0        # 0 disables clipping; the pseudo-gradient is noisy, not large
    sigma: float = 0.1                # smoothing radius of the pseudo-gradient
    utility_samples: int = 256        # action pairs per utility evaluation
    # Perturbations drawn per iteration -- the papers' `batch_size`, and the single
    # setting that decides whether the estimator carries any signal at all. One
    # antithetic pair (`2`) is the reference implementation's *default*, not the setting
    # its experiments run: `g = (u+ - u-)/(2 sigma) * z` has cosine ~0.02 with the true
    # pseudo-gradient at `d ~ 5e3` parameters, because a random direction in `R^d` is
    # nearly orthogonal to the gradient. JPSPG's experiments use 256, which is what
    # brings the estimate above the noise floor; see the module docstring.
    perturbation_batch: int = 256
    antithetic: bool = True
    dynamics: str = "simultaneous"    # simultaneous | extragradient | optimistic
    # Polyak-averaged copy of each player's params, updated every step as
    # `target = target_tau * live + (1 - target_tau) * target`. No magnet: unlike
    # `training.trainer_common`, the live params never regularize *towards* the target --
    # there is no density to take a KL between, so nothing here plays that role. The
    # target exists purely as the slower-moving read-out; `0` disables tracking (the
    # target stays frozen at its init). Sourced from the shared `ppo.target_tau` field so
    # the same config value both baselines are compared against also governs this one.
    target_tau: float = 0.001

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RandomizedPolicyHyperparams":
        data = dict(data)
        data["hidden_dims"] = tuple(data["hidden_dims"])
        data["low"], data["high"] = tuple(data["low"]), tuple(data["high"])
        return cls(**data)


class RandomizedPolicy(nn.Module):
    """`a = f(obs, z)`: an implicit distribution over actions, squashed into the box.

    The noise is concatenated to the observation rather than added anywhere deeper, which
    is the simplest thing that makes every layer able to use it. Note what this module
    does *not* have: a log-probability. Nothing downstream may ask for one.
    """

    action_dim: int
    hidden_dims: tuple[int, ...]
    low: chex.Array
    high: chex.Array
    activation: str = "tanh"
    squash: str = "tanh"

    @nn.compact
    def __call__(self, obs: chex.Array, noise: chex.Array) -> chex.Array:
        # He initialization, as both papers specify. Flax's default is `lecun_normal`
        # (variance `1/fan_in`); He is `2/fan_in`, so the default starts the network a
        # factor `sqrt(2)` per layer quieter -- and a quiet network barely propagates
        # `noise`, which is the one input that makes this policy mixed rather than pure.
        dense = lambda dim, **kw: nn.Dense(dim, kernel_init=nn.initializers.he_normal(), **kw)
        x = jnp.concatenate([obs, noise], axis=-1)
        for dim in self.hidden_dims:
            x = Activation(kind=self.activation)(dense(dim)(x))
        raw = dense(self.action_dim, name="action_head")(x)
        if self.squash == "wrap":
            return self.low + (self.high - self.low) * jnp.mod(raw, 1.0)
        # Squashed rather than clipped: a clip would give the whole outside of the box
        # zero gradient, and a zeroth-order estimator cannot route around that.
        return self.low + (self.high - self.low) * 0.5 * (jnp.tanh(raw) + 1.0)


def build_policy(hyperparams: RandomizedPolicyHyperparams) -> RandomizedPolicy:
    return RandomizedPolicy(
        action_dim=hyperparams.action_dim,
        hidden_dims=tuple(hyperparams.hidden_dims),
        low=jnp.asarray(hyperparams.low),
        high=jnp.asarray(hyperparams.high),
        activation=hyperparams.activation,
        squash=hyperparams.squash,
    )


def sample_actions(policy: RandomizedPolicy, params, obs: chex.Array, key: chex.PRNGKey,
                   num_samples: int, noise_dim: int) -> chex.Array:
    noise = jax.random.normal(key, (num_samples, noise_dim))
    return jax.vmap(lambda z: policy.apply(params, obs, z))(noise)


# --------------------------------------------------------------------------- pseudo-gradient


def tree_normal(key: chex.PRNGKey, tree) -> chex.ArrayTree:
    """A standard normal draw shaped like `tree` -- the parameter-space perturbation."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef, [jax.random.normal(k, leaf.shape, leaf.dtype) for k, leaf in zip(keys, leaves)])


def tree_add_scaled(tree, other, scale) -> chex.ArrayTree:
    return jax.tree_util.tree_map(lambda a, b: a + scale * b, tree, other)


def tree_scale(tree, scale) -> chex.ArrayTree:
    """`scale * tree`, keeping each leaf's dtype.

    The cast matters: the utility estimate is float64 (the metric grid needs x64) while
    the parameters are float32, so an uncast product would silently promote the whole
    parameter tree and break the `scan` carry's type agreement.
    """
    return jax.tree_util.tree_map(lambda a: (scale * a).astype(a.dtype), tree)


def build_utility_fn(game: ZeroSumGame, policy: RandomizedPolicy,
                     hyperparams: RandomizedPolicyHyperparams):
    """`(params_pair, key) -> E[payoff]` for player 0, estimated from sampled action pairs.

    The same `key` is deliberately reused across the perturbed evaluations of one
    iteration (common random numbers): the pseudo-gradient is a *difference* of two
    utility estimates, and letting them differ in their sampling noise as well as in the
    perturbation is the fastest way to drown the signal.
    """
    observations = tuple(game.observation(player, bo.OBSERVATION_KEY) for player in (0, 1))
    samples, noise_dim = hyperparams.utility_samples, hyperparams.noise_dim

    def utility(params, key: chex.PRNGKey) -> chex.Array:
        key_0, key_1 = jax.random.split(key)
        actions_0 = sample_actions(policy, params[0], observations[0], key_0, samples, noise_dim)
        actions_1 = sample_actions(policy, params[1], observations[1], key_1, samples, noise_dim)
        return jnp.mean(game.payoff_batch(actions_0, actions_1))

    return utility


def noise_batch(key: chex.PRNGKey, tree, batch: int, antithetic: bool) -> chex.ArrayTree:
    """`batch` parameter-space perturbations stacked on a leading axis.

    Antithetic sampling draws `batch // 2` normals and mirrors them, so the stack is
    `[z_1..z_p, -z_1..-z_p]` -- exactly the reference implementation's
    `jnp.concatenate([z, -z])`, which turns the weighted sum below into the
    centred-difference stencil without ever forming the difference explicitly.
    """
    draws = batch // 2 if antithetic else batch
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))

    def draw(leaf_key, leaf):
        z = jax.random.normal(leaf_key, (draws,) + leaf.shape, leaf.dtype)
        return jnp.concatenate([z, -z]) if antithetic else z

    return jax.tree_util.tree_unflatten(
        treedef, [draw(k, leaf) for k, leaf in zip(keys, leaves)])


def batch_utility_keys(key: chex.PRNGKey, batch: int, antithetic: bool) -> chex.Array:
    """One key per perturbation, shared within each antithetic pair.

    Sharing it is the common-random-numbers trick: `u(x + sigma z)` and `u(x - sigma z)`
    differ only in the perturbation, not in which action pairs were sampled. Across
    *different* pairs the keys differ, so the batch still averages over game randomness.
    """
    draws = batch // 2 if antithetic else batch
    keys = jax.random.split(key, draws)
    return jnp.concatenate([keys, keys]) if antithetic else keys


def contract_noise(noise, values: chex.Array, scale: float) -> chex.ArrayTree:
    """`scale * sum_k values[k] * noise[k]`, the pseudo-gradient's weighted sum.

    The cast keeps each leaf's dtype: the utility estimate is float64 (the metric grid
    needs x64) while the parameters are float32, and an uncast product would silently
    promote the whole parameter tree and break the `scan` carry's type agreement.
    """
    return jax.tree_util.tree_map(
        lambda z: (jnp.tensordot(values, z, (0, 0)) * scale).astype(z.dtype), noise)


def perturb(base, noise):
    """`base + noise`, broadcasting the unbatched parameters over the leading batch axis."""
    return jax.tree_util.tree_map(lambda b, z: b + z, base, noise)


def spg_pseudo_gradients(utility, params, key: chex.PRNGKey, sigma: float,
                         antithetic: bool = True, batch: int = 2) -> tuple[tuple, int]:
    """Simultaneous pseudo-gradient: perturb **one player at a time** (IJCAI 2023).

    `batch` perturbations are drawn per player and averaged, which is the paper's
    `1/(2n sigma) sum_k (u(x + sigma z_k) - u(x - sigma z_k)) z_k`. Returns
    `(gradients, evaluations)`; `evaluations` is what `jpspg.py` improves on -- it grows
    linearly in the number of players, `n * batch` here.
    """
    gradients, evaluations = [], 0
    for player in (0, 1):
        key, noise_key, utility_key = jax.random.split(key, 3)
        noise = noise_batch(noise_key, params[player], batch, antithetic)
        perturbed = perturb(params[player], tree_scale(noise, sigma))
        other = params[1 - player]

        def evaluate(own, utility_sub_key, player=player, other=other):
            profile = (own, other) if player == 0 else (other, own)
            return utility(profile, utility_sub_key)

        values = jax.vmap(evaluate)(
            perturbed, batch_utility_keys(utility_key, batch, antithetic))
        # Player 1's utility is the negated payoff.
        sign = 1.0 if player == 0 else -1.0
        gradients.append(contract_noise(noise, values, sign / (batch * sigma)))
        evaluations += batch
    return tuple(gradients), evaluations


# --------------------------------------------------------------------------- training


def _apply_dynamics(dynamics: str, estimate, params, key, previous):
    """One iteration's ascent direction, under the dynamics their appendix runs.

    `estimate(params, key) -> (gradients, evaluations)`. Extragradient re-estimates at a
    look-ahead point (doubling the evaluation count); optimistic reuses the previous
    estimate as the look-ahead instead, which is the whole point of it.
    """
    gradients, evaluations = estimate(params, key)
    if dynamics == "simultaneous":
        return gradients, gradients, evaluations
    if dynamics == "optimistic":
        # 2 g_t - g_{t-1}: the standard optimistic correction, with the previous estimate
        # standing in for the look-ahead.
        direction = tuple(
            jax.tree_util.tree_map(lambda g, p: 2.0 * g - p, gradients[i], previous[i])
            for i in (0, 1))
        return direction, gradients, evaluations
    if dynamics == "extragradient":
        raise ValueError("extragradient is handled in `run_pseudo_gradient`, not here")
    raise ValueError(f"unknown dynamics {dynamics!r} "
                     "(choices: simultaneous, extragradient, optimistic)")


def run_pseudo_gradient(
    game: ZeroSumGame,
    oracle: GridOracle,
    hyperparams: RandomizedPolicyHyperparams,
    estimator: PseudoGradientFn = spg_pseudo_gradients,
    iterations: int = 20_000,
    log_every: int | None = None,
    samples: int = 4096,
    seed: int = 0,
    writer: RunWriter | None = None,
    estimator_name: str = "spg",
    score: bool = True,
) -> dict:
    """Zeroth-order self-play with a randomized policy per player.

    `estimator(utility, params, key, sigma, antithetic) -> (gradients, evaluations)` is
    the only thing that separates this from `jpspg.py`; everything else -- the policy,
    the utility estimate, the optimizer, the metric -- is shared, so a comparison between
    the two estimators is a comparison of the estimators.

    `score=False` records checkpoints, wall-time and payoff-evaluation counts but skips
    the exploitability computation, which is scored offline instead (see
    `experiments/one_shot_neural/`).
    """
    log_every = log_every or max(iterations // 50, 1)
    policy = build_policy(hyperparams)
    observations = tuple(game.observation(player, bo.OBSERVATION_KEY) for player in (0, 1))
    utility = build_utility_fn(game, policy, hyperparams)

    key = jax.random.PRNGKey(seed)
    init_keys = jax.random.split(key, 3)
    params = tuple(
        policy.init(init_keys[player], observations[player], jnp.zeros(hyperparams.noise_dim))
        for player in (0, 1))
    key = init_keys[2]

    transformations = [build_optimizer(hyperparams.optimizer, hyperparams.learning_rate)]
    if hyperparams.max_grad_norm > 0:
        transformations.insert(0, optax.clip_by_global_norm(hyperparams.max_grad_norm))
    optimizer = optax.chain(*transformations)
    opt_states = tuple(optimizer.init(params[player]) for player in (0, 1))

    def estimate(current, estimate_key):
        return estimator(utility, current, estimate_key, hyperparams.sigma,
                         hyperparams.antithetic, hyperparams.perturbation_batch)

    def step(carry, step_key):
        params, opt_states, previous, target_params = carry
        if hyperparams.dynamics == "extragradient":
            look_key, apply_key = jax.random.split(step_key)
            gradients, evaluations = estimate(params, look_key)
            # A plain scaled step to the look-ahead point, then the real update from the
            # original point using the gradient measured there (Korpelevich).
            lookahead = tuple(
                tree_add_scaled(params[i], gradients[i], hyperparams.learning_rate) for i in (0, 1))
            direction, extra = estimate(lookahead, apply_key)
            evaluations += extra
            raw = direction
        else:
            direction, raw, evaluations = _apply_dynamics(
                hyperparams.dynamics, estimate, params, step_key, previous)

        new_params, new_states, new_targets = [], [], []
        for player in (0, 1):
            # optax minimizes; these players ascend their own utility.
            updates, state = optimizer.update(
                tree_scale(direction[player], -1.0), opt_states[player], params[player])
            updated = optax.apply_updates(params[player], updates)
            new_params.append(updated)
            new_states.append(state)
            new_targets.append(optax.incremental_update(
                updated, target_params[player], hyperparams.target_tau))
        metrics = {"evaluations": jnp.asarray(evaluations, dtype=jnp.float64),
                   "grad_norm_0": optax.tree.norm(direction[0])}
        return (tuple(new_params), tuple(new_states), raw, tuple(new_targets)), metrics

    runner = ChunkRunner(lambda n: lambda c, k: jax.lax.scan(step, c, k))

    zero_gradients = tuple(jax.tree_util.tree_map(jnp.zeros_like, params[i]) for i in (0, 1))
    carry = (params, opt_states, zero_gradients, params)

    local_history: list[dict] = []
    history = writer.history if writer is not None else local_history

    def record(t: int, metrics: dict | None) -> dict:
        nonlocal key
        key, key_0, key_1, target_key_0, target_key_1 = jax.random.split(key, 5)
        strategies = [
            empirical_strategy(sample_actions(policy, carry[0][player], observations[player],
                                              sample_key, samples, hyperparams.noise_dim))
            for player, sample_key in ((0, key_0), (1, key_1))]
        target_strategies = [
            empirical_strategy(sample_actions(policy, carry[3][player], observations[player],
                                              sample_key, samples, hyperparams.noise_dim))
            for player, sample_key in ((0, target_key_0), (1, target_key_1))]
        entry = {"t": int(t), "wall_time": runner.run_seconds,
                 "compile_time": runner.compile_seconds,
                 # Each utility evaluation averages `utility_samples` action pairs.
                 "payoff_evals": total_evaluations * hyperparams.utility_samples}
        if score:
            entry.update(strategy_row(oracle, strategies[0][0], strategies[0][1],
                                      strategies[1][0], strategies[1][1]))
            entry["target_expl"] = float(oracle.exploitability(
                target_strategies[0][0], target_strategies[0][1],
                target_strategies[1][0], target_strategies[1][1]))
        if metrics:
            entry.update({k: float(v) for k, v in metrics.items()})
        if writer is not None:
            writer.record(entry, StrategyPair(
                t=int(t), support_0=strategies[0][0], weights_0=strategies[0][1],
                support_1=strategies[1][0], weights_1=strategies[1][1]))
        else:
            local_history.append(entry)
        return entry, strategies, target_strategies

    done, total_evaluations = 0, 0
    entry, strategies, target_strategies = record(0, None)
    print_row(entry, ("support_0", "support_1"))

    while done < iterations:
        length = min(log_every, iterations - done)
        key, chunk_key = jax.random.split(key)
        carry, metrics_stack = runner(carry, jax.random.split(chunk_key, length), length)
        done += length
        metrics = jax.device_get(metrics_stack)
        total_evaluations += int(np.sum(np.asarray(metrics["evaluations"])))
        entry, strategies, target_strategies = record(done, {
            "grad_norm_0": float(np.mean(np.asarray(metrics["grad_norm_0"]))),
            "utility_evaluations": float(total_evaluations),
        })
        print_row(entry, ("grad_norm_0", "utility_evaluations", "support_0", "support_1"))

    if writer is not None:
        for player in (0, 1):
            writer.save_params(f"player_{player}", hyperparams, carry[0][player])
            writer.save_params(f"target_{player}", hyperparams, carry[3][player])
    return {"history": history, "params": carry[0], "target_params": carry[3], "policy": policy,
            "support_0": strategies[0][0], "weights_0": strategies[0][1],
            "support_1": strategies[1][0], "weights_1": strategies[1][1],
            "target_support_0": target_strategies[0][0], "target_weights_0": target_strategies[0][1],
            "target_support_1": target_strategies[1][0], "target_weights_1": target_strategies[1][1],
            "utility_evaluations": total_evaluations, "estimator": estimator_name,
            "train_seconds": runner.run_seconds, "compile_seconds": runner.compile_seconds}


def hyperparams_from_config(game: ZeroSumGame, config, args) -> RandomizedPolicyHyperparams:
    lo, hi = action_bounds(game, 0)
    return RandomizedPolicyHyperparams(
        action_dim=lo.shape[0],
        hidden_dims=tuple(config.network.hidden_dims),
        noise_dim=args.noise_dim,
        activation=config.network.activation,
        low=tuple(float(x) for x in lo),
        high=tuple(float(x) for x in hi),
        squash=getattr(args, "squash", "tanh"),
        learning_rate=args.lr,
        optimizer=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        sigma=args.sigma,
        utility_samples=args.utility_samples,
        perturbation_batch=args.perturbation_batch,
        antithetic=not args.no_antithetic,
        dynamics=args.dynamics,
        target_tau=config.ppo.target_tau,
    )


def add_arguments(ap) -> None:
    """The flags shared by this module's CLI and `jpspg.py`'s."""
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=0.1, help="pseudo-gradient smoothing radius")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="adabelief", help="see training.optimizers.OPTIMIZERS")
    ap.add_argument("--noise-dim", type=int, default=8, help="latent noise fed to the policy")
    ap.add_argument("--squash", choices=("tanh", "wrap"), default="tanh",
                    help="map into the action box; 'wrap' (modulo the box) for periodic "
                         "games such as circle")
    ap.add_argument("--max-grad-norm", type=float, default=0.0,
                    help="clip the pseudo-gradient to this global norm; 0 disables clipping")
    ap.add_argument("--utility-samples", type=int, default=256,
                    help="action pairs averaged per utility evaluation")
    ap.add_argument("--perturbation-batch", type=int, default=256,
                    help="perturbations drawn and averaged per iteration (the papers' "
                         "`batch_size`; JPSPG's experiments use 256, its reference "
                         "implementation defaults to 2)")
    ap.add_argument("--no-antithetic", action="store_true",
                    help="single-point estimator instead of the central difference")
    ap.add_argument("--dynamics", choices=("simultaneous", "extragradient", "optimistic"),
                    default="simultaneous")


def report(args, game, game_config, oracle, hyperparams, result, writer, meta) -> None:
    """Shared tail of both CLIs."""
    last = result["history"][-1]
    print(f"\nfinal exploitability {last['expl']:+.5f}  |  "
          f"target {last['target_expl']:+.5f}  |  "
          f"{result['utility_evaluations']} utility evaluations")
    report_final(oracle, result["support_0"], result["weights_0"],
                 result["support_1"], result["weights_1"])
    print("  target:")
    report_final(oracle, result["target_support_0"], result["target_weights_0"],
                 result["target_support_1"], result["target_weights_1"])
    if writer is not None:
        writer.finish({"final": last, "utility_evaluations": result["utility_evaluations"]})
        print(f"run -> {writer.directory}")
    elif args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": meta, "history": result["history"]}, indent=2))
        print(f"saved history -> {args.out}")


def main() -> None:
    ap = neural_parser(__doc__)
    add_arguments(ap)
    args = ap.parse_args()

    game, game_config, config = load_run(args.config)
    oracle = GridOracle(game, points=args.grid)
    hyperparams = hyperparams_from_config(game, config, args)

    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"policy  : randomized network a=f(o,z), noise {hyperparams.noise_dim}, "
          f"hidden {hyperparams.hidden_dims} (no density -- no magnet, no ratio)  "
          f"target_tau={hyperparams.target_tau}")
    print(f"solver  : simultaneous pseudo-gradient  sigma={hyperparams.sigma}  "
          f"lr={hyperparams.learning_rate}  {hyperparams.optimizer}  "
          f"{'antithetic' if hyperparams.antithetic else 'single-point'}  "
          f"batch={hyperparams.perturbation_batch}  "
          f"dynamics={hyperparams.dynamics}\n")

    meta = {"algorithm": "randomized_policy_spg", "config": args.config,
            "hyperparams": hyperparams.to_dict(), **{k: v for k, v in vars(args).items()
                                                     if k != "config"}}
    writer = RunWriter(args.checkpoint_dir, meta) if args.checkpoint_dir else None
    result = run_pseudo_gradient(game, oracle, hyperparams, spg_pseudo_gradients,
                                 iterations=args.iters, log_every=args.log_every,
                                 samples=args.samples, seed=args.seed, writer=writer,
                                 estimator_name="spg")
    report(args, game, game_config, oracle, hyperparams, result, writer, meta)


if __name__ == "__main__":
    main()
