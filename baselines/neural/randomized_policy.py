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

the central-difference (antithetic) pseudo-gradient, one perturbation per player per
iteration -- hence *simultaneous pseudo-gradient*, and hence `2n` utility evaluations
for `n` players. `jpspg.py` is the follow-up that makes that count constant.

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
    # Optimization. The paper's settings: AdaBelief, alpha = 1e-4, sigma = 0.1.
    learning_rate: float = 1e-4
    optimizer: str = "adabelief"
    max_grad_norm: float = 0.0        # 0 disables clipping; the pseudo-gradient is noisy, not large
    sigma: float = 0.1                # smoothing radius of the pseudo-gradient
    utility_samples: int = 256        # action pairs per utility evaluation
    antithetic: bool = True
    dynamics: str = "simultaneous"    # simultaneous | extragradient | optimistic

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

    @nn.compact
    def __call__(self, obs: chex.Array, noise: chex.Array) -> chex.Array:
        x = jnp.concatenate([obs, noise], axis=-1)
        for dim in self.hidden_dims:
            x = Activation(kind=self.activation)(nn.Dense(dim)(x))
        raw = nn.Dense(self.action_dim, name="action_head")(x)
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


def spg_pseudo_gradients(utility, params, key: chex.PRNGKey, sigma: float,
                         antithetic: bool = True) -> tuple[tuple, int]:
    """Simultaneous pseudo-gradient: perturb **one player at a time** (IJCAI 2023).

    Returns `(gradients, evaluations)`; `evaluations` is what `jpspg.py` improves on --
    it grows linearly in the number of players, `2n` with antithetic sampling.
    """
    gradients, evaluations = [], 0
    for player in (0, 1):
        key, noise_key, utility_key = jax.random.split(key, 3)
        noise = tree_normal(noise_key, params[player])
        sign = 1.0 if player == 0 else -1.0     # player 1's utility is the negated payoff

        plus = list(params)
        plus[player] = tree_add_scaled(params[player], noise, sigma)
        u_plus = sign * utility(tuple(plus), utility_key)
        evaluations += 1
        if antithetic:
            minus = list(params)
            minus[player] = tree_add_scaled(params[player], noise, -sigma)
            u_minus = sign * utility(tuple(minus), utility_key)
            evaluations += 1
            coefficient = (u_plus - u_minus) / (2.0 * sigma)
        else:
            coefficient = u_plus / sigma
        gradients.append(tree_scale(noise, coefficient))
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
        return estimator(utility, current, estimate_key, hyperparams.sigma, hyperparams.antithetic)

    def step(carry, step_key):
        params, opt_states, previous = carry
        if hyperparams.dynamics == "extragradient":
            look_key, apply_key = jax.random.split(step_key)
            gradients, evaluations = estimate(params, look_key)
            # A plain scaled step to the look-ahead point, then the real update from the
            # original point using the gradient measured there (Korpelevich).
            lookahead = tuple(
                tree_add_scaled(params[i], gradients[i], hyperparams.learning_rate) for i in (0, 1))
            direction, raw, extra = estimate(lookahead, apply_key)
            evaluations += extra
        else:
            direction, raw, evaluations = _apply_dynamics(
                hyperparams.dynamics, estimate, params, step_key, previous)

        new_params, new_states = [], []
        for player in (0, 1):
            # optax minimizes; these players ascend their own utility.
            updates, state = optimizer.update(
                tree_scale(direction[player], -1.0), opt_states[player], params[player])
            new_params.append(optax.apply_updates(params[player], updates))
            new_states.append(state)
        metrics = {"evaluations": jnp.asarray(evaluations, dtype=jnp.float64),
                   "grad_norm_0": optax.tree.norm(direction[0])}
        return (tuple(new_params), tuple(new_states), raw), metrics

    runner = ChunkRunner(lambda n: lambda c, k: jax.lax.scan(step, c, k))

    zero_gradients = tuple(jax.tree_util.tree_map(jnp.zeros_like, params[i]) for i in (0, 1))
    carry = (params, opt_states, zero_gradients)

    local_history: list[dict] = []
    history = writer.history if writer is not None else local_history

    def record(t: int, metrics: dict | None) -> dict:
        nonlocal key
        key, key_0, key_1 = jax.random.split(key, 3)
        strategies = [
            empirical_strategy(sample_actions(policy, carry[0][player], observations[player],
                                              sample_key, samples, hyperparams.noise_dim))
            for player, sample_key in ((0, key_0), (1, key_1))]
        entry = {"t": int(t), "wall_time": runner.run_seconds,
                 "compile_time": runner.compile_seconds,
                 # Each utility evaluation averages `utility_samples` action pairs.
                 "payoff_evals": total_evaluations * hyperparams.utility_samples}
        if score:
            entry.update(strategy_row(oracle, strategies[0][0], strategies[0][1],
                                      strategies[1][0], strategies[1][1]))
        if metrics:
            entry.update({k: float(v) for k, v in metrics.items()})
        if writer is not None:
            writer.record(entry, StrategyPair(
                t=int(t), support_0=strategies[0][0], weights_0=strategies[0][1],
                support_1=strategies[1][0], weights_1=strategies[1][1]))
        else:
            local_history.append(entry)
        return entry, strategies

    done, total_evaluations = 0, 0
    entry, strategies = record(0, None)
    print_row(entry, ("support_0", "support_1"))

    while done < iterations:
        length = min(log_every, iterations - done)
        key, chunk_key = jax.random.split(key)
        carry, metrics_stack = runner(carry, jax.random.split(chunk_key, length), length)
        done += length
        metrics = jax.device_get(metrics_stack)
        total_evaluations += int(np.sum(np.asarray(metrics["evaluations"])))
        entry, strategies = record(done, {
            "grad_norm_0": float(np.mean(np.asarray(metrics["grad_norm_0"]))),
            "utility_evaluations": float(total_evaluations),
        })
        print_row(entry, ("grad_norm_0", "utility_evaluations", "support_0", "support_1"))

    if writer is not None:
        for player in (0, 1):
            writer.save_params(f"player_{player}", hyperparams, carry[0][player])
    return {"history": history, "params": carry[0], "policy": policy,
            "support_0": strategies[0][0], "weights_0": strategies[0][1],
            "support_1": strategies[1][0], "weights_1": strategies[1][1],
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
        learning_rate=args.lr,
        optimizer=args.optimizer,
        sigma=args.sigma,
        utility_samples=args.utility_samples,
        antithetic=not args.no_antithetic,
        dynamics=args.dynamics,
    )


def add_arguments(ap) -> None:
    """The flags shared by this module's CLI and `jpspg.py`'s."""
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=0.1, help="pseudo-gradient smoothing radius")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="adabelief", help="see training.optimizers.OPTIMIZERS")
    ap.add_argument("--noise-dim", type=int, default=8, help="latent noise fed to the policy")
    ap.add_argument("--utility-samples", type=int, default=256,
                    help="action pairs averaged per utility evaluation")
    ap.add_argument("--no-antithetic", action="store_true",
                    help="single-point estimator instead of the central difference")
    ap.add_argument("--dynamics", choices=("simultaneous", "extragradient", "optimistic"),
                    default="simultaneous")


def report(args, game, game_config, oracle, hyperparams, result, writer, meta) -> None:
    """Shared tail of both CLIs."""
    last = result["history"][-1]
    print(f"\nfinal exploitability {last['expl']:+.5f}  |  "
          f"{result['utility_evaluations']} utility evaluations")
    report_final(oracle, result["support_0"], result["weights_0"],
                 result["support_1"], result["weights_1"])
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
          f"hidden {hyperparams.hidden_dims} (no density -- no magnet, no ratio)")
    print(f"solver  : simultaneous pseudo-gradient  sigma={hyperparams.sigma}  "
          f"lr={hyperparams.learning_rate}  {hyperparams.optimizer}  "
          f"{'antithetic' if hyperparams.antithetic else 'single-point'}  "
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
