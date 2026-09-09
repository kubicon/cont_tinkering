"""Randomized policy networks trained by **exact pathwise gradients**.

The same representation as `randomized_policy.py` -- an implicit generative policy
`a = f_theta(o, z)`, `z ~ N(0, I)` -- and a different training signal, which is the whole
point of having both. Call it the second version of that method:

**What the papers do.** An implicit policy has no tractable density, so Martin & Sandholm
stop differentiating the policy at all and estimate `grad_i u_i` by Gaussian smoothing in
*parameter* space (`randomized_policy.py`, and `jpspg.py` for the joint-perturbation
version). That buys a black-box access model -- the payoff is only ever *evaluated* --
and pays for it in variance: a random direction in `R^d` is nearly orthogonal to the
gradient it estimates, so the estimate needs `perturbation_batch` draws averaged to carry
any signal at all.

**What this module does.** The games in `games/` have differentiable payoffs and the
noise enters as `a = f(theta, z)` with `z` drawn independently of `theta`, so
`grad_theta E_z[u(f(theta, z), ...)]` can be taken *pathwise* -- the reparametrization
gradient, straight through the network and the payoff with `jax.grad`. No density is
needed for that, only differentiability, so the one obstacle the papers work around does
not exist here. The estimate is exact up to the action sample, and one iteration costs
`2 * batch_size` payoff evaluations instead of `2 * utility_samples * perturbation_batch`.

So this is the honest upper bound on what the *representation* can do, separated from
what the zeroth-order *estimator* costs. `spg` and `jpspg` answer "how well does an
implicit policy do when you may only evaluate the payoff"; this answers "how well does an
implicit policy do at all". A gap between them is the price of the black-box access model;
no gap means the estimator was never the bottleneck. Note the access model is strictly
stronger and `run_cell.ACCESS_MODEL` records it as such -- this is not a drop-in rival to
those two, it is the control for them.

`--smooth` puts the zeroth-order estimator back, via `optax.perturbations`, which
implements the score-function (REINFORCE) estimator of the same smoothed objective using
only *values* of the payoff. That makes the two access models switchable inside one module
with everything else held fixed, which is the comparison worth running. Its default of one
perturbation sample per step is the IJCAI'23 pairing (a tiny step, and the trajectory
averages the noise), not IJCAI'25's (`batch = 256` under AdaBelief); see
`RandomizedPolicyHyperparams` for why that pairing, not the sample count alone, is what
decides whether it converges.

The defaults here are the reference implementation's for this variant, and several differ
from `randomized_policy.py`'s on purpose: `mish` activations under RMSNorm, a `sigmoid`
squash, flax's default (`lecun_normal`) initialization rather than He, and optimistic
gradient descent rather than AdaBelief. The OGD coefficients are the unusual part -- a
step `alpha = lr` against a negative-momentum `beta = 1.0`, so the extrapolation term is
`1 / lr` times the gradient term. That reads like a scale error and is not one: matching
them makes `glicksberg_gross` markedly worse.

Usage:
    python -m baselines.neural.randomized_policy_pathwise configs/glicksberg_gross.yaml
    python -m baselines.neural.randomized_policy_pathwise configs/two_point.yaml --smooth 1
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from games.base import ZeroSumGame
from nets.activations import Activation
from nets.normalization import Normalization
from training.optimizers import build_optimizer

from ..common import ChunkRunner, GridOracle, StrategyPair, action_bounds
from . import br_oracle as bo
from .common import RunWriter, empirical_strategy, load_run, neural_parser, print_row, \
    report_final, strategy_row


@dataclasses.dataclass(frozen=True)
class PathwisePolicyHyperparams:
    """Everything needed to rebuild the policy and reproduce the run."""

    action_dim: int
    hidden_dims: tuple[int, ...]
    noise_dim: int = 16
    activation: str = "mish"
    normalization: str = "rms_norm"
    low: tuple[float, ...] = (0.0,)
    high: tuple[float, ...] = (1.0,)
    # Optimization. `optimistic` is this variant's own optimizer and is built here rather
    # than by `training.optimizers`, which exposes no `alpha`/`beta`; every other name is
    # looked up there, so the repo's optimizers remain available for comparison.
    optimizer: str = "optimistic"
    learning_rate: float = 1e-2
    optimism: float = 1.0             # OGD's `beta`; only read when optimizer == "optimistic"
    max_grad_norm: float = 0.0        # 0 disables clipping
    # Action pairs per gradient estimate. Unlike the pseudo-gradient's `utility_samples`
    # this is the *only* sampling in the update, because the gradient itself is exact:
    # there is no perturbation batch to average on top of it.
    batch_size: int = 64
    # Zeroth-order fallback. `smooth = 0` keeps the exact pathwise gradient; nonzero
    # replaces it with the score-function estimator of the objective smoothed at radius
    # `smooth_scale`, averaged over `smooth` perturbation samples -- which is the papers'
    # access model, reachable from inside this module for a controlled comparison.
    smooth: int = 0
    smooth_scale: float = 0.1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PathwisePolicyHyperparams":
        data = dict(data)
        data["hidden_dims"] = tuple(data["hidden_dims"])
        data["low"], data["high"] = tuple(data["low"]), tuple(data["high"])
        return cls(**data)

    def payoff_evals_per_iteration(self) -> int:
        return payoff_evals_per_iteration(self.batch_size, self.smooth)


def payoff_evals_per_iteration(batch_size: int, smooth: int) -> int:
    """Payoff evaluations one iteration costs, in scored action pairs.

    `batch_size` pairs per player's gradient, both players every iteration. Under
    `--smooth` the score-function estimator evaluates the utility `smooth` times at
    perturbed parameters plus once at the unperturbed point for its variance-reduction
    baseline, so the same batch is scored `smooth + 1` times per player.

    Module-level so `experiments/one_shot_neural/run_cell.py` can price an iteration
    before it has built the hyperparameters -- the budget is in payoff evaluations, and a
    cost formula that lives in two places is one that will disagree with itself.
    """
    return 2 * batch_size * (smooth + 1 if smooth else 1)


class PathwisePolicy(nn.Module):
    """`a = f(obs, z)`, squashed into the action box -- differentiable end to end.

    Structurally `randomized_policy.RandomizedPolicy` with this variant's choices: the
    noise is concatenated to the observation, each hidden layer is normalized before its
    activation, and the squash is a `sigmoid` rather than a scaled `tanh`.

    The initialization is flax's default (`lecun_normal`), *not* the He initialization the
    pseudo-gradient policy specifies. That difference is load-bearing in one direction
    only: He exists there because a quiet network barely propagates `noise`, and a
    zeroth-order estimator cannot see that it should. Here the gradient flows through the
    noise path, so the network can learn to amplify it.
    """

    action_dim: int
    hidden_dims: tuple[int, ...]
    low: chex.Array
    high: chex.Array
    activation: str = "mish"
    normalization: str = "rms_norm"

    @nn.compact
    def __call__(self, obs: chex.Array, noise: chex.Array) -> chex.Array:
        x = jnp.concatenate([obs, noise], axis=-1)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = Normalization(kind=self.normalization)(x, use_running_average=True)
            x = Activation(kind=self.activation)(x)
        raw = nn.Dense(self.action_dim, name="action_head")(x)
        return self.low + (self.high - self.low) * nn.sigmoid(raw)


def build_policy(hyperparams: PathwisePolicyHyperparams) -> PathwisePolicy:
    return PathwisePolicy(
        action_dim=hyperparams.action_dim,
        hidden_dims=tuple(hyperparams.hidden_dims),
        low=jnp.asarray(hyperparams.low),
        high=jnp.asarray(hyperparams.high),
        activation=hyperparams.activation,
        normalization=hyperparams.normalization,
    )


def sample_actions(policy: PathwisePolicy, params, obs: chex.Array, key: chex.PRNGKey,
                   num_samples: int, noise_dim: int) -> chex.Array:
    """`num_samples` actions from the policy at a single `obs`.

    Differentiable in `params`: the noise is drawn from `key` alone, so the sample is a
    deterministic function of the parameters given the draw. That is the reparametrization
    the pathwise gradient rests on.
    """
    noise = jax.random.normal(key, (num_samples, noise_dim))
    return jax.vmap(lambda z: policy.apply(params, obs, z))(noise)


def build_utility_fn(game: ZeroSumGame, policy: PathwisePolicy,
                     hyperparams: PathwisePolicyHyperparams):
    """`(profile, key) -> E[payoff]` for player 0, over `batch_size` sampled action pairs.

    The pairing matters and is the same one `randomized_policy.build_utility_fn` uses:
    each player's `i`-th sampled action is scored against the other's `i`-th, so the batch
    is `batch_size` payoff evaluations rather than an outer product of them.
    """
    observations = tuple(game.observation(player, bo.OBSERVATION_KEY) for player in (0, 1))
    samples, noise_dim = hyperparams.batch_size, hyperparams.noise_dim

    def utility(profile, key: chex.PRNGKey) -> chex.Array:
        key_0, key_1 = jax.random.split(key)
        actions_0 = sample_actions(policy, profile[0], observations[0], key_0, samples, noise_dim)
        actions_1 = sample_actions(policy, profile[1], observations[1], key_1, samples, noise_dim)
        return jnp.mean(game.payoff_batch(actions_0, actions_1))

    return utility


def smoothed_utility_fn(utility, scale: float, samples: int):
    """`utility` replaced by a score-function estimate of its Gaussian-smoothed value.

    `optax.perturbations.make_perturbed_fun` evaluates the wrapped function only at
    `stop_gradient`-ed perturbed inputs and carries the derivative on a DiCE "magic box"
    factor, so differentiating the result gives the zeroth-order estimator and *no*
    pathwise term -- switching the access model rather than adding to it.

    The whole profile is perturbed, not just the player being differentiated. Only that
    player's block receives a gradient (the other is a constant of the call), so the
    readout is per-player while the perturbation is joint.
    """

    def new_utility(profile, key: chex.PRNGKey):
        key, subkey = jax.random.split(key)
        proxy = optax.perturbations.make_perturbed_fun(
            fun=lambda perturbed: utility(perturbed, subkey),
            num_samples=samples,
            sigma=scale,
            noise=optax.perturbations.Normal(),
        )
        return proxy(key, profile)

    return new_utility


def player_gradients(utility, params, key: chex.PRNGKey) -> tuple:
    """Both players' gradients of their own utility, each w.r.t. its own parameters.

    One `jax.grad` per player rather than one over the pair: player 1 maximizes `-u`, so
    the two are gradients of *different* objectives and a single call would give the
    wrong sign on one of them. The opponent's parameters are a constant of each call --
    this is simultaneous ascent, not a differentiation through the opponent's response.
    """
    gradients = []
    for player in (0, 1):
        # Player 1's utility is the negated payoff.
        sign = 1.0 if player == 0 else -1.0

        def own_utility(own, player=player, sign=sign):
            profile = (own, params[1]) if player == 0 else (params[0], own)
            return sign * utility(profile, key)

        gradients.append(jax.grad(own_utility)(params[player]))
    return tuple(gradients)


def build_pathwise_optimizer(hyperparams: PathwisePolicyHyperparams) -> optax.GradientTransformation:
    """This variant's optimizer, with `optimistic` built here and everything else deferred.

    `optax.optimistic_gradient_descent(1.0, alpha, beta)` updates by
    `(alpha + beta) g_t - beta g_{t-1}`, so passing the learning rate as `alpha` and
    keeping `beta = optimism` reproduces the reference implementation exactly.
    `training.optimizers` only exposes the single-argument form, which is why this one
    case is not routed through it.
    """
    if hyperparams.optimizer == "optimistic":
        transformation = optax.optimistic_gradient_descent(
            learning_rate=1.0, alpha=hyperparams.learning_rate, beta=hyperparams.optimism)
    else:
        transformation = build_optimizer(hyperparams.optimizer, hyperparams.learning_rate)
    if hyperparams.max_grad_norm > 0:
        return optax.chain(optax.clip_by_global_norm(hyperparams.max_grad_norm), transformation)
    return transformation


def run_pathwise(
    game: ZeroSumGame,
    oracle: GridOracle,
    hyperparams: PathwisePolicyHyperparams,
    iterations: int = 20_000,
    log_every: int | None = None,
    samples: int = 4096,
    seed: int = 0,
    writer: RunWriter | None = None,
    score: bool = True,
) -> dict:
    """Simultaneous gradient ascent on both players' utilities, exactly differentiated.

    Deliberately the same shape as `randomized_policy.run_pseudo_gradient` -- same chunked
    scan, same history rows, same checkpoints -- so the two differ in the gradient and
    nothing else. There is no `dynamics` switch here: the optimism lives in the optimizer
    (`optimistic` is OGD), which is where this variant puts it.

    `score=False` records checkpoints, wall time and payoff-evaluation counts but skips
    the exploitability computation, which is scored offline instead (see
    `experiments/one_shot_neural/`).
    """
    log_every = log_every or max(iterations // 50, 1)
    policy = build_policy(hyperparams)
    observations = tuple(game.observation(player, bo.OBSERVATION_KEY) for player in (0, 1))

    utility = build_utility_fn(game, policy, hyperparams)
    if hyperparams.smooth:
        utility = smoothed_utility_fn(utility, hyperparams.smooth_scale, hyperparams.smooth)

    key = jax.random.PRNGKey(seed)
    init_keys = jax.random.split(key, 3)
    params = tuple(
        policy.init(init_keys[player], observations[player], jnp.zeros(hyperparams.noise_dim))
        for player in (0, 1))
    key = init_keys[2]

    optimizer = build_pathwise_optimizer(hyperparams)
    opt_states = tuple(optimizer.init(params[player]) for player in (0, 1))
    per_iteration = hyperparams.payoff_evals_per_iteration()

    def step(carry, step_key):
        params, opt_states = carry
        gradients = player_gradients(utility, params, step_key)

        new_params, new_states = [], []
        for player in (0, 1):
            # optax minimizes; these players ascend their own utility.
            updates, state = optimizer.update(
                jax.tree_util.tree_map(jnp.negative, gradients[player]),
                opt_states[player], params[player])
            new_params.append(optax.apply_updates(params[player], updates))
            new_states.append(state)
        metrics = {"grad_norm_0": optax.tree.norm(gradients[0])}
        return (tuple(new_params), tuple(new_states)), metrics

    runner = ChunkRunner(lambda n: lambda c, k: jax.lax.scan(step, c, k))
    carry = (params, opt_states)

    local_history: list[dict] = []
    history = writer.history if writer is not None else local_history

    def record(t: int, metrics: dict | None) -> tuple[dict, list]:
        nonlocal key
        key, key_0, key_1 = jax.random.split(key, 3)
        strategies = [
            empirical_strategy(sample_actions(policy, carry[0][player], observations[player],
                                              sample_key, samples, hyperparams.noise_dim))
            for player, sample_key in ((0, key_0), (1, key_1))]
        entry = {"t": int(t), "wall_time": runner.run_seconds,
                 "compile_time": runner.compile_seconds,
                 "payoff_evals": t * per_iteration}
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

    done = 0
    entry, strategies = record(0, None)
    print_row(entry, ("support_0", "support_1"))

    while done < iterations:
        length = min(log_every, iterations - done)
        key, chunk_key = jax.random.split(key)
        carry, metrics_stack = runner(carry, jax.random.split(chunk_key, length), length)
        done += length
        metrics = jax.device_get(metrics_stack)
        entry, strategies = record(done, {
            "grad_norm_0": float(np.mean(np.asarray(metrics["grad_norm_0"]))),
        })
        print_row(entry, ("grad_norm_0", "support_0", "support_1"))

    if writer is not None:
        for player in (0, 1):
            writer.save_params(f"player_{player}", hyperparams, carry[0][player])
    return {"history": history, "params": carry[0], "policy": policy,
            "support_0": strategies[0][0], "weights_0": strategies[0][1],
            "support_1": strategies[1][0], "weights_1": strategies[1][1],
            "payoff_evals": done * per_iteration,
            "train_seconds": runner.run_seconds, "compile_seconds": runner.compile_seconds}


def hyperparams_from_config(game: ZeroSumGame, config, args) -> PathwisePolicyHyperparams:
    lo, hi = action_bounds(game, 0)
    return PathwisePolicyHyperparams(
        action_dim=lo.shape[0],
        hidden_dims=tuple(config.network.hidden_dims),
        noise_dim=args.noise_dim,
        activation=args.activation,
        normalization=args.normalization,
        low=tuple(float(x) for x in lo),
        high=tuple(float(x) for x in hi),
        optimizer=args.optimizer,
        learning_rate=args.lr,
        optimism=args.optimism,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        smooth=args.smooth,
        smooth_scale=args.smooth_scale,
    )


def add_arguments(ap) -> None:
    """This module's flags. Defaults are the reference implementation's for this variant,
    which is why several differ from `randomized_policy.add_arguments`."""
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--optimizer", default="optimistic",
                    help="'optimistic' is this variant's OGD; any other name is looked up "
                         "in training.optimizers.OPTIMIZERS")
    ap.add_argument("--optimism", type=float, default=1.0,
                    help="OGD's negative-momentum coefficient (only used by 'optimistic')")
    ap.add_argument("--max-grad-norm", type=float, default=0.0,
                    help="clip the gradient to this global norm; 0 disables clipping")
    ap.add_argument("--noise-dim", type=int, default=16, help="latent noise fed to the policy")
    ap.add_argument("--activation", default="mish", help="see nets.activations.ACTIVATIONS")
    ap.add_argument("--normalization", default="rms_norm",
                    help="see nets.normalization.NORMALIZATIONS")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="action pairs averaged per gradient estimate")
    ap.add_argument("--smooth", type=int, default=0,
                    help="0 keeps the exact pathwise gradient; N > 0 replaces it with the "
                         "zeroth-order score-function estimator over N perturbation samples")
    ap.add_argument("--smooth-scale", type=float, default=0.1,
                    help="smoothing radius used by --smooth (the papers' sigma)")


def main() -> None:
    ap = neural_parser(__doc__)
    add_arguments(ap)
    args = ap.parse_args()

    game, game_config, config = load_run(args.config)
    oracle = GridOracle(game, points=args.grid)
    hyperparams = hyperparams_from_config(game, config, args)

    gradient = ("exact pathwise (reparametrization)" if not hyperparams.smooth
                else f"zeroth-order score function, {hyperparams.smooth} sample(s), "
                     f"sigma={hyperparams.smooth_scale}")
    optimizer = (f"OGD alpha={hyperparams.learning_rate} beta={hyperparams.optimism}"
                 if hyperparams.optimizer == "optimistic"
                 else f"{hyperparams.optimizer} lr={hyperparams.learning_rate}")
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"policy  : randomized network a=f(o,z), noise {hyperparams.noise_dim}, "
          f"hidden {hyperparams.hidden_dims}, {hyperparams.normalization}/"
          f"{hyperparams.activation} (no density -- no magnet, no ratio)")
    print(f"solver  : {gradient}  |  {optimizer}  |  batch={hyperparams.batch_size}  "
          f"({hyperparams.payoff_evals_per_iteration()} payoff evaluations/iteration)\n")

    meta = {"algorithm": "randomized_policy_pathwise", "config": args.config,
            "hyperparams": hyperparams.to_dict(),
            **{k: v for k, v in vars(args).items() if k != "config"}}
    writer = RunWriter(args.checkpoint_dir, meta) if args.checkpoint_dir else None
    result = run_pathwise(game, oracle, hyperparams, iterations=args.iters,
                          log_every=args.log_every, samples=args.samples, seed=args.seed,
                          writer=writer)

    last = result["history"][-1]
    print(f"\nfinal exploitability {last['expl']:+.5f}  |  "
          f"{result['payoff_evals']} payoff evaluations")
    report_final(oracle, result["support_0"], result["weights_0"],
                 result["support_1"], result["weights_1"])
    if writer is not None:
        writer.finish({"final": last, "payoff_evals": result["payoff_evals"]})
        print(f"run -> {writer.directory}")
    elif args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": meta, "history": result["history"]}, indent=2))
        print(f"saved history -> {args.out}")


if __name__ == "__main__":
    main()
