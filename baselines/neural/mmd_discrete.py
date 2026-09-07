"""MMD on a discretized action space: our algorithm, a categorical head.

This is the sharpest neural baseline in the repo, because it changes exactly one
thing. The rollouts, the PPO surrogate, the entropy bonus, the trust region, the
magnet and its periodic snapshot, the Polyak-averaged iterate -- all of it is the
mixture method's, with the same coefficients read off the same config section (the
*categorical head's* coefficients, `ppo.category_entropy_coef` /
`ppo.trpo_category_kl_coef` / `ppo.magnet_category_kl_coef`, which is what a policy
that is nothing but a categorical head should inherit). What differs is the policy:
instead of `K` Gaussian components whose means move, the network emits one logit per
cell of a fixed `bins^d` grid over the action box.

That is the trade the mixture head exists to avoid, and running it makes the trade
concrete rather than rhetorical:

  * the categorical policy is *complete* on its grid -- any mixed strategy supported on
    the grid is representable, so the decoy-well trap that catches a K=2 mixture cannot
    catch it for representational reasons;
  * but it has `bins^d` parameters in the head and no notion that neighbouring cells are
    nearby actions, so what it buys in one dimension it pays for immediately in more;
  * and its equilibrium is only ever as accurate as the grid: the Nash of these games
    sits on a peak, and the policy can only put mass on the nearest cell.

One practical note, since it bites immediately: a categorical head collapses. At a
learning rate an order of magnitude above the config's, the policy puts all its mass on
one cell within a few hundred iterations and never recovers -- worse than uniform, and
worse than anything the mixture head does, because a Gaussian component that overshoots
can still slide back while a dead logit gets no gradient. The entropy bonus and the
magnet are what hold the head open, which is the same role they play in the method under
test.

Because the policy *is* a distribution over known points, its exploitability is
computed exactly (no sampling), against a finer deviation grid than the policy's own
(`--grid` vs `--bins`) so the metric never flatters the discretization.

Usage:
    python -m baselines.neural.mmd_discrete configs/two_point.yaml --bins 51 --iters 4000
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.base import ZeroSumGame
from training.config import PPOHyperparams
from training.ppo import ppo_update
from training.trainer_common import create_mixture_train_state, update_target_and_magnet

from ..common import ChunkRunner, GridOracle, StrategyPair, action_bounds
from .common import DiscreteActorCritic, RunWriter, action_grid, discrete_probs, \
    grid_strategy, load_run, neural_parser, print_row, report_final, strategy_row

# See `br_oracle.OBSERVATION_KEY`: the observation of a one-shot game is constant.
_OBSERVATION_KEY = jax.random.PRNGKey(0)


@dataclasses.dataclass(frozen=True)
class DiscreteMMDHyperparams(PPOHyperparams):
    """`PPOHyperparams` plus the discretized head and MMD's regularizers.

    The three coefficients are the categorical-head counterparts of
    `MixturePPOHyperparams`'s six: a policy with no Gaussian factor has no Gaussian
    entropy, Gaussian trust region, or Gaussian magnet to weigh.
    """

    bins: int = 51
    num_actions: int = 51          # `bins ** action_dim`; stored so the head can be rebuilt
    target_tau: float = 0.005      # Polyak coefficient for the averaged iterate
    magnet_interval: int = 1000    # iterations between magnet snapshots
    magnet_kl_coef: float = 0.0    # KL(current || magnet)
    trpo_kl_coef: float = 0.0      # KL(rollout policy || current)

    @classmethod
    def from_dict(cls, data: dict) -> "DiscreteMMDHyperparams":
        data = dict(data)
        data["hidden_dims"] = tuple(data["hidden_dims"])
        return cls(**data)


@chex.dataclass
class DiscreteBatch:
    """One rollout of a categorical policy. `index` is the sampled grid cell."""

    obs: chex.Array          # (B, obs_dim)
    index: chex.Array        # (B,)
    log_prob: chex.Array     # (B,) log pi(index) at rollout time
    logits: chex.Array       # (B, A) rollout-time logits -- the trust region's anchor
    magnet_logits: chex.Array  # (B, A)
    value: chex.Array        # (B,)
    reward: chex.Array       # (B,)


def build_network(hyperparams: DiscreteMMDHyperparams) -> DiscreteActorCritic:
    return DiscreteActorCritic(
        num_actions=hyperparams.num_actions,
        hidden_dims=hyperparams.hidden_dims,
        activation=hyperparams.activation,
        normalization=hyperparams.normalization,
    )


def collect_self_play_batch(game: ZeroSumGame, network: DiscreteActorCritic,
                            params, magnet_params, grids, key: chex.PRNGKey,
                            num_envs: int) -> tuple[DiscreteBatch, DiscreteBatch]:
    """One simultaneous rollout of both categorical policies.

    Mirrors `training.mixture.collect_mixture_self_play_episode`: both sides act from
    their live policies, the magnet's logits at the same observation are recorded
    alongside (the loss needs them, and re-running the magnet network inside the loss
    would evaluate it at parameters that have already moved).
    """
    keys = jax.random.split(key, 4)
    batches = []
    logits_all, index_all, actions = [], [], []
    for player in (0, 1):
        obs = game.observation(player, keys[player], (num_envs,))
        logits, value = jax.vmap(lambda o: network.apply(params[player], o))(obs)
        magnet_logits, _ = jax.vmap(lambda o: network.apply(magnet_params[player], o))(obs)
        index = jax.random.categorical(keys[2 + player], logits)
        log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), index[:, None], axis=-1)[:, 0]
        logits_all.append((obs, logits, magnet_logits, value, log_prob))
        index_all.append(index)
        actions.append(jnp.asarray(grids[player])[index])

    reward = game.payoff_batch(actions[0], actions[1])
    for player in (0, 1):
        obs, logits, magnet_logits, value, log_prob = logits_all[player]
        batches.append(DiscreteBatch(
            obs=obs, index=index_all[player], log_prob=log_prob, logits=logits,
            magnet_logits=magnet_logits, value=value,
            reward=reward if player == 0 else -reward,
        ))
    return batches[0], batches[1]


def build_loss_fn(magnet_kl_coef: float, trpo_kl_coef: float):
    """The MMD loss for a categorical policy, in `training.ppo.ppo_update`'s signature.

    `clip_eps`, `value_coef` and `entropy_coef` arrive from the hyperparams the way
    `ppo_update` passes them; the two KL coefficients are closed over, exactly as
    `training.trainer_common.build_loss_fn` closes over the mixture's six.
    """

    def loss_fn(params, network, batch: DiscreteBatch, clip_eps: float,
                value_coef: float, entropy_coef: float):
        logits, value = jax.vmap(lambda o: network.apply(params, o))(batch.obs)
        log_probs = jax.nn.log_softmax(logits)
        new_log_prob = jnp.take_along_axis(log_probs, batch.index[:, None], axis=-1)[:, 0]
        ratio = jnp.exp(new_log_prob - batch.log_prob)

        advantage = batch.reward - batch.value
        advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
        surrogate = jnp.minimum(ratio * advantage,
                                jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage)
        policy_loss = -jnp.mean(surrogate)
        value_loss = jnp.mean(jnp.square(value - batch.reward))

        probs = jnp.exp(log_probs)
        entropy = jnp.mean(-jnp.sum(probs * log_probs, axis=-1))
        # KL(current || magnet): the magnet pulls the policy back towards its own past,
        # which is what makes this MMD rather than PPO self-play.
        magnet_kl = jnp.mean(jnp.sum(
            probs * (log_probs - jax.nn.log_softmax(batch.magnet_logits)), axis=-1))
        # KL(rollout || current), the direction `mixture_ppo_loss` penalizes.
        old_log_probs = jax.nn.log_softmax(batch.logits)
        trpo_kl = jnp.mean(jnp.sum(
            jnp.exp(old_log_probs) * (old_log_probs - log_probs), axis=-1))

        loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy
                + magnet_kl_coef * magnet_kl + trpo_kl_coef * trpo_kl)
        metrics = {
            "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy, "magnet_kl": magnet_kl, "trpo_kl": trpo_kl,
            "approx_kl": jnp.mean(batch.log_prob - new_log_prob),
            "clip_frac": jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float64)),
        }
        return loss, metrics

    return loss_fn


def run_mmd_discrete(
    game: ZeroSumGame,
    oracle: GridOracle,
    hyperparams: DiscreteMMDHyperparams,
    iterations: int = 4000,
    log_every: int | None = None,
    seed: int = 0,
    writer: RunWriter | None = None,
    score: bool = True,
) -> dict:
    """Self-play MMD with a discretized head. Returns the final strategies and history.

    `score=False` skips the exploitability computation at every logged iteration and
    records only the checkpoint, the training wall-time and the payoff-evaluation count
    -- the fast path a benchmark uses, with scoring done afterwards from the checkpoints
    (`experiments/one_shot_neural/score.py`). One in-training score costs a best response
    over the whole deviation grid against a 4096-point support, which can exceed the
    training it is measuring.
    """
    log_every = log_every or max(iterations // 50, 1)
    network = build_network(hyperparams)
    grids = (action_grid(game, 0, hyperparams.bins), action_grid(game, 1, hyperparams.bins))
    observations = tuple(game.observation(player, _OBSERVATION_KEY) for player in (0, 1))

    key = jax.random.PRNGKey(seed)
    init_keys = jax.random.split(key, 3)
    states = tuple(
        create_mixture_train_state(
            network, network.init(init_keys[player], observations[player]), hyperparams)
        for player in (0, 1)
    )
    key = init_keys[2]
    loss_fn = build_loss_fn(hyperparams.magnet_kl_coef, hyperparams.trpo_kl_coef)

    def train_step(states, step_key):
        params = (states[0].params, states[1].params)
        magnet = (states[0].magnet_params, states[1].magnet_params)
        batch_0, batch_1 = collect_self_play_batch(
            game, network, params, magnet, grids, step_key, hyperparams.num_envs)
        new_states, metrics = [], {}
        for player, batch in ((0, batch_0), (1, batch_1)):
            state, player_metrics = ppo_update(
                states[player], network, batch, hyperparams, loss_fn=loss_fn)
            new_states.append(update_target_and_magnet(state, hyperparams))
            metrics.update({f"{k}_{player}": v for k, v in player_metrics.items()})
        metrics["mean_reward_0"] = jnp.mean(batch_0.reward)
        return tuple(new_states), metrics

    runner = ChunkRunner(lambda n: lambda s, k: jax.lax.scan(lambda c, kk: train_step(c, kk), s, k))

    def strategies(which: str = "params"):
        """Both players' exact strategies: grid, and the policy's probabilities on it."""
        return tuple(
            grid_strategy(grids[player], discrete_probs(network, getattr(states[player], which),
                                                        observations[player]))
            for player in (0, 1)
        )

    # `history` is the writer's own list when there is one, so a run with a
    # checkpoint directory and one without keep the same in-memory history.
    local_history: list[dict] = []
    history = writer.history if writer is not None else local_history

    def record(t: int, metrics: dict | None) -> dict:
        (s0, w0), (s1, w1) = strategies("params")
        (t0, u0), (t1, u1) = strategies("target_params")
        # One self-play rollout per iteration, `num_envs` action pairs scored in it.
        entry = {"t": int(t), "wall_time": runner.run_seconds,
                 "compile_time": runner.compile_seconds,
                 "payoff_evals": int(t) * hyperparams.num_envs}
        if score:
            entry.update(strategy_row(oracle, s0, w0, s1, w1))
            entry["target_expl"] = float(oracle.exploitability(t0, u0, t1, u1))
        if metrics:
            entry.update({k: float(v) for k, v in metrics.items()})
        if writer is not None:
            writer.record(entry, StrategyPair(
                t=int(t), support_0=s0, weights_0=w0, support_1=s1, weights_1=w1,
                extra={"target_weights_0": u0, "target_weights_1": u1},
            ))
        else:
            local_history.append(entry)
        return entry

    entry = record(0, None)
    print_row(entry, ("target_expl", "support_0", "support_1"))

    done = 0
    while done < iterations:
        length = min(log_every, iterations - done)
        key, chunk_key = jax.random.split(key)
        # `runner` accumulates training time and compile time separately; logging,
        # scoring and checkpointing all happen outside its timer.
        states, metrics_stack = runner(states, jax.random.split(chunk_key, length), length)
        done += length
        metrics = {k: float(np.mean(np.asarray(v))) for k, v in jax.device_get(metrics_stack).items()}
        entry = record(done, metrics)
        print_row(entry, ("target_expl", "support_0", "support_1"))

    (s0, w0), (s1, w1) = strategies("params")
    if writer is not None:
        writer.save_params("player_0", hyperparams, states[0].params)
        writer.save_params("player_1", hyperparams, states[1].params)
    return {"history": history, "support_0": s0, "weights_0": w0,
            "support_1": s1, "weights_1": w1, "network": network, "states": states,
            "train_seconds": runner.run_seconds, "compile_seconds": runner.compile_seconds}


def hyperparams_from_config(game: ZeroSumGame, config, bins: int, args) -> DiscreteMMDHyperparams:
    """Read the network/optimizer/ppo sections the way `train.py` does, then take the
    *categorical* head's regularizer coefficients as this policy's."""
    lo, _ = action_bounds(game, 0)
    network, optimizer, ppo = config.network, config.optimizer, config.ppo
    return DiscreteMMDHyperparams(
        action_dim=lo.shape[0],
        hidden_dims=tuple(network.hidden_dims),
        activation=network.activation,
        normalization=network.normalization,
        learning_rate=args.lr if args.lr is not None else optimizer.learning_rate,
        max_grad_norm=optimizer.max_grad_norm,
        optimizer=optimizer.optimizer,
        weight_decay=optimizer.weight_decay,
        clip_eps=ppo.clip_eps,
        value_coef=ppo.value_coef,
        entropy_coef=args.entropy if args.entropy is not None else ppo.category_entropy_coef,
        num_envs=args.batch if args.batch is not None else ppo.batch_size,
        num_epochs=ppo.ppo_epochs,
        bins=bins,
        num_actions=bins ** lo.shape[0],
        target_tau=ppo.target_tau,
        magnet_interval=args.magnet_interval if args.magnet_interval is not None else ppo.magnet_interval,
        magnet_kl_coef=args.magnet_kl if args.magnet_kl is not None else ppo.magnet_category_kl_coef,
        trpo_kl_coef=args.trpo_kl if args.trpo_kl is not None else ppo.trpo_category_kl_coef,
    )


def main() -> None:
    ap = neural_parser(__doc__)
    ap.add_argument("--bins", type=int, default=51, help="cells per action axis in the policy head")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="override optimizer.learning_rate")
    ap.add_argument("--batch", type=int, default=None, help="override ppo.batch_size")
    ap.add_argument("--entropy", type=float, default=None, help="override ppo.category_entropy_coef")
    ap.add_argument("--magnet-kl", type=float, default=None,
                    help="override ppo.magnet_category_kl_coef")
    ap.add_argument("--trpo-kl", type=float, default=None, help="override ppo.trpo_category_kl_coef")
    ap.add_argument("--magnet-interval", type=int, default=None, help="override ppo.magnet_interval")
    args = ap.parse_args()

    game, game_config, config = load_run(args.config)
    oracle = GridOracle(game, points=args.grid)
    hyperparams = hyperparams_from_config(game, config, args.bins, args)

    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"policy  : categorical over {hyperparams.num_actions} cells "
          f"({args.bins} per axis), hidden {hyperparams.hidden_dims}")
    print(f"solver  : MMD  lr={hyperparams.learning_rate}  batch={hyperparams.num_envs}  "
          f"entropy={hyperparams.entropy_coef}  magnet_kl={hyperparams.magnet_kl_coef} "
          f"(every {hyperparams.magnet_interval})  trpo_kl={hyperparams.trpo_kl_coef}")
    print(f"metric  : exact, against a {oracle.grid0.shape[0]}-point deviation grid\n")

    meta = {"algorithm": "mmd_discrete", "config": args.config, "grid": args.grid,
            "bins": args.bins, "iters": args.iters, "seed": args.seed,
            "hyperparams": dataclasses.asdict(hyperparams)}
    writer = RunWriter(args.checkpoint_dir, meta) if args.checkpoint_dir else None
    result = run_mmd_discrete(game, oracle, hyperparams, iterations=args.iters,
                              log_every=args.log_every, seed=args.seed, writer=writer)

    last = result["history"][-1]
    print(f"\nfinal exploitability {last['expl']:+.5f}  |  averaged iterate {last['target_expl']:+.5f}")
    report_final(oracle, result["support_0"], result["weights_0"],
                 result["support_1"], result["weights_1"])
    if writer is not None:
        writer.finish({"final": last})
        print(f"run -> {writer.directory}")
    elif args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": meta, "history": result["history"]}, indent=2))
        print(f"saved history -> {args.out}")


if __name__ == "__main__":
    main()
