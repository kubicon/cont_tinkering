"""Neural fictitious self-play.

NFSP (Heinrich & Silver, 2016) is the established deep baseline whose guarantee is
about the *average* iterate: each player keeps two policies -- a reinforcement-learned
best response `beta` to what the opponents have been doing, and a supervised `pi` that
imitates its own past best responses -- and plays the anticipatory mixture
`eta * beta + (1 - eta) * pi`. `pi` approximates the fictitious-play average, which in a
zero-sum game converges to a Nash equilibrium; `beta` does not and is not supposed to.
Both are reported here, because "NFSP converged" is a statement about `pi` only.

What is reused, and what is a deviation:

  * the best response is `br_oracle.train_best_response`, i.e. the repo's own PPO
    trainer against a fixed opponent sampler, restarted from a fresh initialization
    each round (see below). **Deviation**: the paper uses DQN with a circular replay
    buffer; this is PPO on fresh on-policy rollouts. The structure NFSP actually rests
    on -- best response to the opponent's average, supervised imitation of one's own
    best responses -- is unchanged, and PPO is what this repo has; a DQN oracle would
    be a different implementation, not a different algorithm.
  * the average policy is a `MixtureActorCritic` fit by maximum likelihood to a
    reservoir of the best response's own actions (`--average-head mixture`), which is
    the paper's supervised-learning step. `--average-head reservoir` skips the network
    and uses the reservoir's empirical distribution directly: the same algorithm with a
    perfect density estimator, which separates "fictitious play did not converge" from
    "the density estimator did not fit".

The average net is a density estimator here, not the object under test, so it is given
its own capacity (`--average-components`, default 8) rather than the config's
`network.num_components`: handicapping NFSP's function approximator the same way the
method under test is handicapped would be measuring the wrong thing.

**Why the best response is not warm-started.** The obvious optimization -- start each
round's `beta` from the previous round's, so it improves continuously rather than
relearning from scratch -- is what the paper's DQN oracle effectively does, and it is
wrong for this one. DQN keeps exploring however peaked its Q-net gets, because
epsilon-greedy is external to the network; an on-policy Gaussian policy explores only
through its own scale, `br_hyperparams` zeroes `gaussian_entropy_coef` (see
`br_oracle`'s docstring for why a best response must be unregularized), and so within a
few rounds `beta` collapses to a scale of ~0.01 and can never re-widen. Every subsequent
"best response" is then local gradient ascent from last round's answer.

Measured on `configs/circle.yaml`, against the round-39 opponent of a warm-started run
and on the default budget, the warm-started oracle reaches +0.356 where the exact grid
best response is +0.703; cold-started on the same opponent and budget it reaches +0.700.
Over a run this is fatal rather than merely lossy: the per-round best responses drift by
a few hundredths instead of leaping across the action space, the reservoir becomes a
narrow smear rather than the sweep fictitious play averages over, and exploitability
plateaus (~1.4 on `circle`, against PSRO's ~0.000 out of the same oracle -- the absence
of a warm start being the only structural difference between the two).

**The best-response budget, by contrast, is not the binding knob.** Measured on
`configs/two_point.yaml` against the exact Nash (whose exact best-response value is
0.000, since every deviation is worthless at an equilibrium), the cold-started PPO
oracle reaches -0.005 after ~1000 iterations and only -0.72 after 400, so a round whose
"best response" is 100 iterations long is running fictitious play on something that is
not a best response. `--br-steps 50 --br-epochs 20` is 1000 iterations per round and is
the default for that reason; treat lowering it as changing the algorithm. But raising it
does not rescue a warm-started run -- the 1000 iterations above were enough to be exact
from a cold start and left a 0.35 gap from a warm one.

Usage:
    python -m baselines.neural.nfsp configs/two_point.yaml --rounds 30
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.base import ZeroSumGame
from training.config import MixturePPOHyperparams
from training.mixture import build_mixture_network, mixture_marginal_log_prob, sample_mixture_actions
from training.ppo import create_train_state

from ..common import GridOracle, StrategyPair
from . import br_oracle as bo
from .common import RunWriter, empirical_strategy, load_run, neural_parser, print_row, \
    report_final, strategy_row


class Reservoir:
    """Reservoir sample of the best response's actions -- the supervised set.

    Reservoir rather than a sliding window on purpose: the supervised target is the
    *average* of every best response the player has ever played, so early rounds must
    keep their weight instead of ageing out. A fixed-capacity device array with a live
    count is what lets the fitting step keep one compiled shape across rounds.
    """

    def __init__(self, capacity: int, dim: int, seed: int = 0):
        self.capacity = capacity
        self.buffer = np.zeros((capacity, dim), dtype=np.float64)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, actions) -> None:
        for action in np.asarray(actions, dtype=np.float64):
            self.seen += 1
            if self.size < self.capacity:
                self.buffer[self.size] = action
                self.size += 1
            else:
                # Classic algorithm R: the `n`-th item replaces a uniformly chosen slot
                # with probability `capacity / n`, which keeps the buffer a uniform
                # sample of everything seen.
                j = self.rng.integers(0, self.seen)
                if j < self.capacity:
                    self.buffer[j] = action

    def strategy(self) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("empty reservoir")
        return empirical_strategy(self.buffer[:self.size])


def fit_average_policy(
    game: ZeroSumGame,
    player: int,
    hyperparams: MixturePPOHyperparams,
    reservoir: Reservoir,
    steps: int,
    batch_size: int,
    key: chex.PRNGKey,
    init_params=None,
):
    """Maximum-likelihood fit of a mixture policy to the reservoir -- NFSP's SL step.

    The likelihood is the mixture's *marginal* density at the stored action
    (`training.mixture.mixture_marginal_log_prob`), i.e. plain behaviour cloning with
    the component identity marginalized out, since the reservoir records what was
    played and not which component played it.
    """
    network = build_mixture_network(hyperparams)
    obs = game.observation(player, bo.OBSERVATION_KEY)
    init_key, train_key = jax.random.split(key)
    params = network.init(init_key, obs) if init_params is None else init_params
    state = create_train_state(network, params, hyperparams)

    buffer = jnp.asarray(reservoir.buffer)
    size = jnp.asarray(reservoir.size)

    def loss_fn(params, actions):
        logits, means, scale_trils, _ = network.apply(params, obs)
        mask = jnp.ones_like(logits, dtype=bool)
        log_probs = jax.vmap(
            lambda a: mixture_marginal_log_prob(logits, means, scale_trils, mask, a, 0)
        )(actions)
        return -jnp.mean(log_probs)

    grad_fn = jax.value_and_grad(loss_fn)

    def step(carry, step_key):
        state = carry
        index = jax.random.randint(step_key, (batch_size,), 0, size)
        loss, grads = grad_fn(state.params, buffer[index])
        return state.apply_gradients(grads=grads), loss

    state, losses = jax.jit(
        lambda s, keys: jax.lax.scan(step, s, keys)
    )(state, jax.random.split(train_key, steps))
    return network, state.params, float(jnp.mean(losses))


def run_nfsp(
    game: ZeroSumGame,
    oracle: GridOracle,
    config,
    rounds: int = 30,
    br_steps: int = 50,
    br_epochs: int = 20,
    eta: float = 0.1,
    reservoir_capacity: int = 200_000,
    reservoir_samples: int = 4096,
    sl_steps: int = 400,
    sl_batch: int = 256,
    average_head: str = "mixture",
    average_components: int = 8,
    samples: int = 4096,
    seed: int = 0,
    writer: RunWriter | None = None,
    score: bool = True,
    batch_size: int | None = None,
) -> dict:
    """`rounds` of NFSP. Returns the final average/best-response strategies and history.

    `score=False` skips the exploitability computation per round (scored offline from
    the checkpoints instead) and is what a wall-time comparison should use: measuring
    NFSP's average policy costs a best response of its own.

    `batch_size`, if given, overrides the game config's `ppo.batch_size` (PPO's
    `num_envs`) for the best responses -- the batch every other method in a comparison
    still reads from config.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    key = jax.random.PRNGKey(seed)
    dims = tuple(game.action_space(player).shape[0] for player in (0, 1))
    reservoirs = [Reservoir(reservoir_capacity, dims[player], seed + player) for player in (0, 1)]

    envs_override = {"num_envs": batch_size} if batch_size is not None else {}
    br_hp = [bo.br_hyperparams(game, player, config, **envs_override) for player in (0, 1)]
    sl_hp = [dataclasses.replace(bo.br_hyperparams(game, player, config, **envs_override),
                                 num_components=average_components)
             for player in (0, 1)]

    # Round 0: no best response and no reservoir yet, so both roles start as uniform
    # random play -- PSRO and NFSP both need *some* opponent to respond to first.
    br_policies: list[dict | None] = [None, None]
    average_policies: list[dict | None] = [None, None]
    history = writer.history if writer is not None else []
    # Training time and payoff evaluations, both excluding the metric: one PPO iteration
    # of a best response scores `num_envs` action pairs, and each round trains two.
    train_seconds = 0.0
    evals_per_round = 2 * br_steps * br_epochs * br_hp[0].num_envs

    def strategy_of(policy: dict | None, player: int, key: chex.PRNGKey):
        """`(support, weights)` for one side, whatever kind of policy it is."""
        if policy is None:
            return empirical_strategy(game.action_space(player).sample(key, (samples,)))
        if policy["kind"] == "finite":
            return policy["support"], policy["weights"]
        actions = sample_mixture_actions(
            policy["network"], policy["params"],
            game.observation(player, bo.OBSERVATION_KEY),
            game.action_space(player), key, samples)
        return empirical_strategy(actions)

    def opponent_of(policy: dict | None, player: int, key: chex.PRNGKey) -> bo.Opponent:
        if policy is None:
            return bo.uniform_opponent(game, player, samples, key)
        if policy["kind"] == "finite":
            return bo.finite_opponent(policy["support"], policy["weights"], "average")
        return bo.policy_opponent(game, player, policy["network"], policy["params"],
                                  samples, key, policy["kind"])

    for round_index in range(1, rounds + 1):
        key, *round_keys = jax.random.split(key, 9)

        # Each player best-responds to the opponent's anticipatory mixture. Both sides
        # use the opponents as they stood at the start of the round (simultaneous, not
        # alternating), which is what fictitious play prescribes.
        round_started = time.monotonic()
        opponents = []
        for player in (0, 1):
            br_opp = opponent_of(br_policies[player], player, round_keys[player])
            avg_opp = opponent_of(average_policies[player], player, round_keys[2 + player])
            opponents.append(bo.mix_opponents([br_opp, avg_opp], [eta, 1.0 - eta], samples,
                                              round_keys[4 + player]))

        for player in (0, 1):
            # Cold-started on purpose: see "Why the best response is not warm-started"
            # in the module docstring.
            response = bo.train_best_response(
                game, player, opponents[1 - player], br_hp[player],
                steps=br_steps, epochs=br_epochs, seed=seed + round_index,
            )
            br_policies[player] = {"kind": "br", "network": response.network,
                                   "params": response.params, "mean_reward": response.mean_reward}

            # The supervised set: what this round's best response actually plays.
            actions = sample_mixture_actions(
                response.network, response.params,
                game.observation(player, bo.OBSERVATION_KEY),
                game.action_space(player), round_keys[6 + player], reservoir_samples)
            reservoirs[player].add(actions)

        # ... and the average policy is refit to it.
        sl_losses = []
        for player in (0, 1):
            if average_head == "reservoir":
                support, weights = reservoirs[player].strategy()
                average_policies[player] = {"kind": "finite", "support": support, "weights": weights}
                sl_losses.append(float("nan"))
                continue
            key, fit_key = jax.random.split(key)
            network, params, loss = fit_average_policy(
                game, player, sl_hp[player], reservoirs[player], sl_steps, sl_batch, fit_key,
                init_params=average_policies[player]["params"] if average_policies[player] else None,
            )
            average_policies[player] = {"kind": "average", "network": network, "params": params}
            sl_losses.append(loss)

        train_seconds += time.monotonic() - round_started

        key, *metric_keys = jax.random.split(key, 5)
        avg = [strategy_of(average_policies[p], p, metric_keys[p]) for p in (0, 1)]
        br = [strategy_of(br_policies[p], p, metric_keys[2 + p]) for p in (0, 1)]
        entry = {
            "t": round_index,
            "wall_time": train_seconds,
            "payoff_evals": round_index * evals_per_round,
            "reservoir": reservoirs[0].size,
            "sl_loss_0": sl_losses[0], "sl_loss_1": sl_losses[1],
            "br_reward_0": br_policies[0]["mean_reward"],
        }
        if score:
            entry.update(strategy_row(oracle, avg[0][0], avg[0][1], avg[1][0], avg[1][1]))
            # The best-response pair is not expected to converge; it is logged because a
            # *rising* `br_expl` beside a falling `expl` is what NFSP working looks like.
            entry["br_expl"] = float(
                oracle.exploitability(br[0][0], br[0][1], br[1][0], br[1][1]))
        if writer is None:
            history.append(entry)
        else:
            writer.record(entry, StrategyPair(
                t=round_index, support_0=avg[0][0], weights_0=avg[0][1],
                support_1=avg[1][0], weights_1=avg[1][1],
                extra={"br_support_0": br[0][0], "br_support_1": br[1][0]},
            ))
        print_row(entry, ("br_expl", "reservoir", "support_0", "support_1"))

    if writer is not None:
        for player in (0, 1):
            if average_policies[player] and average_policies[player]["kind"] == "average":
                writer.save_params(f"average_{player}", sl_hp[player],
                                   average_policies[player]["params"])
            if br_policies[player]:
                writer.save_params(f"br_{player}", br_hp[player], br_policies[player]["params"])

    return {"history": history, "average": avg, "br": br,
            "average_policies": average_policies, "br_policies": br_policies,
            "train_seconds": train_seconds}


def main() -> None:
    ap = neural_parser(__doc__)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--br-steps", type=int, default=50, help="training chunks per best response")
    ap.add_argument("--br-epochs", type=int, default=20, help="iterations per chunk")
    ap.add_argument("--eta", type=float, default=0.1, help="anticipatory parameter")
    ap.add_argument("--reservoir-capacity", type=int, default=200_000)
    ap.add_argument("--reservoir-samples", type=int, default=4096,
                    help="best-response actions added to the reservoir each round")
    ap.add_argument("--sl-steps", type=int, default=400, help="supervised steps per round")
    ap.add_argument("--sl-batch", type=int, default=256)
    ap.add_argument("--average-head", choices=("mixture", "reservoir"), default="mixture")
    ap.add_argument("--average-components", type=int, default=8)
    args = ap.parse_args()

    game, game_config, config = load_run(args.config)
    oracle = GridOracle(game, points=args.grid)
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"solver  : NFSP  {args.rounds} rounds  eta={args.eta}  "
          f"BR {args.br_steps}x{args.br_epochs} PPO iterations/round  "
          f"average head: {args.average_head}"
          f"{f' (K={args.average_components})' if args.average_head == 'mixture' else ''}")
    print(f"metric  : `expl` is the AVERAGE policies'; `br_expl` the best responses'\n")

    meta = {"algorithm": "nfsp", "config": args.config, "grid": args.grid,
            **{k: v for k, v in vars(args).items() if k not in ("config", "grid")}}
    writer = RunWriter(args.checkpoint_dir, meta) if args.checkpoint_dir else None
    result = run_nfsp(
        game, oracle, config, rounds=args.rounds, br_steps=args.br_steps,
        br_epochs=args.br_epochs, eta=args.eta, reservoir_capacity=args.reservoir_capacity,
        reservoir_samples=args.reservoir_samples, sl_steps=args.sl_steps, sl_batch=args.sl_batch,
        average_head=args.average_head, average_components=args.average_components,
        samples=args.samples, seed=args.seed, writer=writer,
    )

    last = result["history"][-1]
    print(f"\nfinal average-policy exploitability {last['expl']:+.5f}  "
          f"(best responses {last['br_expl']:+.5f})  |  {last['wall_time']:.1f}s training")
    (s0, w0), (s1, w1) = result["average"]
    report_final(oracle, s0, w0, s1, w1)
    if writer is not None:
        writer.finish({"final": last})
        print(f"run -> {writer.directory}")
    elif args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": meta, "history": result["history"]}, indent=2))
        print(f"saved history -> {args.out}")


if __name__ == "__main__":
    main()
