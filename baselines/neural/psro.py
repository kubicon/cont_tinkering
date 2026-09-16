"""PSRO: policy-space response oracles.

The deep counterpart of `baselines/double_oracle.py`, and the same algorithm at heart:
keep a finite *population* of policies per player, solve the empirical game between them
exactly, and grow each population by a best response to the opponent's current mixture.
What changes going from double oracle to PSRO is only what the two oracles are --

  * the best response is trained by RL against a *sampler* of the opponent's mixture
    (`br_oracle.train_best_response`, i.e. this repo's PPO trainer) rather than found by
    a global grid argmax, so it is local, noisy, and might not be a best response at all;
  * the empirical payoff matrix is estimated by Monte-Carlo play between population
    members rather than evaluated exactly, so the meta-solver's LP is fed noisy numbers.

Both of those are the honest cost of not having an exact oracle, and they are what the
metrics here separate: `expl` is the true exploitability of the meta-strategy (exact
deviation grid, so it is not fooled by the population's own noise) while `rl_gap` is
what PSRO itself believes it has left to gain, from the RL responses' realized payoffs.
`expl` >> `rl_gap` means the RL oracle is the bottleneck, not the population.

The meta-solver is the same LP `double_oracle` uses, so a PSRO run and a double-oracle
run on the same game differ *only* in their oracle -- which is the comparison worth
having. `--meta-solver uniform` replaces it with uniform weights over the population
(self-play-ish), the standard ablation showing what the meta-solve buys.

**Budget the oracle properly.** Each round's best response is what the whole algorithm
rests on: on `configs/two_point.yaml` the PPO oracle needs on the order of 1000 iterations
to actually reach the best-response value, so the default is `--br-steps 100 --br-epochs 20`
(2000). A cheap oracle shows up as `rl_gap` collapsing to nearly zero while `expl` stays
high -- PSRO believing it has converged because its responses stopped improving, which is a
statement about the oracle and not about the population.

Usage:
    python -m baselines.neural.psro configs/two_point.yaml --rounds 12
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
from training.mixture import build_mixture_network, sample_mixture_actions

from ..common import GridOracle, StrategyPair
from ..double_oracle import solve_matrix_game
from . import br_oracle as bo
from .common import RunWriter, load_run, neural_parser, print_row, report_final, strategy_row


def _pairwise_payoff_fn(game: ZeroSumGame):
    """`(actions_0, actions_1) -> E[u]` over the outer product, jitted once.

    See `br_oracle.payoff_estimate` for why the outer product rather than paired rows;
    this is that function, compiled, because PSRO calls it once per population pair per
    round.
    """

    @jax.jit
    def estimate(actions_0, actions_1):
        return jnp.mean(jax.vmap(lambda x: jax.vmap(lambda y: game.payoff(x, y))(actions_1))(actions_0))

    return estimate


class EmpiricalGame:
    """The population payoff matrix, filled in one row/column at a time.

    Each member's action sample is drawn once and kept: a population member is a frozen
    policy, so re-sampling it every round would add noise to entries that are supposed
    to be fixed, and the meta-solver's LP would chase that noise.
    """

    def __init__(self, game: ZeroSumGame, estimate, samples: int):
        self.game = game
        self.estimate = estimate
        self.samples = samples
        self.actions: tuple[list, list] = ([], [])
        self.payoff = np.zeros((0, 0))

    def add_policy(self, player: int, network, params, key: chex.PRNGKey) -> None:
        actions = sample_mixture_actions(
            network, params, self.game.observation(player, bo.OBSERVATION_KEY),
            self.game.action_space(player), key, self.samples)
        self.actions[player].append(actions)

    def refresh(self) -> np.ndarray:
        """Extend the payoff matrix to cover every population pair, and return it."""
        rows, cols = len(self.actions[0]), len(self.actions[1])
        payoff = np.zeros((rows, cols))
        payoff[:self.payoff.shape[0], :self.payoff.shape[1]] = self.payoff
        for i in range(rows):
            for j in range(cols):
                if i < self.payoff.shape[0] and j < self.payoff.shape[1]:
                    continue
                payoff[i, j] = float(self.estimate(self.actions[0][i], self.actions[1][j]))
        self.payoff = payoff
        return payoff

    def meta_strategy_pair(self, weights_0, weights_1) -> tuple:
        """The two meta-strategies as finitely supported strategies.

        Each member contributes its own action sample, reweighted by its meta-probability
        -- so the snapshot carries the *exact* meta weights and only the members'
        sampling error, not a second round of sampling on top.
        """
        supports, weights = [], []
        for player, meta in ((0, weights_0), (1, weights_1)):
            actions = np.concatenate([np.asarray(a, dtype=np.float64) for a in self.actions[player]])
            per_member = np.repeat(np.asarray(meta, dtype=np.float64) / self.samples, self.samples)
            supports.append(actions)
            weights.append(per_member)
        return supports[0], weights[0], supports[1], weights[1]


def run_psro(
    game: ZeroSumGame,
    oracle: GridOracle,
    config,
    rounds: int = 12,
    br_steps: int = 100,
    br_epochs: int = 20,
    payoff_samples: int = 256,
    meta_solver: str = "nash",
    seed: int = 0,
    writer: RunWriter | None = None,
    score: bool = True,
    batch_size: int | None = None,
) -> dict:
    """`rounds` of PSRO. Returns the final meta-strategies, populations, and history.

    `score=False` skips the true-exploitability computation per round (scored offline
    from the checkpoints instead); `rl_gap`, which PSRO computes anyway, is kept.

    `batch_size`, if given, overrides the game config's `ppo.batch_size` (PPO's
    `num_envs`) for the best responses -- the batch every other method in a comparison
    still reads from config.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")

    def solve_meta(payoff: np.ndarray):
        """The meta-strategies over the current populations, and the meta-game's value."""
        if meta_solver == "nash":
            return solve_matrix_game(payoff)
        if meta_solver == "uniform":
            weights_0 = np.full(payoff.shape[0], 1.0 / payoff.shape[0])
            weights_1 = np.full(payoff.shape[1], 1.0 / payoff.shape[1])
            return weights_0, weights_1, float(weights_0 @ payoff @ weights_1)
        raise ValueError(f"unknown meta_solver {meta_solver!r} (choices: nash, uniform)")

    key = jax.random.PRNGKey(seed)
    envs_override = {"num_envs": batch_size} if batch_size is not None else {}
    hyperparams = [bo.br_hyperparams(game, player, config, **envs_override) for player in (0, 1)]
    networks = [build_mixture_network(hyperparams[player]) for player in (0, 1)]
    empirical = EmpiricalGame(game, _pairwise_payoff_fn(game), payoff_samples)

    # Seed each population with one untrained policy -- PSRO's usual random initial
    # strategy, and the only thing available before any best response exists.
    populations: tuple[list, list] = ([], [])
    for player in (0, 1):
        key, init_key, sample_key = jax.random.split(key, 3)
        params = networks[player].init(
            init_key, game.observation(player, bo.OBSERVATION_KEY))
        populations[player].append(params)
        empirical.add_policy(player, networks[player], params, sample_key)

    history = writer.history if writer is not None else []
    meta = ([1.0], [1.0])
    # Training time and payoff evaluations, both excluding the metric. Two costs per
    # round: the best responses (`num_envs` pairs per PPO iteration) and the empirical
    # payoff matrix, whose every new population pair is an outer product of samples.
    train_seconds = 0.0
    payoff_evals = payoff_samples ** 2      # the seed policies' own pair
    br_evals_per_round = 2 * br_steps * br_epochs * hyperparams[0].num_envs

    for round_index in range(1, rounds + 1):
        round_started = time.monotonic()
        before = (len(empirical.actions[0]), len(empirical.actions[1]))
        payoff = empirical.refresh()
        weights_0, weights_1, value = solve_meta(payoff)
        meta = (list(map(float, weights_0)), list(map(float, weights_1)))

        # Each player best-responds to the opponent's meta-mixture over its population.
        responses = []
        for player in (0, 1):
            key, opponent_key = jax.random.split(key)
            opponent = bo.population_opponent(
                game, 1 - player, networks[1 - player], populations[1 - player],
                weights_1 if player == 0 else weights_0, payoff_samples, opponent_key)
            responses.append(bo.train_best_response(
                game, player, opponent, hyperparams[player],
                steps=br_steps, epochs=br_epochs, seed=seed + round_index))

        # What PSRO believes is left on the table: each response's realized payoff
        # against the opponent's meta-mixture, minus the meta-game's own value. Both
        # responses are measured *before* they join the population.
        rl_gap = (responses[0].mean_reward - value) + (responses[1].mean_reward + value)
        train_seconds += time.monotonic() - round_started
        payoff_evals += br_evals_per_round + (
            before[0] * before[1] - (before[0] - 1) * (before[1] - 1)
            if round_index > 1 else 0) * payoff_samples ** 2

        s0, w0, s1, w1 = empirical.meta_strategy_pair(weights_0, weights_1)
        entry = {
            "t": round_index,
            "wall_time": train_seconds,
            "payoff_evals": payoff_evals,
            "rl_gap": float(rl_gap),
            "meta_value": float(value),
            "population_0": len(populations[0]),
            "population_1": len(populations[1]),
        }
        if score:
            entry.update(strategy_row(oracle, s0, w0, s1, w1))
        if writer is not None:
            writer.record(entry, StrategyPair(
                t=round_index, support_0=s0, weights_0=w0, support_1=s1, weights_1=w1,
                extra={"meta_weights_0": np.asarray(meta[0]), "meta_weights_1": np.asarray(meta[1])},
            ))
        else:
            history.append(entry)
        print_row(entry, ("rl_gap", "population_0", "population_1"))

        for player in (0, 1):
            key, sample_key = jax.random.split(key)
            populations[player].append(responses[player].params)
            empirical.add_policy(player, networks[player], responses[player].params, sample_key)

    # One last meta-solve, so the reported strategy uses the responses just added.
    payoff = empirical.refresh()
    weights_0, weights_1, value = solve_meta(payoff)
    s0, w0, s1, w1 = empirical.meta_strategy_pair(weights_0, weights_1)
    final = {"t": rounds + 1, "wall_time": train_seconds, "payoff_evals": payoff_evals,
             "rl_gap": float("nan"), "meta_value": float(value),
             "population_0": len(populations[0]), "population_1": len(populations[1])}
    if score:
        final.update(strategy_row(oracle, s0, w0, s1, w1))
    if writer is not None:
        writer.record(final, StrategyPair(
            t=rounds + 1, support_0=s0, weights_0=w0, support_1=s1, weights_1=w1,
            extra={"meta_weights_0": weights_0, "meta_weights_1": weights_1}))
        for player in (0, 1):
            for index, params in enumerate(populations[player]):
                writer.save_params(f"player{player}_policy{index}", hyperparams[player], params)
    else:
        history.append(final)
    print_row(final, ("meta_value", "population_0", "population_1"))

    return {"history": history, "support_0": s0, "weights_0": w0, "support_1": s1,
            "weights_1": w1, "populations": populations, "payoff": payoff,
            "meta": (weights_0, weights_1), "train_seconds": train_seconds}


def main() -> None:
    ap = neural_parser(__doc__)
    ap.add_argument("--rounds", type=int, default=12, help="best responses added per player")
    ap.add_argument("--br-steps", type=int, default=100, help="training chunks per best response")
    ap.add_argument("--br-epochs", type=int, default=20, help="iterations per chunk")
    ap.add_argument("--payoff-samples", type=int, default=256,
                    help="actions sampled per population member for the empirical payoff matrix")
    ap.add_argument("--meta-solver", choices=("nash", "uniform"), default="nash")
    args = ap.parse_args()

    game, game_config, config = load_run(args.config)
    oracle = GridOracle(game, points=args.grid)
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"solver  : PSRO  {args.rounds} rounds  meta-solver: {args.meta_solver}  "
          f"BR {args.br_steps}x{args.br_epochs} PPO iterations  "
          f"payoff samples {args.payoff_samples}")
    print(f"metric  : `expl` is the meta-strategy's true exploitability; "
          f"`rl_gap` what the RL responses claim\n")

    meta = {"algorithm": "psro", "config": args.config,
            **{k: v for k, v in vars(args).items() if k != "config"}}
    writer = RunWriter(args.checkpoint_dir, meta) if args.checkpoint_dir else None
    result = run_psro(game, oracle, config, rounds=args.rounds, br_steps=args.br_steps,
                      br_epochs=args.br_epochs, payoff_samples=args.payoff_samples,
                      meta_solver=args.meta_solver, seed=args.seed, writer=writer)

    last = result["history"][-1]
    print(f"\nfinal meta-strategy exploitability {last['expl']:+.5f}  "
          f"|  population {last['population_0']}/{last['population_1']}")
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
