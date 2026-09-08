"""PSRO on a game tree: policy-space response oracles for Kuhn, Leduc and Blotto.

The algorithm is `baselines/neural/psro.py`'s, unchanged: keep a population of
policies per player, solve the empirical game between them exactly with the same
LP `double_oracle` uses, and grow each population by an RL best response to the
opponent's meta-strategy. What the tree changes is only what a "policy" and a
"payoff entry" are:

  * a population member is a `MixtureActorCritic` over *infosets* rather than a
    distribution over a single action, so it is not summarizable as a support and
    a weight vector -- it is carried as parameters and played;
  * a payoff-matrix entry is estimated by *playing hands* between two members
    (`training.best_response.PairEvaluator`, the same rollout `best_response.py`
    measures with) instead of by an outer product over sampled actions;
  * the meta-strategy is a mixture over policies, drawn once per hand. That is a
    mixed strategy, not a behavioral one, and the two are different objects --
    which is exactly why `expl` on Kuhn goes through
    `games.kuhn_best_response.mix_strategies` rather than averaging the members'
    strategy tables.

**The columns.** `rl_gap` is what PSRO itself believes is left on the table (its
two responses' realized payoffs against the meta-strategy, minus the meta-game's
value) and is available on every game. `expl` is the meta-strategy's true
exploitability and exists only on Kuhn, where an exact tree best response does
(`baselines/neural/sequential_scoring.py`); `--score-every N` adds `expl_lb`, an
RL best-response lower bound, on the games where it does not. `expl` >> `rl_gap`
means the RL oracle is the bottleneck, not the population.

**Budget the oracle.** Everything rests on the inner best response being one; a
cheap oracle shows up as `rl_gap` collapsing to nearly zero while the population
keeps growing and `expl` does not fall. The defaults here (`--br-steps 50
--br-epochs 20`, i.e. 1000 PPO iterations per response) are Kuhn-sized; Leduc's
tree is far larger and wants several times that, the same way
`configs/leduc.yaml` gives self-play 200k iterations where `configs/kuhn.yaml`
gives 16k.

Usage:
    python -m baselines.neural.sequential_psro configs/kuhn.yaml --rounds 12
    python -m baselines.neural.sequential_psro configs/leduc.yaml --rounds 10 --score-every 5
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax
import numpy as np

from games.sequential import SequentialZeroSumGame
from training.best_response import PairEvaluator

from ..double_oracle import solve_matrix_game
from . import sequential_oracle as so
from .common import RunWriter, print_row
from .sequential_common import load_sequential_run, sequential_parser
from .sequential_run import Budget, mean_columns
from .sequential_scoring import build_scorer


class SequentialEmpiricalGame:
    """The population payoff matrix, filled in one row/column at a time by playing.

    Entries are Monte-Carlo -- `episodes` hands per pair -- and, once measured,
    are kept: a population member is a frozen policy, so re-playing old pairs
    every round would jitter entries that are supposed to be fixed and the
    meta-solver's LP would chase that noise. The same choice `psro.py` makes for
    its sampled actions, for the same reason.

    One `PairEvaluator` serves every entry: the networks are fixed for the run
    and the parameters are arguments, so the rollout is compiled once no matter
    how large the populations get.
    """

    def __init__(self, game: SequentialZeroSumGame, networks, episodes: int = 20_000):
        self.game = game
        self.episodes = episodes
        self.evaluator = PairEvaluator(game, networks)
        self.members: tuple[list, list] = ([], [])
        self.payoff = np.zeros((0, 0))
        # Entries the last `refresh` actually played out, for the cost accounting:
        # the only caller that can know this is the one filling the matrix.
        self.last_new_entries = 0

    def add_policy(self, player: int, params) -> None:
        self.members[player].append(params)

    def refresh(self, key) -> np.ndarray:
        """Extend the payoff matrix to cover every population pair, and return it."""
        rows, cols = len(self.members[0]), len(self.members[1])
        payoff = np.zeros((rows, cols))
        payoff[:self.payoff.shape[0], :self.payoff.shape[1]] = self.payoff
        played = 0
        for i in range(rows):
            for j in range(cols):
                if i < self.payoff.shape[0] and j < self.payoff.shape[1]:
                    continue
                key, entry_key = jax.random.split(key)
                payoff[i, j] = self.evaluator.evaluate(
                    (self.members[0][i], self.members[1][j]), 0, entry_key,
                    num_episodes=self.episodes,
                ).value
                played += 1
        self.payoff = payoff
        self.last_new_entries = played
        return payoff


def run_sequential_psro(
    game: SequentialZeroSumGame,
    config,
    rounds: int = 12,
    br_steps: int = 50,
    br_epochs: int = 20,
    payoff_episodes: int = 20_000,
    meta_solver: str = "nash",
    seed: int = 0,
    scorer=None,
    writer: RunWriter | None = None,
    checkpoint_fn=None,
) -> dict:
    """`rounds` of PSRO on a tree. Returns the meta-strategies, populations and history.

    `checkpoint_fn(step, entries, arrays)` -- `SequentialRunLog.checkpoint`'s
    signature -- is called once per round with the whole population and the
    meta-weights that go with it. Once per *round*, because a population without
    its meta-weights is not a strategy: the two have to be written together or
    the checkpoint cannot be played.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")

    def solve_meta(payoff: np.ndarray):
        if meta_solver == "nash":
            return solve_matrix_game(payoff)
        if meta_solver == "uniform":
            weights_0 = np.full(payoff.shape[0], 1.0 / payoff.shape[0])
            weights_1 = np.full(payoff.shape[1], 1.0 / payoff.shape[1])
            return weights_0, weights_1, float(weights_0 @ payoff @ weights_1)
        raise ValueError(f"unknown meta_solver {meta_solver!r} (choices: nash, uniform)")

    key = jax.random.PRNGKey(seed)
    hyperparams = [so.br_hyperparams(game, player, config) for player in (0, 1)]
    # The whole run shares one architecture per player, so the seed policies'
    # networks are the networks -- see sequential_oracle's docstring.
    seeds = [so.initial_policy(game, player, hyperparams[player], seed + player) for player in (0, 1)]
    networks = (seeds[0].network, seeds[1].network)
    empirical = SequentialEmpiricalGame(game, networks, episodes=payoff_episodes)

    populations: tuple[list, list] = ([], [])
    for player in (0, 1):
        populations[player].append(seeds[player].params[0])
        empirical.add_policy(player, seeds[player].params[0])

    def meta_mixture(player: int, weights) -> so.PolicyMixture:
        return so.population(hyperparams[player], populations[player], weights,
                             network=networks[player], label=f"meta(player {player})")

    history = writer.history if writer is not None else []
    train_seconds = 0.0
    # Two costs per round, both in hands played: the best responses (`num_envs`
    # per PPO iteration, two responses -- charged exactly, from their own
    # histories) and the payoff-matrix entries (charged at the running mean
    # episode length, since `PairEvaluator` reports payoffs and not lengths).
    budget = Budget(default_episode_length=float(game.max_steps))
    mixtures = (meta_mixture(0, [1.0]), meta_mixture(1, [1.0]))

    def checkpoint(step: int, weights_0, weights_1) -> None:
        if checkpoint_fn is None:
            return
        checkpoint_fn(
            step,
            {f"player{player}_policy{index}": (hyperparams[player], params)
             for player in (0, 1) for index, params in enumerate(populations[player])},
            {"meta_weights_0": np.asarray(weights_0), "meta_weights_1": np.asarray(weights_1),
             "payoff": empirical.payoff},
        )

    for round_index in range(1, rounds + 1):
        round_started = time.monotonic()
        key, matrix_key = jax.random.split(key)
        payoff = empirical.refresh(matrix_key)
        weights_0, weights_1, value = solve_meta(payoff)
        mixtures = (meta_mixture(0, weights_0), meta_mixture(1, weights_1))

        responses = [
            so.train_sequential_best_response(
                game, player, mixtures[1 - player], hyperparams[player],
                steps=br_steps, epochs=br_epochs, seed=seed + round_index)
            for player in (0, 1)
        ]
        # What PSRO believes is left: each response's realized payoff against the
        # opponent's meta-mixture, minus the meta-game's own value. Both are
        # measured before the responses join the population.
        rl_gap = (responses[0].br_value - value) + (responses[1].br_value + value)
        train_seconds += time.monotonic() - round_started
        for player in (0, 1):
            budget.add_training(responses[player].history, hyperparams[player].num_envs)
        budget.add_episodes(empirical.last_new_entries * payoff_episodes)

        entry = {
            "t": round_index,
            "wall_time": train_seconds,
            **budget.row(),
            "rl_gap": float(rl_gap),
            "meta_value": float(value),
            "population_0": len(populations[0]),
            "population_1": len(populations[1]),
            # The optimizer's own numbers, averaged over each response's whole
            # training: this round's oracle, not the algorithm's state.
            **mean_columns(responses[0].history, prefix="br0_"),
            **mean_columns(responses[1].history, prefix="br1_"),
        }
        if scorer is not None:
            entry.update(scorer(mixtures, round_index))
        (writer.record(entry) if writer is not None else history.append(entry))
        print_row(entry, ("h2h", "rl_gap", "expl_lb", "population_0", "population_1"))
        checkpoint(round_index, weights_0, weights_1)

        for player in (0, 1):
            populations[player].append(responses[player].params)
            empirical.add_policy(player, responses[player].params)

    # One last meta-solve, so the reported strategy uses the responses just added.
    # Its matrix entries are algorithm work like any other round's, so they are on
    # the training clock too.
    final_started = time.monotonic()
    key, matrix_key = jax.random.split(key)
    payoff = empirical.refresh(matrix_key)
    weights_0, weights_1, value = solve_meta(payoff)
    mixtures = (meta_mixture(0, weights_0), meta_mixture(1, weights_1))
    train_seconds += time.monotonic() - final_started
    budget.add_episodes(empirical.last_new_entries * payoff_episodes)
    final = {"t": rounds + 1, "wall_time": train_seconds, **budget.row(),
             "rl_gap": float("nan"), "meta_value": float(value),
             "population_0": len(populations[0]), "population_1": len(populations[1])}
    if scorer is not None:
        final.update(scorer(mixtures, rounds + 1, force=True))
    (writer.record(final) if writer is not None else history.append(final))
    print_row(final, ("h2h", "meta_value", "expl_lb", "population_0", "population_1"))
    checkpoint(rounds + 1, weights_0, weights_1)

    if writer is not None:
        for player in (0, 1):
            for index, params in enumerate(populations[player]):
                writer.save_params(f"player{player}_policy{index}", hyperparams[player], params)
        writer.save_arrays("meta", meta_weights_0=weights_0, meta_weights_1=weights_1,
                           payoff=empirical.payoff)

    return {"history": history, "populations": populations, "payoff": empirical.payoff,
            "mixtures": mixtures, "meta": (weights_0, weights_1),
            "train_seconds": train_seconds, "budget": budget}


def main() -> None:
    ap = sequential_parser(__doc__)
    ap.add_argument("--rounds", type=int, default=12, help="best responses added per player")
    ap.add_argument("--br-steps", type=int, default=50, help="training chunks per best response")
    ap.add_argument("--br-epochs", type=int, default=20, help="iterations per chunk")
    ap.add_argument("--payoff-episodes", type=int, default=20_000,
                    help="hands played per population pair for the empirical payoff matrix")
    ap.add_argument("--meta-solver", choices=("nash", "uniform"), default="nash")
    args = ap.parse_args()

    game, game_config, config = load_sequential_run(args.config)
    hyperparams = tuple(so.br_hyperparams(game, player, config) for player in (0, 1))
    scorer = build_scorer(game, hyperparams, exact_grid=args.exact_grid,
                          score_every=args.score_every, br_steps=args.score_br_steps,
                          br_epochs=args.score_br_epochs, episodes=args.score_episodes,
                          seed=args.seed)
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"solver  : sequential PSRO  {args.rounds} rounds  meta-solver: {args.meta_solver}  "
          f"BR {args.br_steps}x{args.br_epochs} PPO iterations  "
          f"payoff {args.payoff_episodes} hands/pair")
    print(f"metric  : `h2h` is the meta-strategies' own value; "
          f"`expl` exact (Kuhn only); `expl_lb` an RL best-response bound; "
          f"`rl_gap` what PSRO claims\n")

    meta = {"algorithm": "sequential_psro", "config": args.config,
            **{k: v for k, v in vars(args).items() if k != "config"}}
    writer = RunWriter(args.checkpoint_dir, meta) if args.checkpoint_dir else None
    result = run_sequential_psro(
        game, config, rounds=args.rounds, br_steps=args.br_steps, br_epochs=args.br_epochs,
        payoff_episodes=args.payoff_episodes, meta_solver=args.meta_solver, seed=args.seed,
        scorer=scorer, writer=writer)

    last = result["history"][-1]
    expl = last.get("expl", last.get("expl_lb"))
    label = "exploitability" if "expl" in last else "exploitability lower bound"
    if expl is None:
        print("\nfinal meta-strategy: not scored (no exact best response for this game; "
              "pass --score-every N for an RL bound)")
    else:
        print(f"\nfinal meta-strategy {label} {expl:+.5f}")
    print(f"population {last['population_0']}/{last['population_1']}  "
          f"|  {last['wall_time']:.1f}s training")
    print(f"meta weights p0 {np.round(result['meta'][0], 3)}")
    print(f"meta weights p1 {np.round(result['meta'][1], 3)}")
    if writer is not None:
        writer.finish({"final": last})
        print(f"run -> {writer.directory}")
    elif args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": meta, "history": result["history"]}, indent=2))
        print(f"saved history -> {args.out}")


if __name__ == "__main__":
    main()
