"""CLI entry point for solving a *sequential* game by any of the three methods.

`train.py` runs one algorithm -- self-play PPO on the Gaussian-mixture policy --
on either family of game. This script runs all three algorithms this repo has for
a game tree, off the same YAML schema, into the same run directory layout, and
under the same metric:

  self_play   `training.sequential_trainer.SequentialSelfPlayPPOTrainer`, the
              method under test: both players' `MixtureActorCritic`s learning
              simultaneously, scheduled by `train.steps`/`train.epochs`.
  nfsp        `baselines.neural.sequential_nfsp`, scheduled by the `nfsp:` section.
  psro        `baselines.neural.sequential_psro`, scheduled by the `psro:` section.
  rpn         `baselines.neural.sequential_rpn`, scheduled by the `rpn:` section:
              randomized policy networks (Martin & Sandholm) trained by
              zeroth-order pseudo-gradient. A different *policy class* -- implicit,
              `a = f(o, z)`, no density -- so it shares the torso shape with the
              others and nothing else: no PPO, no magnet, no trust region, because
              none of them can be written without a `log pi`.
  discrete_mmd
              the *same* self-play run, on a game whose continuous action has
              been split into `discrete.bins` evenly spaced actions
              (`games.discretized`) -- so the policy plays through its
              categorical head alone. The discretization baseline: identical
              update, identical rollout, identical regularizers, a different
              action set. Scheduled by `train.steps`/`train.epochs` like
              `self_play`, which is the run it is meant to be read against.

`train.solver` picks between them. The three want *vastly* different budgets, so
each brings its own section rather than sharing one set of fields; what they do
share is `network:`, `optimizer:` and `ppo:` -- the policy and its optimizer are
the same object in all three, and a comparison in which they were not would not
be a comparison.

**What is recorded, and why.** Each solver's natural unit of progress (a chunk of
PPO iterations, a fictitious-play round, a PSRO round) is not comparable with the
others', so every log point also carries the two units that are:

    wall_time        seconds spent *training*, excluding measurement
    total_wall_time  seconds since the run started, including it
    episodes         hands played (exact)
    env_steps        decision nodes visited (exact for training, estimated for
                     evaluation rollouts -- see `baselines/neural/sequential_run.py`)
    iterations       PPO iterations

alongside the optimizer's own numbers (`loss`, `grad_norm`, `value_loss`,
`approx_kl`, ... averaged over the interval the row covers) and whatever the
metric supports for this game (`expl` exactly on Kuhn, `expl_lb` on request
elsewhere, `h2h` always). Rows stream to `metrics.jsonl` as they happen, so a run
killed by a scheduler keeps everything it reached.

The run directory (`train.checkpoint_dir`):

    meta.json          the full config, the solver, and the run's totals
    history.json       every log point (also streamed to metrics.jsonl)
    checkpoints/{t}.pkl  weights per log point; `player_0`/`player_1` are the
                       strategy to measure, so `best_response.py` reads them
                       directly -- except under PSRO, whose strategy is a
                       population (`player{p}_policy{k}` plus `{t}.npz` holding
                       the meta-weights and the payoff matrix)

Examples:
  python train_sequential.py configs/kuhn.yaml                  # self-play (the default)
  python train_sequential.py configs/kuhn_psro.yaml
  python train_sequential.py configs/leduc_nfsp.yaml --solver psro --checkpoint-dir data/x
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time
from pathlib import Path

import jax

from baselines.neural import sequential_oracle as so
from baselines.neural.sequential_nfsp import run_sequential_nfsp
from baselines.neural.sequential_psro import run_sequential_psro
from baselines.neural import sequential_rpn as rpn
from baselines.neural.sequential_run import Budget, SequentialRunLog, Stopwatch, mean_columns
from baselines.neural.sequential_scoring import build_scorer, has_exact_exploitability
from games.discretized import DiscretizedSequentialGame, discretize
from games.sequential import SequentialZeroSumGame
from training.hyperparams import build_hyperparams
from training.run_config import SOLVERS, RunConfig, load_run_config
from training.sequential_trainer import SequentialSelfPlayPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a YAML run config naming a sequential game")
    parser.add_argument("--solver", choices=SOLVERS, default=None,
                        help="overrides train.solver")
    parser.add_argument("--checkpoint-dir", default=None, help="overrides train.checkpoint_dir")
    parser.add_argument("--seed", type=int, default=None, help="overrides train.seed")
    parser.add_argument("--score-every", type=int, default=None,
                        help="overrides scoring.score_every: train an RL best-response bound "
                             "every N log points (0 = never; the only bound off Kuhn)")
    return parser.parse_args()


def apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    """Fold the CLI flags into the config, so everything downstream reads one object."""
    train = dataclasses.replace(config.train, **{
        key: value for key, value in (
            ("solver", args.solver),
            ("checkpoint_dir", args.checkpoint_dir),
            ("seed", args.seed),
        ) if value is not None
    })
    scoring = config.scoring
    if args.score_every is not None:
        scoring = dataclasses.replace(scoring, score_every=args.score_every)
    return dataclasses.replace(config, train=train, scoring=scoring)


def check_directory_is_free(config: RunConfig) -> None:
    """Refuse to write a second solver's run into another's directory.

    `--solver` on a shared config is the natural way to produce the three runs
    being compared, and forgetting `--checkpoint-dir` with it would leave one
    directory holding two solvers' `checkpoints/{t}.pkl` -- the later run's rows
    beside the earlier run's weights, silently, since the step numbering is per
    solver. Re-running the *same* solver overwrites its own output, which is what
    re-running means.
    """
    directory = config.train.checkpoint_dir
    if directory is None:
        return
    meta_path = Path(directory) / "meta.json"
    if not meta_path.exists():
        return
    try:
        previous = json.loads(meta_path.read_text()).get("solver")
    except (OSError, json.JSONDecodeError):
        return
    if previous is not None and previous != config.train.solver:
        raise ValueError(
            f"{directory} already holds a {previous!r} run; writing a "
            f"{config.train.solver!r} one into it would mix their checkpoints. Pass "
            "--checkpoint-dir (or set train.checkpoint_dir) to a directory of its own."
        )


def build_run_log(config: RunConfig, config_path: str) -> SequentialRunLog:
    """The run directory, with the whole config recorded in `meta.json` up front.

    The config is stored as data rather than as a path: a comparison read months
    later has to be able to say what the run's `eta` or `br_steps` actually were,
    and the file on disk may have moved on.
    """
    meta = {
        "entry_point": "train_sequential.py",
        "solver": config.train.solver,
        "config_path": str(config_path),
        "config": {
            "game": {"name": type(config.game).__name__, **dataclasses.asdict(config.game)},
            "network": dataclasses.asdict(config.network),
            "optimizer": dataclasses.asdict(config.optimizer),
            "ppo": dataclasses.asdict(config.ppo),
            "train": dataclasses.asdict(config.train),
            "nfsp": dataclasses.asdict(config.nfsp),
            "psro": dataclasses.asdict(config.psro),
            "scoring": dataclasses.asdict(config.scoring),
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "jax_devices": [str(device) for device in jax.devices()],
    }
    return SequentialRunLog(config.train.checkpoint_dir, meta)


def solver_hyperparams(game: SequentialZeroSumGame, player: int, config: RunConfig,
                       best_response: bool = False):
    """The policy's hyperparameters for this game, `best_response` stripping the
    regularizers the way `br_oracle` does.

    The one thing added on top of `training.hyperparams`: a discretized run zeroes
    `mean_box_penalty_coef`. That penalty acts on the Gaussian head's means, which
    such a run never samples from -- but the heads share a torso, so leaving it on
    would let a dead head push the live one around, and the whole point of the
    discretization baseline is that nothing but the action set differs.
    """
    hyperparams = (so.br_hyperparams(game, player, config) if best_response
                   else build_hyperparams(game, player, config))
    if isinstance(game, DiscretizedSequentialGame):
        hyperparams = dataclasses.replace(hyperparams, mean_box_penalty_coef=0.0)
    return hyperparams


def build_run_scorer(game: SequentialZeroSumGame, config: RunConfig):
    """One metric for every solver -- see `baselines/neural/sequential_scoring.py`.

    Deliberately *not* `train.py`'s `build_sequential_hooks`, whose exploitability
    is Kuhn-shaped and reads a pair of networks: NFSP's and PSRO's strategies are
    not a pair of networks, and a comparison in which the solvers were scored by
    different code would be worth very little.

    `rpn` is the one that needs its own, and for a reason rather than by omission:
    its policy has no density, so its strategy has to be *sampled* out of it and
    its best-response bound has to be trained by its own zeroth-order oracle. The
    columns and the rules for producing them are the same.
    """
    if config.train.solver == "rpn":
        hyperparams = rpn.hyperparams_from_config(game, config)
        return rpn.build_scorer(game, rpn.build_policy(hyperparams), hyperparams, config)
    hyperparams = tuple(solver_hyperparams(game, player, config, best_response=True)
                        for player in (0, 1))
    return build_scorer(
        game, hyperparams,
        exact_grid=config.scoring.exact_grid,
        score_every=config.scoring.score_every,
        br_steps=config.scoring.br_steps,
        br_epochs=config.scoring.br_epochs,
        episodes=config.scoring.episodes,
        seed=config.train.seed,
    )


# --------------------------------------------------------------------------- solvers


def run_self_play(game: SequentialZeroSumGame, config: RunConfig, log: SequentialRunLog,
                  scorer) -> dict:
    """The method under test: both mixture policies learning against each other.

    The trainer already logs per-iteration metrics and checkpoints per chunk, so
    all that is added here is a `metric_fn` that closes each chunk's row: it stops
    the training clock, prices the chunk's iterations, scores the current policies
    through the shared scorer, and hands `run_training_chunks` a short dict to
    print.
    """
    hyperparams = tuple(solver_hyperparams(game, player, config) for player in (0, 1))
    trainer = SequentialSelfPlayPPOTrainer(game, hyperparams[0], hyperparams[1],
                                           seed=config.train.seed)
    budget = Budget(default_episode_length=float(game.max_steps))
    clock = Stopwatch(running=True)
    recorded = 0     # history rows already priced, so a chunk is charged exactly once
    chunk = 0

    def metric_fn(trainer: SequentialSelfPlayPPOTrainer) -> dict[str, float]:
        nonlocal recorded, chunk
        clock.stop()
        chunk += 1
        rows = trainer.history[recorded:]
        recorded = len(trainer.history)
        budget.add_training(rows, hyperparams[0].num_envs)

        live = tuple(so.single(trainer.networks[player], hyperparams[player],
                               trainer.params[player], f"live(player {player})")
                     for player in (0, 1))
        scores = scorer(live, chunk)
        # The Polyak-averaged iterate is the better-behaved one in self-play and
        # is what the checkpoints carry, so it is scored too -- but only where the
        # metric is exact and therefore free. Paying for a second RL bound would
        # come out of the very budget being compared.
        if config.scoring.include_target and has_exact_exploitability(game):
            target = tuple(so.single(trainer.networks[player], hyperparams[player],
                                     trainer.target_params[player], f"target(player {player})")
                           for player in (0, 1))
            scores.update({f"target_{k}": v for k, v in scorer(target, chunk, full=False).items()})

        entry = {"t": chunk, "wall_time": clock.seconds, **budget.row(),
                 **mean_columns(rows), **scores}
        log.record(entry)
        clock.start()
        # What `run_training_chunks` prints beside its own per-iteration line.
        return {k: float(v) for k, v in scores.items() if k in ("expl", "expl_lb", "h2h")}

    trainer.train(
        config.train.steps,
        epochs=config.train.epochs,
        checkpoint_dir=(Path(log.directory) / log.checkpoint_name) if log.directory else None,
        metric_fn=metric_fn,
    )
    if log.directory is not None:
        for player in (0, 1):
            log.save_params(f"player_{player}", hyperparams[player], trainer.params[player])
    return {"history": log.history, "train_seconds": clock.seconds, "budget": budget}


def run_nfsp(game: SequentialZeroSumGame, config: RunConfig, log: SequentialRunLog,
             scorer) -> dict:
    return run_sequential_nfsp(
        game, config,
        rounds=config.nfsp.rounds,
        br_steps=config.nfsp.br_steps,
        br_epochs=config.nfsp.br_epochs,
        eta=config.nfsp.eta,
        reservoir_capacity=config.nfsp.reservoir_capacity,
        reservoir_episodes=config.nfsp.reservoir_episodes,
        sl_steps=config.nfsp.sl_steps,
        sl_batch=config.nfsp.sl_batch,
        seed=config.train.seed,
        scorer=scorer,
        writer=log,
        checkpoint_fn=log.checkpoint,
    )


def run_psro(game: SequentialZeroSumGame, config: RunConfig, log: SequentialRunLog,
             scorer) -> dict:
    return run_sequential_psro(
        game, config,
        rounds=config.psro.rounds,
        br_steps=config.psro.br_steps,
        br_epochs=config.psro.br_epochs,
        payoff_episodes=config.psro.payoff_episodes,
        meta_solver=config.psro.meta_solver,
        seed=config.train.seed,
        scorer=scorer,
        writer=log,
        checkpoint_fn=log.checkpoint,
    )


# `discrete_mmd` is `run_self_play` on purpose, and this line is the claim: the
# discretization baseline differs from the method under test in the *game* it is
# handed (`prepare_game` below) and in nothing else -- not the trainer, not the
# loss, not the schedule, not the metric.
def run_rpn(game: SequentialZeroSumGame, config: RunConfig, log: SequentialRunLog,
            scorer) -> dict:
    return rpn.run_sequential_rpn(
        game, config,
        iterations=config.rpn.iterations,
        log_every=config.rpn.log_every,
        estimator=config.rpn.estimator,
        seed=config.train.seed,
        scorer=scorer,
        writer=log,
        checkpoint_fn=log.checkpoint,
    )


SOLVER_RUNNERS = {"self_play": run_self_play, "nfsp": run_nfsp, "psro": run_psro,
                  "discrete_mmd": run_self_play, "rpn": run_rpn}


def prepare_game(game: SequentialZeroSumGame, config: RunConfig) -> SequentialZeroSumGame:
    """The game the solver actually plays -- discretized for `discrete_mmd`, as is otherwise.

    Applied before the hyperparameters and the scorer are built, because both
    read the action space: the wrapped game has `base_atoms + bins**d` atoms and
    no legal continuous branch, and everything downstream picks that up on its own.
    """
    if config.train.solver != "discrete_mmd":
        return game
    return discretize(game, config.discrete.bins, max_actions=config.discrete.max_actions)


# --------------------------------------------------------------------------- entry point


def describe(game: SequentialZeroSumGame, config: RunConfig) -> str:
    """The schedule line: what this run is about to spend, before it spends it."""
    if config.train.solver in ("self_play", "discrete_mmd"):
        iterations = config.train.steps * config.train.epochs
        schedule = (f"{config.train.steps} chunks x {config.train.epochs} iterations "
                    f"= {iterations} PPO iterations at batch {config.ppo.batch_size}")
        if isinstance(game, DiscretizedSequentialGame):
            schedule += (f"  |  {game.num_grid} actions "
                         f"({config.discrete.bins} bins/axis) + {game.base_atoms} atoms")
    elif config.train.solver == "rpn":
        per_iteration = config.rpn.perturbation_batch * (1 if config.rpn.estimator == "joint" else 2)
        schedule = (f"{config.rpn.iterations} iterations x {per_iteration} utility "
                    f"evaluations x {config.rpn.utility_episodes} hands "
                    f"({config.rpn.estimator} pseudo-gradient, sigma={config.rpn.sigma}, "
                    f"{config.rpn.dynamics})")
    elif config.train.solver == "nfsp":
        per_round = 2 * config.nfsp.br_steps * config.nfsp.br_epochs
        schedule = (f"{config.nfsp.rounds} rounds x {per_round} PPO iterations "
                    f"(eta={config.nfsp.eta}, SL {config.nfsp.sl_steps}x{config.nfsp.sl_batch})")
    else:
        per_round = 2 * config.psro.br_steps * config.psro.br_epochs
        schedule = (f"{config.psro.rounds} rounds x {per_round} PPO iterations "
                    f"(meta-solver {config.psro.meta_solver}, "
                    f"{config.psro.payoff_episodes} hands/pair)")
    metric = ("exact tree best response (`expl`)" if has_exact_exploitability(game)
              else ("an RL best-response bound (`expl_lb`) every "
                    f"{config.scoring.score_every} log points"
                    if config.scoring.score_every else "head-to-head value only (`h2h`)"))
    return f"solver  : {config.train.solver}  {schedule}\nmetric  : {metric}"


def main() -> None:
    # Importing `baselines.*` turns x64 on process-wide -- the one-shot metric
    # integrates differences at the 1e-3 level and needs it (see
    # `baselines/common.py`). Nothing here does: a tree metric is a traversal of
    # small tables and a Monte-Carlo payoff, and float64 would silently make
    # every network in the run twice the width `train.py` builds. Since this
    # script exists to compare wall-clock cost between solvers, running in a
    # different precision from the rest of the repo is exactly the wrong default.
    jax.config.update("jax_enable_x64", False)

    args = parse_args()
    config = apply_overrides(load_run_config(args.config), args)
    game = config.game.build()

    if not isinstance(game, SequentialZeroSumGame):
        raise ValueError(
            f"{type(game).__name__} is a one-shot game; train_sequential.py solves game trees. "
            "Use train.py (self-play) or `python -m baselines.neural.psro` / `.nfsp`."
        )
    if config.network.policy != "gaussian_mixture":
        raise ValueError(
            f"network.policy {config.network.policy!r} is one-shot only; a game tree needs the "
            "mixture policy's atoms and legality masks (see training/expfam.py's scope note)"
        )

    check_directory_is_free(config)
    game = prepare_game(game, config)

    log = build_run_log(config, args.config)
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(config.game)}")
    print(describe(game, config))
    print(f"run     : {log.directory}\n")

    result = SOLVER_RUNNERS[config.train.solver](game, config, log, build_run_scorer(game, config))

    last = log.history[-1] if log.history else {}
    totals = {
        "rows": len(log.history),
        "train_seconds": float(result.get("train_seconds") or 0.0),
        "total_seconds": log.clock.seconds,
        **(result["budget"].row() if "budget" in result else {}),
        "mean_episode_length": (result["budget"].mean_episode_length
                                if "budget" in result else None),
    }
    log.finish({"final": last, "totals": totals, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    expl = last.get("expl", last.get("expl_lb"))
    label = "exploitability" if "expl" in last else "exploitability lower bound"
    print("\n" + (f"final {label} {expl:+.5f}" if expl is not None
                  else "final strategy not scored (no exact best response for this game; "
                       "set scoring.score_every for an RL bound)"))
    print(f"cost    : {totals['train_seconds']:.1f}s training / {totals['total_seconds']:.1f}s total"
          f"  |  {totals.get('episodes', 0):,} episodes  |  "
          f"{totals.get('env_steps', 0):,.0f} decisions")
    if log.directory is not None:
        print(f"run -> {log.directory}  ({totals['rows']} rows in history.json, "
              f"weights in {log.checkpoint_name}/)")


if __name__ == "__main__":
    main()
