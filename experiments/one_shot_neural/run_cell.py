"""Run ONE (game, method, seed) cell of the neural comparison, fast, and checkpoint it.

    python experiments/one_shot_neural/run_cell.py --game configs/two_point.yaml \
        --method psro --seed 0 --budget 2000000 --out data/one_shot_neural

This is the unit `run_all.py` parallelizes over, and it is also the thing to run by hand
when you want one method on one game. It deliberately does **no scoring**: exploitability
is computed afterwards by `score.py` from the checkpoints this writes. That is not a
convenience -- a single in-training exploitability call best-responds over the whole
deviation grid against a few thousand sampled actions, which on the cheaper methods costs
more than the training between two checkpoints, and it would land inside the wall-time
being compared. Pass `--score` to put it back (useful when watching one run interactively).

**The budget is in payoff evaluations, not iterations.** An iteration of PSRO and an
iteration of a Gaussian mixture have nothing in common; a scored action pair does. Each
method converts the budget into its own iteration or round count with the cost formula in
`plan_units`, so every cell in a sweep consumes the same amount of game.

Each run directory holds

    meta.json          the plan, the settings, timings, the git commit
    history.json       per checkpoint: t, wall_time, compile_time, payoff_evals, cheap metrics
    checkpoints/       one StrategyPair npz per logged point, in the format every
                       baseline in this repo shares

`wall_time` is *training* time with compilation excluded (`compile_time` is reported
separately) and with logging, checkpointing and scoring outside the timer, so a wall-time
plot compares the methods rather than this harness.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repo root

import jax  # noqa: E402
import numpy as np  # noqa: E402

from baselines import sisa as sisa_module  # noqa: E402
from baselines.common import GridOracle, StrategyPair, load_game  # noqa: E402
from baselines.neural import jpspg as jpspg_module  # noqa: E402
from baselines.neural import mmd_discrete as mmd_module  # noqa: E402
from baselines.neural import nfsp as nfsp_module  # noqa: E402
from baselines.neural import psro as psro_module  # noqa: E402
from baselines.neural import randomized_policy as spg_module  # noqa: E402
from baselines.neural.common import RunWriter, empirical_strategy, load_run  # noqa: E402
from training.hyperparams import build_hyperparams  # noqa: E402
from training.mixture import sample_mixture_actions  # noqa: E402
from training.mixture_trainer import MixtureSelfPlayPPOTrainer  # noqa: E402

METHODS = ("mixture", "mmd_discrete", "nfsp", "psro", "spg", "jpspg", "sisa")

# What each method assumes it can do to the game. Recorded in `meta.json` and printed by
# the summary, because a comparison that hides it is not a fair one: exact gradients are
# a strictly stronger access model than sampled payoffs, which are stronger than a
# black-box payoff oracle.
ACCESS_MODEL = {
    "mixture": "sampled payoffs, policy gradients",
    "mmd_discrete": "sampled payoffs, policy gradients",
    "nfsp": "sampled payoffs, policy gradients",
    "psro": "sampled payoffs, policy gradients",
    "spg": "black-box payoff (zeroth order)",
    "jpspg": "black-box payoff (zeroth order)",
    "sisa": "exact payoff gradients",
}


@dataclasses.dataclass
class Settings:
    """Per-method knobs. Defaults match each baseline's own, so a cell run through this
    harness is the same run as the baseline's CLI at the same budget."""

    checkpoints: int = 40           # logged points per run, evenly spaced
    grid: int = 401                 # deviation grid (only used when --score)
    samples: int = 4096             # actions sampled from a continuous policy per checkpoint
    # mmd_discrete
    bins: int = 51
    # spg / jpspg
    sigma: float = 0.1
    pseudo_lr: float = 1e-4
    utility_samples: int = 256
    noise_dim: int = 8
    dynamics: str = "simultaneous"
    # nfsp / psro
    br_steps: int = 50
    br_epochs: int = 20
    eta: float = 0.1
    sl_steps: int = 400
    average_head: str = "mixture"
    average_components: int = 8
    payoff_samples: int = 256
    meta_solver: str = "nash"
    # sisa
    atoms: int = 8
    lr_support: float = 1e-2
    lr_weight: float = 1e-2
    sisa_init: str = "spread"


def plan_units(method: str, budget: int, settings: Settings, batch_size: int) -> dict:
    """How many iterations/rounds `budget` payoff evaluations buys this method.

    The cost formulas, all in scored action pairs:

      mixture, mmd_discrete   one self-play rollout per iteration      `batch`
      spg, jpspg              `utility_samples` per utility evaluation, 4 (separate) or
                              2 (joint) evaluations per iteration
      sisa                    two payoff matrices and two Jacobian passes  `4 n^2`
      nfsp                    two best responses per round               `2 * br_iters * batch`
      psro                    the same, plus every *new* population pair's outer product
                              of `payoff_samples` actions -- which grows with the round,
                              so its round count is solved for rather than divided out

    Sampling for checkpoints and for the metric is excluded on purpose: instrumentation
    is not the algorithm, and counting it would penalize whichever method logs most.
    """
    if method in ("mixture", "mmd_discrete"):
        per = batch_size
        return {"iterations": max(int(budget // per), 1), "evals_per_unit": per}
    if method in ("spg", "jpspg"):
        per = settings.utility_samples * (2 if method == "jpspg" else 4)
        return {"iterations": max(int(budget // per), 1), "evals_per_unit": per}
    if method == "sisa":
        per = 4 * settings.atoms ** 2
        return {"iterations": max(int(budget // per), 1), "evals_per_unit": per}

    br_iterations = settings.br_steps * settings.br_epochs
    per_round = 2 * br_iterations * batch_size
    if method == "nfsp":
        return {"rounds": max(int(budget // per_round), 1), "evals_per_unit": per_round}
    if method == "psro":
        # Cumulative cost after `r` rounds: `r * per_round` for the best responses plus
        # `(r + 1)^2 * payoff_samples^2` for the empirical matrix over the grown
        # populations. Solved by walking `r` up, which is exact and costs nothing.
        matrix = settings.payoff_samples ** 2
        rounds = 0
        while (rounds + 1) * per_round + (rounds + 2) ** 2 * matrix <= budget:
            rounds += 1
        return {"rounds": max(rounds, 1), "evals_per_unit": per_round,
                "matrix_evals_per_pair": matrix}
    raise ValueError(f"unknown method {method!r} (choices: {METHODS})")


def budget_warning(method: str, plan: dict, settings: Settings, budget: int) -> str | None:
    """Warn when a method's *smallest* unit already costs more than the budget.

    Every method's unit count is floored at 1, so a budget below one unit silently buys
    more game than it asked for -- PSRO is the one that hits this, because a single round
    pays for the whole empirical payoff matrix (`payoff_samples^2` per population pair)
    before any best response is trained. Better to say so than to publish a row whose
    x-axis quietly disagrees with the others.
    """
    units = plan.get("iterations", plan.get("rounds", 1))
    spent = units * plan["evals_per_unit"]
    if method == "psro":
        spent += (units + 1) ** 2 * plan["matrix_evals_per_pair"]
    if spent <= budget:
        return None
    return (f"WARNING: one unit of {method} costs {spent:.3g} payoff evaluations, above the "
            f"{budget:.3g} budget -- this cell will overspend. Raise --budget"
            + (" or lower --payoff-samples." if method == "psro" else "."))


def compile_warning(method: str, plan: dict, settings: Settings) -> str | None:
    """Warn when a round-based method will spend most of its wall-time compiling.

    `nfsp` and `psro` build a fresh best-response trainer every round, so each round pays
    an XLA compilation -- and unlike the single-scan methods, that cost is *inside* their
    `wall_time` because it cannot be separated from the trainer's own construction. It is
    a fixed cost per round, so it is harmless when a round trains for a thousand
    iterations and ruinous when it trains for ten: at `--br-steps 2 --br-epochs 5` a round
    of NFSP measured here was ~60s of compilation around ~2s of training. Since a best
    response needs on the order of a thousand iterations to be one at all (see
    `baselines/neural/README.md`), the fix for both problems is the same.
    """
    if method not in ("nfsp", "psro"):
        return None
    iterations = settings.br_steps * settings.br_epochs
    if iterations >= 500:
        return None
    return (f"WARNING: {plan['rounds']} rounds of only {iterations} best-response "
            f"iterations each. These methods recompile per round, so most of the measured "
            f"wall-time will be compilation -- and a {iterations}-iteration best response "
            f"is not a best response. Raise --br-steps/--br-epochs (>= 500 iterations) "
            f"and the budget with it.")


# --------------------------------------------------------------------------- runners
#
# One adapter per method, all `(game, oracle, config, plan, settings, seed, writer,
# score) -> dict`. Each returns the run's history plus whatever the caller needs for the
# final line; the writer already holds the checkpoints.


def _log_every(units: int, checkpoints: int) -> int:
    return max(units // max(checkpoints, 1), 1)


def _aligned(units: int, checkpoints: int) -> tuple[int, int]:
    """`(units, log_every)` with `units` rounded down to a whole number of chunks.

    A leftover chunk of a different length is a *second* XLA compilation of the same
    scan, which on a short run can cost more than the training -- and it lands in
    `compile_time`, where it makes the harness look like the method. Losing at most
    `log_every` iterations of budget is the cheaper trade.
    """
    log_every = _log_every(units, checkpoints)
    return max((units // log_every) * log_every, log_every), log_every


def run_mixture(game, oracle, config, plan, settings, seed, writer, score):
    """The method under test: `MixtureSelfPlayPPOTrainer`, driven chunk by chunk.

    Uses the trainer's public API with its per-chunk printing and exploitability turned
    off, so what is timed is the same training `train.py` would do and nothing else. The
    checkpoint is the mixture's *sampled* strategy, which is what makes it comparable to
    every other method here; the Polyak-averaged iterate is sampled alongside into
    `extra`, since that is the iterate this repo usually reports.
    """
    hyperparams = [build_hyperparams(game, player, config) for player in (0, 1)]
    trainer = MixtureSelfPlayPPOTrainer(game, hyperparams[0], hyperparams[1], seed=seed)
    spaces = [game.action_space(player) for player in (0, 1)]
    observations = [trainer._obs_1, trainer._obs_2]
    epochs = plan["log_every"]
    chunks = max(plan["iterations"] // epochs, 1)
    key = jax.random.PRNGKey(seed + 7919)
    train_seconds = 0.0

    def snapshot(t: int) -> dict:
        nonlocal key
        key, *keys = jax.random.split(key, 5)
        live, target = [], []
        for player, params in enumerate(trainer.params):
            live.append(empirical_strategy(sample_mixture_actions(
                trainer.network_1 if player == 0 else trainer.network_2, params,
                observations[player], spaces[player], keys[player], settings.samples)))
        for player, params in enumerate(trainer.target_params):
            target.append(empirical_strategy(sample_mixture_actions(
                trainer.network_1 if player == 0 else trainer.network_2, params,
                observations[player], spaces[player], keys[2 + player], settings.samples)))
        entry = {"t": t, "wall_time": train_seconds, "payoff_evals": t * hyperparams[0].num_envs}
        if score:
            entry["expl"] = float(oracle.exploitability(live[0][0], live[0][1],
                                                        live[1][0], live[1][1]))
            entry["target_expl"] = float(oracle.exploitability(target[0][0], target[0][1],
                                                               target[1][0], target[1][1]))
        writer.record(entry, StrategyPair(
            t=t, support_0=live[0][0], weights_0=live[0][1],
            support_1=live[1][0], weights_1=live[1][1],
            extra={"target_support_0": target[0][0], "target_support_1": target[1][0]}))
        return entry

    snapshot(0)
    for chunk in range(chunks):
        started = time.monotonic()
        trainer.train(1, epochs, verbose=False, measure_exploitability=False)
        jax.block_until_ready(trainer.params)
        train_seconds += time.monotonic() - started
        entry = snapshot((chunk + 1) * epochs)
        print(f"  t={entry['t']:7d}  {entry['wall_time']:7.1f}s  "
              f"{entry['payoff_evals']:9.3g} evals"
              + (f"  expl={entry['expl']:+.5f}" if score else ""))
    for player, params in enumerate(trainer.params):
        writer.save_params(f"player_{player}", hyperparams[player], params)
    return {"history": writer.history, "train_seconds": train_seconds, "compile_seconds": 0.0}


def run_mmd_discrete(game, oracle, config, plan, settings, seed, writer, score):
    class _Args:
        lr = batch = entropy = magnet_kl = trpo_kl = magnet_interval = None

    hyperparams = mmd_module.hyperparams_from_config(game, config, settings.bins, _Args())
    result = mmd_module.run_mmd_discrete(
        game, oracle, hyperparams, iterations=plan["iterations"],
        log_every=plan["log_every"], seed=seed, writer=writer, score=score)
    return result


def run_pseudo_gradient(method: str):
    def run(game, oracle, config, plan, settings, seed, writer, score):
        class _Args:
            noise_dim = settings.noise_dim
            lr = settings.pseudo_lr
            optimizer = "adabelief"
            sigma = settings.sigma
            utility_samples = settings.utility_samples
            no_antithetic = False
            dynamics = settings.dynamics

        hyperparams = spg_module.hyperparams_from_config(game, config, _Args())
        estimator = (jpspg_module.jpspg_pseudo_gradients if method == "jpspg"
                     else spg_module.spg_pseudo_gradients)
        return spg_module.run_pseudo_gradient(
            game, oracle, hyperparams, estimator, iterations=plan["iterations"],
            log_every=plan["log_every"], samples=settings.samples, seed=seed,
            writer=writer, score=score, estimator_name=method)

    return run


def run_nfsp(game, oracle, config, plan, settings, seed, writer, score):
    return nfsp_module.run_nfsp(
        game, oracle, config, rounds=plan["rounds"], br_steps=settings.br_steps,
        br_epochs=settings.br_epochs, eta=settings.eta, sl_steps=settings.sl_steps,
        average_head=settings.average_head, average_components=settings.average_components,
        samples=settings.samples, seed=seed, writer=writer, score=score)


def run_psro(game, oracle, config, plan, settings, seed, writer, score):
    return psro_module.run_psro(
        game, oracle, config, rounds=plan["rounds"], br_steps=settings.br_steps,
        br_epochs=settings.br_epochs, payoff_samples=settings.payoff_samples,
        meta_solver=settings.meta_solver, seed=seed, writer=writer, score=score)


def run_sisa(game, oracle, config, plan, settings, seed, writer, score):
    result = sisa_module.run_sisa(
        game, oracle, atoms=settings.atoms, iters=plan["iterations"],
        lr_support=settings.lr_support, lr_weight=settings.lr_weight,
        init=settings.sisa_init, seed=seed, log_every=plan["log_every"],
        checkpoint_fn=writer.checkpoints, score=score)
    writer.history.extend(result.history)
    return {"history": result.history}


RUNNERS = {
    "mixture": run_mixture,
    "mmd_discrete": run_mmd_discrete,
    "nfsp": run_nfsp,
    "psro": run_psro,
    "spg": run_pseudo_gradient("spg"),
    "jpspg": run_pseudo_gradient("jpspg"),
    "sisa": run_sisa,
}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parents[1],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def run_cell(game_config_path: str, method: str, seed: int, budget: int, settings: Settings,
             out_root: Path, score: bool = False, overwrite: bool = False) -> dict:
    """One cell. Returns its summary row; failures are recorded rather than raised."""
    game_tag = Path(game_config_path).stem
    directory = out_root / game_tag / method / f"seed{seed}"
    row = {"game": game_tag, "method": method, "seed": seed, "budget": budget,
           "config": game_config_path, "dir": str(directory),
           "access_model": ACCESS_MODEL[method]}

    if (directory / "meta.json").exists() and not overwrite:
        stored = json.loads((directory / "meta.json").read_text())
        print(f"  skip (already done): {directory}")
        return {**row, **{k: stored.get(k) for k in
                          ("status", "train_seconds", "compile_seconds", "payoff_evals",
                           "checkpoints")}}

    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        game, game_cfg, config = load_run(game_config_path)
        oracle = GridOracle(game, points=settings.grid)
        batch = build_hyperparams(game, 0, config).num_envs
        plan = plan_units(method, budget, settings, batch)
        for warning in (budget_warning(method, plan, settings, budget),
                        compile_warning(method, plan, settings)):
            if warning:
                print(f"  {warning}")
        for key in ("iterations", "rounds"):
            if key in plan:
                plan[key], plan["log_every"] = _aligned(plan[key], settings.checkpoints)
        meta = {**row, "plan": plan, "settings": dataclasses.asdict(settings),
                "game_config": dataclasses.asdict(game_cfg), "scored_during_run": score,
                "commit": _git_commit(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        writer = RunWriter(directory, meta)
        result = RUNNERS[method](game, oracle, config, plan, settings, seed, writer, score)
    except Exception as exc:                    # noqa: BLE001 -- recorded, not swallowed
        (directory / "error.txt").write_text(traceback.format_exc())
        print(f"  FAILED after {time.monotonic() - started:.1f}s: {type(exc).__name__}: {exc}")
        return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    history = result["history"]
    meta = writer.finish({
        "status": "ok",
        "train_seconds": result.get("train_seconds", history[-1].get("wall_time", 0.0)),
        "compile_seconds": result.get("compile_seconds", history[-1].get("compile_time", 0.0)),
        "elapsed_seconds": time.monotonic() - started,
        "payoff_evals": history[-1].get("payoff_evals"),
        "checkpoints": len(writer.checkpoints.entries),
        "units": plan,
    })
    print(f"  done: {meta['payoff_evals']:.3g} payoff evals in {meta['train_seconds']:.1f}s "
          f"training (+{meta['compile_seconds']:.1f}s compile), "
          f"{meta['checkpoints']} checkpoints -> {directory}")
    return {**row, "status": "ok", "train_seconds": meta["train_seconds"],
            "compile_seconds": meta["compile_seconds"], "payoff_evals": meta["payoff_evals"],
            "checkpoints": meta["checkpoints"]}


def add_settings_arguments(ap: argparse.ArgumentParser) -> None:
    defaults = Settings()
    for field in dataclasses.fields(Settings):
        flag = "--" + field.name.replace("_", "-")
        kind = {int: int, float: float, str: str}[field.type if not isinstance(field.type, str)
                                                  else {"int": int, "float": float,
                                                        "str": str}[field.type]]
        ap.add_argument(flag, type=kind, default=getattr(defaults, field.name))


def settings_from_args(args) -> Settings:
    return Settings(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Settings)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, help="path to a config in configs/")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=2_000_000,
                    help="payoff evaluations this cell may spend")
    ap.add_argument("--out", default="data/one_shot_neural")
    ap.add_argument("--score", action="store_true",
                    help="also compute exploitability during the run (slower; normally left "
                         "to score.py)")
    ap.add_argument("--overwrite", action="store_true")
    add_settings_arguments(ap)
    args = ap.parse_args()

    settings = settings_from_args(args)
    print(f"{args.method} / {Path(args.game).stem} / seed{args.seed} / "
          f"budget {args.budget:.3g} payoff evals  ({ACCESS_MODEL[args.method]})")
    row = run_cell(args.game, args.method, args.seed, args.budget, settings,
                   Path(args.out), score=args.score, overwrite=args.overwrite)
    if row.get("status") == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
