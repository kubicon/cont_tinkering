"""CLI entry point for measuring a trained strategy by best-responding to it.
"""

from __future__ import annotations

import argparse
import dataclasses

import jax

from games.sequential import SequentialZeroSumGame
from training.best_response import (
    Evaluation,
    SequentialBestResponseTrainer,
    latest_checkpoint_step,
    load_frozen_policy,
    warn_if_regularized,
)
from training.hyperparams import build_hyperparams
from training.run_config import RunConfig, load_run_config


def _eval_seed(settings, responder: int, iterate: str) -> int:
    """A distinct rng stream per (responder, iterate), so no two runs share a sample."""
    return settings.eval_seed + responder + (1000 if iterate == "target" else 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a YAML run config with a `best_response` section")
    parser.add_argument(
        "--responder", choices=("0", "1", "both"), default=None,
        help="which player learns a best response; overrides best_response.responder",
    )
    parser.add_argument(
        "--opponent-iterate", choices=("live", "target", "both"), default=None,
        help="which of the checkpoint's strategies to measure; overrides best_response.opponent_iterate",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="overrides best_response.checkpoint_dir")
    parser.add_argument(
        "--checkpoint-step", type=int, default=None,
        help="overrides best_response.checkpoint_step (default: the last one written)",
    )
    parser.add_argument("--steps", type=int, default=None, help="overrides train.steps")
    parser.add_argument("--epochs", type=int, default=None, help="overrides train.epochs")
    parser.add_argument("--eval-episodes", type=int, default=None, help="overrides best_response.eval_episodes")
    return parser.parse_args()


def apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    """Fold the CLI flags into the config, so everything downstream reads one object."""
    best_response = dataclasses.replace(
        config.best_response,
        **{
            key: value
            for key, value in (
                ("responder", args.responder),
                ("opponent_iterate", args.opponent_iterate),
                ("checkpoint_dir", args.checkpoint_dir),
                ("checkpoint_step", args.checkpoint_step),
                ("eval_episodes", args.eval_episodes),
            )
            if value is not None
        },
    )
    train = dataclasses.replace(
        config.train,
        **{
            key: value
            for key, value in (("steps", args.steps), ("epochs", args.epochs))
            if value is not None
        },
    )
    return dataclasses.replace(config, best_response=best_response, train=train)


def responder_players(responder: str | int) -> tuple[int, ...]:
    """`best_response.responder` as the list of players to train, in order."""
    if responder in ("both", "BOTH"):
        return (0, 1)
    if responder in (0, 1, "0", "1"):
        return (int(responder),)
    raise ValueError(f"best_response.responder must be 0, 1 or 'both', got {responder!r}")


def opponent_iterates(setting: str) -> tuple[str, ...]:
    """`best_response.opponent_iterate` as the list of strategies to measure, in order."""
    if setting == "both":
        return ("live", "target")
    if setting in ("live", "target"):
        return (setting,)
    raise ValueError(f"best_response.opponent_iterate must be 'live', 'target' or 'both', got {setting!r}")


def build_progress_metric_fn(config: RunConfig, key: jax.Array):
    """A cheap per-chunk evaluation, so the log shows the bound flattening out.
    """
    episodes = config.best_response.progress_episodes

    def metric_fn(trainer: SequentialBestResponseTrainer) -> dict[str, float]:
        nonlocal key
        key, eval_key = jax.random.split(key)
        return {"br_value_eval": trainer.evaluate(eval_key, num_episodes=episodes).value}

    return metric_fn


def run_one_direction(
    game: SequentialZeroSumGame,
    config: RunConfig,
    responder: int,
    step: int,
    iterate: str = "live",
) -> dict[str, Evaluation]:
    """Train one best response and measure it; returns the sampled and greedy bounds."""
    settings = config.best_response
    opponent = load_frozen_policy(
        settings.checkpoint_dir, step, 1 - responder, target=(iterate == "target")
    )
    hyperparams = build_hyperparams(game, responder, config)

    regularized = warn_if_regularized(hyperparams)
    if regularized:
        print(
            f"  warning: {', '.join(regularized)} nonzero -- these pull the responder towards its "
            "own past iterate, weakening it and so *understating* exploitability. Set them to 0."
        )

    print(
        f"\n=== best response for player {responder} vs the checkpoint's {iterate} "
        f"player {1 - responder} ==="
    )
    trainer = SequentialBestResponseTrainer(
        game, opponent, hyperparams, responder=responder, seed=config.train.seed
    )
    trainer.train(
        config.train.steps,
        epochs=config.train.epochs,
        metric_fn=build_progress_metric_fn(config, jax.random.PRNGKey(_eval_seed(settings, responder, iterate))),
    )

    # One key per readout: the two estimates are then independent, so agreement
    # between them is evidence and not an artifact of a shared sample.
    keys = jax.random.split(jax.random.PRNGKey(_eval_seed(settings, responder, iterate) + 100), 2)
    return {
        mode: trainer.evaluate(
            keys[index],
            num_episodes=settings.eval_episodes,
            greedy=(mode == "greedy"),
            batch_size=settings.eval_batch_size,
        )
        for index, mode in enumerate(("sampled", "greedy"))
    }


def report(results: dict[tuple[str, int], dict[str, Evaluation]]) -> None:
    """Print each direction's bound, and the exploitability of each iterate measured."""
    print("\n=== best-response values (the responder's own payoff; higher = more exploitable) ===")
    exploitability = {}
    for iterate in dict.fromkeys(key[0] for key in results):
        bounds = {}
        for (row_iterate, responder), evaluations in sorted(results.items()):
            if row_iterate != iterate:
                continue
            sampled, greedy = evaluations["sampled"], evaluations["greedy"]
            # Both are valid lower bounds, so the larger is the better bound. Only
            # two candidates, so the selection bias this introduces is negligible.
            bounds[responder] = max(sampled.value, greedy.value)
            print(
                f"  [{iterate}] BR(player {responder}) vs checkpoint player {1 - responder}: "
                f"sampled {sampled}  greedy {greedy}  -> {bounds[responder]:+.4f}"
            )
        if len(bounds) == 2:
            exploitability[iterate] = bounds[0] + bounds[1]

    for iterate, value in exploitability.items():
        print(f"\n  exploitability({iterate}) >= {value:+.4f}   (0 exactly at a Nash equilibrium)")
    if len(exploitability) == 2:
        print(
            "  The two differ because they are two different strategies: self-play can converge "
            "in the average while the live iterate still orbits."
        )
    print("  A bound, not a value: an under-trained responder reports too little.")


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_run_config(args.config), args)
    game = config.game.build()
    if not isinstance(game, SequentialZeroSumGame):
        raise ValueError(
            f"best_response.py needs a sequential game, got {type(game).__name__}; the one-shot "
            "games train against a fixed opponent through `train.py`'s `fixed_opponent` mode"
        )

    settings = config.best_response
    step = settings.checkpoint_step
    if step is None:
        step = latest_checkpoint_step(settings.checkpoint_dir)
    print(f"responding to {settings.checkpoint_dir}/{step}.pkl")

    results = {
        (iterate, responder): run_one_direction(game, config, responder, step, iterate)
        for iterate in opponent_iterates(settings.opponent_iterate)
        for responder in responder_players(settings.responder)
    }
    report(results)


if __name__ == "__main__":
    main()
