"""CLI entry point for training `MixtureActorCritic` players with PPO.

Takes a single argument: a YAML run config (see `configs/*.yaml` for one
example per game, and `training/run_config.py` for the schema). The same
schema and the same policy serve both families of game in `games.configs`;
which trainer runs is decided here, by what `game.build()` returns:

  one-shot (`ZeroSumGame`)
      A single simultaneous move. `train.mode` picks between
      `self_play` (both players learn) and `fixed_opponent` (one player
      learns against a non-learning opponent).

  sequential (`SequentialZeroSumGame`)
      A game tree; the policy reads an infoset observation and emits the local
      behavioral strategy. Self-play only -- there is no fixed-opponent rollout
      for a tree yet.

A sequential run reports exploitability only where an *exact* best response
exists (Kuhn). For the games where it does not, `best_response.py` trains one
instead and reports a lower bound; it reads the same config schema.

Example:
  python train.py configs/blotto.yaml
  python train.py configs/kuhn.yaml
  python train.py configs/leduc.yaml && python best_response.py configs/leduc_br.yaml
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from games.base import ZeroSumGame
from games.sequential import SequentialZeroSumGame
from games.sequential_examples import ContinuousKuhnPoker
from training.expfam_trainer import ExpFamilyPPOTrainer, ExpFamilySelfPlayPPOTrainer
from training.hyperparams import action_bounds, build_expfam_hyperparams, build_hyperparams
from training.kuhn_evaluation import build_kuhn_metric_fn, build_kuhn_strategy_log_fn
from training.mixture import OpponentActionFn
from training.mixture_trainer import MixturePPOTrainer, MixtureSelfPlayPPOTrainer
from training.run_config import RunConfig, load_run_config
from training.sequential_trainer import SequentialSelfPlayPPOTrainer


def build_opponent_action_fn(game: ZeroSumGame, perspective: int, kind: str) -> OpponentActionFn:
    opponent_player = 1 - perspective
    space = game.action_space(opponent_player)

    if kind == "random":
        def opponent_action_fn(key: jax.Array, num_envs: int) -> jax.Array:
            return space.sample(key, batch_shape=(num_envs,))

        return opponent_action_fn

    if kind == "static":
        low, high = action_bounds(space)
        midpoint = space.clip((jnp.asarray(low) + jnp.asarray(high)) / 2)

        def opponent_action_fn(key: jax.Array, num_envs: int) -> jax.Array:
            return jnp.broadcast_to(midpoint, (num_envs,) + midpoint.shape)

        return opponent_action_fn

    raise ValueError(f"unknown opponent {kind!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a YAML run config")
    return parser.parse_args()


def run_one_shot(game: ZeroSumGame, config: RunConfig) -> None:
    """Train on a single simultaneous move (`games.examples`).

    `network.policy` picks the parametrization: the default Gaussian mixture
    (`training/mixture.py`) or the log-linear exponential family
    (`training/expfam.py`). The two trainers take the same config and report the
    same metrics, so a pair of runs differing only in that field is a controlled
    comparison of the two policy classes.
    """
    expfam = config.network.policy == "exp_family"
    build = build_expfam_hyperparams if expfam else build_hyperparams
    self_play_cls = ExpFamilySelfPlayPPOTrainer if expfam else MixtureSelfPlayPPOTrainer
    fixed_opponent_cls = ExpFamilyPPOTrainer if expfam else MixturePPOTrainer

    if config.train.mode == "self_play":
        trainer = self_play_cls(
            game, build(game, 0, config), build(game, 1, config), seed=config.train.seed
        )
    elif config.train.mode == "fixed_opponent":
        hyperparams = build(game, config.train.perspective, config)
        opponent_action_fn = build_opponent_action_fn(game, config.train.perspective, config.train.opponent)
        trainer = fixed_opponent_cls(
            game, hyperparams, opponent_action_fn, perspective=config.train.perspective, seed=config.train.seed
        )
    else:
        raise ValueError(f"unknown train.mode {config.train.mode!r}")

    trainer.train(
        config.train.steps,
        epochs=config.train.epochs,
        checkpoint_dir=config.train.checkpoint_dir,
    )


def build_sequential_hooks(game: SequentialZeroSumGame, config: RunConfig) -> dict:
    """Per-game evaluation and logging hooks for `SequentialSelfPlayPPOTrainer.train`.

    Exploitability in a tree is not one algorithm -- it depends on the shape of
    the tree -- so the trainer takes it as a callback and the choice is made
    here, at the composition root, rather than inside either of them. A
    sequential game with no exact best response simply trains without one.
    """
    if isinstance(game, ContinuousKuhnPoker):
        return {
            "metric_fn": build_kuhn_metric_fn(game, config.game.exploitability_grid_points),
            "strategy_log_fn": build_kuhn_strategy_log_fn(game),
        }
    return {}


def run_sequential(game: SequentialZeroSumGame, config: RunConfig) -> None:
    """Train on a game tree (`games.sequential_examples`)."""
    if config.train.mode != "self_play":
        raise ValueError(
            f"train.mode {config.train.mode!r} is not available for a sequential game; "
            "only 'self_play' is (there is no fixed-opponent tree rollout yet)"
        )

    if config.network.policy != "gaussian_mixture":
        raise ValueError(
            f"network.policy {config.network.policy!r} is one-shot only; a game tree needs the "
            "mixture policy's atoms and legality masks (see training/expfam.py's scope note)"
        )

    hyperparams_0 = build_hyperparams(game, 0, config)
    hyperparams_1 = build_hyperparams(game, 1, config)
    trainer = SequentialSelfPlayPPOTrainer(game, hyperparams_0, hyperparams_1, seed=config.train.seed)
    trainer.train(
        config.train.steps,
        epochs=config.train.epochs,
        checkpoint_dir=config.train.checkpoint_dir,
        **build_sequential_hooks(game, config),
    )


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config)
    game = config.game.build()

    if isinstance(game, SequentialZeroSumGame):
        run_sequential(game, config)
    else:
        run_one_shot(game, config)

    if config.train.checkpoint_dir is not None:
        print(f"saved checkpoints 0.pkl..{config.train.steps}.pkl to {config.train.checkpoint_dir}")


if __name__ == "__main__":
    main()
