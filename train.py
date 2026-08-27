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

Example:
  python train.py configs/blotto.yaml
  python train.py configs/kuhn.yaml
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from games.base import ZeroSumGame
from games.sequential import SequentialZeroSumGame
from games.sequential_examples import ContinuousKuhnPoker
from games.spaces import ActionSpace, BoxSpace, HybridSpace
from training.config import MixturePPOHyperparams
from training.kuhn_evaluation import build_kuhn_metric_fn, build_kuhn_strategy_log_fn
from training.mixture import OpponentActionFn
from training.mixture_trainer import MixturePPOTrainer, MixtureSelfPlayPPOTrainer
from training.run_config import RunConfig, load_run_config
from training.sequential_trainer import SequentialSelfPlayPPOTrainer


def _action_bounds(space: ActionSpace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`(low, high)` for spreading mixture-component means; see `MixturePPOHyperparams`.

    Exact for a `BoxSpace`, and for the `box` inside a `HybridSpace`. A
    `SimplexSpace` has no natural per-axis bounds, so it's approximated as
    `[0, total]` per dimension -- only used for initialization, the actual
    action is still projected onto the simplex.
    """
    if isinstance(space, (BoxSpace, HybridSpace)):
        return tuple(float(x) for x in space.low), tuple(float(x) for x in space.high)
    total = float(getattr(space, "total"))
    dim = space.shape[0]
    return (0.0,) * dim, (total,) * dim


def _num_atoms(space: ActionSpace) -> int:
    """How many parameterless discrete actions the policy's categorical head needs.

    Read off the action space rather than the config: the atoms are a property
    of the *game* (a `HybridSpace` says how many discrete choices its tree
    offers), not a network size to be tuned. Every purely continuous space has
    none.
    """
    return space.num_atoms if isinstance(space, HybridSpace) else 0


def build_hyperparams(game: ZeroSumGame, player: int, config: RunConfig) -> MixturePPOHyperparams:
    space = game.action_space(player)
    low, high = _action_bounds(space)
    network, optimizer, ppo = config.network, config.optimizer, config.ppo
    return MixturePPOHyperparams(
        action_dim=space.shape[0],
        hidden_dims=tuple(network.hidden_dims),
        activation=network.activation,
        normalization=network.normalization,
        learning_rate=optimizer.learning_rate,
        max_grad_norm=optimizer.max_grad_norm,
        optimizer=optimizer.optimizer,
        weight_decay=optimizer.weight_decay,
        clip_eps=ppo.clip_eps,
        value_coef=ppo.value_coef,
        num_envs=ppo.batch_size,
        num_epochs=ppo.ppo_epochs,
        num_components=network.num_components,
        num_atoms=_num_atoms(space),
        clip_means=network.clip_means,
        low=low,
        high=high,
        target_tau=ppo.target_tau,
        magnet_interval=ppo.magnet_interval,
        category_entropy_coef=ppo.category_entropy_coef,
        gaussian_entropy_coef=ppo.gaussian_entropy_coef,
        trpo_category_kl_coef=ppo.trpo_category_kl_coef,
        trpo_gaussian_kl_coef=ppo.trpo_gaussian_kl_coef,
        magnet_category_kl_coef=ppo.magnet_category_kl_coef,
        magnet_gaussian_kl_coef=ppo.magnet_gaussian_kl_coef,
    )


def build_opponent_action_fn(game: ZeroSumGame, perspective: int, kind: str) -> OpponentActionFn:
    opponent_player = 1 - perspective
    space = game.action_space(opponent_player)

    if kind == "random":
        def opponent_action_fn(key: jax.Array, num_envs: int) -> jax.Array:
            return space.sample(key, batch_shape=(num_envs,))

        return opponent_action_fn

    if kind == "static":
        low, high = _action_bounds(space)
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
    """Train on a single simultaneous move (`games.examples`)."""
    if config.train.mode == "self_play":
        hyperparams_1 = build_hyperparams(game, 0, config)
        hyperparams_2 = build_hyperparams(game, 1, config)
        trainer = MixtureSelfPlayPPOTrainer(game, hyperparams_1, hyperparams_2, seed=config.train.seed)
    elif config.train.mode == "fixed_opponent":
        hyperparams = build_hyperparams(game, config.train.perspective, config)
        opponent_action_fn = build_opponent_action_fn(game, config.train.perspective, config.train.opponent)
        trainer = MixturePPOTrainer(
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
