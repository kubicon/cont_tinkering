"""CLI entry point for training `MixtureActorCritic` players with PPO.

Takes a single argument: a YAML run config (see `configs/*.yaml` for one
example per game, and `training/run_config.py` for the schema). Two modes,
set via `train.mode` in the config:

  self_play        -- both players train simultaneously against each other
                       (`MixtureSelfPlayPPOTrainer`).
  fixed_opponent    -- one player trains against a fixed, non-learning
                       opponent (`MixturePPOTrainer`).

Example:
  python train.py configs/blotto.yaml
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from games.base import ZeroSumGame
from games.spaces import ActionSpace, BoxSpace
from training.config import MixturePPOHyperparams
from training.mixture import OpponentActionFn
from training.mixture_trainer import MixturePPOTrainer, MixtureSelfPlayPPOTrainer
from training.run_config import RunConfig, load_run_config


def _action_bounds(space: ActionSpace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`(low, high)` for spreading mixture-component means; see `MixturePPOHyperparams`.

    Exact for a `BoxSpace`. A `SimplexSpace` has no natural per-axis bounds,
    so it's approximated as `[0, total]` per dimension -- only used for
    initialization, the actual action is still projected onto the simplex.
    """
    if isinstance(space, BoxSpace):
        return tuple(float(x) for x in space.low), tuple(float(x) for x in space.high)
    total = float(getattr(space, "total"))
    dim = space.shape[0]
    return (0.0,) * dim, (total,) * dim


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
        clip_eps=ppo.clip_eps,
        value_coef=ppo.value_coef,
        num_envs=ppo.batch_size,
        num_epochs=ppo.ppo_epochs,
        num_components=network.num_components,
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


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config)
    game = config.game.build()

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

    if config.train.checkpoint_dir is not None:
        print(f"saved checkpoints 0.pkl..{config.train.steps}.pkl to {config.train.checkpoint_dir}")


if __name__ == "__main__":
    main()
