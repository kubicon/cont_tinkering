"""Self-play PPO on a `SequentialZeroSumGame`.

The loss itself lives in `training.mixture`: a trajectory batch is an `Episode`
like any other, and `build_mixture_ppo_loss_fn(player, ...)` already weights
every reduction by `Episode.actor == player`, which is exactly what selects one
player's decisions out of an interleaved, padded trajectory. All that is left
here is the trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from games.sequential import TERMINAL, SequentialZeroSumGame

from .checkpoint import load_checkpoint_step_multi, save_checkpoint_step_multi
from .config import MixturePPOHyperparams
from .mixture import MixtureActorCritic, build_mixture_network, build_mixture_ppo_loss_fn
from .mixture_trainer import (
    MixtureTrainState,
    _update_target_and_magnet,
    create_mixture_train_state,
)
from .ppo import ppo_update
from .sequential_rollout import build_episode_sampler, collect_sequential_batch


def _build_self_play_train_step(
    game: SequentialZeroSumGame,
    network_0: MixtureActorCritic,
    network_1: MixtureActorCritic,
    hyperparams_0: MixturePPOHyperparams,
    hyperparams_1: MixturePPOHyperparams,
):
    sample_episode = build_episode_sampler(game, network_0, network_1)
    loss_fns = tuple(
        build_mixture_ppo_loss_fn(
            player,
            hyperparams.category_entropy_coef, hyperparams.gaussian_entropy_coef,
            hyperparams.trpo_category_kl_coef, hyperparams.trpo_gaussian_kl_coef,
            hyperparams.magnet_category_kl_coef, hyperparams.magnet_gaussian_kl_coef,
        )
        for player, hyperparams in enumerate((hyperparams_0, hyperparams_1))
    )

    def step(state_0: MixtureTrainState, state_1: MixtureTrainState, key: jax.Array):
        # One batch of trajectories feeds both updates: the players' decisions are
        # interleaved in the same episodes, so each simply masks to its own.
        batch, payoff = collect_sequential_batch(
            sample_episode,
            state_0.params, state_0.magnet_params,
            state_1.params, state_1.magnet_params,
            key, hyperparams_0.num_envs,
        )
        state_0, metrics_0 = ppo_update(state_0, network_0, batch, hyperparams_0, loss_fn=loss_fns[0])
        state_1, metrics_1 = ppo_update(state_1, network_1, batch, hyperparams_1, loss_fn=loss_fns[1])
        state_0 = _update_target_and_magnet(state_0, hyperparams_0)
        state_1 = _update_target_and_magnet(state_1, hyperparams_1)

        metrics = {
            "payoff": jnp.mean(payoff),
            "episode_length": jnp.mean(jnp.sum((batch.actor != TERMINAL).astype(jnp.float32), axis=-1)),
            **{f"{k}_0": v for k, v in metrics_0.items()},
            **{f"{k}_1": v for k, v in metrics_1.items()},
        }
        return state_0, state_1, metrics

    return step


class SequentialSelfPlayPPOTrainer:
    """Trains both players' `MixtureActorCritic`s on a sequential game, simultaneously.
    """

    def __init__(
        self,
        game: SequentialZeroSumGame,
        hyperparams_0: MixturePPOHyperparams,
        hyperparams_1: MixturePPOHyperparams,
        seed: int = 0,
    ):
        for name, hyperparams in (("hyperparams_0", hyperparams_0), ("hyperparams_1", hyperparams_1)):
            if hyperparams.normalization == "batch_norm":
                raise ValueError(f"{name}: batch_norm is not supported (see PPOTrainer for why).")
        if hyperparams_0.num_envs != hyperparams_1.num_envs:
            raise ValueError("hyperparams_0.num_envs must equal hyperparams_1.num_envs for self-play rollouts")

        self.game = game
        self.hyperparams = (hyperparams_0, hyperparams_1)
        self.networks = (build_mixture_network(hyperparams_0), build_mixture_network(hyperparams_1))

        key = jax.random.PRNGKey(seed)
        init_key_0, init_key_1, state_key, self.key = jax.random.split(key, 4)
        dummy_state = game.initial_state(state_key)
        params = (
            self.networks[0].init(init_key_0, game.observation(0, dummy_state)),
            self.networks[1].init(init_key_1, game.observation(1, dummy_state)),
        )
        self.state_0 = create_mixture_train_state(self.networks[0], params[0], hyperparams_0)
        self.state_1 = create_mixture_train_state(self.networks[1], params[1], hyperparams_1)

        train_step = _build_self_play_train_step(
            game, self.networks[0], self.networks[1], hyperparams_0, hyperparams_1
        )

        def scan_body(states, key):
            state_0, state_1 = states
            state_0, state_1, metrics = train_step(state_0, state_1, key)
            return (state_0, state_1), metrics

        self._run_chunk = jax.jit(lambda states, keys: jax.lax.scan(scan_body, states, keys))
        self.history: list[dict] = []

    @property
    def params(self) -> tuple:
        return (self.state_0.params, self.state_1.params)

    @property
    def target_params(self) -> tuple:
        return (self.state_0.target_params, self.state_1.target_params)

    def train(
        self,
        steps: int,
        epochs: int = 10,
        checkpoint_dir: str | Path | None = None,
        metric_fn: Callable[["SequentialSelfPlayPPOTrainer"], dict[str, float]] | None = None,
        strategy_log_fn: Callable[["SequentialSelfPlayPPOTrainer"], str] | None = None,
    ) -> list[dict]:
        """Trains for `steps` chunks of `epochs` `lax.scan`-ned iterations each,
        logging and checkpointing once per chunk (`steps * epochs` iterations total).
        """
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        if checkpoint_dir is not None:
            self.save(checkpoint_dir, 0)

        for chunk in range(steps):
            self.key, chunk_key = jax.random.split(self.key)
            step_keys = jax.random.split(chunk_key, epochs)
            (self.state_0, self.state_1), metrics_stack = self._run_chunk(
                (self.state_0, self.state_1), step_keys
            )

            # One device-to-host transfer for the whole chunk. Indexing the device
            # arrays per iteration instead costs a dispatch and a sync *per metric
            # per iteration*, which for a 300-iteration chunk takes several times
            # longer than the training it is reporting on.
            metrics_chunk = jax.device_get(metrics_stack)
            for offset in range(epochs):
                iteration = chunk * epochs + offset + 1
                record = {"iteration": iteration, **{k: float(v[offset]) for k, v in metrics_chunk.items()}}
                self.history.append(record)

            # Evaluated once per chunk, on the parameters as they now stand, so
            # it attaches to that chunk's last record rather than to every iteration.
            extra = metric_fn(self) if metric_fn is not None else {}
            record.update(extra)

            print(
                f"iter {iteration:5d} | payoff {record['payoff']:+.4f} "
                f"| len {record['episode_length']:.2f} "
                f"| p0 policy {record['policy_loss_0']:+.4f} value {record['value_loss_0']:.4f} "
                f"cat_H {record['category_entropy_0']:.4f} atom {record['atom_frac_0']:.2f} "
                f"| p1 policy {record['policy_loss_1']:+.4f} value {record['value_loss_1']:.4f} "
                f"cat_H {record['category_entropy_1']:.4f} atom {record['atom_frac_1']:.2f}"
            )
            if extra:
                print("  " + "  ".join(f"{k} {v:+.5f}" for k, v in extra.items()))
            if strategy_log_fn is not None:
                print(strategy_log_fn(self))

            if checkpoint_dir is not None:
                self.save(checkpoint_dir, chunk + 1)

        return self.history

    def save(self, checkpoint_dir: str | Path, step: int) -> None:
        save_checkpoint_step_multi(
            checkpoint_dir,
            step,
            {
                "player_0": (self.hyperparams[0], self.state_0.params),
                "player_1": (self.hyperparams[1], self.state_1.params),
            },
        )

    @classmethod
    def load(
        cls, checkpoint_dir: str | Path, step: int, game: SequentialZeroSumGame
    ) -> "SequentialSelfPlayPPOTrainer":
        entries = load_checkpoint_step_multi(checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams)
        hyperparams_0, params_0 = entries["player_0"]
        hyperparams_1, params_1 = entries["player_1"]
        trainer = cls(game, hyperparams_0, hyperparams_1)
        trainer.state_0 = trainer.state_0.replace(
            params=params_0, target_params=params_0, magnet_params=params_0
        )
        trainer.state_1 = trainer.state_1.replace(
            params=params_1, target_params=params_1, magnet_params=params_1
        )
        return trainer
