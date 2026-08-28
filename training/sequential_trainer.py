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

from .checkpoint import load_checkpoint_step_multi, save_checkpoint_step_multi, target_entry
from .config import MixturePPOHyperparams
from .mixture import MixtureActorCritic, build_mixture_network
from .ppo import ppo_update
from .sequential_rollout import build_episode_sampler, collect_sequential_batch
from .trainer_common import (
    MixtureTrainState,
    build_loss_fn,
    create_mixture_train_state,
    reject_batch_norm,
    run_training_chunks,
    update_target_and_magnet,
)


def _build_self_play_train_step(
    game: SequentialZeroSumGame,
    networks: tuple[MixtureActorCritic, MixtureActorCritic],
    hyperparams: tuple[MixturePPOHyperparams, MixturePPOHyperparams],
):
    sample_episode = build_episode_sampler(game, networks[0], networks[1])
    # `shared_obs` is never right here: a sequential game's whole point is a
    # per-infoset observation, so the forward pass stays inside the per-sample vmap.
    loss_fns = tuple(build_loss_fn(player, hyperparams[player]) for player in (0, 1))

    def step(state_0: MixtureTrainState, state_1: MixtureTrainState, key: jax.Array):
        # One batch of trajectories feeds both updates: the players' decisions are
        # interleaved in the same episodes, so each simply masks to its own.
        batch, payoff = collect_sequential_batch(
            sample_episode,
            state_0.params, state_0.magnet_params,
            state_1.params, state_1.magnet_params,
            key, hyperparams[0].num_envs,
        )
        state_0, metrics_0 = ppo_update(state_0, networks[0], batch, hyperparams[0], loss_fn=loss_fns[0])
        state_1, metrics_1 = ppo_update(state_1, networks[1], batch, hyperparams[1], loss_fn=loss_fns[1])
        state_0 = update_target_and_magnet(state_0, hyperparams[0])
        state_1 = update_target_and_magnet(state_1, hyperparams[1])

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
            reject_batch_norm(name, hyperparams)
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

        train_step = _build_self_play_train_step(game, self.networks, self.hyperparams)

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
        def commit(states) -> None:
            self.state_0, self.state_1 = states

        def format_record(record: dict) -> str:
            return (
                f"iter {record['iteration']:5d} | payoff {record['payoff']:+.4f} "
                f"| len {record['episode_length']:.2f} "
                f"| p0 policy {record['policy_loss_0']:+.4f} value {record['value_loss_0']:.4f} "
                f"cat_H {record['category_entropy_0']:.4f} atom {record['atom_frac_0']:.2f} "
                f"| p1 policy {record['policy_loss_1']:+.4f} value {record['value_loss_1']:.4f} "
                f"cat_H {record['category_entropy_1']:.4f} atom {record['atom_frac_1']:.2f}"
            )

        self.key = run_training_chunks(
            steps=steps,
            epochs=epochs,
            key=self.key,
            states=(self.state_0, self.state_1),
            run_chunk=self._run_chunk,
            commit=commit,
            history=self.history,
            format_record=format_record,
            metric_fn=(lambda: metric_fn(self)) if metric_fn is not None else None,
            strategy_log_fn=(lambda: strategy_log_fn(self)) if strategy_log_fn is not None else None,
            checkpoint_fn=(
                (lambda step: self.save(checkpoint_dir, step)) if checkpoint_dir is not None else None
            ),
        )
        return self.history

    def save(self, checkpoint_dir: str | Path, step: int) -> None:
        """Both players' live params, and both players' Polyak-averaged ones.
        """
        states = (self.state_0, self.state_1)
        save_checkpoint_step_multi(
            checkpoint_dir,
            step,
            {
                **{f"player_{p}": (self.hyperparams[p], states[p].params) for p in (0, 1)},
                **{
                    target_entry(f"player_{p}"): (self.hyperparams[p], states[p].target_params)
                    for p in (0, 1)
                },
            },
        )

    @classmethod
    def load(
        cls, checkpoint_dir: str | Path, step: int, game: SequentialZeroSumGame
    ) -> "SequentialSelfPlayPPOTrainer":
        """Rebuild a trainer at a checkpoint's params (optimizer state and rng restart)."""
        entries = load_checkpoint_step_multi(checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams)
        hyperparams_0, params_0 = entries["player_0"]
        hyperparams_1, params_1 = entries["player_1"]
        trainer = cls(game, hyperparams_0, hyperparams_1)
        for player, (state_name, params) in enumerate((("state_0", params_0), ("state_1", params_1))):
            # Resuming falls back to the live params where a checkpoint predates
            # target params being saved -- unlike a *measurement*, which must not
            # quietly substitute one iterate for the other.
            name = target_entry(f"player_{player}")
            target_params = entries[name][1] if name in entries else params
            state = getattr(trainer, state_name)
            setattr(
                trainer,
                state_name,
                state.replace(params=params, target_params=target_params, magnet_params=params),
            )
        return trainer
