"""Trainer: piece rollout collection, PPO updates, logging, and checkpointing together."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from games.base import ZeroSumGame

from .actor_critic import ActorCritic
from .checkpoint import load_checkpoint, save_checkpoint
from .config import PPOHyperparams
from .ppo import TrainState, create_train_state, ppo_update
from .rollout import OpponentActionFn, collect_episode


def _build_train_step(
    game: ZeroSumGame,
    network: ActorCritic,
    opponent_action_fn: OpponentActionFn,
    hyperparams: PPOHyperparams,
    perspective: int,
):
    """One jit-compiled rollout + PPO update, closing over everything that never changes.

    Compiled once on first call and reused for every subsequent iteration
    (same shapes/hyperparameters throughout a `train()` run) instead of
    re-tracing `lax.scan` from scratch every iteration.
    """

    def step(state: TrainState, key: jax.Array):
        batch = collect_episode(
            game,
            network,
            state.params,
            opponent_action_fn,
            key,
            hyperparams.num_envs,
            perspective,
        )
        state, metrics = ppo_update(state, network, batch, hyperparams)
        metrics = {**metrics, "mean_reward": jnp.mean(batch.reward)}
        return state, metrics

    return jax.jit(step)


class PPOTrainer:
    """Trains one player's `ActorCritic` policy against a (possibly fixed) opponent.

    `opponent_action_fn(key, num_envs) -> action_batch` supplies the other
    player's actions each iteration — pass `lambda k, n: game.action_space(1).sample(k, (n,))`
    for a random opponent, or wrap another trainer's current policy for self-play.
    """

    def __init__(
        self,
        game: ZeroSumGame,
        hyperparams: PPOHyperparams,
        opponent_action_fn: OpponentActionFn,
        perspective: int = 0,
        seed: int = 0,
    ):
        if hyperparams.normalization == "batch_norm":
            raise ValueError(
                "batch_norm is not supported for the actor-critic: PPO treats the network as a "
                "pure function of params, and batch_norm's running stats need a separate mutable "
                "update path this trainer doesn't implement. Use 'layer_norm', 'rms_norm', or 'none'."
            )

        self.game = game
        self.hyperparams = hyperparams
        self.opponent_action_fn = opponent_action_fn
        self.perspective = perspective

        self.network = ActorCritic(
            action_dim=hyperparams.action_dim,
            hidden_dims=hyperparams.hidden_dims,
            activation=hyperparams.activation,
            normalization=hyperparams.normalization,
        )

        key = jax.random.PRNGKey(seed)
        init_key, obs_key, self.key = jax.random.split(key, 3)
        dummy_obs = game.observation(perspective, obs_key)
        params = self.network.init(init_key, dummy_obs)
        self.state = create_train_state(self.network, params, hyperparams)

        self._train_step = _build_train_step(game, self.network, opponent_action_fn, hyperparams, perspective)

        self.history: list[dict] = []

    def train(
        self,
        num_iterations: int,
        log_every: int = 10,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 50,
    ) -> list[dict]:
        for iteration in range(1, num_iterations + 1):
            self.key, step_key = jax.random.split(self.key)
            self.state, metrics = self._train_step(self.state, step_key)

            record = {"iteration": iteration, **{k: float(v) for k, v in metrics.items()}}
            self.history.append(record)

            if iteration == 1 or iteration % log_every == 0:
                print(
                    f"iter {iteration:5d} | reward {record['mean_reward']:+.4f} "
                    f"| policy_loss {record['policy_loss']:+.4f} | value_loss {record['value_loss']:.4f} "
                    f"| entropy {record['entropy']:.4f} | approx_kl {record['approx_kl']:.4f}"
                )

            if checkpoint_dir is not None and iteration % checkpoint_every == 0:
                self.save(Path(checkpoint_dir) / f"iter_{iteration:07d}")

        return self.history

    def save(self, path: str | Path) -> None:
        save_checkpoint(path, self.hyperparams, self.state.params)

    @classmethod
    def load(
        cls,
        path: str | Path,
        game: ZeroSumGame,
        opponent_action_fn: OpponentActionFn,
        perspective: int = 0,
    ) -> "PPOTrainer":
        dummy_obs = game.observation(perspective, jax.random.PRNGKey(0))
        hyperparams, params = load_checkpoint(path, dummy_obs)
        trainer = cls(game, hyperparams, opponent_action_fn, perspective=perspective)
        trainer.state = trainer.state.replace(params=params)
        return trainer
