"""Self-play PPO: both players' policies are updated simultaneously each iteration.

This is simultaneous gradient descent-ascent on a stochastic zero-sum
objective. It has no general convergence guarantee to a Nash equilibrium
(the same family of dynamics as GAN training — it can cycle around an
equilibrium rather than settle on it), but it's the standard way to try to
approximate one for games without closed-form solutions. Pass
`exploitability_every` to track, via `game.exploitability`, how close the
current (mean-action) strategy pair actually gets.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from games.base import ZeroSumGame

from .actor_critic import ActorCritic
from .checkpoint import load_checkpoint, save_checkpoint
from .config import PPOHyperparams
from .ppo import TrainState, create_train_state, ppo_update
from .rollout import collect_self_play_episode


def _build_train_step(
    game: ZeroSumGame,
    network_1: ActorCritic,
    network_2: ActorCritic,
    hyperparams_1: PPOHyperparams,
    hyperparams_2: PPOHyperparams,
):
    """One jit-compiled self-play rollout + both PPO updates; compiled once, reused every iteration."""

    def step(state_1: TrainState, state_2: TrainState, key: jax.Array):
        batch_1, batch_2 = collect_self_play_episode(
            game,
            network_1,
            state_1.params,
            network_2,
            state_2.params,
            key,
            hyperparams_1.num_envs,
        )
        state_1, metrics_1 = ppo_update(state_1, network_1, batch_1, hyperparams_1)
        state_2, metrics_2 = ppo_update(state_2, network_2, batch_2, hyperparams_2)
        metrics = {
            "mean_reward_1": jnp.mean(batch_1.reward),
            **{f"{k}_1": v for k, v in metrics_1.items()},
            **{f"{k}_2": v for k, v in metrics_2.items()},
        }
        return state_1, state_2, metrics

    return jax.jit(step)


class SelfPlayPPOTrainer:
    """Trains both players of a `ZeroSumGame` against each other simultaneously."""

    def __init__(
        self,
        game: ZeroSumGame,
        hyperparams_1: PPOHyperparams,
        hyperparams_2: PPOHyperparams,
        seed: int = 0,
    ):
        for name, hyperparams in (("hyperparams_1", hyperparams_1), ("hyperparams_2", hyperparams_2)):
            if hyperparams.normalization == "batch_norm":
                raise ValueError(f"{name}: batch_norm is not supported (see PPOTrainer for why).")
        if hyperparams_1.num_envs != hyperparams_2.num_envs:
            raise ValueError("hyperparams_1.num_envs must equal hyperparams_2.num_envs for self-play rollouts")

        self.game = game
        self.hyperparams_1 = hyperparams_1
        self.hyperparams_2 = hyperparams_2

        self.network_1 = ActorCritic(
            action_dim=hyperparams_1.action_dim,
            hidden_dims=hyperparams_1.hidden_dims,
            activation=hyperparams_1.activation,
            normalization=hyperparams_1.normalization,
        )
        self.network_2 = ActorCritic(
            action_dim=hyperparams_2.action_dim,
            hidden_dims=hyperparams_2.hidden_dims,
            activation=hyperparams_2.activation,
            normalization=hyperparams_2.normalization,
        )

        key = jax.random.PRNGKey(seed)
        init_key_1, init_key_2, obs_key_1, obs_key_2, self.key = jax.random.split(key, 5)
        params_1 = self.network_1.init(init_key_1, game.observation(0, obs_key_1))
        params_2 = self.network_2.init(init_key_2, game.observation(1, obs_key_2))
        self.state_1 = create_train_state(self.network_1, params_1, hyperparams_1)
        self.state_2 = create_train_state(self.network_2, params_2, hyperparams_2)

        self._train_step = _build_train_step(game, self.network_1, self.network_2, hyperparams_1, hyperparams_2)
        self._exploitability_fn = jax.jit(game.exploitability)

        self.history: list[dict] = []

    def train(
        self,
        num_iterations: int,
        log_every: int = 10,
        exploitability_every: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 50,
    ) -> list[dict]:
        for iteration in range(1, num_iterations + 1):
            self.key, step_key, exploit_key = jax.random.split(self.key, 3)

            self.state_1, self.state_2, metrics = self._train_step(self.state_1, self.state_2, step_key)
            record = {"iteration": iteration, **{k: float(v) for k, v in metrics.items()}}

            if exploitability_every is not None and (iteration == 1 or iteration % exploitability_every == 0):
                record["exploitability"] = float(self._exploitability(exploit_key))

            self.history.append(record)

            if iteration == 1 or iteration % log_every == 0:
                message = (
                    f"iter {iteration:5d} | reward {record['mean_reward_1']:+.4f} "
                    f"| p1 policy_loss {record['policy_loss_1']:+.4f} value_loss {record['value_loss_1']:.4f} "
                    f"| p2 policy_loss {record['policy_loss_2']:+.4f} value_loss {record['value_loss_2']:.4f}"
                )
                if "exploitability" in record:
                    message += f" | exploitability {record['exploitability']:.4f}"
                print(message)

            if checkpoint_dir is not None and iteration % checkpoint_every == 0:
                self.save(Path(checkpoint_dir) / f"iter_{iteration:07d}")

        return self.history

    def _exploitability(self, key: jax.Array) -> jax.Array:
        """Exploitability of the two policies' mean actions (a deterministic strategy snapshot)."""
        obs_1 = self.game.observation(0, jax.random.PRNGKey(0))
        obs_2 = self.game.observation(1, jax.random.PRNGKey(0))
        mean_action_1, _, _ = self.network_1.apply(self.state_1.params, obs_1)
        mean_action_2, _, _ = self.network_2.apply(self.state_2.params, obs_2)
        mean_action_1 = self.game.action_space(0).clip(mean_action_1)
        mean_action_2 = self.game.action_space(1).clip(mean_action_2)
        return self._exploitability_fn(mean_action_1, mean_action_2, key)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        save_checkpoint(path / "player_1", self.hyperparams_1, self.state_1.params)
        save_checkpoint(path / "player_2", self.hyperparams_2, self.state_2.params)

    @classmethod
    def load(cls, path: str | Path, game: ZeroSumGame) -> "SelfPlayPPOTrainer":
        path = Path(path)
        obs_key = jax.random.PRNGKey(0)
        hyperparams_1, params_1 = load_checkpoint(path / "player_1", game.observation(0, obs_key))
        hyperparams_2, params_2 = load_checkpoint(path / "player_2", game.observation(1, obs_key))
        trainer = cls(game, hyperparams_1, hyperparams_2)
        trainer.state_1 = trainer.state_1.replace(params=params_1)
        trainer.state_2 = trainer.state_2.replace(params=params_2)
        return trainer
