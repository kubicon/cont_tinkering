"""Trainers for `MixtureActorCritic`: mirrors `trainer.py`/`self_play.py`, swapping
in the mixture rollout/loss functions from `training/mixture.py`.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.base import ZeroSumGame

from .actor_critic import masked_log_softmax
from .checkpoint import (
    load_checkpoint_step,
    load_checkpoint_step_multi,
    save_checkpoint_step,
    save_checkpoint_step_multi,
)
from .config import MixturePPOHyperparams
from .mixture import (
    MixtureActorCritic,
    OpponentActionFn,
    build_mixture_network,
    collect_mixture_episode,
    collect_mixture_self_play_episode,
    sample_mixture_actions,
)
from .ppo import ppo_update
from .trainer_common import (
    MixtureTrainState,
    append_chunk_records,
    build_loss_fn,
    create_mixture_train_state,
    reject_batch_norm,
    update_target_and_magnet,
)

__all__ = [
    "MixturePPOTrainer",
    "MixtureSelfPlayPPOTrainer",
    "MixtureTrainState",
    "create_mixture_train_state",
    "update_target_and_magnet",
]

# Monte-Carlo samples drawn from each mixture policy when estimating its exploitability.
EXPLOITABILITY_SAMPLES = 512

# The observation of a one-shot game is a constant (see
# `ZeroSumGame.constant_observation`), so the key it is drawn with is irrelevant
# and fixing it keeps every readout comparable across chunks.
_OBSERVATION_KEY = jax.random.PRNGKey(0)


def _format_vector(values) -> tuple[float, ...]:
    return tuple(round(float(v), 3) for v in values)


def _build_strategy_str(network: MixtureActorCritic):
    """`(params, obs, mask=None) -> str`, the full behavioral strategy at `obs`.

    Logging runs once per chunk, but an eager `network.apply` there costs more
    than the whole chunk of training it reports on: every op dispatches
    separately and every `float()` on a device scalar is its own
    device-to-host sync. Jitting the forward pass and pulling the three arrays
    back in one `device_get` makes the whole thing a rounding error again.
    """
    num_atoms, num_components = network.num_atoms, network.num_components
    default_mask = np.ones(num_atoms + num_components, dtype=bool)
    default_mask_device = jnp.asarray(default_mask)

    @jax.jit
    def forward(params, obs: chex.Array, mask: chex.Array):
        logits, means, log_stds, _ = network.apply(params, obs)
        return jnp.exp(masked_log_softmax(logits, mask)), means, jnp.exp(log_stds)

    def strategy_str(params, obs: chex.Array, mask: chex.Array | None = None) -> str:
        if mask is None:
            mask, mask_device = default_mask, default_mask_device
        else:
            mask, mask_device = np.asarray(mask), jnp.asarray(mask)
        probs, means, stds = jax.device_get(forward(params, obs, mask_device))

        parts = [f"atom{i} {probs[i]:.2f}" for i in range(num_atoms) if mask[i]]
        parts += [
            f"{probs[num_atoms + k]:.2f}x{_format_vector(means[k])}±{_format_vector(stds[k])}"
            for k in range(num_components)
            if mask[num_atoms + k]
        ]
        return ", ".join(parts)

    return strategy_str


def _build_train_step(
    game: ZeroSumGame,
    network: MixtureActorCritic,
    opponent_action_fn: OpponentActionFn,
    hyperparams: MixturePPOHyperparams,
    perspective: int,
):
    loss_fn = build_loss_fn(perspective, hyperparams, shared_obs=game.constant_observation)

    def step(state: MixtureTrainState, key: jax.Array):
        batch = collect_mixture_episode(
            game, network, state.params, state.magnet_params, opponent_action_fn,
            key, hyperparams.num_envs, perspective,
        )
        state, metrics = ppo_update(state, network, batch, hyperparams, loss_fn=loss_fn)
        state = update_target_and_magnet(state, hyperparams)
        metrics = {**metrics, "mean_reward": jnp.mean(batch.reward)}
        return state, metrics

    return step


class MixturePPOTrainer:
    """Trains one player's `MixtureActorCritic` against a (possibly fixed) opponent.

    Same contract as `PPOTrainer`, but for a policy that can represent
    multi-modal / finite-support mixed strategies.
    """

    def __init__(
        self,
        game: ZeroSumGame,
        hyperparams: MixturePPOHyperparams,
        opponent_action_fn: OpponentActionFn,
        perspective: int = 0,
        seed: int = 0,
    ):
        reject_batch_norm("hyperparams", hyperparams)
        if hyperparams.num_atoms != 0:
            # `train` reports exploitability every chunk via
            # `sample_mixture_actions`, which is meaningless with atoms. Fail
            # here rather than at the end of the first chunk.
            raise ValueError(
                f"one-shot games have no discrete actions, got num_atoms={hyperparams.num_atoms}"
            )

        self.game = game
        self.hyperparams = hyperparams
        self.opponent_action_fn = opponent_action_fn
        self.perspective = perspective

        self.network = build_mixture_network(hyperparams)

        key = jax.random.PRNGKey(seed)
        init_key, obs_key, self.key = jax.random.split(key, 3)
        dummy_obs = game.observation(perspective, obs_key)
        params = self.network.init(init_key, dummy_obs)
        self.state = create_mixture_train_state(self.network, params, hyperparams)

        train_step = _build_train_step(game, self.network, opponent_action_fn, hyperparams, perspective)
        self._run_chunk = jax.jit(lambda state, keys: jax.lax.scan(train_step, state, keys))
        self._strategy_str = _build_strategy_str(self.network)

        # Constant for the whole run, so they are computed once rather than
        # rebuilt (and re-transferred) on every chunk's log line.
        self._obs = game.observation(perspective, _OBSERVATION_KEY)
        self._opponent_sample = _format_vector(
            jnp.squeeze(opponent_action_fn(_OBSERVATION_KEY, 1), axis=0)
        )
        self._exploitability = self._build_exploitability_fn()

        self.history: list[dict] = []

    def _build_exploitability_fn(self):
        """`(params, target_params, key) -> (live, target)` exploitabilities, jitted whole.

        Samples a batch of actions from each mixture policy (rather than
        collapsing it to its mode), so a genuinely multi-modal equilibrium --
        which no single action can represent -- reads as unexploitable.

        Both iterates go through one jitted call: the sampling used to run
        eagerly around a jitted best-response, which dispatched the policy
        forward pass op by op, and the two measurements then cost two
        device-to-host syncs instead of one.
        """
        game, network, perspective = self.game, self.network, self.perspective
        space = game.action_space(perspective)
        opponent_action_fn = self.opponent_action_fn
        obs = self._obs

        def exploitability(params, key: jax.Array) -> jax.Array:
            sample_key, opponent_key, exploit_key = jax.random.split(key, 3)
            actions = sample_mixture_actions(
                network, params, obs, space, sample_key, EXPLOITABILITY_SAMPLES
            )
            opponent_actions = opponent_action_fn(opponent_key, EXPLOITABILITY_SAMPLES)
            if perspective == 0:
                return game.mixture_exploitability(actions, opponent_actions, exploit_key)
            return game.mixture_exploitability(opponent_actions, actions, exploit_key)

        @jax.jit
        def both(params, target_params, key: jax.Array):
            live_key, target_key = jax.random.split(key)
            return exploitability(params, live_key), exploitability(target_params, target_key)

        return both

    def train(
        self,
        steps: int,
        epochs: int = 10,
        checkpoint_dir: str | Path | None = None,
    ) -> list[dict]:
        """Trains for `steps` chunks of `epochs` `lax.scan`-ned iterations each,
        logging and checkpointing once per chunk (`steps * epochs` iterations total).

        Checkpoints are `{step}.pkl` inside `checkpoint_dir`: `0.pkl` is the
        untrained, freshly initialized params, `1.pkl` is after the first chunk, etc.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        if checkpoint_dir is not None:
            self.save(checkpoint_dir, 0)

        for chunk in range(steps):
            self.key, chunk_key, exploit_key = jax.random.split(self.key, 3)
            step_keys = jax.random.split(chunk_key, epochs)
            self.state, metrics_stack = self._run_chunk(self.state, step_keys)

            record = append_chunk_records(self.history, metrics_stack, chunk, epochs)
            live, target = jax.device_get(
                self._exploitability(self.state.params, self.state.target_params, exploit_key)
            )
            record["exploitability"] = float(live)
            record["exploitability_target"] = float(target)

            print(
                f"iter {record['iteration']:5d} | reward {record['mean_reward']:+.4f} "
                f"| policy_loss {record['policy_loss']:+.4f} | value_loss {record['value_loss']:.4f} "
                f"| entropy {record['entropy']:.4f} (category {record['category_entropy']:.4f}) "
                f"| approx_kl {record['approx_kl']:.4f} "
                f"| exploitability {record['exploitability']:.4f}"
                f" (target {record['exploitability_target']:.4f})"
            )
            print(
                f"  player {self.perspective} strategy: "
                f"{self._strategy_str(self.state.params, self._obs)}"
            )
            print(
                f"  player {self.perspective} target strategy: "
                f"{self._strategy_str(self.state.target_params, self._obs)}"
            )
            print(f"  opponent sample: {self._opponent_sample}")

            if checkpoint_dir is not None:
                self.save(checkpoint_dir, chunk + 1)

        return self.history

    def save(self, checkpoint_dir: str | Path, step: int) -> None:
        save_checkpoint_step(checkpoint_dir, step, self.hyperparams, self.state.params)

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        step: int,
        game: ZeroSumGame,
        opponent_action_fn: OpponentActionFn,
        perspective: int = 0,
    ) -> "MixturePPOTrainer":
        """Rebuild a trainer at a checkpoint's params.

        Only params are stored, so the optimizer state and rng stream restart
        from scratch: this resumes a *policy*, not a run.
        """
        hyperparams, params = load_checkpoint_step(checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams)
        trainer = cls(game, hyperparams, opponent_action_fn, perspective=perspective)
        trainer.state = trainer.state.replace(params=params, target_params=params, magnet_params=params)
        return trainer


def _build_self_play_train_step(
    game: ZeroSumGame,
    networks: tuple[MixtureActorCritic, MixtureActorCritic],
    hyperparams: tuple[MixturePPOHyperparams, MixturePPOHyperparams],
):
    loss_fns = tuple(
        build_loss_fn(player, hyperparams[player], shared_obs=game.constant_observation)
        for player in (0, 1)
    )

    def step(state_1: MixtureTrainState, state_2: MixtureTrainState, key: jax.Array):
        batch_1, batch_2 = collect_mixture_self_play_episode(
            game, networks[0], state_1.params, state_1.magnet_params,
            networks[1], state_2.params, state_2.magnet_params,
            key, hyperparams[0].num_envs,
        )
        state_1, metrics_1 = ppo_update(state_1, networks[0], batch_1, hyperparams[0], loss_fn=loss_fns[0])
        state_2, metrics_2 = ppo_update(state_2, networks[1], batch_2, hyperparams[1], loss_fn=loss_fns[1])
        state_1 = update_target_and_magnet(state_1, hyperparams[0])
        state_2 = update_target_and_magnet(state_2, hyperparams[1])
        metrics = {
            "mean_reward_1": jnp.mean(batch_1.reward),
            **{f"{k}_1": v for k, v in metrics_1.items()},
            **{f"{k}_2": v for k, v in metrics_2.items()},
        }
        return state_1, state_2, metrics

    return step


class MixtureSelfPlayPPOTrainer:
    """Trains both players' `MixtureActorCritic`s against each other simultaneously."""

    def __init__(
        self,
        game: ZeroSumGame,
        hyperparams_1: MixturePPOHyperparams,
        hyperparams_2: MixturePPOHyperparams,
        seed: int = 0,
    ):
        for name, hyperparams in (("hyperparams_1", hyperparams_1), ("hyperparams_2", hyperparams_2)):
            reject_batch_norm(name, hyperparams)
            if hyperparams.num_atoms != 0:
                raise ValueError(
                    f"{name}: one-shot games have no discrete actions, "
                    f"got num_atoms={hyperparams.num_atoms}"
                )
        if hyperparams_1.num_envs != hyperparams_2.num_envs:
            raise ValueError("hyperparams_1.num_envs must equal hyperparams_2.num_envs for self-play rollouts")

        self.game = game
        self.hyperparams_1 = hyperparams_1
        self.hyperparams_2 = hyperparams_2

        self.network_1 = build_mixture_network(hyperparams_1)
        self.network_2 = build_mixture_network(hyperparams_2)

        key = jax.random.PRNGKey(seed)
        init_key_1, init_key_2, obs_key_1, obs_key_2, self.key = jax.random.split(key, 5)
        params_1 = self.network_1.init(init_key_1, game.observation(0, obs_key_1))
        params_2 = self.network_2.init(init_key_2, game.observation(1, obs_key_2))
        self.state_1 = create_mixture_train_state(self.network_1, params_1, hyperparams_1)
        self.state_2 = create_mixture_train_state(self.network_2, params_2, hyperparams_2)

        train_step = _build_self_play_train_step(
            game, (self.network_1, self.network_2), (hyperparams_1, hyperparams_2)
        )

        def scan_body(states, key):
            state_1, state_2 = states
            state_1, state_2, metrics = train_step(state_1, state_2, key)
            return (state_1, state_2), metrics

        self._run_chunk = jax.jit(lambda states, keys: jax.lax.scan(scan_body, states, keys))
        self._strategy_str_1 = _build_strategy_str(self.network_1)
        self._strategy_str_2 = _build_strategy_str(self.network_2)

        # Constant for the whole run (see `_OBSERVATION_KEY`).
        self._obs_1 = game.observation(0, _OBSERVATION_KEY)
        self._obs_2 = game.observation(1, _OBSERVATION_KEY)
        self._exploitability = self._build_exploitability_fn()

        self.history: list[dict] = []

    @property
    def params(self) -> tuple:
        return (self.state_1.params, self.state_2.params)

    @property
    def target_params(self) -> tuple:
        return (self.state_1.target_params, self.state_2.target_params)

    def _build_exploitability_fn(self):
        """`(params, target_params, key) -> (live, target)` exploitabilities, jitted whole.

        Each mixture is Monte-Carlo sampled rather than collapsed to its mode,
        so a multi-modal equilibrium (which no single action can represent)
        reads as unexploitable instead of maximally exploitable. Both iterates
        share one jitted call and so one device-to-host sync.
        """
        game = self.game
        networks = (self.network_1, self.network_2)
        spaces = (game.action_space(0), game.action_space(1))
        observations = (self._obs_1, self._obs_2)

        def exploitability(params: tuple, key: jax.Array) -> jax.Array:
            sample_key_1, sample_key_2, exploit_key = jax.random.split(key, 3)
            actions = tuple(
                sample_mixture_actions(
                    networks[i], params[i], observations[i], spaces[i], sample_key, EXPLOITABILITY_SAMPLES
                )
                for i, sample_key in enumerate((sample_key_1, sample_key_2))
            )
            return game.mixture_exploitability(actions[0], actions[1], exploit_key)

        @jax.jit
        def both(params: tuple, target_params: tuple, key: jax.Array):
            live_key, target_key = jax.random.split(key)
            return exploitability(params, live_key), exploitability(target_params, target_key)

        return both

    def train(
        self,
        steps: int,
        epochs: int = 10,
        checkpoint_dir: str | Path | None = None,
    ) -> list[dict]:
        """Trains for `steps` chunks of `epochs` `lax.scan`-ned iterations each,
        logging and checkpointing once per chunk (`steps * epochs` iterations total).

        Checkpoints are `{step}.pkl` inside `checkpoint_dir`, holding both
        players' params: `0.pkl` is the untrained, freshly initialized params,
        `1.pkl` is after the first chunk, etc.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        if checkpoint_dir is not None:
            self.save(checkpoint_dir, 0)

        for chunk in range(steps):
            self.key, chunk_key, exploit_key = jax.random.split(self.key, 3)
            step_keys = jax.random.split(chunk_key, epochs)
            (self.state_1, self.state_2), metrics_stack = self._run_chunk(
                (self.state_1, self.state_2), step_keys
            )

            record = append_chunk_records(self.history, metrics_stack, chunk, epochs)
            live, target = jax.device_get(
                self._exploitability(self.params, self.target_params, exploit_key)
            )
            record["exploitability"] = float(live)
            record["exploitability_target"] = float(target)

            print(
                f"iter {record['iteration']:5d} | reward {record['mean_reward_1']:+.4f} "
                f"| p1 policy_loss {record['policy_loss_1']:+.4f} value_loss {record['value_loss_1']:.4f} "
                f"category_entropy {record['category_entropy_1']:.4f} "
                f"| p2 policy_loss {record['policy_loss_2']:+.4f} value_loss {record['value_loss_2']:.4f} "
                f"category_entropy {record['category_entropy_2']:.4f} "
                f"| exploitability {record['exploitability']:.4f}"
                f" (target {record['exploitability_target']:.4f})"
            )
            print(f"  p1 strategy: {self._strategy_str_1(self.state_1.params, self._obs_1)}")
            print(f"  p2 strategy: {self._strategy_str_2(self.state_2.params, self._obs_2)}")
            print(f"  p1 target strategy: {self._strategy_str_1(self.state_1.target_params, self._obs_1)}")
            print(f"  p2 target strategy: {self._strategy_str_2(self.state_2.target_params, self._obs_2)}")

            if checkpoint_dir is not None:
                self.save(checkpoint_dir, chunk + 1)

        return self.history

    def save(self, checkpoint_dir: str | Path, step: int) -> None:
        save_checkpoint_step_multi(
            checkpoint_dir,
            step,
            {
                "player_1": (self.hyperparams_1, self.state_1.params),
                "player_2": (self.hyperparams_2, self.state_2.params),
            },
        )

    @classmethod
    def load(cls, checkpoint_dir: str | Path, step: int, game: ZeroSumGame) -> "MixtureSelfPlayPPOTrainer":
        """Rebuild a trainer at a checkpoint's params (optimizer state and rng restart)."""
        entries = load_checkpoint_step_multi(checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams)
        hyperparams_1, params_1 = entries["player_1"]
        hyperparams_2, params_2 = entries["player_2"]
        trainer = cls(game, hyperparams_1, hyperparams_2)
        trainer.state_1 = trainer.state_1.replace(params=params_1, target_params=params_1, magnet_params=params_1)
        trainer.state_2 = trainer.state_2.replace(params=params_2, target_params=params_2, magnet_params=params_2)
        return trainer
