"""Trainers for `MixtureActorCritic`: mirrors `trainer.py`/`self_play.py`, swapping
in the mixture rollout/loss functions from `training/mixture.py`.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

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
    build_mixture_ppo_loss_fn,
    collect_mixture_episode,
    collect_mixture_self_play_episode,
    sample_mixture_actions,
)
from .ppo import TrainState, create_train_state, ppo_update

# Monte-Carlo samples drawn from each mixture policy when estimating its exploitability.
EXPLOITABILITY_SAMPLES = 512


def _build_strategy_fn(network: MixtureActorCritic):
    """Jitted `(params, obs, mask) -> (probs, means, stds)` for `_strategy_str`.

    Logging runs once per chunk, but an eager `network.apply` there costs more
    than the whole chunk of training it reports on: every op dispatches
    separately and every `float()` on a device scalar is its own
    device-to-host sync. Jitting the forward pass and pulling the three arrays
    back in one `device_get` makes the whole thing a rounding error again.
    """

    @jax.jit
    def strategy_fn(params, obs: chex.Array, mask: chex.Array):
        logits, means, log_stds, _ = network.apply(params, obs)
        return jnp.exp(masked_log_softmax(logits, mask)), means, jnp.exp(log_stds)

    return strategy_fn


def _strategy_str(
    network: MixtureActorCritic,
    strategy_fn,
    params,
    obs: chex.Array,
    mask: chex.Array | None = None,
) -> str:
    """The full behavioral strategy at `obs`.

    """
    if mask is None:
        mask = np.ones(network.num_atoms + network.num_components, dtype=bool)
    probs, means, stds = jax.device_get(strategy_fn(params, obs, jnp.asarray(mask)))
    mask = np.asarray(mask)

    parts = [f"atom{i} {probs[i]:.2f}" for i in range(network.num_atoms) if mask[i]]
    parts += [
        f"{probs[network.num_atoms + k]:.2f}x"
        f"{tuple(round(float(x), 3) for x in means[k])}±{tuple(round(float(s), 3) for s in stds[k])}"
        for k in range(network.num_components)
        if mask[network.num_atoms + k]
    ]
    return ", ".join(parts)


class MixtureTrainState(TrainState):
    """`TrainState` plus two extra, non-trained copies of `params`.
    """

    target_params: chex.ArrayTree
    magnet_params: chex.ArrayTree
    magnet_step: chex.Array


def create_mixture_train_state(
    network: MixtureActorCritic, params, hyperparams: MixturePPOHyperparams
) -> MixtureTrainState:
    base = create_train_state(network, params, hyperparams)
    return MixtureTrainState(
        step=base.step,
        apply_fn=base.apply_fn,
        params=base.params,
        tx=base.tx,
        opt_state=base.opt_state,
        target_params=params,
        magnet_params=params,
        magnet_step=jnp.zeros((), dtype=jnp.int32),
    )


def _update_target_and_magnet(
    state: MixtureTrainState, hyperparams: MixturePPOHyperparams
) -> MixtureTrainState:
    magnet_step = state.magnet_step + 1
    return state.replace(
        target_params=optax.incremental_update(state.params, state.target_params, hyperparams.target_tau),
        magnet_params=optax.periodic_update(
            state.params, state.magnet_params, magnet_step, hyperparams.magnet_interval
        ),
        magnet_step=magnet_step,
    )


def _build_train_step(
    game: ZeroSumGame,
    network: MixtureActorCritic,
    opponent_action_fn: OpponentActionFn,
    hyperparams: MixturePPOHyperparams,
    perspective: int,
):
    loss_fn = build_mixture_ppo_loss_fn(
        perspective,
        hyperparams.category_entropy_coef, hyperparams.gaussian_entropy_coef,
        hyperparams.trpo_category_kl_coef, hyperparams.trpo_gaussian_kl_coef,
        hyperparams.magnet_category_kl_coef, hyperparams.magnet_gaussian_kl_coef,
        shared_obs=game.constant_observation,
    )

    def step(state: MixtureTrainState, key: jax.Array):
        batch = collect_mixture_episode(
            game, network, state.params, state.magnet_params, opponent_action_fn,
            key, hyperparams.num_envs, perspective,
        )
        state, metrics = ppo_update(state, network, batch, hyperparams, loss_fn=loss_fn)
        state = _update_target_and_magnet(state, hyperparams)
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
        if hyperparams.normalization == "batch_norm":
            raise ValueError("batch_norm is not supported (see PPOTrainer for why).")

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
        self._exploitability_fn = jax.jit(game.mixture_exploitability)
        self._strategy_fn = _build_strategy_fn(self.network)

        self.history: list[dict] = []

    def _exploitability(self, params, key: jax.Array) -> jax.Array:
        """Exploitability of `params`'s mixed strategy against the opponent's action distribution.

        Samples a batch of actions from the mixture policy (rather than
        collapsing it to its mode), so a genuinely multi-modal equilibrium --
        which no single action can represent -- reads as unexploitable.
        """
        obs = self.game.observation(self.perspective, jax.random.PRNGKey(0))
        sample_key, opponent_key, exploit_key = jax.random.split(key, 3)
        actions = sample_mixture_actions(
            self.network, params, obs, self.game.action_space(self.perspective),
            sample_key, EXPLOITABILITY_SAMPLES,
        )
        opponent_actions = self.opponent_action_fn(opponent_key, EXPLOITABILITY_SAMPLES)

        actions_1, actions_2 = (actions, opponent_actions) if self.perspective == 0 else (opponent_actions, actions)
        return self._exploitability_fn(actions_1, actions_2, exploit_key)

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

            # One device-to-host transfer for the whole chunk. Indexing the device
            # arrays per iteration instead costs a dispatch and a sync *per metric
            # per iteration*, which for a 300-iteration chunk takes several times
            # longer than the training it is reporting on.
            metrics_chunk = jax.device_get(metrics_stack)
            for offset in range(epochs):
                iteration = chunk * epochs + offset + 1
                record = {"iteration": iteration, **{k: float(v[offset]) for k, v in metrics_chunk.items()}}
                self.history.append(record)

            exploit_key, target_exploit_key = jax.random.split(exploit_key)
            record["exploitability"] = float(self._exploitability(self.state.params, exploit_key))
            record["exploitability_target"] = float(
                self._exploitability(self.state.target_params, target_exploit_key)
            )

            message = (
                f"iter {iteration:5d} | reward {record['mean_reward']:+.4f} "
                f"| policy_loss {record['policy_loss']:+.4f} | value_loss {record['value_loss']:.4f} "
                f"| entropy {record['entropy']:.4f} (category {record['category_entropy']:.4f}) "
                f"| approx_kl {record['approx_kl']:.4f} "
                f"| exploitability {record['exploitability']:.4f}"
                f" (target {record['exploitability_target']:.4f})"
            )
            print(message)
            obs = self.game.observation(self.perspective, jax.random.PRNGKey(0))
            opponent_sample = jnp.squeeze(self.opponent_action_fn(jax.random.PRNGKey(0), 1), axis=0)
            print(f"  player {self.perspective} strategy: {_strategy_str(self.network, self._strategy_fn, self.state.params, obs)}")
            print(
                f"  player {self.perspective} target strategy: "
                f"{_strategy_str(self.network, self._strategy_fn, self.state.target_params, obs)}"
            )
            print(f"  opponent sample: {tuple(round(float(x), 3) for x in opponent_sample)}")

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
        hyperparams, params = load_checkpoint_step(checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams)
        trainer = cls(game, hyperparams, opponent_action_fn, perspective=perspective)
        trainer.state = trainer.state.replace(params=params, target_params=params, magnet_params=params)
        return trainer


def _build_self_play_train_step(
    game: ZeroSumGame,
    network_1: MixtureActorCritic,
    network_2: MixtureActorCritic,
    hyperparams_1: MixturePPOHyperparams,
    hyperparams_2: MixturePPOHyperparams,
):
    loss_fn_1 = build_mixture_ppo_loss_fn(
        0,
        hyperparams_1.category_entropy_coef, hyperparams_1.gaussian_entropy_coef,
        hyperparams_1.trpo_category_kl_coef, hyperparams_1.trpo_gaussian_kl_coef,
        hyperparams_1.magnet_category_kl_coef, hyperparams_1.magnet_gaussian_kl_coef,
        shared_obs=game.constant_observation,
    )
    loss_fn_2 = build_mixture_ppo_loss_fn(
        1,
        hyperparams_2.category_entropy_coef, hyperparams_2.gaussian_entropy_coef,
        hyperparams_2.trpo_category_kl_coef, hyperparams_2.trpo_gaussian_kl_coef,
        hyperparams_2.magnet_category_kl_coef, hyperparams_2.magnet_gaussian_kl_coef,
        shared_obs=game.constant_observation,
    )

    def step(state_1: MixtureTrainState, state_2: MixtureTrainState, key: jax.Array):
        batch_1, batch_2 = collect_mixture_self_play_episode(
            game, network_1, state_1.params, state_1.magnet_params,
            network_2, state_2.params, state_2.magnet_params,
            key, hyperparams_1.num_envs,
        )
        state_1, metrics_1 = ppo_update(state_1, network_1, batch_1, hyperparams_1, loss_fn=loss_fn_1)
        state_2, metrics_2 = ppo_update(state_2, network_2, batch_2, hyperparams_2, loss_fn=loss_fn_2)
        state_1 = _update_target_and_magnet(state_1, hyperparams_1)
        state_2 = _update_target_and_magnet(state_2, hyperparams_2)
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
            if hyperparams.normalization == "batch_norm":
                raise ValueError(f"{name}: batch_norm is not supported (see PPOTrainer for why).")
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
            game, self.network_1, self.network_2, hyperparams_1, hyperparams_2
        )

        def scan_body(states, key):
            state_1, state_2 = states
            state_1, state_2, metrics = train_step(state_1, state_2, key)
            return (state_1, state_2), metrics

        self._run_chunk = jax.jit(lambda states, keys: jax.lax.scan(scan_body, states, keys))
        self._exploitability_fn = jax.jit(game.mixture_exploitability)
        self._strategy_fn_1 = _build_strategy_fn(self.network_1)
        self._strategy_fn_2 = _build_strategy_fn(self.network_2)

        self.history: list[dict] = []

    def _exploitability(self, params_1, params_2, key: jax.Array) -> jax.Array:
        """Exploitability of the two players' mixed strategies (0 at a Nash equilibrium).

        Each mixture is Monte-Carlo sampled rather than collapsed to its mode,
        so a multi-modal equilibrium (which no single action can represent)
        reads as unexploitable instead of maximally exploitable.
        """
        obs_1 = self.game.observation(0, jax.random.PRNGKey(0))
        obs_2 = self.game.observation(1, jax.random.PRNGKey(0))
        sample_key_1, sample_key_2, exploit_key = jax.random.split(key, 3)
        actions_1 = sample_mixture_actions(
            self.network_1, params_1, obs_1, self.game.action_space(0), sample_key_1, EXPLOITABILITY_SAMPLES
        )
        actions_2 = sample_mixture_actions(
            self.network_2, params_2, obs_2, self.game.action_space(1), sample_key_2, EXPLOITABILITY_SAMPLES
        )
        return self._exploitability_fn(actions_1, actions_2, exploit_key)

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

            # One device-to-host transfer for the whole chunk. Indexing the device
            # arrays per iteration instead costs a dispatch and a sync *per metric
            # per iteration*, which for a 300-iteration chunk takes several times
            # longer than the training it is reporting on.
            metrics_chunk = jax.device_get(metrics_stack)
            for offset in range(epochs):
                iteration = chunk * epochs + offset + 1
                record = {"iteration": iteration, **{k: float(v[offset]) for k, v in metrics_chunk.items()}}
                self.history.append(record)

            exploit_key, target_exploit_key = jax.random.split(exploit_key)
            record["exploitability"] = float(
                self._exploitability(self.state_1.params, self.state_2.params, exploit_key)
            )
            record["exploitability_target"] = float(
                self._exploitability(self.state_1.target_params, self.state_2.target_params, target_exploit_key)
            )

            message = (
                f"iter {iteration:5d} | reward {record['mean_reward_1']:+.4f} "
                f"| p1 policy_loss {record['policy_loss_1']:+.4f} value_loss {record['value_loss_1']:.4f} "
                f"category_entropy {record['category_entropy_1']:.4f} "
                f"| p2 policy_loss {record['policy_loss_2']:+.4f} value_loss {record['value_loss_2']:.4f} "
                f"category_entropy {record['category_entropy_2']:.4f} "
                f"| exploitability {record['exploitability']:.4f}"
                f" (target {record['exploitability_target']:.4f})"
            )
            print(message)
            obs_1 = self.game.observation(0, jax.random.PRNGKey(0))
            obs_2 = self.game.observation(1, jax.random.PRNGKey(0))
            print(f"  p1 strategy: {_strategy_str(self.network_1, self._strategy_fn_1, self.state_1.params, obs_1)}")
            print(f"  p2 strategy: {_strategy_str(self.network_2, self._strategy_fn_2, self.state_2.params, obs_2)}")
            print(f"  p1 target strategy: {_strategy_str(self.network_1, self._strategy_fn_1, self.state_1.target_params, obs_1)}")
            print(f"  p2 target strategy: {_strategy_str(self.network_2, self._strategy_fn_2, self.state_2.target_params, obs_2)}")

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
        entries = load_checkpoint_step_multi(checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams)
        hyperparams_1, params_1 = entries["player_1"]
        hyperparams_2, params_2 = entries["player_2"]
        trainer = cls(game, hyperparams_1, hyperparams_2)
        trainer.state_1 = trainer.state_1.replace(params=params_1, target_params=params_1, magnet_params=params_1)
        trainer.state_2 = trainer.state_2.replace(params=params_2, target_params=params_2, magnet_params=params_2)
        return trainer
