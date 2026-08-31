"""Trainers for `ExpFamilyActorCritic`: `mixture_trainer.py` with the log-linear policy.

Same two classes, same chunked loop, same checkpoint layout and the same
sampled exploitability readout -- only the policy underneath differs, which is
the point: a run of `wo_magnet_expfam_ppo.yaml` differs from
`wo_magnet_ppo.yaml` in the parametrization and in nothing else.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.base import ZeroSumGame

from .checkpoint import (
    load_checkpoint_step,
    load_checkpoint_step_multi,
    save_checkpoint_step,
    save_checkpoint_step_multi,
    target_entry,
)
from .config import ExpFamilyPPOHyperparams
from .expfam import (
    ExpFamilyActorCritic,
    OpponentActionFn,
    build_expfam_network,
    build_expfam_ppo_loss_fn,
    collect_expfam_episode,
    collect_expfam_self_play_episode,
    density_entropy,
    density_moments,
    grid_log_probs,
    sample_expfam_actions,
)
from .ppo import ppo_update
from .trainer_common import (
    MixtureTrainState,
    append_chunk_records,
    create_mixture_train_state,
    reject_batch_norm,
    update_target_and_magnet,
)

__all__ = ["ExpFamilyPPOTrainer", "ExpFamilySelfPlayPPOTrainer"]

EXPLOITABILITY_SAMPLES = 512

# See `mixture_trainer._OBSERVATION_KEY`: a one-shot observation is a constant,
# so fixing the key keeps every chunk's readout comparable.
_OBSERVATION_KEY = jax.random.PRNGKey(0)

# Coarse bins the logged density profile is collapsed onto. The policy has
# `grid_points` of them, which is far too many to read; this is enough to see
# where the mass is and whether it is multi-modal.
_PROFILE_BINS = 10


def _build_loss_fn(hyperparams: ExpFamilyPPOHyperparams, shared_obs: bool):
    return build_expfam_ppo_loss_fn(
        hyperparams.density_entropy_coef,
        hyperparams.trpo_density_kl_coef,
        hyperparams.magnet_density_kl_coef,
        shared_obs=shared_obs,
    )


def _build_strategy_str(network: ExpFamilyActorCritic):
    """`(params, obs) -> str`: mean +/- std per dimension, plus a coarse density profile.

    A Gaussian mixture is summarized by listing its components; a log-linear
    density has none, so what is printed instead is the shape of the density
    itself -- `_PROFILE_BINS` bucket masses across the box. That is strictly
    more informative for the question these runs ask (where is the mass?) and it
    is the same readout whatever the basis size.

    Jitted, and pulled back in one `device_get`, for the reason
    `mixture_trainer._build_strategy_str` gives: an eager forward pass per chunk
    costs more than the chunk of training it reports on.
    """
    basis = network.basis
    # Every grid bin lands in exactly one bucket, whether or not `_PROFILE_BINS`
    # divides `grid_points`: reshaping instead would silently drop the remainder,
    # and the remainder is at the top of the box -- precisely where a policy that
    # has collapsed onto the boundary keeps its mass, so the printed profile
    # would stop summing to 1 exactly when it matters most.
    bucket = (jnp.arange(basis.grid_points) * _PROFILE_BINS) // basis.grid_points
    buckets = jax.nn.one_hot(bucket, _PROFILE_BINS)  # (G, _PROFILE_BINS)

    @jax.jit
    def forward(params, obs: chex.Array):
        theta, _ = network.apply(params, obs)
        probs = jnp.exp(grid_log_probs(theta, basis))  # (dim, G)
        profile = probs @ buckets
        mean, std = density_moments(theta, basis)
        return mean, std, profile, density_entropy(theta, basis)

    def strategy_str(params, obs: chex.Array) -> str:
        mean, std, profile, entropy = jax.device_get(forward(params, obs))
        parts = []
        for d in range(np.shape(mean)[0]):
            bars = " ".join(f"{p:.2f}" for p in profile[d])
            parts.append(f"dim{d} mean {mean[d]:+.3f} ±{std[d]:.3f} | [{bars}]")
        return f"H {float(entropy):+.3f}  " + "  ".join(parts)

    return strategy_str


def _build_train_step(
    game: ZeroSumGame,
    network: ExpFamilyActorCritic,
    opponent_action_fn: OpponentActionFn,
    hyperparams: ExpFamilyPPOHyperparams,
    perspective: int,
):
    loss_fn = _build_loss_fn(hyperparams, shared_obs=game.constant_observation)

    def step(state: MixtureTrainState, key: jax.Array):
        batch = collect_expfam_episode(
            game, network, state.params, state.magnet_params, opponent_action_fn,
            key, hyperparams.num_envs, perspective,
        )
        state, metrics = ppo_update(state, network, batch, hyperparams, loss_fn=loss_fn)
        state = update_target_and_magnet(state, hyperparams)
        metrics = {**metrics, "mean_reward": jnp.mean(batch.reward)}
        return state, metrics

    return step


class ExpFamilyPPOTrainer:
    """Trains one player's `ExpFamilyActorCritic` against a (possibly fixed) opponent."""

    def __init__(
        self,
        game: ZeroSumGame,
        hyperparams: ExpFamilyPPOHyperparams,
        opponent_action_fn: OpponentActionFn,
        perspective: int = 0,
        seed: int = 0,
    ):
        reject_batch_norm("hyperparams", hyperparams)

        self.game = game
        self.hyperparams = hyperparams
        self.opponent_action_fn = opponent_action_fn
        self.perspective = perspective

        self.network = build_expfam_network(hyperparams)

        key = jax.random.PRNGKey(seed)
        init_key, obs_key, self.key = jax.random.split(key, 3)
        params = self.network.init(init_key, game.observation(perspective, obs_key))
        self.state = create_mixture_train_state(self.network, params, hyperparams)

        train_step = _build_train_step(game, self.network, opponent_action_fn, hyperparams, perspective)
        self._run_chunk = jax.jit(lambda state, keys: jax.lax.scan(train_step, state, keys))
        self._strategy_str = _build_strategy_str(self.network)

        self._obs = game.observation(perspective, _OBSERVATION_KEY)
        self._exploitability = self._build_exploitability_fn()
        self.history: list[dict] = []

    def _build_exploitability_fn(self):
        game, network, perspective = self.game, self.network, self.perspective
        opponent_action_fn = self.opponent_action_fn
        obs = self._obs

        def exploitability(params, key: jax.Array) -> jax.Array:
            sample_key, opponent_key, exploit_key = jax.random.split(key, 3)
            actions = sample_expfam_actions(network, params, obs, sample_key, EXPLOITABILITY_SAMPLES)
            opponent_actions = opponent_action_fn(opponent_key, EXPLOITABILITY_SAMPLES)
            if perspective == 0:
                return game.mixture_exploitability(actions, opponent_actions, exploit_key)
            return game.mixture_exploitability(opponent_actions, actions, exploit_key)

        @jax.jit
        def both(params, target_params, key: jax.Array):
            live_key, target_key = jax.random.split(key)
            return exploitability(params, live_key), exploitability(target_params, target_key)

        return both

    def train(self, steps: int, epochs: int = 10, checkpoint_dir: str | Path | None = None) -> list[dict]:
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        if checkpoint_dir is not None:
            self.save(checkpoint_dir, 0)

        for chunk in range(steps):
            self.key, chunk_key, exploit_key = jax.random.split(self.key, 3)
            self.state, metrics_stack = self._run_chunk(self.state, jax.random.split(chunk_key, epochs))

            record = append_chunk_records(self.history, metrics_stack, chunk, epochs)
            live, target = jax.device_get(
                self._exploitability(self.state.params, self.state.target_params, exploit_key)
            )
            record["exploitability"] = float(live)
            record["exploitability_target"] = float(target)

            print(
                f"iter {record['iteration']:5d} | reward {record['mean_reward']:+.4f} "
                f"| policy_loss {record['policy_loss']:+.4f} | value_loss {record['value_loss']:.4f} "
                f"| entropy {record['entropy']:+.4f} | magnet_kl {record['magnet_kl']:.4f} "
                f"| approx_kl {record['approx_kl']:+.4f} "
                f"| exploitability {record['exploitability']:.4f}"
                f" (target {record['exploitability_target']:.4f})"
            )
            print(f"  player {self.perspective}: {self._strategy_str(self.state.params, self._obs)}")
            print(f"  player {self.perspective} target: {self._strategy_str(self.state.target_params, self._obs)}")

            if checkpoint_dir is not None:
                self.save(checkpoint_dir, chunk + 1)

        return self.history

    def save(self, checkpoint_dir: str | Path, step: int) -> None:
        save_checkpoint_step(
            checkpoint_dir, step, self.hyperparams, self.state.params, self.state.target_params
        )

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        step: int,
        game: ZeroSumGame,
        opponent_action_fn: OpponentActionFn,
        perspective: int = 0,
    ) -> "ExpFamilyPPOTrainer":
        """Rebuild a trainer at a checkpoint's params (optimizer state and rng restart)."""
        hyperparams, params = load_checkpoint_step(
            checkpoint_dir, step, hyperparams_cls=ExpFamilyPPOHyperparams
        )
        try:
            _, target_params = load_checkpoint_step(
                checkpoint_dir, step, hyperparams_cls=ExpFamilyPPOHyperparams, target=True
            )
        except KeyError:
            print(f"No target params found for step {step}, using live params", flush=True)
            target_params = params
        trainer = cls(game, hyperparams, opponent_action_fn, perspective=perspective)
        trainer.state = trainer.state.replace(
            params=params, target_params=target_params, magnet_params=params
        )
        return trainer


def _build_self_play_train_step(
    game: ZeroSumGame,
    networks: tuple[ExpFamilyActorCritic, ExpFamilyActorCritic],
    hyperparams: tuple[ExpFamilyPPOHyperparams, ExpFamilyPPOHyperparams],
):
    loss_fns = tuple(
        _build_loss_fn(hyperparams[player], shared_obs=game.constant_observation) for player in (0, 1)
    )

    def step(state_1: MixtureTrainState, state_2: MixtureTrainState, key: jax.Array):
        batch_1, batch_2 = collect_expfam_self_play_episode(
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


class ExpFamilySelfPlayPPOTrainer:
    """Trains both players' `ExpFamilyActorCritic`s against each other simultaneously."""

    def __init__(
        self,
        game: ZeroSumGame,
        hyperparams_1: ExpFamilyPPOHyperparams,
        hyperparams_2: ExpFamilyPPOHyperparams,
        seed: int = 0,
    ):
        for name, hyperparams in (("hyperparams_1", hyperparams_1), ("hyperparams_2", hyperparams_2)):
            reject_batch_norm(name, hyperparams)
        if hyperparams_1.num_envs != hyperparams_2.num_envs:
            raise ValueError("hyperparams_1.num_envs must equal hyperparams_2.num_envs for self-play rollouts")

        self.game = game
        self.hyperparams_1 = hyperparams_1
        self.hyperparams_2 = hyperparams_2

        self.network_1 = build_expfam_network(hyperparams_1)
        self.network_2 = build_expfam_network(hyperparams_2)

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
        game = self.game
        networks = (self.network_1, self.network_2)
        observations = (self._obs_1, self._obs_2)

        def exploitability(params: tuple, key: jax.Array) -> jax.Array:
            sample_key_1, sample_key_2, exploit_key = jax.random.split(key, 3)
            actions = tuple(
                sample_expfam_actions(
                    networks[i], params[i], observations[i], sample_key, EXPLOITABILITY_SAMPLES
                )
                for i, sample_key in enumerate((sample_key_1, sample_key_2))
            )
            return game.mixture_exploitability(actions[0], actions[1], exploit_key)

        @jax.jit
        def both(params: tuple, target_params: tuple, key: jax.Array):
            live_key, target_key = jax.random.split(key)
            return exploitability(params, live_key), exploitability(target_params, target_key)

        return both

    def train(self, steps: int, epochs: int = 10, checkpoint_dir: str | Path | None = None) -> list[dict]:
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        if checkpoint_dir is not None:
            self.save(checkpoint_dir, 0)

        for chunk in range(steps):
            self.key, chunk_key, exploit_key = jax.random.split(self.key, 3)
            (self.state_1, self.state_2), metrics_stack = self._run_chunk(
                (self.state_1, self.state_2), jax.random.split(chunk_key, epochs)
            )

            record = append_chunk_records(self.history, metrics_stack, chunk, epochs)
            live, target = jax.device_get(
                self._exploitability(self.params, self.target_params, exploit_key)
            )
            record["exploitability"] = float(live)
            record["exploitability_target"] = float(target)

            print(
                f"iter {record['iteration']:5d} | reward {record['mean_reward_1']:+.4f} "
                f"| p1 entropy {record['entropy_1']:+.4f} magnet_kl {record['magnet_kl_1']:.4f} "
                f"| p2 entropy {record['entropy_2']:+.4f} magnet_kl {record['magnet_kl_2']:.4f} "
                f"| exploitability {record['exploitability']:.4f}"
                f" (target {record['exploitability_target']:.4f})"
            )
            print(f"  p1: {self._strategy_str_1(self.state_1.params, self._obs_1)}")
            print(f"  p2: {self._strategy_str_2(self.state_2.params, self._obs_2)}")

            if checkpoint_dir is not None:
                self.save(checkpoint_dir, chunk + 1)

        return self.history

    def save(self, checkpoint_dir: str | Path, step: int) -> None:
        players = ((1, self.hyperparams_1, self.state_1), (2, self.hyperparams_2, self.state_2))
        save_checkpoint_step_multi(
            checkpoint_dir,
            step,
            {
                **{f"player_{i}": (hp, state.params) for i, hp, state in players},
                **{target_entry(f"player_{i}"): (hp, state.target_params) for i, hp, state in players},
            },
        )

    @classmethod
    def load(
        cls, checkpoint_dir: str | Path, step: int, game: ZeroSumGame
    ) -> "ExpFamilySelfPlayPPOTrainer":
        entries = load_checkpoint_step_multi(
            checkpoint_dir, step, hyperparams_cls=ExpFamilyPPOHyperparams
        )
        hyperparams_1, params_1 = entries["player_1"]
        hyperparams_2, params_2 = entries["player_2"]
        trainer = cls(game, hyperparams_1, hyperparams_2)
        for index, state_name, params in ((1, "state_1", params_1), (2, "state_2", params_2)):
            name = target_entry(f"player_{index}")
            target_params = entries[name][1] if name in entries else params
            state = getattr(trainer, state_name)
            setattr(
                trainer,
                state_name,
                state.replace(params=params, target_params=target_params, magnet_params=params),
            )
        return trainer
