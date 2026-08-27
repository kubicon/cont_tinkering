"""PPO on a `SequentialZeroSumGame`: the per-episode loss, and the self-play trainer.

Sampling lives in `training/sequential_rollout.py`; this module only scores what
came back and takes the gradient step. The split matters because the two halves
have different shapes of concern -- sampling is about walking a tree with traced
control flow, training is about masked reductions over the padding that walk
leaves behind.

Three differences from the one-shot loss in `training/mixture.py`, all of them
consequences of an episode being a *path* rather than a point:

  * **Masking, not slicing.** A trajectory holds both players' decisions
    interleaved, and its tail is padding once the episode ended. Player `p`'s
    loss weights each step by `live & (player == p)` -- there is no compaction,
    because which steps belong to whom varies per episode and a `vmap`ed batch
    cannot have ragged rows. Padding rows are still forward-passed; they are
    finite and multiplied by zero.
  * **Masked statistics.** Advantage normalization and every logged metric are
    means over the steps that survive the mask, not over the padded array. The
    normalizer is the batch-wide count of that player's live decisions, so the
    loss scale does not depend on how often a player happens to act.
  * **Monte-Carlo returns, no bootstrapping.** Payoffs land only at leaves, so
    the return at every decision is the episode's terminal payoff signed for the
    player who made it (`SequentialEpisode.reward`), and the advantage is still
    just `reward - value`. There is no GAE here and no need for one; adding
    intermediate rewards is what would call for it.

The unit of work is **one episode**: `sequential_ppo_loss` scores exactly one,
and `build_sequential_ppo_loss_fn` `vmap`s it over the batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from games.sequential import SequentialZeroSumGame

from .checkpoint import load_checkpoint_step_multi, save_checkpoint_step_multi
from .config import MixturePPOHyperparams
from .mixture import MixtureActorCritic, build_mixture_network, mixture_ppo_loss
from .mixture_trainer import (
    MixtureTrainState,
    _update_target_and_magnet,
    create_mixture_train_state,
)
from .ppo import ppo_update
from .sequential_rollout import (
    SequentialEpisode,
    build_episode_sampler,
    collect_sequential_batch,
)


def masked_mean(values: chex.Array, weight: chex.Array) -> chex.Array:
    """`sum(weight * values) / sum(weight)`, safe when nothing is selected."""
    return jnp.sum(weight * values) / jnp.maximum(jnp.sum(weight), 1.0)


def normalized_advantage(raw: chex.Array, weight: chex.Array) -> chex.Array:
    """Standardize `raw` using only the entries `weight` selects.

    Normalizing over the padded array instead would let the number of dead steps
    -- a property of how long episodes happened to run, not of the policy --
    shift and rescale every real advantage in the batch.
    """
    mean = masked_mean(raw, weight)
    variance = masked_mean(jnp.square(raw - mean), weight)
    return (raw - mean) / (jnp.sqrt(variance) + 1e-8)


def player_weight(episode: SequentialEpisode, player: int) -> chex.Array:
    """`1.0` on the steps where `player` really acted, `0.0` on everything else."""
    return (episode.live & (episode.player == player)).astype(jnp.float32)


def sequential_ppo_loss(
    params,
    network: MixtureActorCritic,
    episode: SequentialEpisode,
    advantage: chex.Array,
    player: int,
    clip_eps: float,
    value_coef: float,
    category_entropy_coef: float,
    gaussian_entropy_coef: float,
    trpo_category_kl_coef: float,
    trpo_gaussian_kl_coef: float,
    magnet_category_kl_coef: float,
    magnet_gaussian_kl_coef: float,
) -> tuple[chex.Array, dict[str, chex.Array], chex.Array]:
    """`player`'s PPO loss over a **single** episode, summed over their live steps.

    Each row of the trajectory is exactly the `Episode` that `mixture_ppo_loss`
    already scores -- atoms, legality masks and all -- so this adds only the two
    things that are new in a tree: `vmap` over the time axis, and the weighting
    that discards the other player's steps and the post-terminal padding.

    Returns `(loss_sum, metric_sums, count)` rather than means. The caller sums
    all three across the batch and divides once, which is the only way to get a
    true masked mean out of a `vmap` whose rows contain different numbers of
    live steps.
    """
    weight = player_weight(episode, player)

    per_step_loss, per_step_metrics = jax.vmap(
        mixture_ppo_loss, in_axes=(None, None, 0, 0, None, None, None, None, None, None, None, None)
    )(
        params, network, episode.to_transitions(), advantage, clip_eps, value_coef,
        category_entropy_coef, gaussian_entropy_coef,
        trpo_category_kl_coef, trpo_gaussian_kl_coef,
        magnet_category_kl_coef, magnet_gaussian_kl_coef,
    )

    loss_sum = jnp.sum(weight * per_step_loss)
    metric_sums = jax.tree_util.tree_map(lambda m: jnp.sum(weight * m), per_step_metrics)
    return loss_sum, metric_sums, jnp.sum(weight)


def build_sequential_ppo_loss_fn(
    player: int,
    category_entropy_coef: float,
    gaussian_entropy_coef: float,
    trpo_category_kl_coef: float,
    trpo_gaussian_kl_coef: float,
    magnet_category_kl_coef: float,
    magnet_gaussian_kl_coef: float,
):
    """Batch `sequential_ppo_loss` over a `SequentialEpisode`'s leading (env) axis.

    Binds `player` and the six per-head coefficients and returns a function
    matching `ppo_update`'s `loss_fn` contract, so the epoch scan and the
    optimizer step are shared with every other trainer here. `entropy_coef` (the
    generic `PPOHyperparams` field) is accepted and unused, exactly as in
    `build_mixture_ppo_loss_fn`.
    """

    def loss_fn(
        params,
        network: MixtureActorCritic,
        batch: SequentialEpisode,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        del entropy_coef

        weight = player_weight(batch, player)
        advantage = normalized_advantage(batch.reward - batch.value, weight)

        loss_sums, metric_sums, counts = jax.vmap(
            sequential_ppo_loss,
            in_axes=(None, None, 0, 0, None, None, None, None, None, None, None, None, None),
        )(
            params, network, batch, advantage, player, clip_eps, value_coef,
            category_entropy_coef, gaussian_entropy_coef,
            trpo_category_kl_coef, trpo_gaussian_kl_coef,
            magnet_category_kl_coef, magnet_gaussian_kl_coef,
        )

        total = jnp.maximum(jnp.sum(counts), 1.0)
        metrics = jax.tree_util.tree_map(lambda m: jnp.sum(m) / total, metric_sums)
        metrics["decisions_per_episode"] = jnp.mean(counts)
        return jnp.sum(loss_sums) / total, metrics

    return loss_fn


def _build_self_play_train_step(
    game: SequentialZeroSumGame,
    network_0: MixtureActorCritic,
    network_1: MixtureActorCritic,
    hyperparams_0: MixturePPOHyperparams,
    hyperparams_1: MixturePPOHyperparams,
):
    sample_episode = build_episode_sampler(game, network_0, network_1)
    loss_fns = tuple(
        build_sequential_ppo_loss_fn(
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
        batch = collect_sequential_batch(
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
            "payoff": jnp.mean(batch.payoff),
            "episode_length": jnp.mean(jnp.sum(batch.live.astype(jnp.float32), axis=-1)),
            **{f"{k}_0": v for k, v in metrics_0.items()},
            **{f"{k}_1": v for k, v in metrics_1.items()},
        }
        return state_0, state_1, metrics

    return step


class SequentialSelfPlayPPOTrainer:
    """Trains both players' `MixtureActorCritic`s on a sequential game, simultaneously.

    The counterpart of `MixtureSelfPlayPPOTrainer` for extensive-form games. The
    policies are unchanged -- the same mixture-with-atoms head, now reading an
    infoset observation instead of a constant -- and so are the target/magnet
    mechanics; what differs is that a rollout is a tree walk and the loss is
    masked. Both players learn from the *same* batch of trajectories, since one
    episode contains decisions belonging to both.

    Exploitability is not built in: best-responding in a tree with a continuous
    action is nothing like the projected-gradient ascent
    `ZeroSumGame.mixture_exploitability` performs, and how to do it depends on
    the shape of the game's tree. Pass a `metric_fn` to `train` to report one --
    `training.kuhn_evaluation.build_kuhn_metric_fn` supplies it for
    `ContinuousKuhnPoker`.
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

        Both hooks are called once per chunk with this trainer. `metric_fn`
        returns numbers, which are merged into that chunk's history record and
        printed; `strategy_log_fn` returns a block of text to print. They are
        separate because what counts as "the strategy" -- and what counts as
        exploitability -- depends on the game's tree, which this trainer knows
        nothing about.
        """
        if checkpoint_dir is not None:
            self.save(checkpoint_dir, 0)

        for chunk in range(steps):
            self.key, chunk_key = jax.random.split(self.key)
            step_keys = jax.random.split(chunk_key, epochs)
            (self.state_0, self.state_1), metrics_stack = self._run_chunk(
                (self.state_0, self.state_1), step_keys
            )

            for offset in range(epochs):
                iteration = chunk * epochs + offset + 1
                record = {"iteration": iteration, **{k: float(v[offset]) for k, v in metrics_stack.items()}}
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
