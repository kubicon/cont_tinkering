"""Pieces every mixture trainer repeats: train state, loss wiring, chunk logging.

`mixture_trainer` (one-shot games) and `sequential_trainer` (game trees) differ
only in how a batch is collected. Everything around that -- the target/magnet
copies carried alongside the trained params, the six-coefficient loss
construction, and the per-chunk metric bookkeeping -- was written out once per
trainer; it lives here instead, so `sequential_trainer` no longer has to reach
into `mixture_trainer` for private helpers.checkpointed.
"""

from __future__ import annotations

from typing import Callable

import chex
import jax
import jax.numpy as jnp
import optax

from .config import MixturePPOHyperparams, PPOHyperparams
from .mixture import MixtureActorCritic, build_mixture_ppo_loss_fn
from .ppo import TrainState, create_train_state


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


def update_target_and_magnet(
    state: MixtureTrainState, hyperparams: MixturePPOHyperparams
) -> MixtureTrainState:
    """Polyak-average `target_params` towards `params`; snapshot `magnet_params` every `L` steps."""
    magnet_step = state.magnet_step + 1
    return state.replace(
        target_params=optax.incremental_update(state.params, state.target_params, hyperparams.target_tau),
        magnet_params=optax.periodic_update(
            state.params, state.magnet_params, magnet_step, hyperparams.magnet_interval
        ),
        magnet_step=magnet_step,
    )


def build_loss_fn(player: int, hyperparams: MixturePPOHyperparams, shared_obs: bool = False):
    """`build_mixture_ppo_loss_fn` with the six per-head coefficients read off `hyperparams`.

    The coefficients always travel together and always come from the same
    dataclass; naming them one by one at five call sites only created five
    places to forget one.
    """
    return build_mixture_ppo_loss_fn(
        player,
        hyperparams.category_entropy_coef,
        hyperparams.gaussian_entropy_coef,
        hyperparams.trpo_category_kl_coef,
        hyperparams.trpo_gaussian_kl_coef,
        hyperparams.magnet_category_kl_coef,
        hyperparams.magnet_gaussian_kl_coef,
        shared_obs=shared_obs,
    )


def reject_batch_norm(name: str, hyperparams: PPOHyperparams) -> None:
    if hyperparams.normalization == "batch_norm":
        raise ValueError(f"{name}: batch_norm is not supported (see PPOTrainer for why).")


def append_chunk_records(
    history: list[dict], metrics_stack, chunk: int, epochs: int
) -> dict:
    """Append one chunk's per-iteration metrics to `history`, and return its last record.

    One device-to-host transfer for the whole chunk. Indexing the device arrays
    per iteration instead costs a dispatch and a sync *per metric per
    iteration*, which for a 300-iteration chunk takes several times longer than
    the training it is reporting on.

    The returned dict is the one now sitting at the end of `history`, not a
    copy: anything measured once per chunk (exploitability, a caller's
    `metric_fn`) attaches to that chunk's last iteration by mutating it.
    """
    metrics_chunk = jax.device_get(metrics_stack)
    records = [
        {
            "iteration": chunk * epochs + offset + 1,
            **{key: float(value[offset]) for key, value in metrics_chunk.items()},
        }
        for offset in range(epochs)
    ]
    history.extend(records)
    return records[-1]


def run_training_chunks(
    steps: int,
    epochs: int,
    key: chex.PRNGKey,
    states: chex.ArrayTree,
    run_chunk: Callable[[chex.ArrayTree, chex.Array], tuple[chex.ArrayTree, dict]],
    commit: Callable[[chex.ArrayTree], None],
    history: list[dict],
    format_record: Callable[[dict], str],
    metric_fn: Callable[[], dict[str, float]] | None = None,
    strategy_log_fn: Callable[[], str] | None = None,
    checkpoint_fn: Callable[[int], None] | None = None,
) -> chex.PRNGKey:
    """The outer loop shared by every mixture trainer: `steps` chunks of `epochs` iterations.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if checkpoint_fn is not None:
        checkpoint_fn(0)

    for chunk in range(steps):
        key, chunk_key = jax.random.split(key)
        states, metrics_stack = run_chunk(states, jax.random.split(chunk_key, epochs))
        commit(states)

        record = append_chunk_records(history, metrics_stack, chunk, epochs)
        # Evaluated once per chunk, on the parameters as they now stand, so it
        # attaches to that chunk's last record rather than to every iteration.
        extra = metric_fn() if metric_fn is not None else {}
        record.update(extra)

        print(format_record(record))
        if extra:
            print("  " + "  ".join(f"{k} {v:+.5f}" for k, v in extra.items()))
        if strategy_log_fn is not None:
            print(strategy_log_fn())
        if checkpoint_fn is not None:
            checkpoint_fn(chunk + 1)

    return key
