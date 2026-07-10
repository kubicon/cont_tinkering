"""PPO loss and the parameter update built from it.

Because episodes here are one-shot (a single simultaneous move, no
bootstrapping), the usual GAE recursion collapses to the trivial case:
the return is just the payoff received, and the advantage is
`reward - value` using the value estimate recorded at rollout time. The
clipped surrogate objective and value/entropy terms are otherwise the
standard PPO recipe.
"""

from __future__ import annotations

from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from .actor_critic import ActorCritic, gaussian_entropy, gaussian_log_prob
from .config import PPOHyperparams
from .optimizers import build_optimizer
from .rollout import Transition

TrainState = train_state.TrainState
LossFn = Callable[..., tuple[chex.Array, dict[str, chex.Array]]]


def create_train_state(network: nn.Module, params, hyperparams: PPOHyperparams) -> TrainState:
    tx = optax.chain(
        optax.clip_by_global_norm(hyperparams.max_grad_norm),
        build_optimizer(hyperparams.optimizer, hyperparams.learning_rate, hyperparams.weight_decay),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def ppo_loss(
    params,
    network: ActorCritic,
    batch: Transition,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    mean, log_std, value_pred = jax.vmap(lambda o: network.apply(params, o))(batch.obs)
    new_log_prob = gaussian_log_prob(batch.raw_action, mean, log_std)
    ratio = jnp.exp(new_log_prob - batch.log_prob)

    advantage = batch.reward - batch.value
    advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

    surrogate_1 = ratio * advantage
    surrogate_2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy_loss = -jnp.mean(jnp.minimum(surrogate_1, surrogate_2))

    value_loss = jnp.mean(jnp.square(value_pred - batch.reward))
    entropy = gaussian_entropy(log_std)

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": jnp.mean(batch.log_prob - new_log_prob),
        "clip_frac": jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)),
    }
    return loss, metrics


def ppo_update(
    state: TrainState,
    network: nn.Module,
    batch,
    hyperparams: PPOHyperparams,
    loss_fn: LossFn = ppo_loss,
) -> tuple[TrainState, dict[str, chex.Array]]:
    """Run `num_epochs` gradient steps on the *whole* batch (no minibatching/shuffling).

    `loss_fn` defaults to the diagonal-Gaussian `ppo_loss`; pass
    `mixture_ppo_loss` (see `training/mixture.py`) to train a
    `MixtureActorCritic` instead — the epoch scanning here is identical for
    either kind of actor.
    """
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def epoch_step(state: TrainState, _):
        (_, metrics), grads = grad_fn(
            state.params,
            network,
            batch,
            hyperparams.clip_eps,
            hyperparams.value_coef,
            hyperparams.entropy_coef,
        )
        state = state.apply_gradients(grads=grads)
        return state, metrics

    state, metrics = jax.lax.scan(epoch_step, state, xs=None, length=hyperparams.num_epochs)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return state, metrics
