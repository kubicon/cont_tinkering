"""On-policy episode collection for a `ZeroSumGame`.

The games in `games/` are one-shot: there's no state to reset or step
through, both players act simultaneously once, and the payoff is the whole
episode. So "sampling an episode" here means: draw each side's observation
from the game itself (`game.observation(player, ...)` -- a constant by
default, since the game is stateless, but the game owns that decision, not
the trainer), sample each side's action from its current Gaussian policy
conditioned on that observation, and score the payoff. Everything is
batched over `num_envs` independent one-shot episodes at once.

Two variants: `collect_episode` samples one side from its policy against an
`opponent_action_fn` that isn't being trained (fixed action, random, or a
frozen snapshot); `collect_self_play_episode` samples both sides from their
own current, live policies for simultaneous self-play training.
"""

from __future__ import annotations

from typing import Callable

import chex
import jax
import jax.numpy as jnp

from games.base import ZeroSumGame

from .actor_critic import ActorCritic, gaussian_log_prob

OpponentActionFn = Callable[[chex.PRNGKey, int], chex.Array]


@chex.dataclass
class Transition:
    obs: chex.Array
    raw_action: chex.Array  # unclipped action sampled from the Gaussian; used to train the policy
    action: chex.Array  # `raw_action` clipped to the action space; used to play the game
    log_prob: chex.Array
    value: chex.Array
    reward: chex.Array


def _sample_batch(
    network: ActorCritic, params, obs: chex.Array, key: chex.PRNGKey, space
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Sample `(raw_action, action, log_prob, value)` for a batch of `obs`, one env per row."""
    mean, log_std, value = jax.vmap(lambda o: network.apply(params, o))(obs)
    noise = jax.random.normal(key, mean.shape)
    raw_action = mean + jnp.exp(log_std) * noise
    log_prob = gaussian_log_prob(raw_action, mean, log_std)
    action = jax.vmap(space.clip)(raw_action)
    return raw_action, action, log_prob, value


def collect_episode(
    game: ZeroSumGame,
    network: ActorCritic,
    params,
    opponent_action_fn: OpponentActionFn,
    key: chex.PRNGKey,
    num_envs: int,
    perspective: int = 0,
) -> Transition:
    """Collect `num_envs` one-shot episodes played from `perspective`'s (0 or 1) point of view.

    The opponent's action comes from `opponent_action_fn`, e.g. a fixed
    action, a random sampler, or another (frozen) policy — it is not
    updated. For training both players against each other simultaneously,
    use `collect_self_play_episode` instead.
    """
    if perspective not in (0, 1):
        raise ValueError(f"perspective must be 0 or 1, got {perspective}")

    obs_key, action_key, opponent_key = jax.random.split(key, 3)
    own_space = game.action_space(perspective)
    obs = game.observation(perspective, obs_key, (num_envs,))
    raw_action, action, log_prob, value = _sample_batch(network, params, obs, action_key, own_space)

    opponent_action = opponent_action_fn(opponent_key, num_envs)
    if perspective == 0:
        reward = game.payoff_batch(action, opponent_action)
    else:
        reward = -game.payoff_batch(opponent_action, action)

    return Transition(
        obs=obs, raw_action=raw_action, action=action, log_prob=log_prob, value=value, reward=reward
    )


def collect_self_play_episode(
    game: ZeroSumGame,
    network_1: ActorCritic,
    params_1,
    network_2: ActorCritic,
    params_2,
    key: chex.PRNGKey,
    num_envs: int,
) -> tuple[Transition, Transition]:
    """Collect `num_envs` one-shot episodes with both players' *current* policies acting.

    Unlike `collect_episode`, neither side is frozen: both actions are drawn
    fresh from `params_1`/`params_2` each call, so the two `Transition`
    batches returned are what each player's own PPO update should train on
    (player 2's reward is `-payoff`, since the game is zero-sum).
    """
    obs_key_1, obs_key_2, key_1, key_2 = jax.random.split(key, 4)
    obs_1 = game.observation(0, obs_key_1, (num_envs,))
    obs_2 = game.observation(1, obs_key_2, (num_envs,))

    raw_action_1, action_1, log_prob_1, value_1 = _sample_batch(
        network_1, params_1, obs_1, key_1, game.action_space(0)
    )
    raw_action_2, action_2, log_prob_2, value_2 = _sample_batch(
        network_2, params_2, obs_2, key_2, game.action_space(1)
    )

    reward = game.payoff_batch(action_1, action_2)

    batch_1 = Transition(
        obs=obs_1, raw_action=raw_action_1, action=action_1, log_prob=log_prob_1, value=value_1, reward=reward
    )
    batch_2 = Transition(
        obs=obs_2, raw_action=raw_action_2, action=action_2, log_prob=log_prob_2, value=value_2, reward=-reward
    )
    return batch_1, batch_2
