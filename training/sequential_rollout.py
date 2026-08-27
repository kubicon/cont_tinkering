"""Sampling trajectories from a `SequentialZeroSumGame`. Sampling only -- no losses.

A trajectory is a `training.mixture.Episode` with a `(max_steps, ...)` time axis
in front of each field: the very same record the one-shot rollouts in
`training.mixture` produce, one row per *decision* instead of one row per
episode. What makes that work is `Episode.actor` -- who owned the decision on
that row, `TERMINAL` on the padding steps a finished episode is carried through
-- which is all the loss needs to weight both players' interleaved decisions and
the padding correctly. There is no separate trajectory record and no conversion
step between sampling and the loss.

The episode's payoff to player 0 is returned *alongside* the `Episode` rather
than stored in it: it is one number for the whole trajectory, not a per-decision
field, and `Episode.reward` already carries it on every row, signed for whoever
acted there.
"""

from __future__ import annotations

from typing import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.sequential import SequentialZeroSumGame, select_by_player
from games.spaces import HybridAction

from .mixture import (
    Episode,
    MixtureActorCritic,
    component_to_kind,
    expand_kind_mask,
    sample_mixture_component,
)


def _validate_players_match(game: SequentialZeroSumGame, networks: tuple[MixtureActorCritic, ...]) -> None:
    """Both players' observations and categorical heads must be the same shape.
    """
    if game.obs_dim(0) != game.obs_dim(1):
        raise ValueError(
            f"both players need the same obs_dim for a batched rollout, got "
            f"{game.obs_dim(0)} and {game.obs_dim(1)}"
        )
    if game.num_kinds(0) != game.num_kinds(1):
        raise ValueError(
            f"both players need the same number of action kinds, got "
            f"{game.num_kinds(0)} and {game.num_kinds(1)}"
        )
    if networks[0].num_atoms != networks[1].num_atoms:
        raise ValueError(
            f"both networks need the same num_atoms, got "
            f"{networks[0].num_atoms} and {networks[1].num_atoms}"
        )
    if networks[0].num_components != networks[1].num_components:
        raise ValueError(
            f"both networks need the same num_components, got "
            f"{networks[0].num_components} and {networks[1].num_components}"
        )
    if networks[0].num_atoms != game.action_space(0).num_atoms:
        raise ValueError(
            f"network num_atoms ({networks[0].num_atoms}) must match the game's action space "
            f"({game.action_space(0).num_atoms})"
        )


def _same_box(space_0, space_1) -> bool:
    """Do both players bet into the identical continuous range? (A build-time check.)"""
    return bool(
        np.array_equal(np.asarray(space_0.box.low), np.asarray(space_1.box.low))
        and np.array_equal(np.asarray(space_0.box.high), np.asarray(space_1.box.high))
    )


def build_episode_sampler(
    game: SequentialZeroSumGame,
    network_0: MixtureActorCritic,
    network_1: MixtureActorCritic,
) -> Callable[..., tuple[Episode, chex.Array]]:
    """Bind the (static) game and networks; return a sampler for **one** episode.

    The returned function is `(params_0, magnet_params_0, params_1,
    magnet_params_1, key) -> (Episode, payoff)`, with a `(max_steps, ...)` time
    axis on every `Episode` field and a scalar `payoff` to player 0.
    """
    _validate_players_match(game, (network_0, network_1))

    networks = (network_0, network_1)
    spaces = (game.action_space(0), game.action_space(1))
    num_atoms = network_0.num_atoms
    num_components = network_0.num_components
    shared_box = _same_box(*spaces)

    def sample_episode(params_0, magnet_params_0, params_1, magnet_params_1, key: chex.PRNGKey):
        params = (params_0, params_1)
        magnet_params = (magnet_params_0, magnet_params_1)

        def step(state, step_key: chex.PRNGKey):
            sample_key, transition_key = jax.random.split(step_key)
            actor = game.current_player(state)  # `TERMINAL` once the episode is over

            def evaluate(index: int):
                obs = game.observation(index, state)
                mask = expand_kind_mask(game.action_mask(index, state), num_components)
                logits, means, log_stds, value = networks[index].apply(params[index], obs)
                magnet = networks[index].apply(magnet_params[index], obs)
                return obs, mask, logits, means, log_stds, value, magnet[:3]

            # Select the acting player's view. `actor` is traced, so this is a
            # select rather than a branch: both networks have already run.
            is_first = actor == 0
            obs, mask, logits, means, log_stds, value, magnet = select_by_player(
                is_first, evaluate(0), evaluate(1)
            )
            magnet_logits, magnet_means, magnet_log_stds = magnet

            component, raw_action = sample_mixture_component(
                logits, means, log_stds, mask, num_atoms, sample_key
            )
            action_kind = component_to_kind(component, num_atoms)
            # Only the continuous part is clipped here (the kind comes from the
            # masked categorical and is legal by construction). Each player's box
            # may differ, so unless they coincide, clip under both and select.
            if shared_box:
                action_value = spaces[0].box.clip(raw_action)
            else:
                action_value = select_by_player(
                    is_first, spaces[0].box.clip(raw_action), spaces[1].box.clip(raw_action)
                )

            next_state = game.step(
                state, HybridAction(kind=action_kind, value=action_value), transition_key
            )
            record = dict(
                actor=actor, obs=obs, action_mask=mask,
                logits=logits, means=means, log_stds=log_stds,
                magnet_logits=magnet_logits, magnet_means=magnet_means,
                magnet_log_stds=magnet_log_stds, component=component,
                raw_action=raw_action, action_kind=action_kind,
                action_value=action_value, value=value,
            )
            return next_state, record

        init_key, scan_key = jax.random.split(key)
        step_keys = jax.random.split(scan_key, game.max_steps)
        final_state, record = jax.lax.scan(step, game.initial_state(init_key), step_keys)

        # Terminal-only payoff, so every decision in the episode shares one
        # return: the leaf value, signed for whoever made that decision.
        payoff = game.payoff(final_state)
        reward = jnp.where(record["actor"] == 0, payoff, -payoff)
        return Episode(**record, reward=reward), payoff

    return sample_episode


def collect_sequential_batch(
    sample_episode: Callable[..., tuple[Episode, chex.Array]],
    params_0,
    magnet_params_0,
    params_1,
    magnet_params_1,
    key: chex.PRNGKey,
    num_envs: int,
) -> tuple[Episode, chex.Array]:
    """`num_envs` independent episodes: `sample_episode` `vmap`ed over rng keys.

    The env axis lands in front of the time axis, so the returned `Episode`'s
    fields are `(num_envs, max_steps, ...)` and `payoff` is `(num_envs,)`.
    """
    keys = jax.random.split(key, num_envs)
    return jax.vmap(sample_episode, in_axes=(None, None, None, None, 0))(
        params_0, magnet_params_0, params_1, magnet_params_1, keys
    )
