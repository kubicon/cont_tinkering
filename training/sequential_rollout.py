"""Sampling trajectories from a `SequentialZeroSumGame`. Sampling only -- no losses.

"""

from __future__ import annotations

from typing import Callable

import chex
import jax
import jax.numpy as jnp

from games.sequential import SequentialZeroSumGame
from games.spaces import HybridAction

from .mixture import (
    Episode,
    MixtureActorCritic,
    component_to_kind,
    expand_kind_mask,
    sample_mixture_component,
)


@chex.dataclass
class SequentialEpisode:
    """One trajectory. Every field carries a leading `max_steps` (time) axis -- except
    `payoff`, which is one number for the whole episode.
    """

    player: chex.Array  # (T,) int32, who acted -- `TERMINAL` on padding steps
    live: chex.Array  # (T,) bool, was this step a real decision
    obs: chex.Array  # (T, obs_dim) the acting player's infoset
    action_mask: chex.Array  # (T, num_atoms + num_components) legality, at sample time
    logits: chex.Array  # (T, num_atoms + num_components)
    means: chex.Array  # (T, num_components, action_dim)
    log_stds: chex.Array  # (T, num_components, action_dim)
    magnet_logits: chex.Array  # (T, num_atoms + num_components) under the magnet snapshot
    magnet_means: chex.Array  # (T, num_components, action_dim)
    magnet_log_stds: chex.Array  # (T, num_components, action_dim)
    component: chex.Array  # (T,) int32, the categorical entry drawn
    raw_action: chex.Array  # (T, action_dim) unclipped Gaussian draw
    action_kind: chex.Array  # (T,) int32, the `HybridAction.kind` played
    action_value: chex.Array  # (T, action_dim) the clipped continuous value played
    value: chex.Array  # (T,) the acting player's state-value estimate
    reward: chex.Array  # (T,) terminal payoff, signed for the acting player
    payoff: chex.Array  # () the episode's terminal payoff to *player 0* -- no time axis

    def to_transitions(self) -> Episode:
        """The same rows as a plain `Episode`, ready for `mixture_ppo_loss`.
        """
        return Episode(
            obs=self.obs,
            action_mask=self.action_mask,
            logits=self.logits,
            means=self.means,
            log_stds=self.log_stds,
            magnet_logits=self.magnet_logits,
            magnet_means=self.magnet_means,
            magnet_log_stds=self.magnet_log_stds,
            component=self.component,
            raw_action=self.raw_action,
            action_kind=self.action_kind,
            action_value=self.action_value,
            value=self.value,
            reward=self.reward,
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


def build_episode_sampler(
    game: SequentialZeroSumGame,
    network_0: MixtureActorCritic,
    network_1: MixtureActorCritic,
) -> Callable[..., SequentialEpisode]:
    """Bind the (static) game and networks; return a sampler for **one** episode.

    """
    _validate_players_match(game, (network_0, network_1))

    networks = (network_0, network_1)
    spaces = (game.action_space(0), game.action_space(1))
    num_atoms = network_0.num_atoms
    num_components = network_0.num_components

    def sample_episode(params_0, magnet_params_0, params_1, magnet_params_1, key: chex.PRNGKey):
        params = (params_0, params_1)
        magnet_params = (magnet_params_0, magnet_params_1)

        def step(state, step_key: chex.PRNGKey):
            sample_key, transition_key = jax.random.split(step_key)
            player = game.current_player(state)
            live = ~game.is_terminal(state)
            is_first = player == 0

            def evaluate(index: int):
                obs = game.observation(index, state)
                mask = expand_kind_mask(game.action_mask(index, state), num_components)
                logits, means, log_stds, value = networks[index].apply(params[index], obs)
                magnet = networks[index].apply(magnet_params[index], obs)
                return obs, mask, logits, means, log_stds, value, magnet[:3]

            evaluated = (evaluate(0), evaluate(1))
            # Select the acting player's view. `player` is traced, so this is a
            # select rather than a branch: both networks have already run.
            pick = lambda a, b: jnp.where(is_first, a, b)
            obs, mask, logits, means, log_stds, value, magnet = jax.tree_util.tree_map(
                pick, evaluated[0], evaluated[1]
            )
            magnet_logits, magnet_means, magnet_log_stds = magnet

            component, raw_action = sample_mixture_component(
                logits, means, log_stds, mask, num_atoms, sample_key
            )
            action_kind = component_to_kind(component, num_atoms)
            # Only the continuous part is clipped here (the kind comes from the
            # masked categorical and is legal by construction). Each player's box
            # may differ, so clip under both and select.
            action_value = pick(spaces[0].box.clip(raw_action), spaces[1].box.clip(raw_action))

            next_state = game.step(
                state, HybridAction(kind=action_kind, value=action_value), transition_key
            )
            record = dict(
                player=player, live=live, obs=obs, action_mask=mask,
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
        reward = jnp.where(record["player"] == 0, payoff, -payoff)
        return SequentialEpisode(**record, reward=reward, payoff=payoff)

    return sample_episode


def collect_sequential_batch(
    sample_episode: Callable[..., SequentialEpisode],
    params_0,
    magnet_params_0,
    params_1,
    magnet_params_1,
    key: chex.PRNGKey,
    num_envs: int,
) -> SequentialEpisode:
    """`num_envs` independent episodes: `sample_episode` `vmap`ed over rng keys.
    """
    keys = jax.random.split(key, num_envs)
    return jax.vmap(sample_episode, in_axes=(None, None, None, None, 0))(
        params_0, magnet_params_0, params_1, magnet_params_1, keys
    )
