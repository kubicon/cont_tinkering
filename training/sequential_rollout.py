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
field. Per row, `Episode.step_reward` holds player 0's reward for that row's
transition (the game's optional dense `reward`, plus the terminal payoff on the
last decision), and `Episode.reward` the Monte Carlo return from that row on,
signed for whoever acted there -- with a terminal-only game, the leaf value on
every row. `training.vtrace` bootstraps from `step_reward` instead.
"""

from __future__ import annotations

from typing import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.sequential import TERMINAL, SequentialZeroSumGame, select_by_player
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
    explore_eps: tuple[float, float] = (0.0, 0.0),
) -> Callable[..., tuple[Episode, chex.Array]]:
    """Bind the (static) game and networks; return a sampler for **one** episode.

    The returned function is `(params_0, magnet_params_0, params_1,
    magnet_params_1, key) -> (Episode, payoff)`, with a `(max_steps, ...)` time
    axis on every `Episode` field and a scalar `payoff` to player 0.

    `explore_eps[p]` makes player `p` act from its exploring behavior policy:
    each continuous action it draws is, with that probability, replaced by a
    uniform draw on its network's box (see `sample_mixture_component`). Any
    nonzero entry records `Episode.behavior_eps`, which switches the loss to its
    importance-weighted form for that player's own decisions. The default
    `(0.0, 0.0)` samples the policies themselves -- what every best response,
    evaluator and baseline wants -- and leaves the rollout exactly as before.
    """
    _validate_players_match(game, (network_0, network_1))

    networks = (network_0, network_1)
    spaces = (game.action_space(0), game.action_space(1))
    num_atoms = network_0.num_atoms
    num_components = network_0.num_components
    shared_box = _same_box(*spaces)
    explore_eps = tuple(float(eps) for eps in explore_eps)
    for player, eps in enumerate(explore_eps):
        if not 0.0 <= eps < 1.0:
            raise ValueError(f"explore_eps[{player}] must lie in [0, 1), got {eps}")
        if eps > 0.0 and not np.all(np.asarray(networks[player].high) > np.asarray(networks[player].low)):
            raise ValueError(
                f"explore_eps[{player}] > 0 needs a box of positive width to draw from, got "
                f"low={networks[player].low}, high={networks[player].high}"
            )
    exploring = any(eps > 0.0 for eps in explore_eps)

    def sample_episode(params_0, magnet_params_0, params_1, magnet_params_1, key: chex.PRNGKey):
        params = (params_0, params_1)
        magnet_params = (magnet_params_0, magnet_params_1)

        def step(state, step_key: chex.PRNGKey):
            sample_key, transition_key = jax.random.split(step_key)
            actor = game.current_player(state)  # `TERMINAL` once the episode is over

            def evaluate(index: int):
                obs = game.observation(index, state)
                mask = expand_kind_mask(game.action_mask(index, state), num_components)
                logits, means, scale_trils, value = networks[index].apply(params[index], obs)
                magnet = networks[index].apply(magnet_params[index], obs)
                return obs, mask, logits, means, scale_trils, value, magnet[:3]

            # Select the acting player's view. `actor` is traced, so this is a
            # select rather than a branch: both networks have already run.
            is_first = actor == 0
            obs, mask, logits, means, scale_trils, value, magnet = select_by_player(
                is_first, evaluate(0), evaluate(1)
            )
            magnet_logits, magnet_means, magnet_scale_trils = magnet

            if exploring:
                # The acting player's rate and box; a padding step's choice is
                # irrelevant, since the loss weights it to zero.
                eps = jnp.where(is_first, explore_eps[0], explore_eps[1])
                low, high = select_by_player(
                    is_first, (networks[0].low, networks[0].high), (networks[1].low, networks[1].high)
                )
                component, raw_action = sample_mixture_component(
                    logits, means, scale_trils, mask, num_atoms, sample_key,
                    explore_eps=eps, low=low, high=high,
                )
            else:
                component, raw_action = sample_mixture_component(
                    logits, means, scale_trils, mask, num_atoms, sample_key
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

            action = HybridAction(kind=action_kind, value=action_value)
            next_state = game.step(state, action, transition_key)
            step_reward = jnp.where(
                actor == TERMINAL, 0.0, game.reward(state, action, next_state)
            ).astype(jnp.float32)
            record = dict(
                actor=actor, obs=obs, action_mask=mask,
                logits=logits, means=means, scale_trils=scale_trils,
                magnet_logits=magnet_logits, magnet_means=magnet_means,
                magnet_scale_trils=magnet_scale_trils, component=component,
                raw_action=raw_action, action_kind=action_kind,
                action_value=action_value, value=value, step_reward=step_reward,
            )
            if exploring:
                record["behavior_eps"] = eps
            return next_state, record

        init_key, scan_key = jax.random.split(key)
        step_keys = jax.random.split(scan_key, game.max_steps)
        final_state, record = jax.lax.scan(step, game.initial_state(init_key), step_keys)

        # The terminal payoff is credited to the last real decision: the row
        # whose transition ended the episode. (Decisions are a prefix of the
        # scan -- once terminal, a state stays terminal.)
        # `payoff` is the episode's total to player 0: dense rewards plus leaf.
        terminal_payoff = game.payoff(final_state)
        payoff = jnp.sum(record["step_reward"]) + terminal_payoff
        num_decisions = jnp.sum(record["actor"] != TERMINAL)
        last_decision = jax.nn.one_hot(num_decisions - 1, game.max_steps, dtype=jnp.float32)
        record["step_reward"] = record["step_reward"] + last_decision * terminal_payoff

        # The Monte Carlo return of each decision: player 0's rewards-to-go from
        # that row on, signed for whoever made it. With no per-step rewards this
        # is the one leaf value, shared by every decision in the episode.
        to_go = jnp.cumsum(record["step_reward"][::-1])[::-1]
        reward = jnp.where(record["actor"] == 0, to_go, -to_go)
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
    param_axes: tuple = (None, None, None, None),
) -> tuple[Episode, chex.Array]:
    """`num_envs` independent episodes: `sample_episode` `vmap`ed over rng keys.

    The env axis lands in front of the time axis, so the returned `Episode`'s
    fields are `(num_envs, max_steps, ...)` and `payoff` is `(num_envs,)`.

    `param_axes` gives the `in_axes` of the four parameter arguments, in order.
    The default maps none of them: one strategy per player, shared by the whole
    batch, which is what self-play and a single frozen opponent need. Passing
    `0` for a player's two entries instead makes that player's parameters a
    *stacked* pytree with a leading `num_envs` axis -- one strategy per episode,
    which is how a mixture over policies is played: the member is drawn at the
    start of the episode and held for all of it (a mixed strategy), rather than
    re-drawn at every decision (which would be a different, behavioral, object).
    See `baselines.neural.sequential_oracle`.
    """
    keys = jax.random.split(key, num_envs)
    return jax.vmap(sample_episode, in_axes=(*param_axes, 0))(
        params_0, magnet_params_0, params_1, magnet_params_1, keys
    )
