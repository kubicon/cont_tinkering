"""Reading an explicit `KuhnStrategy` out of a trained mixture policy.

`games.kuhn_best_response` does the game arithmetic on plain probability tables
and knows nothing about networks. This module is the bridge: it queries a
`MixtureActorCritic` at every Kuhn infoset and turns its mixture-with-atoms head
into the tables that module expects.

The one genuinely delicate step is the bet-size distribution. A rollout plays
`box.clip(raw_action)`, so the distribution the *game* sees is not the Gaussian
mixture the policy parameterizes: it is that mixture pushed through a clip,
which puts point masses on `min_bet` and `max_bet`. Those atoms are not a
rounding detail -- early in training, when component means sit outside the box,
they carry most of the mass. `clipped_mixture_grid_probs` handles them exactly
and, pleasingly, for free.

Note also that `mixture_log_probs` scores the *unclipped* `raw_action` -- the
usual continuous-control convention -- so the density the policy trains against
is deliberately not the distribution evaluated here. Best response has to use
the one the game actually sees.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from games.kuhn_best_response import (
    KuhnStrategy,
    bet_grid,
    best_response_value_first,
    best_response_value_second,
    game_value,
)
from games.sequential_examples import KIND_CALL, KIND_PASSIVE, ContinuousKuhnPoker

from .actor_critic import masked_log_softmax
from .mixture import MixtureActorCritic, expand_kind_mask


def clipped_mixture_grid_probs(
    weights: chex.Array, means: chex.Array, log_stds: chex.Array, grid: chex.Array
) -> chex.Array:
    """Probability that a *clipped* Gaussian mixture lands in each grid cell.

    Cells are the Voronoi intervals of `grid`, so cell `j` is bounded by the
    midpoints either side of `grid[j]`. Its probability is a difference of
    normal CDFs -- exact, with no quadrature error, since the components are
    Gaussian and the action is one-dimensional.

    Clipping is handled by the endpoints rather than as a special case. `clip` is
    monotone, so the preimage of the first cell is `(-inf, midpoint_0)` and of
    the last is `(midpoint_last, +inf)`: bounding the outer cells by `0` and `1`
    instead of by a CDF *is* the statement that everything below `min_bet` piles
    up on `min_bet` and everything above `max_bet` on `max_bet`.

    `weights` is taken as *joint* -- the probability of choosing each Gaussian
    component at all, not renormalized to sum to 1 -- so the result sums to the
    probability of betting rather than to 1, which is exactly what
    `KuhnStrategy.open_bet` holds.

    The remaining approximation is that every bet inside a cell is treated as
    sitting at `grid[j]`. With the payoff linear in the size, that error is
    bounded by half a cell width; halve it further by refining the grid.
    """
    if grid.shape[0] == 1:
        return jnp.sum(weights, keepdims=True)

    midpoints = 0.5 * (grid[:-1] + grid[1:])
    z = (midpoints[None, :] - means[:, None]) / jnp.exp(log_stds)[:, None]
    cdf = jax.scipy.stats.norm.cdf(z)  # (num_components, num_grid - 1)

    pad = jnp.ones((cdf.shape[0], 1))
    upper = jnp.concatenate([cdf, pad], axis=1)
    lower = jnp.concatenate([jnp.zeros_like(pad), cdf], axis=1)
    return weights @ (upper - lower)


def strategy_from_network(
    game: ContinuousKuhnPoker,
    network: MixtureActorCritic,
    params,
    player: int,
    grid: chex.Array,
) -> KuhnStrategy:
    """Query `player`'s policy at every Kuhn infoset and tabulate it on `grid`.

    Costs `num_cards * (1 + num_grid)` forward passes -- one per card at the
    open node, and one per (card, size) at the node facing a bet, where the
    observed size is part of the infoset. All of it is `vmap`ed into two batched
    calls.
    """
    open_node, faced_node = game.decision_nodes(player)
    num_atoms = network.num_atoms
    open_mask = expand_kind_mask(game.infoset_action_mask(open_node), network.num_components)
    faced_mask = expand_kind_mask(game.infoset_action_mask(faced_node), network.num_components)
    cards = jnp.arange(game.num_cards)

    def at_open(card: chex.Array) -> tuple[chex.Array, chex.Array]:
        logits, means, log_stds, _ = network.apply(
            params, game.infoset_observation(card, open_node, 0.0)
        )
        probs = jnp.exp(masked_log_softmax(logits, open_mask))
        # The Gaussian components' own probabilities stay joint, so the returned
        # size distribution already carries "and it bet at all".
        sizes = clipped_mixture_grid_probs(
            probs[num_atoms:], means[:, 0], log_stds[:, 0], grid
        )
        return probs[KIND_PASSIVE], sizes

    def at_faced(card: chex.Array, bet: chex.Array) -> chex.Array:
        logits, _, _, _ = network.apply(
            params, game.infoset_observation(card, faced_node, bet)
        )
        return jnp.exp(masked_log_softmax(logits, faced_mask))[KIND_CALL]

    open_check, open_bet = jax.vmap(at_open)(cards)
    call = jax.vmap(jax.vmap(at_faced, in_axes=(None, 0)), in_axes=(0, None))(cards, grid)
    return KuhnStrategy(open_check=open_check, open_bet=open_bet, call=call)


def evaluate_networks(
    game: ContinuousKuhnPoker,
    networks: tuple[MixtureActorCritic, MixtureActorCritic],
    params: tuple,
    grid: chex.Array | None = None,
) -> dict[str, chex.Array]:
    """Exploitability, both best-response values, and the game value of a policy pair.

    `exploitability` is the headline: `br_first + br_second`, zero exactly at a
    Nash equilibrium. It is a *lower bound* -- the responder may only bet one of
    the grid's sizes -- so check it has converged by doubling the grid rather
    than trusting a single resolution.
    """
    grid = bet_grid(game) if grid is None else grid
    strategy_0 = strategy_from_network(game, networks[0], params[0], 0, grid)
    strategy_1 = strategy_from_network(game, networks[1], params[1], 1, grid)

    br_first = best_response_value_first(game, grid, strategy_1)
    br_second = best_response_value_second(game, grid, strategy_0)
    return {
        "exploitability": br_first + br_second,
        "br_first": br_first,
        "br_second": br_second,
        "value": game_value(game, grid, strategy_0, strategy_1),
    }


def build_kuhn_metric_fn(game: ContinuousKuhnPoker, num_grid_points: int | None = None):
    """A `metric_fn` for `SequentialSelfPlayPPOTrainer.train`, reporting exploitability.

    Evaluates both the live parameters and the Polyak-averaged `target_params`.
    The averaged iterate is usually the better-behaved of the two in a self-play
    game -- the live one can orbit an equilibrium without ever settling on it --
    so watching only `params` can make a converging run look like a diverging one.
    """
    grid = bet_grid(game) if num_grid_points is None else bet_grid(game, num_grid_points)
    evaluate = None

    def metric_fn(trainer) -> dict[str, float]:
        nonlocal evaluate
        if evaluate is None:
            # Closed over rather than passed as a static argument: a flax Module
            # holding array attributes is not hashable, and the architecture is
            # fixed for the whole run anyway.
            networks = trainer.networks
            evaluate = jax.jit(lambda params: evaluate_networks(game, networks, params, grid))

        live = evaluate(trainer.params)
        target = evaluate(trainer.target_params)
        return {
            **{k: float(v) for k, v in live.items()},
            **{f"{k}_target": float(v) for k, v in target.items()},
        }

    return metric_fn


def build_kuhn_strategy_log_fn(game: ContinuousKuhnPoker, num_grid_points: int = 65):
    """A `strategy_log_fn` for `SequentialSelfPlayPPOTrainer.train`: the whole policy, per card.

    Prints, for each player and card, the probability of betting and the mean
    size conditional on betting, plus the probability of calling a bet of the
    middle size. A coarse grid is plenty -- this is a readout, not a measurement,
    and `build_kuhn_metric_fn` is what produces numbers to trust.
    """
    grid = bet_grid(game) if game.max_bet <= game.min_bet else bet_grid(game, num_grid_points)
    middle = grid.shape[0] // 2
    labels = "JQKA23456789"[: game.num_cards]

    def strategy_log_fn(trainer) -> str:
        lines = []
        for player in (0, 1):
            strategy = strategy_from_network(
                game, trainer.networks[player], trainer.params[player], player, grid
            )
            bet_prob = jnp.sum(strategy.open_bet, axis=-1)
            mean_size = jnp.sum(strategy.open_bet * grid, axis=-1) / jnp.maximum(bet_prob, 1e-9)
            cards = "  ".join(
                f"{labels[c]} bet {float(bet_prob[c]):.2f}@{float(mean_size[c]):.2f}"
                f" call {float(strategy.call[c, middle]):.2f}"
                for c in range(game.num_cards)
            )
            lines.append(f"  p{player} | {cards}")
        return "\n".join(lines)

    return strategy_log_fn
