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
from games.discretized import DiscretizedSequentialGame
from games.sequential_examples import KIND_CALL, KIND_PASSIVE, ContinuousKuhnPoker

from .actor_critic import masked_log_softmax
from .gaussian import marginal_std
from .mixture import MixtureActorCritic, expand_kind_mask


def clipped_mixture_grid_probs(
    weights: chex.Array, means: chex.Array, stds: chex.Array, grid: chex.Array
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
    z = (midpoints[None, :] - means[:, None]) / stds[:, None]
    cdf = jax.scipy.stats.norm.cdf(z)  # (num_components, num_grid - 1)

    pad = jnp.ones((cdf.shape[0], 1))
    upper = jnp.concatenate([cdf, pad], axis=1)
    lower = jnp.concatenate([jnp.zeros_like(pad), cdf], axis=1)
    return weights @ (upper - lower)


def greedy_mixture_grid_probs(
    weights: chex.Array, means: chex.Array, grid: chex.Array
) -> chex.Array:
    """`clipped_mixture_grid_probs` in the limit `stds -> 0`: every component bets its mean.

    Each component's whole (joint) weight lands in the grid cell holding
    `clip(mean)`. Cells are the Voronoi intervals of `grid` and its endpoints are
    the box's, so that is simply the grid point nearest the unclipped mean -- a
    mean below `min_bet` snaps to the first point, above `max_bet` to the last.
    Only the Gaussians are made greedy: the weights, i.e. whether to bet and
    with which component, stay a distribution.
    """
    if grid.shape[0] == 1:
        return jnp.sum(weights, keepdims=True)

    cell = jnp.argmin(jnp.abs(means[:, None] - grid[None, :]), axis=-1)
    return jnp.zeros(grid.shape[0], dtype=weights.dtype).at[cell].add(weights)


def strategy_from_network(
    game: ContinuousKuhnPoker,
    network: MixtureActorCritic,
    params,
    player: int,
    grid: chex.Array,
    greedy_gaussians: bool = False,
) -> KuhnStrategy:
    """Query `player`'s policy at every Kuhn infoset and tabulate it on `grid`.

    Costs `num_cards * (1 + num_grid)` forward passes -- one per card at the
    open node, and one per (card, size) at the node facing a bet, where the
    observed size is part of the infoset. All of it is `vmap`ed into two batched
    calls.

    `greedy_gaussians` evaluates the policy with every Gaussian's sigma set to
    zero (see `greedy_mixture_grid_probs`); the categorical head -- check/bet,
    the component weights, fold/call -- is read as the distribution it is.
    """
    open_node, faced_node = game.decision_nodes(player)
    num_atoms = network.num_atoms
    open_mask = expand_kind_mask(game.infoset_action_mask(open_node), network.num_components)
    faced_mask = expand_kind_mask(game.infoset_action_mask(faced_node), network.num_components)
    cards = jnp.arange(game.num_cards)

    def at_open(card: chex.Array) -> tuple[chex.Array, chex.Array]:
        logits, means, scale_trils, _ = network.apply(
            params, game.infoset_observation(card, open_node, 0.0)
        )
        probs = jnp.exp(masked_log_softmax(logits, open_mask))
        # The Gaussian components' own probabilities stay joint, so the returned
        # size distribution already carries "and it bet at all".
        # The bet size is action coordinate 0; its *marginal* standard deviation
        # is what a one-dimensional CDF over that coordinate needs.
        if greedy_gaussians:
            sizes = greedy_mixture_grid_probs(probs[num_atoms:], means[:, 0], grid)
        else:
            sizes = clipped_mixture_grid_probs(
                probs[num_atoms:], means[:, 0], marginal_std(scale_trils)[:, 0], grid
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


def discrete_strategy_from_network(
    game: DiscretizedSequentialGame,
    network: MixtureActorCritic,
    params,
    player: int,
    grid: chex.Array,
) -> KuhnStrategy:
    """`strategy_from_network` for a policy playing a *discretized* Kuhn.

    Only one of the two decisions changes. The bet-size distribution is now a
    plain categorical over the grid the game offers -- no clipped mixture, no
    CDFs -- and it is reported on the caller's evaluation grid by putting each
    action's probability on the nearest point of it, so the tables mean the same
    thing they do for a continuous policy and the same best-response arithmetic
    reads them. (Snapping is exact whenever the evaluation grid contains the
    action grid, which it does at any sane resolution, and otherwise costs half
    an evaluation cell -- the approximation the continuous reader already makes.)

    The response to a bet does *not* change: a discretized player still observes
    the real size it is facing (see `games.discretized`), so `call` is queried
    across the full evaluation grid exactly as before. That is what keeps the
    exploitability computed from these tables honest about the grid's cost --
    the best response may bet between the points, and it will.
    """
    base = game.game
    if not isinstance(base, ContinuousKuhnPoker):
        raise ValueError(f"expected a discretized Kuhn game, got one wrapping {type(base).__name__}")
    open_node, faced_node = base.decision_nodes(player)
    open_mask = expand_kind_mask(
        game.discretize_kind_mask(base.infoset_action_mask(open_node)), network.num_components)
    faced_mask = expand_kind_mask(
        game.discretize_kind_mask(base.infoset_action_mask(faced_node)), network.num_components)
    cards = jnp.arange(base.num_cards)

    # Bet size is action coordinate 0, and the action grid is one point per kind
    # from `base_atoms` onwards.
    sizes = game.grid(player)[:, 0]
    column = jnp.argmin(jnp.abs(grid[None, :] - sizes[:, None]), axis=-1)  # (num_grid,)
    first_bet, last_bet = game.base_atoms, game.base_atoms + game.num_grid

    def at_open(card: chex.Array) -> tuple[chex.Array, chex.Array]:
        logits, _, _, _ = network.apply(
            params, base.infoset_observation(card, open_node, 0.0))
        probs = jnp.exp(masked_log_softmax(logits, open_mask))
        # Joint, like the continuous reader's: these already carry "and it bet at all".
        sizes_row = jnp.zeros(grid.shape[0]).at[column].add(probs[first_bet:last_bet])
        return probs[KIND_PASSIVE], sizes_row

    def at_faced(card: chex.Array, bet: chex.Array) -> chex.Array:
        logits, _, _, _ = network.apply(
            params, base.infoset_observation(card, faced_node, bet))
        return jnp.exp(masked_log_softmax(logits, faced_mask))[KIND_CALL]

    open_check, open_bet = jax.vmap(at_open)(cards)
    call = jax.vmap(jax.vmap(at_faced, in_axes=(None, 0)), in_axes=(0, None))(cards, grid)
    return KuhnStrategy(open_check=open_check, open_bet=open_bet, call=call)


def strategy_from_policy(
    game,
    network: MixtureActorCritic,
    params,
    player: int,
    grid: chex.Array,
) -> KuhnStrategy:
    """The right reader for whichever Kuhn `game` is -- continuous or discretized.

    One dispatch point, so a caller measuring a run never has to know which
    action set the policy it was handed was trained on.
    """
    if isinstance(game, DiscretizedSequentialGame):
        return discrete_strategy_from_network(game, network, params, player, grid)
    return strategy_from_network(game, network, params, player, grid)


def evaluate_networks(
    game: ContinuousKuhnPoker,
    networks: tuple[MixtureActorCritic, MixtureActorCritic],
    params: tuple,
    grid: chex.Array | None = None,
    greedy_gaussians: bool = False,
) -> dict[str, chex.Array]:
    """Exploitability, both best-response values, and the game value of a policy pair.

    `exploitability` is the headline: `br_first + br_second`, zero exactly at a
    Nash equilibrium. It is a *lower bound* -- the responder may only bet one of
    the grid's sizes -- so check it has converged by doubling the grid rather
    than trusting a single resolution.

    `greedy_gaussians` scores both policies with their Gaussians' sigmas at zero
    (see `strategy_from_network`).
    """
    grid = bet_grid(game) if grid is None else grid
    strategy_0 = strategy_from_network(game, networks[0], params[0], 0, grid, greedy_gaussians)
    strategy_1 = strategy_from_network(game, networks[1], params[1], 1, grid, greedy_gaussians)

    br_first = best_response_value_first(game, grid, strategy_1)
    br_second = best_response_value_second(game, grid, strategy_0)
    return {
        "exploitability": br_first + br_second,
        "br_first": br_first,
        "br_second": br_second,
        "value": game_value(game, grid, strategy_0, strategy_1),
    }


def build_kuhn_metric_fn(
    game: ContinuousKuhnPoker, num_grid_points: int | None = None, greedy_gaussians: bool = False
):
    """A `metric_fn` for `SequentialSelfPlayPPOTrainer.train`, reporting exploitability.

    Evaluates both the live parameters and the Polyak-averaged `target_params`.
    The averaged iterate is usually the better-behaved of the two in a self-play
    game -- the live one can orbit an equilibrium without ever settling on it --
    so watching only `params` can make a converging run look like a diverging one.

    `greedy_gaussians` scores both with every Gaussian's sigma at zero, the
    categorical head still a distribution (see `evaluate_networks`).
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
            evaluate = jax.jit(
                lambda params: evaluate_networks(game, networks, params, grid, greedy_gaussians)
            )

        live = evaluate(trainer.params)
        target = evaluate(trainer.target_params)
        return {
            **{k: float(v) for k, v in live.items()},
            **{f"{k}_target": float(v) for k, v in target.items()},
        }

    return metric_fn


def build_kuhn_strategy_log_fn(game: ContinuousKuhnPoker, num_call_sizes: int = 8):
    """A `strategy_log_fn` for `SequentialSelfPlayPPOTrainer.train`: both live policies, infoset by infoset.

    Each player acts at two kinds of infoset, and each gets its own block:

      * **no bet outstanding** -- player 0's opening node, player 1's after
        player 0 checks. One row per card: the probability of checking, then of
        betting, split into the mixture's Gaussian components as
        `weight x (mean ± sigma)`. Check and the component weights are the whole
        categorical head, so `check + bet = 1` and `bet` is the sum of the
        weights. A drawn size outside `[min_bet, max_bet]` is clipped to the
        nearer end when played.
      * **facing a bet** -- player 1's after player 0 bets, player 0's after
        checking and being bet into. The bet's size is part of the infoset, so
        one column per size in the header and one row per card: the probability
        of calling that size. Folding is the rest.

    A readout, not a measurement: `build_kuhn_metric_fn` is what scores the
    clipped bet-size distribution the game actually sees.
    """
    labels = "JQKA23456789"[: game.num_cards]
    cards = jnp.arange(game.num_cards)
    sizes = jnp.linspace(
        game.min_bet, game.max_bet, 1 if game.max_bet <= game.min_bet else num_call_sizes
    )
    titles = (
        ("P0 opens, no bet yet", "P0 checked, P1 bet b"),
        ("P1 after P0 checks", "P1 facing P0's bet b"),
    )

    def strategy_log_fn(trainer) -> str:
        lines = []
        for player in (0, 1):
            network, params = trainer.networks[player], trainer.params[player]
            num_atoms = network.num_atoms
            open_node, faced_node = game.decision_nodes(player)
            open_title, faced_title = titles[player]

            open_mask = expand_kind_mask(game.infoset_action_mask(open_node), network.num_components)
            logits, means, scale_trils, _ = jax.vmap(
                lambda card: network.apply(params, game.infoset_observation(card, open_node, 0.0))
            )(cards)
            probs = jnp.exp(jax.vmap(masked_log_softmax, in_axes=(0, None))(logits, open_mask))
            sigmas = marginal_std(scale_trils)[..., 0]
            lines.append(f"  {open_title}: check + bet = 1, bet split as weight x (mean ± sigma)")
            for c in range(game.num_cards):
                weights = probs[c, num_atoms:]
                components = "  ".join(
                    f"{float(w):.2f}x({float(m):.3f}±{float(s):.3f})"
                    for w, m, s in zip(weights, means[c, :, 0], sigmas[c])
                )
                lines.append(
                    f"    {labels[c]} | check {float(probs[c, KIND_PASSIVE]):.2f}"
                    f"  bet {float(jnp.sum(weights)):.2f} = {components}"
                )

            faced_mask = expand_kind_mask(game.infoset_action_mask(faced_node), network.num_components)

            def call_prob(card, bet):
                logits, _, _, _ = network.apply(params, game.infoset_observation(card, faced_node, bet))
                return jnp.exp(masked_log_softmax(logits, faced_mask))[KIND_CALL]

            call = jax.vmap(jax.vmap(call_prob, in_axes=(None, 0)), in_axes=(0, None))(cards, sizes)
            lines.append(f"  {faced_title}: P(call) for each bet size b, fold = 1 - call")
            lines.append("    b | " + "  ".join(f"{float(b):.2f}" for b in sizes))
            for c in range(game.num_cards):
                lines.append(f"    {labels[c]} | " + "  ".join(f"{float(p):.2f}" for p in call[c]))
        return "\n".join(lines)

    return strategy_log_fn
