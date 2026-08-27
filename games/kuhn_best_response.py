"""Exact best response and exploitability for `ContinuousKuhnPoker`.

Kuhn's tree is infinitely branching -- a bet may be any size in
`[min_bet, max_bet]` -- but only in a very contained way, and that is what makes
an exact evaluation possible at all:

  * The **kind-skeleton is finite**: five terminal lines (check-check,
    check-bet-fold, check-bet-call, bet-fold, bet-call) over `num_cards *
    (num_cards - 1)` deals.
  * **No line carries two bet sizes.** Kuhn has no raise, so a bet is answered by
    fold or call and never by another bet. The continuum is one-dimensional per
    line, never a product -- add raises and this whole approach becomes an `M^2`
    grid and stops being cheap.
  * The response to a bet is **binary**, so the only continuous decision is *how
    much to bet*, at one node per player.

What the grid discretizes is therefore not the tree but **the responder's policy
at its continuum of infosets**: a player facing a bet observes its size, so it
has one infoset per real number, and a best response there is a decision
*function* of `b`. The grid represents that function at `M` points.

Everything here operates on explicit `KuhnStrategy` tables, never on a network,
so the arithmetic can be checked against the closed-form Kuhn equilibrium with
no policy, optimizer or sampler in the picture. `training.kuhn_evaluation`
supplies the bridge that reads such a table out of a trained policy.

**The reported number is a lower bound.** The responder may only bet one of `M`
sizes, so its value -- and hence the exploitability -- can only be
under-reported. Confirm the grid is fine enough by doubling `M` and checking the
number stops moving, rather than by trying to bound the policy network's
Lipschitz constant.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp

from .sequential_examples import ContinuousKuhnPoker

# Grid resolution that is plenty for a tree this size; a best response costs one
# batched forward pass of `3 + 3M` observations per player.
DEFAULT_GRID_POINTS = 1025


@chex.dataclass(frozen=True)
class KuhnStrategy:
    """One player's behavioral strategy, as explicit probabilities on a bet grid.

    Both players have exactly the same two decision points -- one where no bet is
    live and one facing a bet -- so a single layout serves either of them:

      * player 0: `NODE_P0_OPEN` and `NODE_P0_AFTER_CHECK_BET`
      * player 1: `NODE_P1_AFTER_CHECK` and `NODE_P1_AFTER_BET`

    `open_check[c] + open_bet[c].sum()` must be 1: `open_bet[c, j]` is the *joint*
    probability of betting at all and of the size landing in grid cell `j`, not
    a conditional. `call[c, j]` is the probability of calling a bet whose size is
    `grid[j]`; folding is its complement.
    """

    open_check: chex.Array  # (num_cards,)
    open_bet: chex.Array  # (num_cards, num_grid)
    call: chex.Array  # (num_cards, num_grid)

    def validate(self, tolerance: float = 1e-4) -> None:
        total = self.open_check + jnp.sum(self.open_bet, axis=-1)
        if not bool(jnp.all(jnp.abs(total - 1.0) < tolerance)):
            raise ValueError(f"open_check + open_bet must sum to 1 per card, got {total}")
        for name, array in (("open_bet", self.open_bet), ("call", self.call), ("open_check", self.open_check)):
            if not bool(jnp.all((array >= -tolerance) & (array <= 1.0 + tolerance))):
                raise ValueError(f"{name} must hold probabilities")


def bet_grid(game: ContinuousKuhnPoker, num_points: int = DEFAULT_GRID_POINTS) -> chex.Array:
    """`num_points` candidate bet sizes spanning `[min_bet, max_bet]`.

    Collapses to the single legal size when the game fixes one
    (`min_bet == max_bet`), which is the classic-Kuhn case -- and there the grid
    introduces no approximation whatsoever, so it is the setting in which this
    module can be checked against a closed form.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be at least 1, got {num_points}")
    if game.max_bet <= game.min_bet:
        return jnp.asarray([game.min_bet], dtype=jnp.float32)
    return jnp.linspace(game.min_bet, game.max_bet, num_points, dtype=jnp.float32)


def _deal(num_cards: int) -> tuple[chex.Array, chex.Array]:
    """`(pair, showdown)`: the deal distribution, and who wins at a showdown.

    `pair[a, b]` is the probability of the dealer handing *this* player `a` and
    the other player `b` -- zero on the diagonal, since the deal is without
    replacement. `showdown[a, b]` is `+1` when `a` beats `b`.

    Both are written from the perspective of "my card, their card" rather than
    "player 0's card, player 1's card", which is what lets the same tensors serve
    both directions of the best response: a showdown is worth `+1` to whoever
    holds the higher card, whichever seat they sit in.
    """
    cards = jnp.arange(num_cards)
    distinct = (cards[:, None] != cards[None, :]).astype(jnp.float32)
    pair = distinct / (num_cards * (num_cards - 1))
    showdown = jnp.where(cards[:, None] > cards[None, :], 1.0, -1.0)
    return pair, showdown


def game_value(
    game: ContinuousKuhnPoker,
    grid: chex.Array,
    strategy_0: KuhnStrategy,
    strategy_1: KuhnStrategy,
) -> chex.Array:
    """Player 0's expected payoff when the two strategies play each other.

    Not needed for exploitability, but the cheapest possible check on both the
    strategies and this module: at a known equilibrium it must equal the known
    game value (`-1/18` for classic Kuhn).
    """
    pair, showdown = _deal(game.num_cards)
    scale = 1.0 + grid  # (M,) pot won or lost when a bet of this size is called

    # Player 0 checks. Player 1 checks back (showdown) or bets, and player 0
    # folds (-1) or calls.
    checked_down = strategy_1.open_check[None, :] * showdown  # (c0, c1)
    faced_bet = strategy_1.open_bet[None, :, :] * (
        -(1.0 - strategy_0.call[:, None, :]) + strategy_0.call[:, None, :] * showdown[..., None] * scale
    )  # (c0, c1, M)
    after_check = strategy_0.open_check[:, None] * (checked_down + jnp.sum(faced_bet, axis=-1))

    # Player 0 bets. Player 1 folds (+1 to player 0) or calls.
    after_bet = strategy_0.open_bet[:, None, :] * (
        (1.0 - strategy_1.call[None, :, :]) + strategy_1.call[None, :, :] * showdown[..., None] * scale
    )  # (c0, c1, M)

    return jnp.sum(pair * (after_check + jnp.sum(after_bet, axis=-1)))


def best_response_value_first(
    game: ContinuousKuhnPoker, grid: chex.Array, opponent: KuhnStrategy
) -> chex.Array:
    """Best achievable value for **player 0** against a fixed player 1.

    Player 0 acts twice on one line -- open, then possibly face a bet after
    checking -- so the value of checking must already contain the optimal
    continuation. That is why `V3` is summed *into* `value_check` rather than
    added on at the end, and it is the whole first-mover asymmetry between this
    function and `best_response_value_second`.

    All values are counterfactual: reach-weighted by chance and the opponent
    only, never by player 0's own actions, and never normalized into a belief.
    An infoset the opponent never puts mass on then contributes exactly zero
    instead of dividing by zero, which is the correct answer -- exploitability
    does not care what a best response does where it cannot be reached.
    """
    pair, showdown = _deal(game.num_cards)
    weighted_showdown = pair * showdown
    scale = 1.0 + grid

    # Facing player 1's bet, having checked. One infoset per (own card, size).
    fold_here = -(pair @ opponent.open_bet)  # (c0, M)
    call_here = (weighted_showdown @ opponent.open_bet) * scale
    value_faced = jnp.maximum(fold_here, call_here)

    # Checking: player 1 checks back for a showdown, or bets and we continue optimally.
    value_check = weighted_showdown @ opponent.open_check + jnp.sum(value_faced, axis=-1)

    # Betting size `grid[j]`: player 1 folds (+1) or calls.
    value_bet = pair @ (1.0 - opponent.call) + (weighted_showdown @ opponent.call) * scale

    return jnp.sum(jnp.maximum(value_check, jnp.max(value_bet, axis=-1)))


def best_response_value_second(
    game: ContinuousKuhnPoker, grid: chex.Array, opponent: KuhnStrategy
) -> chex.Array:
    """Best achievable value for **player 1** against a fixed player 0.

    Player 1 acts exactly once per line -- either responding to a bet or
    responding to a check -- so its two infoset families are reached under
    disjoint events and their optimal values simply add. No continuation is
    nested anywhere, unlike `best_response_value_first`.

    `showdown` is indexed "my card, their card" throughout (see `_deal`), so a
    called bet is worth `+1 * scale` to whoever holds the better card and no
    sign flip is needed to move to player 1's seat.
    """
    pair, showdown = _deal(game.num_cards)
    weighted_showdown = pair * showdown
    scale = 1.0 + grid

    # Facing player 0's opening bet. One infoset per (own card, size), terminal after.
    fold_here = -(pair @ opponent.open_bet)  # (c1, M)
    call_here = (weighted_showdown @ opponent.open_bet) * scale
    value_faced = jnp.maximum(fold_here, call_here)

    # Facing player 0's check: check back for a showdown, or bet and let player 0 respond.
    reached = opponent.open_check  # (c0,) -- player 0's own reach into this infoset
    value_check = weighted_showdown @ reached
    value_bet = pair @ ((1.0 - opponent.call) * reached[:, None]) + (
        weighted_showdown @ (opponent.call * reached[:, None])
    ) * scale
    value_after_check = jnp.maximum(value_check, jnp.max(value_bet, axis=-1))

    return jnp.sum(value_after_check) + jnp.sum(value_faced)


def exploitability(
    game: ContinuousKuhnPoker,
    grid: chex.Array,
    strategy_0: KuhnStrategy,
    strategy_1: KuhnStrategy,
) -> chex.Array:
    """`BR_0(pi_1) + BR_1(pi_0)`: zero exactly at a Nash equilibrium, positive otherwise.

    Each term is what that player could earn by abandoning its strategy and
    playing optimally against the other. At equilibrium they are `v` and `-v` and
    cancel; away from it both are at least as large, so the sum measures how much
    the pair jointly leaves on the table. Restricted to the grid, so it is a
    lower bound -- see the module docstring.

    Only the *sum* is non-negative. The individual terms are bounded below by the
    game value and its negation (`-1/18` and `+1/18` for classic Kuhn), not by
    zero, so a negative `best_response_value_first` against a strong opponent is
    correct rather than a bug.
    """
    return best_response_value_first(game, grid, strategy_1) + best_response_value_second(
        game, grid, strategy_0
    )


def analytic_equilibrium(game: ContinuousKuhnPoker, alpha: float = 1.0 / 6.0) -> tuple[KuhnStrategy, KuhnStrategy]:
    """The closed-form equilibrium family of *classic* three-card Kuhn poker.

    Valid only when the bet size is fixed (`min_bet == max_bet`) and the deck has
    three cards -- that is the game Kuhn solved. The equilibria form a
    one-parameter family in `alpha` in `[0, 1/3]`; player 1's strategy is unique.
    With cards ordered J < Q < K:

      * player 0 bets J with probability `alpha`, never Q, and K with `3 * alpha`;
        facing a bet it folds J, calls Q with `alpha + 1/3`, and always calls K.
      * player 1, facing a check, bets J with `1/3`, never Q, always K; facing a
        bet it folds J, calls Q with `1/3`, and always calls K.

    Every such pair has value `-1/18` to player 0. This exists so the arithmetic
    above can be checked against something that was not produced by it.
    """
    if game.num_cards != 3:
        raise ValueError(f"the analytic Kuhn equilibrium is for a three-card deck, got {game.num_cards}")
    if game.max_bet > game.min_bet:
        raise ValueError("the analytic Kuhn equilibrium is for a fixed bet size (min_bet == max_bet)")
    if not 0.0 <= alpha <= 1.0 / 3.0:
        raise ValueError(f"alpha must lie in [0, 1/3], got {alpha}")

    def strategy(bet: list[float], call: list[float]) -> KuhnStrategy:
        bet_array = jnp.asarray(bet, dtype=jnp.float32)
        return KuhnStrategy(
            open_check=1.0 - bet_array,
            open_bet=bet_array[:, None],
            call=jnp.asarray(call, dtype=jnp.float32)[:, None],
        )

    first = strategy(bet=[alpha, 0.0, 3.0 * alpha], call=[0.0, alpha + 1.0 / 3.0, 1.0])
    second = strategy(bet=[1.0 / 3.0, 0.0, 1.0], call=[0.0, 1.0 / 3.0, 1.0])
    return first, second
