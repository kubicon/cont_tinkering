"""Colonel Blotto played one battlefield at a time, with continuous bids.

The classic game is one-shot: both colonels split a budget across every front
simultaneously, and `games.examples.ContinuousBlottoGame` is that game. This is
its *extensive-form* counterpart. The fronts are contested one after another;
each front is settled before the next one opens, and all a player learns is
**who won it** -- never how much the opponent actually committed. So a loss says
"they outbid me here" and nothing more, which is a bound on their remaining
budget rather than a number: beliefs about how much powder the opponent has left
are the whole informational content of the game, and spending big early to look
rich is a bluff in the poker sense.

**Why bids are fractions.** `games.sequential` requires an action space that
does not depend on the state, but the amount a colonel may commit obviously
does -- it is capped by what they have left. The action here is therefore the
*fraction of the remaining budget* to commit, always `[0, 1]`, with the game
multiplying it by the budget actually left. The constraint is then satisfied by
construction: no mask, no projection, no illegal action to reject.

**Why the two bids on a front are simultaneous.** If player 1 saw player 0's bid
before answering it, they would answer it with "that plus epsilon" -- the
supremum is not attained, and the game has no equilibrium in the naive
continuous strategy space. The tree is still turn-taking (player 0 bids, then
player 1 bids), but player 1's observation omits the live bid, which is the
standard extensive-form encoding of a simultaneous move.

**Resolving a front.** With `sharpness=None` the bigger bid wins outright, ties
on a coin. Otherwise the front is a logistic contest -- player 0 wins with
probability `sigmoid(sharpness * (x - y))` -- which is the same softening the
one-shot `ContinuousBlottoGame` applies, and recovers the hard rule as
`sharpness -> inf`. Under PPO the payoff is never differentiated (the gradient
goes through `log pi`, not through the reward), so the hard rule is perfectly
trainable; what it costs is *directional* information. With a smooth contest a
near-deterministic policy still learns "bid a little more, win a little more",
while under the hard rule a policy whose mass has drifted entirely to one side
of the opponent sees the same payoff on every sample and gets no gradient at all
until exploration wanders back. Given that a mixture policy's failure mode is
exactly variance collapse, `sharpness` is the knob to reach for first when a run
flatlines.

Either way the winner is *sampled*, so the observed win/loss is genuinely all
the information a player gets, and the payoff is a sum of indicators rather than
of probabilities -- the hard limit is then a limit of the same game, not a
different one.

**Perfect recall.** A player's observation carries their own per-front spends,
not merely what is left of their budget: the remaining budget alone would forget
*where* the money went, and a player who cannot recall their own past actions is
playing a different (imperfect-recall) game.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .sequential import TERMINAL, SequentialZeroSumGame
from .spaces import HybridAction, HybridSpace, hybrid

# The bid is a pure continuous action: there is nothing discrete to choose
# alongside it (unlike a poker tree's fold/check/call), so the space has no
# atoms and `num_kinds` is 1.
NUM_ATOMS = 0


@chex.dataclass(frozen=True)
class BlottoState:
    """Fixed-shape state of one `ContinuousSequentialBlotto` game.

    `turn` is the entire public clock: front `turn // 2` is being contested and
    player `turn % 2` is to act, so the state needs no separate "whose move"
    field and no history list. It runs to `2 * num_fields`, which is terminal.

    `spends` holds both players' commitments per front -- player 0's bid on the
    live front is written there by their move and read back by the resolution
    half a step later, which is why there is no separate "pending bid" field.
    Only a player's *own* row ever reaches their observation.
    """

    spends: chex.Array  # (2, num_fields) float32, committed so far; 0 on unplayed fronts
    results: chex.Array  # (num_fields,) float32, +1 player 0 took the front, -1 player 1, 0 unplayed
    turn: chex.Array  # () int32 in [0, 2 * num_fields]


class ContinuousSequentialBlotto(SequentialZeroSumGame):
    """Sequential, continuous-bid Colonel Blotto with win/loss-only feedback.

    `num_fields` fronts are contested in order. On each one player 0 commits a
    fraction of their remaining budget, then player 1 does so without seeing
    that commitment, the front is awarded (see the module docstring), the result
    is announced to both, and play moves on. Both commitments are spent whether
    they won or lost -- this is an all-pay contest, which is what makes the
    budget the strategic object.

    Player 0's payoff is `sum(field_values * results)`: the value of the fronts
    they took minus the value of the fronts they lost.

    **Two useful degenerate settings.** `num_fields=1` is a single all-pay
    auction over one prize, whose equilibrium is known. Large `sharpness` with
    many fronts approaches the classic hard-argmax game, whose one-shot version
    has Roberson's known equilibrium marginals -- a sanity reference for what
    "spread out" is supposed to mean here, though the sequential game's
    equilibrium is not that.
    """

    def __init__(
        self,
        num_fields: int = 3,
        field_values: tuple[float, ...] | None = None,
        budget: float = 1.0,
        sharpness: float | None = 100.0,
    ):
        if num_fields < 1:
            raise ValueError(f"num_fields must be at least 1, got {num_fields}")
        if budget <= 0.0:
            raise ValueError(f"budget must be positive, got {budget}")
        if sharpness is not None and sharpness <= 0.0:
            raise ValueError(
                f"sharpness must be positive or None (a hard argmax), got {sharpness}"
            )
        if field_values is None:
            field_values = (1.0,) * num_fields
        if len(field_values) != num_fields:
            raise ValueError(
                f"field_values must have one entry per front: got {len(field_values)} "
                f"for num_fields={num_fields}"
            )
        if any(v <= 0.0 for v in field_values):
            raise ValueError(f"field values must be positive, got {field_values}")

        self.num_fields = int(num_fields)
        self.budget = float(budget)
        self.sharpness = None if sharpness is None else float(sharpness)
        self.field_values = tuple(float(v) for v in field_values)
        self._values = jnp.asarray(self.field_values, dtype=jnp.float32)
        # The bid is a fraction of what is left, hence a fixed `[0, 1]` box; see
        # the module docstring on why it is not the amount itself.
        self._space = hybrid(NUM_ATOMS, [0.0], [1.0])

    # ---- shape/static information ------------------------------------------

    @property
    def max_steps(self) -> int:
        """One decision per player per front."""
        return 2 * self.num_fields

    def action_space(self, player: int) -> HybridSpace:
        return self._space

    def obs_dim(self, player: int) -> int:
        return 3 * self.num_fields + 2

    # ---- the game tree ------------------------------------------------------

    def initial_state(self, key: chex.PRNGKey) -> BlottoState:
        del key  # no deal: the only chance moves are the front resolutions
        return BlottoState(
            spends=jnp.zeros((2, self.num_fields), dtype=jnp.float32),
            results=jnp.zeros((self.num_fields,), dtype=jnp.float32),
            turn=jnp.zeros((), dtype=jnp.int32),
        )

    def current_player(self, state: BlottoState) -> chex.Array:
        return jnp.where(state.turn >= self.max_steps, TERMINAL, state.turn % 2).astype(jnp.int32)

    def observation(self, player: int, state: BlottoState) -> chex.Array:
        """`(3 * num_fields + 2,)`: own spends, public results, own budget left, the clock.

        The opponent's row of `spends` is deliberately absent -- including it
        would hand over both the live bid on this front (turning the
        simultaneous move into a sequential one) and the opponent's remaining
        budget (deleting the game's entire hidden state).

        `results` is signed from `player`'s point of view, so both players read
        "+1 means I took that front"; the network for player 1 therefore does not
        have to learn a sign convention off its own index.
        """
        sign = 1.0 if player == 0 else -1.0
        own_spends = state.spends[player] / self.budget
        remaining = self.remaining_budget(player, state) / self.budget
        # `num_fields + 1` slots: one per front, plus the finished game. Clamped
        # because `observation` is called on terminal states too, where
        # `turn // 2 == num_fields`.
        front = jax.nn.one_hot(
            jnp.minimum(state.turn // 2, self.num_fields), self.num_fields + 1
        )
        return jnp.concatenate([own_spends, sign * state.results, remaining[None], front])

    def action_mask(self, player: int, state: BlottoState) -> chex.Array:
        """`(1,)` all-`True`: bidding is the only kind, and every fraction is legal."""
        del player, state  # nothing is ever illegal here -- the box is the whole rule
        return jnp.ones((self.num_kinds(0),), dtype=bool)

    def payoff(self, state: BlottoState) -> chex.Array:
        return jnp.sum(self._values * state.results)

    def _step(self, state: BlottoState, action: HybridAction, key: chex.PRNGKey) -> BlottoState:
        player = state.turn % 2
        # Clamped for the same reason as in `observation`: `step`'s terminal
        # guard throws this result away, but the indexed write below still has to
        # stay in bounds while it is being computed.
        front = jnp.minimum(state.turn // 2, self.num_fields - 1)

        fraction = jnp.squeeze(self._space.box.clip(action.value), axis=-1)
        spend = fraction * self._remaining(state.spends, player)
        spends = state.spends.at[player, front].set(spend)

        # Player 1's move closes the front: both bids are now in `spends`, so the
        # contest reads them from there rather than from a carried pending bid.
        resolves = player == 1
        win_0 = jax.random.bernoulli(key, self._win_probability(spends[0, front], spends[1, front]))
        results = jnp.where(
            resolves, state.results.at[front].set(jnp.where(win_0, 1.0, -1.0)), state.results
        )

        return BlottoState(spends=spends, results=results, turn=state.turn + 1)

    # ---- rules helpers ------------------------------------------------------

    def _win_probability(self, spend_0: chex.Array, spend_1: chex.Array) -> chex.Array:
        """Probability player 0 takes a front given the two commitments on it."""
        if self.sharpness is None:
            return jnp.where(spend_0 > spend_1, 1.0, jnp.where(spend_0 < spend_1, 0.0, 0.5))
        return jax.nn.sigmoid(self.sharpness * (spend_0 - spend_1))

    def _remaining(self, spends: chex.Array, player: chex.Array) -> chex.Array:
        """Budget left to `player` (a traced index) given every commitment so far."""
        return jnp.maximum(self.budget - jnp.sum(spends[player]), 0.0)

    def remaining_budget(self, player: int, state: BlottoState) -> chex.Array:
        """What `player` has left to commit -- the budget minus everything spent.

        Derived rather than stored: `spends` is the single source of truth, so
        the two can never disagree.
        """
        return self._remaining(state.spends, player)
