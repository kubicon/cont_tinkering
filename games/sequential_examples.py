"""Concrete `SequentialZeroSumGame`s, mainly to exercise/demonstrate the base class.

The other one lives in `games.leduc`: continuous-raise Leduc Hold'em, whose two
rounds and raise cap put it past the point where every betting history can be
enumerated as a node the way Kuhn's are here.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .sequential import TERMINAL, SequentialZeroSumGame
from .spaces import HybridAction, HybridSpace, hybrid

# The four decision nodes of the Kuhn betting tree, plus the terminal marker.
# `node` is the entire public history: the tree is small enough that every
# distinct betting sequence is one of these, which is what lets the state stay a
# fixed-shape pytree (see `games.sequential`).
NODE_P0_OPEN = 0  # player 0 to act, nothing bet yet
NODE_P1_AFTER_CHECK = 1  # player 1 to act, player 0 checked
NODE_P1_AFTER_BET = 2  # player 1 to act, facing player 0's bet
NODE_P0_AFTER_CHECK_BET = 3  # player 0 to act, facing player 1's bet after checking
NODE_TERMINAL = 4
NUM_NODES = 5

# Action kinds, indexing `HybridAction.kind` (and the categorical head's first
# `NUM_ATOMS` entries). The two atoms carry no parameter; `KIND_BET` is the
# continuous branch and is the only kind whose `value` is read.
KIND_PASSIVE = 0  # check when no bet is live, fold when facing one
KIND_CALL = 1  # match the outstanding bet
KIND_BET = 2  # bet `value`; == NUM_ATOMS, i.e. `HybridSpace.continuous_kind`
NUM_ATOMS = 2

# Which player acts at each node, indexed by `node`.
_NODE_PLAYER = (0, 1, 1, 0, TERMINAL)

# Decision nodes belonging to each player, indexed by player.
_PLAYER_NODES = ((NODE_P0_OPEN, NODE_P0_AFTER_CHECK_BET), (NODE_P1_AFTER_CHECK, NODE_P1_AFTER_BET))

# Legal action kinds at each node: `(passive, call, bet)`, indexed by `node`.
# Calling needs a bet outstanding; betting needs none (Kuhn has no raise). The
# terminal row is all-`True` purely to keep the masked softmax well posed -- see
# `SequentialZeroSumGame.action_mask`; nothing is sampled there.
_NODE_ACTION_MASK = (
    (True, False, True),  # NODE_P0_OPEN:            check or bet
    (True, False, True),  # NODE_P1_AFTER_CHECK:     check or bet
    (True, True, False),  # NODE_P1_AFTER_BET:       fold or call
    (True, True, False),  # NODE_P0_AFTER_CHECK_BET: fold or call
    (True, True, True),   # NODE_TERMINAL:           unused
)


@chex.dataclass(frozen=True)
class KuhnState:
    """Fixed-shape state of one `ContinuousKuhnPoker` hand.

    `payoff` is carried in the state rather than recomputed from the history:
    the terminating transition is the only place where the fold/showdown
    distinction and the outstanding bet are both in hand, so it writes the
    result once and `step`'s terminal guard preserves it for the remaining
    (no-op) steps of the scan.
    """

    cards: chex.Array  # (2,) int32, each player's private card
    node: chex.Array  # () int32, one of the `NODE_*` constants
    bet: chex.Array  # () float32, the outstanding bet size (0 when no bet is live)
    payoff: chex.Array  # () float32, player 0's payoff; only final once terminal


class ContinuousKuhnPoker(SequentialZeroSumGame):
    """Kuhn poker with a *continuously sized* bet.

    The tree is Kuhn's: both players ante 1 and are dealt one private card each
    from a `num_cards`-card deck without replacement; player 0 checks or bets;
    a check is answered by a check (showdown) or a bet, which player 0 may fold
    to or call; a bet is answered by a fold or a call. What is continuous is
    *how much* the bet is -- so a strategy at a betting node is a distribution
    over `[min_bet, max_bet]` together with the probability of not betting at
    all, exactly the mixed discrete/continuous object a `MixtureActorCritic`
    with atoms represents.

    **Actions.** Two atoms and one continuous branch (`KIND_PASSIVE`,
    `KIND_CALL`, `KIND_BET`), with `action_mask` making exactly two of the three
    legal at any decision node: check-or-bet with no bet outstanding, fold-or-
    call facing one. Only `KIND_BET` reads `HybridAction.value`, which *is* the
    bet size -- the continuous action space is `[min_bet, max_bet]` directly,
    with no rescaling, so the mixture components' means live in the same units
    the game is played in.

    `min_bet > 0` is a real rule here rather than a representational
    workaround: a bet of 0 would still be a distinct branch of the tree (the
    opponent could fold to it and forfeit an ante), so allowing it would make
    the game degenerate rather than continuous.

    **Recovering classic Kuhn.** With `min_bet == max_bet == 1` the size is
    constant and this is exactly the textbook game, whose equilibria are known
    in closed form -- the baseline to check a sequential trainer against before
    trusting it on the continuous version.

    **Infosets.** Player `p` acts at the two nodes in `decision_nodes(p)`, so
    with a discrete deck there are `2 * num_cards` infosets per player *plus*
    the bet size a player faces, which is continuous at
    `NODE_P1_AFTER_BET`/`NODE_P0_AFTER_CHECK_BET`. That continuum is why
    exploitability here needs the bet axis discretized (or an approximate
    best-responder) rather than a plain tabular traversal.
    """

    def __init__(self, num_cards: int = 3, min_bet: float = 0.5, max_bet: float = 2.0):
        if num_cards < 2:
            raise ValueError(f"num_cards must be at least 2, got {num_cards}")
        if not 0.0 < min_bet <= max_bet:
            raise ValueError(f"need 0 < min_bet <= max_bet, got {min_bet} and {max_bet}")

        self.num_cards = num_cards
        self.min_bet = float(min_bet)
        self.max_bet = float(max_bet)
        self._space = hybrid(NUM_ATOMS, [min_bet], [max_bet])
        self._node_player = jnp.asarray(_NODE_PLAYER, dtype=jnp.int32)
        self._node_action_mask = jnp.asarray(_NODE_ACTION_MASK, dtype=bool)

    # ---- shape/static information ------------------------------------------

    @property
    def max_steps(self) -> int:
        """Three: the longest line is check, bet, call."""
        return 3

    def action_space(self, player: int) -> HybridSpace:
        return self._space

    def obs_dim(self, player: int) -> int:
        return self.num_cards + NUM_NODES + 1

    # ---- the game tree ------------------------------------------------------

    def initial_state(self, key: chex.PRNGKey) -> KuhnState:
        cards = jax.random.permutation(key, self.num_cards)[:2].astype(jnp.int32)
        return KuhnState(
            cards=cards,
            node=jnp.asarray(NODE_P0_OPEN, dtype=jnp.int32),
            bet=jnp.zeros((), dtype=jnp.float32),
            payoff=jnp.zeros((), dtype=jnp.float32),
        )

    def current_player(self, state: KuhnState) -> chex.Array:
        return self._node_player[state.node]

    def observation(self, player: int, state: KuhnState) -> chex.Array:
        """`(num_cards + NUM_NODES + 1,)`: own card, public node, outstanding bet.

        Only `player`'s own card goes in -- the opponent's is in `state.cards`
        but never observed, which is the whole imperfect-information content of
        the game. `node` is public (it is the betting history), and so is the
        outstanding bet, scaled by `max_bet` to keep the feature O(1).
        """
        card = jax.nn.one_hot(state.cards[player], self.num_cards)
        node = jax.nn.one_hot(state.node, NUM_NODES)
        return jnp.concatenate([card, node, (state.bet / self.max_bet)[None]])

    def action_mask(self, player: int, state: KuhnState) -> chex.Array:
        """`(3,)` bool over `(KIND_PASSIVE, KIND_CALL, KIND_BET)`; depends only on the node."""
        del player  # legality here is public: it is a function of the betting history alone
        return self._node_action_mask[state.node]

    def payoff(self, state: KuhnState) -> chex.Array:
        return state.payoff

    def _step(self, state: KuhnState, action: HybridAction, key: chex.PRNGKey) -> KuhnState:
        del key  # the only chance move is the deal, already resolved in `initial_state`
        clipped = self._space.clip(action)
        size = jnp.squeeze(clipped.value, axis=-1)
        # `KIND_CALL` is masked out where a bet would be legal and vice versa, so
        # the two aggressive kinds never compete: whichever is legal at this node
        # is the one this branch means.
        aggressive = clipped.kind != KIND_PASSIVE

        node = state.node
        at_open = node == NODE_P0_OPEN
        at_check = node == NODE_P1_AFTER_CHECK
        facing_bet = (node == NODE_P1_AFTER_BET) | (node == NODE_P0_AFTER_CHECK_BET)
        opens_bet = (at_open | at_check) & aggressive

        next_node = jnp.where(
            at_open,
            jnp.where(aggressive, NODE_P1_AFTER_BET, NODE_P1_AFTER_CHECK),
            jnp.where(at_check & aggressive, NODE_P0_AFTER_CHECK_BET, NODE_TERMINAL),
        ).astype(jnp.int32)
        next_bet = jnp.where(opens_bet, size, state.bet).astype(jnp.float32)

        # Payoffs are in ante units: each player has 1 in before any bet, so a
        # fold hands over one ante and a called bet of `b` is worth `1 + b`.
        showdown = jnp.where(state.cards[0] > state.cards[1], 1.0, -1.0)
        called = showdown * (1.0 + state.bet)
        folded = jnp.where(node == NODE_P1_AFTER_BET, 1.0, -1.0)  # whoever faces the bet is the folder
        payoff = jnp.where(
            at_check & ~aggressive,
            showdown,  # check-check: showdown for the antes alone
            jnp.where(facing_bet, jnp.where(aggressive, called, folded), 0.0),
        )
        payoff = jnp.where(next_node == NODE_TERMINAL, payoff, 0.0).astype(jnp.float32)

        return KuhnState(cards=state.cards, node=next_node, bet=next_bet, payoff=payoff)

    # ---- infoset enumeration (logging, and tabular best responses) ----------

    def decision_nodes(self, player: int) -> tuple[int, int]:
        """The `NODE_*` constants at which `player` acts."""
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player}")
        return _PLAYER_NODES[player]

    def infoset_observation(self, card: int, node: int, bet: float = 0.0) -> chex.Array:
        """The observation of one infoset, built directly rather than reached by play.

        Lets a caller sweep every (card, node) pair -- and, at the two nodes
        that face a bet, a grid over `bet` -- to print or best-respond to a
        policy's full behavioral strategy without having to roll out to each
        node. The layout matches `observation` exactly.
        """
        one_hot_card = jax.nn.one_hot(jnp.asarray(card), self.num_cards)
        one_hot_node = jax.nn.one_hot(jnp.asarray(node), NUM_NODES)
        scaled_bet = jnp.asarray(bet, dtype=jnp.float32)[None] / self.max_bet
        return jnp.concatenate([one_hot_card, one_hot_node, scaled_bet])

    def infoset_action_mask(self, node: int) -> chex.Array:
        """The action mask of one infoset, keyed by node -- the `infoset_observation` counterpart."""
        return self._node_action_mask[node]
