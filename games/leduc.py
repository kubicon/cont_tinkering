"""Leduc Hold'em with a *continuously sized* raise.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .sequential import TERMINAL, SequentialZeroSumGame
from .spaces import HybridAction, HybridSpace, hybrid


KIND_PASSIVE = 0  # check when no bet is live, fold when facing one
KIND_CALL = 1  # match the outstanding bet
KIND_RAISE = 2  # bet/raise by `value`; == NUM_ATOMS, i.e. `HybridSpace.continuous_kind`
NUM_ATOMS = 2

# Leduc is a two-round game by definition: one round, the board card, one more.
NUM_ROUNDS = 2

# Chips each player puts in before the deal. Payoffs are quoted in these units,
# so an uncontested fold on the first action is worth exactly `ANTE`.
ANTE = 1.0

# Both betting rounds are opened by player 0, as in the standard two-player game.
FIRST_TO_ACT = 0


@chex.dataclass(frozen=True)
class LeducState:
    """Fixed-shape state of one `ContinuousLeducHoldem` hand.
    """

    cards: chex.Array  # (2,) int32, each player's private card *rank*
    board: chex.Array  # () int32, the public card's rank (hidden while `round == 0`)
    round: chex.Array  # () int32, 0 (pre-board) or 1 (post-board)
    contrib: chex.Array  # (2,) float32, chips each player has put in, antes included
    to_act: chex.Array  # () int32, 0, 1, or `TERMINAL`
    raises: chex.Array  # () int32, aggressive actions taken *this round*
    actions: chex.Array  # () int32, decisions taken *this round*
    payoff: chex.Array  # () float32, player 0's payoff; only final once terminal


class ContinuousLeducHoldem(SequentialZeroSumGame):
    """Leduc Hold'em in which the size of a bet or raise is a continuous choice.
    """

    def __init__(
        self,
        num_ranks: int = 3,
        num_suits: int = 2,
        min_bet: float = 0.5,
        max_bet: float = 2.0,
        max_raises: int = 2,
        second_round_scale: float = 1.0,
    ):
        if num_ranks < 2:
            raise ValueError(f"num_ranks must be at least 2, got {num_ranks}")
        if num_suits < 1:
            raise ValueError(f"num_suits must be at least 1, got {num_suits}")
        if num_ranks * num_suits < 3:
            raise ValueError(
                "the deck must hold at least 3 cards (two private and one board), got "
                f"{num_ranks * num_suits}"
            )
        if not 0.0 < min_bet <= max_bet:
            raise ValueError(f"need 0 < min_bet <= max_bet, got {min_bet} and {max_bet}")
        if max_raises < 0:
            raise ValueError(f"max_raises must be non-negative, got {max_raises}")
        if second_round_scale <= 0.0:
            raise ValueError(f"second_round_scale must be positive, got {second_round_scale}")

        self.num_ranks = num_ranks
        self.num_suits = num_suits
        self.min_bet = float(min_bet)
        self.max_bet = float(max_bet)
        self.max_raises = int(max_raises)
        self.second_round_scale = float(second_round_scale)

        self._space = hybrid(NUM_ATOMS, [min_bet], [max_bet])
        self._round_scale = jnp.asarray([1.0, self.second_round_scale], dtype=jnp.float32)
        # The most either player can ever have in: the ante plus, in each round,
        # `max_raises` raises of the largest size that round allows. Used only to
        # keep the pot features in the observation O(1).
        self._max_contrib = ANTE + self.max_raises * self.max_bet * (1.0 + self.second_round_scale)

    # ---- shape/static information ------------------------------------------

    @property
    def max_steps(self) -> int:
        """`2 * (max_raises + 2)`: the longest round is check, bet, raise..., call."""
        return NUM_ROUNDS * (self.max_raises + 2)

    def action_space(self, player: int) -> HybridSpace:
        return self._space

    def obs_dim(self, player: int) -> int:
        return 2 * self.num_ranks + NUM_ROUNDS + 5

    # ---- the game tree ------------------------------------------------------

    def initial_state(self, key: chex.PRNGKey) -> LeducState:
        """Deal two private cards and the board from one deck, without replacement.
        """
        deck = jax.random.permutation(key, self.num_ranks * self.num_suits)[:3]
        ranks = (deck // self.num_suits).astype(jnp.int32)
        return LeducState(
            cards=ranks[:2],
            board=ranks[2],
            round=jnp.zeros((), dtype=jnp.int32),
            contrib=jnp.full((2,), ANTE, dtype=jnp.float32),
            to_act=jnp.asarray(FIRST_TO_ACT, dtype=jnp.int32),
            raises=jnp.zeros((), dtype=jnp.int32),
            actions=jnp.zeros((), dtype=jnp.int32),
            payoff=jnp.zeros((), dtype=jnp.float32),
        )

    def current_player(self, state: LeducState) -> chex.Array:
        return state.to_act

    def observation(self, player: int, state: LeducState) -> chex.Array:
        """`player`'s infoset: own card, the board once it is out, and the betting state.
        """
        opponent = 1 - player
        return self._encode_observation(
            card=state.cards[player],
            board=state.board,
            round_index=state.round,
            own_contrib=state.contrib[player],
            opponent_contrib=state.contrib[opponent],
            raises=state.raises,
            actions=state.actions,
        )

    def action_mask(self, player: int, state: LeducState) -> chex.Array:
        """`(3,)` bool over `(KIND_PASSIVE, KIND_CALL, KIND_RAISE)`.

        """
        opponent = 1 - player
        facing_bet = state.contrib[opponent] > state.contrib[player]
        may_raise = state.raises < self.max_raises
        mask = jnp.stack([jnp.asarray(True), facing_bet, may_raise])
        # All-`True` on terminal states keeps the masked softmax well posed; see
        # `SequentialZeroSumGame.action_mask`. Nothing is sampled there.
        return mask | (state.to_act == TERMINAL)

    def payoff(self, state: LeducState) -> chex.Array:
        return state.payoff

    def _step(self, state: LeducState, action: HybridAction, key: chex.PRNGKey) -> LeducState:
        del key  # the only chance move is the deal, already resolved in `initial_state`
        clipped = self._space.clip(action)
        raise_size = jnp.squeeze(clipped.value, axis=-1) * self._round_scale[state.round]

        actor = state.to_act
        opponent = 1 - actor
        own = state.contrib[actor]
        outstanding = state.contrib[opponent]

        passive = clipped.kind == KIND_PASSIVE
        aggressive = clipped.kind == KIND_RAISE
        facing_bet = outstanding > own
        folds = passive & facing_bet

        # A raise matches the outstanding amount and adds `raise_size` on top; a
        # call matches it; a check (or a fold) adds nothing. With nothing
        # outstanding `outstanding == own`, so the same expression is an opening
        # bet -- and `KIND_CALL` is masked out there, so it never means "check".
        new_own = jnp.where(aggressive, outstanding + raise_size, jnp.where(passive, own, outstanding))
        contrib = state.contrib.at[actor].set(new_own)

        raises = state.raises + aggressive.astype(jnp.int32)
        actions = state.actions + 1
        # Both players have acted and nobody is facing a bet: check-check,
        # bet-call, or bet-raise-...-call. A lone opening check leaves the
        # contributions equal too, which is what `actions >= 2` rules out.
        round_over = (new_own == outstanding) & (actions >= 2)
        hand_over = folds | (round_over & (state.round == NUM_ROUNDS - 1))

        next_round = jnp.minimum(state.round + round_over, NUM_ROUNDS - 1).astype(jnp.int32)
        next_to_act = jnp.where(
            hand_over, TERMINAL, jnp.where(round_over, FIRST_TO_ACT, opponent)
        ).astype(jnp.int32)
        next_raises = jnp.where(round_over, 0, raises).astype(jnp.int32)
        next_actions = jnp.where(round_over, 0, actions).astype(jnp.int32)

        # A folder forfeits what they themselves have put in; at a showdown the
        # contributions are equal by construction, so either one is the stake.
        folded = jnp.where(actor == 0, -contrib[0], contrib[1])
        called = self._showdown_sign(state) * contrib[0]
        payoff = jnp.where(folds, folded, jnp.where(hand_over, called, 0.0)).astype(jnp.float32)

        return LeducState(
            cards=state.cards,
            board=state.board,
            round=next_round,
            contrib=contrib,
            to_act=next_to_act,
            raises=next_raises,
            actions=next_actions,
            payoff=payoff,
        )

    # ---- helpers -------------------------------------------------------------

    def _showdown_sign(self, state: LeducState) -> chex.Array:
        """`+1` if player 0 wins the showdown, `-1` if player 1 does, `0` on a split.

        Pairing the board is worth more than any unpaired hand, so a paired rank
        is ranked `num_ranks` above every unpaired one; two players pairing the
        same board card (possible once `num_suits >= 3`) tie, as do two equal
        unpaired ranks.
        """
        strength = state.cards + self.num_ranks * (state.cards == state.board)
        return jnp.sign(strength[0] - strength[1]).astype(jnp.float32)

    def _encode_observation(
        self,
        card: chex.Array,
        board: chex.Array,
        round_index: chex.Array,
        own_contrib: chex.Array,
        opponent_contrib: chex.Array,
        raises: chex.Array,
        actions: chex.Array,
    ) -> chex.Array:
        """The one place the observation layout is defined; see `obs_dim` for its width.

        """
        revealed = (round_index == NUM_ROUNDS - 1).astype(jnp.float32)
        scalars = jnp.stack(
            [
                own_contrib / self._max_contrib,
                opponent_contrib / self._max_contrib,
                raises / max(self.max_raises, 1),
                actions / (self.max_raises + 2),
                (opponent_contrib > own_contrib).astype(jnp.float32),
            ]
        )
        return jnp.concatenate(
            [
                jax.nn.one_hot(card, self.num_ranks),
                jax.nn.one_hot(board, self.num_ranks) * revealed,
                jax.nn.one_hot(round_index, NUM_ROUNDS),
                scalars,
            ]
        ).astype(jnp.float32)

    # ---- infoset construction (logging, and probing a trained policy) --------

    def infoset_observation(
        self,
        card: int,
        board: int = 0,
        round_index: int = 0,
        own_contrib: float = ANTE,
        opponent_contrib: float = ANTE,
        raises: int = 0,
        actions: int = 0,
    ) -> chex.Array:
        """One infoset's observation, built directly rather than reached by play.

        """
        return self._encode_observation(
            card=jnp.asarray(card, dtype=jnp.int32),
            board=jnp.asarray(board, dtype=jnp.int32),
            round_index=jnp.asarray(round_index, dtype=jnp.int32),
            own_contrib=jnp.asarray(own_contrib, dtype=jnp.float32),
            opponent_contrib=jnp.asarray(opponent_contrib, dtype=jnp.float32),
            raises=jnp.asarray(raises, dtype=jnp.int32),
            actions=jnp.asarray(actions, dtype=jnp.int32),
        )

    def infoset_action_mask(self, facing_bet: bool, raises: int = 0) -> chex.Array:
        """The action mask of one infoset -- the `infoset_observation` counterpart."""
        return jnp.asarray(
            [True, bool(facing_bet), int(raises) < self.max_raises], dtype=bool
        )
