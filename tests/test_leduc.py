"""Rules-level checks for `games.leduc.ContinuousLeducHoldem`.

Split in two: the `games.sequential` contract a batched, `jit`ed rollout relies
on (fixed shapes, a bounded horizon, absorbing terminal states, masks that never
admit an illegal kind, observations that hide the opponent's card *and* the board
until it is turned), and the poker itself -- pot arithmetic over two rounds,
raises, pairing the board -- walked by hand line by line.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.leduc import (
    ANTE,
    KIND_CALL,
    KIND_PASSIVE,
    KIND_RAISE,
    NUM_ATOMS,
    ContinuousLeducHoldem,
    LeducState,
)
from games.sequential import TERMINAL
from games.spaces import HybridAction

BATCH = 512


def _game(**kwargs) -> ContinuousLeducHoldem:
    return ContinuousLeducHoldem(**kwargs)


def _state(cards, board, round_index=0, contrib=(ANTE, ANTE), to_act=0, raises=0, actions=0) -> LeducState:
    return LeducState(
        cards=jnp.asarray(cards, dtype=jnp.int32),
        board=jnp.asarray(board, dtype=jnp.int32),
        round=jnp.asarray(round_index, dtype=jnp.int32),
        contrib=jnp.asarray(contrib, dtype=jnp.float32),
        to_act=jnp.asarray(to_act, dtype=jnp.int32),
        raises=jnp.asarray(raises, dtype=jnp.int32),
        actions=jnp.asarray(actions, dtype=jnp.int32),
        payoff=jnp.zeros((), dtype=jnp.float32),
    )


def _action(kind: int, value: float = 1.0) -> HybridAction:
    return HybridAction(
        kind=jnp.asarray(kind, dtype=jnp.int32), value=jnp.asarray([value], dtype=jnp.float32)
    )


def _play(game: ContinuousLeducHoldem, state: LeducState, *actions) -> LeducState:
    key = jax.random.PRNGKey(0)
    for kind, value in actions:
        state = game.step(state, _action(kind, value), key)
    return state


CHECK = (KIND_PASSIVE, 0.0)
FOLD = (KIND_PASSIVE, 0.0)
CALL = (KIND_CALL, 0.0)


def RAISE(size: float):
    return (KIND_RAISE, size)


# ---- the deal ------------------------------------------------------------


def test_the_deal_takes_three_distinct_cards_from_one_deck():
    """Ranks may repeat (two suits), but never more often than the deck allows."""
    game = _game(num_ranks=3, num_suits=2)
    states = jax.vmap(game.initial_state)(jax.random.split(jax.random.PRNGKey(0), BATCH))
    ranks = jnp.concatenate([states.cards, states.board[:, None]], axis=1)

    assert ranks.shape == (BATCH, 3)
    assert jnp.all((ranks >= 0) & (ranks < game.num_ranks))
    counts = jnp.sum(jax.nn.one_hot(ranks, game.num_ranks), axis=1)
    assert jnp.all(counts <= game.num_suits)
    # Both players start with only their ante in, player 0 to act.
    assert jnp.all(states.contrib == ANTE)
    assert jnp.all(states.to_act == 0)


def test_a_single_suit_deck_never_repeats_a_rank():
    game = _game(num_ranks=4, num_suits=1)
    states = jax.vmap(game.initial_state)(jax.random.split(jax.random.PRNGKey(0), BATCH))
    ranks = jnp.concatenate([states.cards, states.board[:, None]], axis=1)
    assert jnp.all(jnp.sum(jax.nn.one_hot(ranks, game.num_ranks), axis=1) <= 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_ranks=1),
        dict(num_suits=0),
        dict(num_ranks=2, num_suits=1),  # a two-card deck cannot fill three seats
        dict(min_bet=0.0),
        dict(min_bet=2.0, max_bet=1.0),
        dict(max_raises=-1),
        dict(second_round_scale=0.0),
    ],
)
def test_impossible_rules_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _game(**kwargs)


# ---- the rollout contract ------------------------------------------------


@pytest.mark.parametrize("max_raises", [0, 1, 2, 4])
def test_random_play_always_terminates_within_max_steps(max_raises):
    """The horizon bound `max_steps` is what a rollout's `lax.scan` length is set from."""
    game = _game(max_raises=max_raises)
    action_fns = (game.random_action_fn(0), game.random_action_fn(1))
    play = jax.jit(jax.vmap(lambda k: game.play_episode(action_fns, k)))
    final_state, payoff = play(jax.random.split(jax.random.PRNGKey(0), BATCH))

    assert jnp.all(game.is_terminal(final_state))
    assert jnp.all(game.current_player(final_state) == TERMINAL)
    assert payoff.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(payoff))


def test_stepping_a_terminal_state_is_a_noop():
    """`step`'s guard is what lets finished episodes ride out the rest of the scan."""
    game = _game()
    terminal = _play(game, _state([2, 0], 1), RAISE(1.0), FOLD)
    assert bool(game.is_terminal(terminal))

    for action in (CHECK, CALL, RAISE(2.0)):
        stepped = _play(game, terminal, action)
        jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), stepped, terminal)


def test_state_shapes_and_dtypes_survive_a_step():
    """A rollout `lax.scan`s over the state, so a step must not change its pytree."""
    game = _game()
    state = _state([1, 0], 2)
    stepped = _play(game, state, RAISE(1.0))

    assert jax.tree_util.tree_structure(state) == jax.tree_util.tree_structure(stepped)
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        assert before.shape == after.shape
        assert before.dtype == after.dtype


# ---- legality masks ------------------------------------------------------


@pytest.mark.parametrize(
    "contrib, player, raises, expected",
    [
        # nothing outstanding: check or bet, but calling is meaningless
        ((1.0, 1.0), 0, 0, (True, False, True)),
        # facing a bet: fold, call or raise
        ((1.0, 3.0), 0, 1, (True, True, True)),
        ((3.0, 1.0), 1, 1, (True, True, True)),
        # the raise cap is spent: fold or call only
        ((1.0, 3.0), 0, 2, (True, True, False)),
        # ... and it also shuts off an opening bet
        ((1.0, 1.0), 0, 2, (True, False, False)),
    ],
)
def test_action_mask_matches_the_betting_rules(contrib, player, raises, expected):
    game = _game(max_raises=2)
    state = _state([1, 0], 2, contrib=contrib, to_act=player, raises=raises)
    mask = game.action_mask(player, state)

    assert mask.shape == (game.num_kinds(player),) == (NUM_ATOMS + 1,)
    assert tuple(bool(x) for x in mask) == expected
    assert tuple(bool(x) for x in game.infoset_action_mask(expected[1], raises)) == expected


def test_every_state_leaves_at_least_one_legal_kind():
    """An all-False mask would make the masked softmax uniform garbage, not an error."""
    game = _game(max_raises=0)  # the tightest cap there is
    for to_act in (0, 1, TERMINAL):
        for contrib in ((1.0, 1.0), (1.0, 3.0), (3.0, 1.0)):
            for player in (0, 1):
                state = _state([1, 0], 2, contrib=contrib, to_act=to_act)
                assert bool(jnp.any(game.action_mask(player, state)))


def test_random_play_never_takes_an_illegal_kind():
    """`random_action_fn` samples from masked logits, so illegal kinds have probability 0."""
    game = _game(max_raises=1)
    action_fn = game.random_action_fn(0)
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)

    cases = ((((1.0, 1.0)), 0, KIND_CALL), (((1.0, 3.0)), 1, KIND_RAISE))
    for contrib, raises, forbidden in cases:
        state = _state([1, 0], 2, contrib=contrib, raises=raises)
        mask, obs = game.action_mask(0, state), game.observation(0, state)
        kinds = jax.vmap(lambda k: action_fn(obs, mask, k).kind)(keys)
        assert not bool(jnp.any(kinds == forbidden))
        assert len(jnp.unique(kinds)) == 2  # both legal kinds do get drawn


# ---- observations --------------------------------------------------------


def test_observation_hides_the_opponents_card():
    game = _game(num_ranks=3, num_suits=2)
    for player in (0, 1):
        opponent = 1 - player
        cards = [0, 0]
        cards[player] = 1
        observations = []
        for opponent_card in range(game.num_ranks):
            cards[opponent] = opponent_card
            observations.append(game.observation(player, _state(cards, 2, round_index=1)))
        for other in observations[1:]:
            np.testing.assert_array_equal(observations[0], other)


def test_observation_hides_the_board_until_it_is_turned():
    """The board is dealt up front for `jit`'s sake; only round 1 may reveal it."""
    game = _game()
    pre = [game.observation(0, _state([1, 0], board)) for board in range(game.num_ranks)]
    for other in pre[1:]:
        np.testing.assert_array_equal(pre[0], other)

    post = [game.observation(0, _state([1, 0], board, round_index=1)) for board in range(game.num_ranks)]
    assert all(not np.array_equal(post[0], other) for other in post[1:])


def test_observation_reveals_own_card_the_board_and_the_pot():
    game = _game(num_ranks=3)
    state = _state([2, 0], 1, round_index=1, contrib=(1.0, 3.0), raises=1, actions=1)
    obs_0 = game.observation(0, state)
    obs_1 = game.observation(1, state)

    assert obs_0.shape == (game.obs_dim(0),) == (2 * game.num_ranks + 7,)
    assert jnp.argmax(obs_0[: game.num_ranks]) == 2
    assert jnp.argmax(obs_1[: game.num_ranks]) == 0
    assert jnp.argmax(obs_0[game.num_ranks : 2 * game.num_ranks]) == 1  # the board, both ways
    assert jnp.argmax(obs_1[game.num_ranks : 2 * game.num_ranks]) == 1
    # The two contributions swap with the observer, and so does "facing a bet".
    np.testing.assert_allclose(obs_0[-5:-3], obs_1[-5:-3][::-1], rtol=1e-6)
    assert float(obs_0[-1]) == 1.0
    assert float(obs_1[-1]) == 0.0


def test_infoset_observation_matches_played_observation():
    game = _game(num_ranks=3, min_bet=0.5, max_bet=2.0)
    state = _play(game, _state([2, 0], 1), RAISE(1.5))
    built = game.infoset_observation(
        card=0, board=1, round_index=0, own_contrib=ANTE, opponent_contrib=ANTE + 1.5, actions=1, raises=1
    )
    np.testing.assert_allclose(game.observation(1, state), built, rtol=1e-6)


# ---- the betting rounds --------------------------------------------------


def test_a_checked_round_hands_over_to_the_board():
    """Check-check closes round 0 without money changing hands; player 0 opens again."""
    game = _game()
    state = _play(game, _state([2, 0], 1), CHECK)
    assert int(state.to_act) == 1 and int(state.round) == 0
    state = _play(game, state, CHECK)
    assert int(state.round) == 1 and int(state.to_act) == 0
    assert not bool(game.is_terminal(state))
    np.testing.assert_allclose(state.contrib, [ANTE, ANTE])


def test_a_call_closes_the_round_and_levels_the_pot():
    game = _game(min_bet=0.5, max_bet=2.0)
    state = _play(game, _state([2, 0], 1), RAISE(1.5), CALL)
    assert int(state.round) == 1 and int(state.to_act) == 0
    np.testing.assert_allclose(state.contrib, [ANTE + 1.5, ANTE + 1.5])
    assert int(state.raises) == 0 and int(state.actions) == 0  # both reset for the new round


def test_a_raise_stacks_on_top_of_the_call():
    """`value` is what a raise adds *above* matching the outstanding bet."""
    game = _game(min_bet=0.5, max_bet=2.0, max_raises=2)
    state = _play(game, _state([2, 0], 1), RAISE(2.0), RAISE(1.0))
    np.testing.assert_allclose(state.contrib, [ANTE + 2.0, ANTE + 3.0])
    assert int(state.raises) == 2 and int(state.to_act) == 0
    assert not bool(game.action_mask(0, state)[KIND_RAISE])  # the cap is spent


def test_the_second_round_scale_multiplies_post_board_raises():
    game = _game(min_bet=2.0, max_bet=2.0, second_round_scale=2.0)
    state = _play(game, _state([2, 0], 1), RAISE(2.0), CALL)  # round 0: bets of 2
    np.testing.assert_allclose(state.contrib, [3.0, 3.0])
    state = _play(game, state, RAISE(2.0))  # round 1: the same choice is worth 4
    np.testing.assert_allclose(state.contrib, [7.0, 3.0])


def test_the_raise_value_is_clipped_to_the_action_box():
    """`value` is the raise size in game units, so the box bounds are the betting limits."""
    game = _game(min_bet=0.5, max_bet=2.0)
    for proposed, clipped in ((-5.0, 0.5), (0.1, 0.5), (99.0, 2.0)):
        state = _play(game, _state([2, 0], 1), RAISE(proposed))
        np.testing.assert_allclose(state.contrib, [ANTE + clipped, ANTE], rtol=1e-6)


def test_the_raise_value_is_ignored_when_an_atom_is_played():
    game = _game()
    payoffs = {
        float(game.payoff(_play(game, _state([2, 0], 1), (KIND_PASSIVE, v), CHECK, CHECK, CHECK)))
        for v in (-99.0, 0.0, 1.3, 99.0)
    }
    assert payoffs == {ANTE}


# ---- payoffs -------------------------------------------------------------


@pytest.mark.parametrize(
    "cards, board, actions, expected",
    [
        # every check: the antes alone go to the better hand
        ([2, 0], 1, (CHECK, CHECK, CHECK, CHECK), 1.0),
        ([0, 2], 1, (CHECK, CHECK, CHECK, CHECK), -1.0),
        # a tie splits: equal ranks, neither paired
        ([1, 1], 2, (CHECK, CHECK, CHECK, CHECK), 0.0),
        # player 1 folds to the opening bet -> they forfeit their ante only
        ([0, 2], 1, (RAISE(2.0), FOLD), 1.0),
        # ... and a fold after the board costs the folder their whole stake
        ([0, 2], 1, (RAISE(2.0), CALL, CHECK, RAISE(1.0), FOLD), -3.0),
        # a called bet in each round: the showdown is worth the full pot leg
        ([2, 0], 1, (RAISE(2.0), CALL, RAISE(1.5), CALL), 4.5),
        ([0, 2], 1, (RAISE(2.0), CALL, RAISE(1.5), CALL), -4.5),
        # pairing the board beats a higher unpaired card
        ([0, 2], 0, (CHECK, CHECK, CHECK, CHECK), 1.0),
        ([2, 0], 0, (CHECK, CHECK, CHECK, CHECK), -1.0),
        # a raise war, called: 1 + 2 + 1 each
        ([2, 0], 1, (RAISE(2.0), RAISE(1.0), CALL, CHECK, CHECK), 4.0),
    ],
)
def test_known_lines(cards, board, actions, expected):
    """Whole hands, walked by hand, against payoffs in ante units."""
    game = _game(min_bet=0.5, max_bet=2.0, max_raises=2)
    state = _state(cards, board)
    for action in actions[:-1]:
        state = _play(game, state, action)
        assert not bool(game.is_terminal(state))
    state = _play(game, state, actions[-1])

    assert bool(game.is_terminal(state))
    assert float(game.payoff(state)) == pytest.approx(expected)


def test_a_showdown_is_worth_the_whole_stake():
    game = _game(min_bet=0.5, max_bet=2.0)
    for size in (0.5, 1.25, 2.0):
        state = _play(game, _state([2, 0], 1), RAISE(size), CALL, CHECK, CHECK)
        assert float(game.payoff(state)) == pytest.approx(ANTE + size, rel=1e-5)


def test_payoff_is_zero_sum_over_swapped_deals():
    """Swapping the two hands and mirroring the players negates the payoff.

    Both players play the same fixed line here, so the game is symmetric and the
    check is a genuine zero-sum assertion rather than a tautology.
    """
    game = _game()
    line = (RAISE(1.3), CALL, RAISE(0.7), CALL)
    payoff = float(game.payoff(_play(game, _state([2, 0], 1), *line)))
    mirrored = float(game.payoff(_play(game, _state([0, 2], 1), *line)))
    assert payoff == pytest.approx(-mirrored)


def test_classic_leduc_payoffs_when_the_bet_size_is_fixed():
    """Fixed 2/4 betting collapses the continuum back to the textbook game.

    Every reachable pot leg is then an integer, so a payoff is an odd number of
    ante units (or 0 on a split): 1 + 2k pre-board and 4k more after it.
    """
    game = _game(min_bet=2.0, max_bet=2.0, max_raises=2, second_round_scale=2.0)
    action_fns = (game.random_action_fn(0), game.random_action_fn(1))
    play = jax.jit(jax.vmap(lambda k: game.play_episode(action_fns, k)))
    _, payoff = play(jax.random.split(jax.random.PRNGKey(0), 20_000))

    values = {float(p) for p in jnp.unique(payoff)}
    assert values <= {0.0} | {float(s * v) for s in (-1, 1) for v in (1, 3, 5, 7, 9, 11, 13)}
    assert max(values) == 13.0  # ante + two raises in each round, both called


def test_every_finished_hand_settles_the_pot_correctly():
    """The one accounting invariant, checked over the whole reachable tree at once.

    A fold pays the folder's own contribution to the other player; a showdown
    pays the (necessarily equal) stake to whoever has the better hand, or
    nothing on a split. Random play reaches every kind of line, so this pins the
    payoff arithmetic far more broadly than the hand-walked cases above.
    """
    game = _game(min_bet=0.5, max_bet=2.0)
    action_fns = (game.random_action_fn(0), game.random_action_fn(1))
    play = jax.jit(jax.vmap(lambda k: game.play_episode(action_fns, k)))
    state, payoff = play(jax.random.split(jax.random.PRNGKey(0), 20_000))

    contrib_0, contrib_1 = state.contrib[:, 0], state.contrib[:, 1]
    showdown = contrib_0 == contrib_1
    strength = state.cards + game.num_ranks * (state.cards == state.board[:, None])
    expected = jnp.where(
        showdown,
        jnp.sign(strength[:, 0] - strength[:, 1]) * contrib_0,
        # Otherwise somebody folded, and it is whoever put in less.
        jnp.where(contrib_0 < contrib_1, -contrib_0, contrib_1),
    )
    np.testing.assert_allclose(payoff, expected, rtol=1e-6)


def test_uniform_random_play_is_worth_almost_nothing_to_either_player():
    """Unlike Kuhn's 1/8, position is worth next to nothing under random play here.

    The deal is symmetric, so every showdown is worth 0 in expectation and only
    the folding lines can move the value -- and with two rounds, raises, and both
    of them opened by player 0, those very nearly cancel. Loose bounds: the point
    is that the value sits near 0 while individual hands are worth several antes,
    so a trainer's `value_target` drifting far from 0 is a bug, not position.
    """
    game = _game(min_bet=0.5, max_bet=2.0)
    action_fns = (game.random_action_fn(0), game.random_action_fn(1))
    play = jax.jit(jax.vmap(lambda k: game.play_episode(action_fns, k)))
    _, payoff = play(jax.random.split(jax.random.PRNGKey(0), 200_000))

    assert abs(float(jnp.mean(payoff))) < 0.05
    assert float(jnp.std(payoff)) > 1.0
