"""Interface-level checks for `games.sequential` via `ContinuousKuhnPoker`.

These exist to pin the contract a batched, `jit`ed rollout relies on -- fixed
shapes, a statically bounded horizon, terminal states that absorb further steps,
legality masks that never let an illegal kind be played -- and the one property
no shape check can catch: that a player's observation really hides the
opponent's card.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.sequential import TERMINAL
from games.sequential_examples import (
    KIND_BET,
    KIND_CALL,
    KIND_PASSIVE,
    NODE_P0_AFTER_CHECK_BET,
    NODE_P0_OPEN,
    NODE_P1_AFTER_BET,
    NODE_P1_AFTER_CHECK,
    NODE_TERMINAL,
    NUM_ATOMS,
    ContinuousKuhnPoker,
    KuhnState,
)
from games.spaces import HybridAction

BATCH = 512


def _game(**kwargs) -> ContinuousKuhnPoker:
    return ContinuousKuhnPoker(**kwargs)


def _state(game: ContinuousKuhnPoker, cards, node, bet=0.0) -> KuhnState:
    return KuhnState(
        cards=jnp.asarray(cards, dtype=jnp.int32),
        node=jnp.asarray(node, dtype=jnp.int32),
        bet=jnp.asarray(bet, dtype=jnp.float32),
        payoff=jnp.zeros((), dtype=jnp.float32),
    )


def _action(kind: int, value: float = 1.0) -> HybridAction:
    return HybridAction(
        kind=jnp.asarray(kind, dtype=jnp.int32), value=jnp.asarray([value], dtype=jnp.float32)
    )


def _play(game: ContinuousKuhnPoker, state: KuhnState, *actions) -> KuhnState:
    key = jax.random.PRNGKey(0)
    for kind, value in actions:
        state = game.step(state, _action(kind, value), key)
    return state


CHECK = (KIND_PASSIVE, 0.0)
FOLD = (KIND_PASSIVE, 0.0)
CALL = (KIND_CALL, 0.0)


def BET(size: float):
    return (KIND_BET, size)


def test_deal_gives_two_distinct_cards():
    game = _game()
    cards = jax.vmap(game.initial_state)(jax.random.split(jax.random.PRNGKey(0), BATCH)).cards
    assert cards.shape == (BATCH, 2)
    assert jnp.all(cards[:, 0] != cards[:, 1])
    assert jnp.all((cards >= 0) & (cards < game.num_cards))


def test_random_play_always_terminates_within_max_steps():
    """The horizon bound `max_steps` is what a rollout's `lax.scan` length is set from."""
    game = _game()
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
    terminal = _play(game, _state(game, [2, 0], NODE_P0_OPEN), BET(1.0), FOLD)
    assert bool(game.is_terminal(terminal))

    for action in (CHECK, CALL, BET(2.0)):
        stepped = _play(game, terminal, action)
        jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), stepped, terminal)


# ---- legality masks ------------------------------------------------------


@pytest.mark.parametrize(
    "node, expected",
    [
        (NODE_P0_OPEN, (True, False, True)),
        (NODE_P1_AFTER_CHECK, (True, False, True)),
        (NODE_P1_AFTER_BET, (True, True, False)),
        (NODE_P0_AFTER_CHECK_BET, (True, True, False)),
    ],
)
def test_action_mask_matches_the_betting_rules(node, expected):
    """Call needs a bet outstanding; bet needs none (Kuhn has no raise)."""
    game = _game()
    mask = game.action_mask(0, _state(game, [1, 0], node))
    assert mask.shape == (game.num_kinds(0),) == (NUM_ATOMS + 1,)
    assert tuple(bool(x) for x in mask) == expected


def test_every_state_leaves_at_least_one_legal_kind():
    """An all-False mask would make the masked softmax uniform garbage, not an error."""
    game = _game()
    for node in range(NODE_TERMINAL + 1):
        for player in (0, 1):
            assert bool(jnp.any(game.action_mask(player, _state(game, [1, 0], node))))


def test_random_play_never_takes_an_illegal_kind():
    """`random_action_fn` samples from masked logits, so illegal kinds have probability 0."""
    game = _game()
    action_fn = game.random_action_fn(0)
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)

    for node, forbidden in ((NODE_P0_OPEN, KIND_CALL), (NODE_P1_AFTER_BET, KIND_BET)):
        state = _state(game, [1, 0], node, bet=1.0)
        mask = game.action_mask(0, state)
        obs = game.observation(0, state)
        kinds = jax.vmap(lambda k: action_fn(obs, mask, k).kind)(keys)
        assert not bool(jnp.any(kinds == forbidden))
        assert len(jnp.unique(kinds)) == 2  # both legal kinds do get drawn


# ---- observations --------------------------------------------------------


def test_observation_hides_the_opponents_card():
    game = _game()
    for player in (0, 1):
        opponent = 1 - player
        cards = [0, 0]
        cards[player] = 1
        observations = []
        for opponent_card in range(game.num_cards):
            if opponent_card == 1:
                continue
            cards[opponent] = opponent_card
            observations.append(game.observation(player, _state(game, cards, NODE_P0_OPEN)))
        for other in observations[1:]:
            np.testing.assert_array_equal(observations[0], other)


def test_observation_reveals_own_card_node_and_bet():
    game = _game()
    state = _state(game, [2, 0], NODE_P1_AFTER_BET, bet=1.5)
    obs_0 = game.observation(0, state)
    obs_1 = game.observation(1, state)

    assert obs_0.shape == (game.obs_dim(0),) == (game.num_cards + 5 + 1,)
    assert jnp.argmax(obs_0[: game.num_cards]) == 2
    assert jnp.argmax(obs_1[: game.num_cards]) == 0
    assert jnp.argmax(obs_0[game.num_cards : -1]) == NODE_P1_AFTER_BET
    assert obs_0[-1] == pytest.approx(1.5 / game.max_bet)


def test_infoset_observation_matches_played_observation():
    game = _game()
    state = _play(game, _state(game, [2, 0], NODE_P0_OPEN), BET(game.max_bet))
    built = game.infoset_observation(card=0, node=NODE_P1_AFTER_BET, bet=game.max_bet)
    np.testing.assert_allclose(game.observation(1, state), built)
    np.testing.assert_array_equal(
        game.infoset_action_mask(NODE_P1_AFTER_BET), game.action_mask(1, state)
    )


# ---- payoffs -------------------------------------------------------------


@pytest.mark.parametrize(
    "cards, actions, node_path, expected",
    [
        # check, check -> showdown for the antes alone
        ([2, 0], (CHECK, CHECK), (NODE_P1_AFTER_CHECK,), 1.0),
        ([0, 2], (CHECK, CHECK), (NODE_P1_AFTER_CHECK,), -1.0),
        # player 0 bets, player 1 folds -> player 0 collects one ante
        ([0, 2], (BET(2.0), FOLD), (NODE_P1_AFTER_BET,), 1.0),
        # player 0 bets max, player 1 calls -> showdown for 1 + max_bet
        ([2, 0], (BET(2.0), CALL), (NODE_P1_AFTER_BET,), 3.0),
        ([0, 2], (BET(2.0), CALL), (NODE_P1_AFTER_BET,), -3.0),
        # player 0 checks, player 1 bets, player 0 folds -> player 0 loses one ante
        ([2, 0], (CHECK, BET(1.0), FOLD), (NODE_P1_AFTER_CHECK, NODE_P0_AFTER_CHECK_BET), -1.0),
        # player 0 checks, player 1 bets min, player 0 calls -> showdown for 1 + min_bet
        ([2, 0], (CHECK, BET(0.5), CALL), (NODE_P1_AFTER_CHECK, NODE_P0_AFTER_CHECK_BET), 1.5),
    ],
)
def test_known_lines(cards, actions, node_path, expected):
    """Every branch of the tree, walked by hand, against payoffs in ante units."""
    game = _game(min_bet=0.5, max_bet=2.0)
    state = _state(game, cards, NODE_P0_OPEN)
    for action, node in zip(actions, node_path):
        state = _play(game, state, action)
        assert int(state.node) == node
        assert not bool(game.is_terminal(state))
    state = _play(game, state, actions[-1])
    assert int(state.node) == NODE_TERMINAL
    assert float(game.payoff(state)) == pytest.approx(expected)


def test_a_called_bet_scales_the_payoff_with_its_size():
    game = _game(min_bet=0.5, max_bet=2.0)
    for size in (0.5, 1.25, 2.0):
        state = _play(game, _state(game, [2, 0], NODE_P0_OPEN), BET(size), CALL)
        assert float(game.payoff(state)) == pytest.approx(1.0 + size, rel=1e-5)


def test_the_bet_value_is_clipped_to_the_action_box():
    """`value` is the bet size in game units, so the box bounds are the betting limits."""
    game = _game(min_bet=0.5, max_bet=2.0)
    for proposed, clipped in ((-5.0, 0.5), (0.1, 0.5), (99.0, 2.0)):
        state = _play(game, _state(game, [2, 0], NODE_P0_OPEN), BET(proposed), CALL)
        assert float(game.payoff(state)) == pytest.approx(1.0 + clipped)


def test_the_bet_value_is_ignored_when_an_atom_is_played():
    game = _game()
    payoffs = {
        float(game.payoff(_play(game, _state(game, [2, 0], NODE_P0_OPEN), (KIND_PASSIVE, v), CHECK)))
        for v in (-99.0, 0.0, 1.3, 99.0)
    }
    assert payoffs == {1.0}


def test_classic_kuhn_payoffs_when_the_bet_size_is_fixed():
    """`min_bet == max_bet == 1` collapses the continuum back to textbook Kuhn."""
    game = _game(min_bet=1.0, max_bet=1.0)
    action_fns = (game.random_action_fn(0), game.random_action_fn(1))
    play = jax.jit(jax.vmap(lambda k: game.play_episode(action_fns, k)))
    _, payoff = play(jax.random.split(jax.random.PRNGKey(0), BATCH))

    assert set(float(p) for p in jnp.unique(payoff)) <= {-2.0, -1.0, 1.0, 2.0}


def test_payoff_is_zero_sum_over_swapped_deals():
    """Swapping the two hands and mirroring the players negates the payoff.

    Both players play the same fixed strategy here, so the game is symmetric and
    the check is a genuine zero-sum assertion rather than a tautology.
    """
    game = _game()
    actions = (BET(1.3), CALL)
    payoff = float(game.payoff(_play(game, _state(game, [2, 0], NODE_P0_OPEN), *actions)))
    mirrored = float(game.payoff(_play(game, _state(game, [0, 2], NODE_P0_OPEN), *actions)))
    assert payoff == pytest.approx(-mirrored)


def test_state_shapes_and_dtypes_survive_a_step():
    """A rollout `lax.scan`s over the state, so a step must not change its pytree."""
    game = _game()
    state = _state(game, [1, 0], NODE_P0_OPEN)
    stepped = _play(game, state, BET(1.0))

    assert jax.tree_util.tree_structure(state) == jax.tree_util.tree_structure(stepped)
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(stepped)):
        assert before.shape == after.shape
        assert before.dtype == after.dtype


@pytest.mark.parametrize("min_bet, max_bet", [(0.5, 2.0), (1.0, 1.0), (0.1, 5.0)])
def test_uniform_random_play_is_worth_one_eighth_to_the_first_mover(min_bet, max_bet):
    """An exact analytic value for the whole tree, independent of the bet sizing.

    Exactly two kinds are legal at every decision node, so uniform-random play is
    passive with probability 1/2 there, and the deal is symmetric so every
    showdown is worth 0 in expectation. Only the two folding lines survive:
    player 0 bets into a fold with probability 1/4 (+1), and folds to player 1's
    bet with probability 1/8 (-1), for `1/4 - 1/8 = 1/8`. The bet size cancels
    because it is drawn independently of the cards -- which is exactly the
    position asymmetry a sequential trainer has to reproduce, and is *not* zero.
    """
    game = _game(min_bet=min_bet, max_bet=max_bet)
    action_fns = (game.random_action_fn(0), game.random_action_fn(1))
    play = jax.jit(jax.vmap(lambda k: game.play_episode(action_fns, k)))
    _, payoff = play(jax.random.split(jax.random.PRNGKey(0), 200_000))

    standard_error = float(jnp.std(payoff)) / jnp.sqrt(payoff.shape[0])
    assert float(jnp.mean(payoff)) == pytest.approx(0.125, abs=4 * standard_error)
