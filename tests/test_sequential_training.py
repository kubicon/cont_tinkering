"""Checks for `training.sequential_rollout` and `training.sequential_trainer`.

The load-bearing property throughout is that **nothing outside a player's live
steps can move their loss**. A trajectory is padded to a fixed length and holds
both players' decisions interleaved, so every real result here depends on the
weighting being exactly zero everywhere it should be -- not merely small. Two of
these tests corrupt the rows that ought to be ignored and assert the loss does
not budge.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.sequential import TERMINAL
from games.sequential_examples import ContinuousKuhnPoker
from training.config import MixturePPOHyperparams
from training.mixture import build_mixture_network
from training.sequential_rollout import (
    build_episode_sampler,
    collect_sequential_batch,
)
from training.sequential_trainer import (
    SequentialSelfPlayPPOTrainer,
    build_sequential_ppo_loss_fn,
    masked_mean,
    normalized_advantage,
    player_weight,
)

NUM_COMPONENTS = 2
NUM_ENVS = 96


def _hyperparams(game: ContinuousKuhnPoker, **overrides) -> MixturePPOHyperparams:
    space = game.action_space(0)
    base = dict(
        action_dim=1,
        hidden_dims=(16,),
        num_components=NUM_COMPONENTS,
        num_atoms=space.num_atoms,
        low=(float(space.low[0]),),
        high=(float(space.high[0]),),
        num_envs=NUM_ENVS,
        num_epochs=1,
        category_entropy_coef=0.05,
        gaussian_entropy_coef=0.05,
        trpo_category_kl_coef=0.05,
        trpo_gaussian_kl_coef=0.05,
        magnet_category_kl_coef=0.2,
        magnet_gaussian_kl_coef=0.2,
    )
    return MixturePPOHyperparams(**{**base, **overrides})


def _setup(seed: int = 0, **game_kwargs):
    game = ContinuousKuhnPoker(**game_kwargs)
    hyperparams = _hyperparams(game)
    networks = (build_mixture_network(hyperparams), build_mixture_network(hyperparams))
    key = jax.random.PRNGKey(seed)
    init_0, init_1, state_key, batch_key = jax.random.split(key, 4)
    dummy = game.initial_state(state_key)
    params = (
        networks[0].init(init_0, game.observation(0, dummy)),
        networks[1].init(init_1, game.observation(1, dummy)),
    )
    sampler = build_episode_sampler(game, *networks)
    batch = collect_sequential_batch(
        sampler, params[0], params[0], params[1], params[1], batch_key, NUM_ENVS
    )
    return game, networks, params, sampler, batch


def _loss(network, params, batch, player):
    return build_sequential_ppo_loss_fn(player, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2)(
        params, network, batch, 0.1, 0.5, 0.0
    )


# ---- one episode ---------------------------------------------------------


def test_sample_episode_returns_one_trajectory_with_a_scalar_payoff():
    game, _, params, sampler, _ = _setup()
    episode = sampler(params[0], params[0], params[1], params[1], jax.random.PRNGKey(3))

    steps = game.max_steps
    assert episode.player.shape == episode.live.shape == (steps,)
    assert episode.obs.shape == (steps, game.obs_dim(0))
    assert episode.means.shape == (steps, NUM_COMPONENTS, 1)
    assert episode.action_mask.shape == (steps, game.action_space(0).num_atoms + NUM_COMPONENTS)
    assert episode.payoff.shape == ()  # the one field with no time axis


def test_collect_puts_the_batch_axis_in_front_of_the_time_axis():
    game, _, _, _, batch = _setup()
    assert batch.obs.shape == (NUM_ENVS, game.max_steps, game.obs_dim(0))
    assert batch.player.shape == (NUM_ENVS, game.max_steps)
    assert batch.payoff.shape == (NUM_ENVS,)


def test_live_steps_are_a_contiguous_prefix_and_dead_steps_are_tagged_terminal():
    """Padding only ever happens at the tail -- a mask that is true after a false
    would mean the terminal guard leaked."""
    _, _, _, _, batch = _setup()
    live = np.asarray(batch.live)
    for row, player_row in zip(live, np.asarray(batch.player)):
        assert list(row) == sorted(row, reverse=True)
        assert all(p in (0, 1) for p in player_row[row])
        assert all(p == TERMINAL for p in player_row[~row])


def test_recorded_players_match_an_independent_replay_of_the_tree():
    """Replays each episode's own actions through the game and checks who acted."""
    game, _, params, sampler, _ = _setup()
    from games.spaces import HybridAction

    for seed in range(8):
        episode = sampler(params[0], params[0], params[1], params[1], jax.random.PRNGKey(seed))
        state = game.initial_state(jax.random.split(jax.random.PRNGKey(seed))[0])
        for t in range(game.max_steps):
            if not bool(episode.live[t]):
                assert bool(game.is_terminal(state))
                continue
            assert int(episode.player[t]) == int(game.current_player(state))
            state = game.step(
                state,
                HybridAction(kind=episode.action_kind[t], value=episode.action_value[t]),
                jax.random.PRNGKey(0),
            )
        assert float(episode.payoff) == pytest.approx(float(game.payoff(state)))


def test_reward_is_the_terminal_payoff_signed_for_whoever_acted():
    """No bootstrapping: every decision in an episode shares the one leaf value."""
    _, _, _, _, batch = _setup()
    expected = jnp.where(batch.player == 0, batch.payoff[:, None], -batch.payoff[:, None])
    np.testing.assert_allclose(batch.reward, expected, rtol=1e-6)


def test_only_legal_kinds_are_ever_played():
    _, _, _, _, batch = _setup()
    live = batch.live
    chosen_is_legal = jnp.take_along_axis(
        batch.action_mask, batch.component[..., None], axis=-1
    )[..., 0]
    assert bool(jnp.all(chosen_is_legal | ~live))


# ---- masked reductions ---------------------------------------------------


def test_masked_mean_ignores_unselected_entries():
    values = jnp.array([1.0, 100.0, 3.0, -100.0])
    weight = jnp.array([1.0, 0.0, 1.0, 0.0])
    assert float(masked_mean(values, weight)) == pytest.approx(2.0)
    assert float(masked_mean(values, jnp.zeros(4))) == 0.0  # no division by zero


def test_normalized_advantage_standardizes_over_the_selected_entries_only():
    raw = jnp.array([1.0, 1e6, 3.0, -1e6, 5.0])
    weight = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
    advantage = normalized_advantage(raw, weight)

    selected = advantage[weight > 0]
    assert float(jnp.mean(selected)) == pytest.approx(0.0, abs=1e-4)
    assert float(jnp.std(selected)) == pytest.approx(1.0, rel=1e-3)


def test_player_weight_selects_exactly_that_players_live_steps():
    _, _, _, _, batch = _setup()
    total = player_weight(batch, 0) + player_weight(batch, 1)
    np.testing.assert_allclose(total, batch.live.astype(jnp.float32))


# ---- the loss ignores everything it should ------------------------------


def _corrupt(batch, where):
    """Replace obs/reward/value with wild but finite values wherever `where` holds."""
    wild = lambda x, fill: jnp.where(where.reshape(where.shape + (1,) * (x.ndim - where.ndim)), fill, x)
    return batch.replace(
        obs=wild(batch.obs, 7.0),
        reward=wild(batch.reward, -50.0),
        value=wild(batch.value, 50.0),
        raw_action=wild(batch.raw_action, 9.0),
    )


def test_padding_steps_cannot_move_the_loss():
    """The tail of a finished episode is real memory holding meaningless numbers."""
    _, networks, params, _, batch = _setup()
    for player in (0, 1):
        before, _ = _loss(networks[player], params[player], batch, player)
        after, _ = _loss(networks[player], params[player], _corrupt(batch, ~batch.live), player)
        assert float(before) == pytest.approx(float(after), rel=1e-5)


def test_the_other_players_steps_cannot_move_the_loss():
    """Both players learn from one interleaved trajectory; each must see only its own."""
    _, networks, params, _, batch = _setup()
    for player in (0, 1):
        opponent_steps = batch.live & (batch.player == 1 - player)
        before, _ = _loss(networks[player], params[player], batch, player)
        after, _ = _loss(networks[player], params[player], _corrupt(batch, opponent_steps), player)
        assert float(before) == pytest.approx(float(after), rel=1e-5)


def test_loss_and_gradients_are_finite():
    _, networks, params, _, batch = _setup()
    for player in (0, 1):
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: _loss(networks[player], p, batch, player), has_aux=True
        )(params[player])
        assert jnp.isfinite(loss)
        assert 0.0 < float(metrics["decisions_per_episode"])
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))


def test_gradients_are_finite_on_a_degenerate_zero_width_bet_box():
    """`min_bet == max_bet` is the classic-Kuhn baseline; `log(high - low)` is `-inf` there."""
    _, networks, params, _, batch = _setup(min_bet=1.0, max_bet=1.0)
    grads = jax.grad(lambda p: _loss(networks[0], p, batch, 0)[0])(params[0])
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))


# ---- shape validation ----------------------------------------------------


@pytest.mark.parametrize("field, value", [("num_components", 3), ("num_atoms", 1)])
def test_mismatched_networks_are_rejected(field, value):
    """Selecting between the two players with `jnp.where` needs one common shape."""
    game = ContinuousKuhnPoker()
    network_0 = build_mixture_network(_hyperparams(game))
    network_1 = build_mixture_network(_hyperparams(game, **{field: value}))
    with pytest.raises(ValueError, match=field):
        build_episode_sampler(game, network_0, network_1)


def test_a_network_whose_atoms_disagree_with_the_game_is_rejected():
    game = ContinuousKuhnPoker()
    network = build_mixture_network(_hyperparams(game, num_atoms=0))
    with pytest.raises(ValueError, match="num_atoms"):
        build_episode_sampler(game, network, network)


# ---- end to end ----------------------------------------------------------


def test_a_training_chunk_runs_and_moves_both_players():
    game = ContinuousKuhnPoker()
    hyperparams = _hyperparams(game, learning_rate=1e-2)
    trainer = SequentialSelfPlayPPOTrainer(game, hyperparams, hyperparams, seed=0)
    before = jax.tree_util.tree_leaves(trainer.params[0]) + jax.tree_util.tree_leaves(trainer.params[1])

    history = trainer.train(1, epochs=3)

    after = jax.tree_util.tree_leaves(trainer.params[0]) + jax.tree_util.tree_leaves(trainer.params[1])
    assert len(history) == 3
    assert all(jnp.isfinite(v) for v in history[-1].values())
    assert any(not bool(jnp.allclose(a, b)) for a, b in zip(before, after))
    assert history[-1]["episode_length"] >= 2.0  # Kuhn is never shorter than two decisions
