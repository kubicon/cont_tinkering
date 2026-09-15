"""Checks for `training.vtrace` and the `advantage="vtrace"` branch of the mixture loss.

Three layers. The associative-scan recurrence against a plain Python loop; the
V-trace targets against a direct transcription of Espeholt et al.'s definition
on the player's own decisions (compacted out of the interleaved rows), plus its
two textbook limits -- Monte Carlo at `lambda = 1` on-policy, TD(0) at
`lambda = 0`; and the loss itself, which on-policy with `lambda = 1` has to
reproduce the Monte Carlo loss *exactly*, gradients included.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.disk_sumo import DiskSumo
from games.sequential import TERMINAL
from games.sequential_examples import ContinuousKuhnPoker
from training.config import MixturePPOHyperparams
from training.mixture import (
    behavior_log_probs,
    build_mixture_network,
    build_mixture_ppo_loss_fn,
    category_floor_penalty,
    hybrid_action_log_prob,
    neurd_category_loss,
    neurd_log_weight,
    sample_mixture_component,
    uniform_component_log_probs,
)
from training.run_config import run_config_from_dict
from training.sequential_rollout import build_episode_sampler, collect_sequential_batch
from training.vtrace import opponent_past_weight, reverse_linear_recurrence, vtrace

NUM_ENVS = 64


# ---- the recurrence ----------------------------------------------------------


@pytest.mark.parametrize("shape", [(9,), (4, 13), (2, 3, 16)])
def test_reverse_linear_recurrence_matches_a_loop(shape):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=shape), rng.normal(size=shape)
    expected = np.zeros(shape)
    acc = np.zeros(shape[:-1])
    for t in reversed(range(shape[-1])):
        acc = b[..., t] + a[..., t] * acc
        expected[..., t] = acc
    got = reverse_linear_recurrence(jnp.asarray(a), jnp.asarray(b))
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


# ---- the targets -------------------------------------------------------------


def _reference(own, rewards, values, log_rhos, gamma, lambda_, rho_bar, c_bar, opponent_log_rhos=None):
    """V-trace straight from the definition, on one episode's own decisions.

    With `opponent_log_rhos`, the opponent rows of each segment are part of its action.
    """
    T = len(own)
    idx = [t for t in range(T) if own[t]]
    vs, adv = np.zeros(T), np.zeros(T)
    if not idx:
        return vs, adv
    bounds = idx[1:] + [T]
    seg = [rewards[s:e].sum() for s, e in zip(idx, bounds)]
    V = [values[s] for s in idx] + [0.0]
    opponent = np.zeros(T) if opponent_log_rhos is None else opponent_log_rhos
    q = [np.exp(opponent[s + 1:e].sum()) for s, e in zip(idx, bounds)]
    ratio = [np.exp(log_rhos[s]) * q[k] for k, s in enumerate(idx)]
    rho = [min(rho_bar, r) for r in ratio]
    c = [lambda_ * min(c_bar, r) for r in ratio]
    n = len(idx)
    v = [0.0] * (n + 1)  # v[n] = 0: nothing past the last decision
    for k in reversed(range(n)):
        delta = rho[k] * (seg[k] + gamma * V[k + 1] - V[k])
        v[k] = V[k] + delta + gamma * c[k] * (v[k + 1] - V[k + 1])
    for k, s in enumerate(idx):
        vs[s] = v[k]
        adv[s] = min(rho_bar, q[k]) * (seg[k] + gamma * v[k + 1] - V[k])
    return vs, adv


def _random_inputs(seed, num_envs=5, T=17):
    rng = np.random.default_rng(seed)
    own = rng.random((num_envs, T)) < 0.5
    # Padding at the end, as a finished episode has.
    for row in own:
        row[rng.integers(T // 2, T + 1):] = False
    rewards = rng.normal(size=(num_envs, T))
    values = rng.normal(size=(num_envs, T))
    log_rhos = rng.normal(scale=0.7, size=(num_envs, T))
    return own, rewards, values, log_rhos


@pytest.mark.parametrize("gamma,lambda_,rho_bar,c_bar", [
    (1.0, 0.95, 1.0, 1.0), (0.9, 0.7, 1.0, 1.0), (1.0, 1.0, 2.0, 0.5), (0.99, 0.0, 1.0, 1.0),
])
def test_vtrace_matches_the_definition_on_interleaved_rows(gamma, lambda_, rho_bar, c_bar):
    own, rewards, values, log_rhos = _random_inputs(seed=1)
    out = vtrace(jnp.asarray(own), jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(log_rhos),
                 gamma=gamma, lambda_=lambda_, rho_bar=rho_bar, c_bar=c_bar)
    for e in range(own.shape[0]):
        vs, adv = _reference(own[e], rewards[e], values[e], log_rhos[e], gamma, lambda_, rho_bar, c_bar)
        np.testing.assert_allclose(np.asarray(out.vs[e]), vs, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(out.pg_advantage[e]), adv, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("gamma,lambda_,rho_bar,c_bar", [(1.0, 0.95, 2.0, 1.0), (0.9, 1.0, 1.0, 0.5)])
def test_opponent_future_correction_matches_the_definition(gamma, lambda_, rho_bar, c_bar):
    own, rewards, values, log_rhos = _random_inputs(seed=4)
    opponent = np.random.default_rng(5).normal(scale=0.5, size=own.shape)
    out = vtrace(jnp.asarray(own), jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(log_rhos),
                 gamma=gamma, lambda_=lambda_, rho_bar=rho_bar, c_bar=c_bar,
                 opponent_log_rhos=jnp.asarray(opponent))
    for e in range(own.shape[0]):
        # Own rows' entries must be ignored.
        masked = np.where(own[e], 0.0, opponent[e])
        vs, adv = _reference(own[e], rewards[e], values[e], log_rhos[e], gamma, lambda_, rho_bar, c_bar,
                             opponent_log_rhos=masked)
        np.testing.assert_allclose(np.asarray(out.vs[e]), vs, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(out.pg_advantage[e]), adv, rtol=1e-5, atol=1e-5)


def test_zero_opponent_ratios_change_nothing():
    own, rewards, values, log_rhos = _random_inputs(seed=6)
    args = (jnp.asarray(own), jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(log_rhos))
    plain = vtrace(*args, lambda_=0.9)
    corrected = vtrace(*args, lambda_=0.9, opponent_log_rhos=jnp.zeros(own.shape))
    np.testing.assert_allclose(np.asarray(corrected.vs), np.asarray(plain.vs), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(corrected.pg_advantage), np.asarray(plain.pg_advantage), rtol=1e-6, atol=1e-6
    )


def test_opponent_past_weight_is_the_floored_product_of_earlier_opponent_rows():
    own, _, _, _ = _random_inputs(seed=7)
    opponent = np.random.default_rng(8).normal(scale=1.5, size=own.shape)
    floor = 0.05
    got = np.asarray(opponent_past_weight(jnp.asarray(own), jnp.asarray(opponent), floor))
    for e in range(own.shape[0]):
        for t in range(own.shape[1]):
            if not own[e, t]:
                assert got[e, t] == 0.0
                continue
            past = sum(opponent[e, k] for k in range(t) if not own[e, k])
            np.testing.assert_allclose(got[e, t], max(floor, np.exp(past)), rtol=1e-5)


def test_on_policy_lambda_one_is_the_monte_carlo_return():
    own, rewards, values, _ = _random_inputs(seed=2)
    out = vtrace(jnp.asarray(own), jnp.asarray(rewards), jnp.asarray(values),
                 jnp.zeros(own.shape), gamma=1.0, lambda_=1.0)
    to_go = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
    np.testing.assert_allclose(np.asarray(out.vs), np.where(own, to_go, 0.0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(out.pg_advantage), np.where(own, to_go - values, 0.0), rtol=1e-5, atol=1e-5
    )


def test_lambda_zero_is_one_step_td():
    own = np.array([[True, False, True, False, True, False, False]])
    rewards = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]])
    values = np.array([[1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 9.0]])
    out = vtrace(jnp.asarray(own), jnp.asarray(rewards), jnp.asarray(values),
                 jnp.zeros(own.shape), gamma=0.5, lambda_=0.0)
    # R + gamma * V(next own), the opponent's rows' rewards folded into R.
    expected = [0.3 + 0.5 * 2.0, 0.7 + 0.5 * 3.0, 1.1 + 0.0]
    np.testing.assert_allclose(np.asarray(out.vs[0, [0, 2, 4]]), expected, rtol=1e-6)


def test_truncation_caps_the_importance_weights():
    own, rewards, values, _ = _random_inputs(seed=3)
    out = vtrace(jnp.asarray(own), jnp.asarray(rewards), jnp.asarray(values),
                 jnp.full(own.shape, 5.0), rho_bar=1.5)
    np.testing.assert_allclose(np.asarray(out.rho), np.where(own, 1.5, 0.0), rtol=1e-6)


# ---- the action probability ----------------------------------------------------


def test_hybrid_log_prob_sums_over_components_and_mixes_in_uniform_exploration():
    logits = jnp.array([0.3, -0.2, 0.5, 0.1])  # 2 atoms, 2 Gaussians
    mask = jnp.array([True, True, True, True])
    means = jnp.array([[0.2], [0.8]])
    scale_trils = jnp.array([[[0.1]], [[0.3]]])
    x = jnp.array([0.5])
    low, high = jnp.array([0.0]), jnp.array([1.0])
    probs = jax.nn.softmax(logits)
    normal = jax.scipy.stats.norm.pdf(0.5, means[:, 0], scale_trils[:, 0, 0])
    expected = jnp.log(jnp.sum(probs[2:] * normal))
    for component in (2, 3):  # whichever Gaussian drew it, the action's probability is the same
        got = hybrid_action_log_prob(logits, means, scale_trils, mask, component, x, 2)
        np.testing.assert_allclose(float(got), float(expected), rtol=1e-5)
    eps = 0.3
    explored = hybrid_action_log_prob(logits, means, scale_trils, mask, 2, x, 2, jnp.array(eps), low, high)
    # Exploring picks one of the three legal kinds (two atoms, the continuous
    # one) uniformly, and a continuous value uniformly on the unit box.
    mixed = (1 - eps) * jnp.exp(expected) + eps * (1 / 3) * 1.0
    np.testing.assert_allclose(float(explored), float(jnp.log(mixed)), rtol=1e-5)
    atom = hybrid_action_log_prob(logits, means, scale_trils, mask, 1, x, 2, jnp.array(eps), low, high)
    np.testing.assert_allclose(float(atom), float(jnp.log((1 - eps) * probs[1] + eps / 3)), rtol=1e-5)


def test_exploration_draws_the_categorical_uniformly_over_legal_kinds():
    logits = jnp.array([-30.0, 0.0, 0.0, 0.0])  # 2 atoms, 2 Gaussians; the policy never checks
    mask = jnp.array([True, False, True, True])  # check or bet
    means = jnp.array([[0.5], [1.5]])
    scale_trils = jnp.array([[[0.1]], [[0.1]]])
    low, high = jnp.array([0.25]), jnp.array([2.0])
    eps = 0.3
    # Uniform over {check, bet}, the bet split evenly over its two components.
    uniform = np.exp(np.asarray(uniform_component_log_probs(mask, 2)))
    np.testing.assert_allclose(uniform[[0, 2, 3]], [0.5, 0.25, 0.25], rtol=1e-6)
    components, actions = jax.vmap(
        lambda k: sample_mixture_component(
            logits, means, scale_trils, mask, 2, k, jnp.float32(eps), low, high
        )
    )(jax.random.split(jax.random.PRNGKey(0), 20000))
    components = np.asarray(components)
    assert not np.any(components == 1)
    assert abs(np.mean(components == 0) - eps / 2) < 0.015
    assert np.all(np.isfinite(np.asarray(actions)))
    check = hybrid_action_log_prob(
        logits, means, scale_trils, mask, 0, jnp.array([1.0]), 2, jnp.array(eps), low, high
    )
    np.testing.assert_allclose(float(check), np.log(eps / 2), rtol=1e-5)


# ---- the rollout's rewards -------------------------------------------------------


def _hyperparams(game, **overrides) -> MixturePPOHyperparams:
    space = game.action_space(0)
    base = dict(
        action_dim=space.box.low.shape[0],
        hidden_dims=(16,),
        num_components=2,
        num_atoms=space.num_atoms,
        low=tuple(float(v) for v in space.box.low),
        high=tuple(float(v) for v in space.box.high),
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


def _batch(game, explore_eps=(0.0, 0.0), seed=0):
    hyperparams = _hyperparams(game)
    networks = (build_mixture_network(hyperparams), build_mixture_network(hyperparams))
    init_0, init_1, state_key, batch_key = jax.random.split(jax.random.PRNGKey(seed), 4)
    dummy = game.initial_state(state_key)
    params = (
        networks[0].init(init_0, game.observation(0, dummy)),
        networks[1].init(init_1, game.observation(1, dummy)),
    )
    sampler = build_episode_sampler(game, *networks, explore_eps=explore_eps)
    batch, payoff = collect_sequential_batch(
        sampler, params[0], params[0], params[1], params[1], batch_key, NUM_ENVS
    )
    return networks, params, batch, payoff


def _sumo():
    return DiskSumo(horizon=15, shaping_weight=0.5, margin_weight=0.3)


def test_step_rewards_sum_to_the_payoff_and_the_monte_carlo_return_is_their_tail():
    game = _sumo()
    _, _, batch, payoff = _batch(game)
    step = np.asarray(batch.step_reward)
    np.testing.assert_allclose(step.sum(axis=1), np.asarray(payoff), rtol=1e-5, atol=1e-6)
    assert np.all(step[np.asarray(batch.actor) == TERMINAL] == 0.0)
    to_go = np.cumsum(step[:, ::-1], axis=1)[:, ::-1]
    actor = np.asarray(batch.actor)
    live = actor != TERMINAL
    expected = np.where(actor == 0, to_go, -to_go)
    np.testing.assert_allclose(np.asarray(batch.reward)[live], expected[live], rtol=1e-5, atol=1e-6)


def test_sumo_shaping_telescopes_into_the_return():
    """Undiscounted, the dense shaping is `w * (phi(s_T) - phi(s_0))` on top of the payoff."""
    game = _sumo()
    unshaped = DiskSumo(horizon=15, shaping_weight=0.0, margin_weight=0.3)
    policy = (game.random_action_fn(0), game.random_action_fn(1))
    for seed in range(4):
        key = jax.random.PRNGKey(seed)
        final, total = game.play_episode(policy, key)
        _, leaf = unshaped.play_episode(policy, key)
        start = game.initial_state(jax.random.split(key)[0])
        shaping = 0.5 * (game.potential(final) - game.potential(start))
        assert float(total) == pytest.approx(float(leaf + shaping), abs=1e-5)


# ---- the loss ----------------------------------------------------------------------


def _loss_fn(player, advantage, **kwargs):
    return build_mixture_ppo_loss_fn(
        player, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, advantage_estimator=advantage, **kwargs
    )


@pytest.mark.parametrize("make_game", [ContinuousKuhnPoker, _sumo])
@pytest.mark.parametrize("player", [0, 1])
def test_on_policy_lambda_one_vtrace_is_exactly_the_monte_carlo_loss(make_game, player):
    game = make_game()
    networks, params, batch, _ = _batch(game)
    args = (params[player], networks[player], batch, 0.1, 0.5, 0.0)
    mc_fn = jax.value_and_grad(_loss_fn(player, "monte_carlo"), has_aux=True)
    vt_fn = jax.value_and_grad(_loss_fn(player, "vtrace", gamma=1.0, vtrace_lambda=1.0), has_aux=True)
    (mc_loss, mc_metrics), mc_grads = mc_fn(*args)
    (vt_loss, vt_metrics), vt_grads = vt_fn(*args)
    np.testing.assert_allclose(float(vt_loss), float(mc_loss), rtol=1e-4, atol=1e-6)
    for g_mc, g_vt in zip(jax.tree_util.tree_leaves(mc_grads), jax.tree_util.tree_leaves(vt_grads)):
        np.testing.assert_allclose(np.asarray(g_vt), np.asarray(g_mc), rtol=1e-3, atol=1e-5)
    # Sampled by these very parameters: every importance ratio is one.
    assert float(vt_metrics["vtrace_rho"]) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("make_game", [ContinuousKuhnPoker, _sumo])
def test_an_exploring_batch_is_importance_weighted(make_game):
    game = make_game()
    networks, params, batch, _ = _batch(game, explore_eps=(0.3, 0.3))
    # rho <= 1 / (1 - eps) under exploration, so only a rho_bar below that clips.
    loss_fn = jax.value_and_grad(_loss_fn(0, "vtrace", vtrace_rho_bar=1.0), has_aux=True)
    (loss, metrics), grads = loss_fn(params[0], networks[0], batch, 0.1, 0.5, 0.0)
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree_util.tree_leaves(grads))
    # Exploration moves the ratio off one, and the truncation bites somewhere.
    assert float(metrics["vtrace_rho_clip_frac"]) > 0.0


def test_exploring_rollout_records_bounded_behavior_log_ratios():
    game = ContinuousKuhnPoker()
    eps = 0.3
    _, _, batch, _ = _batch(game, explore_eps=(eps, eps))
    ratio = np.asarray(batch.behavior_log_ratio)
    real = np.asarray(batch.actor) != TERMINAL
    assert np.all(np.isfinite(ratio))
    assert np.all(ratio <= -np.log1p(-eps) + 1e-5)
    assert np.all(ratio[~real] == 0.0)
    assert np.any(ratio[real] != 0.0)


@pytest.mark.parametrize("correction", ["future", "future_and_past"])
def test_opponent_correction_on_an_on_policy_batch_changes_nothing(correction):
    game = _sumo()
    networks, params, batch, _ = _batch(game)
    args = (params[1], networks[1], batch, 0.1, 0.5, 0.0)
    base, _ = _loss_fn(1, "vtrace")(*args)
    corrected, metrics = _loss_fn(1, "vtrace", vtrace_opponent_correction=correction)(*args)
    assert float(corrected) == pytest.approx(float(base), rel=1e-6)
    assert float(metrics["vtrace_opponent_rho"]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("make_game", [ContinuousKuhnPoker, _sumo])
@pytest.mark.parametrize("correction", ["future", "future_and_past"])
def test_opponent_correction_reweights_an_exploring_batch(make_game, correction):
    game = make_game()
    networks, params, batch, _ = _batch(game, explore_eps=(0.3, 0.3))
    # Player 0: in Kuhn, player 1's later moves can be explored -- a bet size, or
    # the check/bet or fold/call choice itself.
    args = (params[0], networks[0], batch, 0.1, 0.5, 0.0)
    base, _ = _loss_fn(0, "vtrace")(*args)
    fn = jax.value_and_grad(_loss_fn(0, "vtrace", vtrace_opponent_correction=correction), has_aux=True)
    (loss, metrics), grads = fn(*args)
    assert np.isfinite(float(loss)) and float(loss) != pytest.approx(float(base), rel=1e-6)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree_util.tree_leaves(grads))
    if correction == "future_and_past":
        assert 0.0 < float(metrics["opponent_past_ess_frac"]) <= 1.0 + 1e-6


def test_vtrace_ignores_whatever_sits_on_rows_that_are_not_the_players():
    """The current value and policy are evaluated on every row; only own rows may count."""
    game = _sumo()
    networks, params, batch, _ = _batch(game)
    fn = _loss_fn(0, "vtrace")
    base, _ = fn(params[0], networks[0], batch, 0.1, 0.5, 0.0)
    not_mine = (batch.actor != 0)[..., None]
    corrupted = batch.replace(obs=jnp.where(not_mine, 50.0, batch.obs))
    again, _ = fn(params[0], networks[0], corrupted, 0.1, 0.5, 0.0)
    assert float(again) == pytest.approx(float(base), rel=1e-6)


# ---- NeuRD ---------------------------------------------------------------------------


def test_neurd_moves_each_logit_by_its_gated_advantage():
    logits = jnp.array([10.0, -10.0, 0.0, 3.0])
    mask = jnp.array([True, True, True, False])  # the illegal entry takes no part
    beta = 2.0

    def gradient(advantages):
        return np.asarray(jax.grad(lambda z: neurd_category_loss(z, mask, advantages, beta)[0])(logits))

    # Centered legal logits (10, -10, 0): the first sits above beta, the second
    # below -beta. Pushing the first further up is dropped; pushing the second
    # back inside goes through in full, however small its probability.
    force = np.array([0.0, 0.5, -0.3])
    grad = gradient(jnp.array([1.0, 0.5, -0.3, 7.0]))
    np.testing.assert_allclose(grad[:3], -(force - force.mean()), atol=1e-6)
    assert grad[3] == 0.0
    # Pushing both saturated entries further out is dropped for both.
    outward = jnp.array([1.0, -0.5, 0.0, 7.0])
    np.testing.assert_allclose(gradient(outward), 0.0, atol=1e-6)
    _, gated = neurd_category_loss(logits, mask, outward, beta)
    assert float(gated) == pytest.approx(2 / 3)


def test_neurd_weights_a_bet_by_the_density_of_its_own_size():
    eps, pi_k, u_k = 0.2, 0.5, 0.25
    low, high = jnp.array([0.0]), jnp.array([1.0])  # uniform density 1 on the box
    # A size near the component's mean, and one exploration drew deep in its tail.
    for x, density in ((jnp.array([0.5]), 1.6), (jnp.array([0.95]), 1e-3)):
        log_density = jnp.log(density)
        category, joint = behavior_log_probs(
            jnp.log(pi_k), log_density, jnp.log(u_k), x, jnp.array(eps), low, high
        )
        weight = float(jnp.exp(neurd_log_weight(jnp.array(1.0), log_density, category, joint)))
        assert weight == pytest.approx(density / ((1 - eps) * pi_k * density + eps * u_k), rel=1e-4)
    assert weight < 0.05  # the tail draw barely counts towards the component's advantage
    # An atom is weighted by the probability it was drawn with alone.
    category, joint = behavior_log_probs(
        jnp.log(pi_k), jnp.array(0.0), jnp.log(u_k), low, jnp.array(eps), low, high
    )
    atom_weight = float(jnp.exp(neurd_log_weight(jnp.array(0.0), jnp.array(0.0), category, joint)))
    assert atom_weight == pytest.approx(1 / ((1 - eps) * pi_k + eps * u_k), rel=1e-4)


@pytest.mark.parametrize(
    "make_game, advantage, explore_eps",
    [(ContinuousKuhnPoker, "vtrace", 0.3), (ContinuousKuhnPoker, "monte_carlo", 0.0), (_sumo, "vtrace", 0.3)],
)
def test_neurd_loss_trains_on_and_off_policy(make_game, advantage, explore_eps):
    game = make_game()
    networks, params, batch, _ = _batch(game, explore_eps=(explore_eps, explore_eps))
    args = (params[0], networks[0], batch, 0.1, 0.5, 0.0)
    fn = jax.value_and_grad(_loss_fn(0, advantage, category_update="neurd"), has_aux=True)
    (loss, metrics), grads = fn(*args)
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree_util.tree_leaves(grads))
    assert 0.0 <= float(metrics["neurd_gated_frac"]) <= 1.0
    ppo_loss, _ = _loss_fn(0, advantage)(*args)
    assert float(loss) != pytest.approx(float(ppo_loss), rel=1e-6)


def test_category_floor_is_zero_above_the_floor_and_pushes_a_rare_entry_up():
    mask = jnp.ones(4, dtype=bool)
    logits = jnp.array([0.0, 2.0, -8.0, 1.0])  # entry 2 sits near 3e-5, the rest well above 1e-3
    probs = np.asarray(jax.nn.softmax(logits))
    floor = 1e-3
    penalty, frac = category_floor_penalty(logits, mask, 1, floor, "entry")
    assert float(penalty) == pytest.approx(np.log(floor) - np.log(probs[2]), rel=1e-4)
    assert float(frac) == pytest.approx(0.25)
    # `pi_k - 1` on the rare entry however small it is; the rest give up mass in proportion.
    grad = np.asarray(jax.grad(lambda z: category_floor_penalty(z, mask, 1, floor, "entry")[0])(logits))
    np.testing.assert_allclose(grad, probs - np.eye(4)[2], rtol=1e-4, atol=1e-6)
    assert float(category_floor_penalty(jnp.zeros(4), mask, 1, floor, "entry")[0]) == 0.0


def test_category_floor_by_kind_sums_the_components_of_a_bet():
    # One atom and two components, each component below the floor but the bet above it.
    mask = jnp.ones(3, dtype=bool)
    logits = jnp.log(jnp.array([0.9986, 0.0007, 0.0007]))
    assert float(category_floor_penalty(logits, mask, 1, 1e-3, "kind")[0]) == 0.0
    assert float(category_floor_penalty(logits, mask, 1, 1e-3, "entry")[0]) > 0.0
    # An illegal entry never counts, however low its probability.
    penalty, frac = category_floor_penalty(logits, jnp.array([False, True, True]), 1, 1e-3, "kind")
    assert float(penalty) == 0.0 and float(frac) == 0.0


@pytest.mark.parametrize("category_update", ["ppo", "neurd"])
def test_category_floor_enters_the_loss_under_either_update(category_update):
    game = ContinuousKuhnPoker()
    networks, params, batch, _ = _batch(game, explore_eps=(0.3, 0.3))
    args = (params[0], networks[0], batch, 0.1, 0.5, 0.0)
    base, _ = _loss_fn(0, "vtrace", category_update=category_update)(*args)
    # At initialization the four bet components outweigh check, so check sits below 0.45.
    fn = jax.value_and_grad(
        _loss_fn(0, "vtrace", category_update=category_update, category_floor=0.45), has_aux=True
    )
    (loss, metrics), grads = fn(*args)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree_util.tree_leaves(grads))
    assert float(metrics["category_floor_frac"]) > 0.0
    assert float(loss) > float(base)


@pytest.mark.parametrize(
    "ppo", [{"category_floor": 1.0}, {"category_floor": -0.1}, {"category_floor_mode": "component"}]
)
def test_run_config_rejects_a_bad_category_floor(ppo):
    with pytest.raises(ValueError, match="category_floor"):
        run_config_from_dict({"game": {"name": "kuhn"}, "ppo": ppo})


def test_run_config_rejects_an_unknown_category_update():
    with pytest.raises(ValueError, match="category_update"):
        run_config_from_dict({"game": {"name": "kuhn"}, "ppo": {"category_update": "reinforce"}})


def test_vtrace_on_a_one_shot_batch_is_rejected():
    game = ContinuousKuhnPoker()
    networks, params, batch, _ = _batch(game)
    flat = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
    with pytest.raises(ValueError, match="sequential"):
        _loss_fn(0, "vtrace")(params[0], networks[0], flat, 0.1, 0.5, 0.0)


# ---- config ------------------------------------------------------------------------


@pytest.mark.parametrize("ppo", [
    dict(advantage="gae"), dict(advantage="vtrace", gamma=0.0), dict(advantage="vtrace", vtrace_lambda=1.5),
    dict(advantage="vtrace", vtrace_rho_bar=0.0),
    dict(advantage="vtrace", vtrace_opponent_correction="past"),
    dict(advantage="monte_carlo", vtrace_opponent_correction="future"),
    dict(advantage="vtrace", vtrace_opponent_correction="future_and_past", vtrace_opponent_past_floor=0.0),
])
def test_bad_vtrace_settings_are_rejected(ppo):
    with pytest.raises(ValueError):
        run_config_from_dict({"game": {"name": "disk_sumo"}, "ppo": ppo})


def test_vtrace_settings_reach_the_hyperparams():
    import train
    config = run_config_from_dict({
        "game": {"name": "disk_sumo"},
        "ppo": {"advantage": "vtrace", "gamma": 0.99, "vtrace_lambda": 0.8,
                "vtrace_opponent_correction": "future_and_past", "vtrace_opponent_past_floor": 0.1},
    })
    hyperparams = train.build_hyperparams(config.game.build(), 0, config)
    assert (hyperparams.advantage, hyperparams.gamma, hyperparams.vtrace_lambda) == ("vtrace", 0.99, 0.8)
    assert (hyperparams.vtrace_opponent_correction, hyperparams.vtrace_opponent_past_floor) == (
        "future_and_past", 0.1
    )
