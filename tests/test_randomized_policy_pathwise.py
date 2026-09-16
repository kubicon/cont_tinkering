"""Checks for `baselines/neural/randomized_policy_pathwise.py`.

The module is the second version of the randomized policy network: the representation of
`baselines/neural/randomized_policy.py` trained by exact pathwise gradients instead of a
zeroth-order pseudo-gradient. What is pinned is the part that differs, plus the parts a
harness depends on:

  * the gradient really is the gradient -- a directional derivative against a central
    finite difference of the same utility, which is the reference the zeroth-order tests
    cannot have (there, the estimator is checked against a synthetic utility instead);
  * player 1 ascends `-u`, so a sign slip that would make both players maximize the same
    thing is caught rather than showing up later as a run that will not converge;
  * `--smooth` genuinely *replaces* the pathwise gradient rather than adding to it, which
    is what makes the flag a switch between access models;
  * the cost formula `experiments/one_shot_neural/run_cell.py` budgets with matches what
    a run actually spends.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_X64_BEFORE = jax.config.jax_enable_x64

from baselines.common import GridOracle, load_game  # noqa: E402
from baselines.neural.randomized_policy_pathwise import PathwisePolicyHyperparams, \
    build_pathwise_optimizer, build_policy, build_utility_fn, extragradient_gradients, \
    payoff_evals_per_iteration, player_gradients, run_pathwise, sample_actions, \
    smoothed_utility_fn  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)


@pytest.fixture(scope="module")
def game():
    jax.config.update("jax_enable_x64", True)
    return load_game("configs/two_point.yaml")[0]


@pytest.fixture(scope="module")
def oracle(game):
    return GridOracle(game, points=201)


def _setup(game, **overrides):
    """A small policy, its utility, and an initialized parameter pair."""
    hyperparams = PathwisePolicyHyperparams(
        action_dim=1, hidden_dims=(32,), noise_dim=4, low=(-2.0,), high=(2.0,),
        batch_size=64, **overrides)
    policy = build_policy(hyperparams)
    utility = build_utility_fn(game, policy, hyperparams)
    obs = tuple(game.observation(player, jax.random.PRNGKey(0)) for player in (0, 1))
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    params = tuple(
        policy.init(keys[player], obs[player], jnp.zeros(hyperparams.noise_dim))
        for player in (0, 1))
    return hyperparams, policy, utility, params


def _direction(tree, seed):
    return jax.tree_util.tree_map(
        lambda x: jax.random.normal(jax.random.PRNGKey(seed), x.shape, x.dtype), tree)


def _dot(a, b):
    return sum(float(jnp.vdot(x, y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def _shift(tree, direction, eps):
    return jax.tree_util.tree_map(lambda p, z: p + eps * z, tree, direction)


# --------------------------------------------------------------------------- the gradient


@pytest.mark.parametrize("player", [0, 1])
def test_pathwise_gradient_matches_a_finite_difference(game, player):
    """The whole claim of this variant, checked the only way it can be: against the
    finite difference of the very same utility.

    The key is held fixed across the three evaluations, so the noise draw is common to
    them and the difference measures the parameters alone -- without that, the sampling
    noise swamps an `eps`-sized change and this test would be measuring nothing.
    """
    _, _, utility, params = _setup(game)
    key = jax.random.PRNGKey(7)
    gradients = player_gradients(utility, params, key)

    direction = _direction(params[player], 3)
    autodiff = _dot(gradients[player], direction)
    # Player 1 maximizes the negated payoff, so its objective is `-u`.
    sign = 1.0 if player == 0 else -1.0
    eps = 1e-4

    def at(shifted):
        profile = (shifted, params[1]) if player == 0 else (params[0], shifted)
        return float(utility(profile, key))

    finite = sign * (at(_shift(params[player], direction, eps))
                     - at(_shift(params[player], direction, -eps))) / (2 * eps)
    assert autodiff == pytest.approx(finite, rel=1e-3)


def test_the_two_players_ascend_opposite_objectives(game):
    """Both players differentiating `+u` is the sign slip this module could plausibly
    have, and it would look like a plain failure to converge rather than a bug."""
    _, _, utility, params = _setup(game)
    key = jax.random.PRNGKey(7)
    gradients = player_gradients(utility, params, key)

    # Player 1's gradient must be the gradient of `-u`, i.e. the negation of what it
    # would be if it maximized the payoff like player 0.
    def player_1_utility(own):
        return utility((params[0], own), key)

    ascending_payoff = jax.grad(player_1_utility)(params[1])
    direction = _direction(params[1], 4)
    assert _dot(gradients[1], direction) == pytest.approx(-_dot(ascending_payoff, direction),
                                                          rel=1e-6)


def test_smoothing_replaces_the_pathwise_gradient(game):
    """`--smooth` switches the access model rather than adding a term to it.

    `optax.perturbations` evaluates the wrapped function only at `stop_gradient`-ed
    inputs, so a *constant* utility must have exactly zero gradient through the proxy --
    if any pathwise path survived, a function whose value never changes could still
    produce one. This is what makes the flag a genuine zeroth-order fallback.
    """
    hyperparams, _, utility, params = _setup(game)
    constant = lambda profile, key: jnp.asarray(1.0)
    proxy = smoothed_utility_fn(constant, scale=0.1, samples=4)
    gradient = jax.grad(lambda own: proxy((own, params[1]), jax.random.PRNGKey(0)))(params[0])
    assert _dot(gradient, gradient) == pytest.approx(0.0, abs=1e-12)

    # And on a real utility the smoothed gradient is finite and not identically zero,
    # so the estimator is live rather than merely detached.
    smoothed = smoothed_utility_fn(utility, scale=0.1, samples=4)
    estimate = player_gradients(smoothed, params, jax.random.PRNGKey(3))
    norm = _dot(estimate[0], estimate[0])
    assert np.isfinite(norm) and norm > 0.0


# --------------------------------------------------------------------------- policy / cost


def test_pathwise_policy_stays_in_the_action_box(game):
    hyperparams, policy, _, params = _setup(game)
    obs = game.observation(0, jax.random.PRNGKey(0))
    actions = np.asarray(sample_actions(policy, params[0], obs, jax.random.PRNGKey(1), 512, 4))
    assert actions.min() >= -2.0 and actions.max() <= 2.0
    # A randomized policy is supposed to be *random*: a degenerate one would be a bug.
    assert float(np.std(actions)) > 1e-3


def test_optimistic_optimizer_uses_the_reference_coefficients():
    """`optimistic` must be OGD with `alpha = lr` and `beta = optimism`, not the
    single-argument form `training.optimizers` exposes -- the coefficients are the part
    of this variant that looks wrong and is not."""
    hyperparams = PathwisePolicyHyperparams(
        action_dim=1, hidden_dims=(4,), learning_rate=0.01, optimism=1.0)
    optimizer = build_pathwise_optimizer(hyperparams)
    params = {"w": jnp.zeros(3)}
    state = optimizer.init(params)

    # optax substitutes the current gradient for the absent previous one on the first
    # step, so `u = (alpha + beta) g - beta g = alpha g`: the run opens with a plain
    # `lr`-sized step rather than a `beta`-sized jump.
    updates, state = optimizer.update({"w": jnp.ones(3)}, state, params)
    assert np.asarray(updates["w"])[0] == pytest.approx(-0.01, rel=1e-6)

    # Thereafter the extrapolation is live: at a *changed* gradient the negative-momentum
    # term is `beta = 1.0` times the change, which dwarfs the `alpha = 0.01` step. That
    # ratio is the variant's actual behaviour and what a "corrected" beta would remove.
    updates, _ = optimizer.update({"w": jnp.full((3,), 2.0)}, state, params)
    assert np.asarray(updates["w"])[0] == pytest.approx(-((0.01 + 1.0) * 2.0 - 1.0), rel=1e-6)


def test_payoff_eval_cost_matches_what_a_run_reports(game, oracle):
    """The budget in `experiments/one_shot_neural/` is in payoff evaluations, so the
    formula `run_cell.plan_units` prices with has to be the one a run spends."""
    assert payoff_evals_per_iteration(64, 0) == 2 * 64
    assert payoff_evals_per_iteration(64, 3) == 2 * 64 * 4

    hyperparams = PathwisePolicyHyperparams(
        action_dim=1, hidden_dims=(32,), noise_dim=4, low=(-2.0,), high=(2.0,), batch_size=8)
    result = run_pathwise(game, oracle, hyperparams, iterations=20, log_every=10,
                          samples=256, score=False)
    assert result["payoff_evals"] == 20 * payoff_evals_per_iteration(8, 0)
    assert result["history"][-1]["payoff_evals"] == result["payoff_evals"]


def test_extragradient_measures_the_gradient_at_the_lookahead_point(game):
    """The look-ahead gradient is the plain gradient at `theta + step * g`, for both
    players at once and with the same key -- not a second sample at the current point."""
    _, _, utility, params = _setup(game)
    key = jax.random.PRNGKey(7)
    step = 0.05

    first = player_gradients(utility, params, key)
    lookahead = tuple(_shift(params[player], first[player], step) for player in (0, 1))
    expected = player_gradients(utility, lookahead, key)
    actual = extragradient_gradients(utility, params, key, step)

    for player in (0, 1):
        direction = _direction(params[player], 5 + player)
        assert _dot(actual[player], direction) == pytest.approx(
            _dot(expected[player], direction), rel=1e-6)
    # And the look-ahead is a real move: the result is not the gradient at `theta`.
    assert not np.isclose(_dot(actual[0], first[0]), _dot(first[0], first[0]), rtol=1e-6)


def test_extragradient_doubles_the_payoff_eval_cost(game, oracle):
    assert payoff_evals_per_iteration(64, 0, "extragradient") == 4 * 64
    assert payoff_evals_per_iteration(64, 3, "extragradient") == 4 * 64 * 4

    hyperparams = PathwisePolicyHyperparams(
        action_dim=1, hidden_dims=(32,), noise_dim=4, low=(-2.0,), high=(2.0,), batch_size=8,
        smooth=0, dynamics="extragradient")
    result = run_pathwise(game, oracle, hyperparams, iterations=20, log_every=10,
                          samples=256, score=False)
    assert result["payoff_evals"] == 20 * payoff_evals_per_iteration(8, 0, "extragradient")


def test_run_produces_a_history_and_checkpoints(game, oracle, tmp_path):
    from baselines.common import load_checkpoints
    from baselines.neural.common import RunWriter

    hyperparams = PathwisePolicyHyperparams(
        action_dim=1, hidden_dims=(32,), noise_dim=4, low=(-2.0,), high=(2.0,), batch_size=16)
    writer = RunWriter(tmp_path / "rpn", {"algorithm": "randomized_policy_pathwise"})
    result = run_pathwise(game, oracle, hyperparams, iterations=20, log_every=10,
                          samples=256, writer=writer)
    writer.finish()
    assert [h["t"] for h in result["history"]] == [0, 10, 20]
    assert len(load_checkpoints(tmp_path / "rpn" / "checkpoints")) == 3
    assert (tmp_path / "rpn" / "params" / "player_0" / "params.msgpack").exists()


def test_hyperparams_round_trip():
    """`meta.json` has to be enough to rebuild the run, which `from_dict` is what makes
    true -- the tuple fields come back from JSON as lists otherwise."""
    hyperparams = PathwisePolicyHyperparams(
        action_dim=2, hidden_dims=(64, 64), low=(-1.0, 0.0), high=(1.0, 2.0), smooth=3)
    assert PathwisePolicyHyperparams.from_dict(hyperparams.to_dict()) == hyperparams
