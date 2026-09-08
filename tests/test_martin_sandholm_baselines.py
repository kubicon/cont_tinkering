"""Checks for the Martin & Sandholm baselines: the pseudo-gradient family and SISAMS.

The three implementations are `baselines/neural/randomized_policy.py` (randomized policy
networks + simultaneous pseudo-gradient, IJCAI 2023), `baselines/neural/jpspg.py` (the
joint-perturbation estimator, IJCAI 2025), and `baselines/sisa.py` (simultaneous
incremental support adjustment and metagame solving, arXiv:2406.08683).

What is pinned:

  * the zeroth-order estimators actually estimate the gradient -- checked against a
    synthetic utility whose gradient is known, since on a real game there is nothing to
    compare an estimate to;
  * the two estimators agree in expectation while the joint one uses half the utility
    evaluations, which is the entire claim of the second paper;
  * SISAMS's metagame subgradient finds the metagame equilibrium on a support where the
    answer is known in closed form, and the whole algorithm reaches the true Nash when
    its support covers the equilibrium's basins.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_X64_BEFORE = jax.config.jax_enable_x64

from baselines.common import GridOracle, load_game  # noqa: E402
from baselines.neural.jpspg import jpspg_pseudo_gradients  # noqa: E402
from baselines.neural.randomized_policy import RandomizedPolicyHyperparams, build_policy, \
    run_pseudo_gradient, sample_actions, spg_pseudo_gradients, tree_normal  # noqa: E402
from baselines.sisa import collapse_warning, metagame_exploitability, run_sisa  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)

PEAKS = (-1.0, 1.0)
NASH_WEIGHTS = np.array([0.3, 0.7])


@pytest.fixture(scope="module")
def game():
    jax.config.update("jax_enable_x64", True)
    return load_game("configs/two_point.yaml")[0]


@pytest.fixture(scope="module")
def oracle(game):
    return GridOracle(game, points=201)


# --------------------------------------------------------------------------- estimators


def _quadratic_utility(centre_0, centre_1):
    """A synthetic two-player utility with a known gradient in each player's parameters.

    `u = -||x0 - c0||^2 + ||x1 - c1||^2`, so player 0 (maximizing `u`) has gradient
    `-2(x0 - c0)` and player 1 (maximizing `-u`) has `-2(x1 - c1)`: both point at their
    own centre. A real game gives no such reference, which is exactly why the estimator
    is checked here instead.
    """

    def utility(params, key):
        del key      # deterministic: the estimator's noise is the only randomness
        return (-jnp.sum((params[0]["w"] - centre_0) ** 2)
                + jnp.sum((params[1]["w"] - centre_1) ** 2))

    return utility


def _average_estimate(estimator, utility, params, draws: int, sigma: float, seed: int = 0):
    keys = jax.random.split(jax.random.PRNGKey(seed), draws)
    totals = [jnp.zeros_like(params[i]["w"]) for i in (0, 1)]
    evaluations = 0
    for key in keys:
        gradients, count = estimator(utility, params, key, sigma, True)
        evaluations += count
        totals = [totals[i] + gradients[i]["w"] for i in (0, 1)]
    return [total / draws for total in totals], evaluations


def _cosine(a, b) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("estimator", [spg_pseudo_gradients, jpspg_pseudo_gradients])
def test_pseudo_gradient_points_where_the_true_gradient_does(estimator):
    centre_0, centre_1 = jnp.array([1.0, -2.0]), jnp.array([-0.5, 0.5])
    params = ({"w": jnp.zeros(2)}, {"w": jnp.zeros(2)})
    utility = _quadratic_utility(centre_0, centre_1)

    estimates, _ = _average_estimate(estimator, utility, params, draws=400, sigma=0.1)
    # Both players ascend towards their own centre.
    assert _cosine(estimates[0], -2.0 * (params[0]["w"] - centre_0)) > 0.9
    assert _cosine(estimates[1], -2.0 * (params[1]["w"] - centre_1)) > 0.9


def test_joint_perturbation_matches_separate_at_half_the_cost():
    """The IJCAI 2025 claim, on a utility where both estimators can be averaged to
    convergence: same expectation, constant rather than linear evaluation count."""
    centre_0, centre_1 = jnp.array([1.0, -2.0]), jnp.array([-0.5, 0.5])
    params = ({"w": jnp.zeros(2)}, {"w": jnp.zeros(2)})
    utility = _quadratic_utility(centre_0, centre_1)

    separate, separate_evaluations = _average_estimate(
        spg_pseudo_gradients, utility, params, draws=600, sigma=0.1, seed=1)
    joint, joint_evaluations = _average_estimate(
        jpspg_pseudo_gradients, utility, params, draws=600, sigma=0.1, seed=2)

    assert separate_evaluations == 2 * joint_evaluations      # 4 per iteration vs 2
    for player in (0, 1):
        assert _cosine(separate[player], joint[player]) > 0.9


def test_single_point_estimator_is_cheaper_than_the_central_difference():
    """`batch` counts *evaluations*, so one antithetic pair costs what two single-point
    draws cost. Halving the batch is what makes the single-point stencil the cheap one."""
    params = ({"w": jnp.zeros(2)}, {"w": jnp.zeros(2)})
    utility = _quadratic_utility(jnp.zeros(2), jnp.zeros(2))
    _, antithetic = spg_pseudo_gradients(utility, params, jax.random.PRNGKey(0), 0.1, True, 2)
    _, single = spg_pseudo_gradients(utility, params, jax.random.PRNGKey(0), 0.1, False, 1)
    assert (antithetic, single) == (4, 2)


@pytest.mark.parametrize("estimator", [spg_pseudo_gradients, jpspg_pseudo_gradients])
def test_perturbation_batch_sharpens_the_estimate(estimator):
    """The papers' `batch_size`. A single draw is a random direction in parameter space
    and is nearly orthogonal to the gradient; averaging `batch` of them is the only thing
    that makes a *single* iteration informative, which is why the reference
    implementation's default of 2 does not reproduce the papers' experiments."""
    centre_0, centre_1 = jnp.array([1.0, -2.0]), jnp.array([-0.5, 0.5])
    params = ({"w": jnp.zeros(8)}, {"w": jnp.zeros(8)})
    utility = _quadratic_utility(jnp.zeros(8).at[:2].set(centre_0[:2]) * 0 + 1.0,
                                 jnp.zeros(8) - 0.5)
    truth = -2.0 * (params[0]["w"] - 1.0)

    def mean_cosine(batch):
        keys = jax.random.split(jax.random.PRNGKey(4), 64)
        return float(np.mean([
            _cosine(estimator(utility, params, k, 0.05, True, batch)[0][0]["w"], truth)
            for k in keys]))

    single, batched = mean_cosine(2), mean_cosine(256)
    assert batched > single + 0.2
    assert batched > 0.9

    # And the evaluation count scales with the batch, so the gain is paid for.
    _, evaluations = estimator(utility, params, jax.random.PRNGKey(0), 0.05, True, 64)
    assert evaluations == (64 if estimator is jpspg_pseudo_gradients else 128)


def test_tree_normal_matches_the_parameter_structure():
    params = {"a": jnp.zeros((3, 2)), "b": jnp.zeros(5)}
    noise = tree_normal(jax.random.PRNGKey(0), params)
    assert jax.tree_util.tree_structure(noise) == jax.tree_util.tree_structure(params)
    assert noise["a"].shape == (3, 2) and noise["b"].shape == (5,)
    assert abs(float(jnp.std(noise["a"])) - 1.0) < 0.5


# --------------------------------------------------------------------------- policy


def test_randomized_policy_stays_in_the_action_box(game):
    hyperparams = RandomizedPolicyHyperparams(
        action_dim=1, hidden_dims=(32,), noise_dim=4, low=(-2.0,), high=(2.0,))
    policy = build_policy(hyperparams)
    obs = game.observation(0, jax.random.PRNGKey(0))
    params = policy.init(jax.random.PRNGKey(0), obs, jnp.zeros(4))
    actions = np.asarray(sample_actions(policy, params, obs, jax.random.PRNGKey(1), 512, 4))
    assert actions.min() >= -2.0 and actions.max() <= 2.0
    # A randomized policy is supposed to be *random*: a degenerate one would be a bug.
    assert float(np.std(actions)) > 1e-3


def test_pseudo_gradient_run_produces_a_history_and_checkpoints(game, oracle, tmp_path):
    from baselines.common import load_checkpoints
    from baselines.neural.common import RunWriter

    hyperparams = RandomizedPolicyHyperparams(
        action_dim=1, hidden_dims=(32,), noise_dim=4, low=(-2.0,), high=(2.0,),
        utility_samples=64, perturbation_batch=2)
    writer = RunWriter(tmp_path / "spg", {"algorithm": "randomized_policy_spg"})
    result = run_pseudo_gradient(game, oracle, hyperparams, spg_pseudo_gradients,
                                 iterations=20, log_every=10, samples=256, writer=writer)
    writer.finish()
    assert [h["t"] for h in result["history"]] == [0, 10, 20]
    assert result["utility_evaluations"] == 20 * 4          # antithetic, two players
    assert len(load_checkpoints(tmp_path / "spg" / "checkpoints")) == 3
    assert (tmp_path / "spg" / "params" / "player_0" / "params.msgpack").exists()


# --------------------------------------------------------------------------- SISAMS


def test_metagame_subgradient_finds_the_metagame_equilibrium(game, oracle):
    """With the support pinned at the Nash's own atoms and the support step off, the
    weight rule alone has to recover the (0.3, 0.7) mix."""
    support = np.array([[-1.0], [1.0]])
    result = run_sisa(game, oracle, iters=3000, lr_support=0.0, lr_weight=0.05,
                      init_support=(support, support), seed=0)
    assert np.allclose(result.weights_0, NASH_WEIGHTS, atol=0.03), result.weights_0
    assert np.allclose(result.weights_1, NASH_WEIGHTS, atol=0.03), result.weights_1
    assert result.history[-1]["expl"] < 0.05


def test_sisa_reaches_the_nash_when_its_support_spans_the_basins(game, oracle):
    result = run_sisa(game, oracle, atoms=4, iters=3000, init="spread", seed=0)
    assert result.history[-1]["expl"] < 0.05, result.history[-1]
    # The atoms carrying weight sit on the peaks, with the Nash mix over them.
    carrying = result.weights_0 > 1e-2
    assert sorted(np.round(result.support_0[carrying].ravel(), 1)) == [-1.0, 1.0]
    assert np.allclose(sorted(result.weights_0[carrying]), NASH_WEIGHTS, atol=0.05)


def test_metagame_exploitability_is_zero_at_a_matrix_game_equilibrium():
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])
    half = np.array([0.5, 0.5])
    assert abs(metagame_exploitability(payoff, half, half)) < 1e-12
    assert metagame_exploitability(payoff, np.array([1.0, 0.0]), half) > 0.0


def test_collapse_warning_fires_only_on_a_collapsed_support():
    assert collapse_warning({"metagame_expl": 0.0, "expl": 1.2}) is not None
    assert collapse_warning({"metagame_expl": 0.0, "expl": 0.001}) is None
    assert collapse_warning({"metagame_expl": 0.3, "expl": 1.2}) is None
