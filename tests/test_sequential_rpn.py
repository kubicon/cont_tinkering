"""Checks for `baselines/neural/sequential_rpn.py`.

The method's defining properties are what is pinned here, because they are what
make it a different thing from every other solver in the repo rather than a
differently tuned one:

  * the policy is *implicit* -- deterministic given `z`, mixed because `z` is
    random -- and it plays legal actions only;
  * the update never differentiates it: the pseudo-gradient is a difference of
    two played-out utilities, which is checked against a case whose answer is
    known (a utility that depends on the parameters in a way one can compute);
  * the strategy read back out is sampled, and is the strategy actually played;
  * the run produces the same cost and checkpoint layout as the other solvers.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_X64_BEFORE = jax.config.jax_enable_x64

import train_sequential as ts  # noqa: E402
from baselines.neural import sequential_rpn as rpn  # noqa: E402
from baselines.neural.sequential_run import SequentialRunLog  # noqa: E402
from games.kuhn_best_response import bet_grid, game_value  # noqa: E402
from games.sequential_examples import KIND_BET, KIND_CALL, KIND_PASSIVE  # noqa: E402
from training.checkpoint import load_checkpoint_step_multi  # noqa: E402
from training.run_config import load_run_config  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)


def _tiny():
    """Classic Kuhn (exact metric) on the smallest schedule that exercises everything."""
    config = load_run_config("configs/kuhn_classic.yaml")
    config = dataclasses.replace(
        config,
        train=dataclasses.replace(config.train, solver="rpn", seed=0),
        rpn=dataclasses.replace(
            config.rpn, iterations=4, log_every=2, utility_episodes=8,
            perturbation_batch=4, strategy_samples=32, br_iterations=2),
        scoring=dataclasses.replace(config.scoring, score_every=0, episodes=500),
    )
    return config.game.build(), config


@pytest.fixture(scope="module")
def setup():
    game, config = _tiny()
    hyperparams = rpn.hyperparams_from_config(game, config)
    policy = rpn.build_policy(hyperparams)
    params = rpn.initial_params(game, policy, hyperparams, seed=0)
    return game, config, hyperparams, policy, params


# --------------------------------------------------------------------------- the policy

def test_the_policy_reads_its_shape_off_the_game_and_the_shared_network_block(setup):
    game, config, hyperparams, _, _ = setup
    assert hyperparams.num_kinds == game.num_kinds(0)
    assert hyperparams.obs_dim == game.obs_dim(0)
    assert hyperparams.hidden_dims == tuple(config.network.hidden_dims)
    assert hyperparams.activation == config.network.activation


def test_the_same_noise_gives_the_same_action_and_different_noise_does_not(setup):
    """Implicit, not stochastic: every bit of randomness enters through `z`."""
    game, _, hyperparams, policy, params = setup
    state = game.initial_state(jax.random.PRNGKey(0))
    obs, mask = game.observation(0, state), game.action_mask(0, state)
    action_fn = rpn.policy_action_fn(policy, params[0], hyperparams.noise_dim)

    first = action_fn(obs, mask, jax.random.PRNGKey(7))
    again = action_fn(obs, mask, jax.random.PRNGKey(7))
    assert int(first.kind) == int(again.kind)
    assert np.allclose(first.value, again.value)

    actions = jax.vmap(lambda k: action_fn(obs, mask, k))(
        jax.random.split(jax.random.PRNGKey(1), 256))
    assert np.std(np.asarray(actions.value)) > 0.0        # the size really varies with z


def test_the_policy_only_ever_plays_legal_kinds_and_sizes_in_the_box(setup):
    game, _, hyperparams, policy, params = setup
    state = game.initial_state(jax.random.PRNGKey(0))
    action_fn = rpn.policy_action_fn(policy, params[0], hyperparams.noise_dim)
    # Facing a bet, checking is a fold and betting is illegal: only PASSIVE/CALL.
    faced = game.decision_nodes(0)[1]
    obs = game.infoset_observation(0, faced, 1.0)
    mask = game.infoset_action_mask(faced)
    kinds = jax.vmap(lambda k: action_fn(obs, mask, k).kind)(
        jax.random.split(jax.random.PRNGKey(2), 128))
    assert set(np.asarray(kinds).tolist()) <= {KIND_PASSIVE, KIND_CALL}
    assert KIND_BET not in np.asarray(kinds).tolist()

    space = game.action_space(0)
    sizes = jax.vmap(lambda k: action_fn(obs, mask, k).value)(
        jax.random.split(jax.random.PRNGKey(3), 128))
    assert np.all(np.asarray(sizes) >= float(space.low[0]) - 1e-6)
    assert np.all(np.asarray(sizes) <= float(space.high[0]) + 1e-6)


# --------------------------------------------------------------------------- the update

def test_the_pseudo_gradient_points_uphill_on_a_utility_it_can_be_checked_against(setup):
    """The estimator is game-agnostic, so it can be checked on a utility whose
    gradient is known: `u = -||theta_0 - 1||^2 + ||theta_1||^2`. Player 0 ascends
    `u`, player 1 ascends `-u`, so both pseudo-gradients must have a positive
    inner product with the true ones."""
    _, _, hyperparams, _, params = setup

    def utility(profile, key):
        del key
        first = -sum(jnp.sum((leaf - 1.0) ** 2) for leaf in jax.tree_util.tree_leaves(profile[0]))
        second = sum(jnp.sum(leaf ** 2) for leaf in jax.tree_util.tree_leaves(profile[1]))
        return first + second

    gradients, evaluations = rpn.ESTIMATORS["separate"](
        utility, params, jax.random.PRNGKey(0), sigma=0.05, antithetic=True, batch=64)
    assert evaluations == 128                       # `batch` per player, two players

    # Both players' ascent directions are `-2 * (theta - target)`: player 0 pulls
    # its parameters towards 1 to raise `u`, player 1 pulls its own towards 0 to
    # raise `-u`. The estimator already carries player 1's sign flip.
    for player, target in ((0, 1.0), (1, 0.0)):
        true = jax.tree_util.tree_map(lambda p, target=target: -2.0 * (p - target),
                                      params[player])
        inner = sum(float(jnp.sum(a * b)) for a, b in
                    zip(jax.tree_util.tree_leaves(gradients[player]),
                        jax.tree_util.tree_leaves(true)))
        assert inner > 0.0, (player, inner)


def test_a_single_player_gradient_is_finite_and_ascends(setup):
    _, _, _, _, params = setup

    def utility(profile, key):
        del key
        return sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(profile[0]))

    gradient, evaluations = rpn.player_pseudo_gradient(
        utility, params, 0, jax.random.PRNGKey(0), sigma=0.05, antithetic=True, batch=32)
    assert evaluations == 32
    # `u` is linear in player 0's parameters with unit slope, so the estimate is
    # ~1 everywhere; what matters is that it is positive and finite.
    leaves = jax.tree_util.tree_leaves(gradient)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert float(sum(jnp.sum(leaf) for leaf in leaves)) > 0.0


# --------------------------------------------------------------------------- measurement

def test_the_sampled_strategy_is_the_one_that_gets_played(setup):
    """The reader has no density to consult, so it samples; the check is that the
    value it implies is the value of actually playing the policies."""
    game, _, hyperparams, policy, params = setup
    grid = bet_grid(game)
    strategies = [
        rpn.kuhn_strategy_of(game, policy, params[player], player, grid,
                             jax.random.PRNGKey(player), 4096, hyperparams.noise_dim)
        for player in (0, 1)
    ]
    for strategy in strategies:
        strategy.validate()
    exact = float(game_value(game, grid, strategies[0], strategies[1]))

    played, stderr = rpn.RPNEvaluator(game, policy, hyperparams.noise_dim).evaluate(
        params, jax.random.PRNGKey(0), episodes=40_000)
    assert exact == pytest.approx(played, abs=5 * stderr + 0.05)


def test_the_scorer_reports_the_same_columns_as_the_other_solvers(setup):
    game, config, hyperparams, policy, params = setup
    score = rpn.build_scorer(game, policy, hyperparams, config)
    row = score(params, 1)
    assert {"h2h", "expl", "value"} <= set(row)
    assert np.isfinite(row["expl"])
    assert score(params, 1, full=False).keys() == {"expl", "br_0", "br_1", "value"}


def test_the_bound_is_reported_when_asked_for(setup):
    game, config, hyperparams, policy, params = setup
    config = dataclasses.replace(
        config, scoring=dataclasses.replace(config.scoring, score_every=1, episodes=500))
    row = rpn.build_scorer(game, policy, hyperparams, config)(params, 1)
    assert np.isfinite(row["expl_lb"])


# --------------------------------------------------------------------------- end to end

def test_a_run_logs_the_shared_cost_columns_and_checkpoints(tmp_path):
    game, config = _tiny()
    config = dataclasses.replace(
        config, train=dataclasses.replace(config.train, checkpoint_dir=str(tmp_path / "rpn")))
    log = SequentialRunLog(config.train.checkpoint_dir, {"solver": "rpn"})
    result = ts.SOLVER_RUNNERS["rpn"](game, config, log, ts.build_run_scorer(game, config))
    log.finish()

    assert [row["t"] for row in log.history] == [1, 2]        # 4 iterations, log_every 2
    for row in log.history:
        for column in ("wall_time", "total_wall_time", "iterations", "episodes", "env_steps"):
            assert column in row, column
        assert np.isfinite(row["expl"]) and np.isfinite(row["grad_norm_0"])
    # Two players, `perturbation_batch` perturbations each, `utility_episodes` hands
    # per evaluation -- exactly, since every iteration costs the same.
    per_iteration = 2 * config.rpn.perturbation_batch * config.rpn.utility_episodes
    assert log.history[-1]["episodes"] == 4 * per_iteration
    assert log.history[-1]["iterations"] == 4
    assert 0 < log.history[-1]["env_steps"] <= 4 * per_iteration * game.max_steps
    assert result["budget"].mean_episode_length > 0

    entries = load_checkpoint_step_multi(
        tmp_path / "rpn" / "checkpoints", 2,
        hyperparams_cls=rpn.SequentialRPNHyperparams)
    assert set(entries) == {"player_0", "player_1"}
