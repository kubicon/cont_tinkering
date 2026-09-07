"""Checks for the neural baselines in `baselines/neural/`.

These are stochastic RL runs, so what is pinned here is what *must* hold regardless of
how a particular seed's training went:

  * the opponent samplers really sample the strategy they claim to (a best response is
    only meaningful against the distribution it was trained on, and every one of these
    algorithms is a stack of best responses),
  * the best-response oracle actually maximizes on a game whose best response is known
    in closed form,
  * the discretized MMD policy learns (its exploitability falls a long way from the
    uniform policy's), and
  * NFSP and PSRO run end to end and leave behind the same checkpoint format as every
    other baseline in the repo.

Anything sharper -- "NFSP reaches exploitability X" -- is a claim about a run, not about
the code, and belongs in an experiment, not in the test suite.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_X64_BEFORE = jax.config.jax_enable_x64

from baselines.common import GridOracle, load_checkpoints  # noqa: E402
from baselines.neural import br_oracle as bo  # noqa: E402
from baselines.neural.common import action_grid, empirical_strategy, load_run  # noqa: E402
from baselines.neural.mmd_discrete import hyperparams_from_config, run_mmd_discrete  # noqa: E402
from baselines.neural.nfsp import Reservoir, run_nfsp  # noqa: E402
from baselines.neural.psro import run_psro  # noqa: E402
from games.examples import MultiPointGame, QuadraticZeroSumGame  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)

PEAKS = (-1.0, 1.0)
WEIGHTS = (0.3, 0.7)


@pytest.fixture(scope="module")
def loaded():
    jax.config.update("jax_enable_x64", True)
    return load_run("configs/two_point.yaml")


@pytest.fixture(scope="module")
def oracle(loaded):
    return GridOracle(loaded[0], points=201)


def test_finite_opponent_samples_its_own_weights():
    opponent = bo.finite_opponent(np.array([[-1.0], [1.0]]), np.array([0.3, 0.7]))
    actions = np.asarray(opponent(jax.random.PRNGKey(0), 20_000))
    assert abs(float(np.mean(actions > 0)) - 0.7) < 0.02


def test_mix_opponents_respects_its_probabilities():
    left = bo.finite_opponent(np.array([[-1.0]]), np.array([1.0]))
    right = bo.finite_opponent(np.array([[1.0]]), np.array([1.0]))
    mixed = bo.mix_opponents([left, right], [0.25, 0.75])
    actions = np.asarray(mixed(jax.random.PRNGKey(1), 20_000))
    assert abs(float(np.mean(actions > 0)) - 0.75) < 0.02


def test_population_opponent_of_one_policy_is_that_policy(loaded):
    """Two copies of the same policy, any weights, must sample like the policy itself --
    the check that the stack-and-select trick in `population_opponent` is not scrambling
    parameters between members."""
    game, _, config = loaded
    from training.mixture import build_mixture_network

    hyperparams = bo.br_hyperparams(game, 1, config)
    network = build_mixture_network(hyperparams)
    params = network.init(jax.random.PRNGKey(0), game.observation(1, bo.OBSERVATION_KEY))

    single = bo.policy_opponent(game, 1, network, params, samples=8192)
    population = bo.population_opponent(game, 1, network, [params, params], [0.4, 0.6], samples=8192)
    assert abs(float(np.mean(single.support)) - float(np.mean(population.support))) < 0.15


def test_payoff_estimate_matches_a_direct_average():
    game = MultiPointGame(peaks=PEAKS, weights=WEIGHTS)
    a0 = np.array([[-1.0], [1.0]])
    a1 = np.array([[1.0], [0.0], [-1.0]])
    direct = np.mean([[float(game.payoff(jnp.asarray(x), jnp.asarray(y))) for y in a1] for x in a0])
    assert abs(bo.payoff_estimate(game, a0, a1) - direct) < 1e-9


def test_br_hyperparams_strip_the_regularizers(loaded):
    game, _, config = loaded
    hyperparams = bo.br_hyperparams(game, 0, config)
    assert all(getattr(hyperparams, name) == 0.0 for name in bo.BR_REGULARIZERS)
    # ... but a deliberate override survives, and the architecture still comes from the config.
    kept = bo.br_hyperparams(game, 0, config, category_entropy_coef=0.01)
    assert kept.category_entropy_coef == 0.01
    assert kept.hidden_dims == tuple(config.network.hidden_dims)


def test_best_response_maximizes_on_a_game_with_a_known_answer():
    """`QuadraticZeroSumGame` with no coupling: player 0's payoff is `-||a||^2`, so its
    best response is the origin whatever the opponent does. A best-response oracle that
    does not find it is not one."""
    jax.config.update("jax_enable_x64", True)
    game = QuadraticZeroSumGame(coupling=jnp.zeros((1, 1)), bound=3.0)
    from training.run_config import run_config_from_dict

    config = run_config_from_dict({
        "game": {"name": "quadratic"},
        "optimizer": {"learning_rate": 0.01},
        "ppo": {"batch_size": 256, "ppo_epochs": 4},
    })
    hyperparams = bo.br_hyperparams(game, 0, config)
    opponent = bo.uniform_opponent(game, 1, samples=512)
    response = bo.train_best_response(game, 0, opponent, hyperparams, steps=15, epochs=10, seed=0)
    actions = np.asarray(response.opponent(game, 0, samples=2048).support)
    assert abs(float(np.mean(actions))) < 0.5, float(np.mean(actions))


def test_reservoir_keeps_capacity_and_stays_representative():
    reservoir = Reservoir(capacity=500, dim=1, seed=0)
    reservoir.add(np.full((400, 1), -1.0))
    assert reservoir.size == 400
    reservoir.add(np.full((3600, 1), 1.0))
    assert reservoir.size == 500 and reservoir.seen == 4000
    support, weights = reservoir.strategy()
    # A uniform sample of 4000 items that are 10% -1 and 90% +1.
    assert abs(float(np.mean(support > 0)) - 0.9) < 0.06
    assert np.allclose(weights.sum(), 1.0)


def test_action_grid_refuses_an_unrepresentable_discretization(loaded):
    game = loaded[0]
    with pytest.raises(ValueError, match="Lower --bins"):
        action_grid(game, 0, 400_000)


class _Args:
    """The override fields `hyperparams_from_config` reads off argparse.

    All `None`, i.e. "take the config's values" -- the point of the test is that the
    baseline learns under the *same* settings the mixture method is configured with, not
    under settings picked to make it pass. (It is genuinely sensitive to them: at
    `lr=0.01` the categorical head collapses onto a single cell within a few hundred
    iterations and the run is worse than uniform.)
    """

    lr = None
    batch = None
    entropy = None
    magnet_kl = None
    trpo_kl = None
    magnet_interval = None


def test_mmd_discrete_learns(loaded, oracle):
    """The uniform policy on this game is exploitable by about 2.5; 400 iterations of
    MMD on the discretized head must make a real dent in that."""
    game, _, config = loaded
    hyperparams = hyperparams_from_config(game, config, bins=41, args=_Args())
    result = run_mmd_discrete(game, oracle, hyperparams, iterations=400, log_every=200, seed=0)
    history = result["history"]
    assert history[0]["expl"] > 2.0
    assert history[-1]["expl"] < 0.75 * history[0]["expl"], [h["expl"] for h in history]
    # The policy is a distribution over the grid, so its snapshot is exact.
    assert np.allclose(result["weights_0"].sum(), 1.0)
    assert result["support_0"].shape[0] == 41


def test_psro_runs_and_checkpoints(loaded, oracle, tmp_path):
    from baselines.neural.common import RunWriter

    game, _, config = loaded
    writer = RunWriter(tmp_path / "psro", {"algorithm": "psro"})
    result = run_psro(game, oracle, config, rounds=2, br_steps=2, br_epochs=5,
                      payoff_samples=64, seed=0, writer=writer)
    writer.finish()

    assert [h["t"] for h in result["history"]] == [1, 2, 3]
    assert result["history"][-1]["population_0"] == 3      # seed policy + one per round
    assert result["payoff"].shape == (3, 3)
    checkpoints = load_checkpoints(tmp_path / "psro" / "checkpoints")
    assert len(checkpoints) == 3
    assert np.allclose(checkpoints[-1].weights_0.sum(), 1.0)
    assert "meta_weights_0" in checkpoints[-1].extra
    # Population weights are saved in the repo's own checkpoint format, not a pickle.
    assert (tmp_path / "psro" / "params" / "player0_policy0" / "hyperparams.json").exists()
    assert json.loads((tmp_path / "psro" / "meta.json").read_text())["algorithm"] == "psro"


def test_nfsp_runs_with_both_average_heads(loaded, oracle, tmp_path):
    from baselines.neural.common import RunWriter

    game, _, config = loaded
    for head in ("reservoir", "mixture"):
        writer = RunWriter(tmp_path / head, {"algorithm": "nfsp", "average_head": head})
        result = run_nfsp(game, oracle, config, rounds=2, br_steps=2, br_epochs=5,
                          reservoir_samples=256, sl_steps=50, sl_batch=64, samples=512,
                          average_head=head, seed=0, writer=writer)
        writer.finish()
        assert [h["t"] for h in result["history"]] == [1, 2]
        assert result["history"][-1]["reservoir"] == 512
        assert len(load_checkpoints(tmp_path / head / "checkpoints")) == 2


@pytest.mark.slow
def test_best_response_reaches_the_exact_value_at_the_nash(loaded, oracle):
    """The oracle every other baseline is built on, against ground truth.

    At the Nash of `two_point` no deviation is worth anything, so the exact
    best-response value is 0. A PPO oracle given enough iterations has to find that --
    and the same measurement at smaller budgets is what set NFSP's and PSRO's defaults
    (see `baselines/neural/README.md`); this pins the converged end so a regression in
    the oracle cannot hide behind "RL is noisy".
    """
    game, _, config = loaded
    opponent = bo.finite_opponent(np.array([[-1.0], [1.0]]), np.array([0.3, 0.7]), "nash")
    _, exact_value = oracle.best_response(0, opponent.support, opponent.weights)
    assert abs(exact_value) < 1e-2

    hyperparams = bo.br_hyperparams(game, 0, config)
    response = bo.train_best_response(game, 0, opponent, hyperparams, steps=50, epochs=20, seed=0)
    actions = np.asarray(response.opponent(game, 0, samples=4096).support)
    # `payoff(actions, support) @ weights` is one value per sampled action; the policy's
    # value is their mean.
    value = float(np.mean(oracle.payoff(actions, opponent.support) @ opponent.weights))
    assert value > exact_value - 0.15, value
    # ... by concentrating on the peaks, which is where the exact response sits.
    assert abs(float(np.mean(np.abs(actions))) - 1.0) < 0.15
