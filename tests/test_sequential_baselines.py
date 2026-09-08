"""Checks for the sequential baselines in `baselines/neural/sequential_*.py`.

Same standard as `tests/test_neural_baselines.py`: these are stochastic RL runs,
so what is pinned is what must hold whatever a seed's training did.

  * a mixture over policies is *played* as a mixture -- drawn once per hand and
    held -- and the Kuhn conversion that turns one into a behavioral strategy is
    exact, which is checked against the only thing that can check it: a mixed
    strategy's value is linear in its members' values, and the entrywise average
    that the conversion is easy to confuse with is not;
  * the supervised memory records the right player's decisions, from the right
    member, under masks the game says are legal;
  * both algorithms run end to end on all three sequential games and leave
    behind loadable weights.

Anything sharper -- "PSRO reaches exploitability X on Leduc" -- is a claim about
a run, not about the code.
"""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

_X64_BEFORE = jax.config.jax_enable_x64

from baselines.common import load_game  # noqa: E402
from baselines.neural import sequential_oracle as so  # noqa: E402
from baselines.neural.common import RunWriter  # noqa: E402
from baselines.neural.sequential_common import load_sequential_run  # noqa: E402
from baselines.neural.sequential_nfsp import DecisionReservoir, run_sequential_nfsp  # noqa: E402
from baselines.neural.sequential_psro import run_sequential_psro  # noqa: E402
from baselines.neural.sequential_scoring import (  # noqa: E402
    build_scorer,
    exact_kuhn_exploitability,
    has_exact_exploitability,
)
from games.kuhn_best_response import KuhnStrategy, bet_grid, game_value, mix_strategies  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)


@pytest.fixture(scope="module")
def kuhn():
    """Classic (fixed-bet) Kuhn: the grid holds one point, so `expl` is exact."""
    return load_sequential_run("configs/kuhn_classic.yaml")


def _tiny(config):
    """The same run config with a batch small enough for a test to afford."""
    import dataclasses

    return dataclasses.replace(config, ppo=dataclasses.replace(config.ppo, batch_size=64))


# --------------------------------------------------------------------------- config

def test_the_two_baseline_families_reject_each_other_s_games():
    with pytest.raises(ValueError, match="one-shot game"):
        load_sequential_run("configs/two_point.yaml")
    with pytest.raises(ValueError, match="sequential game"):
        load_game("configs/kuhn.yaml")


def test_exactness_is_claimed_only_where_it_exists(kuhn):
    assert has_exact_exploitability(kuhn[0])
    assert not has_exact_exploitability(load_sequential_run("configs/leduc.yaml")[0])
    assert not has_exact_exploitability(load_sequential_run("configs/sequential_blotto.yaml")[0])


# --------------------------------------------------------------------------- mixtures

def _random_kuhn_strategy(rng, num_cards, num_grid):
    bet = rng.random((num_cards, num_grid))
    check = rng.random(num_cards)
    total = check + bet.sum(axis=1)
    return KuhnStrategy(open_check=check / total, open_bet=bet / total[:, None],
                        call=rng.random((num_cards, num_grid)))


@pytest.mark.parametrize("player", [0, 1])
def test_mixing_kuhn_strategies_is_linear_in_the_members_values(player):
    """The property that *defines* a mixed strategy, and the one the naive
    entrywise average of behavioral strategies fails: playing member `k` with
    probability `w_k` must be worth `sum_k w_k * value(member k)`."""
    game = load_sequential_run("configs/kuhn.yaml")[0]
    grid = bet_grid(game, 9)
    rng = np.random.default_rng(0)
    members = [_random_kuhn_strategy(rng, game.num_cards, grid.shape[0]) for _ in range(2)]
    opponent = _random_kuhn_strategy(rng, game.num_cards, grid.shape[0])
    weights = [0.35, 0.65]

    def value(strategy):
        pair = (strategy, opponent) if player == 0 else (opponent, strategy)
        return float(game_value(game, grid, *pair))

    mixed = value(mix_strategies(members, weights, player))
    linear = sum(w * value(m) for w, m in zip(weights, members))
    assert abs(mixed - linear) < 1e-5

    naive = KuhnStrategy(
        open_check=sum(w * m.open_check for w, m in zip(weights, members)),
        open_bet=sum(w * m.open_bet for w, m in zip(weights, members)),
        call=sum(w * m.call for w, m in zip(weights, members)),
    )
    if player == 0:  # player 1 never acts twice on a line, so there the two agree
        assert abs(value(naive) - linear) > 1e-4


def test_mix_flattens_and_rescales_the_member_weights(kuhn):
    game, _, config = kuhn
    hyperparams = so.br_hyperparams(game, 0, config)
    left = so.initial_policy(game, 0, hyperparams, 0)
    right = so.population(hyperparams, [left.params[0], left.params[0]], [0.25, 0.75],
                          network=left.network)
    mixed = so.mix([left, right], [0.2, 0.8])
    assert len(mixed) == 3
    assert np.allclose(mixed.probs, [0.2, 0.8 * 0.25, 0.8 * 0.75])


def test_mix_rejects_a_member_of_a_different_shape(kuhn):
    import dataclasses

    game, _, config = kuhn
    hyperparams = so.br_hyperparams(game, 0, config)
    wider = dataclasses.replace(hyperparams, num_components=hyperparams.num_components + 1)
    with pytest.raises(ValueError, match="one architecture"):
        so.mix([so.initial_policy(game, 0, hyperparams, 0),
                so.initial_policy(game, 0, wider, 1)], [0.5, 0.5])


def test_a_mixture_of_one_policy_repeated_is_that_policy(kuhn):
    """Two copies of the same weights, any probabilities, must play identically --
    the check that gathering per-episode parameters is not scrambling members."""
    game, _, config = kuhn
    hyperparams = so.br_hyperparams(game, 0, config)
    policy = so.initial_policy(game, 0, hyperparams, 0)
    opponent = so.initial_policy(game, 1, so.br_hyperparams(game, 1, config), 1)
    doubled = so.population(hyperparams, [policy.params[0], policy.params[0]], [0.4, 0.6],
                            network=policy.network)

    evaluator = so.MixtureEvaluator(game, (policy.network, opponent.network))
    single, _ = evaluator.evaluate(policy, opponent, jax.random.PRNGKey(0), num_episodes=20_000)
    twice, _ = evaluator.evaluate(doubled, opponent, jax.random.PRNGKey(0), num_episodes=20_000)
    assert abs(single - twice) < 0.02


def test_the_exact_metric_agrees_with_the_trainers_own(kuhn):
    """A one-member mixture is a policy, so `exact_kuhn_exploitability` has to
    reproduce `training.kuhn_evaluation.evaluate_networks` exactly -- the number
    `train.py` prints for a self-play run, i.e. the same measurement."""
    from training.kuhn_evaluation import evaluate_networks

    game, _, config = kuhn
    policies = [so.initial_policy(game, p, so.br_hyperparams(game, p, config), p) for p in (0, 1)]
    ours = exact_kuhn_exploitability(game, (policies[0], policies[1]))
    theirs = evaluate_networks(game, (policies[0].network, policies[1].network),
                               (policies[0].params[0], policies[1].params[0]))
    assert abs(ours["expl"] - float(theirs["exploitability"])) < 1e-6
    assert abs(ours["value"] - float(theirs["value"])) < 1e-6


# --------------------------------------------------------------------------- oracle

def test_the_best_response_trainer_reports_and_moves(kuhn):
    game, _, config = kuhn
    hyperparams = so.br_hyperparams(game, 0, _tiny(config))
    opponent = so.mix([so.initial_policy(game, 1, hyperparams, 1),
                       so.initial_policy(game, 1, hyperparams, 2)], [0.5, 0.5])
    response = so.train_sequential_best_response(
        game, 0, opponent, hyperparams, steps=2, epochs=3, seed=0)
    assert len(response.history) == 6
    assert np.isfinite(response.br_value)
    before = so.initial_policy(game, 0, hyperparams, 0).params[0]
    moved = jax.tree_util.tree_reduce(
        lambda acc, leaf: acc or bool(np.any(np.asarray(leaf) != 0)),
        jax.tree_util.tree_map(lambda a, b: np.asarray(a) - np.asarray(b),
                               response.params, before), False)
    assert moved


def test_decision_rows_come_from_the_right_player_and_member(kuhn):
    game, _, config = kuhn
    hyperparams = [so.br_hyperparams(game, p, config) for p in (0, 1)]
    beta = so.initial_policy(game, 1, hyperparams[1], 11)
    pi = so.initial_policy(game, 1, hyperparams[1], 12)
    own = so.mix([beta, pi], [0.5, 0.5])
    opponent = so.initial_policy(game, 0, hyperparams[0], 0)

    rows = so.sample_decision_rows(game, 1, own, opponent, jax.random.PRNGKey(0), 512,
                                   own_member=0)
    assert rows["obs"].shape[1] == game.obs_dim(1)
    assert 0 < rows["obs"].shape[0] < 512 * game.max_steps
    # Every recorded action was legal at the infoset it was played in: the mask is
    # the expanded per-logit one, so a kind maps to the atom entry or to any
    # Gaussian component.
    num_atoms = hyperparams[1].num_atoms
    for mask, kind in zip(rows["action_mask"], rows["action_kind"]):
        assert mask[kind] if kind < num_atoms else mask[num_atoms:].any()
    # Halving the recorded member's probability roughly halves the rows.
    fewer = so.sample_decision_rows(game, 1, so.mix([beta, pi], [0.1, 0.9]), opponent,
                                    jax.random.PRNGKey(0), 512, own_member=0)
    assert fewer["obs"].shape[0] < rows["obs"].shape[0]


def test_decision_reservoir_keeps_capacity(kuhn):
    reservoir = DecisionReservoir(capacity=100, obs_dim=3, mask_width=4, action_dim=1, seed=0)
    rows = {"obs": np.ones((80, 3)), "action_mask": np.ones((80, 4), dtype=bool),
            "action_kind": np.zeros(80, dtype=np.int32), "raw_action": np.ones((80, 1))}
    reservoir.add(rows)
    assert reservoir.size == 80
    reservoir.add({k: np.concatenate([v] * 10) for k, v in rows.items()})
    assert reservoir.size == 100 and reservoir.seen == 880


# --------------------------------------------------------------------------- end to end

def test_sequential_psro_runs_and_saves_its_population(kuhn, tmp_path):
    game, _, config = kuhn
    config = _tiny(config)
    scorer = build_scorer(game, tuple(so.br_hyperparams(game, p, config) for p in (0, 1)),
                          episodes=2_000)
    writer = RunWriter(tmp_path / "psro", {"algorithm": "sequential_psro"})
    result = run_sequential_psro(game, config, rounds=2, br_steps=1, br_epochs=3,
                                 payoff_episodes=2_000, seed=0, scorer=scorer, writer=writer)
    writer.finish()

    assert [h["t"] for h in result["history"]] == [1, 2, 3]
    assert result["history"][-1]["population_0"] == 3       # seed policy + one per round
    assert result["payoff"].shape == (3, 3)
    assert all("expl" in row for row in result["history"])   # Kuhn: exact, every round
    assert np.isclose(result["meta"][0].sum(), 1.0)
    assert (tmp_path / "psro" / "params" / "player0_policy0" / "hyperparams.json").exists()
    saved = np.load(tmp_path / "psro" / "meta.npz")
    assert saved["payoff"].shape == (3, 3) and len(saved["meta_weights_1"]) == 3
    assert json.loads((tmp_path / "psro" / "meta.json").read_text())["algorithm"] == "sequential_psro"


def test_sequential_nfsp_runs_and_writes_a_loadable_average(kuhn, tmp_path):
    from training.best_response import load_frozen_policy

    game, _, config = kuhn
    config = _tiny(config)
    scorer = build_scorer(game, tuple(so.br_hyperparams(game, p, config) for p in (0, 1)),
                          episodes=2_000)
    writer = RunWriter(tmp_path / "nfsp", {"algorithm": "sequential_nfsp"})
    result = run_sequential_nfsp(game, config, rounds=2, br_steps=1, br_epochs=3,
                                 reservoir_episodes=256, sl_steps=20, sl_batch=32,
                                 seed=0, scorer=scorer, writer=writer)
    writer.finish()

    assert [h["t"] for h in result["history"]] == [1, 2]
    assert result["history"][-1]["reservoir"] > 0
    assert all("expl" in row and "br_expl" in row for row in result["history"])
    # The averages are written where `best_response.py` looks, so a finished run
    # can be scored offline by the repo's own tool rather than an NFSP-aware loader.
    frozen = load_frozen_policy(tmp_path / "nfsp" / "average_checkpoint", 2, player=1)
    assert frozen.hyperparams.num_components == config.network.num_components


@pytest.mark.parametrize("config_path", ["configs/leduc.yaml", "configs/sequential_blotto.yaml"])
def test_both_solvers_run_on_the_games_with_no_exact_best_response(config_path):
    """Leduc and Blotto have no tree best response, so a run there must still
    produce its cost columns and its RL bound without ever reaching for `expl`."""
    game, _, config = load_sequential_run(config_path)
    config = _tiny(config)
    hyperparams = tuple(so.br_hyperparams(game, p, config) for p in (0, 1))
    scorer = build_scorer(game, hyperparams, score_every=1, br_steps=1, br_epochs=2,
                          episodes=1_000)
    psro = run_sequential_psro(game, config, rounds=1, br_steps=1, br_epochs=2,
                              payoff_episodes=1_000, seed=0, scorer=scorer)
    assert "expl" not in psro["history"][-1]
    assert np.isfinite(psro["history"][-1]["expl_lb"])

    nfsp = run_sequential_nfsp(game, config, rounds=1, br_steps=1, br_epochs=2,
                               reservoir_episodes=128, sl_steps=10, sl_batch=16, seed=0,
                               scorer=scorer)
    assert "expl" not in nfsp["history"][-1]
    assert np.isfinite(nfsp["history"][-1]["h2h"])
