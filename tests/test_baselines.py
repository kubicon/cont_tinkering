"""Checks for the representation baselines in `baselines/`.

These solvers exist to be *believed* when they disagree with the mixture solver --
"tabular MMD reaches the Nash on the game where our K=2 mixture is trapped" is only
an argument if the tabular solver is right. So what is pinned here is correctness
against ground truth, not behaviour:

  * the shared exploitability metric is ~0 at the analytically known Nash of
    `MultiPointGame` (weights over peaks) and positive away from it,
  * the restricted-game LP reproduces the textbook matching-pennies equilibrium,
  * and each of the three algorithms, run briefly on the two-peak game whose Nash is
    known in closed form, ends near that Nash rather than merely "somewhere".
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import jax
import pytest

# See `test_idealized_multidim.py`: importing a solver turns x64 on process-wide, so
# the flag is read before the import and restored after it.
_X64_BEFORE = jax.config.jax_enable_x64

from baselines.common import GridOracle, cluster_atoms, load_game  # noqa: E402
from baselines.double_oracle import run_double_oracle, solve_matrix_game  # noqa: E402
from baselines.grid_mmd import run_grid_mmd  # noqa: E402
from baselines.particle_mean_field import run_particle_mean_field  # noqa: E402
from games.examples import MultiPointGame  # noqa: E402

jax.config.update("jax_enable_x64", _X64_BEFORE)

# Peaks at -1 and +1 with Nash weights 0.3/0.7 -- `MultiPointGame`'s docstring proves
# each player's Nash marginal over the peaks is exactly `weights`.
PEAKS = (-1.0, 1.0)
WEIGHTS = (0.3, 0.7)
NASH_SUPPORT = np.array([[-1.0], [1.0]])
NASH_WEIGHTS = np.array(WEIGHTS)


@pytest.fixture(scope="module")
def game():
    jax.config.update("jax_enable_x64", True)
    return MultiPointGame(peaks=PEAKS, weights=WEIGHTS, width=0.1, coupling=1.0)


@pytest.fixture(scope="module")
def oracle(game):
    return GridOracle(game, points=401)


def test_exploitability_vanishes_at_the_known_nash(oracle):
    expl = oracle.exploitability(NASH_SUPPORT, NASH_WEIGHTS, NASH_SUPPORT, NASH_WEIGHTS)
    assert expl < 5e-3, expl


def test_exploitability_is_positive_off_the_nash(oracle):
    """The *wrong* mix over the right support is still exploitable -- which is the
    whole reason these games are interesting, and the property a metric that only
    checked the support would miss."""
    wrong = np.array([0.7, 0.3])
    assert oracle.exploitability(NASH_SUPPORT, wrong, NASH_SUPPORT, wrong) > 0.1


def test_best_response_finds_a_peak(oracle):
    """Against the Nash mix the coupling term vanishes identically, leaving the well:
    the best reply is a peak, and it earns exactly the value of the game (zero here,
    since both players sit on equally tall peaks and the well terms cancel) -- which
    is the same statement as "the Nash is unexploitable", read off the oracle."""
    action, value = oracle.best_response(0, NASH_SUPPORT, NASH_WEIGHTS)
    assert np.min(np.abs(action[0] - np.array(PEAKS))) < 0.05
    assert abs(value - oracle.value(NASH_SUPPORT, NASH_WEIGHTS,
                                    NASH_SUPPORT, NASH_WEIGHTS)) < 1e-3


def test_matrix_game_lp_matches_matching_pennies():
    x, y, value = solve_matrix_game(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert np.allclose(x, 0.5, atol=1e-9)
    assert np.allclose(y, 0.5, atol=1e-9)
    assert abs(value) < 1e-9


def test_matrix_game_lp_matches_a_dominated_row():
    """A game with a pure equilibrium: row 1 dominates row 0, column 0 dominates
    column 1, so the value is the corner entry."""
    x, y, value = solve_matrix_game(np.array([[0.0, -1.0], [2.0, 1.0]]))
    assert np.allclose(x, [0.0, 1.0], atol=1e-9)
    assert np.allclose(y, [0.0, 1.0], atol=1e-9)
    assert abs(value - 1.0) < 1e-9


def test_cluster_atoms_merges_adjacent_cells():
    support = np.array([[-1.01], [-1.0], [-0.99], [1.0], [1.01]])
    weights = np.array([0.1, 0.1, 0.1, 0.35, 0.35])
    centers, masses = cluster_atoms(support, weights, radius=0.05)
    order = np.argsort(centers[:, 0])
    assert centers.shape[0] == 2
    assert np.allclose(masses[order], [0.3, 0.7], atol=1e-9)
    assert np.allclose(centers[order][:, 0], [-1.0, 1.005], atol=1e-6)


def _nash_error(support, weights, radius=0.08):
    """Total-variation-ish distance to the known Nash after merging adjacent atoms:
    `max_k |mass near peak k - weights_k|`, plus whatever mass sits nowhere near a peak."""
    centers, masses = cluster_atoms(support, weights, radius)
    near = np.abs(centers[:, 0][:, None] - np.array(PEAKS)[None, :]) < 0.1   # (C, 2)
    mass_per_peak = masses @ near
    stray = float(np.sum(masses) - np.sum(mass_per_peak))
    return float(np.max(np.abs(mass_per_peak - NASH_WEIGHTS))) + stray


def test_double_oracle_reaches_the_nash(game, oracle):
    result = run_double_oracle(game, oracle, iters=12, tol=1e-3, dedup=1e-3, polish=True)
    assert result.converged, result.history[-1]
    assert result.history[-1]["expl"] < 1e-3
    assert _nash_error(result.support_0, result.weights_0) < 0.02
    assert _nash_error(result.support_1, result.weights_1) < 0.02


def test_grid_mmd_reaches_the_nash(game, oracle):
    result = run_grid_mmd(game, oracle, iters=6000, lr=10.0, tau=1.0, magnet_interval=200)
    assert result.history[-1]["expl"] < 0.05, result.history[-1]
    assert _nash_error(result.grid_0, result.weights_0) < 0.05
    assert _nash_error(result.grid_1, result.weights_1) < 0.05


def test_grid_mmd_exploitability_matches_the_shared_oracle(game, oracle):
    """`grid_mmd` computes its metric from the payoff matrix instead of calling the
    oracle; the two must agree, or its curve cannot be plotted next to the others."""
    result = run_grid_mmd(game, oracle, iters=200, lr=10.0, tau=1.0, magnet_interval=200)
    from_oracle = oracle.exploitability(result.grid_0, result.weights_0,
                                        result.grid_1, result.weights_1)
    assert abs(from_oracle - result.history[-1]["expl"]) < 1e-9


def test_particle_mean_field_reaches_the_nash(game, oracle):
    result = run_particle_mean_field(game, oracle, particles=32, iters=4000,
                                     lr_position=1e-2, lr_weight=10.0, tau=1.0,
                                     magnet_interval=200, seed=0)
    assert result.history[-1]["expl"] < 0.05, result.history[-1]
    assert _nash_error(result.support_0, result.weights_0) < 0.05
    assert _nash_error(result.support_1, result.weights_1) < 0.05


def test_frozen_weights_particles_cannot_fix_the_mixture(game, oracle):
    """The pure-Wasserstein ablation: with the birth-death term off, the weights stay
    uniform, so the 0.3/0.7 Nash mix is unreachable however well the positions move --
    the check that `--freeze-weights` really does disable the Fisher-Rao half."""
    result = run_particle_mean_field(game, oracle, particles=32, iters=2000,
                                     lr_position=1e-2, freeze_weights=True, seed=0)
    assert np.allclose(result.weights_0, 1.0 / 32, atol=1e-12)
    assert result.history[-1]["expl"] > 0.1


def test_load_game_reads_a_config(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("game:\n  name: multi_point\n  peaks: [-1.0, 1.0]\n  weights: [0.3, 0.7]\n")
    loaded, config = load_game(path)
    assert isinstance(loaded, MultiPointGame)
    assert config.weights == (0.3, 0.7)


def test_load_game_rejects_an_unknown_field(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("game:\n  name: multi_point\n  peeks: [-1.0, 1.0]\n")
    with pytest.raises(ValueError, match="peeks"):
        load_game(path)


def test_checkpoints_round_trip(game, oracle, tmp_path):
    """A checkpoint has to come back as the strategy that was written, and be scorable
    without the algorithm that produced it -- that is the whole reason the three
    baselines share one format."""
    from baselines.common import CheckpointWriter, load_checkpoints, latest_checkpoint

    writer = CheckpointWriter(tmp_path / "ck")
    result = run_grid_mmd(game, oracle, iters=200, lr=10.0, tau=1.0, magnet_interval=200,
                          log_every=50, checkpoint_fn=writer)
    writer.write_index({"algorithm": "grid_mmd"})

    checkpoints = load_checkpoints(tmp_path / "ck")
    assert [c.t for c in checkpoints] == [0, 50, 100, 150, 200]
    last = latest_checkpoint(tmp_path / "ck")
    assert np.allclose(last.weights_0, result.weights_0)
    assert np.allclose(last.extra["avg_weights_0"], result.avg_weights_0)
    # Re-scored from the file alone, it reproduces the metric the run reported.
    expl = oracle.exploitability(last.support_0, last.weights_0, last.support_1, last.weights_1)
    assert abs(expl - result.history[-1]["expl"]) < 1e-9

    index = json.loads((tmp_path / "ck" / "index.json").read_text())
    assert [e["t"] for e in index["checkpoints"]] == [0, 50, 100, 150, 200]


def test_double_oracle_checkpoints_track_the_growing_support(game, oracle, tmp_path):
    from baselines.common import CheckpointWriter, load_checkpoints

    writer = CheckpointWriter(tmp_path / "ck")
    run_double_oracle(game, oracle, iters=4, tol=1e-3, dedup=1e-3, checkpoint_fn=writer)
    sizes = [c.support_0.shape[0] for c in load_checkpoints(tmp_path / "ck")]
    assert sizes == sorted(sizes) and sizes[0] == 1 and len(set(sizes)) > 1


def _load_driver():
    """`experiments/one_shot_tabular/run_baselines.py` is a script, not a module."""
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "one_shot_tabular" \
        / "run_baselines.py"
    import sys

    spec = importlib.util.spec_from_file_location("run_baselines", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `Settings` is a dataclass, and `dataclasses` resolves
    # its annotations through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_driver_writes_a_cell_and_then_skips_it(tmp_path):
    """The sweep must be restartable: a finished cell is re-reported from its
    `meta.json`, not re-run (which is what makes adding a game later cheap)."""
    driver = _load_driver()
    settings = driver.Settings(grid=201, iters=100, checkpoints=2)
    config = "configs/two_point.yaml"

    row = driver.run_cell(config, "grid_mmd", 0, settings, tmp_path)
    assert row["status"] == "ok"
    run_dir = tmp_path / "two_point" / "grid_mmd" / "seed0"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["settings"]["iters"] == 100
    assert list(meta["game_config"]["peaks"]) == list(PEAKS)   # the game is recorded too
    assert len(list((run_dir / "checkpoints").glob("*.npz"))) == meta["checkpoints"] > 0
    assert len(json.loads((run_dir / "history.json").read_text())) > 1

    stamp = (run_dir / "meta.json").stat().st_mtime_ns
    again = driver.run_cell(config, "grid_mmd", 0, settings, tmp_path)
    assert again["expl"] == row["expl"]
    assert (run_dir / "meta.json").stat().st_mtime_ns == stamp   # not rewritten


def test_driver_records_a_failed_cell_instead_of_raising(tmp_path):
    """One game the discretization cannot handle must not take the rest of an
    overnight sweep with it."""
    driver = _load_driver()
    settings = driver.Settings(grid=201, iters=10)
    row = driver.run_cell("configs/blotto.yaml", "grid_mmd", 0, settings, tmp_path)
    assert row["status"] == "failed"
    assert "SimplexSpace" in row["error"] or "simplex" in row["error"].lower()
    assert (tmp_path / "blotto" / "grid_mmd" / "seed0" / "error.txt").exists()
