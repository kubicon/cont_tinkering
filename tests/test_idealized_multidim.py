"""Checks for `run_idealized.py` above one dimension.

The solver used to be scalar-only, with a separate script for the multidimensional
decoy-well game. Both now run through the same loop on a `(K, d)` mean and a
`(K, d, d)` Cholesky factor, so the properties worth pinning are the ones that
could silently break in the lift:

  * the `d`-dimensional backends agree with each other on a game where two of them
    apply, and the multidim closed form reduces to the scalar one at `d == 1`,
  * `full_covariance` actually allocates off-diagonal entries that something can
    move, while a separable game's payoff leaves an uncorrelated factor exactly
    where it is,
  * and the quadrature backend refuses a grid it cannot hold instead of dying in
    the allocator, since `grid_points^(2d)` reaches terabytes at the 1-D default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# See `test_idealized_sampled.py`: importing the solver turns x64 on process-wide, so
# the flag is read before the import and restored after it, then set per test.
_X64_BEFORE = jax.config.jax_enable_x64

from games.configs import GAME_CONFIGS  # noqa: E402 -- must follow the line above
from run_idealized import (  # noqa: E402
    IdealizedSection,
    Params,
    SolverConfig,
    build_backend,
    build_init,
    run,
)

jax.config.update("jax_enable_x64", _X64_BEFORE)


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _X64_BEFORE)


def _cfg(**idealized) -> SolverConfig:
    return SolverConfig(chunks=(1,), num_components=2,
                        idealized=IdealizedSection(**idealized))


def _well(dim: int, decoys=()):
    return GAME_CONFIGS["multidim_decoy_well"](
        dim=dim, peaks=(-1.0, 1.0), decoys=list(decoys), peak_width=0.1, coupling=1.0
    ).build()


def _centered(q) -> np.ndarray:
    """`q` up to the additive constant the categorical mirror update cannot see."""
    q = np.asarray(q, dtype=float)
    return q - q.mean()


def _params(means, stds) -> Params:
    """A mixture with the given per-component means and (diagonal) per-axis stds."""
    means = jnp.asarray(means, dtype=jnp.float64)
    stds = jnp.asarray(stds, dtype=jnp.float64)
    scale = jax.vmap(jnp.diag)(stds)
    return Params(logits=jnp.zeros(means.shape[0], dtype=jnp.float64),
                  means=means, scale_tril=scale)


# --- the backends agree with each other -------------------------------------


def test_multidim_closed_form_reduces_to_the_scalar_one_at_dim_1() -> None:
    """`dim=1` is the seam between the two closed forms; they must meet there."""
    game = _well(1, decoys=[(0.0, 0.7, 0.45)])
    scalar = build_backend(game, _cfg(backend="closed_form"))
    multi = build_backend(game, _cfg(backend="closed_form_multidim"))
    p0 = _params([[-1.4], [0.3]], [[0.25], [0.4]])
    p1 = _params([[0.9], [-0.2]], [[0.3], [0.35]])

    assert float(multi.expected_payoff(p0, p1)) == pytest.approx(
        float(scalar.expected_payoff(p0, p1)), rel=1e-10)
    for sign, p, opp in ((+1.0, p0, p1), (-1.0, p1, p0)):
        assert np.asarray(multi.component_q(p, opp, sign)) == pytest.approx(
            np.asarray(scalar.component_q(p, opp, sign)), rel=1e-10)
    # the two search their best-response grids at different resolutions, so the
    # exploitability agrees to grid error rather than to machine precision
    assert float(multi.exploitability(p0, p1)) == pytest.approx(
        float(scalar.exploitability(p0, p1)), abs=1e-3)


def test_sampled_backend_matches_the_exact_payoff_in_two_dimensions() -> None:
    """The backend that carries the solver above 1-D, against the one that is exact."""
    game = _well(2)
    sampled = build_backend(game, _cfg(backend="sampled", samples=200_000, grid_points=41))
    exact = build_backend(game, _cfg(backend="closed_form_multidim", grid_points=41))
    p0 = _params([[-1.1, 0.8], [0.4, -0.9]], [[0.3, 0.4], [0.35, 0.3]])
    p1 = _params([[0.7, -1.2], [-0.5, 0.6]], [[0.4, 0.3], [0.3, 0.45]])
    noise = sampled.draw_noise(jax.random.PRNGKey(0), p0, p1)

    got = sampled.expected_payoff(sampled.handle(p0, noise, 0), sampled.handle(p1, noise, 1))
    assert float(got) == pytest.approx(float(exact.expected_payoff(p0, p1)), abs=0.02)

    for sign, p, opp, player in ((+1.0, p0, p1, 1), (-1.0, p1, p0, 0)):
        q_hat = sampled.component_q(p, sampled.handle(opp, noise, player), sign, noise)
        q = exact.component_q(p, opp, sign)
        # the closed form drops the opponent's own well, a constant across components
        # that the softmax update cannot see; compare what the update does see
        assert _centered(q_hat) == pytest.approx(_centered(q), abs=0.05)


def test_quadrature_matches_the_closed_form_in_two_dimensions() -> None:
    """A coarse tensor grid is still the same integral, to grid error."""
    game = _well(2)
    quad = build_backend(game, _cfg(backend="quadrature", grid_points=101, std_max=0.5))
    exact = build_backend(game, _cfg(backend="closed_form_multidim", std_max=0.5))
    p0 = _params([[-1.0, 1.0], [1.0, -1.0]], [[0.2, 0.2], [0.2, 0.2]])
    p1 = _params([[0.9, -0.8], [-1.1, 1.2]], [[0.25, 0.2], [0.2, 0.25]])

    got = quad.expected_payoff(quad.handle(p0), quad.handle(p1))
    assert float(got) == pytest.approx(float(exact.expected_payoff(p0, p1)), abs=0.02)


# --- the covariance the lift exists for -------------------------------------


def test_full_covariance_allocates_off_diagonals_that_a_diagonal_run_cannot() -> None:
    game = _well(2)
    diagonal, _ = build_init(game, _cfg(full_covariance=False))
    full, _ = build_init(game, _cfg(full_covariance=True))
    # both start uncorrelated -- the flag changes what may move, not where it starts
    for p in (diagonal, full):
        assert p.scale_tril.shape == (2, 2, 2)
        assert np.allclose(np.asarray(p.scale_tril)[:, 1, 0], 0.0)

    # a term that couples a player's own coordinates: the mixture's differential
    # entropy is not a sum over axes, so it is the one force in this game that
    # reaches an off-diagonal entry at all
    backend = build_backend(game, _cfg(full_covariance=True, grid_points=41, std_max=0.5))
    correlated = full._replace(scale_tril=full.scale_tril.at[:, 1, 0].set(0.1))
    grad = jax.grad(backend.entropy)(correlated).scale_tril
    assert np.max(np.abs(np.asarray(grad)[:, 1, 0])) > 1e-6


def test_a_diagonal_parametrization_stays_exactly_diagonal() -> None:
    """`full_covariance: false` must mean the off-diagonals are not parameters.

    `training/mixture.py` gets this for free -- it never allocates them -- but here
    the factor *is* the parameter vector, so the step has to be masked. Without the
    mask a term the payoff does not separate over (the mixture's differential
    entropy) has a nonzero gradient on an off-diagonal entry even at a factor that
    is exactly diagonal, and a `full_covariance: false` run would quietly train
    correlations the config says it does not have.
    """
    game = GAME_CONFIGS["quadratic"](dim=2, coupling=0.5, bound=3.0).build()

    def final_factor(full_covariance: bool):
        cfg = SolverConfig(
            chunks=(50,) * 4, lr=0.05, num_components=2, gaussian_entropy_coef=0.1,
            category_entropy_coef=0.1, magnet_category_kl_coef=0.2, magnet_interval=50,
            idealized=IdealizedSection(backend="quadrature", grid_points=41, std_max=0.5,
                                       full_covariance=full_covariance, verbose=False),
        )
        p0, p1 = build_init(game, cfg)
        return np.asarray(run(game, cfg, p0, p1)[0].scale_tril)

    assert final_factor(False)[:, 1, 0] == pytest.approx(0.0, abs=0.0)
    # and the flag is not merely cosmetic: with the entries allocated, the same run
    # does move them (this game's *payoff* cannot, but its entropy term can)
    assert np.max(np.abs(final_factor(True)[:, 1, 0])) > 1e-4


def test_a_separable_payoff_does_not_move_an_uncorrelated_factor_off_the_diagonal() -> None:
    """Why `full_covariance` is inert on `multidim_decoy_well` (see MultiDim.md).

    The payoff is a sum of per-coordinate terms, so its expectation depends only on
    the per-axis marginal variances -- and an off-diagonal entry enters those only
    through its own square (`Sigma_ii = sum_j A_ij^2`). Its gradient therefore
    *vanishes at zero*: a run that starts uncorrelated, as `build_init` always does,
    can never acquire a correlation from this payoff, whatever `full_covariance`
    allocates. Off zero the entry does feel the payoff, through the row norm alone
    -- see `tests/test_gaussian_scale.py`, which pins the rotation invariance that
    is the exact statement of "the payoff cannot see correlations".
    """
    game = _well(2)
    backend = build_backend(game, _cfg(backend="closed_form_multidim", full_covariance=True))
    p0 = _params([[-1.0, 0.5], [0.6, -0.7]], [[0.3, 0.4], [0.35, 0.3]])
    p1 = _params([[0.8, -0.6], [-0.9, 0.4]], [[0.4, 0.3], [0.3, 0.4]])

    payoff_grad = jax.grad(lambda p: backend.expected_payoff(p, p1))
    assert np.asarray(payoff_grad(p0).scale_tril)[:, 1, 0] == pytest.approx(0.0, abs=1e-12)

    correlated = p0._replace(scale_tril=p0.scale_tril.at[:, 1, 0].set(0.15))
    assert np.max(np.abs(np.asarray(payoff_grad(correlated).scale_tril)[:, 1, 0])) > 1e-6


# --- the grid backend's hard limit ------------------------------------------


def test_quadrature_refuses_a_grid_it_cannot_hold() -> None:
    """`grid_points^(2d)` is terabytes at the 1-D default, and must say so."""
    game = _well(2)
    with pytest.raises(ValueError, match="max_quadrature_gib"):
        build_backend(game, _cfg(backend="quadrature", grid_points=801))
    # the same grid is unremarkable in 1-D, where the default has always run
    build_backend(_well(1), _cfg(backend="quadrature", grid_points=801))


# --- end to end --------------------------------------------------------------


def test_a_two_dimensional_run_reaches_the_diagonal_nash() -> None:
    """MultiDim.md's exp1, shortened: two peaks per axis, no decoy, K=2.

    Not a tuning check -- it is the end-to-end path (init, both heads' updates, the
    natural-gradient step in factor coordinates, the metrics) exercised at `d > 1`.
    The config is exp1's, and the run reproduces its published 0.0492.
    """
    game = _well(2)
    cfg = SolverConfig(
        chunks=(200,) * 30, lr=0.05, num_components=2, magnet_interval=200,
        magnet_category_kl_coef=0.2, magnet_gaussian_kl_coef=0.2,
        idealized=IdealizedSection(gaussian_entropy="component", kl_weighting="uniform",
                                   std_max=1.0, verbose=False),
    )
    p0, p1 = build_init(game, cfg)
    final0, _, history, _ = run(game, cfg, p0, p1)

    assert history[-1]["expl"] < 0.1 < history[0]["expl"]
    assert history[-1]["expl"] == pytest.approx(0.0492, abs=5e-4)
    # both components end on the diagonal, one per peak, in every coordinate
    means = np.sort(np.asarray(final0.means), axis=0)
    assert means == pytest.approx(np.array([[-1.0, -1.0], [1.0, 1.0]]), abs=0.05)
