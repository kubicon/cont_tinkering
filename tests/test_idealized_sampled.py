"""Checks for `run_idealized.py`'s `sampled` backend -- the middle ground with `train.py`.

The backend keeps the exact tabular mirror step and replaces only the payoff
integral by a Monte-Carlo average over `samples` action draws, so the property
that makes it worth having is that it is *the same solver* as the quadrature
one, up to sampling error. Pinned here:

  * every `q_estimator` estimates the same per-component q-values the exact
    backend computes,
  * both `grad_estimator`s estimate the same Gaussian-head gradient,
  * a large-sample run reproduces a quadrature run end to end,
  * and the one term that does NOT converge to its exact counterpart -- the
    sampled entropy bonus, whose gradient is mean-zero, exactly as
    `mixture_ppo_loss`'s is -- stays mean-zero, so that surprise is documented
    rather than discovered.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Importing the solver (`idealized_mmd` and `run_idealized` both do it) turns x64 on
# process-wide, and a test module is imported at collection time -- i.e. before every
# other module's tests run. Left alone that would silently promote the whole suite's
# dtypes, so the flag is read *before* those imports, put back after them, and switched
# on per test instead.
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
    """The solver runs in double precision; keep that contained to this module."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _X64_BEFORE)


def _game():
    return GAME_CONFIGS["quadratic"](dim=1, coupling=0.5, bound=3.0).build()


def _scalar_params(logits, means, stds) -> Params:
    """A 1-D mixture in the solver's `(means (K, 1), scale_tril (K, 1, 1))` shape."""
    return Params(logits=jnp.asarray(logits),
                  means=jnp.asarray(means)[:, None],
                  scale_tril=jnp.asarray(stds)[:, None, None])


def _params(seed: int = 0) -> tuple[Params, Params]:
    """Two deliberately asymmetric 2-component mixtures (nothing at the Nash)."""
    return (_scalar_params([0.4, -0.9], [-1.2, 0.7], [0.6, 0.9]),
            _scalar_params([-0.3, 0.5], [0.4, 1.6], [0.8, 0.5]))


def _cfg(**idealized) -> SolverConfig:
    return SolverConfig(chunks=(1,), num_components=2,
                        idealized=IdealizedSection(**idealized))


def _backends(**idealized):
    """A sampled backend and the exact one it should agree with."""
    game = _game()
    cfg = _cfg(backend="sampled", **idealized)
    sampled = build_backend(game, cfg)
    return game, sampled, sampled.metrics


@pytest.mark.parametrize("q_estimator", ["responsibility", "per_component", "onpolicy"])
def test_q_estimators_match_the_exact_q_values(q_estimator: str) -> None:
    game, sampled, exact = _backends(samples=200_000, q_estimator=q_estimator)
    p0, p1 = _params()
    noise = sampled.draw_noise(jax.random.PRNGKey(0), p0, p1)

    for sign, p, opp in ((+1.0, p0, p1), (-1.0, p1, p0)):
        player = 0 if sign > 0 else 1
        got = sampled.component_q(p, sampled.handle(opp, noise, 1 - player), sign, noise)
        want = exact.component_q(p, exact.handle(opp), sign)
        # q enters the simplex update through `eta * q`, so absolute error is what matters
        assert jnp.max(jnp.abs(got - want)) < 0.05, (q_estimator, got, want)


@pytest.mark.parametrize("grad_estimator", ["pathwise", "score"])
def test_gaussian_gradients_match_the_exact_ones(grad_estimator: str) -> None:
    game, sampled, exact = _backends(samples=400_000, grad_estimator=grad_estimator)
    p0, p1 = _params()
    noise = sampled.draw_noise(jax.random.PRNGKey(1), p0, p1)

    h_opp_sampled = sampled.handle(p1, noise, 1)
    g_hat = jax.grad(lambda pp: sampled.expected_payoff(sampled.handle(pp, noise, 0),
                                                        h_opp_sampled))(p0)
    h_opp_exact = exact.handle(p1)
    g = jax.grad(lambda pp: exact.expected_payoff(exact.handle(pp), h_opp_exact))(p0)

    # the score estimator is unbiased but far noisier, so it gets a looser leash
    tol = 0.05 if grad_estimator == "pathwise" else 0.5
    assert jnp.max(jnp.abs(g_hat.means - g.means)) < tol, (g_hat.means, g.means)
    assert jnp.max(jnp.abs(g_hat.scale_tril - g.scale_tril)) < tol, (g_hat.scale_tril, g.scale_tril)


def test_sampled_entropy_bonus_has_a_mean_zero_gradient() -> None:
    """`entropy_source: sampled` is what `mixture_ppo_loss` adds -- `-log p(a)` at a
    *detached* sampled action -- and `E_{a~p}[grad -log p(a)]` is `-grad int p = 0`.

    So the sampled entropy bonus supplies variance, not the outward push on the stds
    that the exact term supplies. This is a property of the trainer's objective, not a
    bug in the backend, and `entropy_source: exact` is the knob that isolates it.
    """
    game, sampled, exact = _backends(samples=100_000)
    p0, p1 = _params()

    def sampled_grad(seed: int):
        noise = sampled.draw_noise(jax.random.PRNGKey(seed), p0, p1)
        return jax.grad(lambda pp: sampled.entropy(pp, noise, 0))(p0).scale_tril[:, 0, 0]

    averaged = jnp.mean(jnp.stack([sampled_grad(s) for s in range(16)]), axis=0)
    exact_grad = jax.grad(exact.entropy)(p0).scale_tril[:, 0, 0]

    assert jnp.max(jnp.abs(averaged)) < 0.05, averaged
    # the exact term is a real force pushing every std outward -- weighted by the
    # component's probability, so the minority one feels much less of it than the
    # dominant one, but no component ever feels a push inward.
    assert jnp.min(exact_grad) > 0.0, exact_grad
    assert jnp.sum(exact_grad) > 10 * jnp.abs(jnp.sum(averaged)), (exact_grad, averaged)


def test_large_sample_run_reproduces_the_quadrature_run() -> None:
    """The whole point of the backend: `samples -> inf` is the quadrature solver.

    Run with `entropy_source: exact` so the only difference under test is the payoff
    integral -- see the entropy test above for why the sampled term never converges to
    its exact counterpart.
    """
    game = _game()
    shared = dict(chunks=(100,) * 5, lr=0.05, num_components=2,
                  gaussian_entropy_coef=0.1, category_entropy_coef=0.1,
                  magnet_category_kl_coef=0.2, magnet_interval=100)

    exact_cfg = SolverConfig(**shared, idealized=IdealizedSection(backend="quadrature",
                                                                 verbose=False))
    p0, p1 = build_init(game, exact_cfg)
    _, _, exact_history, _ = run(game, exact_cfg, p0, p1)

    sampled_cfg = SolverConfig(**shared, idealized=IdealizedSection(
        backend="sampled", samples=200_000, entropy_source="exact", verbose=False))
    _, _, sampled_history, _ = run(game, sampled_cfg, p0, p1)

    for exact_row, sampled_row in zip(exact_history, sampled_history):
        assert sampled_row["expl"] == pytest.approx(exact_row["expl"], abs=0.02)
        for field in ("means0", "std0", "w0", "means1", "std1", "w1"):
            # means/stds are per-component *vectors* now (one entry per action
            # dimension), so compare them flattened
            got = np.ravel(np.asarray(sampled_row[field], dtype=float))
            want = np.ravel(np.asarray(exact_row[field], dtype=float))
            assert got == pytest.approx(want, abs=0.02), field


def test_sampled_backend_rejects_unknown_estimators() -> None:
    game = _game()
    for field in ("q_estimator", "grad_estimator", "entropy_source"):
        with pytest.raises(ValueError, match=field):
            build_backend(game, _cfg(backend="sampled", **{field: "nope"}))
