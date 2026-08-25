"""Exp 3 — how the monotone island and the empirical basin scale with coupling c.

Game: MultiPointGame(peaks=(-1,0,1), width=0.08, coupling=c), K=3 components.
(Three peaks so the feature map contains u^2 — a *curved* feature. With two
peaks the single feature is linear, f'' = 0, and the coupling cannot bend the
own-player Hessian at all: the island size is then c-independent by
construction. That itself is a theory prediction, noted in THEORY.md.)

Measured per coupling c:

(a) **Certified monotone radius r_mono**: largest perturbation radius r such
    that for all sampled configurations with both players' means at
    peaks + r*(unit vector), weights uniform, std at the empirical fixed-point
    value, both players' own-mean Hessian blocks stay negative semidefinite
    (block-skew lemma => field monotone there).
    Theory: r_mono ~ min( sqrt(w^2+s^2) [well concavity cap],
                          scale at which c * F(nu) * f'' overturns D_s'' ).
    For w=0.08 the well curvature h/w^2 ~ 156 dominates any moderate c, so
    r_mono should be ~flat in c and collapse only for c = O(100).

(b) **Empirical basin radius r_basin**: convergence rate of the actual MMD
    dynamics from inits means = peaks + r*(unit vector) (both players,
    independent directions), as a function of r. The funnel outside the island
    is where the coupling force c*F(nu)*f'(mu) competes with the (tiny,
    exponentially-decaying) well pull, so r_basin CAN degrade with c even
    while r_mono stays flat.

Run:  .venv/bin/python theory/exp3_coupling_sweep.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from games.examples import MultiPointGame
from idealized_mmd import (MMDConfig, _tail_expl, expected_payoff, make_init,
                           run)

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PEAKS = np.array([-1.0, 0.0, 1.0])
WIDTH = 0.08
CFG = MMDConfig(lr=0.05, magnet_coef=0.2, magnet_interval=200, steps=12000)
STD_FP = 0.007  # empirical fixed-point std from exp2 (same lr/tau)


def unit(rng, k=3):
    v = rng.normal(size=k)
    return v / np.linalg.norm(v)


def hessian_nsd_fraction(game, r, rng, n_samples=100, tol=1e-9):
    """Fraction of sampled r-perturbed configs where BOTH own-mean Hessians are NSD."""
    logstd = jnp.full(3, jnp.log(STD_FP))
    logits = jnp.zeros(3)

    @jax.jit
    def max_eigs(m0, m1):
        p0 = make_init(m0)._replace(log_std=logstd, logits=logits)
        p1 = make_init(m1)._replace(log_std=logstd, logits=logits)
        h0 = jax.hessian(lambda m: expected_payoff(p0._replace(means=m), p1, game))(p0.means)
        h1 = jax.hessian(lambda m: -expected_payoff(p0, p1._replace(means=m), game))(p1.means)
        return jnp.max(jnp.linalg.eigvalsh(h0)), jnp.max(jnp.linalg.eigvalsh(h1))

    ok = 0
    for _ in range(n_samples):
        m0 = jnp.asarray(PEAKS + r * unit(rng))
        m1 = jnp.asarray(PEAKS + r * unit(rng))
        e0, e1 = max_eigs(m0, m1)
        ok += bool(max(float(e0), float(e1)) <= tol)
    return ok / n_samples


def basin_fraction(game, r, rng, n_dirs=6):
    conv = 0
    for _ in range(n_dirs):
        i0 = make_init((PEAKS + r * unit(rng)).tolist())
        i1 = make_init((PEAKS + r * unit(rng)).tolist())
        _, _, h = run(game, CFG, i0, i1, log_every=500)
        conv += bool(_tail_expl(h) < 0.1)
    return conv / n_dirs


def main():
    out = {"width": WIDTH, "std_fp": STD_FP, "mono": {}, "basin": {}}

    # ---- (a) certified monotone radius, incl. extreme c --------------------
    r_grid_mono = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]
    print("=== (a) fraction of configs with both own-Hessians NSD ===")
    print("  c \\ r " + "".join(f"{r:>7.2f}" for r in r_grid_mono))
    for c in [0.5, 1.0, 2.0, 4.0, 8.0, 32.0, 100.0, 300.0]:
        game = MultiPointGame(peaks=tuple(PEAKS), width=WIDTH, coupling=c)
        rng = np.random.default_rng(1)
        fr = [hessian_nsd_fraction(game, r, rng) for r in r_grid_mono]
        out["mono"][c] = dict(zip(map(str, r_grid_mono), fr))
        print(f"  c={c:<6}" + "".join(f"{f:>7.2f}" for f in fr))

    # ---- (b) empirical basin ------------------------------------------------
    r_grid_basin = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    print("\n=== (b) fraction of runs converged (tail expl < 0.1) ===")
    print("  c \\ r " + "".join(f"{r:>7.2f}" for r in r_grid_basin))
    for c in [0.5, 1.0, 2.0, 4.0, 8.0]:
        game = MultiPointGame(peaks=tuple(PEAKS), width=WIDTH, coupling=c)
        rng = np.random.default_rng(2)
        fr = [basin_fraction(game, r, rng) for r in r_grid_basin]
        out["basin"][c] = dict(zip(map(str, r_grid_basin), fr))
        print(f"  c={c:<6}" + "".join(f"{f:>7.2f}" for f in fr))

    with open(RESULTS / "exp3.json", "w") as f:
        json.dump(out, f)
    print(f"\nsaved -> {RESULTS / 'exp3.json'}")


if __name__ == "__main__":
    main()
