"""Exp 4 — P3: the decoy trap is a *second monotone island*, not a mystery.

DecoyWellGame (peaks +-1 h=1 w=0.05, decoy at 0 h=0.7 w=0.45, symmetric
weights). The theory says the all-mass-on-the-decoy configuration is:

  (i)  feature-matched: u(0) = target first moment = 0, so F(nu) = 0 and the
       coupling term vanishes identically — each player faces the well D alone;
  (ii) locally concave: the decoy is a strict local max of D_s, so the
       own-mean Hessians are negative definite;
  (iii) hence (block-skew lemma) a locally *monotone*, locally *stable* fixed
       point of the same dynamics that converge to Nash from the Nash island —
       while being 0.72-exploitable.

Checks: (A) reproduce the trap from the natural spread init; (B) one-step
residual + Jacobian spectral radius at the trapped endpoint; (C) F(nu) = 0 and
own-Hessian eigenvalues there; (D) same measurements at the *Nash* island of
the same game (components walked onto the peaks) — two coexisting stable
islands of one vector field.

Run:  .venv/bin/python theory/exp4_decoy_island.py
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

from games.examples import DecoyWellGame
from idealized_mmd import (MMDConfig, Params, exploitability, expected_payoff,
                           make_init, mixture_stats, run)
from exp2_local_monotonicity import mmd_step, flatten

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

GAME = DecoyWellGame()
CFG = MMDConfig(lr=0.05, magnet_coef=0.2, magnet_interval=200, steps=20000)
BOX = 3.0  # DecoyWellGame default box [-3, 3]


def analyze(label, p0, p1, out):
    z = flatten(p0, p1)
    step_kw = dict(cfg=CFG, game=GAME)
    resid = float(jnp.linalg.norm(
        mmd_step_box(z, z) - z))
    J = np.asarray(jax.jacobian(lambda zz: mmd_step_box(zz, z))(z))
    rho = float(np.max(np.abs(np.linalg.eigvals(J))))
    _, _, feat0, _ = mixture_stats(p0, GAME)
    _, _, feat1, _ = mixture_stats(p1, GAME)
    h0 = np.asarray(jax.hessian(
        lambda m: expected_payoff(p0._replace(means=m), p1, GAME))(p0.means))
    h1 = np.asarray(jax.hessian(
        lambda m: -expected_payoff(p0, p1._replace(means=m), GAME))(p1.means))
    e0, e1 = np.linalg.eigvalsh(h0), np.linalg.eigvalsh(h1)
    expl = float(exploitability(p0, p1, GAME))
    print(f"\n--- {label} ---")
    print(f"  means0={np.array(p0.means).round(4)}  w0={np.array(jax.nn.softmax(p0.logits)).round(3)}"
          f"  std0={np.exp(np.array(p0.log_std)).round(4)}")
    print(f"  exploitability = {expl:.4f}")
    print(f"  one-step residual = {resid:.3e}   Jacobian spectral radius = {rho:.6f}"
          f"  ({'STABLE' if rho < 1 else 'UNSTABLE'})")
    print(f"  F(nu0) = {np.array(feat0).round(6)}   F(nu1) = {np.array(feat1).round(6)}")
    print(f"  eig(H_mu0) = {e0.round(3)}   eig(H_mu1) = {e1.round(3)}"
          f"  ({'both NSD -> monotone island' if e0.max() < 1e-9 and e1.max() < 1e-9 else 'NOT NSD'})")
    out[label] = {"expl": expl, "residual": resid, "rho": rho,
                  "feat0": np.array(feat0).tolist(),
                  "eig_h0": e0.tolist(), "eig_h1": e1.tolist(),
                  "means0": np.array(p0.means).tolist()}


def mmd_step_box(z, zmag):
    """mmd_step with the DecoyWellGame box [-3,3] instead of TWO_POINT's."""
    import exp2_local_monotonicity as e2
    old = e2.LOW, e2.HIGH, e2.CFG, e2.GAME
    e2.LOW, e2.HIGH, e2.CFG, e2.GAME = -BOX, BOX, CFG, GAME
    try:
        return mmd_step(z, zmag, cfg=CFG, game=GAME)
    finally:
        e2.LOW, e2.HIGH, e2.CFG, e2.GAME = old


def main():
    out = {}

    # (A) the trainer's spread init: means +-1.5, std 1.5 (broad). The broad std
    # makes the component feel the SMOOTHED landscape, where the decoy's 6.3x
    # mass dominates -> walks into the trap. (With a narrow init std of 0.1 the
    # same means converge to the peaks: the initial smoothing scale selects the
    # island. Both runs below.)
    init_narrow = make_init([-1.5, 1.5])  # std 0.1
    pn0, pn1, hist_n = run(GAME, CFG, init_narrow, init_narrow)
    print("=== (A) spread means +-1.5, init std 0.1 (narrow) ===")
    print(f"  final expl = {hist_n[-1]['expl']:.4f}  means0 = {np.array(pn0.means).round(4)}")
    init = make_init([-1.5, 1.5], log_std=float(np.log(1.5)))
    p0, p1, hist = run(GAME, CFG, init, init)
    print("=== (A') spread means +-1.5, init std 1.5 (the trainer's init) ===")
    print(f"  final expl = {hist[-1]['expl']:.4f}  means0 = {np.array(p0.means).round(4)}")

    # (B)+(C) the trapped endpoint
    analyze("trap endpoint (from spread init)", p0, p1, out)

    # (D) the Nash island of the same game: init components ON the peaks
    initN = make_init([-1.0, 1.0])
    q0, q1, histN = run(GAME, CFG, initN, initN)
    analyze("Nash island (init on peaks)", q0, q1, out)

    with open(RESULTS / "exp4.json", "w") as f:
        json.dump(out, f)
    print(f"\nsaved -> {RESULTS / 'exp4.json'}")


if __name__ == "__main__":
    main()
