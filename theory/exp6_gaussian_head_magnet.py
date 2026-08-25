"""Exp 6 — Is the magnet (MMD proximal) term on the *Gaussian head* helpful?

MMD's parametric update runs two heads (see idealized_mmd.py / THEORY.md):

  * categorical head — the exact closed-form MMD simplex proximal step. Its
    payoff is *linear* in pi, so it has NO own curvature; the magnet
    tau*KL(pi||magnet) is what injects strong convexity. Core MMD.
  * Gaussian head — a natural-gradient (Fisher-Rao) step. At a Nash-realizing
    config each mean sits at a strict local MAX of the smoothed well, so the
    payoff is already strongly *concave* in the means (Hessian ~ -w*h/w^2 << 0,
    THEORY.md exp2). The magnet -tau*KL(current||magnet) is *added on top* of
    that curvature.

Question: does the Gaussian-head magnet do anything useful, given the head is
already strongly monotone from curvature? Theory prediction: it is redundant for
convergence (curvature already contracts), acts only as a small transport
*brake*, and — since the snapshot converges to the current policy at a fixed
point — does not move the fixed point. So dropping it should not hurt convergent
cases and should not help (may mildly hurt) trapped ones; dropping the
*categorical* magnet, by contrast, should break convergence.

We separate the two coefficients (cat_tau, gauss_tau) and compare, per scenario:
  A) cat=0.2 gauss=0.2  — baseline (current code)
  B) cat=0.2 gauss=0.0  — the question: Gaussian magnet OFF
  C) cat=0.0 gauss=0.2  — contrast: categorical magnet OFF (should break)

Metrics: final expl, tail(30%) expl, steps-to-reach-0.1, and the Jacobian
spectral radius of the step map at the Nash fixed point (does gauss magnet add
local stability?).

Run:  .venv/bin/python theory/exp6_gaussian_head_magnet.py
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from idealized_mmd import (Params, TWO_POINT, THREE_POINT,
                           categorical_mirror_update, component_q,
                           expected_payoff, exploitability, gaussian_kl,
                           make_init)
from games.configs import DecoyWellConfig

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

DECOY = DecoyWellConfig().build()  # peaks +-1 (h1,w0.05) + decoy at 0 (h0.7,w0.45), box[-3,3]


@dataclasses.dataclass(frozen=True)
class Cfg:
    lr: float = 0.05
    cat_tau: float = 0.2        # magnet weight on the categorical head
    gauss_tau: float = 0.2      # magnet weight on the Gaussian head  <-- the knob
    entropy_coef: float = 0.0
    magnet_interval: int = 200
    steps: int = 20000
    std_floor: float = 1e-3
    std_ceil: float = 1.0


class _CatShim:
    """categorical_mirror_update reads .lr/.magnet_coef/.entropy_coef."""
    def __init__(self, cfg: Cfg):
        self.lr, self.magnet_coef, self.entropy_coef = cfg.lr, cfg.cat_tau, cfg.entropy_coef


def _step(p0, p1, m0, m1, game, cfg: Cfg, lo, hi):
    catshim = _CatShim(cfg)

    def apply(p, opp, magnet, sign):
        logits = categorical_mirror_update(
            p.logits, component_q(p, opp, game, sign), magnet.logits, catshim)

        def obj(pp):
            pay = expected_payoff(pp, opp, game) if sign > 0 else -expected_payoff(opp, pp, game)
            ent = cfg.entropy_coef * jnp.sum(pp.log_std)
            mag = cfg.gauss_tau * gaussian_kl(pp.means, pp.log_std, magnet.means, magnet.log_std)
            return pay + ent - mag

        g = jax.grad(obj)(p)
        means = jnp.clip(p.means + cfg.lr * jnp.exp(2 * p.log_std) * g.means, lo, hi)
        log_std = jnp.clip(p.log_std + cfg.lr * 0.5 * g.log_std,
                           jnp.log(cfg.std_floor), jnp.log(cfg.std_ceil))
        return Params(logits, means, log_std)

    return apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)


def run(game, cfg: Cfg, init0, init1):
    lo = float(game.action_space(0).low[0])
    hi = float(game.action_space(0).high[0])
    jstep = jax.jit(lambda a, b, c, d: _step(a, b, c, d, game, cfg, lo, hi))
    p0, p1 = init0, init1
    m0, m1 = init0, init1
    expls = []
    reach = None
    for t in range(cfg.steps):
        p0, p1 = jstep(p0, p1, m0, m1)
        if (t + 1) % cfg.magnet_interval == 0:
            m0, m1 = p0, p1
        if t % 100 == 0 or t == cfg.steps - 1:
            e = float(exploitability(p0, p1, game))
            expls.append(e)
            if reach is None and e < 0.1:
                reach = t
    tail = float(np.mean(expls[int(len(expls) * 0.7):]))
    return {"final": expls[-1], "tail": tail, "reach": reach, "best": float(min(expls))}, (p0, p1)


# ---- Jacobian spectral radius at a fixed point (magnet frozen at z*) ----------
def flatten(p0, p1):
    return jnp.concatenate([p0.logits, p0.means, p0.log_std, p1.logits, p1.means, p1.log_std])


def unflatten(z):
    q = jnp.split(z, 6)
    return Params(q[0], q[1], q[2]), Params(q[3], q[4], q[5])


def spectral_radius(zstar, game, cfg, lo, hi):
    def stepmap(z):
        p0, p1 = unflatten(z)
        m0, m1 = unflatten(zstar)
        q0, q1 = _step(p0, p1, m0, m1, game, cfg, lo, hi)
        return flatten(q0, q1)
    J = np.asarray(jax.jacobian(stepmap)(zstar))
    return float(np.max(np.abs(np.linalg.eigvals(J))))


# ---- scenarios ---------------------------------------------------------------
def peak_init(game, means, log_std):
    return make_init(means, log_std=float(np.log(log_std)))


SCENARIOS = [
    ("two_point / Nash basin", TWO_POINT,
     peak_init(TWO_POINT, [0.05, 0.95], 0.1), peak_init(TWO_POINT, [0.05, 0.95], 0.1)),
    ("two_point / weight-starvation (K=3>2)", TWO_POINT,
     make_init([-0.08, -0.02, 0.1]), make_init([-0.05, 0.0, 0.12])),
    ("three_point / Nash basin", THREE_POINT,
     make_init([-0.95, 0.0, 0.95]), make_init([-0.95, 0.0, 0.95])),
    ("decoy_well / on-peaks init (clean)", DECOY,
     peak_init(DECOY, [-1.0, 1.0], 0.1), peak_init(DECOY, [-1.0, 1.0], 0.1)),
    ("decoy_well / spread init (the TRAP)", DECOY,
     peak_init(DECOY, [-1.5, 1.5], 1.5), peak_init(DECOY, [-1.5, 1.5], 1.5)),
]

VARIANTS = [
    ("A both magnets   (cat=.2 g=.2)", Cfg(cat_tau=0.2, gauss_tau=0.2)),
    ("B gauss magnet OFF(cat=.2 g=0 )", Cfg(cat_tau=0.2, gauss_tau=0.0)),
    ("C cat magnet OFF  (cat=0  g=.2)", Cfg(cat_tau=0.0, gauss_tau=0.2)),
]


def main():
    out = {}
    for sname, game, i0, i1 in SCENARIOS:
        print(f"\n=== {sname} ===")
        print(f"  {'variant':32s} {'final':>8s} {'tail':>8s} {'reach0.1':>9s} {'best':>8s}")
        out[sname] = {}
        for vname, cfg in VARIANTS:
            res, _ = run(game, cfg, i0, i1)
            reach = "never" if res["reach"] is None else str(res["reach"])
            print(f"  {vname:32s} {res['final']:8.4f} {res['tail']:8.4f} {reach:>9s} {res['best']:8.4f}")
            out[sname][vname] = res

    # ---- local stability: does the Gaussian magnet change rho at the Nash? ----
    print("\n=== Jacobian spectral radius at the two_point Nash fixed point ===")
    game = TWO_POINT
    lo = float(game.action_space(0).low[0]); hi = float(game.action_space(0).high[0])
    # settle to the Nash with both magnets, then measure rho under each variant
    _, (pf0, pf1) = run(game, Cfg(cat_tau=0.2, gauss_tau=0.2),
                        peak_init(game, [0.05, 0.95], 0.1), peak_init(game, [0.05, 0.95], 0.1))
    zstar = flatten(pf0, pf1)
    out["rho"] = {}
    for vname, cfg in VARIANTS[:2]:
        rho = spectral_radius(zstar, game, cfg, lo, hi)
        print(f"  {vname:32s} rho = {rho:.6f}  1-rho = {1-rho:.5f}   (eta*tau = {cfg.lr*0.2:.5f})")
        out["rho"][vname] = rho

    (RESULTS / "exp6.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'exp6.json'}")


if __name__ == "__main__":
    main()
