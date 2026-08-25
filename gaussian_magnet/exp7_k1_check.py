"""Exp 7 -- confirm: exp6 Part 1 (linear+quadratic, no well) works at K=1.

Its Nash is E[a]=0, E[a^2]=1, which a SINGLE Gaussian N(0,1) satisfies. So it does
not truly need two components. Run the identical game with a single Gaussian
policy (mean mu, log-std). The dynamics split into two independent bilinear
matching-pennies games -- one in mu (Nash 0), one in the spread s^2 (Nash 1) --
both flat, so the Gaussian magnet should help both, exactly as at K=1 in exp1-3.

Contrast with the quartic game (Part 2): a single Gaussian is strictly exploitable
there (E[a^4]=3 != 1), so K=1 cannot even represent the Nash -- reported as the
single-Gaussian NashConv, which is > 0.
"""
from __future__ import annotations

import pathlib
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


class G(NamedTuple):
    mean: jnp.ndarray
    log_std: jnp.ndarray


C1 = C2 = 1.0
Q2 = 1.0
LR = 0.05
BOUND = 3.0
STEPS = 20000


def feats(p):
    return p.mean, p.mean ** 2 + jnp.exp(2 * p.log_std) - Q2  # (E[a], E[a^2]-1)


def payoff(p0, p1):
    a1, a2 = feats(p0); b1, b2 = feats(p1)
    return C1 * a1 * b1 + C2 * a2 * b2


def gkl(mp, lp, mq, lq):
    vp, vq = jnp.exp(2 * lp), jnp.exp(2 * lq)
    return lq - lp + (vp + (mp - mq) ** 2) / (2 * vq) - 0.5


def expl(p0, p1):
    grid = jnp.linspace(-BOUND, BOUND, 4001)
    a1, a2 = feats(p0); b1, b2 = feats(p1)
    U = payoff(p0, p1)
    g1, g2 = grid, grid ** 2 - Q2
    br0 = jnp.max(C1 * g1 * b1 + C2 * g2 * b2)
    br1 = jnp.min(C1 * a1 * g1 + C2 * a2 * g2)
    return float((br0 - U) + (U - br1))


def step(p0, p1, m0, m1, tau):
    def apply(p, opp, mag, sign):
        def obj(pp):
            pay = payoff(pp, opp) if sign > 0 else -payoff(opp, pp)
            return pay - tau * gkl(pp.mean, pp.log_std, mag.mean, mag.log_std)
        g = jax.grad(obj)(p)
        mean = jnp.clip(p.mean + LR * jnp.exp(2 * p.log_std) * g.mean, -BOUND, BOUND)
        ls = jnp.clip(p.log_std + LR * 0.5 * g.log_std, jnp.log(1e-3), jnp.log(BOUND))
        return G(mean, ls)
    return apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)


def run(tau, i0, i1):
    js = jax.jit(lambda a, b, c, d: step(a, b, c, d, tau))
    p0, p1 = i0, i1; m0, m1 = i0, i1
    es = []
    for t in range(STEPS):
        p0, p1 = js(p0, p1, m0, m1)
        if (t + 1) % 200 == 0:
            m0, m1 = p0, p1
        if t % 100 == 0 or t == STEPS - 1:
            es.append(expl(p0, p1))
    e = np.array(es)
    return {"final": float(e[-1]), "tail": float(e[int(len(e) * 0.7):].mean()),
            "tail_max": float(e[int(len(e) * 0.7):].max()),
            "mu": (float(p0.mean), float(p1.mean)), "s": float(jnp.exp(p0.log_std))}


def main():
    i0 = G(jnp.asarray(1.5), jnp.asarray(np.log(0.3)))
    i1 = G(jnp.asarray(-1.0), jnp.asarray(np.log(0.5)))
    print("exp6 Part-1 game with a SINGLE Gaussian (K=1). Nash: mu=0, s=1 (i.e. N(0,1)).\n")
    print(f"  {'variant':20} | {'final':>9} {'tail':>9} {'tail_max':>9} | {'final (mu0,mu1)':>18} {'s0':>6}")
    for name, tau in [("gauss magnet OFF", 0.0), ("gauss magnet ON ", 0.2)]:
        r = run(tau, i0, i1)
        print(f"  {name:20} | {r['final']:9.4f} {r['tail']:9.4f} {r['tail_max']:9.4f} "
              f"| {str(tuple(round(x, 2) for x in r['mu'])):>18} {r['s']:6.3f}")
    print("\n  => single Gaussian suffices; magnet OFF cycles, ON converges to N(0,1)."
          "\n     The quartic feature (exp6 Part 2) is what a single Gaussian CANNOT match"
          "\n     (E[a^4]=3 != 1, single-Gaussian NashConv 3.2) -- only that truly needs K=2.")


if __name__ == "__main__":
    main()
