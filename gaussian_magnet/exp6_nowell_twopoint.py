"""Exp 6 -- the minimal 2-Gaussian counterexample that actually works: a mixed
Nash over two points with NO well, so the Gaussian head has no own curvature.

exp4/exp5 showed that a two-ISOLATED-ATOM Nash built from a well (MultiPointGame)
never benefits from the Gaussian magnet: resolving two peaks forces a strong
mean-curvature (~h/w^2) that pins the means by itself. To get the K=1 bilinear
situation (flat mean, magnet-supplied pinning) WITH a two-point mixed Nash, we
remove the well entirely and hold the two atoms with moment-matching coupling
alone -- a linear feature f1(a)=a and a quadratic feature f2(a)=a^2 - q:

    U(a, b) = c1 * f1(a) f1(b) + c2 * f2(a) f2(b)          (no well term)

Zero-sum, player 0 max / player 1 min. A player is unexploitable iff
E[f1] = E[a] = 0 AND E[f2] = E[a^2] = q, i.e. mean 0 and second moment q. The
minimal-support distribution achieving it is the 50/50 mixture over {-sqrt(q),
+sqrt(q)} -- a genuine two-point mixed strategy (a single Gaussian with mean 0
and variance q is ALSO unexploitable in value, but the two-point mix is the
identified support and what K=2 converges toward; either way the update below
has no curvature to lean on).

Why the magnet matters here. Player 0's best-response landscape in x = a is
c1*E[f1_opp]*x + c2*E[f2_opp]*x^2: its own curvature is 2*c2*E[f2_opp], which
FLIPS SIGN with the opponent's spread error and is exactly 0 at the Nash. So the
mean has no intrinsic restoring curvature (unlike the well case) -- natural-
gradient descent-ascent rotates/drifts, and only the magnet's proximal term
supplies last-iterate convergence, exactly as in the K=1 bilinear game (exp1-3).

Single self-contained closed-form engine (K=2, no well); reuses the repo's
categorical mirror step.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from idealized_mmd import categorical_mirror_update  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


class P(NamedTuple):
    logits: jnp.ndarray   # (2,)
    means: jnp.ndarray    # (2,)
    log_std: jnp.ndarray  # (2,)


@dataclasses.dataclass(frozen=True)
class Cfg:
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 1.0           # weight on the quartic feature (forces K>=2; see below)
    q2: float = 1.0           # target E[a^2]  -> atoms at +-1
    q4: float = 1.0           # target E[a^4]. A single Gaussian with E[a]=0,E[a^2]=1
                              # has E[a^4]=3, so matching q4=1 rules out ANY single
                              # Gaussian -> the two-point mix is genuinely required.
    lr: float = 0.05
    cat_tau: float = 0.2      # categorical magnet (kept ON)
    gauss_tau: float = 0.0    # Gaussian magnet -- the knob
    entropy_coef: float = 0.0
    magnet_interval: int = 200
    steps: int = 20000
    bound: float = 3.0
    std_floor: float = 1e-3
    std_ceil: float = 3.0


def _comp_moments(means, var):
    """Per-component (E[a], E[a^2], E[a^4]) for N(mean, var). Gaussian moments."""
    m2 = means ** 2 + var
    m4 = means ** 4 + 6 * means ** 2 * var + 3 * var ** 2
    return means, m2, m4


def _feats(p: P, cfg: Cfg):
    """(E[f1], E[f2], E[f3]) = (E[a], E[a^2]-q2, E[a^4]-q4) under the K=2 mixture."""
    w = jax.nn.softmax(p.logits)
    c1, c2, c4 = _comp_moments(p.means, jnp.exp(2 * p.log_std))
    return (jnp.sum(w * c1), jnp.sum(w * c2) - cfg.q2, jnp.sum(w * c4) - cfg.q4)


def expected_payoff(p0: P, p1: P, cfg: Cfg):
    a1, a2, a3 = _feats(p0, cfg)
    b1, b2, b3 = _feats(p1, cfg)
    return cfg.c1 * a1 * b1 + cfg.c2 * a2 * b2 + cfg.c3 * a3 * b3


def component_q(p: P, opp: P, cfg: Cfg, sign: float):
    """Per-component utility for the categorical mirror step."""
    ob1, ob2, ob3 = _feats(opp, cfg)
    c1, c2, c4 = _comp_moments(p.means, jnp.exp(2 * p.log_std))
    return sign * (cfg.c1 * c1 * ob1 + cfg.c2 * c2 * ob2 + cfg.c3 * c4 * ob3)


def gaussian_kl(m_p, ls_p, m_q, ls_q):
    vp, vq = jnp.exp(2 * ls_p), jnp.exp(2 * ls_q)
    return jnp.sum(ls_q - ls_p + (vp + (m_p - m_q) ** 2) / (2 * vq) - 0.5)


def exploitability(p0: P, p1: P, cfg: Cfg):
    B = cfg.bound
    grid = jnp.linspace(-B, B, 4001)
    a1, a2, a3 = _feats(p0, cfg); b1, b2, b3 = _feats(p1, cfg)
    U = expected_payoff(p0, p1, cfg)
    g1 = grid; g2 = grid ** 2 - cfg.q2; g3 = grid ** 4 - cfg.q4
    br0 = jnp.max(cfg.c1 * g1 * b1 + cfg.c2 * g2 * b2 + cfg.c3 * g3 * b3)   # player 0 deviates
    br1 = jnp.min(cfg.c1 * a1 * g1 + cfg.c2 * a2 * g2 + cfg.c3 * a3 * g3)   # player 1 deviates
    return float((br0 - U) + (U - br1))


class _CatShim:
    def __init__(self, cfg): self.lr, self.magnet_coef, self.entropy_coef = cfg.lr, cfg.cat_tau, cfg.entropy_coef


def _step(p0, p1, m0, m1, cfg: Cfg):
    shim = _CatShim(cfg)

    def apply(p, opp, magnet, sign):
        logits = categorical_mirror_update(p.logits, component_q(p, opp, cfg, sign), magnet.logits, shim)

        def obj(pp):
            pay = expected_payoff(pp, opp, cfg) if sign > 0 else -expected_payoff(opp, pp, cfg)
            ent = cfg.entropy_coef * jnp.sum(pp.log_std)
            mag = cfg.gauss_tau * gaussian_kl(pp.means, pp.log_std, magnet.means, magnet.log_std)
            return pay + ent - mag

        g = jax.grad(obj)(p)
        means = jnp.clip(p.means + cfg.lr * jnp.exp(2 * p.log_std) * g.means, -cfg.bound, cfg.bound)
        log_std = jnp.clip(p.log_std + cfg.lr * 0.5 * g.log_std, jnp.log(cfg.std_floor), jnp.log(cfg.std_ceil))
        return P(logits, means, log_std)

    return apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)


def run(cfg: Cfg, i0, i1):
    jstep = jax.jit(lambda a, b, c, d: _step(a, b, c, d, cfg))
    p0, p1 = i0, i1; m0, m1 = i0, i1
    expls = []
    for t in range(cfg.steps):
        p0, p1 = jstep(p0, p1, m0, m1)
        if (t + 1) % cfg.magnet_interval == 0:
            m0, m1 = p0, p1
        if t % 100 == 0 or t == cfg.steps - 1:
            expls.append(exploitability(p0, p1, cfg))
    e = np.array(expls)
    w0 = np.array(jax.nn.softmax(p0.logits))
    return {
        "final": float(e[-1]), "tail": float(e[int(len(e) * 0.7):].mean()),
        "tail_max": float(e[int(len(e) * 0.7):].max()), "best": float(e.min()),
        "means0": [float(x) for x in np.sort(np.array(p0.means))],
        "w0": sorted(float(x) for x in w0),
    }, (p0, p1)


def mk(means, logits, std):
    return P(jnp.asarray(logits, float), jnp.asarray(means, float), jnp.full((2,), np.log(std)))


def main():
    # off-Nash, asymmetric init: means same-side-ish, uneven weights, wrong spread.
    i0 = mk([-0.2, 0.6], [0.4, -0.4], 0.3)
    i1 = mk([0.1, 0.5], [-0.3, 0.3], 0.3)
    out = {}

    # ---- Part 1: the clean, trustworthy game -- linear + quadratic features, no well ----
    print("PART 1 -- no-well two-point game, features f=(a, a^2-1)  (c1=c2=1, box +-3)")
    print("  Nash: E[a]=0, E[a^2]=1. categorical magnet ON in both.\n")
    base = dict(c1=1.0, c2=1.0, c3=0.0, q2=1.0, bound=3.0, steps=20000)
    hdr = f"  {'variant':22} | {'final':>9} {'tail':>9} {'tail_max':>9} | {'means':>16} {'weights':>12}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, gt in [("gauss magnet OFF (0.0)", 0.0), ("gauss magnet ON  (0.2)", 0.2)]:
        res, _ = run(Cfg(gauss_tau=gt, **base), i0, i1)
        out[f"part1_{name}"] = res
        m = "[" + ",".join(f"{x:+.2f}" for x in res["means0"]) + "]"
        wv = "/".join(f"{x:.2f}" for x in res["w0"])
        print(f"  {name:22} | {res['final']:9.4f} {res['tail']:9.4f} {res['tail_max']:9.4f} | {m:>16} {wv:>12}")

    # ---- Part 2: forcing strict K=2 with a quartic feature (numerically delicate) ----
    # A single Gaussian with E[a]=0, E[a^2]=1 has E[a^4]=3, so adding a matched
    # quartic feature f3=a^4-1 makes one Gaussian strictly exploitable -> K=2 required.
    print("\nPART 2 -- add quartic f3=a^4-1 (c3>0) to make K=2 STRICTLY required (single Gaussian exploitable).")
    print("  NOTE: a^4 is unbounded and interacts with the box, so magnitudes here are less clean than Part 1.\n")
    print(f"  {'c3':>5} {'single-G expl':>13} | {'variant':6} | {'tail':>9} {'tail_max':>9} | {'means':>16} {'weights':>12}")
    for c3 in [0.1]:
        cfg_s = Cfg(c1=2.0, c2=2.0, c3=c3, bound=2.0)
        se = exploitability(mk([0, 0], [0, 0], 1.0), mk([0, 0], [0, 0], 1.0), cfg_s)
        for name, gt in [("OFF", 0.0), ("ON", 0.2)]:
            res, _ = run(Cfg(c1=2.0, c2=2.0, c3=c3, gauss_tau=gt, bound=2.0, steps=20000), i0, i1)
            out[f"part2_c3{c3}_{name}"] = res
            m = "[" + ",".join(f"{x:+.2f}" for x in res["means0"]) + "]"
            wv = "/".join(f"{x:.2f}" for x in res["w0"])
            ses = f"{se:13.4f}" if name == "OFF" else " " * 13
            print(f"  {c3:5.2f} {ses} | {name:6} | {res['tail']:9.4f} {res['tail_max']:9.4f} | {m:>16} {wv:>12}")

    (RESULTS / "exp6.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'exp6.json'}")


if __name__ == "__main__":
    main()
