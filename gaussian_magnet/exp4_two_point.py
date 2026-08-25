"""Exp 4 -- the minimal 2-Gaussian counterexample: mixed Nash over TWO points.

Game: MultiPointGame(peaks=(-1, +1), width=w, coupling=C), K=2 components. Its
unique Nash is the 50/50 mixture over the two peaks {-1, +1} -- a genuine mixed
strategy that a single Gaussian cannot represent, so two components are required.

Two heads move (unlike the K=1 exp1-3, where only the Gaussian head was live):
  * categorical head  -> must reach weights (0.5, 0.5); its magnet is ESSENTIAL
    and is kept ON in every run here (cat_tau=0.2). Not the variable under test.
  * Gaussian head     -> must place the two means at -1 and +1. THIS is where we
    test the magnet (gauss_tau = 0 vs 0.2).

The two peaks pin the atom LOCATIONS with curvature ~ h/w^2, so with a sharp well
(small w) the Gaussian head is curvature-carried and the magnet is redundant
(exp6). We WEAKEN that curvature (wider w) and STRENGTHEN the coupling C. The
coupling's force on a mean is C*f'(mu)*E[f_opp], a rotation between the players'
mean actions; once it dominates the well curvature the two means rotate on the
way in and plain natural-gradient descent-ascent stops converging in last iterate
-- exactly exp3's curvature-vs-coupling crossover, now on a real two-atom game.

Reuses theory/exp6's split-tau step (cat_tau, gauss_tau) verbatim.
"""
from __future__ import annotations

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "theory"))

from idealized_mmd import MultiPointGame, exploitability, make_init, mixture_stats  # noqa: E402
from exp6_gaussian_head_magnet import Cfg, _step  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

WIDTH = 0.4          # weak curvature (sharp default is 0.1); peaks still strict maxima
COUPLINGS = [1.0, 2.0, 4.0, 8.0, 16.0]
# off-Nash init: both means on the same side, weights not yet split -> the head
# must both separate the means AND transport one across, feeling the coupling.
INIT0 = make_init([-0.3, 0.3], log_std=float(np.log(0.3)))
INIT1 = make_init([-0.2, 0.4], log_std=float(np.log(0.3)))


def run(game, cfg: Cfg, init0, init1):
    lo = float(game.action_space(0).low[0]); hi = float(game.action_space(0).high[0])
    jstep = jax.jit(lambda a, b, c, d: _step(a, b, c, d, game, cfg, lo, hi))
    p0, p1 = init0, init1
    m0, m1 = init0, init1
    expls = []
    for t in range(cfg.steps):
        p0, p1 = jstep(p0, p1, m0, m1)
        if (t + 1) % cfg.magnet_interval == 0:
            m0, m1 = p0, p1
        if t % 100 == 0 or t == cfg.steps - 1:
            expls.append(float(exploitability(p0, p1, game)))
    e = np.array(expls)
    w0, _, _, _ = mixture_stats(p0, game)
    return {
        "final": float(e[-1]),
        "tail": float(e[int(len(e) * 0.7):].mean()),
        "tail_max": float(e[int(len(e) * 0.7):].max()),   # cycling shows here
        "best": float(e.min()),
        "final_means0": [float(x) for x in np.sort(np.array(p0.means))],
        "final_w0": sorted(float(x) for x in np.array(w0)),
    }, (p0, p1)


def main():
    out = {}
    print(f"MultiPointGame(peaks=(-1,1), width={WIDTH}); Nash = 50/50 over {{-1,+1}} (needs K=2)")
    print(f"categorical magnet ON (cat_tau=0.2) in all runs; testing the GAUSSIAN magnet\n")
    hdr = (f"{'C':>5} | {'OFF tail':>9} {'OFF tailmax':>11} {'OFF means':>16} {'OFF w':>12} "
           f"| {'ON tail':>9} {'ON means':>16} {'ON w':>12} | verdict")
    print(hdr); print("-" * len(hdr))
    for C in COUPLINGS:
        game = MultiPointGame(peaks=(-1.0, 1.0), width=WIDTH, coupling=C)
        base = dict(lr=0.05, cat_tau=0.2, steps=30000)
        off, _ = run(game, Cfg(gauss_tau=0.0, **base), INIT0, INIT1)
        on, _ = run(game, Cfg(gauss_tau=0.2, **base), INIT0, INIT1)
        helps = on["tail"] < 0.1 and off["tail"] > 0.2
        verdict = "MAGNET HELPS" if helps else ("both ok" if on["tail"] < 0.1 and off["tail"] < 0.1 else "-")
        out[C] = {"off": off, "on": on, "helps": helps}
        m_off = "[" + ",".join(f"{x:+.2f}" for x in off["final_means0"]) + "]"
        m_on = "[" + ",".join(f"{x:+.2f}" for x in on["final_means0"]) + "]"
        w_off = "/".join(f"{x:.2f}" for x in off["final_w0"])
        w_on = "/".join(f"{x:.2f}" for x in on["final_w0"])
        print(f"{C:5.1f} | {off['tail']:9.4f} {off['tail_max']:11.4f} {m_off:>16} {w_off:>12} "
              f"| {on['tail']:9.4f} {m_on:>16} {w_on:>12} | {verdict}")
    (RESULTS / "exp4.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'exp4.json'}")


if __name__ == "__main__":
    main()
