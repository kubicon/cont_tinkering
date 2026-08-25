"""Exp 5 -- does ANY (width, coupling) open a Gaussian-magnet window on the
two-atom game? Searches the curvature-vs-coupling plane for a cell where the
GAUSSIAN head is the bottleneck (means fail to place) and the magnet fixes it.

For each cell we run gauss_tau=0 vs 0.2 (categorical magnet always ON). We flag a
genuine Gaussian-magnet win only if:
  * ON reaches the Nash (tail < 0.1) AND OFF does not (tail > 0.2), AND
  * the improvement is in the MEANS: OFF's means are NOT both near +-1 (i.e. the
    Gaussian head genuinely failed), while ON's are. If OFF already places the
    means at +-1 and only the weights are wrong, that's a categorical/step-size
    failure -- not what we're after -- and is flagged 'cat'.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "theory"))
from idealized_mmd import MultiPointGame, exploitability, make_init, mixture_stats  # noqa: E402
from exp6_gaussian_head_magnet import Cfg, _step  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
WIDTHS = [0.5, 0.6, 0.7]
COUPLINGS = [2.0, 3.0, 4.0, 6.0]
INIT0 = make_init([-0.3, 0.3], log_std=float(np.log(0.3)))
INIT1 = make_init([-0.2, 0.4], log_std=float(np.log(0.3)))


def run(game, cfg, i0, i1):
    lo = float(game.action_space(0).low[0]); hi = float(game.action_space(0).high[0])
    jstep = jax.jit(lambda a, b, c, d: _step(a, b, c, d, game, cfg, lo, hi))
    p0, p1 = i0, i1; m0, m1 = i0, i1
    expls = []
    for t in range(cfg.steps):
        p0, p1 = jstep(p0, p1, m0, m1)
        if (t + 1) % cfg.magnet_interval == 0:
            m0, m1 = p0, p1
        if t % 100 == 0 or t == cfg.steps - 1:
            expls.append(float(exploitability(p0, p1, game)))
    e = np.array(expls)
    means = np.sort(np.array(p0.means))
    means_ok = abs(means[0] + 1) < 0.15 and abs(means[1] - 1) < 0.15
    return {"tail": float(e[int(len(e) * 0.7):].mean()), "means": [float(x) for x in means],
            "means_ok": bool(means_ok)}


def main():
    out = {}
    print("cells: '.'=both converge  'G'=GAUSSIAN-magnet win (means)  'cat'=categorical/stepsize fail  'x'=both fail")
    print(f"{'':6}" + "".join(f"C={c:<6}" for c in COUPLINGS))
    for w in WIDTHS:
        row = []
        for C in COUPLINGS:
            game = MultiPointGame(peaks=(-1.0, 1.0), width=w, coupling=C)
            base = dict(lr=0.05, cat_tau=0.2, steps=15000)
            off = run(game, Cfg(gauss_tau=0.0, **base), INIT0, INIT1)
            on = run(game, Cfg(gauss_tau=0.2, **base), INIT0, INIT1)
            if on["tail"] < 0.1 and off["tail"] < 0.1:
                tag = "."
            elif on["tail"] < 0.1 and off["tail"] > 0.2 and not off["means_ok"] and on["means_ok"]:
                tag = "G"   # the target: Gaussian head was the bottleneck, magnet fixed it
            elif off["tail"] > 0.2 and off["means_ok"]:
                tag = "cat"  # means fine, failure is categorical/stepsize
            else:
                tag = "x"
            row.append(tag)
            out[f"w{w}_C{C}"] = {"off": off, "on": on, "tag": tag}
        print(f"w={w:<4} " + "".join(f"  {t:<6}" for t in row))
    (RESULTS / "exp5.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'exp5.json'}")


if __name__ == "__main__":
    main()
