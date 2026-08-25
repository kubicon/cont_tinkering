"""Exp 5 — P5: the residual exploitability of the mixture Nash is the O(s^2)
smoothing bias of representing atoms by width-s Gaussians.

At the exact configuration (means on the peaks, Nash weights, common std s) the
features stay matched (for TWO_POINT the single feature is linear, so symmetric
noise leaves E[f] = 0 exactly), hence the coupling term vanishes and

    expl(s) = 2 * (max_a D(a) - E_{N(p, s^2)} D)  ~=  2h * (1 - w/sqrt(w^2+s^2))
            =  h s^2 / w^2 + O(s^4).

We compute expl(s) with the repo's exact exploitability and compare with the
closed form + the s^2 slope on a log-log plot.

Run:  .venv/bin/python theory/exp5_std_floor_bias.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from idealized_mmd import TWO_POINT, exploitability, make_init

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    w = float(TWO_POINT.width)
    svals = np.geomspace(1e-3, 0.3, 13)
    rows = []
    for s in svals:
        p = make_init([0.0, 1.0], log_std=float(np.log(s)))
        e = float(exploitability(p, p, TWO_POINT))
        # closed form: both bumps contribute to D at each peak; dominant term below
        pred = 2 * (1 - w / np.sqrt(w**2 + s**2))
        rows.append({"s": float(s), "expl": e, "pred": pred})
        print(f"  s={s:8.4f}  expl={e:10.6f}  closed-form 2h(1-w/sqrt(w^2+s^2))={pred:10.6f}")
    small = [r for r in rows if r["s"] <= 0.03]
    slope = np.polyfit(np.log([r["s"] for r in small]),
                       np.log([r["expl"] for r in small]), 1)[0]
    print(f"\n  log-log slope over s<=0.03: {slope:.3f}  (theory: 2)")
    with open(RESULTS / "exp5.json", "w") as f:
        json.dump({"rows": rows, "slope": slope}, f)
    print(f"saved -> {RESULTS / 'exp5.json'}")


if __name__ == "__main__":
    main()
