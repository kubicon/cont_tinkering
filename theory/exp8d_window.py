"""Exp 8d — the init-std WINDOW is the whole story; emptiness is a game property.

Claim assembled from exp8/8b/8c: annealed mixture-MMD from means at distance d
from their atoms converges iff the init std s0 lies in a window

    s_reach(d)  <~  s0  <~  min( s_dom(game), s_ej(game) )

  s_reach ~ d/kappa      : below it the means freeze in the dead zone;
  s_dom               : the scale at which spurious (decoy) mass overtakes the
                        Nash atoms in the smoothed well (continuation oracle);
  s_ej                : the scale at which the coupling's moment-compensation
                        ejects a component to the box edge (weight asymmetry).

The window's non-emptiness is a property of (game, init distance) — a GAME
CLASS assumption, not a tuning question. Test:

  (H1) Original decoy game (mass x6.3, s_dom ~ 0.28), init means +-2.5
       (d = 1.5 -> s_reach ~ 0.3 > s_dom): predicted EMPTY window — no s0
       converges.
  (H2) Same init, light decoy h_d=0.1 (mass x0.9 < 1: never overtakes,
       s_dom = inf): predicted nonempty window — moderate s0 converges.
  (H3) Symmetric control of the ejection edge: two_point-style symmetric
       weights never eject (all of exp8-A converged at s0=1.5), while
       asymmetric weights (0.65, 0.35), NO decoy, broad s0=1.5 eject
       (exp8b-C'); check s0 dependence of the ejection for that game.

Run:  .venv/bin/python theory/exp8d_window.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from games.examples import DecoyWellGame
from exp8b_boundaries import RESULTS, run_mmd, show

S0S = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]


def main():
    out = {}

    print("=== (H1) heavy decoy (mass x6.3), init means +-2.5: predict NO s0 works ===")
    game = DecoyWellGame()
    for s0 in S0S:
        tail, last, dt = run_mmd(game, [-2.5, 2.5], s0)
        out[f"H1_{s0}"] = dict(tail=tail, means=last["means0"])
        show(f"s0={s0:4.2f}", tail, last, dt)

    print("\n=== (H2) light decoy (mass x0.9, never overtakes), init +-2.5: window exists ===")
    game = DecoyWellGame(decoys=((0.0, 0.1, 0.45),))
    for s0 in S0S:
        tail, last, dt = run_mmd(game, [-2.5, 2.5], s0)
        out[f"H2_{s0}"] = dict(tail=tail, means=last["means0"])
        show(f"s0={s0:4.2f}", tail, last, dt)

    print("\n=== (H3) ejection edge: weights (0.65,0.35), no decoy, init +-1.5 ===")
    game = DecoyWellGame(weights=(0.65, 0.35), decoys=())
    for s0 in S0S:
        tail, last, dt = run_mmd(game, [-1.5, 1.5], s0, steps=20000)
        out[f"H3_{s0}"] = dict(tail=tail, means=last["means0"], w=last["w0"])
        show(f"s0={s0:4.2f}", tail, last, dt)

    with open(RESULTS / "exp8d.json", "w") as f:
        json.dump(out, f, default=float)
    print(f"\nsaved -> {RESULTS / 'exp8d.json'}")


if __name__ == "__main__":
    main()
