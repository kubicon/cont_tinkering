"""Exp 8c — pinning the two exp8/8b edges.

  (E) Upper std edge at h_d=0.7: s0 in {0.30, 0.35, 0.40, 0.60, 1.0} at 40k
      steps — is the trap onset real and near the oracle flip s* = 0.226?
  (F) Does the mass-ratio trap threshold grow with the coupling c? Sweep
      c in {0.5, 1, 2} x h_d in {0.16, 0.20, 0.24, 0.30} at 40k, broad init.
      Coupling-assist prediction: larger c -> larger tolerable decoy mass.
  (G) Long-run classification of the fuzzy h_d=0.22 (c=1) endpoint: 150k
      steps — late convergence or a genuine stall?

Run:  .venv/bin/python theory/exp8c_coupling_threshold.py [efg]
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


def part_e(out):
    print("=== (E) upper std edge, h_d=0.7 (mass x6.3), 40k steps ===")
    game = DecoyWellGame()
    for s0 in [0.30, 0.35, 0.40, 0.60, 1.0]:
        tail, last, dt = run_mmd(game, [-1.5, 1.5], s0)
        out[f"E_s0_{s0}"] = dict(tail=tail, means=last["means0"], std=last["std0"])
        show(f"s0={s0:.2f}", tail, last, dt)


def part_f(out):
    print("\n=== (F) mass threshold vs coupling c, broad init (std 1.5), 40k ===")
    for c in [0.5, 1.0, 2.0]:
        for hd in [0.16, 0.20, 0.24, 0.30]:
            game = DecoyWellGame(decoys=((0.0, hd, 0.45),), coupling=c)
            tail, last, dt = run_mmd(game, [-1.5, 1.5], 1.5)
            out[f"F_c{c}_hd{hd}"] = dict(tail=tail, means=last["means0"])
            show(f"c={c:3.1f} h_d={hd:.2f} (mass x{9*hd:4.2f})", tail, last, dt)


def part_g(out):
    print("\n=== (G) h_d=0.22 (c=1) at 150k steps ===")
    game = DecoyWellGame(decoys=((0.0, 0.22, 0.45),))
    tail, last, dt = run_mmd(game, [-1.5, 1.5], 1.5, steps=150000)
    out["G_hd022_150k"] = dict(tail=tail, means=last["means0"], std=last["std0"])
    show("h_d=0.22, 150k", tail, last, dt)


def main(parts: str = "efg"):
    out = {}
    if "e" in parts:
        part_e(out)
    if "f" in parts:
        part_f(out)
    if "g" in parts:
        part_g(out)
    with open(RESULTS / f"exp8c_{parts}.json", "w") as f:
        json.dump(out, f, default=float)
    print(f"\nsaved -> {RESULTS / f'exp8c_{parts}.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "efg")
