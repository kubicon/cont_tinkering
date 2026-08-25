"""Exp 8b — boundary cases and controls for the scale-dominance study (exp8).

Follow-ups:
  (A') Longer runs (40k steps) at exp8's boundary cases: h_d = 0.18 (means
       reached the peaks but expl stuck at 0.30 -- slow weight fixing, or a
       genuine partial trap?), and a bisection of the mass threshold h_d*
       between 0.14 (converged) and 0.25 (trapped).
  (B') Longer runs at the small-std edge: s0 in {0.05, 0.1} trapped at 12k
       steps but exp4 saw s0=0.1 converge at 20k -- slow transport vs frozen?
       Also refine the upper edge between 0.2 and 0.3 (oracle says 0.226).
  (C') Controls for the random-game failures: for each exp8-C "satisfy" game
       that trapped, run (i) the SAME game with the decoy deleted (is the
       failure the decoy's fault at all?), and (ii) the same game with a
       narrower init std 0.3. Print final means/weights to diagnose the
       failure mode (same-peak miscoordination vs decoy capture).

Run:  .venv/bin/python theory/exp8b_boundaries.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from games.examples import DecoyWellGame
from idealized_mmd import MMDConfig, make_init, run

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

PASS = 0.1
PEAK_MASS = 0.05


def run_mmd(game, init_means, init_std, steps=40000):
    cfg = MMDConfig(lr=0.05, magnet_coef=0.2, magnet_interval=200, steps=steps)
    init = make_init(list(init_means), log_std=float(np.log(init_std)))
    t0 = time.time()
    _, _, hist = run(game, cfg, init, init, log_every=1000)
    e = np.array([h["expl"] for h in hist])
    tail = float(e[int(len(e) * 0.85):].mean())
    return tail, hist[-1], time.time() - t0


def show(label, tail, last, dt):
    v = "NASH" if tail < PASS else "trap"
    print(f"  {label}  tail={tail:7.4f} -> {v}  means0={np.round(last['means0'],3)}"
          f"  w0={np.round(last['w0'],2)}  std0={np.round(last['std0'],3)} [{dt:.0f}s]")
    return v


def part_a(out):
    print("=== (A') mass-threshold boundary, 40k steps, broad init (std 1.5) ===")
    for hd in [0.16, 0.18, 0.20, 0.22]:
        game = DecoyWellGame(decoys=((0.0, hd, 0.45),))
        tail, last, dt = run_mmd(game, [-1.5, 1.5], 1.5)
        out[f"A_hd_{hd}"] = dict(tail=tail, means=last["means0"], w=last["w0"])
        show(f"h_d={hd:.2f} (mass x{9*hd:.2f})", tail, last, dt)


def part_b(out):
    print("\n=== (B') std-window edges, original decoy (mass x6.3), 40k steps ===")
    game = DecoyWellGame()
    for s0 in [0.05, 0.10, 0.22, 0.25, 0.28]:
        tail, last, dt = run_mmd(game, [-1.5, 1.5], s0)
        out[f"B_s0_{s0}"] = dict(tail=tail, means=last["means0"], w=last["w0"])
        show(f"s0={s0:.2f}", tail, last, dt)


def part_c(out):
    print("\n=== (C') controls for the random-game failures (20k steps) ===")
    rng = np.random.default_rng(0)
    games = []
    for i in range(8):
        q = float(rng.uniform(0.3, 0.7))
        wd = float(rng.uniform(0.2, 0.6))
        hd = float(rng.uniform(0.2, 0.8)) * PEAK_MASS / wd
        cd = float(rng.uniform(-0.6, 0.6))
        games.append((q, cd, hd, wd))
    rows = []
    for i, (q, cd, hd, wd) in enumerate(games):
        print(f"  --- game {i}: w=({q:.2f},{1-q:.2f}) decoy=({cd:+.2f},h{hd:.2f},w{wd:.2f})"
              f" mass x{hd*wd/PEAK_MASS:.2f} ---")
        g_no = DecoyWellGame(weights=(q, 1 - q), decoys=())
        tail, last, dt = run_mmd(g_no, [-1.5, 1.5], 1.5, steps=20000)
        show("    no-decoy, std 1.5 ", tail, last, dt)
        g = DecoyWellGame(weights=(q, 1 - q), decoys=((cd, hd, wd),))
        tail2, last2, dt2 = run_mmd(g, [-1.5, 1.5], 0.3, steps=20000)
        show("    decoy,    std 0.3 ", tail2, last2, dt2)
        tail3, last3, dt3 = run_mmd(g, [-1.5, 1.5], 1.5, steps=40000)
        show("    decoy,    std 1.5, 40k", tail3, last3, dt3)
        rows.append(dict(game=(q, cd, hd, wd), no_decoy=tail, narrow=tail2,
                         long=tail3))
    out["C_controls"] = rows


def main(parts: str = "abc"):
    out = {}
    if "a" in parts:
        part_a(out)
    if "b" in parts:
        part_b(out)
    if "c" in parts:
        part_c(out)
    with open(RESULTS / f"exp8b_{parts}.json", "w") as f:
        json.dump(out, f, default=float)
    print(f"\nsaved -> {RESULTS / f'exp8b_{parts}.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "abc")
