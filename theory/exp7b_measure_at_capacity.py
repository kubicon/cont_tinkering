"""Exp 7b — the clean measure test, pinned at escape-capable capacity.

exp7 found that below an RBF-sharpness threshold NO measure escapes the decoy
trap (floor effect), so its m=80 measure sweep can't discriminate measures. This
re-runs the measure sweep at m=120 / n=601 -- the capacity where uniform is known
to fully escape (final expl ~7e-4 by 40k) -- so any measure effect has room to show.

Question: once capacity suffices, does the regression measure mu still matter?
Prediction from exp7's partial data (uniform 0.0007 vs on-policy 0.0122 at n=601):
measure is second-order -- on-policy escapes too, just a bit slower.

Run:  .venv/bin/python -u theory/exp7b_measure_at_capacity.py
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

from exp7_ebm_regression_measure import (
    ebm_regression_mmd, measure_fn, on_decoy_logits, payoff_matrix, rbf_basis,
    _tail, RESULTS)


def main():
    # defaults are the knife-edge capacity; override via argv: exp7b.py [m] [steps]
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    n = 601
    lr, tau, magnet_interval = 0.05, 0.2, 200
    grid, A = payoff_matrix(DecoyWellGame(), n)
    Phi = rbf_basis(grid, m)
    i0 = on_decoy_logits(grid)
    i1 = on_decoy_logits(grid).copy()

    print(f"=== decoy on-trap, capacity-pinned | n={n}, m={m}, steps={steps}, "
          f"lr={lr} tau={tau} ===")
    print(f"    {'measure':<16} {'final':>9} {'tail':>9} {'min':>9}   verdict")
    measures = [
        ("uniform",        measure_fn("uniform", 0.0, 1.0)),
        ("on-policy pi",   measure_fn("onpolicy", 0.0, 1.0)),
        ("mix rho=0.05",   measure_fn("mix", 0.05, 1.0)),
        ("mix rho=0.02",   measure_fn("mix", 0.02, 1.0)),
        ("mix rho=0.10",   measure_fn("mix", 0.10, 1.0)),
        ("tempered b=0.5", measure_fn("tempered", 0.0, 0.5)),
    ]
    only = sys.argv[3] if len(sys.argv) > 3 else None
    if only:
        measures = [(lbl, b) for lbl, b in measures if only in lbl]
    out = {"n": n, "m": m, "steps": steps, "runs": []}
    for label, build in measures:
        _, _, hist = ebm_regression_mmd(A, Phi, build, steps, lr, tau,
                                        magnet_interval, i0, i1, log_every=200)
        tail = _tail(hist)
        mn = min(e for _, e in hist)
        verdict = "ESCAPED" if tail < 0.05 else ("crawling" if mn < 0.05 else "stuck")
        print(f"    {label:<16} {hist[-1][1]:9.4f} {tail:9.4f} {mn:9.4f}   {verdict}",
              flush=True)
        out["runs"].append({"measure": label, "history": hist, "tail_expl": tail,
                            "min_expl": mn})

    fname = f"exp7b_m{m}.json"
    with open(RESULTS / fname, "w") as f:
        json.dump(out, f)
    print(f"\nsaved -> {RESULTS / fname}")


if __name__ == "__main__":
    main()
