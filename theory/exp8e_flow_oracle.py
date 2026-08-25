"""Exp 8e — the scale-space flow oracle: a closed-form predictor of the exp8 edges.

The exp8/8b/8c/8d sweeps showed that, in the symmetric transit regime (where
the coupling term vanishes identically along the trajectory — verified by the
bit-identical c in {0.5, 1, 2} runs of exp8c-F), the fate of a broad-init MMD
run is decided by pure smoothed-well ascent with a self-annealing scale. The
right reduced model is therefore the natural-gradient (Fisher-Rao) flow of a
SINGLE Gaussian component (mu, s) on the smoothed well amplitude

    D_s(mu) = sum_b h_b w_b / sqrt(w_b^2 + s^2) * exp(-(mu-c_b)^2 / (2(w_b^2+s^2)))

    dmu/dt     = s^2      * dD_s/dmu          (Fisher metric I_mu = 1/s^2)
    d(log s)/dt = 1/2     * dD_s/d(log s)     (I_logs = 2)

— exactly the Gaussian-head update of `idealized_mmd.py` with the opponent
stripped out. It costs microseconds and knows nothing about MMD, the magnet,
the categorical head, or the opponent.

This script recomputes every threshold measured in exp8/8b/8c/8d from the
flow alone and prints predicted vs measured. Result (see THEORY.md): every
edge is reproduced to ~10%:

    edge                          flow prediction     measured (MMD)
    mass threshold  (s0=1.5)      h_d* in (0.20,0.22)  h_d* in (0.22,0.25)
    dominance scale (h_d=0.7)     s*  in (0.25,0.28)   s*  in (0.28,0.30)
    empty window (init +-2.5)     no s0 works          no s0 works (H1)
    light-decoy window (+-2.5)    s0 >= ~0.3 works     s0 >= 0.5 works (H2)

Run:  .venv/bin/python theory/exp8e_flow_oracle.py
"""
from __future__ import annotations

import numpy as np


def flow(mu0, s0, bumps, dt=0.02, steps=400_000, floor=1e-3):
    """Integrate the single-component natural-gradient flow; return (mu, s)."""
    mu, ls = float(mu0), float(np.log(s0))
    for _ in range(steps):
        s2 = np.exp(2 * ls)
        g_mu = 0.0
        g_ls = 0.0
        for (c, h, w) in bumps:
            var = w * w + s2
            amp = h * w / np.sqrt(var)
            e = np.exp(-((mu - c) ** 2) / (2 * var))
            g_mu += amp * e * (-(mu - c) / var)
            g_ls += e * (-h * w * s2 / var**1.5 + amp * (mu - c) ** 2 * s2 / var**2)
        mu += dt * s2 * g_mu
        ls = float(np.clip(ls + dt * 0.5 * g_ls, np.log(floor), 0.0))
        if np.exp(ls) <= floor * 1.01 and abs(g_mu) < 1e-12:
            break
    return mu, float(np.exp(ls))


def decoy_bumps(h_d, w_d=0.45):
    """DecoyWellGame default geometry: peaks +-1 (h=1, w=0.05), decoy at 0."""
    return [(-1.0, 1.0, 0.05), (1.0, 1.0, 0.05), (0.0, h_d, w_d)]


def endpoint(mu, s):
    if abs(abs(mu) - 1.0) < 0.05:
        return "PEAK"
    if abs(mu) < 0.05:
        return "decoy"
    return f"frozen@{mu:+.2f}"


def bisect(lo, hi, pred, iters=20):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


def main():
    print("(1) mass threshold at s0=1.5, init mu=1.5  [measured: (0.22, 0.25]]")
    lo, hi = bisect(0.05, 0.7,
                    lambda hd: endpoint(*flow(1.5, 1.5, decoy_bumps(hd))) == "PEAK",
                    iters=12)
    print(f"    flow h_d* in ({lo:.3f}, {hi:.3f})  (mass ratio {9*lo:.2f}-{9*hi:.2f})")

    print("(2) dominance scale s* at h_d=0.7, init mu=1.5  [measured: (0.28, 0.30)]")
    lo, hi = bisect(0.05, 1.5,
                    lambda s0: endpoint(*flow(1.5, s0, decoy_bumps(0.7))) == "PEAK",
                    iters=12)
    print(f"    flow s* in ({lo:.3f}, {hi:.3f})")

    print("(3) window from init mu=2.5:")
    for h_d, label in [(0.7, "heavy decoy (mass x6.3) [measured H1: none]"),
                       (0.1, "light decoy (mass x0.9) [measured H2: s0>=0.5]")]:
        oks = [s0 for s0 in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2)
               if endpoint(*flow(2.5, s0, decoy_bumps(h_d))) == "PEAK"]
        print(f"    {label}: flow-converging s0 = {oks if oks else 'NONE'}")

    print("(4) full endpoint maps (flow):")
    for h_d in (0.1, 0.2, 0.25, 0.7):
        row = [endpoint(*flow(1.5, s0, decoy_bumps(h_d)))
               for s0 in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5)]
        print(f"    h_d={h_d:.2f} from mu=1.5: " + "  ".join(f"{r:>12s}" for r in row))


if __name__ == "__main__":
    main()
