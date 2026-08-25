"""Exp 8 — a *global* convergence assumption: scale-space dominance (SSD).

The local theory (exp2–exp4) explains convergence island-by-island. This
experiment tests a candidate GAME-LEVEL assumption that decides which island a
natural (broad / spread) init reaches — i.e. the missing global hypothesis.

Definition. For the well D(a) = sum_b h_b exp(-(a-c_b)^2 / (2 w_b^2)), let
D_s = D * N(0, s^2) be the Gaussian-smoothed well (smoothed bump amplitude
h_b w_b / sqrt(w_b^2 + s^2) — decays with the bump's MASS h_b w_b, not height).

  SSD(s0): for every scale s in [s_floor, s0], gradient-ascent continuation on
  D_s starting from the init means, with s annealing down, terminates with the
  K components on the K distinct Nash atoms.

A cheap "continuation oracle" computes this by hill-climbing each init mean on
D_s while shrinking s — it knows NOTHING about the coupling, the opponent, the
categorical head, or MMD step sizes. The claim under test:

  (H) Under the standing bounds (coupling below exp3's curvature cap, step
  size below exp1/exp3's cap, K = |support|), mixture-MMD from a broad init
  converges to Nash IFF the continuation oracle on D alone succeeds — and the
  oracle's success is governed by a bump-mass inequality, so the assumption is
  checkable in closed form from the game definition.

Parts:
  (A) Decoy-mass sweep. DecoyWellGame with decoy height h_d swept so the
      decoy/peak mass ratio crosses 1. Broad init (means +-1.5, std 1.5).
      Prediction: sharp trap/converge threshold at the oracle's flip point.
  (B) Init-std sweep at fixed h_d = 0.7 (the original trap). Prediction: the
      empirical std threshold matches the oracle's flip scale s*.
  (C) Random-game validation. Random decoy games satisfying / violating the
      closed-form mass-dominance condition; broad init. Prediction:
      satisfying -> converge, violating (moment-matched decoy) -> trap.
  (D) Weakening test. A mass-dominant decoy that is NOT moment-matched
      (center off the target moment). If MMD escapes it, the assumption only
      needs to exclude *moment-matched* dominant decoys — a strictly broader
      game class (the coupling term rescues mismatched decoys).

Run:  .venv/bin/python theory/exp8_scale_dominance.py
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

CFG = MMDConfig(lr=0.05, magnet_coef=0.2, magnet_interval=200, steps=12000)
PASS = 0.1
PEAK_MASS = 1.0 * 0.05  # h_p * w_p of the true peaks


# ---------------- continuation oracle (knows only D) ------------------------
def smoothed_well(mu, s, game):
    c = np.asarray(game.bump_centers, dtype=float)
    h = np.asarray(game.bump_heights, dtype=float)
    w = np.asarray(game.bump_widths, dtype=float)
    var = w**2 + s**2
    amp = h * w / np.sqrt(var)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    return np.sum(amp[None, :] * np.exp(-(mu[:, None] - c[None, :]) ** 2 / (2 * var[None, :])), axis=1)


def _hill_climb(idx, vals):
    """Walk uphill on a 1-D grid from index `idx` to the nearest local max."""
    n = len(vals)
    while True:
        left = vals[idx - 1] if idx > 0 else -np.inf
        right = vals[idx + 1] if idx < n - 1 else -np.inf
        if left <= vals[idx] >= right:
            return idx
        idx = idx - 1 if left > right else idx + 1


def continuation_oracle(init_means, s0, game, s_floor=1e-3, decay=0.95, n_grid=6001):
    """Anneal s from s0 to s_floor; hill-climb each mean on D_s at every scale.

    Returns (ok, endpoints): ok=True iff the endpoints land on all |peaks|
    distinct true peaks (within 3 grid cells + smoothing tolerance).
    """
    lo = float(game.action_space(0).low[0])
    hi = float(game.action_space(0).high[0])
    grid = np.linspace(lo, hi, n_grid)
    idxs = [int(np.argmin(np.abs(grid - m))) for m in init_means]
    s = s0
    while True:
        vals = smoothed_well(grid, s, game)
        idxs = [_hill_climb(i, vals) for i in idxs]
        if s <= s_floor:
            break
        s = max(s * decay, s_floor)
    ends = np.array([grid[i] for i in idxs])
    peaks = np.asarray(game.peaks, dtype=float)
    tol = 5 * (hi - lo) / (n_grid - 1) + 0.02
    assigned = set()
    for e in ends:
        d = np.abs(peaks - e)
        k = int(np.argmin(d))
        if d[k] > tol or k in assigned:
            return False, ends
        assigned.add(k)
    return len(assigned) == len(peaks), ends


# ---------------- MMD run wrapper -------------------------------------------
def run_mmd(game, init_means, init_std, cfg=CFG):
    init = make_init(list(init_means), log_std=float(np.log(init_std)))
    t0 = time.time()
    _, _, hist = run(game, cfg, init, init, log_every=400)
    e = np.array([h["expl"] for h in hist])
    tail = float(e[int(len(e) * 0.7):].mean())
    return tail, hist[-1], time.time() - t0


def verdict(tail):
    return "NASH" if tail < PASS else "trap"


def main():
    out = {}

    # ---- (A) decoy-mass sweep, broad init ---------------------------------
    print("=== (A) decoy height sweep, init means +-1.5, std 1.5 ===")
    print(f"    peak mass = {PEAK_MASS:.3f}; decoy mass = 0.45*h_d "
          f"(mass ratio = 9*h_d)")
    heights = [0.05, 0.08, 0.10, 0.12, 0.14, 0.18, 0.25, 0.4, 0.7]
    rows = []
    for hd in heights:
        game = DecoyWellGame(decoys=((0.0, hd, 0.45),))
        ok, ends = continuation_oracle([-1.5, 1.5], 1.5, game)
        tail, last, dt = run_mmd(game, [-1.5, 1.5], 1.5)
        match = (tail < PASS) == ok
        rows.append(dict(h_d=hd, mass_ratio=9 * hd, oracle=ok, tail=tail,
                         means=last["means0"], match=match))
        print(f"  h_d={hd:.2f} (mass x{9*hd:4.2f})  oracle={'NASH' if ok else 'trap'}"
              f"  mmd tail_expl={tail:7.4f} -> {verdict(tail)}"
              f"  means0={np.round(last['means0'], 3)}  "
              f"{'MATCH' if match else '** MISMATCH **'}  [{dt:.0f}s]")
    out["A_height_sweep"] = rows

    # ---- (B) init-std sweep at h_d = 0.7 ----------------------------------
    print("\n=== (B) init std sweep, original decoy (h_d=0.7, mass x6.3) ===")
    game = DecoyWellGame()
    # oracle flip scale: bisect the largest s0 for which the oracle still succeeds
    lo_s, hi_s = 0.02, 1.5
    if continuation_oracle([-1.5, 1.5], lo_s, game)[0]:
        for _ in range(24):
            mid = 0.5 * (lo_s + hi_s)
            if continuation_oracle([-1.5, 1.5], mid, game)[0]:
                lo_s = mid
            else:
                hi_s = mid
        s_star = 0.5 * (lo_s + hi_s)
    else:
        s_star = float("nan")
    print(f"  oracle flip scale s* = {s_star:.4f} (oracle NASH for s0 < s*)")
    rows = []
    for s0 in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.5]:
        ok = continuation_oracle([-1.5, 1.5], s0, game)[0]
        tail, last, dt = run_mmd(game, [-1.5, 1.5], s0)
        match = (tail < PASS) == ok
        rows.append(dict(s0=s0, oracle=ok, tail=tail, match=match))
        print(f"  s0={s0:4.2f}  oracle={'NASH' if ok else 'trap'}"
              f"  mmd tail_expl={tail:7.4f} -> {verdict(tail)}"
              f"  means0={np.round(last['means0'], 3)}  "
              f"{'MATCH' if match else '** MISMATCH **'}  [{dt:.0f}s]")
    out["B_std_sweep"] = rows
    out["B_s_star"] = s_star

    # ---- (C) random games satisfying / violating mass dominance -----------
    print("\n=== (C) random decoy games, broad init (std 1.5) ===")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(8):
        q = float(rng.uniform(0.3, 0.7))          # weight on peak -1
        wd = float(rng.uniform(0.2, 0.6))         # decoy width
        # satisfying: decoy mass strictly below peak mass (ratio ~U[0.2, 0.8])
        hd = float(rng.uniform(0.2, 0.8)) * PEAK_MASS / wd
        cd = float(rng.uniform(-0.6, 0.6))
        game = DecoyWellGame(weights=(q, 1 - q), decoys=((cd, hd, wd),))
        ok, _ = continuation_oracle([-1.5, 1.5], 1.5, game)
        tail, last, dt = run_mmd(game, [-1.5, 1.5], 1.5)
        rows.append(dict(kind="satisfy", q=q, decoy=(cd, hd, wd), oracle=ok,
                         tail=tail))
        print(f"  [satisfy {i}] w=({q:.2f},{1-q:.2f}) decoy=({cd:+.2f},h{hd:.2f},w{wd:.2f})"
              f" mass x{hd*wd/PEAK_MASS:.2f}  oracle={'NASH' if ok else 'trap'}"
              f"  mmd={tail:7.4f} -> {verdict(tail)} [{dt:.0f}s]")
    for i in range(8):
        q = float(rng.uniform(0.3, 0.7))
        wd = float(rng.uniform(0.3, 0.6))
        hd = float(rng.uniform(2.0, 6.0)) * PEAK_MASS / wd   # mass-dominant
        hd = min(hd, 0.95)                                    # decoy must stay below peak height
        cd = 1 - 2 * q  # moment-matched location: u(cd) = target first moment
        game = DecoyWellGame(weights=(q, 1 - q), decoys=((cd, hd, wd),))
        ok, _ = continuation_oracle([-1.5, 1.5], 1.5, game)
        tail, last, dt = run_mmd(game, [-1.5, 1.5], 1.5)
        rows.append(dict(kind="violate", q=q, decoy=(cd, hd, wd), oracle=ok,
                         tail=tail))
        print(f"  [violate {i}] w=({q:.2f},{1-q:.2f}) decoy=({cd:+.2f},h{hd:.2f},w{wd:.2f})"
              f" mass x{hd*wd/PEAK_MASS:.2f}  oracle={'NASH' if ok else 'trap'}"
              f"  mmd={tail:7.4f} -> {verdict(tail)} [{dt:.0f}s]")
    out["C_random_games"] = rows

    # ---- (D) mass-dominant but moment-MISMATCHED decoy --------------------
    print("\n=== (D) heavy decoy off the moment-matched point ===")
    for cd in [0.3, 0.5]:
        game = DecoyWellGame(decoys=((cd, 0.7, 0.45),))
        ok, _ = continuation_oracle([-1.5, 1.5], 1.5, game)
        tail, last, dt = run_mmd(game, [-1.5, 1.5], 1.5)
        out[f"D_offcenter_{cd}"] = dict(oracle=ok, tail=tail,
                                        means=last["means0"])
        print(f"  decoy center {cd:+.2f} (feat={cd:.2f} != 0), mass x6.3:"
              f"  oracle={'NASH' if ok else 'trap'}  mmd={tail:7.4f}"
              f" -> {verdict(tail)}  means0={np.round(last['means0'], 3)} [{dt:.0f}s]")

    with open(RESULTS / "exp8.json", "w") as f:
        json.dump(out, f, default=float)
    print(f"\nsaved -> {RESULTS / 'exp8.json'}")


if __name__ == "__main__":
    main()
