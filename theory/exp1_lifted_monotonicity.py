"""Exp 1 — Layer 1 of the theory: the lifted (measure-space) game is monotone.

Two checks:

(a) **Exact skewness.** For the mixed extension of a zero-sum game the
    pseudo-gradient field is F(pi0, pi1) = (-A pi1, A^T pi0); for random pairs
    z, z' we check <F(z)-F(z'), z-z'> = 0 to machine precision. This is the
    monotonicity (with equality) that puts measure-space MMD inside the Sokota
    et al. convergence theory.

(b) **Global convergence of tabular MMD** on a fine action grid, for exactly the
    games whose *parametric mixture* dynamics have documented traps:
      - MultiPointGame (two_point): control, converges parametrically too;
      - DecoyWellGame: parametric mixture has a stable non-Nash fixed point on
        the decoy (expl ~0.72). Prediction P1: tabular MMD converges even when
        *initialized on the decoy*;
      - ForsakenGame: classic non-monotone cycling example; its lifted game is
        monotone, so tabular MMD should converge for admissible step sizes and
        may cycle when the step-size condition is violated (docstring reports
        exactly this sensitivity).

Run:  .venv/bin/python theory/exp1_lifted_monotonicity.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from games.examples import DecoyWellGame, ForsakenGame, MultiPointGame

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def payoff_matrix(game, n: int):
    space = game.action_space(0)
    grid = jnp.linspace(float(space.low[0]), float(space.high[0]), n)
    acts = grid[:, None]  # (n, 1) actions
    A = jax.vmap(lambda a: jax.vmap(lambda b: game.payoff(a, b))(acts))(acts)
    return grid, jnp.asarray(A, dtype=jnp.float64)


# ---- (a) exact skewness of the lifted field --------------------------------
def check_skewness(A, seed: int = 0, trials: int = 100) -> float:
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    worst = 0.0
    for _ in range(trials):
        p0, p0p = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
        p1, p1p = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))
        # min-max field: player 0 maximizes -> ascent on A pi1; write descent form
        F = np.concatenate([-(A @ p1), A.T @ p0])
        Fp = np.concatenate([-(A @ p1p), A.T @ p0p])
        dz = np.concatenate([p0 - p0p, p1 - p1p])
        worst = max(worst, abs(float((F - Fp) @ dz)))
    return worst


# ---- (b) tabular MMD --------------------------------------------------------
def tabular_mmd(A, steps: int, lr: float, tau: float, magnet_interval: int,
                init0=None, init1=None, log_every: int = 200):
    n = A.shape[0]
    l0 = jnp.zeros(n) if init0 is None else jnp.asarray(init0, dtype=jnp.float64)
    l1 = jnp.zeros(n) if init1 is None else jnp.asarray(init1, dtype=jnp.float64)
    m0, m1 = l0, l1

    @jax.jit
    def step(l0, l1, m0, m1):
        p0, p1 = jax.nn.softmax(l0), jax.nn.softmax(l1)
        q0, q1 = A @ p1, -(A.T @ p0)
        upd = lambda l, q, m: (lr * q + lr * tau * jax.nn.log_softmax(m)
                               + jax.nn.log_softmax(l)) / (1 + lr * tau)
        return upd(l0, q0, m0), upd(l1, q1, m1)

    @jax.jit
    def expl(l0, l1):
        p0, p1 = jax.nn.softmax(l0), jax.nn.softmax(l1)
        return jnp.max(A @ p1) - jnp.min(A.T @ p0)

    hist = []
    for t in range(steps):
        l0, l1 = step(l0, l1, m0, m1)
        if (t + 1) % magnet_interval == 0:
            m0, m1 = l0, l1
        if t % log_every == 0 or t == steps - 1:
            hist.append((t, float(expl(l0, l1))))
    return l0, l1, hist


def run_case(name, game, n, inits, steps=20000, lr=0.05, tau=0.2,
             magnet_interval=200, seed=0):
    grid, A = payoff_matrix(game, n)
    skew = check_skewness(np.asarray(A), seed=seed)
    print(f"\n=== {name} (grid n={n}) ===")
    print(f"  lifted-field skewness |<F(z)-F(z'), z-z'>| worst over 100 pairs: {skew:.3e}")
    out = {"name": name, "n": n, "skewness": skew, "runs": []}
    rng = np.random.default_rng(seed)
    for label, (i0, i1) in inits.items():
        _, _, hist = tabular_mmd(A, steps, lr, tau, magnet_interval, i0, i1)
        tail = float(np.mean([e for _, e in hist[int(len(hist) * 0.8):]]))
        print(f"  init={label:<22} final expl={hist[-1][1]:.4f}  tail expl={tail:.4f}"
              f"  {'PASS' if tail < 0.05 else 'FAIL'}")
        out["runs"].append({"init": label, "lr": lr, "tau": tau,
                            "history": hist, "tail_expl": tail})
    return out, grid


def main():
    results = []
    rng = np.random.default_rng(0)

    # -- two_point control ----------------------------------------------------
    game = MultiPointGame(peaks=(0.0, 1.0), width=0.1, coupling=1.0)
    n = 401
    rand = lambda: rng.normal(size=n)
    inits = {"uniform": (None, None),
             "random-1": (rand(), rand()),
             "random-2": (rand(), rand())}
    out, _ = run_case("two_point (MultiPointGame)", game, n, inits)
    results.append(out)

    # -- decoy well: the parametric trap game ---------------------------------
    game = DecoyWellGame()  # peaks +-1 h=1 w=0.05, decoy at 0 h=0.7 w=0.45, box [-3,3]
    n = 601
    grid = np.linspace(-3, 3, n)
    on_decoy = -0.5 * (grid / 0.05) ** 2          # logits of a sharp Gaussian at 0
    inits = {"uniform": (None, None),
             "random": (rng.normal(size=n), rng.normal(size=n)),
             "ON THE DECOY (trap)": (on_decoy, on_decoy.copy())}
    out, _ = run_case("decoy_well (DecoyWellGame)", game, n, inits)
    results.append(out)

    # -- forsaken: step-size sensitivity of a monotone-but-not-strongly game --
    game = ForsakenGame()
    n = 301
    grid_f, A = payoff_matrix(game, n)
    print(f"\n=== forsaken (grid n={n}) step-size sensitivity ===")
    out = {"name": "forsaken", "n": n, "runs": []}
    for lr, tau in [(0.05, 0.1), (0.1, 0.05), (0.02, 0.2)]:
        _, _, hist = tabular_mmd(A, 20000, lr, tau, 200)
        tail = float(np.mean([e for _, e in hist[int(len(hist) * 0.8):]]))
        print(f"  lr={lr:<5} tau={tau:<5} final expl={hist[-1][1]:.4f}  tail expl={tail:.4f}"
              f"  {'converges' if tail < 0.05 else 'cycles/stuck'}")
        out["runs"].append({"lr": lr, "tau": tau, "history": hist, "tail_expl": tail})
    results.append(out)

    with open(RESULTS / "exp1.json", "w") as f:
        json.dump(results, f)
    print(f"\nsaved -> {RESULTS / 'exp1.json'}")


if __name__ == "__main__":
    main()
