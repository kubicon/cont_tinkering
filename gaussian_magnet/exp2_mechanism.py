"""Exp 2 -- mechanism: why the magnet is essential at kappa=0, not a tuning fix.

Three checks on the pure bilinear game (kappa=0):

(A) lr-invariance of the divergence. Plain natural-gradient descent-ascent on a
    bilinear saddle has the discrete map z -> z + lr*Rot(z), whose eigenvalues
    are 1 +- i*lr*c, modulus sqrt(1+(lr c)^2) > 1 for EVERY lr>0. So magnet OFF
    diverges at every step size -- it is not rescued by tuning. The magnet ON
    converges across the same lr range.

(B) last-iterate vs average-iterate. Classic GDA on a bilinear game has a
    *bounded, converging time-average* even while the last iterate cycles/blows
    up. The magnet's contribution is specifically LAST-iterate convergence, which
    is what a deployed policy actually uses. We report both.

(C) trajectory radius over time: OFF spirals outward, ON spirals inward.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from magnet_core import Cfg, init, run  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
INIT0, INIT1 = init(1.5, 0.5), init(-1.0, 0.5)


def avg_iterate_expl(hist, cfg):
    """Exploitability of the running time-average of the means (kappa=0: value 0)."""
    mu0 = np.mean(hist["mu0"]); mu1 = np.mean(hist["mu1"])
    # at kappa=0, expl of pure means (m0,m1) = |c*m0*m1 - max_a c*a*m1| + ... ; use engine
    from magnet_core import G, exploitability
    import jax.numpy as jnp
    p0 = G(jnp.asarray(mu0), jnp.asarray(np.log(1e-3)))
    p1 = G(jnp.asarray(mu1), jnp.asarray(np.log(1e-3)))
    return exploitability(p0, p1, cfg)


def main():
    out = {}

    print("=== (A) lr-invariance of the magnet-OFF divergence (kappa=0) ===")
    print(f"{'lr':>6} | {'OFF final':>12} {'OFF |z|':>9} | {'ON final':>10} {'ON |z|':>8}")
    out["lr_sweep"] = {}
    for lr in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
        base = dict(kappa=0.0, coupling=1.0, lr=lr, steps=20000)
        off, _, _ = run(Cfg(tau=0.0, **base), INIT0, INIT1)
        on, _, _ = run(Cfg(tau=0.2, **base), INIT0, INIT1)
        out["lr_sweep"][lr] = {"off": off, "on": on}
        print(f"{lr:6.3f} | {off['final']:12.4f} {off['iterate_radius']:9.3f} "
              f"| {on['final']:10.4f} {on['iterate_radius']:8.3f}")

    print("\n=== (B) last-iterate vs average-iterate (kappa=0, lr=0.05) ===")
    cfg_off = Cfg(kappa=0.0, coupling=1.0, lr=0.05, tau=0.0, steps=20000)
    cfg_on = Cfg(kappa=0.0, coupling=1.0, lr=0.05, tau=0.2, steps=20000)
    off, hoff, _ = run(cfg_off, INIT0, INIT1)
    on, hon, _ = run(cfg_on, INIT0, INIT1)
    off_avg = avg_iterate_expl(hoff, cfg_off)
    on_avg = avg_iterate_expl(hon, cfg_on)
    out["last_vs_avg"] = {
        "off_last": off["final"], "off_avg": off_avg,
        "on_last": on["final"], "on_avg": on_avg,
    }
    print(f"  magnet OFF: last-iterate expl = {off['final']:10.4f}   time-avg expl = {off_avg:.4f}")
    print(f"  magnet ON : last-iterate expl = {on['final']:10.4f}   time-avg expl = {on_avg:.4f}")
    print("  -> OFF: bounded/converging average but diverging last iterate (classic GDA);")
    print("     ON : last iterate itself converges (what the deployed policy uses).")

    print("\n=== (C) trajectory radius |z_t| over time (kappa=0, lr=0.05) ===")
    def radii(h):
        return np.hypot(np.array(h["mu0"]), np.array(h["mu1"]))
    roff, ron = radii(hoff), radii(hon)
    ts = hoff["t"]
    marks = [0, len(ts) // 4, len(ts) // 2, 3 * len(ts) // 4, len(ts) - 1]
    print(f"  {'t':>7} | {'OFF |z|':>9} | {'ON |z|':>9}")
    for i in marks:
        print(f"  {ts[i]:7d} | {roff[i]:9.4f} | {ron[i]:9.4f}")
    out["radius_trace"] = {"t": [ts[i] for i in marks],
                           "off": [float(roff[i]) for i in marks],
                           "on": [float(ron[i]) for i in marks]}

    (RESULTS / "exp2.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'exp2.json'}")


if __name__ == "__main__":
    main()
