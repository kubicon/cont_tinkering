"""Exp 3 -- the structural criterion: magnet helps iff curvature loses to coupling.

Sweep (kappa, coupling) and mark where the magnet flips the outcome from "diverges"
to "reaches Nash". The exp6/THEORY analysis says the Gaussian head is curvature-
carried when the own-mean Hessian (-kappa) dominates; the discrete rotation the
coupling injects has rate ~ lr*c. So the boundary should track kappa* ~ (a
constant) * lr * c^2 / (something) -- empirically, kappa* grows with c. We just
locate the boundary per c and check monotonicity.

Cell content: 'H' magnet HELPS (ON reaches Nash, OFF does not),
              '.' both converge (curvature regime, magnet redundant -> exp6),
              'x' neither converges.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from magnet_core import Cfg, init, run  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
INIT0, INIT1 = init(1.5, 0.5), init(-1.0, 0.5)

KAPPAS = [2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0]
COUPLINGS = [0.5, 1.0, 2.0, 4.0]
LR = 0.05


def classify(kappa, c):
    base = dict(kappa=kappa, coupling=c, lr=LR, steps=20000)
    off, _, _ = run(Cfg(tau=0.0, **base), INIT0, INIT1)
    on, _, _ = run(Cfg(tau=0.2, **base), INIT0, INIT1)
    ok_off = off["tail"] < 0.05
    ok_on = on["tail"] < 0.05
    if ok_on and not ok_off:
        return "H"
    if ok_on and ok_off:
        return "."
    return "x"


def main():
    out = {}
    print("rows = kappa (own curvature), cols = coupling c;  H=magnet helps  .=both ok  x=neither")
    print("        " + "".join(f"c={c:<6}" for c in COUPLINGS))
    boundary = {}
    for kappa in KAPPAS:
        cells = []
        for c in COUPLINGS:
            v = classify(kappa, c)
            cells.append(v)
            out[f"k{kappa}_c{c}"] = v
        print(f"k={kappa:<5} " + "".join(f"  {v:<6}" for v in cells))
    # locate, per coupling, the largest kappa at which the magnet still helps
    print("\nlargest kappa where magnet still HELPS, per coupling (the crossover kappa*):")
    for j, c in enumerate(COUPLINGS):
        kstar = None
        for kappa in KAPPAS:
            if out[f"k{kappa}_c{c}"] == "H":
                kstar = kappa
                break  # KAPPAS descending -> first H is the largest
        boundary[c] = kstar
        print(f"  c={c}:  kappa* = {kstar}")
    out["crossover"] = boundary
    (RESULTS / "exp3.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'exp3.json'}")


if __name__ == "__main__":
    main()
