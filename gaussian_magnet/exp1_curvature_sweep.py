"""Exp 1 -- the counterexample and the crossover.

For each own-curvature kappa, run the isolated Gaussian head with the magnet
OFF (tau=0, pure natural-gradient descent-ascent) vs ON (tau=0.2, MMD proximal),
from an off-Nash init. Report last-iterate exploitability.

Prediction:
  * large kappa: both converge; magnet mildly WORSE or equal (exp6 reproduced).
  * kappa -> 0 : magnet OFF cycles/diverges (tail_max large, iterate flies to the
                 box edge); magnet ON converges to the Nash at the origin.
The crossover kappa* is where own curvature stops dominating the discrete-time
rotation induced by the coupling.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from magnet_core import Cfg, init, run  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

KAPPAS = [4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.0]
INIT0, INIT1 = init(1.5, 0.5), init(-1.0, 0.5)   # off the Nash (origin), asymmetric
COUPLING = 1.0
LR = 0.05


def main():
    out = {}
    hdr = f"{'kappa':>6} | {'OFF final':>10} {'OFF tailmax':>11} {'OFF |z|':>8} | {'ON final':>10} {'ON tailmax':>11} {'ON |z|':>8} | verdict"
    print(hdr)
    print("-" * len(hdr))
    for kappa in KAPPAS:
        base = dict(kappa=kappa, coupling=COUPLING, lr=LR, steps=20000)
        off, _, _ = run(Cfg(tau=0.0, **base), INIT0, INIT1)
        on, _, _ = run(Cfg(tau=0.2, **base), INIT0, INIT1)
        # "magnet helps" = ON reaches Nash (tail<0.05) while OFF does not (tail>0.1)
        helps = on["tail"] < 0.05 and off["tail"] > 0.1
        verdict = "MAGNET HELPS" if helps else ("both ok" if on["tail"] < 0.05 and off["tail"] < 0.05 else "neither")
        out[kappa] = {"off": off, "on": on, "helps": helps}
        print(f"{kappa:6.2f} | {off['final']:10.4f} {off['tail_max']:11.4f} {off['iterate_radius']:8.3f} "
              f"| {on['final']:10.4f} {on['tail_max']:11.4f} {on['iterate_radius']:8.3f} | {verdict}")
    (RESULTS / "exp1.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {RESULTS / 'exp1.json'}")


if __name__ == "__main__":
    main()
