"""Run the with-/without-magnet configs side by side and plot the learning dynamics.

Both configs drive `run_idealized.py` (exact gradients, no sampling noise) on the
bilinear matching-pennies game, whose Nash sits at the midpoint of the action box.
The only intended difference between them is `ppo.magnet_gaussian_kl_coef`.

The figure has three panels:

  * left   -- the phase plane: E[a] of player 0 against E[a] of player 1, one
              curve per config, colored from light (t=0) to dark (t=T). This is
              where the failure is visible: without the magnet the pair orbits
              the Nash instead of spiralling into it.
  * top r. -- exploitability (NashConv) against iteration.
  * bot r. -- distance of (E[a_0], E[a_1]) to the Nash point, and the policy
              std, against iteration.

usage:
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --init 0.5 --log-every 25
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --reuse   # replot cached runs
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, so `import run_idealized` works

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import run_idealized as ri

RUNS = [
    ("wo_magnet.yaml", "no magnet", "Reds"),
    ("with_magnet.yaml", "magnet", "Blues"),
]


def _rechunk(total: int, every: int) -> tuple[int, ...]:
    """Uniform logging chunks summing to `total` (the last one may be shorter)."""
    every = max(1, min(every, total))
    n, rem = divmod(total, every)
    return (every,) * n + ((rem,) if rem else ())


def simulate(config_path: Path, log_every: int, init: float | None) -> dict:
    """Run one config and return `{"history": [...], "lo": ..., "hi": ...}`."""
    game_config, cfg, _ = ri.load_config(config_path)
    game = game_config.build()

    idealized = dataclasses.replace(cfg.idealized, verbose=False, log_out=None)
    if init is not None:
        idealized = dataclasses.replace(idealized, init_means=[init] * cfg.num_components)
    cfg = dataclasses.replace(cfg, idealized=idealized,
                              chunks=_rechunk(cfg.total_iters, log_every))

    p0, p1 = ri.build_init(game, cfg)
    _, _, history, _ = ri.run(game, cfg, p0, p1)
    lo, hi = ri._bounds(game)
    return {"history": history, "lo": lo, "hi": hi,
            "game": type(game).__name__, "config": str(config_path)}


def _curves(result: dict) -> dict[str, np.ndarray]:
    """Per-iteration scalars: the mixture means E[a], exploitability, mean std."""
    h = result["history"]
    w0, m0 = np.array([e["w0"] for e in h]), np.array([e["means0"] for e in h])
    w1, m1 = np.array([e["w1"] for e in h]), np.array([e["means1"] for e in h])
    return {
        "t": np.array([e["t"] for e in h], dtype=float),
        "x": (w0 * m0).sum(axis=1),          # E[a] of player 0
        "y": (w1 * m1).sum(axis=1),          # E[a] of player 1
        "expl": np.array([e["expl"] for e in h]),
        "target_expl": np.array([e["target_expl"] for e in h]),
        "std": np.array([np.mean(e["std0"] + e["std1"]) for e in h]),
    }


def _fade(ax, x, y, t, cmap, label):
    """Draw `(x, y)` as a line whose color darkens with `t`."""
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap), linewidth=1.6, zorder=3)
    lc.set_array(0.25 + 0.75 * t[:-1] / max(t[-1], 1))  # start light, end dark
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)
    solid = plt.get_cmap(cmap)(0.85)
    ax.plot([], [], color=solid, linewidth=1.6, label=label)
    ax.plot(x[0], y[0], "o", color=solid, markersize=7, markerfacecolor="white", zorder=4)
    ax.plot(x[-1], y[-1], "*", color=solid, markersize=14, zorder=4)
    return solid


def plot(results: list[tuple[str, dict, str]], out: Path) -> None:
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.32, wspace=0.24)
    ax_phase, ax_expl, ax_dist = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    lo, hi = results[0][1]["lo"], results[0][1]["hi"]
    nash = 0.5 * (lo + hi)  # bilinear payoff => the Nash mean is the box midpoint

    ax_dist_r = ax_dist.twinx()
    for label, result, cmap in results:
        c = _curves(result)
        solid = _fade(ax_phase, c["x"], c["y"], c["t"], cmap, label)
        ax_expl.plot(c["t"], c["expl"], color=solid, linewidth=1.4, label=f"{label} (current)")
        ax_expl.plot(c["t"], c["target_expl"], color=solid, linewidth=1.0,
                     linestyle="--", alpha=0.7, label=f"{label} (average)")
        dist = np.hypot(c["x"] - nash, c["y"] - nash)
        ax_dist.plot(c["t"], dist, color=solid, linewidth=1.4, label=label)
        ax_dist_r.plot(c["t"], c["std"], color=solid, linewidth=1.0, linestyle=":", alpha=0.8)

    ax_phase.plot(nash, nash, "kx", markersize=11, markeredgewidth=2, zorder=5, label="Nash")
    ax_phase.axhline(nash, color="0.8", linewidth=0.8, zorder=1)
    ax_phase.axvline(nash, color="0.8", linewidth=0.8, zorder=1)
    pad = 0.04 * (hi - lo)
    ax_phase.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
                 xlabel=r"$E[a]$ of player 0", ylabel=r"$E[a]$ of player 1",
                 title=f"phase plane ({results[0][1]['game']})\n"
                       "open circle = start, star = end, color darkens with time")
    ax_phase.set_aspect("equal")
    ax_phase.legend(loc="upper right", fontsize=8)

    ax_expl.set(yscale="log", ylabel="exploitability", title="NashConv")
    ax_expl.legend(fontsize=7)
    ax_dist.set(yscale="log", xlabel="iteration", ylabel=r"$\|(E[a_0], E[a_1]) - $Nash$\|$")
    ax_dist_r.set_ylabel("policy std (dotted)")
    ax_dist.set_title("distance to Nash (solid) and std (dotted)")
    for ax in (ax_expl, ax_dist):
        ax.grid(alpha=0.25)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-every", type=int, default=50,
                    help="iterations between logged points (default: 50)")
    ap.add_argument("--init", type=float, default=None,
                    help="override both configs' idealized.init_means, so the two runs "
                         "differ only in the magnet coefficient")
    ap.add_argument("--out", type=Path, default=HERE / "dynamics.png")
    ap.add_argument("--cache", type=Path, default=HERE / "dynamics_history.json")
    ap.add_argument("--reuse", action="store_true",
                    help="replot from the cached histories instead of re-running")
    args = ap.parse_args()

    if args.reuse:
        cached = json.loads(args.cache.read_text())
        results = [(label, cached[name], cmap) for name, label, cmap in RUNS]
    else:
        results, cache = [], {}
        for name, label, cmap in RUNS:
            print(f"running {name} ...", flush=True)
            result = simulate(HERE / name, args.log_every, args.init)
            print(f"  final exploitability {result['history'][-1]['expl']:+.5f}")
            results.append((label, result, cmap))
            cache[name] = result
        args.cache.write_text(json.dumps(cache))
        print(f"saved histories -> {args.cache}")

    plot(results, args.out)


if __name__ == "__main__":
    main()
