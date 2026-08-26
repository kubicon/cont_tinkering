"""Run the with-/without-magnet configs side by side and plot the learning dynamics.

Both configs drive `run_idealized.py` (exact gradients, no sampling noise) on the
bilinear matching-pennies game `payoff(a0, a1) = a0 * a1`, whose Nash sits at the
midpoint of the action box. The only *intended* difference between the two is
`ppo.magnet_gaussian_kl_coef` -- pass `--init` to also equalize the starting
means, which the two files as written do not share.

The figure has three panels:

  * left   -- the phase plane: E[a] of player 0 against E[a] of player 1, one
              curve per config, colored light (t=0) to dark (t=T). This is where
              the failure is visible: without the magnet the pair orbits the Nash
              instead of spiralling into it.
  * top r. -- exploitability (NashConv) of the current and of the average
              (target-EMA) strategies, against iteration.
  * bot r. -- distance of (E[a_0], E[a_1]) to the Nash point, and the policy std,
              against iteration.

The phase-plane trace is recorded every iteration (the orbit period is only
~2*pi/lr iterations, so logging once per outer step aliases the circle into a
polygon); exploitability, which needs an eager best-response sweep, is evaluated
on a subsample.

usage:
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --init 0.2
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --reuse   # replot cached runs
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, so `import run_idealized` works

import jax
import jax.numpy as jnp
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


def _round(values) -> list[float]:
    """9 significant digits -- plenty for the plot, and a third of the cache size."""
    return [float(f"{v:.9g}") for v in np.asarray(values).tolist()]


def _open(path: Path, mode: str):
    """`open`, transparently gzipping when the cache path ends in `.gz`."""
    return gzip.open(path, mode) if path.suffix == ".gz" else open(path, mode)


def simulate(config_path: Path, init: float | None, expl_points: int) -> dict:
    """Run one config, tracing the mixture every iteration.

    Returns per-iteration `E[a]` for both players (current and average strategy),
    the mean policy std, and exploitability on a subsample of `expl_points`.
    """
    game_config, cfg, _ = ri.load_config(config_path)
    game = game_config.build()
    if cfg.mode != "self_play":
        raise SystemExit(f"{config_path.name}: this script only handles train.mode: self_play")
    if init is not None:
        cfg = dataclasses.replace(
            cfg, idealized=dataclasses.replace(
                cfg.idealized, init_means=[init] * cfg.num_components))

    p0, p1 = ri.build_init(game, cfg)
    backend = ri.build_backend(game, cfg)
    iteration = ri.build_iteration(game, cfg, backend)

    def trace(carry, _):
        carry, _ = iteration(carry, None)
        cur0, cur1, _, _, avg0, avg1, _, _ = carry
        return carry, (cur0, cur1, avg0, avg1)

    # `fixed_handle` is unused in self_play, but the carry shape must still match.
    fixed_handle = jnp.zeros_like(backend.handle(p0)) if backend.name == "quadrature" else None
    carry = (p0, p1, p0, p1, p0, p1, fixed_handle, jnp.zeros((), dtype=jnp.int64))
    _, ys = jax.jit(lambda c: jax.lax.scan(trace, c, None, length=cfg.total_iters))(carry)

    def mean_action(p) -> np.ndarray:
        """E[a] = sum_k w_k mu_k, per traced iteration."""
        return np.asarray(jnp.sum(jax.nn.softmax(p.logits, axis=-1) * p.means, axis=-1))

    cur0, cur1, avg0, avg1 = ys
    t = np.arange(1, cfg.total_iters + 1, dtype=float)
    std = np.asarray(jnp.exp(cur0.log_std).mean(axis=-1) + jnp.exp(cur1.log_std).mean(axis=-1)) / 2

    # exploitability: eager (it builds a best-response grid), so subsample.
    idx = np.unique(np.linspace(0, cfg.total_iters - 1, expl_points).round().astype(int))
    take = lambda p, i: ri.Params(p.logits[i], p.means[i], p.log_std[i])  # noqa: E731
    expl = lambda a, b: float(backend.exploitability(backend.handle(a), backend.handle(b)))  # noqa: E731

    lo, hi = ri._bounds(game)
    return {
        "config": config_path.name,
        "game": type(game).__name__,
        "lo": lo, "hi": hi,
        "lr": cfg.lr,
        "magnet_gaussian_kl_coef": cfg.magnet_gaussian_kl_coef,
        "init_mean0": float(mean_action(take(cur0, 0))), 
        "t": _round(t),
        "x": _round(mean_action(cur0)),
        "y": _round(mean_action(cur1)),
        "x_avg": _round(mean_action(avg0)),
        "y_avg": _round(mean_action(avg1)),
        "std": _round(std),
        "expl_t": _round(t[idx]),
        "expl": [expl(take(cur0, i), take(cur1, i)) for i in idx],
        "expl_avg": [expl(take(avg0, i), take(avg1, i)) for i in idx],
    }


def _fade(ax, x, y, cmap, label):
    """Draw `(x, y)` as a polyline whose color darkens along the trajectory."""
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap), linewidth=1.1, zorder=3)
    lc.set_array(np.linspace(0.25, 1.0, len(segs)))  # start light, end dark
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)
    solid = plt.get_cmap(cmap)(0.85)
    ax.plot([], [], color=solid, linewidth=1.6, label=label)
    ax.plot(x[0], y[0], "o", color=solid, markersize=8, markerfacecolor="white",
            markeredgewidth=1.8, zorder=5)
    ax.plot(x[-1], y[-1], "*", color=solid, markersize=15, zorder=5)
    return solid


def plot(results: list[tuple[str, dict, str]], out: Path) -> None:
    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.33, wspace=0.26)
    ax_phase = fig.add_subplot(gs[:, 0])
    ax_expl = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_std = ax_dist.twinx()

    lo, hi = results[0][1]["lo"], results[0][1]["hi"]
    nash = 0.5 * (lo + hi)  # the payoff is bilinear, so the Nash mean is the box midpoint

    for label, r, cmap in results:
        x, y, t = np.array(r["x"]), np.array(r["y"]), np.array(r["t"])
        tag = f"{label} (init {r['init_mean0']:+.2f}, magnet {r['magnet_gaussian_kl_coef']:g})"
        solid = _fade(ax_phase, x, y, cmap, tag)
        ax_expl.plot(r["expl_t"], r["expl"], color=solid, linewidth=1.4, label=f"{label}: current")
        ax_expl.plot(r["expl_t"], r["expl_avg"], color=solid, linewidth=1.1, linestyle="--",
                     alpha=0.75, label=f"{label}: average")
        ax_dist.plot(t, np.hypot(x - nash, y - nash), color=solid, linewidth=1.4, label=label)
        ax_std.plot(t, r["std"], color=solid, linewidth=1.0, linestyle=":", alpha=0.85)

    ax_phase.plot(nash, nash, "kx", markersize=12, markeredgewidth=2.5, zorder=6, label="Nash")
    ax_phase.axhline(nash, color="0.85", linewidth=0.8, zorder=1)
    ax_phase.axvline(nash, color="0.85", linewidth=0.8, zorder=1)
    pad = 0.04 * (hi - lo)
    ax_phase.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
                 xlabel=r"$E[a]$ of player 0", ylabel=r"$E[a]$ of player 1",
                 title=f"phase plane -- {results[0][1]['game']}\n"
                       "circle = start, star = end, color darkens with iteration")
    ax_phase.set_aspect("equal")
    ax_phase.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8,
                    frameon=False)  # the orbit fills the box, so keep the legend off it

    ax_expl.set(yscale="log", ylabel="exploitability", title="NashConv")
    ax_expl.legend(fontsize=7)
    ax_dist.set(yscale="log", xlabel="iteration",
                ylabel=r"$\|(E[a_0], E[a_1]) - \mathrm{Nash}\|$")
    ax_dist.set_title("distance to Nash (solid) and policy std (dotted)")
    ax_std.set_ylabel("policy std (dotted)")
    for ax in (ax_expl, ax_dist):
        ax.grid(alpha=0.25)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=float, default=None,
                    help="override both configs' idealized.init_means, so the runs differ "
                         "only in the magnet coefficient (default: use each config's own)")
    ap.add_argument("--expl-points", type=int, default=200,
                    help="number of iterations at which exploitability is evaluated")
    ap.add_argument("--out", type=Path, default=HERE / "dynamics.png")
    ap.add_argument("--cache", type=Path, default=HERE / "dynamics_history.json.gz",
                    help="gzipped JSON trace cache (a per-iteration trace is ~4 MB raw)")
    ap.add_argument("--reuse", action="store_true",
                    help="replot from the cached histories instead of re-running")
    args = ap.parse_args()

    if args.reuse:
        cached = json.loads(_open(args.cache, "rt").read())
        results = [(label, cached[name], cmap) for name, label, cmap in RUNS]
    else:
        results, cache = [], {}
        for name, label, cmap in RUNS:
            print(f"running {name} ...", flush=True)
            r = simulate(HERE / name, args.init, args.expl_points)
            print(f"  final exploitability: current {r['expl'][-1]:+.6f} | "
                  f"average {r['expl_avg'][-1]:+.6f}", flush=True)
            results.append((label, r, cmap))
            cache[name] = r
        with _open(args.cache, "wt") as f:
            json.dump(cache, f)
        print(f"saved histories -> {args.cache}")

    plot(results, args.out)


if __name__ == "__main__":
    main()
