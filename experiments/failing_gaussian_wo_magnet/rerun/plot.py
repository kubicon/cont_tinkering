"""Score every seed's checkpoints of the multi-seed magnet grid, and plot mean +- 95% CI.

    python experiments/failing_gaussian_wo_magnet/rerun/plot.py
    python experiments/failing_gaussian_wo_magnet/rerun/plot.py --cells rot3_magnet_ppo --no-plots
    python experiments/failing_gaussian_wo_magnet/rerun/plot.py --no-std --show-seeds

Reads the tree `run_cell.py` writes, `<out>/<cell>/seed<k>/`, and does two things.

**Scoring.** Every checkpoint of every seed is re-scored with the measures of
`analyze.py` (`SpreadMeasure`, or `PointMassMeasure` with `--no-std`) -- the same
instrument for all three engines, so the curves are one quantity. The result is written
beside the checkpoints as `seed<k>/exploitability.pkl` (`exploitability_no_std.pkl`) and
reused on the next call unless `--force`, so redrawing a plot costs seconds. Scoring is
the expensive half; on the cluster run it per cell with `--cells <cell> --no-plots`
(what `generate_failing_gaussian_rerun.py`'s score jobs do), then plot once.

**Plots**, into `<out>/plots/` (or `--plots`):

    expl_<domain>_<engine>[_no_std].png   exploitability over iterations, magnet vs no magnet
    mp_means_<engine>[_no_std].png        matching pennies in the (player 0 mean, player 1 mean) plane

Seeds are pooled per checkpoint with no smoothing: the line is the mean over seeds and
the band the 95% confidence interval of that mean, `mean +- t_{0.975, n-1} * sd / sqrt(n)`,
with its lower edge clipped at 0.
`ppo` cells draw the Polyak-averaged target as a dashed line with its own band. In the
plane, the mean trajectory is drawn bold with 95% CI ellipses (per-axis half-widths) at
evenly spaced checkpoints, and `--show-seeds` adds each seed's own trajectory faintly --
worth looking at, since seeds that circle the Nash out of phase average to a curve that
spirals in even when no single run does.

The `idealized` cells are deterministic and have one run (see `run_cell.py`): they are
drawn as a plain line, with no band, and labelled `n=1`. A seed that has not finished
contributes only the checkpoints it has; the pooled curve stops where the shortest seed
does, and a warning says so.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import (                                          # noqa: E402
    DOMAIN_LABEL,
    DOMAINS,
    ENGINE_LABEL,
    ENGINES,
    MAGNET_COLOR,
    MAGNET_LABEL,
    MAGNETS,
    REPO_ROOT,
    _load_exact_cell,
    _load_ppo_cell,
    _load_saved,
    _symlog_axis,
    build_measure,
    cell_name,
    result_filename,
    score_cell,
)

DEFAULT_OUT = "data/failing_gaussian_rerun_seeds"
CI_LEVEL = 0.95
CI_ELLIPSES = 12


# ------------------------------------------------------------------------- scoring


def seed_dirs(cell_dir: Path) -> list[tuple[int, Path]]:
    found = [(int(p.name[4:]), p) for p in cell_dir.glob("seed*") if p.name[4:].isdigit()]
    return sorted(found)


def has_checkpoints(run_dir: Path, engine: str) -> bool:
    if engine == "ppo":
        return any(p.stem.isdigit() for p in run_dir.glob("*.pkl"))
    return (run_dir / "idealized_history.json").exists()


def score_run(cell: str, seed: int, run_dir: Path, no_std: bool, grid_points: int | None,
              max_gib: float, measure_cache: dict, force: bool) -> dict:
    """One seed of one cell, scored -- or its saved result, if there is one."""
    out_path = run_dir / result_filename(no_std)
    if not force:
        saved = _load_saved(out_path, grid_points)
        if saved is not None:
            saved["seed"] = seed      # the directory is authoritative
            print(f"  {cell}/seed{seed}: loaded {out_path.name} ({len(saved['expl_live'])} checkpoints)")
            return saved

    domain, magnet, engine = cell.split("_")
    key = (domain, no_std, grid_points)
    if key not in measure_cache:
        measure_cache[key] = build_measure(cell, no_std, grid_points, max_gib)
    measure, game, n, epochs = measure_cache[key]

    data = _load_ppo_cell(run_dir, game, epochs) if engine == "ppo" else _load_exact_cell(run_dir)
    scored = score_cell(cell, measure, data)
    result = {
        "cell": cell,
        "seed": seed,
        "domain": domain,
        "engine": engine,
        "magnet": magnet,
        "no_std": no_std,
        "grid_points": n,
        "steps": data["steps"],
        "t": data["t"],
        "expl_live": scored["expl_live"],
        "expl_target": scored["expl_target"],
        "expl_run": data["expl_run"],
        "target_expl_run": data["target_expl_run"],
        "means_0": [np.asarray(s0.means) for s0, _ in data["live"]],
        "means_1": [np.asarray(s1.means) for _, s1 in data["live"]],
        "weights_0": [np.asarray(s0.w) for s0, _ in data["live"]],
        "weights_1": [np.asarray(s1.w) for _, s1 in data["live"]],
    }
    with out_path.open("wb") as f:
        pickle.dump(result, f)
    print(f"  {cell}/seed{seed}: {len(result['expl_live'])} checkpoints, grid {n}/axis "
          f"-> {os.path.relpath(out_path, REPO_ROOT)}")
    return result


# ----------------------------------------------------------------------- pooling


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean over axis 0 and the CI half-width of that mean; `None` for a single run."""
    n = values.shape[0]
    mean = values.mean(axis=0)
    if n < 2:
        return mean, None
    sem = values.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, stats.t.ppf(0.5 + CI_LEVEL / 2, n - 1) * sem


def pool(cell: str, runs: list[dict]) -> dict:
    """Seeds of one cell stacked on their common checkpoints."""
    length = min(len(r["t"]) for r in runs)
    if any(len(r["t"]) != length for r in runs):
        lengths = {f"seed{r['seed']}": len(r["t"]) for r in runs}
        print(f"  warning: {cell} seeds have different checkpoint counts {lengths}; "
              f"pooling the first {length}")
    t = np.asarray(runs[0]["t"][:length], dtype=np.float64)

    def stacked(key):
        return np.asarray([r[key][:length] for r in runs], dtype=np.float64)

    def mean_axis(player):   # (S, L): the first component's first coordinate
        return np.asarray([[m[0, 0] for m in r[f"means_{player}"][:length]] for r in runs])

    pooled = {"t": t, "n": len(runs), "seeds": [r["seed"] for r in runs]}
    pooled["live"] = stacked("expl_live")
    pooled["target"] = None if runs[0]["expl_target"] is None else stacked("expl_target")
    if runs[0]["domain"] == "mp":
        pooled["x"], pooled["y"] = mean_axis(0), mean_axis(1)
    return pooled


# -------------------------------------------------------------------------- plots


def _band(ax, t, values, color, ls, label, show_seeds):
    mean, half = mean_ci(values)
    ax.plot(t, mean, color=color, lw=1.6 if ls == "-" else 1.2, ls=ls, label=label)
    drawn = list(mean)
    if half is not None:
        # Exploitability is >= 0, so an interval reaching below it only says the band is
        # wide; clipping keeps the symlog axis from spending half its height on negatives.
        lower = np.maximum(mean - half, 0.0)
        ax.fill_between(t, lower, mean + half, color=color, alpha=0.18 if ls == "-" else 0.1, lw=0)
        drawn += list(mean + half)
    if show_seeds and values.shape[0] > 1:
        for row in values:
            ax.plot(t, row, color=color, lw=0.5, ls=ls, alpha=0.25)
    return drawn


def plot_exploitability(pooled: dict, domain: str, engine: str, no_std: bool, show_seeds: bool,
                        out_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    everything = []
    for magnet in MAGNETS:
        p = pooled.get(cell_name(domain, magnet, engine))
        if p is None:
            continue
        color = MAGNET_COLOR[magnet]
        label = f"{MAGNET_LABEL[magnet]} (n={p['n']})"
        everything += _band(ax, p["t"], p["live"], color, "-", f"{label}, live", show_seeds)
        if p["target"] is not None:
            everything += _band(ax, p["t"], p["target"], color, "--", f"{label}, target", show_seeds)

    if not everything:
        plt.close(fig)
        return None

    _symlog_axis(ax, [v for v in everything if np.isfinite(v)])
    ax.set_xlabel("iteration")
    ax.set_ylabel("exploitability" + (" (spread set to 0)" if no_std else ""))
    band = f"mean, {CI_LEVEL:.0%} CI over seeds"
    ax.set_title(f"{DOMAIN_LABEL[domain]}\n{ENGINE_LABEL[engine]}  --  {band}"
                 + ("\nmeans only, spread zeroed" if no_std else ""), fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = out_dir / f"expl_{domain}_{engine}{'_no_std' if no_std else ''}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mp_means(pooled: dict, engine: str, no_std: bool, show_seeds: bool, out_dir: Path):
    """The matching-pennies trajectory in the plane of the two players' mean actions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    drawn = False
    for magnet in MAGNETS:
        p = pooled.get(cell_name("mp", magnet, engine))
        if p is None:
            continue
        color = MAGNET_COLOR[magnet]
        if show_seeds and p["n"] > 1:
            for xs, ys in zip(p["x"], p["y"]):
                ax.plot(xs, ys, color=color, lw=0.5, alpha=0.25)
        x, x_half = mean_ci(p["x"])
        y, y_half = mean_ci(p["y"])
        ax.plot(x, y, color=color, lw=1.4, alpha=0.9, label=f"{MAGNET_LABEL[magnet]} (n={p['n']})")
        if x_half is not None:
            for i in np.linspace(0, len(x) - 1, CI_ELLIPSES).round().astype(int):
                ax.add_patch(Ellipse((x[i], y[i]), 2 * x_half[i], 2 * y_half[i], facecolor=color,
                                     edgecolor=color, alpha=0.15, lw=0.6))
        ax.scatter([x[0]], [y[0]], color=color, marker="o", s=45, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.scatter([x[-1]], [y[-1]], color=color, marker="X", s=80, zorder=3,
                   edgecolor="white", linewidth=0.8)
        drawn = True

    if not drawn:
        plt.close(fig)
        return None

    ax.scatter([0], [0], color="black", marker="*", s=160, zorder=4, label="Nash (0, 0)")
    ax.axhline(0, color="0.75", lw=0.7, zorder=0)
    ax.axvline(0, color="0.75", lw=0.7, zorder=0)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel("player 0 mean action")
    ax.set_ylabel("player 1 mean action")
    ax.set_title(f"ContinuousMatchingPennies mean trajectory\n{ENGINE_LABEL[engine]}\n"
                 f"mean over seeds, ellipses = {CI_LEVEL:.0%} CI; circle = start, X = end",
                 fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    path = out_dir / f"mp_means_{engine}{'_no_std' if no_std else ''}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------- cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help="the tree run_cell.py wrote")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="cells to score/plot (default: every cell directory under --out)")
    parser.add_argument("--no-std", action="store_true",
                        help="score the means alone (spread set to 0); see analyze.py")
    parser.add_argument("--grid-points", type=int, default=None,
                        help="best-response grid per axis (default: each cell's idealized.grid_points)")
    parser.add_argument("--max-quadrature-gib", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", help="rescore even if a result file exists")
    parser.add_argument("--no-plots", action="store_true", help="score and save only")
    parser.add_argument("--show-seeds", action="store_true", help="also draw each seed faintly")
    parser.add_argument("--plots", type=Path, default=None, help="default: <out>/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = (REPO_ROOT / args.out).resolve()

    cells = args.cells or [cell_name(d, m, e) for d in DOMAINS for e in ENGINES for m in MAGNETS]
    cells = [c for c in cells if (out / c).is_dir()]
    if not cells:
        raise SystemExit(f"no cell directories under {out}")

    mode = "means only (spread = 0)" if args.no_std else "full mixtures"
    print(f"scoring {len(cells)} cells under {os.path.relpath(out, REPO_ROOT)}, {mode}")

    pooled, measure_cache = {}, {}
    for cell in cells:
        engine = cell.split("_")[2]
        runs = [score_run(cell, seed, run_dir, args.no_std, args.grid_points,
                          args.max_quadrature_gib, measure_cache, args.force)
                for seed, run_dir in seed_dirs(out / cell) if has_checkpoints(run_dir, engine)]
        if not runs:
            print(f"  {cell}: no seed with checkpoints yet, skipping")
            continue
        pooled[cell] = pool(cell, runs)

    if args.no_plots:
        return

    plots = args.plots or out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written = []
    for domain in DOMAINS:
        for engine in ENGINES:
            written.append(plot_exploitability(pooled, domain, engine, args.no_std,
                                               args.show_seeds, plots))
    for engine in ENGINES:
        written.append(plot_mp_means(pooled, engine, args.no_std, args.show_seeds, plots))
    written = [p for p in written if p is not None]

    print(f"\n{len(written)} plots -> {plots}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
