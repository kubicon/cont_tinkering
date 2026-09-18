"""Plot exploitability over training for a scored sequential sweep.

Reads every ``{run}/exploitability.pkl`` that ``score_sequential_sweep.py``
wrote under ``data/sequential_sweep/`` (or ``--out``) and draws, per game, one
line per solver configuration (seeds pooled) against three x-axes:

    checkpoint   the checkpoint index (solver-specific unit: a PPO chunk for
                 self_play / sac / discrete_mmd, a round for nfsp / psro --
                 not comparable across those two families)
    wall_time    training seconds, excluding measurement (``wall_time`` in
                 ``metrics.jsonl``), shown in hours
    samples      hands played (``episodes``) or, with ``--samples env_steps``,
                 decision nodes visited

The x-values come from the ``wall_time`` / ``episodes_train`` / ``env_steps``
arrays the scorer copies out of each run's ``metrics.jsonl``. Checkpoint 0 is
the untrained policy, saved before the clock starts, so its missing values
are taken as 0.

Seeds are pooled per checkpoint, with no resampling or fitting: for each
checkpoint index that every seed has, the point is (mean x over seeds, mean
exploitability over seeds), and the band is the 95% confidence interval of that
mean, mean +- t_{0.975, n-1} * std(ddof=1) / sqrt(n). Consecutive points are
joined by straight lines. Kuhn plots the exact
exploitability; Leduc / Blotto plot the RL lower bound, which can be <= 0 --
such points vanish on the default log y-axis, so pass ``--linear-y`` to see
them.

Writes to ``{out}/plots/`` (or ``--plots``):

    {game}__checkpoint.png, {game}__wall_time.png, {game}__samples.png
    {game}__combined.png   two panels side by side: samples, wall_time (shared y-axis)
    {game}__capacity.png   --capacity only: exploitability against representation size
    overview.png           rows = games, columns = the three x-axes

``--capacity`` is the plot the capacity sweep exists for (see
``generate_sequential_sweep.py --experiment capacity``): the x-axis is the number of
mixture components (``self_play``) or grid bins (``discrete_mmd``) rather than training
progress, one point per setting at the same budget. Runs whose name carries no such
count are left out of it, so pointing ``--capacity`` at the main sweep's tree simply
writes nothing.

    python rci_scripts/plot_sequential_sweep.py
    python rci_scripts/plot_sequential_sweep.py --games kuhn --target
    python rci_scripts/plot_sequential_sweep.py --samples env_steps --show-seeds
    python rci_scripts/plot_sequential_sweep.py --out data/sequential_capacity --capacity
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "sequential_sweep"
RESULT_FILENAME = "exploitability.pkl"
SEED_SUFFIX_RE = re.compile(r"__seed\d+$")
# A capacity run: one solver at one representation size, as
# `generate_sequential_sweep.py --experiment capacity` names it -- `discrete_mmd__bins16`
# or `self_play__num_components8`. The axis name is the config key that was swept, so a
# sweep of anything else still falls through to the generic label and style.
CAPACITY_RE = re.compile(r"^(?P<solver>.+?)__(?P<axis>num_components|bins)(?P<k>\d+)$")
CAPACITY_UNITS = {"num_components": "components", "bins": "bins"}
CI_LEVEL = 0.95

GAME_TITLES = {
    "kuhn_solvers": "Kuhn poker",
    "leduc_solvers": "Leduc poker",
    "sequential_blotto_solvers": "Sequential Blotto (3 fronts)",
    "sequential_blotto5_solvers": "Sequential Blotto (5 fronts)",
}

# Colour follows the configuration, never its rank in a filtered plot. The
# discrete-MMD bin counts are ordered, so they share one light->dark blue ramp;
# every other solver takes its own hue. Line style + marker are a second
# encoding so no pair relies on colour alone.
DISCRETE_MMD_BLUES = {1: "#c3dbf7", 2: "#a5c9f3", 4: "#86b6ef", 8: "#3987e5",
                      16: "#1c5cab", 32: "#0d366b", 64: "#06203f"}
# The mixture's component counts get the same treatment in self_play's own hue, so a
# capacity plot reads as two ramps rather than ten unrelated colours.
MIXTURE_ORANGES = {1: "#fbd5c2", 2: "#f6ab86", 4: "#ef7f4c", 8: "#eb6834",
                   16: "#b8471c", 32: "#7a2c05", 64: "#511d03"}
CAPACITY_COLORS = {"bins": DISCRETE_MMD_BLUES, "num_components": MIXTURE_ORANGES}
SOLVER_STYLES = {
    "self_play": ("#eb6834", "-", "o"),
    "sac": ("#1baf7a", "--", "s"),
    "nfsp": ("#e87ba4", "-.", "^"),
    "psro": ("#4a3aa7", ":", "D"),
}
FALLBACK_STYLE = ("#8a8984", "-", "x")
LEGEND_ORDER = ["self_play", "sac", "nfsp", "psro"]

TEXT = "#3d3d3a"
MUTED = "#73726c"
GRID = "#e4e3dd"

X_AXES = ("checkpoint", "wall_time", "samples")
SAMPLE_LABELS = {"episodes": "Environment interactions", "env_steps": "Environment interactions"}


def config_label(run_name: str, game: str) -> str:
    """``kuhn_solvers__discrete_mmd__bins16__seed3`` -> ``discrete_mmd__bins16``."""
    name = SEED_SUFFIX_RE.sub("", run_name)
    prefix = f"{game}__"
    return name[len(prefix):] if name.startswith(prefix) else name


# Legend names; the keys stay the run-directory names they are matched against.
DISPLAY_NAMES = {"self_play": "Mixture", "sac": "PPO", "nfsp": "NFSP", "psro": "PSRO",
                 "discrete_mmd": "Discrete MMD"}


def capacity_of(config: str) -> tuple[str, str, int] | None:
    """``(solver, axis, K)`` for a capacity run, or ``None`` for anything else."""
    match = CAPACITY_RE.match(config)
    if not match:
        return None
    return match["solver"], match["axis"], int(match["k"])


def pretty_label(config: str) -> str:
    capacity = capacity_of(config)
    if capacity:
        solver, axis, k = capacity
        return f"{DISPLAY_NAMES.get(solver, solver)} ({k} {CAPACITY_UNITS[axis]})"
    return DISPLAY_NAMES.get(config, config.replace("_", " "))


def style_for(config: str) -> tuple[str, str, str]:
    capacity = capacity_of(config)
    if capacity:
        _, axis, k = capacity
        ramp = CAPACITY_COLORS[axis]
        return ramp.get(k, ramp[max(ramp)]), "-", None
    return SOLVER_STYLES.get(config, FALLBACK_STYLE)


def sort_key(config: str) -> tuple:
    capacity = capacity_of(config)
    if capacity:
        solver, axis, k = capacity
        return (0, axis, k, solver)
    if config in LEGEND_ORDER:
        return (1, LEGEND_ORDER.index(config), 0, config)
    return (2, "", 0, config)


def load_results(out_root: Path, games: list[str] | None,
                 solvers: list[str] | None) -> list[dict]:
    results = []
    for path in sorted(out_root.glob(f"*/{RESULT_FILENAME}")):
        with path.open("rb") as f:
            result = pickle.load(f)
        game = str(result["game"])
        if games is not None and not any(g in game for g in games):
            continue
        config = config_label(str(result["run"]), game)
        if solvers is not None and not any(s in config for s in solvers):
            continue
        result["_config"] = config
        results.append(result)
    return results


def run_curve(result: dict, x_axis: str, samples: str,
              target: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One seed's (checkpoint, x, exploitability) with unusable points dropped."""
    steps = np.asarray(result["steps"], dtype=np.float64)
    if result["exact"]:
        y = np.asarray(result["expl"], dtype=np.float64)
        if target and "target_expl" in result:
            y = np.asarray(result["target_expl"], dtype=np.float64)
    else:
        y = np.asarray(result["expl_lb"], dtype=np.float64)

    if x_axis == "checkpoint":
        x = steps.copy()
    elif x_axis == "wall_time":
        x = np.asarray(result["wall_time"], dtype=np.float64) / 3600.0
    else:
        key = "episodes_train" if samples == "episodes" else "env_steps"
        x = np.asarray(result[key], dtype=np.float64)
    # Checkpoint 0 is written before any training, so it has no metrics row.
    x = np.where((steps == 0) & np.isnan(x), 0.0, x)

    keep = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(steps[keep], kind="stable")
    return steps[keep][order], x[keep][order], y[keep][order]


def pool_seeds(curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict | None:
    """Group seeds by checkpoint: mean x, mean exploitability and its t-based CI.

    Only checkpoints every seed has are kept, so each point averages all seeds.
    """
    curves = [c for c in curves if len(c[0]) >= 1]
    if not curves:
        return None
    common = curves[0][0]
    for steps, _, _ in curves[1:]:
        common = np.intersect1d(common, steps)
    if len(common) == 0:
        return None
    xs = np.stack([x[np.searchsorted(steps, common)] for steps, x, _ in curves])
    ys = np.stack([y[np.searchsorted(steps, common)] for steps, _, y in curves])
    n = len(curves)
    mean = ys.mean(axis=0)
    if n > 1:
        sem = ys.std(axis=0, ddof=1) / np.sqrt(n)
        half = stats.t.ppf(0.5 + CI_LEVEL / 2, df=n - 1) * sem
    else:
        half = np.zeros_like(mean)
    return {"x": xs.mean(axis=0), "mean": mean, "lo": mean - half,
            "hi": mean + half, "n": n}


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def x_label(x_axis: str, samples: str) -> str:
    if x_axis == "checkpoint":
        return "Checkpoint"
    if x_axis == "wall_time":
        return "Training wall-time [h]"
    return SAMPLE_LABELS[samples]


def y_label(exact: bool, target: bool) -> str:
    if exact:
        return "Exploitability" if target else "Exploitability"
    return "Approximate Exploitability"


def draw_panel(ax, by_config: dict[str, list[dict]], x_axis: str, *, samples: str,
               target: bool, linear_y: bool, show_seeds: bool) -> None:
    panels = []
    for config in sorted(by_config, key=sort_key):
        curves = [run_curve(r, x_axis, samples, target) for r in by_config[config]]
        pooled = pool_seeds(curves)
        if pooled is not None:
            panels.append((config, curves, pooled))

    # A log axis cannot draw a CI bound <= 0: matplotlib drops those polygon
    # vertices and bridges the survivors with long diagonal wedges. Clip the
    # lower bound to the axis floor instead, so it runs off the bottom.
    floor = None
    if not linear_y:
        positive = [p["mean"][p["mean"] > 0] for _, _, p in panels]
        positive = np.concatenate(positive) if positive else np.array([])
        if positive.size:
            floor = 0.5 * positive.min()

    for config, curves, pooled in panels:
        color, linestyle, marker = style_for(config)
        if show_seeds:
            for _, x, y in curves:
                ax.plot(x, y, color=color, linewidth=0.7, alpha=0.3, linestyle=linestyle)
        if pooled["n"] > 1:
            lo = pooled["lo"] if floor is None else np.maximum(pooled["lo"], floor)
            ax.fill_between(pooled["x"], lo, pooled["hi"],
                            color=color, alpha=0.15, linewidth=0)
        # Sparse markers keep the marker a secondary cue without cluttering.
        markevery = max(1, len(pooled["x"]) // 8)
        ax.plot(pooled["x"], pooled["mean"], color=color, linewidth=1.6,
                linestyle=linestyle, marker=marker, markersize=5,
                markevery=markevery, markeredgecolor="white", markeredgewidth=0.8,
                label=f"{pretty_label(config)}")
    if not linear_y:
        ax.set_yscale("log", nonpositive="mask")
        if floor is not None:
            ax.set_ylim(bottom=floor)
    if x_axis == "samples":
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_color(MUTED)
    style_axes(ax)


def capacity_points(by_config: dict[str, list[dict]], *, samples: str, target: bool,
                    which: str) -> dict[tuple[str, str], dict[int, np.ndarray]]:
    """`{(solver, axis): {K: exploitabilities, one per seed}}` for one game's capacity runs.

    `which` is "final" (the last checkpoint) or "best" (the lowest one the run reached).
    Runs that are not a capacity sweep are skipped -- a baseline has no K to sit at.
    """
    points: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for config, results in by_config.items():
        capacity = capacity_of(config)
        if capacity is None:
            continue
        solver, axis, k = capacity
        for result in results:
            _, _, y = run_curve(result, "checkpoint", samples, target)
            if len(y) == 0:
                continue
            points[(solver, axis)][k].append(float(y[-1] if which == "final" else y.min()))
    return {key: {k: np.asarray(v) for k, v in sorted(ks.items())}
            for key, ks in points.items()}


def draw_capacity_panel(ax, by_config: dict[str, list[dict]], *, samples: str, target: bool,
                        linear_y: bool) -> bool:
    """Exploitability against representation size, one line per swept axis.

    Every point is one (solver, K) setting at the *same* training budget, so the line
    reads as what K buys rather than as what a longer run buys. Solid is the final
    checkpoint, dashed the best one reached: where they separate, the run found a better
    strategy than the one it ended on, which is a training story and not a capacity one.
    Seeds are pooled exactly as `pool_seeds` does, with a t-based CI.
    """
    from matplotlib import ticker      # local, like main()'s: this module imports no backend

    lines = []
    for which, linestyle, alpha, suffix in (("final", "-", 1.0, ""),
                                            ("best", "--", 0.55, " (best checkpoint)")):
        points = capacity_points(by_config, samples=samples, target=target, which=which)
        for (solver, axis), ks in sorted(points.items()):
            xs = sorted(ks)
            means = np.array([ks[k].mean() for k in xs])
            halves = np.array([
                stats.t.ppf(0.5 + CI_LEVEL / 2, df=len(ks[k]) - 1)
                * ks[k].std(ddof=1) / np.sqrt(len(ks[k])) if len(ks[k]) > 1 else 0.0
                for k in xs])
            name = DISPLAY_NAMES.get(solver, solver)
            lines.append((which, linestyle, alpha, axis, xs, means, halves,
                          f"{name} ({CAPACITY_UNITS[axis]}){suffix}"))
    if not lines:
        return False

    # Same trick as `draw_panel`: an RL lower bound can be <= 0, which a log axis cannot
    # draw, so clip the band to the axis floor rather than letting matplotlib bridge the
    # dropped vertices.
    floor = None
    if not linear_y:
        positive = np.concatenate([m[m > 0] for *_, m, _, _ in lines] or [np.array([])])
        if positive.size:
            floor = 0.5 * positive.min()

    for which, linestyle, alpha, axis, xs, means, halves, label in lines:
        # The darkest shade of the family's ramp, so the summary line reads as the same
        # colour family the per-K training curves are drawn in.
        ramp = CAPACITY_COLORS[axis]
        color = ramp[max(ramp)]
        if which == "final":
            lo = means - halves if floor is None else np.maximum(means - halves, floor)
            ax.fill_between(xs, lo, means + halves, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, means, color=color, linewidth=1.6, linestyle=linestyle, marker="o",
                markersize=5, alpha=alpha, markeredgecolor="white",
                markeredgewidth=0.8, label=label)

    ax.set_xscale("log", base=2)
    # Tick at the K's that were actually run, written as plain numbers: a log2 axis
    # otherwise labels powers of two it has no point for.
    ax.set_xticks(sorted({k for *_, xs, _, _, _ in lines for k in xs}))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Representation size (mixture components / grid bins)", color=TEXT)
    if not linear_y:
        ax.set_yscale("log", nonpositive="mask")
        if floor is not None:
            ax.set_ylim(bottom=floor)
    style_axes(ax)
    return True


def group_by_game(results: list[dict]) -> dict[str, dict[str, list[dict]]]:
    games: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        games[str(result["game"])][result["_config"]].append(result)
    return games


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="root of scored runs "
                         f"(default: {DEFAULT_OUT.relative_to(REPO_ROOT)})")
    ap.add_argument("--plots", type=Path, default=None,
                    help="directory for the PNGs (default: {out}/plots)")
    ap.add_argument("--games", nargs="+", default=None,
                    help="only games whose name contains one of these (e.g. kuhn leduc)")
    ap.add_argument("--solvers", nargs="+", default=None,
                    help="only configs containing one of these (e.g. discrete_mmd psro)")
    ap.add_argument("--samples", choices=sorted(SAMPLE_LABELS), default="episodes",
                    help="x-axis of the samples plot (default: episodes)")
    ap.add_argument("--target", action="store_true",
                    help="on Kuhn, plot the Polyak-target exploitability where scored")
    ap.add_argument("--linear-y", action="store_true",
                    help="linear y-axis (shows lower bounds <= 0; default is log)")
    ap.add_argument("--show-seeds", action="store_true",
                    help="also draw each seed as a faint line")
    ap.add_argument("--capacity", action="store_true",
                    help="also write {game}__capacity: final (and best) exploitability "
                         "against the number of mixture components / bins, for the runs "
                         "of a `--experiment capacity` sweep")
    ap.add_argument("--format", default="png", help="image format (png, pdf, svg)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_root = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    plot_dir = args.plots or out_root / "plots"
    plot_dir = plot_dir if plot_dir.is_absolute() else REPO_ROOT / plot_dir
    if not out_root.is_dir():
        raise SystemExit(f"no sweep root at {out_root}")

    results = load_results(out_root, args.games, args.solvers)
    if not results:
        raise SystemExit(f"no {RESULT_FILENAME} under {out_root} matching the filters; "
                         "run score_sequential_sweep.py first")
    games = group_by_game(results)
    plot_dir.mkdir(parents=True, exist_ok=True)
    panel_kwargs = dict(samples=args.samples, target=args.target,
                        linear_y=args.linear_y, show_seeds=args.show_seeds)
    game_names = sorted(games)

    for game in game_names:
        by_config = games[game]
        exact = any(r["exact"] for runs in by_config.values() for r in runs)
        title = GAME_TITLES.get(game, game)
        for x_axis in X_AXES:
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            draw_panel(ax, by_config, x_axis, **panel_kwargs)
            ax.set_xlabel(x_label(x_axis, args.samples), color=TEXT)
            ax.set_ylabel(y_label(exact, args.target), color=TEXT)
            # ax.set_title(f"{title}: mean over seeds, band = 95% CI",
            #              color=TEXT, fontsize=11, loc="left")
            ax.legend(fontsize=8, frameon=False, loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), labelcolor=TEXT)
            fig.tight_layout()
            path = plot_dir / f"{game}__{x_axis}.{args.format}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
        for ax, x_axis in zip(axes, ("samples", "wall_time")):
            draw_panel(ax, by_config, x_axis, **panel_kwargs)
            ax.set_xlabel(x_label(x_axis, args.samples), color=TEXT)
        axes[0].set_ylabel(y_label(exact, args.target), color=TEXT)
        axes[-1].legend(fontsize=8, frameon=False, loc="upper left",
                        bbox_to_anchor=(1.01, 1.0), labelcolor=TEXT)
        fig.tight_layout()
        path = plot_dir / f"{game}__combined.{args.format}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

        if args.capacity:
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            if draw_capacity_panel(ax, by_config, samples=args.samples, target=args.target,
                                   linear_y=args.linear_y):
                ax.set_ylabel(y_label(exact, args.target), color=TEXT)
                ax.legend(fontsize=8, frameon=False, loc="upper left",
                          bbox_to_anchor=(1.01, 1.0), labelcolor=TEXT)
                fig.tight_layout()
                path = plot_dir / f"{game}__capacity.{args.format}"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"wrote {path}")
            else:
                print(f"no capacity runs for {game}; skipped {game}__capacity")
            plt.close(fig)

    fig, axes = plt.subplots(len(game_names), len(X_AXES),
                             figsize=(5.0 * len(X_AXES), 3.6 * len(game_names)),
                             squeeze=False, sharey="row")
    for row, game in enumerate(game_names):
        by_config = games[game]
        exact = any(r["exact"] for runs in by_config.values() for r in runs)
        for col, x_axis in enumerate(X_AXES):
            ax = axes[row][col]
            draw_panel(ax, by_config, x_axis, **panel_kwargs)
            ax.set_xlabel(x_label(x_axis, args.samples), color=TEXT, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{GAME_TITLES.get(game, game)}\n"
                              f"{y_label(exact, args.target)}", color=TEXT, fontsize=9)
        axes[row][-1].legend(fontsize=7, frameon=False, loc="upper left",
                             bbox_to_anchor=(1.01, 1.0), labelcolor=TEXT)
    fig.tight_layout()
    path = plot_dir / f"overview.{args.format}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
