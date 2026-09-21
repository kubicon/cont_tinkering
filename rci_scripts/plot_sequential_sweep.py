"""Plot exploitability over training for a scored sequential sweep.

Reads every ``{run}/exploitability.pkl`` that ``score_sequential_sweep.py``
wrote under ``data/sequential_sweep/`` (or ``--out``) and draws, per game, one
line per solver configuration (seeds pooled) against three x-axes:

    checkpoint   the checkpoint index (solver-specific unit: a PPO chunk for
                 self_play / sac, a round for nfsp / psro --
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
    {game}__combined.png   one figure, two panels sharing a y-axis: environment
                           interactions (left) and wall-time (right)
    {game}__capacity.png   --capacity only: exploitability against representation size
    overview.png           rows = games, columns = the three x-axes

``--capacity`` is the plot the capacity sweep exists for (see
``generate_sequential_sweep.py --experiment capacity``): the x-axis is the number of
mixture components (``self_play``) or grid bins (``discrete_mmd``) rather than training
progress, one point per setting at the same budget. Runs whose name carries no such
count are left out of it, so pointing ``--capacity`` at the main sweep's tree simply
writes nothing. It draws two lines per solver -- the final checkpoint solid, each run's
best checkpoint dashed -- and ``--capacity-which final`` / ``best`` keeps just one of
them, drawn solid and labelled with the solver name alone.

    python rci_scripts/plot_sequential_sweep.py
    python rci_scripts/plot_sequential_sweep.py --games kuhn --target
    python rci_scripts/plot_sequential_sweep.py --samples env_steps --show-seeds
    python rci_scripts/plot_sequential_sweep.py --out data/sequential_capacity --capacity
    python rci_scripts/plot_sequential_sweep.py --out data/sequential_capacity \
        --capacity --capacity-which best
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_style  # noqa: E402
from plot_style import (  # noqa: E402
    BAND_ALPHA,
    SMALL_AXIS_LABEL_SIZE,
    SMALL_LEGEND_SIZE,
    CI_LEVEL,
    LINEWIDTH,
    MARKERSIZE,
    MUTED,
    SECONDARY_BAND_ALPHA,
    SEED_LINEWIDTH,
    display_name,
    style_axes,
    style_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "sequential_sweep"
RESULT_FILENAME = "exploitability.pkl"
SEED_SUFFIX_RE = re.compile(r"__seed\d+$")
# A capacity run: one solver at one representation size, as
# `generate_sequential_sweep.py --experiment capacity` names it -- `self_play__num_components8`
# or a `__bins16` sweep. The axis name is the config key that was swept, so a
# sweep of anything else still falls through to the generic label and style.
CAPACITY_RE = re.compile(r"^(?P<solver>.+?)__(?P<axis>num_components|bins)(?P<k>\d+)$")
CAPACITY_UNITS = {"num_components": "components", "bins": "bins"}

GAME_TITLES = {
    "kuhn_solvers": "Kuhn poker",
    "leduc_solvers": "Leduc poker",
    "sequential_blotto_solvers": "Sequential Blotto (3 fronts)",
    "sequential_blotto5_solvers": "Sequential Blotto (5 fronts)",
}

# Colour, dash pattern, marker, legend name and which solvers are left out of the
# baseline comparison all come from `plot_style` at the repo root, which the one-shot
# experiment's plots read too -- so a method looks the same, and is shown or hidden the
# same way, in both. Add a solver there, not here.

# This experiment's runs are hours long, so the wall-clock axis is in hours. Type
# sizes and the unit's label come from `plot_style` -- turn FONT_SCALE up there to
# enlarge every label in both experiments at once.
WALL_TIME_UNIT = "hours"
WALL_TIME_LABEL, WALL_TIME_PER_SECOND = plot_style.wall_time_axis(WALL_TIME_UNIT)

X_AXES = ("checkpoint", "wall_time", "samples")
# The two panels of `{game}__combined`, left to right.
COMBINED_AXES = ("samples", "wall_time")
SAMPLE_LABELS = {"episodes": plot_style.SAMPLES_LABEL, "env_steps": plot_style.SAMPLES_LABEL}


def config_label(run_name: str, game: str) -> str:
    """``kuhn_solvers__self_play__num_components8__seed3`` -> ``self_play__num_components8``."""
    name = SEED_SUFFIX_RE.sub("", run_name)
    prefix = f"{game}__"
    return name[len(prefix):] if name.startswith(prefix) else name


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
        return f"{display_name(solver)} ({k} {CAPACITY_UNITS[axis]})"
    return display_name(config)


def style_for(config: str) -> tuple[str, str, str | None]:
    """``(colour, linestyle, marker)`` for one config's training curve."""
    capacity = capacity_of(config)
    if capacity:
        solver, _, k = capacity
        # No marker on a capacity cell: ten curves of one method are told apart by
        # their shade, and ten marker shapes would only add noise.
        return plot_style.capacity_color(solver, k), style_of(solver)["linestyle"], None
    style = style_of(config)
    return style["color"], style["linestyle"], style["marker"]


def sort_key(config: str) -> tuple:
    capacity = capacity_of(config)
    if capacity:
        solver, _, k = capacity
        return plot_style.sort_key(solver, k)
    return plot_style.sort_key(config)


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
        capacity = capacity_of(config)
        if plot_style.excluded_baseline(config if capacity is None else capacity[0],
                                        None if capacity is None else capacity[2]):
            continue
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
        x = np.asarray(result["wall_time"], dtype=np.float64) * WALL_TIME_PER_SECOND
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


def x_label(x_axis: str, samples: str) -> str:
    if x_axis == "checkpoint":
        return "Checkpoint"
    if x_axis == "wall_time":
        return WALL_TIME_LABEL
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
                ax.plot(x, y, color=color, linewidth=SEED_LINEWIDTH, alpha=0.3,
                        linestyle=linestyle)
        if pooled["n"] > 1:
            lo = pooled["lo"] if floor is None else np.maximum(pooled["lo"], floor)
            ax.fill_between(pooled["x"], lo, pooled["hi"],
                            color=color, alpha=BAND_ALPHA, linewidth=0)
        # Sparse markers keep the marker a secondary cue without cluttering.
        ax.plot(pooled["x"], pooled["mean"], color=color, linewidth=LINEWIDTH,
                linestyle=linestyle, marker=marker, markersize=MARKERSIZE,
                markevery=plot_style.markevery(len(pooled["x"])),
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"{pretty_label(config)}")
    if not linear_y:
        ax.set_yscale("log", nonpositive="mask")
        if floor is not None:
            ax.set_ylim(bottom=floor)
    if x_axis == "samples":
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_color(MUTED)
        ax.xaxis.get_offset_text().set_fontsize(plot_style.TICK_LABEL_SIZE)
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


# Which checkpoint the capacity plot reads off, and how that series is drawn:
# (linestyle, line alpha, legend suffix). Drawn on its own, a series is the primary one
# -- solid, opaque, and named after the method alone, because there is no second line
# for the legend to tell it apart from.
CAPACITY_SERIES = {"final": ("-", 1.0, ""), "best": ("--", 0.55, " (best checkpoint)")}
CAPACITY_SOLO = ("-", 1.0, "")
CAPACITY_WHICH = ("both", "final", "best")


def capacity_series(which: str) -> list[tuple[str, str, float, str]]:
    """`[(key, linestyle, alpha, suffix)]` for one `--capacity-which` setting."""
    if which == "both":
        return [(key, *CAPACITY_SERIES[key]) for key in CAPACITY_SERIES]
    return [(which, *CAPACITY_SOLO)]


def draw_capacity_panel(ax, by_config: dict[str, list[dict]], *, samples: str, target: bool,
                        linear_y: bool, which: str = "both") -> bool:
    """Exploitability against representation size, one line per swept axis.

    Every point is one (solver, K) setting at the *same* training budget, so the line
    reads as what K buys rather than as what a longer run buys. Solid is the final
    checkpoint, dashed the best one reached: where they separate, the run found a better
    strategy than the one it ended on, which is a training story and not a capacity one.

    Both lines are pooled the same way, exactly as `pool_seeds` does: the *best* is each
    seed's own best checkpoint, so the dashed line is the mean over seeds of a per-run
    minimum -- not one lucky run's -- and it carries a t-based CI of its own, drawn in a
    fainter band than the final line's.

    `which` is "both", or "final" / "best" for that series alone -- which is then drawn
    as the primary line and labelled with the method name only.
    """
    from matplotlib import ticker      # local, like main()'s: this module imports no backend

    lines = []
    for key, linestyle, alpha, suffix in capacity_series(which):
        points = capacity_points(by_config, samples=samples, target=target, which=key)
        for (solver, axis), ks in sorted(points.items()):
            xs = sorted(ks)
            means = np.array([ks[k].mean() for k in xs])
            halves = np.array([
                stats.t.ppf(0.5 + CI_LEVEL / 2, df=len(ks[k]) - 1)
                * ks[k].std(ddof=1) / np.sqrt(len(ks[k])) if len(ks[k]) > 1 else 0.0
                for k in xs])
            lines.append((key, linestyle, alpha, solver, xs, means, halves,
                          f"{plot_style.short_name(solver)} "
                          f"({CAPACITY_UNITS[axis]}){suffix}"))
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

    for key, linestyle, alpha, solver, xs, means, halves, label in lines:
        # The solver's own colour and marker, exactly as the training-curve plots draw
        # it, so a baseline is recognisable across both experiments. Linestyle is spent
        # here on final vs best, so it is the one cue the two plots do not share.
        style = style_of(solver)
        color, marker = style["color"], style["marker"]
        lo = means - halves if floor is None else np.maximum(means - halves, floor)
        ax.fill_between(xs, lo, means + halves, color=color, linewidth=0,
                        alpha=BAND_ALPHA if alpha == 1.0 else SECONDARY_BAND_ALPHA)
        ax.plot(xs, means, color=color, linewidth=LINEWIDTH, linestyle=linestyle,
                marker=marker, markersize=MARKERSIZE, alpha=alpha,
                markeredgecolor="white", markeredgewidth=0.8, label=label)

    ax.set_xscale("log", base=2)
    # Tick at the K's that were actually run, written as plain numbers: a log2 axis
    # otherwise labels powers of two it has no point for.
    ax.set_xticks(sorted({k for *_, xs, _, _, _ in lines for k in xs}))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    plot_style.label_axes(ax, "Representation size (mixture components / grid bins)")
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
                    help="only configs containing one of these (e.g. self_play psro)")
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
    ap.add_argument("--capacity-which", choices=CAPACITY_WHICH, default="both",
                    dest="capacity_which",
                    help="which checkpoint the --capacity plot reads off: 'both' (final "
                         "solid, best dashed), or 'final' / 'best' alone -- a single "
                         "series is drawn solid and its legend drops the qualifier")
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
            plot_style.label_axes(ax, x_label(x_axis, args.samples),
                                  y_label(exact, args.target))
            # ax.set_title(f"{title}: mean over seeds, band = 95% CI",
            #              color=plot_style.TEXT, fontsize=plot_style.TITLE_SIZE,
            #              loc="left")
            plot_style.legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
            fig.tight_layout()
            path = plot_dir / f"{game}__{x_axis}.{args.format}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")

        # The headline figure, laid out like `experiments/one_shot_neural/score.py`:
        # one PNG, environment interactions on the left, wall-time on the right, a
        # shared y-axis, and the legend inside each panel rather than beside the figure.
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
        for ax, x_axis in zip(axes, COMBINED_AXES):
            draw_panel(ax, by_config, x_axis, **panel_kwargs)
            plot_style.label_axes(ax, x_label(x_axis, args.samples),
                                  y_label(exact, args.target))
            plot_style.legend(ax)
        fig.tight_layout()
        path = plot_dir / f"{game}__combined.{args.format}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

        if args.capacity:
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            if draw_capacity_panel(ax, by_config, samples=args.samples, target=args.target,
                                   linear_y=args.linear_y, which=args.capacity_which):
                plot_style.label_axes(ax, ylabel=y_label(exact, args.target))
                plot_style.legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
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
            plot_style.label_axes(ax, x_label(x_axis, args.samples),
                                  fontsize=SMALL_AXIS_LABEL_SIZE)
            if col == 0:
                plot_style.label_axes(
                    ax, ylabel=f"{GAME_TITLES.get(game, game)}\n"
                               f"{y_label(exact, args.target)}",
                    fontsize=SMALL_AXIS_LABEL_SIZE)
        plot_style.legend(axes[row][-1], fontsize=SMALL_LEGEND_SIZE, loc="upper left",
                          bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    path = plot_dir / f"overview.{args.format}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
