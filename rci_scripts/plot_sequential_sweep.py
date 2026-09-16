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
    overview.png           rows = games, columns = the three x-axes

    python rci_scripts/plot_sequential_sweep.py
    python rci_scripts/plot_sequential_sweep.py --games kuhn --target
    python rci_scripts/plot_sequential_sweep.py --samples env_steps --show-seeds
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
BINS_RE = re.compile(r"^discrete_mmd__bins(?P<bins>\d+)$")
CI_LEVEL = 0.95

GAME_TITLES = {
    "kuhn_solvers": "Kuhn poker",
    "leduc_solvers": "Leduc poker",
    "sequential_blotto_solvers": "Sequential Blotto",
}

# Colour follows the configuration, never its rank in a filtered plot. The
# discrete-MMD bin counts are ordered, so they share one light->dark blue ramp;
# every other solver takes its own hue. Line style + marker are a second
# encoding so no pair relies on colour alone.
DISCRETE_MMD_BLUES = {4: "#86b6ef", 8: "#3987e5", 16: "#1c5cab", 32: "#0d366b"}
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
SAMPLE_LABELS = {"episodes": "hands played", "env_steps": "decision nodes visited"}


def config_label(run_name: str, game: str) -> str:
    """``kuhn_solvers__discrete_mmd__bins16__seed3`` -> ``discrete_mmd__bins16``."""
    name = SEED_SUFFIX_RE.sub("", run_name)
    prefix = f"{game}__"
    return name[len(prefix):] if name.startswith(prefix) else name


def pretty_label(config: str) -> str:
    match = BINS_RE.match(config)
    if match:
        return f"discrete MMD ({match.group('bins')} bins)"
    return config.replace("_", " ")


def style_for(config: str) -> tuple[str, str, str]:
    match = BINS_RE.match(config)
    if match:
        return DISCRETE_MMD_BLUES.get(int(match.group("bins")), "#2a78d6"), "-", None
    return SOLVER_STYLES.get(config, FALLBACK_STYLE)


def sort_key(config: str) -> tuple:
    match = BINS_RE.match(config)
    if match:
        return (0, int(match.group("bins")), config)
    if config in LEGEND_ORDER:
        return (1, LEGEND_ORDER.index(config), config)
    return (2, 0, config)


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
        return "checkpoint (chunk / round)"
    if x_axis == "wall_time":
        return "training wall-time [h]"
    return SAMPLE_LABELS[samples]


def y_label(exact: bool, target: bool) -> str:
    if exact:
        return "exploitability (Polyak target)" if target else "exploitability"
    return "exploitability lower bound (Polyak target where saved)"


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
                label=f"{pretty_label(config)}  (n={pooled['n']})")
    if not linear_y:
        ax.set_yscale("log", nonpositive="mask")
        if floor is not None:
            ax.set_ylim(bottom=floor)
    if x_axis == "samples":
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_color(MUTED)
    style_axes(ax)


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
            ax.set_title(f"{title}: mean over seeds, band = 95% CI",
                         color=TEXT, fontsize=11, loc="left")
            ax.legend(fontsize=8, frameon=False, loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), labelcolor=TEXT)
            fig.tight_layout()
            path = plot_dir / f"{game}__{x_axis}.{args.format}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {path}")

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
