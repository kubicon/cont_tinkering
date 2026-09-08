"""Score every checkpoint of every run in a `run_all.py` output tree, after the fact.

    python experiments/one_shot_neural/score.py --out data/one_shot_neural
    python experiments/one_shot_neural/score.py --out data/one_shot_neural --grid 1601 --overwrite
    python experiments/one_shot_neural/score.py --out data/one_shot_neural --game two_point --plot
    python experiments/one_shot_neural/score.py --out data/one_shot_neural --game two_point --plot-only
    python experiments/one_shot_neural/score.py --out data/one_shot_neural --plot   # every game

Training and measurement are deliberately separated. The runs write strategies, wall-time
and payoff-evaluation counts; this computes exploitability for all of them with **one**
oracle on **one** grid, so every method, every game and every seed is measured by the same
instrument -- and so the measurement never lands inside the wall-time being compared.

It also means the metric can be changed without retraining: rerun with a finer `--grid`
and every curve in the tree is recomputed from the same checkpoints.

Writes `scores.json` next to each run's checkpoints, and at the root a `curves.json`
(everything, for plotting) and a `summary.md` (one row per cell). Each scored point
carries `expl` alongside the run's own `wall_time` and `payoff_evals`, which is what makes
the two comparison plots -- exploitability against environment cost, and against wall
clock -- come from the same data.

Where a checkpoint stored a second iterate (the mixture method's Polyak average, the
discretized policy's averaged weights), that is scored too and reported as `target_expl`:
for those methods the averaged iterate is the one their theory is about.

`--plot` / `--plot-only` draw those two axes for one `--game`, or, when `--game` is
omitted, for every game present in the tree: mean exploitability across seeds with a
95% CI band, one curve per method. Budget on the primary plot is `payoff_evals`
(the shared cost unit), not iteration count.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from baselines.common import GridOracle, StrategyPair, effective_support, load_game  # noqa: E402


def find_runs(out_root: Path) -> list[Path]:
    return sorted(p.parent for p in out_root.glob("*/*/*/meta.json"))


def _target_strategy(pair: StrategyPair):
    """The second iterate a checkpoint may carry, in `(s0, w0, s1, w1)` form, or `None`.

    Two shapes occur, because two kinds of policy store an average: a *support* pair
    (sampled actions from the Polyak-averaged mixture) and a *weights* pair (the averaged
    probabilities of a discretized policy, which lives on the same grid as the live one).
    """
    extra = pair.extra
    if "target_support_0" in extra and "target_support_1" in extra:
        support_0, support_1 = extra["target_support_0"], extra["target_support_1"]
        return (support_0, np.full(support_0.shape[0], 1.0 / support_0.shape[0]),
                support_1, np.full(support_1.shape[0], 1.0 / support_1.shape[0]))
    if "target_weights_0" in extra and "target_weights_1" in extra:
        return (pair.support_0, extra["target_weights_0"],
                pair.support_1, extra["target_weights_1"])
    return None


def score_run(directory: Path, oracle: GridOracle, overwrite: bool = False) -> list[dict]:
    """Score one run's checkpoints, merging in the cost columns from its history."""
    scores_path = directory / "scores.json"
    if scores_path.exists() and not overwrite:
        return json.loads(scores_path.read_text())["scores"]

    meta = json.loads((directory / "meta.json").read_text())
    history_path = directory / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    by_t = {int(entry["t"]): entry for entry in history}
    radius = oracle.cluster_radius()

    scores = []
    for path in sorted((directory / "checkpoints").glob("*.npz")):
        pair = StrategyPair.load(path)
        entry = by_t.get(int(pair.t), {})
        row = {
            "t": int(pair.t),
            # The cost columns come from the run, not from here: they were measured
            # while training, with this scoring deliberately outside the timer.
            "wall_time": entry.get("wall_time"),
            "compile_time": entry.get("compile_time"),
            "payoff_evals": entry.get("payoff_evals"),
            "expl": float(oracle.exploitability(pair.support_0, pair.weights_0,
                                                pair.support_1, pair.weights_1)),
            "value": float(oracle.value(pair.support_0, pair.weights_0,
                                        pair.support_1, pair.weights_1)),
            "support_0": effective_support(pair.weights_0, support=pair.support_0, radius=radius),
            "support_1": effective_support(pair.weights_1, support=pair.support_1, radius=radius),
        }
        target = _target_strategy(pair)
        if target is not None:
            row["target_expl"] = float(oracle.exploitability(*target))
        scores.append(row)

    scores_path.write_text(json.dumps(
        {"meta": {**{k: meta.get(k) for k in ("game", "method", "seed", "budget",
                                              "access_model", "plan")},
                  "grid": oracle.points, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
         "scores": scores}, indent=2))
    return scores


def summary_row(meta: dict, scores: list[dict]) -> dict:
    expls = [s["expl"] for s in scores]
    final = scores[-1] if scores else {}
    return {
        "game": meta.get("game"), "method": meta.get("method"), "seed": meta.get("seed"),
        "access_model": meta.get("access_model"),
        "final_expl": expls[-1] if expls else float("nan"),
        "best_expl": min(expls) if expls else float("nan"),
        "target_expl": final.get("target_expl"),
        "payoff_evals": final.get("payoff_evals"),
        "train_seconds": meta.get("train_seconds"),
        "compile_seconds": meta.get("compile_seconds"),
        "checkpoints": len(scores),
    }


def summary_table(rows: list[dict]) -> str:
    header = ("|game|method|seed|final expl|best expl|avg-iterate expl|payoff evals|train s|compile s|access|\n"
              "|-|-|-|-|-|-|-|-|-|-|\n")
    def cell(value, fmt="{:+.5f}"):
        return "-" if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(value)
    lines = []
    for row in sorted(rows, key=lambda r: (r["game"] or "", r["method"] or "", r["seed"] or 0)):
        lines.append(
            f"|{row['game']}|{row['method']}|{row['seed']}|{cell(row['final_expl'])}|"
            f"{cell(row['best_expl'])}|{cell(row['target_expl'])}|"
            f"{cell(row['payoff_evals'], '{:.3g}')}|{cell(row['train_seconds'], '{:.0f}')}|"
            f"{cell(row['compile_seconds'], '{:.0f}')}|{row['access_model']}|")
    return header + "\n".join(lines)


def _xy(scores: list[dict], x_key: str, y_key: str = "expl") -> tuple[np.ndarray, np.ndarray] | None:
    """Extract a monotone (x, y) series, dropping points that lack either coordinate."""
    xs, ys = [], []
    for row in scores:
        x, y = row.get(x_key), row.get(y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def _mean_ci95(series: list[tuple[np.ndarray, np.ndarray]],
               n_grid: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate seed curves onto a shared x-grid; return x, mean, lo, hi (95% CI).

    With one seed the band collapses to the mean. With n>=2 the interval is the
    normal approx mean ± 1.96 * sem across seeds at each grid point.
    """
    if not series:
        return None
    x_lo = max(s[0][0] for s in series)
    x_hi = min(s[0][-1] for s in series)
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        return None
    grid = np.linspace(x_lo, x_hi, n_grid)
    stacked = np.vstack([np.interp(grid, x, y) for x, y in series])
    mean = stacked.mean(axis=0)
    n = stacked.shape[0]
    if n < 2:
        return grid, mean, mean, mean
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(n)
    half = 1.96 * sem
    return grid, mean, mean - half, mean + half


def plot_game(curves: dict, game: str, out_path: Path,
              methods: list[str] | None = None) -> Path:
    """Two panels for one game: exploitability vs payoff budget, and vs wall-time.

    Seeds are summarized as mean with a 95% CI band. Always plots the live strategy
    (`expl`), not the Polyak-averaged `target_expl` some methods also store.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_method: dict[str, list[dict]] = defaultdict(list)
    for entry in curves.values():
        meta = entry["meta"]
        if meta.get("game") != game:
            continue
        method = meta.get("method")
        if methods and method not in methods:
            continue
        by_method[method].append(entry)

    if not by_method:
        raise SystemExit(f"no scored runs for game {game!r} in curves")

    method_names = sorted(by_method)
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(method_names)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    panels = (
        (axes[0], "payoff_evals", "payoff evaluations (budget)"),
        (axes[1], "wall_time", "wall-time (s)"),
    )

    for ax, x_key, xlabel in panels:
        for method in method_names:
            series = []
            for entry in by_method[method]:
                pair = _xy(entry["scores"], x_key, "expl")
                if pair is not None:
                    series.append(pair)
            band = _mean_ci95(series)
            if band is None:
                continue
            x, mean, lo, hi = band
            color = colors[method]
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, mean, color=color, label=method, linewidth=1.8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("exploitability")
        ax.set_title(f"{game}: expl vs {x_key if x_key != 'payoff_evals' else 'budget'}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_games(curves: dict, out_root: Path, game: str | None = None,
               methods: list[str] | None = None) -> list[Path]:
    """Plot `game`, or every game present in `curves` when `game` is None."""
    if game:
        games = [game]
    else:
        games = sorted({entry["meta"].get("game") for entry in curves.values()
                        if entry["meta"].get("game")})
        if not games:
            raise SystemExit("no scored runs to plot")
    return [plot_game(curves, tag, out_root / f"{tag}_curves.png", methods=methods)
            for tag in games]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/one_shot_neural", help="the run_all.py output tree")
    ap.add_argument("--grid", type=int, default=801,
                    help="deviation grid for the metric; finer than the runs' own is fine "
                         "and costs only this pass")
    ap.add_argument("--game", default=None,
                    help="game tag to plot; with --plot / --plot-only and no --game, "
                         "every game in the output tree is plotted")
    ap.add_argument("--games", nargs="+", default=None, help="score only these game tags")
    ap.add_argument("--methods", nargs="+", default=None, help="score / plot only these methods")
    ap.add_argument("--overwrite", action="store_true", help="rescore runs that have scores.json")
    ap.add_argument("--plot", action="store_true",
                    help="after scoring, plot exploitability curves for --game")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip scoring; plot --game from an existing curves.json")
    args = ap.parse_args()

    out_root = Path(args.out)

    if args.plot_only:
        curves_path = out_root / "curves.json"
        if not curves_path.exists():
            raise SystemExit(f"no {curves_path}; run scoring first or drop --plot-only")
        curves = json.loads(curves_path.read_text())
        for path in plot_games(curves, out_root, args.game, methods=args.methods):
            print(f"plot -> {path}")
        return

    runs = find_runs(out_root)
    if not runs:
        raise SystemExit(f"no runs under {out_root} (looked for */*/*/meta.json)")

    oracles: dict[str, GridOracle] = {}
    rows, curves = [], {}
    started = time.monotonic()
    for directory in runs:
        meta = json.loads((directory / "meta.json").read_text())
        if args.games and meta.get("game") not in args.games:
            continue
        if args.methods and meta.get("method") not in args.methods:
            continue
        if meta.get("status") == "failed":
            print(f"  skip (failed run): {directory}")
            continue

        config = meta["config"]
        if config not in oracles:
            game, _ = load_game(config)
            oracles[config] = GridOracle(game, points=args.grid)
        scores = score_run(directory, oracles[config], args.overwrite)
        rows.append(summary_row(meta, scores))
        curves[str(directory.relative_to(out_root))] = {
            "meta": {k: meta.get(k) for k in ("game", "method", "seed", "budget",
                                              "access_model", "train_seconds")},
            "scores": scores,
        }
        last = scores[-1] if scores else {}
        print(f"  {meta['game']:>22s} / {meta['method']:<13s} seed{meta['seed']}  "
              f"expl {last.get('expl', float('nan')):+.5f}  "
              f"{len(scores)} checkpoints")

    (out_root / "curves.json").write_text(json.dumps(curves, indent=2))
    table = summary_table(rows)
    (out_root / "summary.md").write_text(table + "\n")
    print(f"\n{table}")
    print(f"\nscored {len(rows)} runs in {time.monotonic() - started:.1f}s on a "
          f"{args.grid}-point grid")
    print(f"curves -> {out_root / 'curves.json'}   summary -> {out_root / 'summary.md'}")

    if args.plot:
        for path in plot_games(curves, out_root, args.game, methods=args.methods):
            print(f"plot -> {path}")


if __name__ == "__main__":
    main()
