"""Score ``train.py`` mixture checkpoints with ``GridOracle``, then plot.

For one ``--game`` (required unless ``--plot-only``), finds every matching run
under ``data/components_sweep/`` (``{game}__nc*__seed*``), scores each
``{step}.pkl`` into ``exploitability.pkl``, then writes one figure for that
game under ``data/components_sweep/plots/{game}.png``: for each
``num_components``, mean exploitability across seeds with a 95% CI band.

The pickle is a dict with numpy arrays ready for later analysis:

    {
      "run", "config", "game", "num_components", "train_seed",
      "grid", "samples", "seed", "scored_at",
      "steps",        # (T,) int checkpoint indices
      "expl_live",    # (T,) float GridOracle exploitability of the live iterate
      "expl_target",  # (T,) float same for the Polyak-averaged iterate
      "value",        # (T,) float E[u] under the live sampled strategies
      "support_0",    # (T,) int effective support size, player 0
      "support_1",    # (T,) int effective support size, player 1
    }

    python rci_scripts/score_components_sweep.py --game circle
    python rci_scripts/score_components_sweep.py --game all_pay_auction --overwrite
    python rci_scripts/score_components_sweep.py --game circle --plot-only
    python rci_scripts/score_components_sweep.py --grid 1601 --samples 4096 --game glicksberg_gross
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from baselines.common import GridOracle, effective_support  # noqa: E402
from baselines.neural.common import empirical_strategy  # noqa: E402
from training.checkpoint import load_checkpoint_step_multi, target_entry  # noqa: E402
from training.config import MixturePPOHyperparams  # noqa: E402
from training.mixture import build_mixture_network, sample_mixture_actions  # noqa: E402
from training.run_config import load_run_config  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "components_sweep"
DEFAULT_CONFIGS = REPO_ROOT / "configs" / "num_components"
DEFAULT_GRID = 801
DEFAULT_SAMPLES = 4096
DEFAULT_SEED = 0
RESULT_FILENAME = "exploitability.pkl"
RUN_NAME_RE = re.compile(r"^(?P<game>.+)__nc(?P<nc>\d+)__seed(?P<seed>\d+)$")


def parse_run_name(name: str) -> tuple[str, int, int]:
    match = RUN_NAME_RE.match(name)
    if match is None:
        raise ValueError(
            f"run directory name {name!r} does not match "
            f"{{game}}__nc{{N}}__seed{{S}}"
        )
    return match.group("game"), int(match.group("nc")), int(match.group("seed"))


def list_steps(run_dir: Path) -> list[int]:
    return sorted(int(path.stem) for path in run_dir.glob("*.pkl") if path.stem.isdigit())


def resolve_config(run_dir: Path, configs_dir: Path) -> Path:
    candidate = configs_dir / f"{run_dir.name}.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"no config for run {run_dir.name!r}; expected {candidate}"
        )
    return candidate


def find_runs(out_root: Path, game: str | None = None) -> list[Path]:
    runs = []
    for path in sorted(out_root.iterdir()):
        if not path.is_dir() or not list_steps(path):
            continue
        try:
            run_game, _, _ = parse_run_name(path.name)
        except ValueError:
            continue
        if game is not None and run_game != game:
            continue
        runs.append(path)
    return runs


def find_scored_runs(out_root: Path, game: str | None = None) -> list[Path]:
    runs = []
    for path in sorted(out_root.iterdir()):
        if not path.is_dir() or not (path / RESULT_FILENAME).exists():
            continue
        try:
            run_game, _, _ = parse_run_name(path.name)
        except ValueError:
            continue
        if game is not None and run_game != game:
            continue
        runs.append(path)
    return runs


def load_result(run_dir: Path) -> dict:
    with (run_dir / RESULT_FILENAME).open("rb") as f:
        return pickle.load(f)


def _sample_pair(networks, params_pair, observations, spaces, key, samples: int):
    keys = jax.random.split(key, len(networks))
    supports_weights = []
    for i, params in enumerate(params_pair):
        actions = sample_mixture_actions(
            networks[i], params, observations[i], spaces[i], keys[i], samples
        )
        supports_weights.append(empirical_strategy(jax.device_get(actions)))
    return supports_weights


def score_run(
    run_dir: Path,
    config_path: Path,
    *,
    grid: int,
    samples: int,
    seed: int,
    overwrite: bool,
) -> dict:
    out_path = run_dir / RESULT_FILENAME
    if out_path.exists() and not overwrite:
        return load_result(run_dir)

    game_name, num_components, train_seed = parse_run_name(run_dir.name)
    config = load_run_config(config_path)
    game = config.game.build()
    oracle = GridOracle(game, points=grid)
    radius = oracle.cluster_radius()

    steps = list_steps(run_dir)
    if not steps:
        raise FileNotFoundError(f"no {{step}}.pkl checkpoints in {run_dir}")

    first = load_checkpoint_step_multi(run_dir, steps[0], hyperparams_cls=MixturePPOHyperparams)
    hyperparams = (first["player_1"][0], first["player_2"][0])
    networks = tuple(build_mixture_network(hp) for hp in hyperparams)
    observations = (
        game.observation(0, jax.random.PRNGKey(0)),
        game.observation(1, jax.random.PRNGKey(0)),
    )
    spaces = (game.action_space(0), game.action_space(1))

    key = jax.random.PRNGKey(seed)
    expl_live: list[float] = []
    expl_target: list[float] = []
    values: list[float] = []
    support_0: list[int] = []
    support_1: list[int] = []

    for step in steps:
        entries = load_checkpoint_step_multi(run_dir, step, hyperparams_cls=MixturePPOHyperparams)
        live_params = (entries["player_1"][1], entries["player_2"][1])
        target_params = (
            entries[target_entry("player_1")][1],
            entries[target_entry("player_2")][1],
        )

        key, live_key, target_key = jax.random.split(key, 3)
        live = _sample_pair(networks, live_params, observations, spaces, live_key, samples)
        target = _sample_pair(networks, target_params, observations, spaces, target_key, samples)

        live_expl = float(oracle.exploitability(live[0][0], live[0][1], live[1][0], live[1][1]))
        target_expl = float(
            oracle.exploitability(target[0][0], target[0][1], target[1][0], target[1][1])
        )
        value = float(oracle.value(live[0][0], live[0][1], live[1][0], live[1][1]))
        s0 = effective_support(live[0][1], support=live[0][0], radius=radius)
        s1 = effective_support(live[1][1], support=live[1][0], radius=radius)

        expl_live.append(live_expl)
        expl_target.append(target_expl)
        values.append(value)
        support_0.append(s0)
        support_1.append(s1)
        print(
            f"  {run_dir.name} step {step:5d}: "
            f"expl={live_expl:+.5f}  target_expl={target_expl:+.5f}"
        )

    try:
        config_meta = str(config_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        config_meta = str(config_path)

    result = {
        "run": run_dir.name,
        "config": config_meta,
        "game": game_name,
        "num_components": num_components,
        "train_seed": train_seed,
        "grid": grid,
        "samples": samples,
        "seed": seed,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": np.asarray(steps, dtype=np.int32),
        "expl_live": np.asarray(expl_live, dtype=np.float64),
        "expl_target": np.asarray(expl_target, dtype=np.float64),
        "value": np.asarray(values, dtype=np.float64),
        "support_0": np.asarray(support_0, dtype=np.int32),
        "support_1": np.asarray(support_1, dtype=np.int32),
    }
    with out_path.open("wb") as f:
        pickle.dump(result, f)
    return result


def _mean_ci95(
    series: list[tuple[np.ndarray, np.ndarray]],
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate seed curves onto a shared x-grid; return x, mean, lo, hi (95% CI)."""
    if not series:
        return None
    x_lo = max(float(s[0][0]) for s in series)
    x_hi = min(float(s[0][-1]) for s in series)
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


def plot_game(results: list[dict], game: str, out_path: Path) -> Path | None:
    """One figure for ``game``: live (and target) expl vs checkpoint, one curve per K."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_nc: dict[int, list[dict]] = defaultdict(list)
    for result in results:
        if result.get("game") != game:
            continue
        nc = result.get("num_components")
        if nc is None:
            _, nc, _ = parse_run_name(result["run"])
        by_nc[int(nc)].append(result)

    if not by_nc:
        return None

    nc_values = sorted(by_nc)
    cmap = plt.get_cmap("viridis")
    colors = {
        nc: cmap(i / max(len(nc_values) - 1, 1))
        for i, nc in enumerate(nc_values)
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    panels = (
        (axes[0], "expl_live", "live"),
        (axes[1], "expl_target", "target (Polyak)"),
    )

    for ax, y_key, title in panels:
        for nc in nc_values:
            series = []
            for result in by_nc[nc]:
                steps = np.asarray(result["steps"], dtype=np.float64)
                ys = np.asarray(result[y_key], dtype=np.float64)
                if steps.size < 2:
                    continue
                order = np.argsort(steps)
                series.append((steps[order], ys[order]))
            band = _mean_ci95(series)
            if band is None:
                continue
            x, mean, lo, hi = band
            color = colors[nc]
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, mean, color=color, label=f"K={nc}", linewidth=1.8)

        ax.set_xlabel("checkpoint step")
        ax.set_ylabel("exploitability")
        ax.set_title(f"{game}: {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False, title="num_components")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def load_game_results(out_root: Path, game: str) -> list[dict]:
    results = []
    for run_dir in find_scored_runs(out_root, game=game):
        result = load_result(run_dir)
        if "num_components" not in result or "train_seed" not in result:
            parsed_game, nc, train_seed = parse_run_name(result["run"])
            result.setdefault("game", parsed_game)
            result.setdefault("num_components", nc)
            result.setdefault("train_seed", train_seed)
        results.append(result)
    return results


def score_game(
    game: str,
    out_root: Path,
    configs_dir: Path,
    *,
    grid: int,
    samples: int,
    seed: int,
    overwrite: bool,
) -> list[dict]:
    runs = find_runs(out_root, game=game)
    if not runs:
        raise SystemExit(f"no checkpoint runs for game {game!r} under {out_root}")

    results = []
    for run_dir in runs:
        config_path = resolve_config(run_dir, configs_dir)
        print(f"scoring {run_dir.name} with {config_path}")
        result = score_run(
            run_dir,
            config_path,
            grid=grid,
            samples=samples,
            seed=seed,
            overwrite=overwrite,
        )
        print(f"  wrote {len(result['steps'])} scores -> {run_dir / RESULT_FILENAME}")
        results.append(result)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--game", required=True,
                    help="game stem to score/plot, e.g. circle or all_pay_auction")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="root directory of components_sweep runs")
    ap.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS,
                    help="directory of matching train.py YAMLs named {{run}}.yaml")
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID,
                    help="GridOracle points per axis")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                    help="actions sampled from each mixture per checkpoint")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--overwrite", action="store_true",
                    help=f"recompute even when {RESULT_FILENAME} already exists")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip scoring; only build the plot from existing pickles")
    ap.add_argument("--no-plot", action="store_true",
                    help="score without plotting")
    ap.add_argument("--plot-dir", type=Path, default=None,
                    help="directory for the per-game PNG (default: {{out}}/plots)")
    args = ap.parse_args()

    out_root = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    configs_dir = args.configs if args.configs.is_absolute() else REPO_ROOT / args.configs
    plot_dir = args.plot_dir
    if plot_dir is None:
        plot_dir = out_root / "plots"
    elif not plot_dir.is_absolute():
        plot_dir = REPO_ROOT / plot_dir

    if not args.plot_only:
        score_game(
            args.game,
            out_root,
            configs_dir,
            grid=args.grid,
            samples=args.samples,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    if not args.no_plot:
        results = load_game_results(out_root, args.game)
        if not results:
            raise SystemExit(
                f"no {RESULT_FILENAME} files for game {args.game!r} under {out_root}"
            )
        path = plot_game(results, args.game, plot_dir / f"{args.game}.png")
        if path is None:
            raise SystemExit(f"nothing to plot for game {args.game!r}")
        print(f"wrote plot {path}")


if __name__ == "__main__":
    main()
