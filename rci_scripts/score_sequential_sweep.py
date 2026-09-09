"""Score ``train_sequential.py`` checkpoints with exact / RL exploitability.

Walks every run under ``data/sequential_sweep/`` (or ``--out``), loads each
``checkpoints/{step}.pkl`` that exists -- unfinished cluster jobs are fine --
and writes ``exploitability.pkl`` beside that run's checkpoints.

Kuhn has an exact tree best response (``baselines.neural.sequential_scoring``);
everything else trains an approximate BR with ``rl_exploitability_bound`` and
reads the bound from ``--episodes`` hands. BR budget and the final evaluation
are CLI hyperparameters so measurement stays outside the training wall-time.

Reuse is deliberate: this script only finds runs, rebuilds the
``PolicyMixture`` pair a checkpoint stores, and calls the same scoring
functions ``train_sequential.py`` uses online.

The pickle is a dict with numpy arrays ready for later plots:

    {
      "run", "config", "game", "solver", "train_seed",
      "exact", "br_steps", "br_epochs", "episodes", "exact_grid", "seed",
      "scored_at",
      "steps",          # (T,) checkpoint indices
      "expl",           # (T,) exact exploitability when ``exact`` else NaN
      "expl_lb",        # (T,) RL lower bound when not exact else NaN
      "br_0", "br_1",   # (T,) per-player BR values (exact or lb)
      "value",          # (T,) game value under the pair (exact only; else NaN)
      "target_expl",    # (T,) Polyak iterate on Kuhn when present; else absent
      "wall_time", "episodes_train", "env_steps",  # from history.json when known
    }

    python rci_scripts/score_sequential_sweep.py
    python rci_scripts/score_sequential_sweep.py --br-steps 100 --br-epochs 20 --episodes 50000
    python rci_scripts/score_sequential_sweep.py --game kuhn_solvers --overwrite
    python rci_scripts/score_sequential_sweep.py --run kuhn_solvers__self_play__seed0
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from baselines.neural import sequential_oracle as so  # noqa: E402
from baselines.neural.sequential_scoring import (  # noqa: E402
    exact_kuhn_exploitability,
    has_exact_exploitability,
    rl_exploitability_bound,
)
from training.checkpoint import load_checkpoint_step_multi, target_entry  # noqa: E402
from training.config import MixturePPOHyperparams  # noqa: E402
from training.mixture import build_mixture_network  # noqa: E402
from training.run_config import load_run_config  # noqa: E402
from train_sequential import prepare_game, solver_hyperparams  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "sequential_sweep"
DEFAULT_CONFIGS = REPO_ROOT / "configs" / "sequential_sweep"
DEFAULT_BR_STEPS = 50
DEFAULT_BR_EPOCHS = 20
DEFAULT_EPISODES = 20_000
DEFAULT_SEED = 0
RESULT_FILENAME = "exploitability.pkl"
CHECKPOINT_DIRNAME = "checkpoints"
# ``{game}__{solver}[__swept...]__seed{N}`` from generate_sequential_sweep.py
RUN_NAME_RE = re.compile(
    r"^(?P<game>.+?)__(?P<solver>[^_].*?)(?:__.*)?__seed(?P<seed>\d+)$"
)
PSRO_POLICY_RE = re.compile(r"^player(?P<player>[01])_policy(?P<index>\d+)$")


def parse_run_name(name: str) -> tuple[str, str, int]:
    match = RUN_NAME_RE.match(name)
    if match is None:
        raise ValueError(
            f"run directory name {name!r} does not match "
            f"{{game}}__{{solver}}[__...]__seed{{S}}"
        )
    return match.group("game"), match.group("solver"), int(match.group("seed"))


def list_steps(checkpoint_dir: Path) -> list[int]:
    return sorted(
        int(path.stem) for path in checkpoint_dir.glob("*.pkl") if path.stem.isdigit()
    )


def resolve_config(run_dir: Path, configs_dir: Path) -> Path:
    candidate = configs_dir / f"{run_dir.name}.yaml"
    if candidate.exists():
        return candidate
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        config_path = meta.get("config_path")
        if config_path:
            path = Path(config_path)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if path.exists():
                return path
    raise FileNotFoundError(
        f"no config for run {run_dir.name!r}; expected {candidate} "
        f"or a readable config_path in {meta_path}"
    )


def find_runs(
    out_root: Path,
    *,
    game: str | None = None,
    solver: str | None = None,
    run: str | None = None,
) -> list[Path]:
    runs = []
    for path in sorted(out_root.iterdir()):
        if not path.is_dir():
            continue
        if run is not None and path.name != run:
            continue
        ckpt = path / CHECKPOINT_DIRNAME
        if not ckpt.is_dir() or not list_steps(ckpt):
            continue
        try:
            run_game, run_solver, _ = parse_run_name(path.name)
        except ValueError:
            # Still score directories that have checkpoints + meta, even if the name
            # does not match the sweep convention.
            run_game = run_solver = None
            meta_path = path / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                run_solver = meta.get("solver")
                game_block = (meta.get("config") or {}).get("game") or {}
                run_game = game_block.get("name")
        # Sweep names use the config stem (``kuhn_solvers``); substring match
        # lets ``--game kuhn`` select those runs.
        if game is not None and (run_game is None or game not in str(run_game)):
            continue
        if solver is not None and run_solver != solver:
            continue
        runs.append(path)
    return runs


def load_result(run_dir: Path) -> dict:
    with (run_dir / RESULT_FILENAME).open("rb") as f:
        return pickle.load(f)


def history_by_step(run_dir: Path) -> dict[int, dict]:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        # Unfinished jobs still stream rows here.
        stream = run_dir / "metrics.jsonl"
        if not stream.exists():
            return {}
        rows = [json.loads(line) for line in stream.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(history_path.read_text())
    return {int(row["t"]): row for row in rows if "t" in row}


def run_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _single_mixture(name: str, hyperparams, params) -> so.PolicyMixture:
    return so.single(build_mixture_network(hyperparams), hyperparams, params, name)


def load_mixtures(
    checkpoint_dir: Path,
    step: int,
    *,
    target: bool = False,
) -> tuple[so.PolicyMixture, so.PolicyMixture]:
    """Rebuild the strategy pair a sequential checkpoint stores.

    Self-play / NFSP / discrete_mmd / RPN write ``player_0`` / ``player_1``
    (plus optional ``*_target``). PSRO writes ``player{p}_policy{k}`` with
    meta-weights in ``{step}.npz``.
    """
    entries = load_checkpoint_step_multi(
        checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams
    )

    if "player_0" in entries and "player_1" in entries:
        mixtures = []
        for player in (0, 1):
            live = f"player_{player}"
            name = target_entry(live) if target else live
            if name not in entries:
                if target:
                    raise KeyError(
                        f"checkpoint {checkpoint_dir / f'{step}.pkl'} has no {name!r}; "
                        f"it holds {sorted(entries)}"
                    )
                raise KeyError(name)
            hyperparams, params = entries[name]
            mixtures.append(_single_mixture(name, hyperparams, params))
        return mixtures[0], mixtures[1]

    arrays_path = checkpoint_dir / f"{step}.npz"
    arrays = np.load(arrays_path) if arrays_path.exists() else {}
    populations: list[list[tuple[int, object, object]]] = [[], []]
    for name, (hyperparams, params) in entries.items():
        match = PSRO_POLICY_RE.match(name)
        if match is None:
            continue
        player = int(match.group("player"))
        index = int(match.group("index"))
        populations[player].append((index, hyperparams, params))

    mixtures = []
    for player in (0, 1):
        members = sorted(populations[player], key=lambda item: item[0])
        if not members:
            raise KeyError(
                f"checkpoint {checkpoint_dir / f'{step}.pkl'} has neither "
                f"player_{player} nor player{player}_policy*; holds {sorted(entries)}"
            )
        hyperparams = members[0][1]
        params_list = [params for _, _, params in members]
        key = f"meta_weights_{player}"
        if key in arrays:
            weights = np.asarray(arrays[key], dtype=np.float64)
        else:
            weights = np.ones(len(params_list), dtype=np.float64)
        mixtures.append(
            so.population(hyperparams, params_list, weights, label=f"player_{player}")
        )
    return mixtures[0], mixtures[1]


def score_checkpoint(
    game,
    mixtures: tuple[so.PolicyMixture, so.PolicyMixture],
    br_hyperparams: tuple,
    *,
    exact: bool,
    br_steps: int,
    br_epochs: int,
    episodes: int,
    exact_grid: int | None,
    seed: int,
    key: jax.Array,
) -> tuple[dict[str, float], jax.Array]:
    """One checkpoint -> metrics. Kuhn uses the exact tree BR; else trains two BRs."""
    if exact:
        return exact_kuhn_exploitability(game, mixtures, exact_grid), key

    key, br_key = jax.random.split(key)
    metrics = rl_exploitability_bound(
        game,
        mixtures,
        br_hyperparams,
        steps=br_steps,
        epochs=br_epochs,
        seed=seed,
        episodes=episodes,
        key=br_key,
    )
    return metrics, key


def score_run(
    run_dir: Path,
    config_path: Path,
    *,
    br_steps: int,
    br_epochs: int,
    episodes: int,
    exact_grid: int | None,
    seed: int,
    include_target: bool,
    overwrite: bool,
) -> dict:
    out_path = run_dir / RESULT_FILENAME
    if out_path.exists() and not overwrite:
        return load_result(run_dir)

    jax.config.update("jax_enable_x64", False)

    config = load_run_config(config_path)
    game = prepare_game(config.game.build(), config)
    exact = has_exact_exploitability(game)
    br_hps = tuple(
        solver_hyperparams(game, player, config, best_response=True) for player in (0, 1)
    )

    checkpoint_dir = run_dir / CHECKPOINT_DIRNAME
    steps = list_steps(checkpoint_dir)
    if not steps:
        raise FileNotFoundError(f"no {{step}}.pkl checkpoints in {checkpoint_dir}")

    meta = run_meta(run_dir)
    try:
        game_name, solver_name, train_seed = parse_run_name(run_dir.name)
    except ValueError:
        game_name = ((meta.get("config") or {}).get("game") or {}).get("name", "unknown")
        solver_name = meta.get("solver") or config.train.solver
        train_seed = int(config.train.seed)
    solver_name = meta.get("solver") or solver_name or config.train.solver
    by_t = history_by_step(run_dir)

    grid = exact_grid if exact_grid is not None else config.scoring.exact_grid
    key = jax.random.PRNGKey(seed)

    expl: list[float] = []
    expl_lb: list[float] = []
    br_0: list[float] = []
    br_1: list[float] = []
    values: list[float] = []
    target_expl: list[float] = []
    wall_time: list[float] = []
    episodes_train: list[float] = []
    env_steps: list[float] = []
    have_target = False

    for step in steps:
        mixtures = load_mixtures(checkpoint_dir, step, target=False)
        metrics, key = score_checkpoint(
            game,
            mixtures,
            br_hps,
            exact=exact,
            br_steps=br_steps,
            br_epochs=br_epochs,
            episodes=episodes,
            exact_grid=grid,
            seed=seed + step,
            key=key,
        )

        if exact:
            expl.append(float(metrics["expl"]))
            expl_lb.append(float("nan"))
            br_0.append(float(metrics["br_0"]))
            br_1.append(float(metrics["br_1"]))
            values.append(float(metrics["value"]))
            headline = f"expl={metrics['expl']:+.5f}"
        else:
            expl.append(float("nan"))
            expl_lb.append(float(metrics["expl_lb"]))
            br_0.append(float(metrics["br_lb_0"]))
            br_1.append(float(metrics["br_lb_1"]))
            values.append(float("nan"))
            headline = f"expl_lb={metrics['expl_lb']:+.5f}"

        if include_target and exact:
            try:
                target = load_mixtures(checkpoint_dir, step, target=True)
            except KeyError:
                target_expl.append(float("nan"))
            else:
                target_metrics, key = score_checkpoint(
                    game,
                    target,
                    br_hps,
                    exact=True,
                    br_steps=br_steps,
                    br_epochs=br_epochs,
                    episodes=episodes,
                    exact_grid=grid,
                    seed=seed + 10_000 + step,
                    key=key,
                )
                target_expl.append(float(target_metrics["expl"]))
                have_target = True
                headline += f"  target_expl={target_metrics['expl']:+.5f}"

        row = by_t.get(step, {})
        wall_time.append(float(row["wall_time"]) if "wall_time" in row else float("nan"))
        episodes_train.append(float(row["episodes"]) if "episodes" in row else float("nan"))
        env_steps.append(float(row["env_steps"]) if "env_steps" in row else float("nan"))
        print(f"  {run_dir.name} step {step:5d}: {headline}")

    try:
        config_meta = str(config_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        config_meta = str(config_path)

    result = {
        "run": run_dir.name,
        "config": config_meta,
        "game": game_name,
        "solver": solver_name,
        "train_seed": train_seed,
        "exact": exact,
        "br_steps": br_steps,
        "br_epochs": br_epochs,
        "episodes": episodes,
        "exact_grid": grid,
        "seed": seed,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": np.asarray(steps, dtype=np.int32),
        "expl": np.asarray(expl, dtype=np.float64),
        "expl_lb": np.asarray(expl_lb, dtype=np.float64),
        "br_0": np.asarray(br_0, dtype=np.float64),
        "br_1": np.asarray(br_1, dtype=np.float64),
        "value": np.asarray(values, dtype=np.float64),
        "wall_time": np.asarray(wall_time, dtype=np.float64),
        "episodes_train": np.asarray(episodes_train, dtype=np.float64),
        "env_steps": np.asarray(env_steps, dtype=np.float64),
    }
    if have_target:
        result["target_expl"] = np.asarray(target_expl, dtype=np.float64)

    with out_path.open("wb") as f:
        pickle.dump(result, f)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="root directory of sequential_sweep runs "
             f"(default: {DEFAULT_OUT.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--configs",
        type=Path,
        default=DEFAULT_CONFIGS,
        help="directory of matching train_sequential YAMLs named {{run}}.yaml",
    )
    ap.add_argument("--game", default=None, help="only score runs whose game stem matches")
    ap.add_argument("--solver", default=None, help="only score runs of this train.solver")
    ap.add_argument("--run", default=None, help="score only this run directory name")
    ap.add_argument(
        "--br-steps",
        type=int,
        default=DEFAULT_BR_STEPS,
        help="PPO chunks per approximate best response (non-Kuhn only)",
    )
    ap.add_argument(
        "--br-epochs",
        type=int,
        default=DEFAULT_BR_EPOCHS,
        help="PPO iterations per BR chunk (non-Kuhn only)",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="hands for the final exploitability estimate after BR training",
    )
    ap.add_argument(
        "--exact-grid",
        type=int,
        default=None,
        help="bet-grid points for Kuhn's exact metric (default: the game's own)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--no-target",
        action="store_true",
        help="skip Polyak-target scoring on Kuhn even when checkpoints carry it",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=f"recompute even when {RESULT_FILENAME} already exists",
    )
    args = ap.parse_args()

    out_root = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    configs_dir = args.configs if args.configs.is_absolute() else REPO_ROOT / args.configs

    if not out_root.is_dir():
        raise SystemExit(f"no sweep root at {out_root}")

    runs = find_runs(
        out_root, game=args.game, solver=args.solver, run=args.run
    )
    if not runs:
        raise SystemExit(
            f"no checkpoint runs under {out_root}"
            + (f" for game={args.game!r}" if args.game else "")
            + (f" solver={args.solver!r}" if args.solver else "")
            + (f" run={args.run!r}" if args.run else "")
        )

    print(
        f"scoring {len(runs)} run(s) under {out_root}  "
        f"(BR {args.br_steps}x{args.br_epochs}, eval {args.episodes} episodes)"
    )
    for run_dir in runs:
        config_path = resolve_config(run_dir, configs_dir)
        print(f"scoring {run_dir.name} with {config_path}")
        result = score_run(
            run_dir,
            config_path,
            br_steps=args.br_steps,
            br_epochs=args.br_epochs,
            episodes=args.episodes,
            exact_grid=args.exact_grid,
            seed=args.seed,
            include_target=not args.no_target,
            overwrite=args.overwrite,
        )
        print(
            f"  wrote {len(result['steps'])} scores -> {run_dir / RESULT_FILENAME} "
            f"({'exact' if result['exact'] else 'rl bound'})"
        )


if __name__ == "__main__":
    main()
