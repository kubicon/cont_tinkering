"""Run every representation baseline on every one-shot game, keeping the checkpoints.

    python experiments/one_shot_tabular/run_baselines.py [options]

The three solvers in `baselines/` answer the same question the mixture solver in
`run_idealized.py` does -- where does self-play end up on a one-shot game with a
mixed equilibrium -- while representing a strategy without a Gaussian mixture:
a probability vector over a fine grid (`grid_mmd`), a support grown by best
responses (`double_oracle`), or a weighted particle cloud (`particle_mean_field`).
This script runs the grid of (game x algorithm x seed), writes each run's iterates
to disk, and prints one table of final exploitabilities.

Everything a later analysis needs is on disk, so the expensive part never has to be
repeated to answer a new question:

    <out>/<game>/<algorithm>/seed<k>/meta.json      settings, timings, final metrics
                                    /history.json   per-logged-iteration metrics
                                    /checkpoints/   step_*.npz + index.json

A checkpoint is a `baselines.common.StrategyPair`: both players' support and weights
at one iteration, in the *same* format for all three algorithms (see that class for
why that matters). Read one back with

    from baselines.common import StrategyPair, load_checkpoints, latest_checkpoint

and re-score it against any oracle you like -- a finer grid, a different metric, the
`run_idealized` quadrature backend -- without re-running anything.

A run whose `meta.json` already exists is skipped, so the script is restartable and
adding a game later costs only that game. `--overwrite` forces a re-run, `--dry-run`
prints the plan.

Seeds: only `particle_mean_field` reads one (its particles start at random positions).
`grid_mmd` and `double_oracle` as configured here are deterministic, so they are run
once regardless of how many seeds are asked for -- `--seed-all` overrides that if you
have made them stochastic (e.g. `--do-init random`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, so `import baselines` works

import numpy as np  # noqa: E402

from baselines.common import CheckpointWriter, GridOracle, effective_support, load_game, \
    top_atoms  # noqa: E402
from baselines.double_oracle import run_double_oracle  # noqa: E402
from baselines.grid_mmd import run_grid_mmd  # noqa: E402
from baselines.particle_mean_field import run_particle_mean_field  # noqa: E402

# The one-shot, box-action games whose equilibrium is genuinely mixed -- the setting
# the mixture head is meant for. `decoy_well` is the counterexample config: the K=2
# mixture converges to the decoy there, so what the non-parametric baselines do on it
# is the point of the whole comparison.
DEFAULT_CONFIGS = (
    "configs/two_point.yaml",
    "configs/multi_point.yaml",
    "configs/idealized_decoy_well.yaml",
    "configs/idealized_forsaken.yaml",
    "configs/matching_pennies.yaml",
)

ALGORITHMS = ("grid_mmd", "double_oracle", "particle_mean_field")

# Which algorithms actually consume `seed`; the others are run once. See the module
# docstring.
SEEDED = {"particle_mean_field"}


@dataclasses.dataclass
class Settings:
    """Every knob the grid is run with, recorded verbatim into each `meta.json`."""

    grid: int = 401
    # grid_mmd / particle_mean_field share the mirror-descent knobs, and these values
    # are the ones the two-peak game was verified with (see baselines/README.md).
    iters: int = 20_000
    lr_weight: float = 10.0
    tau: float = 1.0
    tau_ent: float = 0.0
    magnet_interval: int = 200
    # particle_mean_field only
    particles: int = 64
    lr_position: float = 1e-2
    temperature: float = 0.0
    # double_oracle only
    do_rounds: int = 30
    do_tol: float = 1e-3
    do_dedup: float = 1e-3
    do_polish: bool = True
    do_init: str = "center"
    # how many checkpoints a gradient-style run writes (double oracle writes one per round)
    checkpoints: int = 100


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE.parents[1],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:      # a tarball with no .git is not a reason to lose the run
        return None


def _strategy_summary(oracle: GridOracle, support_0, weights_0, support_1, weights_1) -> dict:
    """The bit of a final strategy worth having in `meta.json` without opening an npz."""
    radius = oracle.cluster_radius()
    return {
        "support_0": effective_support(weights_0, support=np.asarray(support_0), radius=radius),
        "support_1": effective_support(weights_1, support=np.asarray(support_1), radius=radius),
        "atoms_0": top_atoms(support_0, weights_0, top=8, radius=radius),
        "atoms_1": top_atoms(support_1, weights_1, top=8, radius=radius),
    }


# --------------------------------------------------------------------------- runners
#
# One adapter per algorithm, all with the same signature, so the driver below neither
# knows nor cares which is which: `(game, oracle, seed, settings, checkpoint_fn)` in,
# `(history, extra_meta, strategy)` out, where `strategy` is the final
# `(support_0, weights_0, support_1, weights_1)`.


def _run_grid_mmd(game, oracle, seed, s: Settings, checkpoint_fn):
    result = run_grid_mmd(
        game, oracle, iters=s.iters, lr=s.lr_weight, tau=s.tau, tau_ent=s.tau_ent,
        magnet_interval=s.magnet_interval, log_every=max(s.iters // s.checkpoints, 1),
        checkpoint_fn=checkpoint_fn,
    )
    extra = {"avg_expl": result.history[-1]["avg_expl"]}
    return result.history, extra, (result.grid_0, result.weights_0,
                                   result.grid_1, result.weights_1)


def _run_double_oracle(game, oracle, seed, s: Settings, checkpoint_fn):
    result = run_double_oracle(
        game, oracle, iters=s.do_rounds, tol=s.do_tol, dedup=s.do_dedup, polish=s.do_polish,
        init=s.do_init, seed=seed, checkpoint_fn=checkpoint_fn,
    )
    extra = {"converged": result.converged, "stop_reason": result.stop_reason,
             "value": result.value, "rounds": len(result.history)}
    return result.history, extra, (result.support_0, result.weights_0,
                                   result.support_1, result.weights_1)


def _run_particle(game, oracle, seed, s: Settings, checkpoint_fn):
    result = run_particle_mean_field(
        game, oracle, particles=s.particles, iters=s.iters, lr_position=s.lr_position,
        lr_weight=s.lr_weight, tau=s.tau, tau_ent=s.tau_ent,
        magnet_interval=s.magnet_interval, temperature=s.temperature, seed=seed,
        log_every=max(s.iters // s.checkpoints, 1), checkpoint_fn=checkpoint_fn,
    )
    return result.history, {}, (result.support_0, result.weights_0,
                                result.support_1, result.weights_1)


RUNNERS = {
    "grid_mmd": _run_grid_mmd,
    "double_oracle": _run_double_oracle,
    "particle_mean_field": _run_particle,
}


# --------------------------------------------------------------------------- driver


def run_cell(config: str, algorithm: str, seed: int, settings: Settings,
             out_root: Path, overwrite: bool = False) -> dict:
    """One (game, algorithm, seed) run, written under `out_root`. Returns its summary row.

    Failures are caught and reported rather than raised: one game the discretization
    cannot handle (a `N^d` grid that does not fit, say) should not take the other
    fifteen cells of an overnight sweep with it.
    """
    game_tag = Path(config).stem
    directory = out_root / game_tag / algorithm / f"seed{seed}"
    meta_path = directory / "meta.json"
    row = {"game": game_tag, "algorithm": algorithm, "seed": seed,
           "config": config, "dir": str(directory)}

    if meta_path.exists() and not overwrite:
        stored = json.loads(meta_path.read_text())
        print(f"  skip (already done): {directory}")
        return {**row, **{k: stored.get(k) for k in
                          ("status", "expl", "best_expl", "tail_expl", "seconds",
                           "support_0", "support_1")}}

    directory.mkdir(parents=True, exist_ok=True)
    writer = CheckpointWriter(directory / "checkpoints")

    started = time.time()
    try:
        # Inside the `try` with the run itself: the likeliest failure of a cell is not the
        # solver but its *setup* -- a simplex action space these baselines cannot
        # discretize, or an `N^d` payoff matrix that does not fit.
        game, game_config = load_game(config)
        oracle = GridOracle(game, points=settings.grid)
        history, extra, strategy = RUNNERS[algorithm](game, oracle, seed, settings, writer)
    except Exception as exc:                    # noqa: BLE001 -- recorded, not swallowed
        seconds = time.time() - started
        (directory / "error.txt").write_text(traceback.format_exc())
        print(f"  FAILED after {seconds:.1f}s: {type(exc).__name__}: {exc}")
        return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "seconds": seconds}
    seconds = time.time() - started

    expl = [h["expl"] for h in history]
    meta = {
        **row,
        "status": "ok",
        "game_config": dataclasses.asdict(game_config),
        "settings": dataclasses.asdict(settings),
        "seconds": seconds,
        "expl": expl[-1],
        "best_expl": min(expl),
        # Mean over the last 30% of the run: a last iterate that is orbiting reads much
        # better or worse than it deserves depending on where the log happened to land.
        "tail_expl": float(np.mean(expl[int(len(expl) * 0.7):])),
        "checkpoints": len(writer.entries),
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
        **_strategy_summary(oracle, *strategy),
    }
    writer.write_index(meta)
    (directory / "history.json").write_text(json.dumps(history, indent=2))
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"  expl {meta['expl']:+.5f} (best {meta['best_expl']:+.5f})  "
          f"support {meta['support_0']}/{meta['support_1']}  "
          f"{meta['checkpoints']} checkpoints  {seconds:.1f}s")
    print(f"    P0 {meta['atoms_0']}")
    print(f"    P1 {meta['atoms_1']}")
    return {k: meta[k] for k in
            ("game", "algorithm", "seed", "config", "dir", "status", "expl", "best_expl",
             "tail_expl", "seconds", "support_0", "support_1")}


def summary_table(rows: list[dict]) -> str:
    """The runs as a markdown table, one line each, grouped by game."""
    header = "|game|algorithm|seed|final expl|best expl|support|seconds|\n|-|-|-|-|-|-|-|\n"
    lines = []
    for r in sorted(rows, key=lambda r: (r["game"], r["algorithm"], r["seed"])):
        if r.get("status") != "ok":
            lines.append(f"|{r['game']}|{r['algorithm']}|{r['seed']}|FAILED|||"
                         f"{r.get('seconds', 0):.0f}|")
            continue
        lines.append(
            f"|{r['game']}|{r['algorithm']}|{r['seed']}|{r['expl']:+.5f}|{r['best_expl']:+.5f}|"
            f"{r['support_0']}/{r['support_1']}|{r['seconds']:.0f}|")
    return header + "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                    help="config paths (only their `game:` section is read)")
    ap.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--seed-all", action="store_true",
                    help="run every algorithm at every seed, not just the stochastic ones")
    ap.add_argument("--out", default="data/one_shot_tabular", help="output root")
    ap.add_argument("--overwrite", action="store_true", help="re-run cells that already have a meta.json")
    ap.add_argument("--dry-run", action="store_true", help="print the grid and exit")

    defaults = Settings()
    ap.add_argument("--grid", type=int, default=defaults.grid)
    ap.add_argument("--iters", type=int, default=defaults.iters)
    ap.add_argument("--lr-weight", type=float, default=defaults.lr_weight)
    ap.add_argument("--tau", type=float, default=defaults.tau)
    ap.add_argument("--tau-ent", type=float, default=defaults.tau_ent)
    ap.add_argument("--magnet-interval", type=int, default=defaults.magnet_interval)
    ap.add_argument("--particles", type=int, default=defaults.particles)
    ap.add_argument("--lr-position", type=float, default=defaults.lr_position)
    ap.add_argument("--temperature", type=float, default=defaults.temperature)
    ap.add_argument("--do-rounds", type=int, default=defaults.do_rounds)
    ap.add_argument("--do-tol", type=float, default=defaults.do_tol)
    ap.add_argument("--do-dedup", type=float, default=defaults.do_dedup)
    ap.add_argument("--do-no-polish", action="store_true", help="skip the gradient polish on best responses")
    ap.add_argument("--do-init", choices=("center", "random"), default=defaults.do_init)
    ap.add_argument("--checkpoints", type=int, default=defaults.checkpoints,
                    help="checkpoints per gradient-style run (double oracle writes one per round)")
    args = ap.parse_args()

    settings = Settings(
        grid=args.grid, iters=args.iters, lr_weight=args.lr_weight, tau=args.tau,
        tau_ent=args.tau_ent, magnet_interval=args.magnet_interval, particles=args.particles,
        lr_position=args.lr_position, temperature=args.temperature, do_rounds=args.do_rounds,
        do_tol=args.do_tol, do_dedup=args.do_dedup, do_polish=not args.do_no_polish,
        do_init=args.do_init, checkpoints=args.checkpoints,
    )
    out_root = Path(args.out)

    cells = [(config, algorithm, seed)
             for config in args.configs
             for algorithm in args.algorithms
             for seed in (args.seeds if (args.seed_all or algorithm in SEEDED) else args.seeds[:1])]

    print(f"{len(cells)} runs -> {out_root}")
    print(f"settings: {dataclasses.asdict(settings)}\n")
    if args.dry_run:
        for config, algorithm, seed in cells:
            print(f"  {Path(config).stem:24s} {algorithm:20s} seed{seed}")
        return

    rows: list[dict] = []
    for config, algorithm, seed in cells:
        print(f"[{len(rows) + 1}/{len(cells)}] {Path(config).stem} / {algorithm} / seed{seed}")
        rows.append(run_cell(config, algorithm, seed, settings, out_root, args.overwrite))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    table = summary_table(rows)
    (out_root / "summary.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nsummary -> {out_root / 'summary.json'} and {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
