"""How far the MJX sumo games scale with `ppo.batch_size`, measured rather than guessed.

    python experiments/mjx_batch_scaling/benchmark.py --dry-run
    python experiments/mjx_batch_scaling/benchmark.py            # the full sweep
    python experiments/mjx_batch_scaling/benchmark.py --summarize data/mjx_batch_scaling/results.json

Every `configs/mjx_*_sumo.yaml` is trained for `--iterations` iterations (100 by
default) at each of `--batch-sizes`, and each cell reports what that batch cost:
compile time, seconds per iteration, decisions per second, and the GPU's peak
bytes in use. Nothing here is a *training* run -- no checkpoints are written and
the policy at the end is thrown away; the only outputs are the timings.

The question this answers is where the batch stops paying for itself. An
`mjx.Data` for the whole batch rides in the rollout scan's carry, so raising the
batch raises device memory linearly while throughput only improves until the GPU
is saturated. Past that point a bigger batch buys a better gradient estimate at a
proportional cost in wall time, and eventually fails outright. The `scaling`
column in the summary is exactly that trade-off:

    scaling = (decisions/s at this batch) / (decisions/s at the smallest batch)
              -------------------------------------------------------------
                          this batch / the smallest batch

1.0 is a batch that is free (the GPU was idle anyway); 0.5 means half of the
batch increase came back as wall time; a failed cell is where it stops being
possible at all. Read the largest batch whose `scaling` is still near 1 as the
one to train at.

**One process per cell.** A cell that exhausts the GPU takes its process down
with it, and JAX does not reliably give that memory back inside a surviving
process, so each (config, batch) is a separate `--cell` invocation of this same
file. A failure is therefore recorded, not fatal: by default the remaining
larger batches for that config are skipped (`--no-stop-on-failure` runs them
anyway), and the next config starts from a clean process.

**Warp contact capacity scales with the batch.** `warp_naconmax` is shared by
every vmapped world, so a batch sweep that left it at the config's value would
measure contact-slot overflow rather than batch scaling. Each cell sets
`warp_naconmax = --naconmax-per-world * batch_size`; the per-world defaults in
`NACONMAX_PER_WORLD` are the puck's one contact and a deliberately generous
allowance for the walkers. A cell that dies with a contact-capacity error is
telling you that allowance was too small -- raise the flag, not the batch.

A cell that already has a result file is skipped, *including a failed one*, so
that adding a config or a batch size later re-runs only what is new. Re-running
a cell whose failure you have since fixed (a bigger `--naconmax-per-world`, a
longer `--timeout`) needs `--overwrite`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The MJX sumo family: the 2-D puck, and the three walkers. `mjx_sumo_br.yaml` is a
# best-response *measurement* config with no `game` section, so it is not one of these.
DEFAULT_CONFIGS = (
    "configs/mjx_sumo.yaml",
    "configs/mjx_spider_sumo.yaml",
    "configs/mjx_ant_sumo.yaml",
    "configs/mjx_bug_sumo.yaml",
)

DEFAULT_BATCH_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)

# Warp contact slots per world, by game. The puck model has at most one contact
# (`configs/mjx_sumo.yaml` says so, and sizes `warp_naconmax` for the eval batch on
# exactly that basis); a walker runs "a couple of hundred", so 256 is an allowance
# rather than a count. Both are only ever an upper bound: unused slots cost memory,
# too few is a hard failure, and the failure names itself in the cell's error.
NACONMAX_PER_WORLD = {
    "mjx_sumo": 1,
    "mjx_ant_sumo": 256,
    "mjx_bug_sumo": 384,
    "mjx_spider_sumo": 192,
}

# Constraint rows per world, which unlike naconmax is *not* batched.
NJMAX = {
    "mjx_sumo": 8,
    "mjx_ant_sumo": 512,
    "mjx_bug_sumo": 768,
    "mjx_spider_sumo": 384,
}

DEFAULT_OUT = "data/mjx_batch_scaling"


def cell_name(config_path: str, batch_size: int) -> str:
    return f"{Path(config_path).stem}__batch{batch_size}"


# --------------------------------------------------------------------------- one cell


def run_cell(
    config_path: str,
    batch_size: int,
    iterations: int,
    physics_backend: str,
    naconmax_per_world: int | None,
    njmax: int | None,
    seed: int,
) -> dict:
    """Time `iterations` self-play iterations at one batch size. Runs in its own process.

    Two chunks of `iterations` are run, not one: the first pays for tracing and
    XLA compilation (and, on Warp, for the CUDA graph capture), the second is the
    steady state. Both are reported -- compile time is a real cost of a batch
    size, it just is not a per-iteration one.
    """
    # Imported here rather than at module scope: the driver process must not pull in
    # JAX at all, or every cell inherits an initialized backend through fork.
    from training.hyperparams import build_hyperparams
    from training.run_config import load_run_config
    from training.sequential_trainer import SequentialSelfPlayPPOTrainer

    config = load_run_config(config_path)
    game_name = Path(config_path).stem

    game_overrides: dict[str, object] = {}
    if physics_backend != "config":
        game_overrides["physics_backend"] = physics_backend
    if hasattr(config.game, "warp_naconmax"):
        per_world = naconmax_per_world if naconmax_per_world is not None else NACONMAX_PER_WORLD.get(game_name, 256)
        game_overrides["warp_naconmax"] = per_world * batch_size
        rows = njmax if njmax is not None else (config.game.warp_njmax or NJMAX.get(game_name, 512))
        game_overrides["warp_njmax"] = rows
    game_config = dataclasses.replace(config.game, **game_overrides)
    config = dataclasses.replace(
        config,
        game=game_config,
        ppo=dataclasses.replace(config.ppo, batch_size=batch_size),
    )

    started = time.monotonic()
    game = game_config.build()
    trainer = SequentialSelfPlayPPOTrainer(
        game,
        build_hyperparams(game, 0, config),
        build_hyperparams(game, 1, config),
        seed=seed,
    )
    setup_seconds = time.monotonic() - started

    # `train` blocks: it pulls the chunk's metrics back to the host at the end of
    # each chunk, which is a device synchronization, so these timings are real.
    first_started = time.monotonic()
    trainer.train(1, epochs=iterations, checkpoint_dir=None)
    first_chunk_seconds = time.monotonic() - first_started

    steady_started = time.monotonic()
    history = trainer.train(1, epochs=iterations, checkpoint_dir=None)
    train_seconds = time.monotonic() - steady_started

    record = history[-1]
    # A "decision" is one player's move; an episode holds `episode_length` of them
    # across both players, which is the unit the rollout scan actually costs.
    decisions = batch_size * iterations * float(record["episode_length"])

    result = {
        "config": config_path,
        "game": game_name,
        "batch_size": batch_size,
        "iterations": iterations,
        "status": "ok",
        "physics_backend": getattr(game, "physics_backend", None),
        "warp_naconmax": game_overrides.get("warp_naconmax"),
        "warp_njmax": game_overrides.get("warp_njmax"),
        "horizon": getattr(game, "horizon", None),
        "setup_seconds": setup_seconds,
        # The first chunk minus the second: everything paid once per batch shape.
        "compile_seconds": max(first_chunk_seconds - train_seconds, 0.0),
        "first_chunk_seconds": first_chunk_seconds,
        "train_seconds": train_seconds,
        "seconds_per_iteration": train_seconds / iterations,
        "episode_length": float(record["episode_length"]),
        "decisions": decisions,
        "decisions_per_second": decisions / train_seconds if train_seconds > 0 else None,
        "payoff": float(record["payoff"]),
    }
    result.update(device_memory())
    return result


def device_memory() -> dict:
    """Peak device memory, where the backend reports it (GPU does, CPU does not)."""
    import jax

    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:                        # noqa: BLE001 -- a missing stat is not a failure
        return {}
    return {
        "peak_bytes": stats.get("peak_bytes_in_use"),
        "bytes_in_use": stats.get("bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def classify(message: str) -> str:
    """Why a cell died, as far as the message says -- the interesting part of the answer."""
    lowered = message.lower()
    if "resource_exhausted" in lowered or "out of memory" in lowered or "oom" in lowered:
        return "oom"
    if "naconmax" in lowered or "ncon" in lowered or "contact" in lowered:
        return "contact_overflow"
    return "failed"


def cell_main(args: argparse.Namespace) -> None:
    """The `--cell` entry point: one (config, batch), one JSON file, one process."""
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_cell(
            args.cell_config,
            args.cell_batch_size,
            args.iterations,
            args.physics_backend,
            args.naconmax_per_world,
            args.njmax,
            args.seed,
        )
    except Exception as exc:                 # noqa: BLE001 -- recorded, then re-raised
        import traceback

        message = f"{type(exc).__name__}: {exc}"
        result = {
            "config": args.cell_config,
            "game": Path(args.cell_config).stem,
            "batch_size": args.cell_batch_size,
            "iterations": args.iterations,
            "status": classify(message),
            "error": message,
        }
        result_path.write_text(json.dumps(result, indent=2))
        traceback.print_exc()
        raise SystemExit(1)
    result_path.write_text(json.dumps(result, indent=2))
    print(
        f"  ok: {result['seconds_per_iteration']:.3f} s/iter, "
        f"{result['decisions_per_second']:.3g} decisions/s, "
        f"+{result['compile_seconds']:.1f}s compile"
    )


# --------------------------------------------------------------------------- the sweep


def child_environment(preallocate: bool) -> dict:
    """Environment for one cell.

    JAX preallocates most of the GPU by default, which would make `peak_bytes_in_use`
    a statement about the allocator rather than about the batch, and would turn an
    out-of-memory into an allocator-pool error at an arbitrary threshold. Turning it
    off is what makes both numbers mean what the summary says they mean.
    """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if not preallocate:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return env


def launch_cell(args: argparse.Namespace, config_path: str, batch_size: int, out_root: Path) -> dict:
    """Run one cell as a subprocess and read back its JSON; never raises."""
    name = cell_name(config_path, batch_size)
    result_path = out_root / "cells" / f"{name}.json"
    if result_path.exists() and not args.overwrite:
        print(f"{name}: skip (already measured)")
        return json.loads(result_path.read_text())

    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--cell",
        "--cell-config", config_path,
        "--cell-batch-size", str(batch_size),
        "--result-file", str(result_path),
        "--iterations", str(args.iterations),
        "--physics-backend", args.physics_backend,
        "--seed", str(args.seed),
    ]
    if args.naconmax_per_world is not None:
        command += ["--naconmax-per-world", str(args.naconmax_per_world)]
    if args.njmax is not None:
        command += ["--njmax", str(args.njmax)]

    print(f"{name}: running {args.iterations} iterations", flush=True)
    started = time.monotonic()
    log_path = out_root / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w") as log:
            subprocess.run(
                command, cwd=REPO_ROOT, env=child_environment(args.preallocate),
                stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False,
            )
    except subprocess.TimeoutExpired:
        result = {"config": config_path, "game": Path(config_path).stem,
                  "batch_size": batch_size, "iterations": args.iterations,
                  "status": "timeout",
                  "error": f"exceeded --timeout ({args.timeout}s)"}
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2))

    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        # Killed outright -- an OOM killer, or a Warp failure that took the process
        # down before the handler ran. The log is the only record, so point at it.
        result = {"config": config_path, "game": Path(config_path).stem,
                  "batch_size": batch_size, "iterations": args.iterations,
                  "status": "died", "error": f"no result written; see {log_path}"}
        result_path.write_text(json.dumps(result, indent=2))

    result["wall_seconds"] = time.monotonic() - started
    result["log"] = str(log_path)
    if result["status"] != "ok":
        print(f"  {result['status']}: {result.get('error', '')[:200]}")
    return result


def summarize(results: list[dict]) -> str:
    """One table per game: what each batch cost, and how much of it came back."""
    lines: list[str] = []
    games = sorted({row["game"] for row in results}, key=lambda g: [r["game"] for r in results].index(g))
    for game in games:
        rows = sorted((r for r in results if r["game"] == game), key=lambda r: r["batch_size"])
        ok = [r for r in rows if r["status"] == "ok"]
        baseline = ok[0] if ok else None

        lines.append(f"\n### {game}")
        if baseline is not None:
            lines.append(
                f"_{baseline['iterations']} iterations per cell, "
                f"backend {baseline.get('physics_backend')}, horizon {baseline.get('horizon')}; "
                f"scaling is relative to batch {baseline['batch_size']}._"
            )
        lines.append("")
        lines.append("| batch | s/iter | decisions/s | scaling | compile s | peak GiB | status |")
        lines.append("|------:|-------:|------------:|--------:|----------:|---------:|:-------|")
        for row in rows:
            if row["status"] != "ok":
                lines.append(f"| {row['batch_size']} | | | | | | **{row['status']}** |")
                continue
            scaling = ""
            if baseline is not None and baseline["decisions_per_second"]:
                relative_throughput = row["decisions_per_second"] / baseline["decisions_per_second"]
                relative_batch = row["batch_size"] / baseline["batch_size"]
                scaling = f"{relative_throughput / relative_batch:.2f}"
            peak = row.get("peak_bytes")
            # Blank rather than 0.00 where the backend reports no memory stats (CPU).
            peak_gib = f"{peak / 2**30:.2f}" if peak else ""
            lines.append(
                f"| {row['batch_size']} "
                f"| {row['seconds_per_iteration']:.3f} "
                f"| {row['decisions_per_second']:.3g} "
                f"| {scaling} "
                f"| {row['compile_seconds']:.1f} "
                f"| {peak_gib} "
                f"| ok |"
            )
    lines.append("")
    lines.append("`scaling` is throughput per unit of batch, against the smallest batch that ran:")
    lines.append("1.00 means the larger batch was free, 0.50 means half of it was paid in wall time.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    ap.add_argument("--iterations", type=int, default=100,
                    help="training iterations per cell, timed twice (compile, then steady state)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--physics-backend", default="auto", choices=("config", "auto", "jax", "warp"),
                    help="'config' keeps each YAML's own setting; 'auto' is Warp on an NVIDIA GPU")
    ap.add_argument("--naconmax-per-world", type=int, default=None,
                    help="Warp contact slots per world; the sweep multiplies this by the batch "
                         f"(defaults per game: {NACONMAX_PER_WORLD})")
    ap.add_argument("--njmax", type=int, default=None, help="Warp constraint rows per world")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per cell before it is killed")
    ap.add_argument("--preallocate", action="store_true",
                    help="leave JAX's GPU preallocation on (makes the memory column meaningless)")
    ap.add_argument("--no-stop-on-failure", dest="stop_on_failure", action="store_false",
                    help="keep trying larger batches after one fails, instead of skipping them")
    ap.add_argument("--overwrite", action="store_true", help="re-run cells that already have a result")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", metavar="RESULTS_JSON",
                    help="reprint the summary from a finished run and exit")

    # `--cell` and its arguments: one (config, batch), run by the sweep as a subprocess.
    ap.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cell-config", help=argparse.SUPPRESS)
    ap.add_argument("--cell-batch-size", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--result-file", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.summarize:
        print(summarize(json.loads(Path(args.summarize).read_text())["results"]))
        return

    if args.cell:
        cell_main(args)
        return

    batch_sizes = sorted(args.batch_sizes)
    if args.dry_run:
        print(f"{len(args.configs) * len(batch_sizes)} cells, {args.iterations} iterations each:")
        for config_path in args.configs:
            print(f"  {config_path}: batches {batch_sizes}")
        return

    out_root = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for config_path in args.configs:
        for batch_size in batch_sizes:
            result = launch_cell(args, config_path, batch_size, out_root)
            results.append(result)
            if result["status"] != "ok" and args.stop_on_failure:
                # Bigger is strictly worse on every axis this measures, so the rest
                # of this config's ladder is a foregone conclusion.
                for skipped in [b for b in batch_sizes if b > batch_size]:
                    results.append({"config": config_path, "game": Path(config_path).stem,
                                    "batch_size": skipped, "iterations": args.iterations,
                                    "status": "skipped",
                                    "error": f"batch {batch_size} already {result['status']}"})
                print(f"  skipping larger batches for {Path(config_path).stem}")
                break

    payload = {"args": vars(args), "results": results,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_root / "results.json").write_text(json.dumps(payload, indent=2))
    summary = summarize(results)
    (out_root / "summary.md").write_text(summary + "\n")
    print(summary)
    print(f"\nwrote {out_root / 'results.json'} and {out_root / 'summary.md'}")


if __name__ == "__main__":
    main()
