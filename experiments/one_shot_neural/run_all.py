"""Run the (game x method x seed) grid of neural baselines, several cells at a time.

    python experiments/one_shot_neural/run_all.py --max-parallel 4
    python experiments/one_shot_neural/run_all.py --methods psro --games configs/two_point.yaml
    python experiments/one_shot_neural/run_all.py --dry-run

Each cell is a separate `run_cell.py` **process**, which is what makes the parallelism
useful and the wall-time measurements meaningful: one JAX process per cell, with its
thread pool capped so `--max-parallel` cells do not fight over the same cores (a cell
that is descheduled by three other cells is not a cell whose wall-time means anything).
Cells are independent, so a failure is logged and the rest continue, and a cell that
already has a `meta.json` is skipped -- add a method or a game later and only the new
cells run.

Everything is written under `--out` in the layout `run_cell.py` documents; nothing here
computes exploitability. Score the whole tree afterwards with `score.py`.

Running one method at a time is the normal way to use this: `--methods psro` (or
`--games`, or `--seeds`) filters the grid, and repeated invocations accumulate into the
same output tree.

`--methods` defaults to `run_cell.METHODS`, the grid. `run_cell.OPTIONAL_METHODS` (`spg`,
`jpspg`) are registered and runnable but not in it, so they have to be named:

    python experiments/one_shot_neural/run_all.py --methods spg jpspg

A cell is identified by its directory, `<out>/<game>/<method>/seed<N>`, so running one
method twice at different settings would be running the same cell twice. `--label`
renames that middle level, which is how the capacity sweep keeps one K per cell:

    python experiments/one_shot_neural/run_all.py --methods mixture \\
        --set num_components=8 --label mixture__k8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repo root
sys.path.insert(0, str(HERE))              # this directory, for `run_cell`

from run_cell import ALL_METHODS, METHODS  # noqa: E402 -- same directory, added to sys.path below

# The one-shot games with genuinely mixed equilibria, in two groups.
#
# The first three have a Nash on *finitely many* points, so a K-component mixture can
# represent it exactly and what is being measured is whether the algorithm finds it;
# `idealized_decoy_well` is the counterexample config, the interesting cell there.
#
# The last three have a Nash the mixture head cannot represent exactly at all: uniform
# on an interval (the all-pay auction), a density unbounded at 0 (Glicksberg-Gross), or
# one needing more atoms than the kernel has harmonics (the circle). Their residual
# exploitability is a statement about the representation rather than the optimizer,
# which is the axis the finite-support games cannot measure.
DEFAULT_GAMES = (
    "configs/two_point.yaml",
    "configs/multi_point.yaml",
    "configs/idealized_decoy_well.yaml",
    "configs/all_pay_auction.yaml",
    "configs/circle.yaml",
    "configs/glicksberg_gross.yaml",
)

RUN_CELL = HERE / "run_cell.py"


def cell_name(game: str, label: str, seed: int) -> str:
    return f"{Path(game).stem}__{label}__seed{seed}"


def is_done(out_root: Path, game: str, label: str, seed: int) -> bool:
    return (out_root / Path(game).stem / label / f"seed{seed}" / "meta.json").exists()


def child_environment(threads: int) -> dict:
    """Environment for one cell: capped threads, so parallel cells do not oversubscribe.

    JAX on CPU will otherwise take every core for each process, and `--max-parallel 4`
    then measures contention rather than the methods.
    """
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") +
                        f" --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={threads}"
                        ).strip()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES))
    # Defaults to the grid; `OPTIONAL_METHODS` (spg, jpspg) have to be asked for by name.
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(ALL_METHODS),
                    metavar="METHOD")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=2_000_000,
                    help="payoff evaluations per cell -- the common currency; see run_cell.py")
    ap.add_argument("--out", default="data/one_shot_neural")
    ap.add_argument("--label", default=None,
                    help="directory name for these cells under <out>/<game>/, instead of "
                         "the method's own. One method per invocation, since the label "
                         "names the settings as well: --methods mixture --set "
                         "num_components=8 --label mixture__k8")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--threads-per-cell", type=int, default=0,
                    help="0 divides the machine's cores evenly among the parallel cells")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score", action="store_true",
                    help="score during the runs (slow; normally left to score.py)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="forward a run_cell.py setting, e.g. --set bins=101 --set atoms=4")
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.label and len(args.methods) != 1:
        raise SystemExit("--label names one method's settings; pass a single --methods "
                         f"(got {args.methods})")
    label = args.label
    extra: list[str] = []
    for item in args.set:
        key, _, value = item.partition("=")
        extra += [f"--{key.replace('_', '-')}", value]

    cells = [(game, method, seed)
             for game in args.games for method in args.methods for seed in args.seeds]
    pending = [c for c in cells
               if args.overwrite or not is_done(out_root, c[0], label or c[1], c[2])]

    threads = args.threads_per_cell or max(1, (os.cpu_count() or 4) // max(args.max_parallel, 1))
    print(f"{len(cells)} cells, {len(pending)} to run, {args.max_parallel} at a time "
          f"({threads} threads each), budget {args.budget:.3g} payoff evals -> {out_root}")
    if args.dry_run:
        for game, method, seed in pending:
            print(f"  {cell_name(game, label or method, seed)}")
        return

    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = child_environment(threads)
    started_at = time.monotonic()
    running: list[tuple] = []
    queue = list(pending)
    finished: list[tuple[str, int]] = []

    while queue or running:
        while queue and len(running) < args.max_parallel:
            game, method, seed = queue.pop(0)
            name = cell_name(game, label or method, seed)
            command = [sys.executable, str(RUN_CELL), "--game", game, "--method", method,
                       "--seed", str(seed), "--budget", str(args.budget), "--out", str(out_root)]
            if label:
                command += ["--label", label]
            if args.overwrite:
                command.append("--overwrite")
            if args.score:
                command.append("--score")
            command += extra
            handle = open(logs / f"{name}.log", "w")
            print(f"  [{time.monotonic() - started_at:7.1f}s] start {name}")
            running.append((name, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                                   env=env), handle))
        time.sleep(1.0)
        for entry in list(running):
            name, process, handle = entry
            if process.poll() is None:
                continue
            handle.close()
            running.remove(entry)
            finished.append((name, process.returncode))
            status = "ok" if process.returncode == 0 else f"FAILED ({process.returncode})"
            print(f"  [{time.monotonic() - started_at:7.1f}s] {status:16s} {name}")

    failed = [name for name, code in finished if code != 0]
    print(f"\n{len(finished)} cells in {time.monotonic() - started_at:.1f}s"
          f"{f', {len(failed)} failed: {failed}' if failed else ''}")
    print(f"logs -> {logs}\nnext: python {HERE / 'score.py'} --out {out_root}")


if __name__ == "__main__":
    main()
