"""Generate SLURM shell scripts for experiments/one_shot_neural.

Two experiments, selected with ``--experiment``:

``main`` (default)
    The eval-matched grid: one job per (game, method) across the chosen seeds, via
    ``run_all.py``, every method given the same ``--budget`` in payoff evaluations.

``walltime``
    The wall-time-matched re-run of the cheap zeroth-order methods, via
    ``run_walltime_matched.py``. There is deliberately **no ``--budget``** here: each
    job derives its own, per (game, method), from the reference tree's measured
    throughput so that the run lands at the ``--match`` method's wall-time times
    ``--headroom``. Jobs land in a separate output tree and never write the reference.

Each writes its own ``scripts/<name>/`` directory with a companion ``run_all.sh`` that
sbatches every generated job.

Edit the defaults in ``main()``, or override them on the CLI:

    python rci_scripts/generate_one_shot_neural.py
    python rci_scripts/generate_one_shot_neural.py \\
        --games configs/two_point.yaml \\
        --seeds 0 1 2 \\
        --budget 2000000 \\
        --out data/one_shot_neural

    # the wall-time-matched follow-up (spg / jpspg only)
    python rci_scripts/generate_one_shot_neural.py --experiment walltime
    bash scripts/one_shot_neural_walltime/run_all.sh

The ``walltime`` scripts read ``--reference`` at *run* time to calibrate, so that tree
must already hold finished ``--match`` and pseudo-gradient runs on the cluster. Check
what the budgets will be before submitting:

    python experiments/one_shot_neural/run_walltime_matched.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import prepare_default_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "one_shot_neural"
SCRIPTS_DIR_WALLTIME = REPO_ROOT / "scripts" / "one_shot_neural_walltime"
RUN_ALL = "experiments/one_shot_neural/run_all.py"
RUN_WALLTIME = "experiments/one_shot_neural/run_walltime_matched.py"

DEFAULT_GAMES = (
    "configs/two_point.yaml",
    # "configs/multi_point.yaml",
    # "configs/idealized_decoy_well.yaml",
    "configs/all_pay_auction.yaml",
    "configs/circle.yaml",
    "configs/glicksberg_gross.yaml",
)
DEFAULT_METHODS = (
    "mixture",
    "mmd_discrete",
    "nfsp",
    "psro",
    "spg",
    "jpspg",
    # "sisa",
)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_BUDGET = 20_000_000
DEFAULT_OUT = "data/one_shot_neural"
DEFAULT_MAX_PARALLEL = 1
DEFAULT_TIME_H = 4
DEFAULT_MEMORY_G = 16
DEFAULT_GPU = False

# --- walltime experiment -----------------------------------------------------------
# Only the methods cheap enough per evaluation to be worth re-running for longer; the
# whole point is that these finished ~20x faster than the mixture at an equal budget.
WALLTIME_METHODS = ("spg", "jpspg")
WALLTIME_OUT = "data/one_shot_neural_walltime"
WALLTIME_REFERENCE = "data/one_shot_neural"
WALLTIME_MATCH = "mixture"
WALLTIME_HEADROOM = 1.1
# One cell now costs about what the matched method costs (~950 s on these games), so a
# job of `len(seeds)` cells at --max-parallel 1 is roughly 3 x that plus compile.
WALLTIME_TIME_H = 4


def job_name(game: str, method: str) -> str:
    return f"{Path(game).stem}__{method}"


def write_job_script(
    path: Path,
    *,
    game: str,
    method: str,
    seeds: list[int],
    budget: int,
    out: str,
    max_parallel: int,
    time_h: int,
    memory_g: int,
    gpu: bool,
) -> None:
    header = prepare_default_script(time_h, memory_g, gpu)
    seeds_str = " ".join(str(s) for s in seeds)
    body = f"""
python {RUN_ALL} \\
  --games {game} \\
  --methods {method} \\
  --seeds {seeds_str} \\
  --budget {budget} \\
  --out {out} \\
  --max-parallel {max_parallel}
"""
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_walltime_job_script(
    path: Path,
    *,
    game: str,
    method: str,
    seeds: list[int],
    out: str,
    reference: str,
    match: str,
    headroom: float,
    max_parallel: int,
    time_h: int,
    memory_g: int,
    gpu: bool,
) -> None:
    """One (game, method) cell of the wall-time-matched re-run.

    No `--budget`: `run_walltime_matched.py` derives it from `--reference` so the run
    lands at `match`'s measured wall-time. `--max-parallel` must match whatever the
    reference was measured under, or the wall-times being equalized are not comparable.
    """
    header = prepare_default_script(time_h, memory_g, gpu)
    seeds_str = " ".join(str(s) for s in seeds)
    body = f"""
python {RUN_WALLTIME} \\
  --games {game} \\
  --methods {method} \\
  --seeds {seeds_str} \\
  --reference {reference} \\
  --match {match} \\
  --headroom {headroom} \\
  --out {out} \\
  --max-parallel {max_parallel}
"""
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_submit_all(path: Path, job_scripts: list[Path], log_dir: str,
                     preamble: list[str] = ()) -> None:
    lines = ["#!/bin/sh", ""]
    lines += list(preamble)
    lines.append(f"mkdir -p {log_dir}")
    lines.append("")
    for script in job_scripts:
        name = script.stem
        rel = script.relative_to(REPO_ROOT).as_posix()
        lines.append(f"sbatch -o {log_dir}/{name}.log {rel}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def generate(
    games: list[str],
    methods: list[str],
    seeds: list[int],
    budget: int,
    out: str,
    max_parallel: int,
    time_h: int,
    memory_g: int,
    gpu: bool,
    experiment: str = "main",
    reference: str = WALLTIME_REFERENCE,
    match: str = WALLTIME_MATCH,
    headroom: float = WALLTIME_HEADROOM,
) -> list[Path]:
    walltime = experiment == "walltime"
    scripts_dir = SCRIPTS_DIR_WALLTIME if walltime else SCRIPTS_DIR
    log_dir = f"logs/{scripts_dir.name}"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    job_scripts: list[Path] = []
    for game in games:
        for method in methods:
            path = scripts_dir / f"{job_name(game, method)}.sh"
            if walltime:
                write_walltime_job_script(
                    path,
                    game=game,
                    method=method,
                    seeds=seeds,
                    out=out,
                    reference=reference,
                    match=match,
                    headroom=headroom,
                    max_parallel=max_parallel,
                    time_h=time_h,
                    memory_g=memory_g,
                    gpu=gpu,
                )
            else:
                write_job_script(
                    path,
                    game=game,
                    method=method,
                    seeds=seeds,
                    budget=budget,
                    out=out,
                    max_parallel=max_parallel,
                    time_h=time_h,
                    memory_g=memory_g,
                    gpu=gpu,
                )
            job_scripts.append(path)

    preamble = []
    if walltime:
        # Each job calls `calibrate()` itself, at run time, against `reference`. Submitting
        # before that tree has finished runs gives "nothing to calibrate", not a wrong
        # budget -- but it is still a wasted round trip through the queue.
        preamble = [
            f"# Each job derives its own budget from {reference} when it starts, so that",
            f"# tree must already hold FINISHED {match} and {'/'.join(methods)} runs.",
            "# Check what the budgets will be before submitting:",
            f"#   python {RUN_WALLTIME} --dry-run --reference {reference} --match {match}",
            "",
        ]

    submit_all = scripts_dir / "run_all.sh"
    write_submit_all(submit_all, job_scripts, log_dir, preamble)
    return job_scripts + [submit_all]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=("main", "walltime"), default="main",
                    help="'main': the eval-matched run_all.py grid. 'walltime': the "
                         "wall-time-matched spg/jpspg re-run (no --budget; it is derived "
                         "per cell from --reference)")
    ap.add_argument("--games", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="root directory for checkpoints (passed through to run_all.py)")
    ap.add_argument("--max-parallel", type=int, default=None,
                    help="parallel seeds within one SLURM job")
    ap.add_argument("--time", type=int, default=None, dest="time_h",
                    help="wall-time hours for #SBATCH --time")
    ap.add_argument("--memory", type=int, default=None, dest="memory_g",
                    help="memory in GB for #SBATCH --mem")
    ap.add_argument("--gpu", action="store_true", default=None,
                    help="request one GPU (changes the partition)")
    walltime_group = ap.add_argument_group("walltime experiment")
    walltime_group.add_argument("--reference", default=None,
                                help="tree to calibrate the budget from; read, never written")
    walltime_group.add_argument("--match", default=None,
                                help="method whose wall-time is the target")
    walltime_group.add_argument("--headroom", type=float, default=None,
                                help="multiply the target wall-time by this")
    args = ap.parse_args()

    # Defaults to edit when generating a specific batch by hand. The walltime experiment
    # overrides the ones the main grid's values would be wrong for: it re-runs only the
    # cheap methods, into its own tree, and derives its budget rather than taking one.
    walltime = args.experiment == "walltime"
    games = list(DEFAULT_GAMES)
    methods = list(WALLTIME_METHODS if walltime else DEFAULT_METHODS)
    seeds = list(DEFAULT_SEEDS)
    budget = DEFAULT_BUDGET
    out = WALLTIME_OUT if walltime else DEFAULT_OUT
    max_parallel = DEFAULT_MAX_PARALLEL
    time_h = WALLTIME_TIME_H if walltime else DEFAULT_TIME_H
    memory_g = DEFAULT_MEMORY_G
    gpu = DEFAULT_GPU
    reference = WALLTIME_REFERENCE
    match = WALLTIME_MATCH
    headroom = WALLTIME_HEADROOM

    if args.games is not None:
        games = args.games
    if args.methods is not None:
        methods = args.methods
    if args.seeds is not None:
        seeds = args.seeds
    if args.budget is not None:
        budget = args.budget
    if args.out is not None:
        out = args.out
    if args.max_parallel is not None:
        max_parallel = args.max_parallel
    if args.time_h is not None:
        time_h = args.time_h
    if args.memory_g is not None:
        memory_g = args.memory_g
    if args.gpu is not None:
        gpu = args.gpu
    if args.reference is not None:
        reference = args.reference
    if args.match is not None:
        match = args.match
    if args.headroom is not None:
        headroom = args.headroom

    if walltime and args.budget is not None:
        print("note: --budget is ignored for --experiment walltime; each cell derives "
              "its own from --reference")

    written = generate(
        games=games,
        methods=methods,
        seeds=seeds,
        budget=budget,
        out=out,
        max_parallel=max_parallel,
        time_h=time_h,
        memory_g=memory_g,
        gpu=gpu,
        experiment=args.experiment,
        reference=reference,
        match=match,
        headroom=headroom,
    )
    scripts_dir = SCRIPTS_DIR_WALLTIME if walltime else SCRIPTS_DIR
    print(f"wrote {len(written) - 1} job scripts + run_all.sh under {scripts_dir}")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
