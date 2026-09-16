"""Generate SLURM scripts for the multi-seed magnet grid (experiments/failing_gaussian_wo_magnet/rerun).

The grid is 3 domains x {magnet, no magnet} x 3 engines = 18 *cells*, each one YAML in
that directory (`<domain>_<magnet|nomagnet>_<engine>.yaml`). This script writes **one
training job per cell**, and that job runs every seed of the cell via `run_cell.py`:

    scripts/failing_gaussian_rerun/
        <cell>.sh              train all seeds of one cell (the `idealized` cells run one seed)
        score_<cell>.sh        score every seed's checkpoints of that cell (plot.py --no-plots)
        run_all.sh             sbatch every training job
        run_all_score.sh       sbatch every scoring job -- after training has finished
        plot.sh                pool the saved scores into the plots; run it in the login shell

Hyperparameters are not set here: the cell YAMLs are the source of truth, and
`run_cell.py` only replaces `train.seed` / `train.checkpoint_dir` per seed. What lives
here is the cluster side -- which cells, which seeds, how many at once, and the time
each job needs (`SEED_HOURS` x seeds). **Every job uses one CPU** and runs its seeds
sequentially; parallelism comes only from SLURM running many jobs at once.

    python rci_scripts/generate_failing_gaussian_rerun.py
    python rci_scripts/generate_failing_gaussian_rerun.py --cells mp_magnet_ppo mp_nomagnet_ppo --seeds 0 1 2
    bash scripts/failing_gaussian_rerun/run_all.sh
    bash scripts/failing_gaussian_rerun/run_all_score.sh    # once training is done
    bash scripts/failing_gaussian_rerun/plot.sh
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from utils import prepare_default_script

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "failing_gaussian_wo_magnet" / "rerun"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "failing_gaussian_rerun"
LOG_DIR = "logs/failing_gaussian_rerun"
RUN_CELL = "experiments/failing_gaussian_wo_magnet/rerun/run_cell.py"
PLOT = "experiments/failing_gaussian_wo_magnet/rerun/plot.py"

DEFAULT_OUT = "data/failing_gaussian_rerun_seeds"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_MEMORY_G = 16
DEFAULT_GPU = False

# Every job gets one CPU and runs its seeds one after another, so a job's wall-time is
# (hours per seed) x (number of seeds). Hours per seed on one core, estimated from the
# single-seed local run of this grid at 100k iterations (4 cells sharing an 8-core
# machine, i.e. ~2 cores each): rot3/rot2/mp idealized ~8.5/6.8/2.8 h, sampled
# ~2.4/2.0/0.1 h, ppo ~1.2/0.7/0.2 h -- doubled for one core, then halved for the
# current 50k (`train.steps: 100`). The job requests TIME_SAFETY times that, capped at 72.
SEED_HOURS: dict[tuple[str, str], float] = {
    ("rot3", "idealized"): 8.5,  ("rot2", "idealized"): 7.0,  ("mp", "idealized"): 3.0,
    ("rot3", "sampled"): 2.5,    ("rot2", "sampled"): 2.0,    ("mp", "sampled"): 0.25,
    ("rot3", "ppo"): 1.25,       ("rot2", "ppo"): 0.75,       ("mp", "ppo"): 0.25,
}
TIME_SAFETY = 1.4
MAX_TIME_H = 72
# Scoring every checkpoint of every seed, also on one CPU; the spread measure's payoff
# matrix makes the rotation games the slow ones.
SCORE_TIME_H: dict[str, int] = {"rot3": 12, "rot2": 8, "mp": 4}
SCORE_MEMORY_G = 16

# Slowest first, so the queue starts the long jobs early.
DOMAIN_ORDER = ("rot3", "rot2", "mp")
ENGINE_ORDER = ("idealized", "sampled", "ppo")


def all_cells() -> list[str]:
    cells = [p.stem for p in EXPERIMENT_DIR.glob("*.yaml")]
    return sorted(cells, key=lambda c: (DOMAIN_ORDER.index(c.split("_")[0]),
                                        ENGINE_ORDER.index(c.split("_")[2]), c))


def cell_seeds(cell: str, seeds: list[int]) -> list[int]:
    """`idealized` has no sampling, so its seeds would all be one run."""
    return seeds[:1] if cell.endswith("_idealized") else seeds


def train_hours(cell: str, n_seeds: int) -> int:
    domain, _, engine = cell.split("_")
    hours = math.ceil(SEED_HOURS[(domain, engine)] * n_seeds * TIME_SAFETY)
    if hours > MAX_TIME_H:
        raise SystemExit(f"{cell}: {n_seeds} sequential seeds need ~{hours} h, over SLURM's "
                         f"{MAX_TIME_H} h; pass fewer --seeds or lower SEED_HOURS")
    return hours


def header(time_h: int, memory_g: int, gpu: bool) -> str:
    base = prepare_default_script(time_h, memory_g, gpu)
    first, rest = base.split("\n", 1)
    return f"{first}\n#SBATCH --cpus-per-task=1\n{rest}"


def write_script(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | 0o111)


def write_train_job(path: Path, cell: str, seeds: list[int], out: str, memory_g: int,
                    gpu: bool) -> None:
    seeds_str = " ".join(str(s) for s in seeds)
    body = f"""
export PYTHONUNBUFFERED=1
python {RUN_CELL} {cell} \\
  --seeds {seeds_str} \\
  --out {out}
"""
    write_script(path, header(train_hours(cell, len(seeds)), memory_g, gpu) + body)


def write_score_job(path: Path, cell: str, out: str, no_std: bool) -> None:
    domain = cell.split("_")[0]
    flags = " --no-std" if no_std else ""
    body = f"""
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
python {PLOT} --out {out} --cells {cell} --no-plots{flags}
"""
    write_script(path, header(SCORE_TIME_H[domain], SCORE_MEMORY_G, False) + body)


def write_submit_all(path: Path, jobs: list[Path], note: str = "") -> None:
    lines = ["#!/bin/sh", ""]
    if note:
        lines += [f"# {note}", ""]
    lines += [f"mkdir -p {LOG_DIR}", ""]
    for job in jobs:
        lines.append(f"sbatch -o {LOG_DIR}/{job.stem}.log {job.relative_to(REPO_ROOT).as_posix()}")
    lines += ["", f'echo "submitted {len(jobs)} jobs; logs in {LOG_DIR}"']
    write_script(path, "\n".join(lines) + "\n")


def write_plot(path: Path, out: str, no_std: bool) -> None:
    flags = " --no-std" if no_std else ""
    lines = [
        "#!/bin/sh",
        "",
        "# Run after run_all_score.sh's jobs have finished; it only loads the saved scores.",
        f"python {PLOT} --out {out}{flags}",
        "",
    ]
    write_script(path, "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=None, choices=all_cells(), metavar="CELL",
                    help="subset of the grid (default: all 18)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--out", default=DEFAULT_OUT, help="checkpoint tree, passed to run_cell.py")
    ap.add_argument("--memory", type=int, default=DEFAULT_MEMORY_G, dest="memory_g")
    ap.add_argument("--gpu", action="store_true", default=DEFAULT_GPU)
    ap.add_argument("--no-std", action="store_true",
                    help="score jobs and plot.sh use the means-only measure")
    args = ap.parse_args()

    cells = args.cells or all_cells()
    train_jobs, score_jobs = [], []
    for cell in cells:
        path = SCRIPTS_DIR / f"{cell}.sh"
        write_train_job(path, cell, cell_seeds(cell, args.seeds), args.out,
                        args.memory_g, args.gpu)
        train_jobs.append(path)
        path = SCRIPTS_DIR / f"score_{cell}.sh"
        write_score_job(path, cell, args.out, args.no_std)
        score_jobs.append(path)

    write_submit_all(SCRIPTS_DIR / "run_all.sh", train_jobs)
    write_submit_all(SCRIPTS_DIR / "run_all_score.sh", score_jobs,
                     note="Submit only after every run_all.sh job has finished.")
    write_plot(SCRIPTS_DIR / "plot.sh", args.out, args.no_std)

    rel = SCRIPTS_DIR.relative_to(REPO_ROOT)
    print(f"wrote {len(train_jobs)} training + {len(score_jobs)} scoring jobs under {rel}")
    for cell in cells:
        seeds = cell_seeds(cell, args.seeds)
        print(f"  {cell:26} seeds {seeds}  ({train_hours(cell, len(seeds))} h)")
    print(f"\n1. bash {rel}/run_all.sh\n2. bash {rel}/run_all_score.sh\n3. bash {rel}/plot.sh")


if __name__ == "__main__":
    main()
