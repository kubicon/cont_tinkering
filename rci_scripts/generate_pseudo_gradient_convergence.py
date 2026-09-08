"""Generate SLURM scripts for experiments/pseudo_gradient_convergence.

Each job runs one stage of ``run.py`` (baseline / sigma / samples / dynamics)
for one game. A companion ``run_all.sh`` submits them in order — stages 2–4
default to ``--score-between`` so they can pick up the best knobs from earlier
runs already on disk.

    python rci_scripts/generate_pseudo_gradient_convergence.py
    python rci_scripts/generate_pseudo_gradient_convergence.py \\
        --games configs/matching_pennies.yaml configs/two_point.yaml \\
        --budget 20000000
    bash scripts/pseudo_gradient_convergence/run_all.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import prepare_default_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "pseudo_gradient_convergence"
RUN = "experiments/pseudo_gradient_convergence/run.py"

DEFAULT_GAMES = (
    "configs/matching_pennies.yaml",
    "configs/two_point.yaml",
    "configs/circle.yaml",
    "configs/glicksberg_gross.yaml",
    "configs/all_pay_auction.yaml",
)
DEFAULT_STAGES = ("1", "2", "3", "4")
DEFAULT_SEEDS = (0,)
DEFAULT_BUDGET = 200_000_000
DEFAULT_OUT = "data/pseudo_gradient_convergence"
DEFAULT_TIME_H = 4
DEFAULT_MEMORY_G = 16
DEFAULT_GPU = False


def job_name(game: str, stage: str) -> str:
    return f"{Path(game).stem}__stage{stage}"


def write_job_script(
    path: Path,
    *,
    game: str,
    stage: str,
    seed: int,
    budget: int,
    out: str,
    time_h: int,
    memory_g: int,
    gpu: bool,
) -> None:
    header = prepare_default_script(time_h, memory_g, gpu)
    score_between = "" if stage == "1" else " \\\n  --score-between"
    body = f"""
python {RUN} \\
  --game {game} \\
  --stages {stage} \\
  --seed {seed} \\
  --budget {budget} \\
  --out {out}{score_between}

python {RUN} --report --out {out} --game {game}
"""
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_submit_all(path: Path, job_scripts: list[Path], games: list[str]) -> None:
    """Chain stages per game with ``--dependency=afterok`` so ``--score-between`` works."""
    lines = ["#!/bin/sh", ""]
    log_dir = "logs/pseudo_gradient_convergence"
    lines.append(f"mkdir -p {log_dir}")
    lines.append("")
    by_game: dict[str, list[Path]] = {Path(g).stem: [] for g in games}
    for script in job_scripts:
        game_stem = script.stem.rsplit("__stage", 1)[0]
        by_game.setdefault(game_stem, []).append(script)
    for game_stem, scripts in by_game.items():
        lines.append(f"# {game_stem}: stage chain")
        lines.append("prev=")
        for script in scripts:
            name = script.stem
            rel = script.relative_to(REPO_ROOT).as_posix()
            lines.append(
                f'if [ -z "$prev" ]; then\n'
                f'  prev=$(sbatch --parsable -o {log_dir}/{name}.log {rel})\n'
                f'else\n'
                f'  prev=$(sbatch --parsable --dependency=afterok:$prev '
                f'-o {log_dir}/{name}.log {rel})\n'
                f'fi'
            )
        lines.append(f'echo "{game_stem} chain ends at job $prev"')
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def generate(
    games: list[str],
    stages: list[str],
    seed: int,
    budget: int,
    out: str,
    time_h: int,
    memory_g: int,
    gpu: bool,
) -> list[Path]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    job_scripts: list[Path] = []
    for game in games:
        for stage in stages:
            name = job_name(game, stage)
            path = SCRIPTS_DIR / f"{name}.sh"
            write_job_script(
                path,
                game=game,
                stage=stage,
                seed=seed,
                budget=budget,
                out=out,
                time_h=time_h,
                memory_g=memory_g,
                gpu=gpu,
            )
            job_scripts.append(path)

    submit_all = SCRIPTS_DIR / "run_all.sh"
    write_submit_all(submit_all, job_scripts, games)
    return job_scripts + [submit_all]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES))
    ap.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES),
                    choices=["1", "2", "3", "4"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--time", type=int, default=DEFAULT_TIME_H, dest="time_h")
    ap.add_argument("--memory", type=int, default=DEFAULT_MEMORY_G, dest="memory_g")
    ap.add_argument("--gpu", action="store_true", default=DEFAULT_GPU)
    args = ap.parse_args()

    written = generate(
        games=args.games,
        stages=args.stages,
        seed=args.seed,
        budget=args.budget,
        out=args.out,
        time_h=args.time_h,
        memory_g=args.memory_g,
        gpu=args.gpu,
    )
    print(f"wrote {len(written) - 1} job scripts + run_all.sh under {SCRIPTS_DIR}")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
