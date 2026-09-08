"""Generate per-run configs + SLURM scripts for a num_components sweep.

For each (game, num_components, seed) this writes a standalone train.py YAML
under ``configs/num_components/`` and a matching SLURM train job under
``scripts/num_components/``. Scoring is one SLURM job **per game**
(``score__{game}.sh``): that job scores every run of the game and writes the
per-game plot. Companions ``run_all.sh`` / ``run_all_score.sh`` submit the
train and score jobs with sbatch.

Checkpoints land under ``data/components_sweep/``; scoring writes
``exploitability.pkl`` beside them and PNGs under
``data/components_sweep/plots/`` via ``rci_scripts/score_components_sweep.py``.

Base hyperparameters are copied from the game's default config; only
``network.num_components``, ``train.seed``, and ``train.checkpoint_dir`` change.

Edit the defaults in ``main()``, or override them on the CLI:

    python rci_scripts/generate_components_sweep.py
    python rci_scripts/generate_components_sweep.py \\
        --num-components 2 4 8 \\
        --seeds 0 1 2 \\
        --grid 1601 \\
        --time 8
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from utils import prepare_default_script

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "num_components"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "num_components"
CHECKPOINT_ROOT = "data/components_sweep"
LOG_DIR = "logs/num_components"
SCORE_SCRIPT = "rci_scripts/score_components_sweep.py"

DEFAULT_GAMES = (
    "configs/all_pay_auction.yaml",
    "configs/circle.yaml",
    "configs/glicksberg_gross.yaml",
)
DEFAULT_NUM_COMPONENTS = (1, 2, 3, 4, 6, 8, 12, 16)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TIME_H = 4
DEFAULT_SCORE_TIME_H = 24
DEFAULT_MEMORY_G = 16
DEFAULT_GPU = False
DEFAULT_GRID = 801
DEFAULT_SAMPLES = 4096


def run_name(game: str, num_components: int, seed: int) -> str:
    return f"{Path(game).stem}__nc{num_components}__seed{seed}"


def load_base_config(game: str) -> dict:
    path = REPO_ROOT / game
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if "sweep" in raw:
        raise ValueError(f"{game} is a sweep config; pass a plain train.py config instead")
    return raw


def write_run_config(
    path: Path,
    *,
    base: dict,
    num_components: int,
    seed: int,
    checkpoint_dir: str,
) -> None:
    merged = copy.deepcopy(base)
    merged.setdefault("network", {})["num_components"] = num_components
    train = merged.setdefault("train", {})
    train["seed"] = seed
    train["checkpoint_dir"] = checkpoint_dir
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def write_train_job_script(
    path: Path,
    *,
    config_rel: str,
    time_h: int,
    memory_g: int,
    gpu: bool,
) -> None:
    header = prepare_default_script(time_h, memory_g, gpu)
    body = f"\npython train.py {config_rel}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_score_job_script(
    path: Path,
    *,
    game: str,
    out_rel: str,
    configs_rel: str,
    grid: int,
    samples: int,
    time_h: int,
    memory_g: int,
    gpu: bool,
) -> None:
    header = prepare_default_script(time_h, memory_g, gpu)
    body = f"""
python {SCORE_SCRIPT} \\
  --game {game} \\
  --out {out_rel} \\
  --configs {configs_rel} \\
  --grid {grid} \\
  --samples {samples} \\
  --overwrite
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_submit_all(path: Path, job_scripts: list[Path], log_subdir: str = "") -> None:
    log_dir = f"{LOG_DIR}/{log_subdir}" if log_subdir else LOG_DIR
    lines = ["#!/bin/sh", "", f"mkdir -p {log_dir}", ""]
    for script in job_scripts:
        name = script.stem
        rel = script.relative_to(REPO_ROOT).as_posix()
        lines.append(f"sbatch -o {log_dir}/{name}.log {rel}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def generate(
    games: list[str],
    num_components: list[int],
    seeds: list[int],
    time_h: int,
    score_time_h: int,
    memory_g: int,
    gpu: bool,
    grid: int,
    samples: int,
) -> list[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    bases = {game: load_base_config(game) for game in games}
    train_scripts: list[Path] = []
    score_scripts: list[Path] = []
    written: list[Path] = []
    configs_rel = CONFIGS_DIR.relative_to(REPO_ROOT).as_posix()

    for game in games:
        game_stem = Path(game).stem
        for nc in num_components:
            for seed in seeds:
                name = run_name(game, nc, seed)
                config_path = CONFIGS_DIR / f"{name}.yaml"
                train_script_path = SCRIPTS_DIR / f"{name}.sh"
                checkpoint_dir = f"{CHECKPOINT_ROOT}/{name}"
                config_rel = config_path.relative_to(REPO_ROOT).as_posix()

                write_run_config(
                    config_path,
                    base=bases[game],
                    num_components=nc,
                    seed=seed,
                    checkpoint_dir=checkpoint_dir,
                )
                write_train_job_script(
                    train_script_path,
                    config_rel=config_rel,
                    time_h=time_h,
                    memory_g=memory_g,
                    gpu=gpu,
                )
                written.append(config_path)
                train_scripts.append(train_script_path)
                written.append(train_script_path)

        score_script_path = SCRIPTS_DIR / f"score__{game_stem}.sh"
        write_score_job_script(
            score_script_path,
            game=game_stem,
            out_rel=CHECKPOINT_ROOT,
            configs_rel=configs_rel,
            grid=grid,
            samples=samples,
            time_h=score_time_h,
            memory_g=memory_g,
            gpu=gpu,
        )
        score_scripts.append(score_script_path)
        written.append(score_script_path)

    submit_train = SCRIPTS_DIR / "run_all.sh"
    submit_score = SCRIPTS_DIR / "run_all_score.sh"
    write_submit_all(submit_train, train_scripts)
    write_submit_all(submit_score, score_scripts, log_subdir="score")
    written.append(submit_train)
    written.append(submit_score)
    return written


def main() -> None:
    games = list(DEFAULT_GAMES)
    num_components = list(DEFAULT_NUM_COMPONENTS)
    seeds = list(DEFAULT_SEEDS)
    time_h = DEFAULT_TIME_H
    score_time_h = DEFAULT_SCORE_TIME_H
    memory_g = DEFAULT_MEMORY_G
    gpu = DEFAULT_GPU
    grid = DEFAULT_GRID
    samples = DEFAULT_SAMPLES

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--games", nargs="+", default=None,
                    help="base train.py configs to copy hyperparameters from")
    ap.add_argument("--num-components", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--time", type=int, default=None, dest="time_h",
                    help="wall-time hours for training #SBATCH --time")
    ap.add_argument("--score-time", type=int, default=None, dest="score_time_h",
                    help="wall-time hours for per-game scoring #SBATCH --time")
    ap.add_argument("--memory", type=int, default=None, dest="memory_g",
                    help="memory in GB for #SBATCH --mem")
    ap.add_argument("--gpu", action="store_true", default=None,
                    help="request one GPU (changes the partition)")
    ap.add_argument("--grid", type=int, default=None,
                    help="GridOracle points per axis (passed to the score jobs)")
    ap.add_argument("--samples", type=int, default=None,
                    help="mixture samples per checkpoint (passed to the score jobs)")
    args = ap.parse_args()

    if args.games is not None:
        games = args.games
    if args.num_components is not None:
        num_components = args.num_components
    if args.seeds is not None:
        seeds = args.seeds
    if args.time_h is not None:
        time_h = args.time_h
    if args.score_time_h is not None:
        score_time_h = args.score_time_h
    if args.memory_g is not None:
        memory_g = args.memory_g
    if args.gpu is not None:
        gpu = args.gpu
    if args.grid is not None:
        grid = args.grid
    if args.samples is not None:
        samples = args.samples

    written = generate(
        games=games,
        num_components=num_components,
        seeds=seeds,
        time_h=time_h,
        score_time_h=score_time_h,
        memory_g=memory_g,
        gpu=gpu,
        grid=grid,
        samples=samples,
    )
    n_runs = len(games) * len(num_components) * len(seeds)
    print(
        f"wrote {n_runs} configs + {n_runs} train scripts + {len(games)} score scripts "
        f"+ run_all.sh + run_all_score.sh"
    )
    print(f"  configs: {CONFIGS_DIR.relative_to(REPO_ROOT)}")
    print(f"  scripts: {SCRIPTS_DIR.relative_to(REPO_ROOT)}")
    print(f"  checkpoints: {CHECKPOINT_ROOT}/{{run_name}}")
    print(f"  score via: {SCORE_SCRIPT} --game {{game}} --grid {grid} --samples {samples}")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
