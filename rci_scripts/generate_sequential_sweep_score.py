"""Generate SLURM scripts that score a sequential sweep offline.

Companion to ``generate_sequential_sweep.py``. That script writes one training
job per run under ``scripts/sequential_sweep/`` plus a ``manifest.json`` that
names every run. This one reads that manifest and writes **one scoring job per
run** -- each calls ``rci_scripts/score_sequential_sweep.py --run <name>`` so a
killed Leduc BR job takes nothing else with it -- plus ``run_all_score.sh`` to
submit them.

Kuhn scoring is exact and cheap; Leduc / sequential Blotto train an approximate
best response per checkpoint. BR budget, the final evaluation length, and (for
non-Kuhn) how many checkpoints to score are hyperparameters of *this*
generator (baked into every score script), not of the training sweep::

    python rci_scripts/generate_sequential_sweep.py
    python rci_scripts/generate_sequential_sweep_score.py
    python rci_scripts/generate_sequential_sweep_score.py \\
        --br-steps 100 --br-epochs 20 --episodes 50000 --n-checkpoints 10 --time 24
    bash scripts/sequential_sweep/run_all.sh          # train
    bash scripts/sequential_sweep/run_all_score.sh    # after (partial) training

Filter the same axes the training generator uses (``--games``, ``--solvers``,
``--seeds``) so you can rescore only Leduc, or only one seed, without rewriting
the training scripts. ``--dry-run`` prints the plan and writes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import prepare_default_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "sequential_sweep"
CONFIGS_DIR = REPO_ROOT / "configs" / "sequential_sweep"
CHECKPOINT_ROOT = "data/sequential_sweep"
LOG_DIR = "logs/sequential_sweep/score"
MANIFEST = SCRIPTS_DIR / "manifest.json"
SCORE = "rci_scripts/score_sequential_sweep.py"

DEFAULT_BR_STEPS = 200
DEFAULT_BR_EPOCHS = 100
DEFAULT_EPISODES = 20_000
DEFAULT_SEED = 0
DEFAULT_TIME_H = 24
DEFAULT_MEMORY_G = 16
DEFAULT_GPU = False


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(
            f"no {MANIFEST.relative_to(REPO_ROOT)}; run "
            "rci_scripts/generate_sequential_sweep.py first so the scoring "
            "jobs know which runs exist"
        )
    return json.loads(MANIFEST.read_text())


def filter_runs(
    runs: list[dict],
    *,
    games: list[str] | None,
    solvers: list[str] | None,
    seeds: list[int] | None,
) -> list[dict]:
    """Keep the runs that match the optional CLI filters.

    ``--games`` accepts either the path written in the manifest
    (``configs/kuhn_solvers.yaml``) or its stem (``kuhn_solvers``).
    """
    game_keys = None
    if games is not None:
        game_keys = {Path(game).as_posix() for game in games} | {
            Path(game).stem for game in games
        }
    out = []
    for run in runs:
        if game_keys is not None:
            game = run["game"]
            if game not in game_keys and Path(game).stem not in game_keys:
                continue
        if solvers is not None and run["solver"] not in solvers:
            continue
        if seeds is not None and int(run["seed"]) not in seeds:
            continue
        out.append(run)
    return out


def score_script_name(run_name: str) -> str:
    """Train jobs are ``{run}.sh``; score jobs must not collide with them."""
    return f"score__{run_name}.sh"


def write_score_job_script(
    path: Path,
    *,
    run_name: str,
    checkpoint_root: str,
    configs_rel: str,
    br_steps: int,
    br_epochs: int,
    episodes: int,
    exact_grid: int | None,
    seed: int,
    n_checkpoints: int | None,
    overwrite: bool,
    no_target: bool,
    time_h: int,
    memory_g: int,
    gpu: bool,
) -> None:
    header = prepare_default_script(time_h, memory_g, gpu)
    flags = [
        f"--run {run_name}",
        f"--out {checkpoint_root}",
        f"--configs {configs_rel}",
        f"--br-steps {br_steps}",
        f"--br-epochs {br_epochs}",
        f"--episodes {episodes}",
        f"--seed {seed}",
    ]
    if exact_grid is not None:
        flags.append(f"--exact-grid {exact_grid}")
    if n_checkpoints is not None:
        flags.append(f"--n-checkpoints {n_checkpoints}")
    if overwrite:
        flags.append("--overwrite")
    if no_target:
        flags.append("--no-target")
    joined = " \\\n  ".join(flags)
    body = f"\npython {SCORE} \\\n  {joined}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_submit_all(path: Path, job_scripts: list[Path]) -> None:
    lines = ["#!/bin/sh", "", f"mkdir -p {LOG_DIR}", ""]
    for script in job_scripts:
        rel = script.relative_to(REPO_ROOT).as_posix()
        lines.append(f"sbatch -o {LOG_DIR}/{script.stem}.log {rel}")
    lines.append("")
    lines.append(f'echo "submitted {len(job_scripts)} score jobs; logs in {LOG_DIR}"')
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def write_score_manifest(path: Path, *, checkpoint_root: str, runs: list[dict],
                         settings: dict) -> None:
    path.write_text(json.dumps({
        "checkpoint_root": checkpoint_root,
        "score_script": SCORE,
        "settings": settings,
        "runs": [{"name": run["name"], "game": run["game"], "solver": run["solver"],
                  "seed": run["seed"], "checkpoint_dir": run["checkpoint_dir"]}
                 for run in runs],
    }, indent=2) + "\n")


def generate(
    *,
    games: list[str] | None,
    solvers: list[str] | None,
    seeds: list[int] | None,
    br_steps: int,
    br_epochs: int,
    episodes: int,
    exact_grid: int | None,
    seed: int,
    n_checkpoints: int | None,
    overwrite: bool,
    no_target: bool,
    time_h: int,
    memory_g: int,
    gpu: bool,
    dry_run: bool = False,
) -> list[Path]:
    manifest = load_manifest()
    checkpoint_root = manifest.get("checkpoint_root", CHECKPOINT_ROOT)
    runs = filter_runs(
        manifest["runs"], games=games, solvers=solvers, seeds=seeds
    )
    if not runs:
        raise SystemExit(
            "no runs left after filtering; check --games / --solvers / --seeds "
            f"against {MANIFEST.relative_to(REPO_ROOT)}"
        )

    settings = {
        "br_steps": br_steps,
        "br_epochs": br_epochs,
        "episodes": episodes,
        "exact_grid": exact_grid,
        "seed": seed,
        "n_checkpoints": n_checkpoints,
        "overwrite": overwrite,
        "no_target": no_target,
        "time_h": time_h,
        "memory_g": memory_g,
        "gpu": gpu,
    }

    if dry_run:
        n_ck = (
            f"  n_checkpoints={n_checkpoints}" if n_checkpoints is not None else ""
        )
        print(f"{len(runs)} score jobs  BR {br_steps}x{br_epochs}  "
              f"eval {episodes} episodes  wall {time_h}h{n_ck}\n")
        for run in runs:
            print(f"  score__{run['name']}")
        return []

    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    configs_rel = CONFIGS_DIR.relative_to(REPO_ROOT).as_posix()
    written: list[Path] = []
    job_scripts: list[Path] = []

    for run in runs:
        script_path = SCRIPTS_DIR / score_script_name(run["name"])
        write_score_job_script(
            script_path,
            run_name=run["name"],
            checkpoint_root=checkpoint_root,
            configs_rel=configs_rel,
            br_steps=br_steps,
            br_epochs=br_epochs,
            episodes=episodes,
            exact_grid=exact_grid,
            seed=seed,
            n_checkpoints=n_checkpoints,
            overwrite=overwrite,
            no_target=no_target,
            time_h=time_h,
            memory_g=memory_g,
            gpu=gpu,
        )
        job_scripts.append(script_path)
        written.append(script_path)

    submit_all = SCRIPTS_DIR / "run_all_score.sh"
    write_submit_all(submit_all, job_scripts)
    score_manifest = SCRIPTS_DIR / "score_manifest.json"
    write_score_manifest(
        score_manifest,
        checkpoint_root=checkpoint_root,
        runs=runs,
        settings=settings,
    )
    return written + [submit_all, score_manifest]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--games", nargs="+", default=None,
        help="only score runs whose manifest game matches (path or stem); "
             "default: every run in the training manifest",
    )
    ap.add_argument(
        "--solvers", nargs="+", default=None,
        help="only score these solvers (sweep names, e.g. sac / self_play / nfsp)",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="only score these training seeds",
    )
    ap.add_argument(
        "--br-steps", type=int, default=DEFAULT_BR_STEPS,
        help="PPO chunks per approximate best response (non-Kuhn)",
    )
    ap.add_argument(
        "--br-epochs", type=int, default=DEFAULT_BR_EPOCHS,
        help="PPO iterations per BR chunk (non-Kuhn)",
    )
    ap.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help="hands for the final exploitability estimate after BR training",
    )
    ap.add_argument(
        "--exact-grid", type=int, default=None,
        help="bet-grid points for Kuhn's exact metric (default: the game's own)",
    )
    ap.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="RNG seed passed to score_sequential_sweep.py",
    )
    ap.add_argument(
        "--n-checkpoints", type=int, default=None,
        help="pass --n-checkpoints to score_sequential_sweep.py: for non-Kuhn "
             "runs, score at most this many linearly spaced checkpoints "
             "(always first+last; must be > 2). Kuhn ignores it. Default: all",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="pass --overwrite so existing exploitability.pkl files are recomputed",
    )
    ap.add_argument(
        "--no-target", action="store_true",
        help="skip Polyak-target scoring on Kuhn",
    )
    ap.add_argument(
        "--time", type=int, default=DEFAULT_TIME_H, dest="time_h",
        help="wall-time hours for #SBATCH --time (BR scoring is the long part)",
    )
    ap.add_argument(
        "--memory", type=int, default=DEFAULT_MEMORY_G, dest="memory_g",
        help="memory in GB for #SBATCH --mem",
    )
    ap.add_argument(
        "--gpu", action="store_true", default=DEFAULT_GPU,
        help="request one GPU (changes the partition)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the score-job plan and write nothing",
    )
    args = ap.parse_args()
    if args.n_checkpoints is not None and args.n_checkpoints <= 2:
        raise SystemExit(f"--n-checkpoints must be > 2, got {args.n_checkpoints}")

    written = generate(
        games=args.games,
        solvers=args.solvers,
        seeds=args.seeds,
        br_steps=args.br_steps,
        br_epochs=args.br_epochs,
        episodes=args.episodes,
        exact_grid=args.exact_grid,
        seed=args.seed,
        n_checkpoints=args.n_checkpoints,
        overwrite=args.overwrite,
        no_target=args.no_target,
        time_h=args.time_h,
        memory_g=args.memory_g,
        gpu=args.gpu,
        dry_run=args.dry_run,
    )
    if not written:
        return
    n_jobs = len(written) - 2
    print(
        f"wrote {n_jobs} score job scripts + run_all_score.sh + score_manifest.json"
    )
    print(f"  scripts : {SCRIPTS_DIR.relative_to(REPO_ROOT)}")
    print(f"  scores  : {CHECKPOINT_ROOT}/{{run}}/exploitability.pkl")
    print(f"  BR      : {args.br_steps}x{args.br_epochs}, eval {args.episodes} episodes")
    if args.n_checkpoints is not None:
        print(f"  subsample: --n-checkpoints {args.n_checkpoints} (non-Kuhn only)")
    print(f"\nsubmit: bash {(SCRIPTS_DIR / 'run_all_score.sh').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
