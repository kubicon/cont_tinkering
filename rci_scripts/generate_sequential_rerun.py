"""Re-submit the runs of a sequential sweep that stopped short of their last checkpoint.

A job killed by the SLURM wall-time leaves a run directory with fewer
``checkpoints/{step}.pkl`` files than ``train.steps`` asked for. That is not a
scoring problem -- every checkpoint that exists gets its best response -- but it
truncates the plots twice over: the curve ends early, and, because
``score_sequential_sweep.py`` subsamples checkpoints *per run*, seeds that
stopped at different points share almost no checkpoint index, so
``plot_sequential_sweep.pool_seeds`` intersects them down to a handful of
points. ``train_sequential.py`` has no resume, so the fix is to run the whole
thing again with a longer ``--time``.

This script writes **only** the SLURM scripts for those runs, into a tree of its
own (``scripts/{experiment}_rerun/``). Nothing that
``generate_sequential_sweep.py`` produced is touched: the per-run configs are
reused exactly as they are, and ``manifest.json`` / ``run_all.sh`` keep
describing the full sweep, so the scoring jobs still cover every run.

A run is short when it has fewer than ``train.steps + 1`` checkpoints (step 0 is
the untrained policy). The expectation comes from each run's own config, so a
sweep with per-game budgets is handled without a flag.

    python rci_scripts/generate_sequential_rerun.py --experiment capacity --dry-run
    python rci_scripts/generate_sequential_rerun.py --experiment capacity --time 24
    bash scripts/sequential_capacity_rerun/run_all_rerun.sh

``--clean`` makes each job delete that run's ``checkpoints/``, ``br/`` and
``exploitability.pkl`` before training. Worth passing: a rerun overwrites
checkpoints by step, so if it ends *earlier* than the original did, the
original's later checkpoints survive and the run looks longer than it was.
The stale ``br/`` and ``exploitability.pkl`` are in any case only replaced by
``generate_sequential_sweep_score.py --overwrite``.

After the reruns finish, re-score and re-plot:

    python rci_scripts/generate_sequential_sweep_score.py --experiment capacity --overwrite
    bash scripts/sequential_capacity/run_all_score.sh
    python rci_scripts/plot_sequential_sweep.py --out data/sequential_capacity --capacity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from generate_sequential_sweep import (
    DEFAULT_MEMORY_G,
    EXPERIMENT_DIRS,
    REPO_ROOT,
    TRAIN,
)
from utils import prepare_default_script

DEFAULT_EXPERIMENT = "capacity"
# The wall-time the reruns get. The 4h default of the original sweep is what cut
# them short; 24h is the most `amd` takes (see utils.prepare_sbatch_part) and
# covers the worst observed rate (44 of 201 checkpoints in 4h) with margin.
DEFAULT_TIME_H = 24
CHECKPOINT_DIRNAME = "checkpoints"
RESULT_FILENAME = "exploitability.pkl"
BR_DIRNAME = "br"
SUBMIT_NAME = "run_all_rerun.sh"


class Paths:
    """The sweep's own tree, plus the separate one this script writes into."""

    def __init__(self, experiment: str) -> None:
        if experiment not in EXPERIMENT_DIRS:
            raise SystemExit(f"unknown experiment {experiment!r}; "
                             f"choices: {sorted(EXPERIMENT_DIRS)}")
        name = EXPERIMENT_DIRS[experiment]
        self.experiment = experiment
        self.sweep_scripts = REPO_ROOT / "scripts" / name
        self.checkpoints = REPO_ROOT / "data" / name
        self.scripts = REPO_ROOT / "scripts" / f"{name}_rerun"
        self.logs = f"logs/{name}_rerun"


def load_manifest(paths: Paths) -> list[dict]:
    path = paths.sweep_scripts / "manifest.json"
    if not path.exists():
        raise SystemExit(f"no manifest at {path}; run generate_sequential_sweep.py "
                         f"--experiment {paths.experiment} first")
    return json.loads(path.read_text())["runs"]


def expected_checkpoints(config_path: Path) -> int | None:
    """``train.steps + 1`` for one run, or None when the config cannot say.

    Checkpoint 0 is written before training starts, so a complete run of
    ``steps`` steps leaves ``steps + 1`` files.
    """
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text()) or {}
    steps = (config.get("train") or {}).get("steps")
    return int(steps) + 1 if steps is not None else None


def count_checkpoints(run_dir: Path) -> int:
    directory = run_dir / CHECKPOINT_DIRNAME
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.glob("*.pkl") if path.stem.isdigit())


def short_runs(runs: list[dict], paths: Paths, only: list[str] | None,
               include_missing: bool) -> list[dict]:
    """The runs whose checkpoint directory is not full, newest shortfall first."""
    out = []
    for run in runs:
        name = run["name"]
        if only is not None and not any(pattern in name for pattern in only):
            continue
        config_path = REPO_ROOT / run["config"]
        expected = expected_checkpoints(config_path)
        if expected is None:
            print(f"  ! {name}: no train.steps in {run['config']}; skipped")
            continue
        have = count_checkpoints(paths.checkpoints / name)
        if have >= expected:
            continue
        # A run with nothing at all never started, which is a different failure
        # (a job that never got queued) and is easy to resubmit by accident.
        if have == 0 and not include_missing:
            print(f"  ! {name}: no checkpoints at all; "
                  "skipped (pass --include-missing to rerun it too)")
            continue
        out.append({**run, "have": have, "expected": expected})
    return sorted(out, key=lambda r: (r["have"] / r["expected"], r["name"]))


def write_job_script(path: Path, *, run: dict, paths: Paths, time_h: int,
                     memory_g: int, gpu: bool, clean: bool) -> None:
    """One SLURM script = one full rerun of one training run."""
    lines = [prepare_default_script(time_h, memory_g, gpu), ""]
    lines.append(f"# rerun: {run['have']} of {run['expected']} checkpoints on disk")
    if clean:
        run_dir = (paths.checkpoints / run["name"]).relative_to(REPO_ROOT).as_posix()
        lines.append("# start from an empty directory, so a shorter rerun cannot leave")
        lines.append("# the previous attempt's later checkpoints behind")
        lines.append(f"rm -rf {run_dir}/{CHECKPOINT_DIRNAME} {run_dir}/{BR_DIRNAME} "
                     f"{run_dir}/{RESULT_FILENAME}")
    lines.append(f"python {TRAIN} {run['config']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def write_submit_all(path: Path, job_scripts: list[Path], log_dir: str,
                     chunk: int | None) -> list[Path]:
    """``run_all_rerun.sh``, or ``run_all_rerun{i}.sh`` when ``--chunk`` splits it."""
    groups = ([job_scripts] if not chunk
              else [job_scripts[i:i + chunk] for i in range(0, len(job_scripts), chunk)])
    written = []
    for index, group in enumerate(groups, start=1):
        target = path if len(groups) == 1 else path.with_name(
            f"{path.stem}{index}{path.suffix}")
        lines = ["#!/bin/sh", ""]
        if len(groups) > 1:
            start = (index - 1) * chunk + 1
            lines.append(f"# part {index} of {len(groups)}: "
                         f"jobs {start}-{start + len(group) - 1} of {len(job_scripts)}")
            lines.append("")
        lines += [f"mkdir -p {log_dir}", ""]
        for script in group:
            rel = script.relative_to(REPO_ROOT).as_posix()
            lines.append(f"sbatch -o {log_dir}/{script.stem}.log {rel}")
        lines += ["", f'echo "submitted {len(group)} jobs; logs in {log_dir}"']
        target.write_text("\n".join(lines) + "\n")
        target.chmod(target.stat().st_mode | 0o111)
        written.append(target)
    return written


def report(runs: list[dict]) -> None:
    for run in runs:
        swept = " ".join(f"{k}={v}" for k, v in run.get("swept", {}).items()) or "(defaults)"
        print(f"  {run['name']:<64} {run['have']:>4}/{run['expected']:<4} {swept}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=sorted(EXPERIMENT_DIRS),
                    default=DEFAULT_EXPERIMENT,
                    help=f"which sweep's runs to check (default: {DEFAULT_EXPERIMENT})")
    ap.add_argument("--runs", nargs="+", default=None, metavar="PATTERN",
                    help="only runs whose name contains one of these "
                         "(e.g. num_components32 discrete_mmd)")
    ap.add_argument("--time", type=int, default=DEFAULT_TIME_H, dest="time_h",
                    help=f"wall-time hours for the reruns (default: {DEFAULT_TIME_H}; "
                         "<=4 amdfast, <=24 amd, <=72 amdlong)")
    ap.add_argument("--memory", type=int, default=DEFAULT_MEMORY_G, dest="memory_g",
                    help=f"memory in GB (default: {DEFAULT_MEMORY_G})")
    ap.add_argument("--gpu", action="store_true",
                    help="request one GPU (changes the partition)")
    ap.add_argument("--clean", action="store_true",
                    help="each job first deletes that run's checkpoints/, br/ and "
                         "exploitability.pkl (recommended; see the module docstring)")
    ap.add_argument("--include-missing", action="store_true",
                    help="also rerun runs with no checkpoints at all (never started)")
    ap.add_argument("--chunk", type=int, default=None, metavar="N",
                    help="split the submit script into parts of N jobs")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the short runs, write nothing")
    args = ap.parse_args()

    paths = Paths(args.experiment)
    runs = short_runs(load_manifest(paths), paths, args.runs, args.include_missing)
    if not runs:
        print(f"every {args.experiment} run has its full checkpoint set; nothing to rerun")
        return

    print(f"{len(runs)} short runs (have/expected):")
    report(runs)
    if args.dry_run:
        print("\ndry run; nothing written")
        return

    job_scripts = []
    for run in runs:
        script = paths.scripts / f"{run['name']}.sh"
        write_job_script(script, run=run, paths=paths, time_h=args.time_h,
                         memory_g=args.memory_g, gpu=args.gpu, clean=args.clean)
        job_scripts.append(script)
    submits = write_submit_all(paths.scripts / SUBMIT_NAME, job_scripts,
                               paths.logs, args.chunk)

    print(f"\nwrote {len(job_scripts)} job scripts to "
          f"{paths.scripts.relative_to(REPO_ROOT)}/")
    if not args.clean:
        print("note: --clean was not passed, so a rerun that again ends early will "
              "leave the previous attempt's later checkpoints in place")
    for submit in submits:
        print(f"submit: bash {submit.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
