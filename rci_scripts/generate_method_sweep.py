"""Generate SLURM scripts for a per-method hyperparameter sweep of experiments/one_shot_neural.

`generate_one_shot_neural.py` submits the *comparison*: every method on its own defaults,
one job per (game, method). This submits the sweep that comes before it -- every method on
several settings of its own knobs, so the comparison can be run on settings that were
chosen rather than inherited.

**Edit `METHOD_GRIDS` below; that is the interface.** Each entry maps a `run_cell.py`
setting to either a list (an axis to sweep) or a single value (a fixed override applied to
every variant of that method). The cartesian product of the lists is the set of variants,
and the sweep is the product of that with `--games` and `--seeds`:

    METHOD_GRIDS = {
        "nfsp": {"br_steps": [25, 50, 100], "eta": [0.1, 0.5]},   # 6 variants
        "psro": {"payoff_samples": 512},                          # 1 variant, non-default
    }

Every setting name is checked against `run_cell.py`'s `Settings` dataclass before anything
is written, so a typo fails here rather than after a night in the queue.

**Each variant gets its own output tree**, `<out>/<variant tag>/<game>/<method>/seedN`,
because `run_cell.py` keys a run directory by (game, method, seed) alone -- two settings of
`nfsp` written into one tree would silently be the same run. A method with no swept axis
has no variant to distinguish and lands in a shared `base/` tree, so the untuned methods
still sit in one tree that `score.py` summarizes as a single head-to-head table.

    python rci_scripts/generate_method_sweep.py
    python rci_scripts/generate_method_sweep.py --dry-run          # the plan, nothing written
    python rci_scripts/generate_method_sweep.py --methods nfsp psro --seeds 0 1
    bash scripts/method_sweep/run_all.sh                           # submit
    bash scripts/method_sweep/score_all.sh                         # after they finish

`score_all.sh` scores each variant tree separately (one `summary.md` per tree). To compare
variants of one method against each other, read the `expl` column of each tree's summary;
they share a grid and a budget, so the numbers are comparable across trees.
"""

from __future__ import annotations

import argparse
import ast
import itertools
from pathlib import Path

from utils import prepare_default_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "method_sweep"
RUN_ALL = "experiments/one_shot_neural/run_all.py"
RUN_CELL = REPO_ROOT / "experiments" / "one_shot_neural" / "run_cell.py"
SCORE = "experiments/one_shot_neural/score.py"

DEFAULT_GAMES = (
    "configs/two_point.yaml",
    "configs/all_pay_auction.yaml",
    "configs/circle.yaml",
    "configs/glicksberg_gross.yaml",
)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_BUDGET = 20_000_000
DEFAULT_OUT = "data/method_sweep"
DEFAULT_MAX_PARALLEL = 1
DEFAULT_TIME_H = 4
DEFAULT_MEMORY_G = 16
DEFAULT_GPU = False
DEFAULT_SCORE_GRID = 801

# --------------------------------------------------------------------------- the sweep
#
# What to sweep, per method. A list is an axis; a bare value is a fixed override. An empty
# dict means "this method, on its defaults" -- worth keeping in, since a sweep of one
# method is only readable next to the others on their usual settings.
#
# Anything not named here keeps the default in `run_cell.py`'s `Settings`, which is where
# to look for the full list of knobs and for why the non-obvious defaults are what they are.
METHOD_GRIDS: dict[str, dict[str, object]] = {
    # The best-response budget per round, and how much of the anticipatory mixture is the
    # best response. `br_steps * br_epochs` is the PPO iteration count that decides whether
    # `beta` is a best response at all; see `baselines/neural/nfsp.py`.
    "nfsp": {
        "br_steps": [25, 50, 100],
        "eta": [0.1, 0.5],
    },
    # `payoff_samples` sets the Monte-Carlo error in the meta-game matrix the LP solves,
    # and `br_epochs` the oracle budget (PSRO's default is 2000 iterations/round, twice
    # NFSP's, because a population member is trained once and then frozen forever).
    "psro": {
        "br_epochs": [20, 40],
        "payoff_samples": [256, 512],
    },
    # The discretization. `bins` is the whole method: too coarse and the equilibrium is not
    # representable, too fine and the categorical head is mostly dead mass.
    "mmd_discrete": {
        "bins": [51, 101],
    },
    # The reference point -- swept on nothing, so it lands in `base/` with any other
    # method left on its defaults.
    "mixture": {},
    # Zeroth order. `perturbation_batch` is the one that decides whether these converge at
    # all (see the note in `Settings`); `sigma` keeps the estimator alive once the tanh
    # squash saturates.
    "spg": {
        "perturbation_batch": [64, 256],
        "sigma": [0.5, 2.0],
    },
    "jpspg": {
        "perturbation_batch": [64, 256],
        "sigma": [0.5, 2.0],
    },
}

# Per-method SLURM resources, for the ones whose cost is not the default. A job is
# `len(seeds)` cells run `--max-parallel` at a time, so its wall-time is roughly
# `len(seeds) / max_parallel` times one cell.
METHOD_RESOURCES: dict[str, dict[str, int]] = {
    # ~1300 s/cell on these games at the default budget, and cold-starting the best
    # response (see baselines/neural/nfsp.py) does not make it cheaper.
    "nfsp": {"time_h": 8},
    "psro": {"time_h": 8},
    "mixture": {"time_h": 8},
}


def settings_fields() -> dict[str, str]:
    """`{name: type}` of `run_cell.py`'s `Settings`, read from its source.

    Parsed rather than imported: this script should run anywhere, and importing
    `run_cell` pulls in JAX and every baseline just to read a list of field names.
    """
    tree = ast.parse(RUN_CELL.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            return {item.target.id: ast.unparse(item.annotation)
                    for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
    raise RuntimeError(f"no Settings dataclass found in {RUN_CELL}")


def validate(grids: dict[str, dict[str, object]]) -> None:
    """Fail before writing anything if a knob does not exist or has the wrong type."""
    fields = settings_fields()
    casts = {"int": int, "float": float, "str": str}
    for method, grid in grids.items():
        for key, value in grid.items():
            if key not in fields:
                near = [f for f in fields if key in f or f in key]
                raise SystemExit(
                    f"{method}: {key!r} is not a run_cell.py setting"
                    + (f" (did you mean {near}?)" if near else "")
                    + f"\navailable: {sorted(fields)}")
            kind = casts[fields[key]]
            for item in (value if isinstance(value, list) else [value]):
                if kind is float and isinstance(item, int):
                    continue        # 1 for a float knob is fine
                if not isinstance(item, kind):
                    raise SystemExit(f"{method}: {key}={item!r} should be {fields[key]}")


def variants(grid: dict[str, object]) -> list[dict[str, object]]:
    """Every combination of the swept axes, each already carrying the fixed overrides."""
    fixed = {k: v for k, v in grid.items() if not isinstance(v, list)}
    axes = {k: v for k, v in grid.items() if isinstance(v, list)}
    if not axes:
        return [dict(fixed)]
    return [{**fixed, **dict(zip(axes, combination))}
            for combination in itertools.product(*axes.values())]


def variant_tag(method: str, grid: dict[str, object], variant: dict[str, object]) -> str:
    """The output subtree for one variant.

    Only the *swept* axes go in the name -- a fixed override is constant across the sweep,
    so putting it in every directory name would say nothing while making every path longer.
    It is still recorded in the run's own `meta.json`. A method with no swept axis has
    nothing to distinguish and shares the `base` tree with the other such methods.
    """
    axes = [k for k, v in grid.items() if isinstance(v, list)]
    if not axes:
        return "base"
    return method + "__" + "__".join(f"{k}{variant[k]}" for k in axes)


def job_name(game: str, method: str, tag: str) -> str:
    return f"{Path(game).stem}__{method}" + ("" if tag == "base" else f"__{tag[len(method) + 2:]}")


def write_job_script(
    path: Path,
    *,
    game: str,
    method: str,
    variant: dict[str, object],
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
    sets = "".join(f" \\\n  --set {key}={value}" for key, value in sorted(variant.items()))
    body = f"""
python {RUN_ALL} \\
  --games {game} \\
  --methods {method} \\
  --seeds {seeds_str} \\
  --budget {budget} \\
  --out {out} \\
  --max-parallel {max_parallel}{sets}
"""
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_submit_all(path: Path, job_scripts: list[Path], log_dir: str) -> None:
    lines = ["#!/bin/sh", "", f"mkdir -p {log_dir}", ""]
    for script in job_scripts:
        rel = script.relative_to(REPO_ROOT).as_posix()
        lines.append(f"sbatch -o {log_dir}/{script.stem}.log {rel}")
    lines.append("")
    lines.append(f'echo "submitted {len(job_scripts)} jobs; logs in {log_dir}"')
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def write_score_all(path: Path, trees: list[str], grid: int) -> None:
    """Score each variant tree. Sequential and unsubmitted on purpose: scoring is cheap
    next to training, and running it in the login shell is how you see it fail."""
    lines = ["#!/bin/sh", "",
             "# Run after run_all.sh's jobs have finished. Each tree gets its own",
             "# summary.md / curves.json; the grid is shared so the numbers compare across trees.",
             ""]
    for tree in trees:
        lines.append(f"python {SCORE} --out {tree} --grid {grid} --plot")
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
    score_grid: int,
    dry_run: bool = False,
) -> list[Path]:
    grids = {m: METHOD_GRIDS[m] for m in methods}
    validate(grids)

    plan: list[tuple[str, str, str, dict]] = []      # (game, method, tag, variant)
    for method in methods:
        for variant in variants(grids[method]):
            tag = variant_tag(method, grids[method], variant)
            for game in games:
                plan.append((game, method, tag, variant))

    trees = sorted({f"{out}/{tag}" for _, _, tag, _ in plan})
    if dry_run:
        print(f"{len(plan)} jobs, {len(trees)} output trees, "
              f"{len(plan) * len(seeds)} cells total\n")
        for game, method, tag, variant in plan:
            settings = " ".join(f"{k}={v}" for k, v in sorted(variant.items())) or "(defaults)"
            print(f"  {job_name(game, method, tag):<48} -> {out}/{tag}    {settings}")
        return []

    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    job_scripts: list[Path] = []
    for game, method, tag, variant in plan:
        resources = METHOD_RESOURCES.get(method, {})
        path = SCRIPTS_DIR / f"{job_name(game, method, tag)}.sh"
        write_job_script(
            path,
            game=game,
            method=method,
            variant=variant,
            seeds=seeds,
            budget=budget,
            out=f"{out}/{tag}",
            max_parallel=max_parallel,
            time_h=resources.get("time_h", time_h),
            memory_g=resources.get("memory_g", memory_g),
            gpu=gpu,
        )
        job_scripts.append(path)

    log_dir = f"logs/{SCRIPTS_DIR.name}"
    submit_all = SCRIPTS_DIR / "run_all.sh"
    write_submit_all(submit_all, job_scripts, log_dir)
    score_all = SCRIPTS_DIR / "score_all.sh"
    write_score_all(score_all, trees, score_grid)
    return job_scripts + [submit_all, score_all]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES))
    ap.add_argument("--methods", nargs="+", default=sorted(METHOD_GRIDS),
                    choices=sorted(METHOD_GRIDS),
                    help="which of METHOD_GRIDS to generate; the grids themselves are "
                         "edited in this file")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="payoff evaluations per cell -- the shared cost unit")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="root under which each variant gets its own tree")
    ap.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                    help="parallel seeds within one SLURM job")
    ap.add_argument("--time", type=int, default=DEFAULT_TIME_H, dest="time_h",
                    help="wall-time hours, for methods not in METHOD_RESOURCES")
    ap.add_argument("--memory", type=int, default=DEFAULT_MEMORY_G, dest="memory_g")
    ap.add_argument("--gpu", action="store_true", default=DEFAULT_GPU)
    ap.add_argument("--score-grid", type=int, default=DEFAULT_SCORE_GRID,
                    help="deviation grid score_all.sh uses")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and write nothing")
    args = ap.parse_args()

    written = generate(
        games=args.games,
        methods=args.methods,
        seeds=args.seeds,
        budget=args.budget,
        out=args.out,
        max_parallel=args.max_parallel,
        time_h=args.time_h,
        memory_g=args.memory_g,
        gpu=args.gpu,
        score_grid=args.score_grid,
        dry_run=args.dry_run,
    )
    if not written:
        return
    print(f"wrote {len(written) - 2} job scripts + run_all.sh + score_all.sh "
          f"under {SCRIPTS_DIR.relative_to(REPO_ROOT)}")
    print(f"  {len(written) - 2} jobs x {len(args.seeds)} seeds "
          f"= {(len(written) - 2) * len(args.seeds)} cells")
    print(f"\nsubmit: bash {(SCRIPTS_DIR / 'run_all.sh').relative_to(REPO_ROOT)}")
    print(f"score : bash {(SCRIPTS_DIR / 'score_all.sh').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
