"""Generate configs + SLURM shell scripts for experiments/one_shot_neural.

Two experiments, selected with ``--experiment``:

``main`` (default)
    The eval-matched grid: one job per (game, method) across the chosen seeds, via
    ``run_all.py``, every method given the same ``--budget`` in payoff evaluations.
    ``DEFAULT_METHODS`` is the batch; it tracks ``run_cell.METHODS`` but is kept separate
    and edited by hand, so check it rather than assuming it is the whole grid.

``walltime``
    The wall-time-matched re-run of the cheap zeroth-order methods, via
    ``run_walltime_matched.py``. There is deliberately **no ``--budget``** here: each
    job derives its own, per (game, method), from the reference tree's measured
    throughput so that the run lands at the ``--match`` method's wall-time times
    ``--headroom``. Jobs land in a separate output tree and never write the reference.

Game specs live in ``GAMES`` below -- this script writes a YAML for each under
``configs/one_shot_neural/`` and points every job at those files. Do not hand it
paths into ``configs/*.yaml``; edit ``GAMES`` instead (comment a game out to
drop it from the batch).

Each experiment also writes its own ``scripts/<name>/`` directory with a companion
``run_all.sh`` that sbatches every generated job, and a ``score.sh`` that runs
``score.py`` on the same output tree once those jobs finish.

Edit the defaults in ``main()``, or override them on the CLI:

    python rci_scripts/generate_one_shot_neural.py
    python rci_scripts/generate_one_shot_neural.py \\
        --games two_point idealized_decoy_well \\
        --seeds 0 1 2 \\
        --budget 2000000 \\
        --out data/one_shot_neural

Hyperparameters come from ``SHARED_SETTINGS`` / ``METHOD_SETTINGS`` (every
``run_cell.Settings`` knob, per method). Override on the CLI with ``--set``
(applied on top of those for every job), or by editing the dicts::

    python rci_scripts/generate_one_shot_neural.py \\
        --methods rpn_pathwise --set rpn_lr=1e-3 --set rpn_batch_size=256

    # the wall-time-matched follow-up (spg / jpspg only)
    python rci_scripts/generate_one_shot_neural.py --experiment walltime
    bash scripts/one_shot_neural_walltime/run_all.sh
    bash scripts/one_shot_neural_walltime/score.sh

The ``walltime`` scripts read ``--reference`` at *run* time to calibrate, so that tree
must already hold finished ``--match`` and pseudo-gradient runs on the cluster. Check
what the budgets will be before submitting:

    python experiments/one_shot_neural/run_walltime_matched.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import prepare_default_script

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "one_shot_neural"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "one_shot_neural"
SCRIPTS_DIR_WALLTIME = REPO_ROOT / "scripts" / "one_shot_neural_walltime"
RUN_ALL = "experiments/one_shot_neural/run_all.py"
RUN_WALLTIME = "experiments/one_shot_neural/run_walltime_matched.py"
SCORE = "experiments/one_shot_neural/score.py"

# Shared network / optimizer / PPO block. Method-specific knobs live in
# `run_cell.Settings` (--set), not here; these only configure the networks every
# method shares. Matches the historical train.py-style configs the comparison used.
# Every `training.run_config` field a one-shot run reads is spelled out, at its schema
# default unless noted, so a generated YAML never silently picks up a changed default.
# Left out: `network`'s `exp_family`-only basis fields and `ppo.density_*` (run_cell
# builds the Gaussian mixture), and `train.solver` (train_sequential.py only).
_SHARED = {
    "network": {
        "policy": "gaussian_mixture",
        "hidden_dims": [64, 64],
        "activation": "gelu",
        "normalization": "rms_norm",
        "full_covariance": False,
        "scale_parameterization": "log",
        "max_correlation": 0.0,
        "sigma_min": None,
        "sigma_max": None,
        "bucket_means": False,
        "clip_means": False,
        "mean_box_penalty_coef": 1.0,
    },
    "optimizer": {
        "learning_rate": 0.0003,
        "max_grad_norm": 1.0,
        "optimizer": "muon",
        "weight_decay": 0.01,
    },
    "ppo": {
        "clip_eps": 0.1,
        "value_coef": 0.5,
        "batch_size": 256,
        "ppo_epochs": 2,
        "target_tau": 0.001,
        "magnet_interval": 500,
        "category_entropy_coef": 0.05,
        "gaussian_entropy_coef": 0.05,
        "trpo_category_kl_coef": 0.05,
        "trpo_gaussian_kl_coef": 0.05,
        "magnet_category_kl_coef": 0.2,
        "magnet_gaussian_kl_coef": 0.2,
        # Categorical head update: "ppo" (clipped surrogate) or "neurd".
        "category_update": "ppo",
        "neurd_beta": 2.0,
        "neurd_clip": 10.0,
        "normalize_advantage": True,
        # Probability floor on the categorical head; 0 disables.
        "category_floor": 0.0,
        "category_floor_coef": 0.01,
        "category_floor_mode": "kind",
        # Sequential-only: a one-shot trainer rejects explore_eps > 0, and "vtrace"
        # needs a game tree. Pinned here so the YAML states what the run did.
        "explore_eps": 0.0,
        "advantage": "monte_carlo",
        "gamma": 1.0,
        "vtrace_lambda": 0.95,
        "vtrace_rho_bar": 2.0,
        "vtrace_c_bar": 1.0,
        "vtrace_opponent_correction": "none",
        "vtrace_opponent_past_floor": 0.05,
        "vtrace_opponent_past_floor_mode": "cumulative",
    },
    "train": {
        "mode": "self_play",
        "perspective": 0,
        "opponent": "random",
        "steps": 100,
        "epochs": 500,
        "seed": 0,
    },
}

# Tag -> (comment, game section, num_components). The tag is the stem of the
# written YAML and the run directory name under --out; it need not equal
# game.name (two_point is multi_point with two peaks; decoy_well is the
# counterexample cell of decoy_well).
GAMES: dict[str, dict[str, Any]] = {
    # Bilinear; unique Nash at the origin. Sanity-check game for dynamics.
    "matching_pennies": {
        "comment": "Continuous matching pennies; unique Nash at the origin.",
        "num_components": 1,
        "game": {
            "name": "matching_pennies",
            "dim": 1,
        },
    },
    # Finite Nash support: a K-component mixture can represent it exactly.
    "two_point": {
        "comment": "Two-peak multi_point; K=2 can represent the Nash exactly.",
        "num_components": 2,
        "game": {
            "name": "multi_point",
            "peaks": [-1.0, 1.0],
            "weights": [0.3, 0.7],
            "width": 0.1,
            "coupling": 1.0,
        },
    },
    # "multi_point": {
    #     "comment": "Three-peak multi_point on the default peaks.",
    #     "num_components": 2,
    #     "game": {
    #         "name": "multi_point",
    #         "peaks": [0.0, 1.0, 2.0],
    #         "weights": None,
    #         "width": 0.1,
    #         "coupling": 1.0,
    #     },
    # },
    # Counterexample: |Nash support| == K == 2 and MMD still converges to the decoy.
    # Tag keeps the historical run-directory name; game.name is decoy_well.
    "idealized_decoy_well": {
        "comment": "Decoy-well counterexample; K=2 equals |Nash support|.",
        "num_components": 4,
        "game": {
            "name": "decoy_well",
            "peaks": [-1.0, 1.0],
            "weights": None,
            "peak_width": 0.05,
            "peak_height": 1.0,
            "decoys": [[0.0, 0.7, 0.45]],
            "coupling": 1.0,
            "action_margin": 2.0,
        },
    },
    # Continuum-support Nash: residual exploitability is about the representation.
    "all_pay_auction": {
        "comment": "Hard all-pay auction; unique Nash is uniform on [0, 2*value].",
        "num_components": 4,
        "game": {
            "name": "all_pay_auction",
            "value": 0.5,
            "high": 1.0,
            "sharpness": None,
        },
    },
    "circle": {
        "comment": "Cyclic kernel on the circle; uniform (and M>harmonics grids) are Nash.",
        "num_components": 4,
        "game": {
            "name": "circle",
            "harmonics": 3,
            "coefficients": None,
        },
    },
    "glicksberg_gross": {
        "comment": "Glicksberg-Gross; Nash density ~1/sqrt(t), unbounded at 0.",
        "num_components": 4,
        "game": {
            "name": "glicksberg_gross",
        },
    },
}

# The batch to generate, edited by hand -- deliberately not `run_cell.METHODS`, since a
# cluster batch is usually a subset of the grid (note `sisa` and methods below are
# commented out rather than deleted, so a run that wants them is one edit away).
DEFAULT_METHODS = (
    "mixture",
    "mmd_discrete",
    "nfsp",
    "psro",
    "rpn_pathwise",
    # "spg",
    # "jpspg",
    # "sisa",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
# Shared metric / logging knobs from `run_cell.Settings`, applied to every method's job.
SHARED_SETTINGS: dict[str, object] = {
    "checkpoints": 40,
    "grid": 401,
    "samples": 16384,
}
# Per-method `run_cell.Settings` knobs. Edit values here; each job gets SHARED_SETTINGS
# plus its own block. Values match `Settings` defaults except where this batch tunes them
# (notably `rpn_pathwise`). Mixture has no method-specific Settings -- its capacity is
# `network.num_components` in the generated YAML.
METHOD_SETTINGS: dict[str, dict[str, object]] = {
    "mixture": {},
    # `bins` is not set here: `settings_flags` matches it to each game's
    # `num_components`, so the discrete grid has the mixture's capacity.
    "mmd_discrete": {},
    "nfsp": {
        "br_steps": 20,
        "br_epochs": 50,
        "br_batch_size": 64,
        "eta": 0.1,
        "sl_steps": 400,
        "average_head": "mixture",
        "average_components": 8,
    },
    "psro": {
        "br_steps": 20,
        "br_epochs": 50,
        "br_batch_size": 64,
        "payoff_samples": 256,
        "meta_solver": "nash",
    },
    "rpn_pathwise": {
        "rpn_lr": 1e-3,
        "rpn_optimizer": "optimistic",
        "rpn_optimism": 0.333,
        "rpn_max_grad_norm": 5.0,
        "rpn_batch_size": 256,
        "rpn_noise_dim": 16,
        "rpn_activation": "mish",
        "rpn_normalization": "rms_norm",
        "rpn_squash": "sigmoid",
        "rpn_smooth": 0,
        "rpn_smooth_scale": 0.1,
        # "extragradient" doubles payoff evals per iteration; plan_units accounts for it.
        "rpn_dynamics": "extragradient",
        "rpn_extragradient_step": 0.01,
    },
    # "spg": {
    #     "sigma": 2.0,
    #     "pseudo_lr": 1e-3,
    #     "pseudo_optimizer": "adabelief",
    #     "utility_samples": 256,
    #     "perturbation_batch": 256,
    #     "noise_dim": 16,
    #     "pseudo_max_grad_norm": 1.0,
    #     "dynamics": "simultaneous",
    # },
    # "jpspg": {
    #     "sigma": 2.0,
    #     "pseudo_lr": 1e-3,
    #     "pseudo_optimizer": "adabelief",
    #     "utility_samples": 256,
    #     "perturbation_batch": 256,
    #     "noise_dim": 16,
    #     "pseudo_max_grad_norm": 1.0,
    #     "dynamics": "simultaneous",
    # },
    # "sisa": {
    #     "atoms": 8,
    #     "lr_support": 1e-2,
    #     "lr_weight": 1e-2,
    #     "sisa_init": "spread",
    # },
}
DEFAULT_BUDGET = 20_000_000
DEFAULT_OUT = "data/one_shot_neural"
DEFAULT_MAX_PARALLEL = 1
DEFAULT_TIME_H = 4
DEFAULT_MEMORY_G = 16
DEFAULT_GPU = False
DEFAULT_SCORE_GRID = 801

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


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
    return str(value)


def _dump_yaml_section(name: str, section: dict[str, Any]) -> list[str]:
    lines = [f"{name}:"]
    for key, value in section.items():
        lines.append(f"  {key}: {_yaml_scalar(value)}")
    return lines


def build_config(tag: str) -> dict[str, Any]:
    spec = GAMES[tag]
    network = {**_SHARED["network"], "num_components": spec["num_components"]}
    return {
        "game": dict(spec["game"]),
        "network": network,
        "optimizer": dict(_SHARED["optimizer"]),
        "ppo": dict(_SHARED["ppo"]),
        "train": {**_SHARED["train"], "checkpoint_dir": f"data/one_shot_neural/{tag}"},
    }


def write_game_config(path: Path, tag: str) -> None:
    spec = GAMES[tag]
    config = build_config(tag)
    lines = [f"# {spec['comment']}", f"# Generated by rci_scripts/generate_one_shot_neural.py ({tag}).", ""]
    for section in ("game", "network", "optimizer", "ppo", "train"):
        lines.extend(_dump_yaml_section(section, config[section]))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def config_rel(tag: str) -> str:
    return (CONFIGS_DIR / f"{tag}.yaml").relative_to(REPO_ROOT).as_posix()


def job_name(game_tag: str, method: str) -> str:
    return f"{game_tag}__{method}"


def settings_flags(method: str, game_tag: str, extra: list[str] = ()) -> list[str]:
    """`--set key=value` items for one (game, method) job: shared + that method's block +
    per-game overrides + CLI extras (last, so they still win)."""
    if method not in METHOD_SETTINGS:
        raise SystemExit(
            f"no METHOD_SETTINGS entry for {method!r}; add one or choose from "
            f"{sorted(METHOD_SETTINGS)}")
    merged = {**SHARED_SETTINGS, **METHOD_SETTINGS[method]}
    if method == "mmd_discrete":
        merged["bins"] = GAMES[game_tag]["num_components"]
    flags = [f"{key}={value}" for key, value in merged.items()]
    flags.extend(extra)
    return flags


def write_job_script(
    path: Path,
    *,
    game_config: str,
    method: str,
    seeds: list[int],
    budget: int,
    out: str,
    max_parallel: int,
    time_h: int,
    memory_g: int,
    gpu: bool,
    settings: list[str] = (),
) -> None:
    header = prepare_default_script(time_h, memory_g, gpu)
    seeds_str = " ".join(str(s) for s in seeds)
    # One `--set key=value` per override; `run_all.py` forwards each to `run_cell.py` as
    # `--key value`, so these are `run_cell.Settings` field names.
    settings_str = "".join(f"  --set {item} \\\n" for item in settings)
    body = f"""
python {RUN_ALL} \\
  --games {game_config} \\
  --methods {method} \\
  --seeds {seeds_str} \\
  --budget {budget} \\
  --out {out} \\
{settings_str}  --max-parallel {max_parallel}
"""
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_walltime_job_script(
    path: Path,
    *,
    game_config: str,
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
  --games {game_config} \\
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


def write_score(path: Path, out: str, grid: int) -> None:
    """Score the training output tree. Not submitted: scoring is cheap next to training,
    and running it in the login shell is how you see it fail."""
    lines = [
        "#!/bin/sh",
        "",
        "# Run after run_all.sh's jobs have finished.",
        f"python {SCORE} --out {out} --grid {grid} --plot",
        "",
    ]
    path.write_text("\n".join(lines))
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
    settings_extra: list[str] = (),
    experiment: str = "main",
    reference: str = WALLTIME_REFERENCE,
    match: str = WALLTIME_MATCH,
    headroom: float = WALLTIME_HEADROOM,
    score_grid: int = DEFAULT_SCORE_GRID,
) -> list[Path]:
    unknown = [g for g in games if g not in GAMES]
    if unknown:
        raise SystemExit(
            f"unknown game tag(s) {unknown}; edit GAMES or choose from {sorted(GAMES)}")
    unknown_methods = [m for m in methods if m not in METHOD_SETTINGS]
    if unknown_methods and experiment != "walltime":
        raise SystemExit(
            f"no METHOD_SETTINGS for {unknown_methods}; edit METHOD_SETTINGS or choose "
            f"from {sorted(METHOD_SETTINGS)}")

    walltime = experiment == "walltime"
    scripts_dir = SCRIPTS_DIR_WALLTIME if walltime else SCRIPTS_DIR
    log_dir = f"logs/{scripts_dir.name}"
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    game_configs: dict[str, str] = {}
    for tag in games:
        path = CONFIGS_DIR / f"{tag}.yaml"
        write_game_config(path, tag)
        written.append(path)
        game_configs[tag] = config_rel(tag)

    job_scripts: list[Path] = []
    for tag in games:
        game_config = game_configs[tag]
        for method in methods:
            path = scripts_dir / f"{job_name(tag, method)}.sh"
            if walltime:
                write_walltime_job_script(
                    path,
                    game_config=game_config,
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
                    game_config=game_config,
                    method=method,
                    seeds=seeds,
                    budget=budget,
                    out=out,
                    max_parallel=max_parallel,
                    time_h=time_h,
                    memory_g=memory_g,
                    gpu=gpu,
                    settings=settings_flags(method, tag, settings_extra),
                )
            job_scripts.append(path)
            written.append(path)

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
    score = scripts_dir / "score.sh"
    write_score(score, out, score_grid)
    written.extend([submit_all, score])
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=("main", "walltime"), default="main",
                    help="'main': the eval-matched run_all.py grid. 'walltime': the "
                         "wall-time-matched spg/jpspg re-run (no --budget; it is derived "
                         "per cell from --reference)")
    ap.add_argument("--games", nargs="+", default=None, choices=sorted(GAMES),
                    metavar="TAG",
                    help="subset of GAMES to generate; default is every key in GAMES")
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
    ap.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                    dest="settings_extra",
                    help="extra run_cell.Settings override applied on top of "
                         "SHARED_SETTINGS/METHOD_SETTINGS for every job "
                         "(e.g. --set samples=4096); repeatable. Main experiment only")
    ap.add_argument("--score-grid", type=int, default=None,
                    help="deviation grid score.sh passes to score.py")
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
    games = list(GAMES)
    methods = list(WALLTIME_METHODS if walltime else DEFAULT_METHODS)
    seeds = list(DEFAULT_SEEDS)
    budget = DEFAULT_BUDGET
    out = WALLTIME_OUT if walltime else DEFAULT_OUT
    max_parallel = DEFAULT_MAX_PARALLEL
    time_h = WALLTIME_TIME_H if walltime else DEFAULT_TIME_H
    memory_g = DEFAULT_MEMORY_G
    gpu = DEFAULT_GPU
    settings_extra: list[str] = []
    reference = WALLTIME_REFERENCE
    match = WALLTIME_MATCH
    headroom = WALLTIME_HEADROOM
    score_grid = DEFAULT_SCORE_GRID

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
    if args.settings_extra is not None:
        settings_extra = args.settings_extra
    if args.score_grid is not None:
        score_grid = args.score_grid
    if args.reference is not None:
        reference = args.reference
    if args.match is not None:
        match = args.match
    if args.headroom is not None:
        headroom = args.headroom

    if walltime and args.budget is not None:
        print("note: --budget is ignored for --experiment walltime; each cell derives "
              "its own from --reference")
    if walltime and settings_extra:
        print("note: --set is ignored for --experiment walltime; "
              "run_walltime_matched.py has no --set")
        settings_extra = []

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
        settings_extra=settings_extra,
        experiment=args.experiment,
        reference=reference,
        match=match,
        headroom=headroom,
        score_grid=score_grid,
    )
    scripts_dir = SCRIPTS_DIR_WALLTIME if walltime else SCRIPTS_DIR
    n_configs = len(games)
    n_jobs = len(games) * len(methods)
    print(f"wrote {n_configs} configs under {CONFIGS_DIR.relative_to(REPO_ROOT)}")
    print(f"wrote {n_jobs} job scripts + run_all.sh + score.sh under "
          f"{scripts_dir.relative_to(REPO_ROOT)}")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}")
    print(f"score: bash {(scripts_dir / 'score.sh').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
