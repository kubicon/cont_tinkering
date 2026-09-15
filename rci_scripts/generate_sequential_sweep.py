"""Generate per-run configs + SLURM scripts for a sequential-game sweep.

`train_sequential.py` solves one game tree with one of the five solvers
(`self_play`, `discrete_mmd`, `nfsp`, `psro`, `rpn`) off a single YAML file.
Sweep entries may also be *aliases* (see `SOLVER_ALIASES`): same runner, different
pinned hyperparameters -- `sac` is the Soft-Actor-Critic ablation of `self_play`.
This script writes the cross product of

    games x solvers x that solver's hyperparameter grid x seeds

as **one standalone config and one SLURM script per training run** -- a job here
is exactly one `python train_sequential.py <config>`, so a run that dies takes
nothing else with it and can be resubmitted on its own.

**Edit `COMMON`, `SOLVERS`, and `GAME_OVERRIDES` below; that is the interface.**
Every training hyperparameter lives here so you can change defaults in this file
without opening the base YAMLs. A key is a dotted path into the config
(`nfsp.br_steps`, `ppo.batch_size`); a bare value is a fixed default; wrap a
value in a list to sweep that axis:

    SOLVERS = {
        "nfsp": {
            "nfsp.rounds": 20,                    # fixed default
            "nfsp.br_steps": [50, 100],           # 2-way sweep
            "nfsp.eta": [0.1, 0.5],               # x 2 = 4 variants
        },
        "psro": {"psro.payoff_episodes": 50_000}, # 1 variant, non-base default
        "sac": {                                 # alias -> self_play in SOLVER_ALIASES
            "network.num_components": 1,
            "ppo.trpo_gaussian_kl_coef": 0.0,
        },
    }

List-valued fields that are *not* a sweep (e.g. network widths) must be tuples:
`network.hidden_dims: (64, 64)` -- a Python list always means an axis.

`COMMON` is multiplied into every solver -- that is where a knob the solvers
share (`optimizer.learning_rate`, `ppo.batch_size`) belongs.
`GAME_OVERRIDES` wins last, for per-game budget differences (Kuhn vs Leduc).
Every dotted key is checked against `training/run_config.py`'s dataclasses
before anything is written, so a typo fails here rather than after a night in
the queue. The base `--games` YAML still supplies the `game:` block (and any
key you leave unnamed here).

**Each run gets a directory of its own**, `data/sequential_sweep/<run name>`,
where the run name is `<game>__<solver>[__<swept axes>]__seed<N>`. That is not
cosmetic: `train_sequential.py` refuses to write one solver's checkpoints into
another's directory, and two settings of `nfsp` sharing a directory would
silently be one run. Only the *swept* axes go in the name -- a fixed default is
constant across the sweep, so it would lengthen every path while saying nothing;
it is still recorded in the run's own config and in its `meta.json`.

    python rci_scripts/generate_sequential_sweep.py
    python rci_scripts/generate_sequential_sweep.py --dry-run        # the plan, nothing written
    python rci_scripts/generate_sequential_sweep.py \\
        --games configs/kuhn_solvers.yaml --solvers nfsp psro --seeds 0 1 2
    bash scripts/sequential_sweep/run_all.sh                         # submit
    python rci_scripts/generate_sequential_sweep_score.py            # score jobs
    bash scripts/sequential_sweep/run_all_score.sh                   # after training

`manifest.json` beside the scripts maps each run name to its game, solver, seed
and full set of overrides, which is what an analysis script should read rather
than parsing the directory names back apart. Offline exploitability scoring is
``generate_sequential_sweep_score.py``: one SLURM job per run, reading this
manifest and calling ``score_sequential_sweep.py``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
from pathlib import Path

import yaml

from utils import prepare_default_script

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "sequential_sweep"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "sequential_sweep"
CHECKPOINT_ROOT = "data/sequential_sweep"
LOG_DIR = "logs/sequential_sweep"
RUN_CONFIG = REPO_ROOT / "training" / "run_config.py"
TRAIN = "train_sequential.py"

DEFAULT_GAMES = (
    "configs/kuhn_solvers.yaml",
    # "configs/leduc_solvers.yaml",
    # "configs/sequential_blotto_solvers.yaml",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TIME_H = 4
DEFAULT_MEMORY_G = 16
DEFAULT_GPU = False

# The config section each dotted key's prefix names, and the dataclass in
# `training/run_config.py` that validates it. `game` is deliberately absent: its
# fields are per game (`games.configs.GAME_CONFIGS`), so a `game.*` key is
# checked against the base config's own `game:` block instead.
SECTIONS = {
    "network": "NetworkConfig",
    "optimizer": "OptimizerConfig",
    "ppo": "PPOConfig",
    "train": "TrainConfig",
    "discrete": "DiscreteConfig",
    "nfsp": "NFSPConfig",
    "psro": "PSROConfig",
    "rpn": "RPNConfig",
    "scoring": "ScoringConfig",
}

# --------------------------------------------------------------------------- defaults + grids
#
# Bare value = fixed default written into every run. list = axis to sweep
# (cartesian product). tuple = fixed list-valued field (e.g. hidden widths).
# Merge order for one (game, solver) cell: COMMON < SOLVERS[solver] <
# GAME_OVERRIDES[game]. Shared numbers below match `configs/kuhn.yaml`; per-game
# differences, if any, live in GAME_OVERRIDES. `train.solver` / `train.seed` /
# `train.checkpoint_dir` are set per run and must not appear here.

COMMON: dict[str, object] = {
    # Everything below network/optimizer/ppo/train mirrors `configs/kuhn.yaml`,
    # the setting known to work, and is applied to every game.

    # network -- shared policy architecture for every solver but rpn (which only
    # reads hidden_dims / activation)
    "network.policy": "gaussian_mixture",
    "network.hidden_dims": (128, 128),
    "network.activation": "gelu",
    "network.normalization": "rms_norm",
    "network.num_components": 2,
    "network.full_covariance": False,
    "network.scale_parameterization": "log",
    "network.max_correlation": 0.0,
    "network.sigma_min": 0.1,
    "network.sigma_max": None,
    "network.bucket_means": False,
    "network.clip_means": True,
    "network.mean_box_penalty_coef": 1.0,

    # optimizer
    "optimizer.learning_rate": 0.001,
    "optimizer.max_grad_norm": 0.5,
    "optimizer.optimizer": "adam",
    "optimizer.weight_decay": 0.0,

    # ppo -- self_play / discrete_mmd read the entropy/KL terms in full; NFSP's
    # and PSRO's inner best responses zero them (see `so.br_hyperparams`)
    "ppo.clip_eps": 0.1,
    "ppo.value_coef": 0.5,
    "ppo.batch_size": 512,
    "ppo.ppo_epochs": 1,
    "ppo.target_tau": 0.001,
    "ppo.magnet_interval": 1000,
    "ppo.explore_eps": 0.2,
    "ppo.advantage": "vtrace",
    "ppo.gamma": 1.0,
    "ppo.vtrace_lambda": 0.95,
    "ppo.vtrace_rho_bar": 2.0,
    "ppo.vtrace_c_bar": 1.0,
    "ppo.vtrace_opponent_correction": "future_and_past",
    "ppo.vtrace_opponent_past_floor": 0.1,
    "ppo.category_entropy_coef": 0.04,
    "ppo.gaussian_entropy_coef": 0.04,
    "ppo.trpo_category_kl_coef": 0.05,
    "ppo.trpo_gaussian_kl_coef": 0.05,
    "ppo.magnet_category_kl_coef": 0.2,
    "ppo.magnet_gaussian_kl_coef": 0.2,
    "ppo.category_update": "ppo",
    "ppo.neurd_beta": 2.5,
    "ppo.neurd_clip": 10.0,
    "ppo.normalize_advantage": True,
    "ppo.category_floor": 0.01,
    "ppo.category_floor_coef": 0.05,
    "ppo.category_floor_mode": "entry",

    # train -- steps/epochs schedule self_play and discrete_mmd; other solvers
    # bring their own budget in SOLVERS
    "train.mode": "self_play",
    "train.steps": 200,
    "train.epochs": 1000,

    # scoring -- Kuhn's exact tree BR makes `expl` free; score_every stays 0
    "scoring.exact_grid": None,
    "scoring.score_every": 0,
    "scoring.br_steps": 100,
    "scoring.br_epochs": 20,
    "scoring.episodes": 20_000,
    "scoring.include_target": True,
}

# Sweep-only names that reuse another solver's runner. The key is what appears in
# run names / the manifest; the value is what is written to `train.solver`.
SOLVER_ALIASES: dict[str, str] = {
    "sac": "self_play",
}

# Per-solver defaults and grids. Keys here override COMMON. An empty dict means
# "this solver, on COMMON alone" -- worth keeping in, since a sweep of one
# solver is only readable next to the others on their usual budgets.
SOLVERS: dict[str, dict[str, object]] = {
    # The method under test, on COMMON alone (= `configs/kuhn.yaml`).
    # `steps * epochs` is the whole budget and the entropy bonus is what keeps
    # the mixture from collapsing onto one bet size early.
    "self_play": {},
    # Soft-Actor-Critic ablation of self_play: one Gaussian, no KL / magnet
    # pull -- entropy remains. Same schedule and entropy sweep as self_play so
    # the comparison is the mixture + KL regularizers, nothing else.
    "sac": {
        "network.num_components": 1,
        "ppo.trpo_category_kl_coef": 0.0,
        "ppo.trpo_gaussian_kl_coef": 0.0,
        "ppo.magnet_category_kl_coef": 0.0,
        "ppo.magnet_gaussian_kl_coef": 0.0,
        "ppo.batch_size": 512,
    },
    # The discretization baseline: the same self-play run on a game whose bet is
    # one of `bins` sizes. `bins` *is* the method -- too coarse and no strategy on
    # the grid is near equilibrium, too fine and the categorical head estimates a
    # distribution over hundreds of actions from the same number of hands.
    "discrete_mmd": {
        "discrete.bins": [4, 8, 16, 32],
        "discrete.max_actions": 1024,
    },
    # `br_steps * br_epochs` is the PPO budget behind each best response, and
    # everything fictitious play claims rests on that being an actual best
    # response; `eta` is how often a player plays it rather than the average.
    "nfsp": {
        "nfsp.rounds": 20,
        "nfsp.br_steps": 10000,
        "nfsp.br_epochs": 1,
        "nfsp.eta": 0.1,
        "nfsp.reservoir_capacity": 1_000_000,
        "nfsp.reservoir_episodes": 8192,
        "nfsp.sl_steps": 1000,
        "nfsp.sl_batch": 256,
    },
    # Same oracle budget question, plus the meta-solver: `uniform` is the
    # self-play-ish ablation that says how much of PSRO is the LP.
    "psro": {
        "psro.rounds": 50,
        "psro.br_steps": 10000,
        "psro.br_epochs": 1,
        "psro.payoff_episodes": 20_000,
        "psro.meta_solver": "nash",
    },
    # Zeroth order. `perturbation_batch` decides whether the pseudo-gradient
    # carries signal at all (one random direction in R^d is nearly orthogonal to
    # the gradient it estimates), and `estimator` is the IJCAI'23 per-player rule
    # against the IJCAI'25 joint one, which reads both players off one
    # perturbation for half the evaluations.
    # "rpn": {
    #     "rpn.iterations": 2000,
    #     "rpn.log_every": 50,
    #     "rpn.noise_dim": 8,
    #     "rpn.estimator": "separate",
    #     "rpn.sigma": 1.0,
    #     "rpn.utility_episodes": 64,
    #     "rpn.perturbation_batch": [64, 256],
    #     "rpn.antithetic": True,
    #     "rpn.dynamics": "simultaneous",
    #     "rpn.learning_rate": 1e-4,
    #     "rpn.optimizer": "adabelief",
    #     "rpn.max_grad_norm": 0.0,
    #     "rpn.strategy_samples": 128,
    #     "rpn.br_iterations": 200,
    # },
}

# Per-game overrides of COMMON/SOLVERS (same bare-vs-list rule). Keys must match
# an entry of `--games`. Only shared sections (network/optimizer/ppo/train/scoring)
# and the active solver's own section are applied -- an `nfsp.*` override never
# lands in a self_play run. Leduc's tree is larger, so budgets scale up here
# rather than living as a second copy of every solver block. Do not pin
# `network.num_components` here: `sac` sets it to 1 in SOLVERS, and GAME wins
# last so a game-level 3 would silently undo the ablation.
GAME_OVERRIDES: dict[str, dict[str, object]] = {
    # Empty on purpose: every game runs the `configs/kuhn.yaml` setting in COMMON.
}

# Sections that every solver reads, vs. the private section each solver owns.
# Used so GAME_OVERRIDES for `nfsp.*` do not multiply self_play's grid.
_SHARED_SECTIONS = frozenset({"network", "optimizer", "ppo", "train", "scoring", "game"})
_SOLVER_SECTIONS: dict[str, frozenset[str]] = {
    "self_play": frozenset(),
    "sac": frozenset(),
    "discrete_mmd": frozenset({"discrete"}),
    "nfsp": frozenset({"nfsp"}),
    "psro": frozenset({"psro"}),
    "rpn": frozenset({"rpn"}),
}


def train_solver(solver: str) -> str:
    """What `train.solver` receives -- aliases resolve to a real runner."""
    return SOLVER_ALIASES.get(solver, solver)


# --------------------------------------------------------------------------- validation


def config_fields() -> dict[str, dict[str, str]]:
    """`{ClassName: {field: annotation}}` for `run_config.py`'s dataclasses.

    Parsed rather than imported: this script should run anywhere, and importing
    `training.run_config` pulls in every game module just to read field names.
    """
    tree = ast.parse(RUN_CONFIG.read_text())
    return {
        node.name: {item.target.id: ast.unparse(item.annotation)
                    for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
        for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }


def solver_choices() -> list[str]:
    """`run_config.py`'s `SOLVERS`, so this file cannot drift from it."""
    tree = ast.parse(RUN_CONFIG.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "SOLVERS"):
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"no SOLVERS tuple found in {RUN_CONFIG}")


def type_ok(annotation: str, value: object) -> bool:
    """Whether `value` fits the annotation, leniently: an int is accepted for a
    float knob, and a container annotation is not checked past its head."""
    for part in (piece.strip() for piece in annotation.split("|")):
        if part == "None" and value is None:
            return True
        if part == "bool" and isinstance(value, bool):
            return True
        if isinstance(value, bool):
            continue                      # a bool is an int; do not let it pass as one
        if part == "int" and isinstance(value, int):
            return True
        if part == "float" and isinstance(value, (int, float)):
            return True
        if part == "str" and isinstance(value, str):
            return True
        if part.startswith(("tuple", "list", "Any")):
            return True
    return False


def validate(grid: dict[str, object], where: str, game_fields: set[str]) -> None:
    """Fail before writing anything if a dotted key does not exist or is mistyped."""
    fields = config_fields()
    for key, value in grid.items():
        if "." not in key:
            raise SystemExit(f"{where}: {key!r} should be a dotted path, e.g. 'nfsp.br_steps'")
        section, _, field = key.partition(".")
        if section == "game":
            if field not in game_fields:
                raise SystemExit(f"{where}: {key!r} is not a field of this game's config "
                                 f"(it has {sorted(game_fields)})")
            continue                      # per-game dataclass; the base config is the authority
        if section not in SECTIONS:
            raise SystemExit(f"{where}: no config section {section!r}; "
                             f"choices: {sorted(SECTIONS) + ['game']}")
        annotations = fields[SECTIONS[section]]
        if field not in annotations:
            near = [f for f in annotations if field in f or f in field]
            raise SystemExit(f"{where}: {key!r} is not a field of {SECTIONS[section]}"
                             + (f" (did you mean {near}?)" if near else "")
                             + f"\navailable: {sorted(annotations)}")
        for item in (value if isinstance(value, list) else [value]):
            if not type_ok(annotations[field], item):
                raise SystemExit(f"{where}: {key}={item!r} should be "
                                 f"{annotations[field]}")


# --------------------------------------------------------------------------- the plan


def merge_grid(solver: str, game: str) -> dict[str, object]:
    """COMMON < SOLVERS[solver] < relevant keys of GAME_OVERRIDES[game]."""
    allowed = _SHARED_SECTIONS | _SOLVER_SECTIONS[solver]
    game_over = {
        key: value for key, value in GAME_OVERRIDES.get(game, {}).items()
        if key.partition(".")[0] in allowed
    }
    return {**COMMON, **SOLVERS[solver], **game_over}


def variants(grid: dict[str, object]) -> list[dict[str, object]]:
    """Every combination of the swept axes, each already carrying the fixed overrides."""
    fixed = {k: v for k, v in grid.items() if not isinstance(v, list)}
    axes = {k: v for k, v in grid.items() if isinstance(v, list)}
    if not axes:
        return [dict(fixed)]
    return [{**fixed, **dict(zip(axes, combination))}
            for combination in itertools.product(*axes.values())]


def swept_axes(grid: dict[str, object]) -> list[str]:
    return [key for key, value in grid.items() if isinstance(value, list)]


def short(value: object) -> str:
    """A value as it appears in a run name: no spaces, no path separators."""
    return str(value).replace(" ", "").replace("/", "-")


def variant_tag(grid: dict[str, object], variant: dict[str, object]) -> str:
    """The swept axes of one variant, as a name fragment (empty if nothing is swept)."""
    return "__".join(f"{key.partition('.')[2]}{short(variant[key])}" for key in swept_axes(grid))


def run_name(game: str, solver: str, tag: str, seed: int) -> str:
    return f"{Path(game).stem}__{solver}" + (f"__{tag}" if tag else "") + f"__seed{seed}"


# --------------------------------------------------------------------------- writing


def load_base_config(game: str) -> dict:
    with open(REPO_ROOT / game) as f:
        raw = yaml.safe_load(f) or {}
    if "sweep" in raw:
        raise SystemExit(f"{game} is a sweep config; pass a plain train_sequential.py config")
    if "game" not in raw:
        raise SystemExit(f"{game} has no `game:` section")
    return raw


def apply_override(config: dict, key: str, value: object) -> None:
    section, _, field = key.partition(".")
    # YAML has no tuples; list-valued fields are written as sequences either way.
    if isinstance(value, tuple):
        value = list(value)
    config.setdefault(section, {})[field] = value


def write_run_config(
    path: Path,
    *,
    base: dict,
    solver: str,
    overrides: dict[str, object],
    seed: int,
    checkpoint_dir: str,
) -> None:
    """One standalone config per run: the base file with this variant folded in.

    Written out in full rather than left as `base + flags` because `meta.json`
    records the config as data, and a comparison read months later has to be able
    to say what this run's `eta` actually was without reconstructing the CLI.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        apply_override(merged, key, value)
    train = merged.setdefault("train", {})
    train["solver"] = solver
    train["seed"] = seed
    train["checkpoint_dir"] = checkpoint_dir
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)


def write_job_script(
    path: Path,
    *,
    config_rel: str,
    time_h: int,
    memory_g: int,
    gpu: bool,
) -> None:
    """One SLURM script = one training run."""
    header = prepare_default_script(time_h, memory_g, gpu)
    body = f"\npython {TRAIN} {config_rel}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body)
    path.chmod(path.stat().st_mode | 0o111)


def write_submit_all(path: Path, job_scripts: list[Path]) -> None:
    lines = ["#!/bin/sh", "", f"mkdir -p {LOG_DIR}", ""]
    for script in job_scripts:
        rel = script.relative_to(REPO_ROOT).as_posix()
        lines.append(f"sbatch -o {LOG_DIR}/{script.stem}.log {rel}")
    lines.append("")
    lines.append(f'echo "submitted {len(job_scripts)} jobs; logs in {LOG_DIR}"')
    path.write_text("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def write_manifest(path: Path, runs: list[dict]) -> None:
    """What each run is, for whatever reads the sweep back -- so an analysis
    script never has to parse a directory name apart."""
    path.write_text(json.dumps({"checkpoint_root": CHECKPOINT_ROOT, "runs": runs}, indent=2) + "\n")


# --------------------------------------------------------------------------- generation


def generate(
    games: list[str],
    solvers: list[str],
    seeds: list[int],
    time_h: int,
    memory_g: int,
    gpu: bool,
    dry_run: bool = False,
) -> list[Path]:
    bases = {game: load_base_config(game) for game in games}
    for game in games:
        game_fields = set(bases[game]["game"]) - {"name"}
        validate(COMMON, "COMMON", game_fields)
        for solver in solvers:
            validate(SOLVERS[solver], f"SOLVERS[{solver!r}]", game_fields)
        if game in GAME_OVERRIDES:
            validate(GAME_OVERRIDES[game], f"GAME_OVERRIDES[{game!r}]", game_fields)

    plan: list[dict] = []
    for game in games:
        for solver in solvers:
            grid = merge_grid(solver, game)
            axes = swept_axes(grid)
            for variant in variants(grid):
                tag = variant_tag(grid, variant)
                for seed in seeds:
                    name = run_name(game, solver, tag, seed)
                    plan.append({
                        "name": name,
                        "game": game,
                        "solver": solver,                 # sweep / directory name
                        "train_solver": train_solver(solver),  # what the config writes
                        "seed": seed,
                        "overrides": {k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in sorted(variant.items())},
                        "swept": {k: variant[k] for k in axes},
                        "checkpoint_dir": f"{CHECKPOINT_ROOT}/{name}",
                        "config": (CONFIGS_DIR / f"{name}.yaml").relative_to(REPO_ROOT).as_posix(),
                    })

    if dry_run:
        print(f"{len(plan)} runs = {len(plan) // max(len(seeds), 1)} settings x "
              f"{len(seeds)} seeds\n")
        for run in plan:
            settings = " ".join(f"{k}={v}" for k, v in run["swept"].items()) or "(defaults)"
            print(f"  {run['name']:<64} {settings}")
        return []

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    job_scripts: list[Path] = []
    for run in plan:
        config_path = REPO_ROOT / run["config"]
        write_run_config(
            config_path,
            base=bases[run["game"]],
            solver=run["train_solver"],
            overrides=run["overrides"],
            seed=run["seed"],
            checkpoint_dir=run["checkpoint_dir"],
        )
        script_path = SCRIPTS_DIR / f"{run['name']}.sh"
        write_job_script(
            script_path,
            config_rel=run["config"],
            time_h=time_h,
            memory_g=memory_g,
            gpu=gpu,
        )
        written.append(config_path)
        job_scripts.append(script_path)
        written.append(script_path)

    submit_all = SCRIPTS_DIR / "run_all.sh"
    write_submit_all(submit_all, job_scripts)
    manifest = SCRIPTS_DIR / "manifest.json"
    write_manifest(manifest, plan)
    return written + [submit_all, manifest]


def main() -> None:
    real_solvers = set(solver_choices())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES),
                    help="base sequential configs; each run's config is one of these with "
                         "the variant folded in (game: block comes from here)")
    ap.add_argument("--solvers", nargs="+", default=sorted(SOLVERS), choices=sorted(SOLVERS),
                    help="which SOLVERS entries to generate; aliases (e.g. sac) map via "
                         "SOLVER_ALIASES to a real train.solver")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                    help="one run per seed -- each gets its own config, script and directory")
    ap.add_argument("--time", type=int, default=DEFAULT_TIME_H, dest="time_h",
                    help="wall-time hours for #SBATCH --time")
    ap.add_argument("--memory", type=int, default=DEFAULT_MEMORY_G, dest="memory_g",
                    help="memory in GB for #SBATCH --mem")
    ap.add_argument("--gpu", action="store_true", default=DEFAULT_GPU,
                    help="request one GPU (changes the partition)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    args = ap.parse_args()

    missing = [solver for solver in args.solvers if solver not in SOLVERS]
    if missing:
        raise SystemExit(f"no entry for {missing}; add one to SOLVERS in {Path(__file__).name} "
                         "(an empty dict means 'COMMON alone')")
    bad_alias = [f"{alias}->{target}" for alias, target in SOLVER_ALIASES.items()
                 if alias in args.solvers and target not in real_solvers]
    if bad_alias:
        raise SystemExit(f"SOLVER_ALIASES target not in run_config SOLVERS: {bad_alias}")
    unknown = [s for s in args.solvers if train_solver(s) not in real_solvers]
    if unknown:
        raise SystemExit(f"{unknown} is not a run_config solver and not in SOLVER_ALIASES")

    written = generate(
        games=args.games,
        solvers=args.solvers,
        seeds=args.seeds,
        time_h=args.time_h,
        memory_g=args.memory_g,
        gpu=args.gpu,
        dry_run=args.dry_run,
    )
    if not written:
        return
    runs = (len(written) - 2) // 2
    print(f"wrote {runs} configs + {runs} job scripts (one training run each) "
          f"+ run_all.sh + manifest.json")
    print(f"  configs    : {CONFIGS_DIR.relative_to(REPO_ROOT)}")
    print(f"  scripts    : {SCRIPTS_DIR.relative_to(REPO_ROOT)}")
    print(f"  checkpoints: {CHECKPOINT_ROOT}/{{run_name}}")
    print(f"\nsubmit: bash {(SCRIPTS_DIR / 'run_all.sh').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
