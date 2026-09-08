"""Hand-tune SPG / JPSPG knobs on one game and see, per run, whether it collapsed.

``run.py`` runs a fixed staged sweep and ranks by final exploitability. This is the
knob-twiddling counterpart: every hyperparameter is a CLI flag that takes *one or more*
values, the cartesian product is run, and the summary prints the diagnostics that the
ranking table hides.

That last part is the point. The staged sweep's tables are misleading on their own,
because the dominant failure mode of this estimator does not look like a bad number --
it looks like a *frozen* one. Small ``sigma`` drives the policy into tanh saturation:
every sampled action pins to the box boundary, the policy becomes a point mass,
``grad_norm`` decays to ~1e-7, and exploitability sits at a constant for the rest of the
run. On ``two_point`` that collapse *out-ranks* every healthy run, so a table sorted by
final exploitability recommends it. So each run here also reports

    spread    std of the sampled actions, as a fraction of the box width
    pinned    fraction of sampled coordinates within 0.5% of a box edge
    dexpl     change in exploitability over the last quarter of the run

and is flagged ``COLLAPSED`` when the policy has degenerated to a (near-)pure strategy,
whatever its exploitability says.

Knobs that ``run.py`` cannot reach are exposed here too, because they are the next
suspects: ``--max-grad-norm`` (the baseline disables clipping, so nothing bounds the
weight norm over hundreds of thousands of AdaBelief steps -- the likeliest cause of the
saturation), ``--noise-dim``, ``--hidden-dims``, ``--activation`` and ``--no-antithetic``.

Runs land in the same directory layout as ``one_shot_neural`` / ``run.py``, so
``score.py`` and ``run.py --report`` work on this tree unchanged.

Examples::

    # one run, paper defaults, cheap budget -- the interactive default
    python experiments/pseudo_gradient_convergence/tune.py

    # the recommended setting, on the game that never converged
    python experiments/pseudo_gradient_convergence/tune.py \\
        --game configs/two_point.yaml --sigma 1.0 --utility-samples 1024

    # sweep two knobs at once (4 runs) and rank them
    python experiments/pseudo_gradient_convergence/tune.py \\
        --sigma 0.1 1.0 --utility-samples 256 1024

    # does gradient clipping stop the small-sigma collapse?
    python experiments/pseudo_gradient_convergence/tune.py \\
        --game configs/circle.yaml --sigma 0.1 --max-grad-norm 0 0.5 --seed 0 1

    # how many iterations does a budget buy, without running anything
    python experiments/pseudo_gradient_convergence/tune.py --sigma 0.01 1.0 --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ONE_SHOT = REPO / "experiments" / "one_shot_neural"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ONE_SHOT))

import numpy as np  # noqa: E402

from baselines.common import GridOracle, StrategyPair  # noqa: E402
from baselines.neural import jpspg as jpspg_module  # noqa: E402
from baselines.neural import randomized_policy as spg_module  # noqa: E402
from baselines.neural.common import RunWriter, load_run  # noqa: E402
from run_cell import _aligned, _git_commit  # noqa: E402

DEFAULT_OUT = "data/pseudo_gradient_tuning"
ESTIMATORS = {"spg": spg_module.spg_pseudo_gradients,
              "jpspg": jpspg_module.jpspg_pseudo_gradients}

# What a knob is called in the run tag. Only knobs that were actually swept appear in it,
# so a one-knob sweep gives short, readable directory names.
TAG_KEYS = (
    ("method", "", "{}"),
    ("sigma", "sigma", "{:g}"),
    ("utility_samples", "us", "{}"),
    ("perturbation_batch", "pb", "{}"),
    ("dynamics", "dyn", "{}"),
    ("lr", "lr", "{:g}"),
    ("optimizer", "opt", "{}"),
    ("noise_dim", "nz", "{}"),
    ("max_grad_norm", "clip", "{:g}"),
    ("antithetic", "anti", "{}"),
    ("hidden_dims", "net", "{}"),
    ("activation", "act", "{}"),
)


# --------------------------------------------------------------------------- planning


def iterations_for(budget: int, method: str, utility_samples: int, dynamics: str,
                   perturbation_batch: int = 2) -> tuple[int, int]:
    """`(iterations, evals_per_iteration)` -- the cost formula `run_cell.plan_units` uses.

    Each iteration draws `perturbation_batch` perturbations. SPG perturbs one player at a
    time, so it evaluates the utility twice per perturbation; JPSPG perturbs both jointly
    and evaluates once. Extragradient estimates twice per iteration on top of that.
    """
    per = utility_samples * perturbation_batch * (1 if method == "jpspg" else 2)
    if dynamics == "extragradient":
        per *= 2
    return max(int(budget // per), 1), per


def variant_tag(combo: dict, swept: set[str]) -> str:
    """A directory name carrying the method plus whichever knobs this sweep varied."""
    parts = []
    for key, prefix, fmt in TAG_KEYS:
        if key != "method" and key not in swept:
            continue
        value = combo[key]
        if key == "hidden_dims":
            value = "x".join(str(d) for d in value)
        elif key == "antithetic":
            value = "yes" if value else "no"
        text = fmt.format(value)
        parts.append(text if not prefix else f"{prefix}-{text}")
    return "__".join(parts)


# --------------------------------------------------------------------------- diagnostics


def diagnose(directory: Path, low: np.ndarray, high: np.ndarray, edge: float = 0.005) -> dict:
    """Collapse diagnostics from a finished run's checkpoints.

    `spread` and `pinned` are what separate "converged" from "saturated": a policy that
    has degenerated to a point mass has spread ~0 and, because the squash is a tanh, is
    almost always pinned against a box edge as well.
    """
    paths = sorted((directory / "checkpoints").glob("*.npz"))
    if not paths:
        return {}
    width = np.maximum(high - low, 1e-12)
    pair = StrategyPair.load(paths[-1])

    spreads, pinned = [], []
    for support in (pair.support_0, pair.support_1):
        actions = np.asarray(support, dtype=np.float64).reshape(len(support), -1)
        spreads.append(float(np.mean(actions.std(axis=0) / width)))
        at_edge = (actions <= low + edge * width) | (actions >= high - edge * width)
        pinned.append(float(at_edge.mean()))

    history = json.loads((directory / "history.json").read_text())
    grad_norms = [row["grad_norm_0"] for row in history if row.get("grad_norm_0") is not None]
    scores_path = directory / "scores.json"
    expls = []
    if scores_path.exists():
        expls = [row["expl"] for row in json.loads(scores_path.read_text())["scores"]]
    elif all("expl" in row for row in history):
        expls = [row["expl"] for row in history]

    tail = expls[max(len(expls) * 3 // 4, 1):] if expls else []
    out = {
        "spread": min(spreads),
        "pinned": max(pinned),
        "grad_norm": grad_norms[-1] if grad_norms else None,
        "final_expl": expls[-1] if expls else None,
        "best_expl": min(expls) if expls else None,
        # Movement over the last quarter: a collapsed run is flat here, a healthy one is
        # still moving (for better or worse).
        "tail_range": (max(tail) - min(tail)) if len(tail) > 1 else None,
    }
    out["collapsed"] = bool(out["spread"] < 0.01 or out["pinned"] > 0.9)
    return out


def verdict(row: dict) -> str:
    if row.get("status") != "ok":
        return "FAILED"
    if row.get("collapsed"):
        return "COLLAPSED"
    if row.get("tail_range") is not None and row["tail_range"] < 1e-4:
        return "frozen"
    return "ok"


# --------------------------------------------------------------------------- one run


def run_one(combo: dict, *, game_path: str, budget: int, seed: int, out_root: Path,
            checkpoints: int, grid: int, samples: int, score: bool, tag: str,
            overwrite: bool) -> dict:
    game_tag = Path(game_path).stem
    directory = out_root / game_tag / tag / f"seed{seed}"
    iterations, per_iteration = iterations_for(
        budget, combo["method"], combo["utility_samples"], combo["dynamics"],
        combo["perturbation_batch"])
    row = {"game": game_tag, "method": tag, "estimator": combo["method"], "seed": seed,
           "budget": budget, "config": game_path, "dir": str(directory),
           "iterations": iterations, "evals_per_iteration": per_iteration,
           # `ablation` is the key `run.py --report` reads knobs back out of.
           "ablation": {**combo, "hidden_dims": list(combo["hidden_dims"])}}

    print(f"\n=== {tag}  seed{seed} ===")
    print(f"  {iterations} iterations x {per_iteration} evals = "
          f"{iterations * per_iteration:.4g} payoff evals")

    if (directory / "meta.json").exists() and not overwrite:
        print(f"  skip (already done, --overwrite to redo): {directory}")
        stored = json.loads((directory / "meta.json").read_text())
        return {**row, "status": stored.get("status", "ok"),
                "train_seconds": stored.get("train_seconds"),
                **diagnose(directory, *_bounds_from_meta(stored))}

    directory.mkdir(parents=True, exist_ok=True)
    game, game_config, config = load_run(game_path)
    oracle = GridOracle(game, points=grid)
    hyperparams = spg_module.RandomizedPolicyHyperparams(
        action_dim=len(spg_module.action_bounds(game, 0)[0]),
        hidden_dims=tuple(combo["hidden_dims"] or config.network.hidden_dims),
        noise_dim=combo["noise_dim"],
        activation=combo["activation"] or config.network.activation,
        low=tuple(float(x) for x in spg_module.action_bounds(game, 0)[0]),
        high=tuple(float(x) for x in spg_module.action_bounds(game, 0)[1]),
        learning_rate=combo["lr"],
        optimizer=combo["optimizer"],
        max_grad_norm=combo["max_grad_norm"],
        sigma=combo["sigma"],
        utility_samples=combo["utility_samples"],
        perturbation_batch=combo["perturbation_batch"],
        antithetic=combo["antithetic"],
        dynamics=combo["dynamics"],
    )
    iterations, log_every = _aligned(iterations, checkpoints)
    meta = {**row, "iterations": iterations, "log_every": log_every,
            "hyperparams": hyperparams.to_dict(),
            "game_config": dataclasses.asdict(game_config),
            "scored_during_run": score, "grid": grid,
            "commit": _git_commit(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    writer = RunWriter(directory, meta)

    started = time.monotonic()
    try:
        result = spg_module.run_pseudo_gradient(
            game, oracle, hyperparams, ESTIMATORS[combo["method"]],
            iterations=iterations, log_every=log_every, samples=samples, seed=seed,
            writer=writer, estimator_name=combo["method"], score=score)
    except Exception as exc:  # noqa: BLE001
        (directory / "error.txt").write_text(traceback.format_exc())
        print(f"  FAILED after {time.monotonic() - started:.1f}s: {type(exc).__name__}: {exc}")
        return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    history = result["history"]
    writer.finish({"status": "ok",
                   "train_seconds": result.get("train_seconds", 0.0),
                   "compile_seconds": result.get("compile_seconds", 0.0),
                   "payoff_evals": history[-1].get("payoff_evals"),
                   "checkpoints": len(writer.checkpoints.entries)})
    stats = diagnose(directory, np.asarray(hyperparams.low), np.asarray(hyperparams.high))
    row = {**row, "status": "ok", "train_seconds": result.get("train_seconds", 0.0),
           "compile_seconds": result.get("compile_seconds", 0.0),
           "payoff_evals": history[-1].get("payoff_evals"), **stats}
    print(f"  {verdict(row)}: " + _stats_line(row) + f"  -> {directory}")
    return row


def _bounds_from_meta(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    hyperparams = meta.get("hyperparams", {})
    return (np.asarray(hyperparams.get("low", [0.0])),
            np.asarray(hyperparams.get("high", [1.0])))


def _stats_line(row: dict) -> str:
    def fmt(key: str, spec: str) -> str:
        value = row.get(key)
        return "n/a" if value is None else format(value, spec)
    return (f"expl {fmt('final_expl', '+.5f')} (best {fmt('best_expl', '+.5f')})  "
            f"spread {fmt('spread', '.4f')}  pinned {fmt('pinned', '.2f')}  "
            f"grad {fmt('grad_norm', '.2e')}")


# --------------------------------------------------------------------------- summary


def summarize(rows: list[dict], out_root: Path, game_tag: str, swept: set[str]) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    # `--no-score` leaves `final_expl` empty; rank those by spread instead, so the
    # collapse diagnostics are still readable before `score.py` runs.
    scored = all(r.get("final_expl") is not None for r in ok)
    ok.sort(key=(lambda r: r["final_expl"]) if scored else (lambda r: -r.get("spread", 0.0)))
    columns = ["run", "final expl", "best expl", "spread", "pinned", "grad norm",
               "tail range", "train s", "verdict"]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join("-" for _ in columns) + "|"]
    for row in ok:
        lines.append("|{}|{}|{}|{:.4f}|{:.2f}|{}|{}|{:.0f}|{}|".format(
            f"{row['method']} seed{row['seed']}",
            "n/a" if row["final_expl"] is None else f"{row['final_expl']:+.5f}",
            "n/a" if row["best_expl"] is None else f"{row['best_expl']:+.5f}",
            row["spread"], row["pinned"],
            "n/a" if row["grad_norm"] is None else f"{row['grad_norm']:.2e}",
            "n/a" if row["tail_range"] is None else f"{row['tail_range']:.4f}",
            row.get("train_seconds") or 0.0, verdict(row)))
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"|{row['method']} seed{row['seed']}|FAILED: "
                         f"{row.get('error', '?')}|||||||")
    table = "\n".join(lines) + "\n"

    print("\n" + table)
    if not scored:
        print("ranked by spread -- `--no-score` skipped exploitability; score it with "
              f"`python experiments/one_shot_neural/score.py --out {out_root} "
              f"--games {game_tag}`")
    healthy = [r for r in ok if not r.get("collapsed")]
    if healthy and scored:
        best = healthy[0]
        print(f"best non-collapsed: {best['method']} seed{best['seed']}  "
              f"final expl {best['final_expl']:+.5f}")
        if ok[0] is not best:
            print(f"note: {ok[0]['method']} scores lower ({ok[0]['final_expl']:+.5f}) but its "
                  "policy collapsed to a (near-)pure strategy -- that number is not a "
                  "converged one.")
    elif ok and not healthy:
        print("every run collapsed to a (near-)pure strategy: the policy is a point mass "
              "against the box edge, so its exploitability is not a converged number. "
              "Raise --sigma.")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{game_tag}_tuning.md").write_text(table)
    (out_root / f"{game_tag}_tuning.json").write_text(json.dumps(
        {"game": game_tag, "swept": sorted(swept), "rows": rows}, indent=2, default=str))
    print(f"table -> {out_root / f'{game_tag}_tuning.md'}")
    print(f"rows  -> {out_root / f'{game_tag}_tuning.json'}")
    print(f"curves: python experiments/one_shot_neural/score.py --out {out_root} "
          f"--game {game_tag} --plot")


# --------------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="configs/circle.yaml")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--budget", type=int, default=200_000_000,
                    help="payoff evaluations per run (the unit the sweeps use)" )
    ap.add_argument("--checkpoints", type=int, default=20)
    ap.add_argument("--grid", type=int, default=401, help="deviation grid for exploitability")
    ap.add_argument("--samples", type=int, default=4096,
                    help="actions sampled from the policy per checkpoint")
    ap.add_argument("--no-score", action="store_true",
                    help="skip in-training exploitability (score later with score.py)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the combinations and their iteration counts, run nothing")

    knobs = ap.add_argument_group(
        "knobs", "each takes one or more values; the cartesian product is run")
    knobs.add_argument("--method", nargs="+", default=["spg", "jpspg"], choices=sorted(ESTIMATORS))
    knobs.add_argument("--sigma", nargs="+", type=float, default=[2.0],
                       help="parameter-space smoothing radius (paper: 0.1)")
    knobs.add_argument("--utility-samples", nargs="+", type=int, default=[64])
    knobs.add_argument("--perturbation-batch", nargs="+", type=int, default=[256],
                       help="perturbations averaged per iteration (the papers' "
                            "`batch_size`: their code defaults to 2, their experiments "
                            "use 256). Below ~32 the pseudo-gradient is mostly noise.")
    knobs.add_argument("--dynamics", nargs="+", default=["extragradient"],
                       choices=["simultaneous", "optimistic", "extragradient"])
    knobs.add_argument("--lr", nargs="+", type=float, default=[3e-4])
    knobs.add_argument("--optimizer", nargs="+", default=["adabelief"],
                       help="see training.optimizers.OPTIMIZERS")
    knobs.add_argument("--noise-dim", nargs="+", type=int, default=[4])
    knobs.add_argument("--max-grad-norm", nargs="+", type=float, default=[1.0],
                       help="0 disables clipping (the baseline's default)")
    knobs.add_argument("--no-antithetic", action="store_true",
                       help="single-point estimator instead of the central difference")
    knobs.add_argument("--hidden-dims", nargs="+", type=int, default=[8, 8],
                       help="policy width, e.g. --hidden-dims 64 64 (default: the config's)")
    knobs.add_argument("--activation", default=None, help="default: the config's")
    knobs.add_argument("--seed", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    axes = {
        "method": args.method,
        "sigma": args.sigma,
        "utility_samples": args.utility_samples,
        "perturbation_batch": args.perturbation_batch,
        "dynamics": args.dynamics,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "noise_dim": args.noise_dim,
        "max_grad_norm": args.max_grad_norm,
    }
    swept = {key for key, values in axes.items() if len(values) > 1}
    fixed = {
        "antithetic": not args.no_antithetic,
        "hidden_dims": tuple(args.hidden_dims) if args.hidden_dims else (),
        "activation": args.activation,
    }

    combos = [dict(zip(axes, values), **fixed)
              for values in itertools.product(*axes.values())]
    # Everything in `tagged` goes into the directory name: the swept axes, plus any
    # single-valued knob set away from its default (otherwise two different runs would
    # share a directory and the second would be skipped as "already done").
    tagged = set(swept)
    if args.no_antithetic:
        tagged.add("antithetic")
    if args.hidden_dims:
        tagged.add("hidden_dims")
    if args.activation:
        tagged.add("activation")
    out_root, game_tag = Path(args.out), Path(args.game).stem
    print(f"game    : {args.game}")
    print(f"budget  : {args.budget:.4g} payoff evals per run")
    print(f"sweeping: {', '.join(sorted(swept)) or '(nothing -- single setting)'}")
    print(f"runs    : {len(combos)} x {len(args.seed)} seed(s) = "
          f"{len(combos) * len(args.seed)}")

    if args.dry_run:
        for combo in combos:
            iterations, per = iterations_for(
                args.budget, combo["method"], combo["utility_samples"], combo["dynamics"],
                combo["perturbation_batch"])
            print(f"  {variant_tag(combo, tagged):<50s} {iterations:>9d} iters "
                  f"x {per} evals")
        return

    rows = []
    for combo in combos:
        tag = variant_tag(combo, tagged)
        for seed in args.seed:
            rows.append(run_one(
                combo, game_path=args.game, budget=args.budget, seed=seed,
                out_root=out_root, checkpoints=args.checkpoints, grid=args.grid,
                samples=args.samples, score=not args.no_score, tag=tag,
                overwrite=args.overwrite))

    summarize(rows, out_root, game_tag, swept)
    if any(row.get("status") != "ok" for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
