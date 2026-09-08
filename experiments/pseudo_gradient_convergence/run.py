"""Diagnose / fix SPG and JPSPG non-convergence on one-shot games.

In ``experiments/one_shot_neural`` both methods stayed near their initial
exploitability on every game (see ``data/one_shot_neural/summary.md``). The
follow-up ``ablate_jpspg.py`` swept dynamics / lr / optimizer on ``two_point``
and still finished around expl ~2. This experiment probes the knobs that
ablation left alone — the ones that control *zeroth-order gradient quality*:

  1. **baseline** — paper defaults (AdaBelief, lr=1e-4, sigma=0.1, us=256,
     simultaneous) for both ``spg`` and ``jpspg``
  2. **sigma** — smoothing radius ``{0.01, 0.05, 0.1, 0.3, 1.0}``
  3. **samples** — ``utility_samples`` ``{64, 256, 1024, 4096}``
  4. **dynamics** — simultaneous / optimistic / extragradient at the best
     (sigma, samples) so far

Default game is ``matching_pennies``: bilinear, unique Nash at the origin, so
if the estimator and ascent are sound the last-iterate exploitability should
fall. Re-run with ``--game configs/two_point.yaml`` once a setting works (or to
confirm the same failure mode as ``one_shot_neural``).

Layout (same as ``one_shot_neural``, so ``score.py`` works unchanged)::

    <out>/<game>/<variant_tag>/seed<k>/

Examples::

    # smoke: stage 1 only, cheap budget
    python experiments/pseudo_gradient_convergence/run.py \\
        --stages 1 --budget 500000 --seed 0

    # full staged search on matching pennies (auto-picks between stages)
    python experiments/pseudo_gradient_convergence/run.py \\
        --budget 20000000 --seed 0 --score-between

    # confirm a candidate on two_point
    python experiments/pseudo_gradient_convergence/run.py \\
        --game configs/two_point.yaml --stages 1 2 \\
        --sigma 0.05 --utility-samples 1024 --budget 20000000

    # score + rank + plot
    python experiments/pseudo_gradient_convergence/run.py --report \\
        --out data/pseudo_gradient_convergence
"""

from __future__ import annotations

import argparse
import dataclasses
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

from baselines.common import GridOracle  # noqa: E402
from baselines.neural.common import RunWriter, load_run  # noqa: E402
from run_cell import (  # noqa: E402
    ACCESS_MODEL,
    RUNNERS,
    Settings,
    _aligned,
    _git_commit,
    budget_warning,
    compile_warning,
    plan_units,
)
from training.hyperparams import build_hyperparams  # noqa: E402

DEFAULT_OUT = "data/pseudo_gradient_convergence"
METHODS = ("spg", "jpspg")
SIGMAS = (0.01, 0.05, 0.1, 0.3, 1.0)
UTILITY_SAMPLES = (64, 256, 1024, 4096)
DYNAMICS = ("simultaneous", "optimistic", "extragradient")

# Paper defaults used by one_shot_neural / randomized_policy.py.
PAPER_SIGMA = 0.1
PAPER_UTILITY_SAMPLES = 256
# JPSPG's experiments average 256 perturbations per iteration. Their reference
# implementation's `pgrad(..., batch_size=2)` *default* is one antithetic pair, which is
# what this tree's stored runs used -- and at 2 the pseudo-gradient is ~99% noise, so
# every ranking below it was ranking noise. See `RandomizedPolicyHyperparams`.
PAPER_PERTURBATION_BATCH = 256
PAPER_LR = 1e-4
PAPER_OPTIMIZER = "adabelief"
PAPER_DYNAMICS = "simultaneous"
PAPER_NOISE_DIM = 8
PAPER_MAX_GRAD_NORM = 0.0        # the baseline disables clipping
PAPER_METRIC_SAMPLES = 4096      # actions per checkpoint, as this tree's stored runs used


def _fmt_float(x: float) -> str:
    return f"{x:.0e}".replace("+", "").replace("-0", "-")


def variant_tag(
    method: str,
    *,
    sigma: float,
    utility_samples: int,
    dynamics: str,
    lr: float = PAPER_LR,
    optimizer: str = PAPER_OPTIMIZER,
) -> str:
    return (
        f"{method}__sigma-{_fmt_float(sigma)}__us-{utility_samples}"
        f"__dyn-{dynamics}__lr-{_fmt_float(lr)}__opt-{optimizer}"
    )


def short_label(method: str, sigma: float, utility_samples: int, dynamics: str) -> str:
    return f"{method} | σ={sigma:g} | us={utility_samples} | {dynamics}"


def _parse_tag(tag: str) -> dict:
    method, *rest = tag.split("__")
    parts = dict(piece.split("-", 1) for piece in rest)
    return {
        "method": method,
        "sigma": float(parts["sigma"]),
        "utility_samples": int(parts["us"]),
        "dynamics": parts["dyn"],
        "lr": float(parts["lr"]),
        "optimizer": parts["opt"],
    }


def _final_expl(directory: Path) -> float | None:
    scores_path = directory / "scores.json"
    if not scores_path.exists():
        return None
    scores = json.loads(scores_path.read_text())["scores"]
    if not scores:
        return None
    return float(scores[-1]["expl"])


def _score_run(directory: Path, grid: int, overwrite: bool) -> float | None:
    from baselines.common import load_game
    from score import score_run

    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("status") == "failed":
        return None
    game, _ = load_game(meta["config"])
    scores = score_run(directory, GridOracle(game, points=grid), overwrite=overwrite)
    return float(scores[-1]["expl"]) if scores else None


def _pick_best(
    out_root: Path,
    game: str,
    seed: int,
    tags: list[str],
    grid: int,
    score: bool,
) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for tag in tags:
        directory = out_root / game / tag / f"seed{seed}"
        if not (directory / "meta.json").exists():
            continue
        expl = _final_expl(directory)
        if expl is None and score:
            expl = _score_run(directory, grid=grid, overwrite=False)
        if expl is None:
            continue
        if best is None or expl < best[1]:
            best = (tag, expl)
    return best


def find_runs(out_root: Path, game: str | None = None) -> list[Path]:
    runs = sorted(p.parent for p in out_root.glob("*/*/*/meta.json"))
    if game is None:
        return runs
    return [p for p in runs if p.parts[-3] == game]


def report(out_root: Path, game: str, grid: int, overwrite_scores: bool = False) -> Path:
    """Score every variant, rank by final exploitability, plot curves."""
    from baselines.common import load_game
    from score import plot_game, score_run

    runs = find_runs(out_root, game)
    if not runs:
        raise SystemExit(f"no runs under {out_root / game}")

    oracles: dict[str, GridOracle] = {}
    rows: list[dict] = []
    curves: dict = {}

    print(f"scoring {len(runs)} variants under {out_root / game} (grid={grid})")
    for directory in runs:
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("status") == "failed":
            print(f"  skip failed: {directory}")
            continue
        config = meta["config"]
        if config not in oracles:
            game_obj, _ = load_game(config)
            oracles[config] = GridOracle(game_obj, points=grid)
        scores = score_run(directory, oracles[config], overwrite=overwrite_scores)
        ablation = meta.get("ablation") or {}
        if not ablation and "__sigma-" in meta.get("method", ""):
            ablation = _parse_tag(meta["method"])
        expls = [s["expl"] for s in scores]
        final = scores[-1] if scores else {}
        method = ablation.get("method", meta.get("method", "?").split("__")[0])
        sigma = float(ablation.get("sigma", float("nan")))
        us = int(ablation.get("utility_samples", -1))
        dynamics = ablation.get("dynamics", "?")
        row = {
            "tag": meta.get("method"),
            "method": method,
            "sigma": sigma,
            "utility_samples": us,
            "dynamics": dynamics,
            "final_expl": expls[-1] if expls else float("nan"),
            "best_expl": min(expls) if expls else float("nan"),
            "payoff_evals": final.get("payoff_evals"),
            "train_seconds": meta.get("train_seconds"),
            "label": short_label(method, sigma, us, dynamics),
        }
        rows.append(row)
        curves[str(directory.relative_to(out_root))] = {
            "meta": {
                **{k: meta.get(k) for k in (
                    "game", "method", "seed", "budget", "access_model", "train_seconds")},
                "method": row["label"],
            },
            "scores": scores,
        }
        print(f"  {row['label']:<55s}  final {row['final_expl']:+.5f}  "
              f"best {row['best_expl']:+.5f}")

    rows.sort(key=lambda r: (r["final_expl"], r["best_expl"]))
    lines = [
        "|rank|method|sigma|utility_samples|dynamics|final expl|best expl|train s|",
        "|-|-|-|-|-|-|-|-|",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"|{i}|{row['method']}|{row['sigma']:g}|{row['utility_samples']}|"
            f"{row['dynamics']}|{row['final_expl']:+.5f}|{row['best_expl']:+.5f}|"
            f"{row['train_seconds']:.0f}|")
    table = "\n".join(lines) + "\n"
    table_path = out_root / f"{game}_ranking.md"
    table_path.write_text(table)
    (out_root / "curves.json").write_text(json.dumps(curves, indent=2))
    plot_path = plot_game(curves, game, out_root / f"{game}_curves.png")

    print(f"\n{table}")
    if rows:
        best = rows[0]
        print(f"winner (lowest final expl): {best['label']}  "
              f"final={best['final_expl']:+.5f}  best={best['best_expl']:+.5f}")
        # Matching pennies / two_point: "converged" roughly means last-iterate
        # exploitability well below the random baseline (~1 and ~2 respectively).
        if best["final_expl"] < 0.1:
            print("looks converged (final expl < 0.1). Re-check on two_point / continuum games.")
        else:
            print("still not converging under these knobs — next places to look: "
                  "average iterate, network width/noise_dim, or a bug in the estimator.")
    print(f"ranking -> {table_path}")
    print(f"plot    -> {plot_path}")
    return table_path


def run_variant(
    *,
    game: str,
    method: str,
    seed: int,
    budget: int,
    out_root: Path,
    sigma: float,
    utility_samples: int,
    dynamics: str,
    lr: float,
    optimizer: str,
    base: Settings,
    overwrite: bool,
) -> tuple[dict, str]:
    tag = variant_tag(
        method, sigma=sigma, utility_samples=utility_samples,
        dynamics=dynamics, lr=lr, optimizer=optimizer)
    game_tag = Path(game).stem
    directory = out_root / game_tag / tag / f"seed{seed}"
    settings = dataclasses.replace(
        base,
        sigma=sigma,
        utility_samples=utility_samples,
        dynamics=dynamics,
        pseudo_lr=lr,
        pseudo_optimizer=optimizer,
    )
    row = {
        "game": game_tag,
        "method": tag,
        "seed": seed,
        "budget": budget,
        "config": game,
        "dir": str(directory),
        "access_model": ACCESS_MODEL[method],
        "ablation": {
            "method": method,
            "sigma": sigma,
            "utility_samples": utility_samples,
            "dynamics": dynamics,
            "pseudo_lr": lr,
            "pseudo_optimizer": optimizer,
        },
    }

    print(f"\n=== {tag} ===")
    if (directory / "meta.json").exists() and not overwrite:
        stored = json.loads((directory / "meta.json").read_text())
        print(f"  skip (already done): {directory}")
        return {**row, **{k: stored.get(k) for k in (
            "status", "train_seconds", "compile_seconds", "payoff_evals", "checkpoints")}}, tag

    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        game_obj, game_cfg, config = load_run(game)
        oracle = GridOracle(game_obj, points=settings.grid)
        batch = build_hyperparams(game_obj, 0, config).num_envs
        plan = plan_units(method, budget, settings, batch)
        for warning in (budget_warning(method, plan, settings, budget),
                        compile_warning(method, plan, settings)):
            if warning:
                print(f"  {warning}")
        for key in ("iterations", "rounds"):
            if key in plan:
                plan[key], plan["log_every"] = _aligned(plan[key], settings.checkpoints)
        meta = {
            **row,
            "plan": plan,
            "settings": dataclasses.asdict(settings),
            "game_config": dataclasses.asdict(game_cfg),
            "scored_during_run": False,
            "commit": _git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        writer = RunWriter(directory, meta)
        result = RUNNERS[method](
            game_obj, oracle, config, plan, settings, seed, writer, False)
    except Exception as exc:  # noqa: BLE001
        (directory / "error.txt").write_text(traceback.format_exc())
        print(f"  FAILED after {time.monotonic() - started:.1f}s: "
              f"{type(exc).__name__}: {exc}")
        return {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}, tag

    history = result["history"]
    meta = writer.finish({
        "status": "ok",
        "train_seconds": result.get("train_seconds", history[-1].get("wall_time", 0.0)),
        "compile_seconds": result.get("compile_seconds", history[-1].get("compile_time", 0.0)),
        "elapsed_seconds": time.monotonic() - started,
        "payoff_evals": history[-1].get("payoff_evals"),
        "checkpoints": len(writer.checkpoints.entries),
        "units": plan,
    })
    print(f"  done: {meta['payoff_evals']:.3g} payoff evals in {meta['train_seconds']:.1f}s "
          f"training (+{meta['compile_seconds']:.1f}s compile), "
          f"{meta['checkpoints']} checkpoints -> {directory}")
    return {**row, "status": "ok", "train_seconds": meta["train_seconds"],
            "compile_seconds": meta["compile_seconds"], "payoff_evals": meta["payoff_evals"],
            "checkpoints": meta["checkpoints"]}, tag


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="configs/matching_pennies.yaml",
                    help="start here; use configs/two_point.yaml to match one_shot_neural")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=20_000_000,
                    help="payoff evaluations per variant (same unit as one_shot_neural)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--report", action="store_true",
                    help="score finished variants, print a ranked table, and plot curves")
    ap.add_argument("--stages", nargs="+", default=["all"],
                    choices=["all", "1", "2", "3", "4"],
                    help="1=baseline, 2=sigma, 3=utility_samples, 4=dynamics")
    ap.add_argument("--sigma", type=float, default=None,
                    help="pin sigma for stages 3–4 (else: best of stage 2 if scored, else paper)")
    ap.add_argument("--utility-samples", type=int, default=None,
                    help="pin utility_samples for stage 4 (else: best of stage 3 if scored, else paper)")
    ap.add_argument("--dynamics", default=None, choices=DYNAMICS,
                    help="pin dynamics when not sweeping stage 4")
    ap.add_argument("--score-between", action="store_true",
                    help="score between stages and carry the best sigma / samples forward")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--grid", type=int, default=801)
    ap.add_argument("--checkpoints", type=int, default=None)
    ap.add_argument("--noise-dim", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out)
    game_tag = Path(args.game).stem

    if args.report:
        report(out_root, game_tag, grid=args.grid, overwrite_scores=args.overwrite)
        return

    stages = {"1", "2", "3", "4"} if "all" in args.stages else set(args.stages)
    methods = tuple(args.methods)
    # `run_cell.Settings` now carries this repo's *tuned* pseudo-gradient defaults. This
    # experiment exists to measure the paper's, and its stored runs used the old ones, so
    # every knob that moved is pinned back here rather than inherited.
    base = dataclasses.replace(
        Settings(), noise_dim=PAPER_NOISE_DIM,
        perturbation_batch=PAPER_PERTURBATION_BATCH,
        pseudo_max_grad_norm=PAPER_MAX_GRAD_NORM, samples=PAPER_METRIC_SAMPLES)
    if args.checkpoints is not None:
        base = dataclasses.replace(base, checkpoints=args.checkpoints)
    if args.noise_dim is not None:
        base = dataclasses.replace(base, noise_dim=args.noise_dim)

    rows: list[dict] = []
    stage1_tags: list[str] = []
    stage2_tags: list[str] = []
    stage3_tags: list[str] = []

    sigma_for_later = args.sigma if args.sigma is not None else PAPER_SIGMA
    us_for_later = (args.utility_samples if args.utility_samples is not None
                    else PAPER_UTILITY_SAMPLES)
    dynamics_default = args.dynamics or PAPER_DYNAMICS

    if "1" in stages:
        print(f"\n# Stage 1: paper baseline × {methods}")
        for method in methods:
            row, tag = run_variant(
                game=args.game, method=method, seed=args.seed, budget=args.budget,
                out_root=out_root, sigma=PAPER_SIGMA,
                utility_samples=PAPER_UTILITY_SAMPLES, dynamics=PAPER_DYNAMICS,
                lr=PAPER_LR, optimizer=PAPER_OPTIMIZER, base=base,
                overwrite=args.overwrite)
            rows.append(row)
            stage1_tags.append(tag)
        if args.score_between:
            best = _pick_best(out_root, game_tag, args.seed, stage1_tags,
                              grid=args.grid, score=True)
            if best:
                print(f"\n  stage-1 best: {best[0]}  final expl={best[1]:+.5f}")

    if "2" in stages:
        print(f"\n# Stage 2: sigma sweep @ us={PAPER_UTILITY_SAMPLES}, {PAPER_DYNAMICS}")
        for method in methods:
            for sigma in SIGMAS:
                row, tag = run_variant(
                    game=args.game, method=method, seed=args.seed, budget=args.budget,
                    out_root=out_root, sigma=sigma,
                    utility_samples=PAPER_UTILITY_SAMPLES, dynamics=PAPER_DYNAMICS,
                    lr=PAPER_LR, optimizer=PAPER_OPTIMIZER, base=base,
                    overwrite=args.overwrite)
                rows.append(row)
                stage2_tags.append(tag)
        if args.score_between and args.sigma is None:
            best = _pick_best(out_root, game_tag, args.seed, stage2_tags,
                              grid=args.grid, score=True)
            if best:
                sigma_for_later = _parse_tag(best[0])["sigma"]
                print(f"\n  stage-2 best: {best[0]}  final expl={best[1]:+.5f} "
                      f"-> sigma={sigma_for_later:g}")
    elif args.sigma is None and args.score_between:
        stage2_tags = [
            variant_tag(m, sigma=s, utility_samples=PAPER_UTILITY_SAMPLES,
                        dynamics=PAPER_DYNAMICS)
            for m in methods for s in SIGMAS
        ]
        best = _pick_best(out_root, game_tag, args.seed, stage2_tags,
                          grid=args.grid, score=True)
        if best:
            sigma_for_later = _parse_tag(best[0])["sigma"]

    if "3" in stages:
        print(f"\n# Stage 3: utility_samples sweep @ sigma={sigma_for_later:g}, "
              f"{PAPER_DYNAMICS}")
        for method in methods:
            for us in UTILITY_SAMPLES:
                row, tag = run_variant(
                    game=args.game, method=method, seed=args.seed, budget=args.budget,
                    out_root=out_root, sigma=sigma_for_later, utility_samples=us,
                    dynamics=PAPER_DYNAMICS, lr=PAPER_LR, optimizer=PAPER_OPTIMIZER,
                    base=base, overwrite=args.overwrite)
                rows.append(row)
                stage3_tags.append(tag)
        if args.score_between and args.utility_samples is None:
            best = _pick_best(out_root, game_tag, args.seed, stage3_tags,
                              grid=args.grid, score=True)
            if best:
                us_for_later = _parse_tag(best[0])["utility_samples"]
                print(f"\n  stage-3 best: {best[0]}  final expl={best[1]:+.5f} "
                      f"-> utility_samples={us_for_later}")
    elif args.utility_samples is None and args.score_between:
        stage3_tags = [
            variant_tag(m, sigma=sigma_for_later, utility_samples=us,
                        dynamics=PAPER_DYNAMICS)
            for m in methods for us in UTILITY_SAMPLES
        ]
        best = _pick_best(out_root, game_tag, args.seed, stage3_tags,
                          grid=args.grid, score=True)
        if best:
            us_for_later = _parse_tag(best[0])["utility_samples"]

    if "4" in stages:
        print(f"\n# Stage 4: dynamics @ sigma={sigma_for_later:g}, us={us_for_later}")
        for method in methods:
            for dynamics in DYNAMICS:
                row, _tag = run_variant(
                    game=args.game, method=method, seed=args.seed, budget=args.budget,
                    out_root=out_root, sigma=sigma_for_later,
                    utility_samples=us_for_later, dynamics=dynamics,
                    lr=PAPER_LR, optimizer=PAPER_OPTIMIZER, base=base,
                    overwrite=args.overwrite)
                rows.append(row)

    seen: set[str] = set()
    unique_rows = []
    for row in rows:
        key = row["method"]
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "game": args.game,
        "seed": args.seed,
        "budget": args.budget,
        "stages": sorted(stages),
        "methods": list(methods),
        "sigma_for_later": sigma_for_later,
        "utility_samples_for_later": us_for_later,
        "dynamics_default": dynamics_default,
        "rows": unique_rows,
    }, indent=2))
    print(f"\n{len(unique_rows)} unique variants -> {summary_path}")
    print(f"score / rank with:\n"
          f"  python experiments/pseudo_gradient_convergence/run.py --report "
          f"--out {out_root} --game {args.game}")
    if any(r.get("status") == "failed" for r in unique_rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
