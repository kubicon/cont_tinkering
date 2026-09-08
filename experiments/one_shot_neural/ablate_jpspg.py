"""Single-seed JPSPG ablations suggested after the two_point comparison.

Three stages (run with ``--stages all``, or pick a subset):

  1. **dynamics** — simultaneous / optimistic / extragradient
     (AdaBelief, lr=1e-4)
  2. **lr** — 1e-4 / 3e-4 / 1e-3
     (AdaBelief; dynamics from ``--dynamics`` or the best stage-1 run if scored)
  3. **optimizer** — adabelief / adam
     (lr from ``--lr`` or the best stage-2 run if scored; same dynamics as stage 2)

Each variant is one run-cell-style directory under ``--out``:

    <out>/<game>/<variant_tag>/seed<k>/

so ``score.py`` can score the whole tree afterwards. Variants already present are
skipped unless ``--overwrite``.

Examples::

    # full suite at the main-experiment budget
    python experiments/one_shot_neural/ablate_jpspg.py \\
        --game configs/two_point.yaml --seed 0 --budget 20000000

    # cheap smoke test of stage 1 only
    python experiments/one_shot_neural/ablate_jpspg.py \\
        --game configs/two_point.yaml --seed 0 --budget 500000 --stages 1

    # auto-pick best dynamics / lr between stages (scores each finished variant)
    python experiments/one_shot_neural/ablate_jpspg.py \\
        --game configs/two_point.yaml --seed 0 --budget 20000000 --score-between

    # pin stage 2–3 knobs by hand
    python experiments/one_shot_neural/ablate_jpspg.py \\
        --stages 2 3 --dynamics optimistic --lr 0.001

    # afterwards — score + rank + plot (this is how you see what won)
    python experiments/one_shot_neural/ablate_jpspg.py --report \\
        --out data/one_shot_neural_jpspg_ablate --game configs/two_point.yaml
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
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

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

DEFAULT_OUT = "data/one_shot_neural_jpspg_ablate"
DYNAMICS = ("simultaneous", "optimistic", "extragradient")
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
OPTIMIZERS = ("adabelief", "adam")


def variant_tag(dynamics: str, lr: float, optimizer: str) -> str:
    """Filesystem-safe method folder name encoding the ablation knobs."""
    lr_s = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    return f"jpspg__dyn-{dynamics}__lr-{lr_s}__opt-{optimizer}"


def short_label(dynamics: str, lr: float, optimizer: str) -> str:
    return f"{dynamics} | lr={lr:g} | {optimizer}"


def _parse_tag(tag: str) -> tuple[str, float, str]:
    parts = dict(piece.split("-", 1) for piece in tag.split("__")[1:])
    return parts["dyn"], float(parts["lr"]), parts["opt"]


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


def _pick_best(out_root: Path, game: str, seed: int, tags: list[str],
               grid: int, score: bool) -> tuple[str, float] | None:
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


def find_ablation_runs(out_root: Path, game: str | None = None) -> list[Path]:
    runs = sorted(p.parent for p in out_root.glob("*/*/*/meta.json"))
    if game is None:
        return runs
    return [p for p in runs if p.parts[-3] == game]


def report(out_root: Path, game: str, grid: int, overwrite_scores: bool = False) -> Path:
    """Score every ablation variant, rank by exploitability, and plot curves."""
    from baselines.common import load_game
    from score import plot_game, score_run

    runs = find_ablation_runs(out_root, game)
    if not runs:
        raise SystemExit(f"no ablation runs under {out_root / game}")

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
        if not ablation and meta.get("method", "").startswith("jpspg__"):
            dyn, lr, opt = _parse_tag(meta["method"])
            ablation = {"dynamics": dyn, "pseudo_lr": lr, "pseudo_optimizer": opt}
        expls = [s["expl"] for s in scores]
        final = scores[-1] if scores else {}
        row = {
            "tag": meta.get("method"),
            "dynamics": ablation.get("dynamics"),
            "lr": ablation.get("pseudo_lr"),
            "optimizer": ablation.get("pseudo_optimizer"),
            "final_expl": expls[-1] if expls else float("nan"),
            "best_expl": min(expls) if expls else float("nan"),
            "payoff_evals": final.get("payoff_evals"),
            "train_seconds": meta.get("train_seconds"),
            "label": short_label(
                ablation.get("dynamics", "?"),
                float(ablation.get("pseudo_lr", float("nan"))),
                ablation.get("pseudo_optimizer", "?")),
        }
        rows.append(row)
        curves[str(directory.relative_to(out_root))] = {
            "meta": {**{k: meta.get(k) for k in ("game", "method", "seed", "budget",
                                                  "access_model", "train_seconds")},
                     "method": row["label"]},
            "scores": scores,
        }
        print(f"  {row['label']:<40s}  final {row['final_expl']:+.5f}  "
              f"best {row['best_expl']:+.5f}")

    rows.sort(key=lambda r: (r["final_expl"], r["best_expl"]))
    lines = [
        "|rank|dynamics|lr|optimizer|final expl|best expl|train s|",
        "|-|-|-|-|-|-|-|",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"|{i}|{row['dynamics']}|{row['lr']:g}|{row['optimizer']}|"
            f"{row['final_expl']:+.5f}|{row['best_expl']:+.5f}|"
            f"{row['train_seconds']:.0f}|")
    table = "\n".join(lines) + "\n"
    table_path = out_root / f"{game}_ablation_ranking.md"
    table_path.write_text(table)
    (out_root / "curves.json").write_text(json.dumps(curves, indent=2))

    plot_path = plot_game(curves, game, out_root / f"{game}_ablation_curves.png")
    print(f"\n{table}")
    if rows:
        best = rows[0]
        print(f"winner (lowest final expl): {best['label']}  "
              f"final={best['final_expl']:+.5f}  best={best['best_expl']:+.5f}")
    print(f"ranking -> {table_path}")
    print(f"plot    -> {plot_path}")
    return table_path


def run_jpspg_variant(
    *,
    game: str,
    seed: int,
    budget: int,
    out_root: Path,
    dynamics: str,
    lr: float,
    optimizer: str,
    base: Settings,
    overwrite: bool,
) -> tuple[dict, str]:
    """Like ``run_cell(..., method='jpspg')`` but directory / meta method = variant tag."""
    tag = variant_tag(dynamics, lr, optimizer)
    game_tag = Path(game).stem
    directory = out_root / game_tag / tag / f"seed{seed}"
    settings = dataclasses.replace(
        base, dynamics=dynamics, pseudo_lr=lr, pseudo_optimizer=optimizer,
    )
    row = {"game": game_tag, "method": tag, "seed": seed, "budget": budget,
           "config": game, "dir": str(directory),
           "access_model": ACCESS_MODEL["jpspg"],
           "ablation": {"dynamics": dynamics, "pseudo_lr": lr,
                        "pseudo_optimizer": optimizer}}

    print(f"\n=== {tag} ===")
    if (directory / "meta.json").exists() and not overwrite:
        stored = json.loads((directory / "meta.json").read_text())
        print(f"  skip (already done): {directory}")
        return {**row, **{k: stored.get(k) for k in
                          ("status", "train_seconds", "compile_seconds", "payoff_evals",
                           "checkpoints")}}, tag

    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        game_obj, game_cfg, config = load_run(game)
        oracle = GridOracle(game_obj, points=settings.grid)
        batch = build_hyperparams(game_obj, 0, config).num_envs
        plan = plan_units("jpspg", budget, settings, batch)
        for warning in (budget_warning("jpspg", plan, settings, budget),
                        compile_warning("jpspg", plan, settings)):
            if warning:
                print(f"  {warning}")
        for key in ("iterations", "rounds"):
            if key in plan:
                plan[key], plan["log_every"] = _aligned(plan[key], settings.checkpoints)
        meta = {**row, "plan": plan, "settings": dataclasses.asdict(settings),
                "game_config": dataclasses.asdict(game_cfg), "scored_during_run": False,
                "commit": _git_commit(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        writer = RunWriter(directory, meta)
        result = RUNNERS["jpspg"](
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="configs/two_point.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=20_000_000,
                    help="payoff evaluations per variant (same unit as run_all.py)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true",
                    help="score finished variants, print a ranked table, and plot curves "
                         "(no training)")
    ap.add_argument("--stages", nargs="+", default=["all"],
                    choices=["all", "1", "2", "3"],
                    help="which ablation stages to run")
    ap.add_argument("--dynamics", default=None, choices=DYNAMICS,
                    help="pin dynamics for stages 2–3 (else: best of stage 1 if scored, "
                         "else simultaneous)")
    ap.add_argument("--lr", type=float, default=None,
                    help="pin learning rate for stage 3 (else: best of stage 2 if scored, "
                         "else 1e-3)")
    ap.add_argument("--score-between", action="store_true",
                    help="score finished variants between stages and pick the best "
                         "dynamics / lr for the next stage")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--grid", type=int, default=801,
                    help="scoring grid for --report / --score-between")
    ap.add_argument("--utility-samples", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--noise-dim", type=int, default=None)
    ap.add_argument("--checkpoints", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out)
    game_tag = Path(args.game).stem

    if args.report:
        report(out_root, game_tag, grid=args.grid, overwrite_scores=args.overwrite)
        return

    stages = {"1", "2", "3"} if "all" in args.stages else set(args.stages)
    # Pinned to the paper's pseudo-gradient knobs, not `Settings()`'s tuned ones: this
    # ablation's stored runs used them, and mixing the two would make its table meaningless.
    base = dataclasses.replace(
        Settings(), sigma=0.1, pseudo_lr=1e-4, noise_dim=8,
        pseudo_max_grad_norm=0.0, samples=4096)
    if args.utility_samples is not None:
        base = dataclasses.replace(base, utility_samples=args.utility_samples)
    if args.sigma is not None:
        base = dataclasses.replace(base, sigma=args.sigma)
    if args.noise_dim is not None:
        base = dataclasses.replace(base, noise_dim=args.noise_dim)
    if args.checkpoints is not None:
        base = dataclasses.replace(base, checkpoints=args.checkpoints)

    rows: list[dict] = []
    stage1_tags: list[str] = []
    stage2_tags: list[str] = []

    dynamics_for_later = args.dynamics or "simultaneous"
    if "1" in stages:
        print(f"\n# Stage 1: dynamics @ lr={1e-4:g}, AdaBelief")
        for dynamics in DYNAMICS:
            row, tag = run_jpspg_variant(
                game=args.game, seed=args.seed, budget=args.budget, out_root=out_root,
                dynamics=dynamics, lr=1e-4, optimizer="adabelief",
                base=base, overwrite=args.overwrite)
            rows.append(row)
            stage1_tags.append(tag)
        if args.score_between and args.dynamics is None:
            best = _pick_best(out_root, game_tag, args.seed, stage1_tags,
                              grid=args.grid, score=True)
            if best:
                dynamics_for_later = _parse_tag(best[0])[0]
                print(f"\n  stage-1 best: {best[0]}  final expl={best[1]:+.5f} "
                      f"-> dynamics={dynamics_for_later}")
    elif args.dynamics is None and args.score_between:
        stage1_tags = [variant_tag(d, 1e-4, "adabelief") for d in DYNAMICS]
        best = _pick_best(out_root, game_tag, args.seed, stage1_tags,
                          grid=args.grid, score=True)
        if best:
            dynamics_for_later = _parse_tag(best[0])[0]

    lr_for_later = args.lr if args.lr is not None else 1e-3
    if "2" in stages:
        print(f"\n# Stage 2: learning rate @ dynamics={dynamics_for_later}, AdaBelief")
        for lr in LEARNING_RATES:
            row, tag = run_jpspg_variant(
                game=args.game, seed=args.seed, budget=args.budget, out_root=out_root,
                dynamics=dynamics_for_later, lr=lr, optimizer="adabelief",
                base=base, overwrite=args.overwrite)
            rows.append(row)
            stage2_tags.append(tag)
        if args.score_between and args.lr is None:
            best = _pick_best(out_root, game_tag, args.seed, stage2_tags,
                              grid=args.grid, score=True)
            if best:
                lr_for_later = _parse_tag(best[0])[1]
                print(f"\n  stage-2 best: {best[0]}  final expl={best[1]:+.5f} "
                      f"-> lr={lr_for_later:g}")
    elif args.lr is None and args.score_between:
        stage2_tags = [variant_tag(dynamics_for_later, lr, "adabelief")
                       for lr in LEARNING_RATES]
        best = _pick_best(out_root, game_tag, args.seed, stage2_tags,
                          grid=args.grid, score=True)
        if best:
            lr_for_later = _parse_tag(best[0])[1]

    if "3" in stages:
        print(f"\n# Stage 3: optimizer @ dynamics={dynamics_for_later}, lr={lr_for_later:g}")
        for optimizer in OPTIMIZERS:
            row, _tag = run_jpspg_variant(
                game=args.game, seed=args.seed, budget=args.budget, out_root=out_root,
                dynamics=dynamics_for_later, lr=lr_for_later, optimizer=optimizer,
                base=base, overwrite=args.overwrite)
            rows.append(row)

    # Deduplicate rows that appear in more than one stage (e.g. simultaneous@1e-4
    # is both stage 1 and stage 2 when dynamics stays simultaneous).
    seen: set[str] = set()
    unique_rows = []
    for row in rows:
        key = row["method"]
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    summary_path = out_root / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "game": args.game, "seed": args.seed, "budget": args.budget,
        "stages": sorted(stages),
        "dynamics_for_later": dynamics_for_later,
        "lr_for_later": lr_for_later,
        "rows": unique_rows,
    }, indent=2))
    print(f"\n{len(unique_rows)} unique variants -> {summary_path}")
    print(f"score with:\n  python experiments/one_shot_neural/score.py --out {out_root}")
    if any(r.get("status") == "failed" for r in unique_rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
