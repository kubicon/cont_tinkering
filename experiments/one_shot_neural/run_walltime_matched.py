"""Re-run SPG / JPSPG at a budget that matches the mixture's *wall-time*, not its evals.

`run_all.py` gives every method the same number of payoff evaluations, which is the
right common currency for "how much game did this method get to see". But the two
zeroth-order methods are an order of magnitude cheaper *per evaluation* than everything
else -- on `data/one_shot_neural` at 2e7 evals:

    mixture    ~855 s        spg    ~39 s       jpspg   ~51 s

so the eval-matched comparison also hands the mixture ~20x the compute. A reader is
entitled to ask what SPG does with 855 seconds, and this script answers that: it reads
the reference tree, computes each pseudo-gradient method's measured throughput
(evals/second) and the mixture's measured wall-time per game, and picks the budget that
lands the new run at the same wall-time (times `--headroom`, so the answer is "at least
as much compute, and it still does not catch up" rather than a near-miss).

The budget therefore differs per `(game, method)`, which is exactly what `run_all.py`'s
single `--budget` cannot express -- hence a separate script and, deliberately, a
**separate output tree**: `data/one_shot_neural_walltime` by default. Nothing here
touches `data/one_shot_neural`; it is only ever read.

    # what budgets would be used, and why (runs nothing)
    python experiments/one_shot_neural/run_walltime_matched.py --dry-run

    # run them (same parallelism as the reference, or the wall-times are not comparable)
    python experiments/one_shot_neural/run_walltime_matched.py --max-parallel 4

    # score the new tree and plot it against the reference mixture
    python experiments/one_shot_neural/run_walltime_matched.py --plot

**The wall-times only mean something if the machine does.** The reference runs were
measured with `run_all.py`'s defaults (4 cells at a time, cores divided between them),
so this script defaults to the same and caps child threads the same way. Running with a
different `--max-parallel`, or on a different machine, compares the machines rather than
the methods -- `--target-seconds` overrides the reference wall-time for that case.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repo root
sys.path.insert(0, str(HERE))              # this directory, for run_cell / run_all / score

from run_all import DEFAULT_GAMES, cell_name, child_environment  # noqa: E402

RUN_CELL = HERE / "run_cell.py"
DEFAULT_REFERENCE = "data/one_shot_neural"
DEFAULT_OUT = "data/one_shot_neural_walltime"
PSEUDO_METHODS = ("spg", "jpspg")
# Label the re-runs so a merged plot cannot silently show two different budgets under
# one name.
LABEL = "{method} (wall-matched)"


# --------------------------------------------------------------------------- calibration


def reference_runs(reference: Path) -> dict[tuple[str, str], list[dict]]:
    """`(game_tag, method) -> [meta, ...]` for every finished run in the reference tree."""
    runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(reference.glob("*/*/seed*/meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("status") == "failed" or not meta.get("train_seconds"):
            continue
        runs[(meta.get("game"), meta.get("method"))].append(meta)
    return runs


def calibrate(reference: Path, games: list[str], methods: list[str], match: str,
              headroom: float, target_seconds: float | None) -> list[dict]:
    """One row per `(game, method)`: the budget that should land at the target wall-time.

    Throughput is measured from the reference run's own `payoff_evals / train_seconds`
    rather than assumed, so a method whose cost per evaluation depends on the game (it
    does -- the games have different payoff kernels) is calibrated per game.
    """
    runs = reference_runs(reference)
    rows = []
    for game_path in games:
        game = Path(game_path).stem
        if target_seconds is not None:
            target, target_note = target_seconds, "--target-seconds"
        else:
            reference_metas = runs.get((game, match), [])
            if not reference_metas:
                print(f"  skip {game}: no finished {match!r} runs in {reference}")
                continue
            target = statistics.mean(m["train_seconds"] for m in reference_metas)
            target_note = f"{match} x{len(reference_metas)}"

        for method in methods:
            metas = runs.get((game, method), [])
            if not metas:
                print(f"  skip {game}/{method}: no reference run to measure throughput from")
                continue
            throughput = statistics.mean(
                m["payoff_evals"] / m["train_seconds"] for m in metas)
            rows.append({
                "game": game,
                "config": metas[0]["config"],
                "method": method,
                "target_seconds": target,
                "target_from": target_note,
                "reference_seconds": statistics.mean(m["train_seconds"] for m in metas),
                "reference_evals": statistics.mean(m["payoff_evals"] for m in metas),
                "evals_per_second": throughput,
                "headroom": headroom,
                "budget": int(target * headroom * throughput),
            })
    return rows


def calibration_table(rows: list[dict]) -> str:
    lines = ["|game|method|reference s|target s|evals/s|new budget|x reference|",
             "|-|-|-|-|-|-|-|"]
    for row in rows:
        lines.append(
            f"|{row['game']}|{row['method']}|{row['reference_seconds']:.0f}|"
            f"{row['target_seconds'] * row['headroom']:.0f}|{row['evals_per_second']:.3g}|"
            f"{row['budget']:.3g}|{row['budget'] / row['reference_evals']:.1f}x|")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- running


def launch(rows: list[dict], seeds: list[int], out_root: Path, max_parallel: int,
           threads: int, overwrite: bool) -> list[tuple[str, int]]:
    """`run_cell.py` per cell, at most `max_parallel` at a time -- `run_all.py`'s loop,
    with a per-cell budget instead of one shared one."""
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = child_environment(threads)

    queue = []
    for row in rows:
        for seed in seeds:
            directory = out_root / row["game"] / row["method"] / f"seed{seed}"
            if (directory / "meta.json").exists() and not overwrite:
                print(f"  skip (already done): {directory}")
                continue
            queue.append((row, seed))

    started_at = time.monotonic()
    running: list[tuple] = []
    finished: list[tuple[str, int]] = []
    while queue or running:
        while queue and len(running) < max_parallel:
            row, seed = queue.pop(0)
            name = cell_name(row["config"], row["method"], seed)
            command = [sys.executable, str(RUN_CELL), "--game", row["config"],
                       "--method", row["method"], "--seed", str(seed),
                       "--budget", str(row["budget"]), "--out", str(out_root)]
            if overwrite:
                command.append("--overwrite")
            handle = open(logs / f"{name}.log", "w")
            print(f"  [{time.monotonic() - started_at:7.1f}s] start {name} "
                  f"(budget {row['budget']:.3g}, target {row['target_seconds']:.0f}s)")
            running.append((name, subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, env=env), handle))
        time.sleep(1.0)
        for entry in list(running):
            name, process, handle = entry
            if process.poll() is None:
                continue
            handle.close()
            running.remove(entry)
            finished.append((name, process.returncode))
            status = "ok" if process.returncode == 0 else f"FAILED ({process.returncode})"
            print(f"  [{time.monotonic() - started_at:7.1f}s] {status:16s} {name}")
    return finished


def check_achieved(rows: list[dict], seeds: list[int], out_root: Path) -> None:
    """Did the calibration land? Throughput is measured, not guaranteed."""
    print("\n|game|method|target s|achieved s|ratio|")
    print("|-|-|-|-|-|")
    for row in rows:
        achieved = []
        for seed in seeds:
            path = out_root / row["game"] / row["method"] / f"seed{seed}" / "meta.json"
            if path.exists():
                meta = json.loads(path.read_text())
                if meta.get("train_seconds"):
                    achieved.append(meta["train_seconds"])
        if not achieved:
            continue
        mean = statistics.mean(achieved)
        target = row["target_seconds"] * row["headroom"]
        print(f"|{row['game']}|{row['method']}|{target:.0f}|{mean:.0f}|"
              f"{mean / target:.2f}x|")


# --------------------------------------------------------------------------- plotting


def plot(out_root: Path, reference: Path, games: list[str], methods: list[str],
         match: str, grid: int, overwrite_scores: bool,
         include_reference_pseudo: bool) -> None:
    """Score the new tree, merge in the reference `match` curves, one plot per game.

    The merged plot's *budget* panel is deliberately not a fair comparison any more --
    that is the whole point of this experiment, and the reason the re-runs are relabelled
    `<method> (wall-matched)`. Read the wall-time panel.
    """
    from baselines.common import GridOracle, load_game
    from score import find_runs, plot_game, score_run, summary_row, summary_table

    oracles: dict[str, GridOracle] = {}
    curves: dict = {}
    rows: list[dict] = []
    for directory in find_runs(out_root):
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("status") == "failed":
            print(f"  skip (failed run): {directory}")
            continue
        if games and meta.get("game") not in games:
            continue
        config = meta["config"]
        if config not in oracles:
            game_obj, _ = load_game(config)
            oracles[config] = GridOracle(game_obj, points=grid)
        scores = score_run(directory, oracles[config], overwrite_scores)
        rows.append(summary_row(meta, scores))
        label = LABEL.format(method=meta["method"])
        curves[f"walltime/{directory.relative_to(out_root)}"] = {
            "meta": {**{k: meta.get(k) for k in ("game", "seed", "budget", "access_model",
                                                 "train_seconds")},
                     "method": label},
            "scores": scores,
        }
        print(f"  {meta['game']:>22s} / {label:<22s} seed{meta['seed']}  "
              f"expl {scores[-1]['expl']:+.5f}  {len(scores)} checkpoints")

    reference_curves_path = reference / "curves.json"
    if not reference_curves_path.exists():
        raise SystemExit(
            f"no {reference_curves_path}: score the reference tree first with\n"
            f"  python {HERE / 'score.py'} --out {reference}")
    keep = {match} | (set(methods) if include_reference_pseudo else set())
    reference_curves = json.loads(reference_curves_path.read_text())
    merged = dict(curves)
    for key, entry in reference_curves.items():
        if entry["meta"].get("method") in keep:
            merged[f"reference/{key}"] = entry

    (out_root / "curves.json").write_text(json.dumps(curves, indent=2))
    (out_root / "curves_merged.json").write_text(json.dumps(merged, indent=2))
    table = summary_table(rows)
    (out_root / "summary.md").write_text(table + "\n")
    print(f"\n{table}")

    print(f"curves -> {out_root / 'curves.json'} (+ curves_merged.json with {match})")
    plotted = sorted({entry["meta"]["game"] for entry in curves.values()})
    for game in plotted:
        try:
            path = plot_game(merged, game, out_root / f"{game}_walltime_curves.png")
        except ImportError as exc:
            # Everything above is already on disk; only the drawing needs matplotlib,
            # which the runtime environment does not have to carry.
            print(f"cannot plot ({exc}). Install matplotlib and re-run --plot; the "
                  "scores are cached in each run's scores.json, so nothing is rescored.")
            break
        print(f"plot -> {path}")


# --------------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES))
    ap.add_argument("--methods", nargs="+", default=list(PSEUDO_METHODS),
                    help="the cheap methods to re-run at a matched wall-time")
    ap.add_argument("--match", default="mixture",
                    help="method whose wall-time is the target (the main technique)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    help="the run_all.py tree to calibrate against; read only, never written")
    ap.add_argument("--out", default=DEFAULT_OUT, help="new tree; the reference is untouched")
    ap.add_argument("--headroom", type=float, default=1.1,
                    help="multiply the target wall-time by this ('slightly higher')")
    ap.add_argument("--target-seconds", type=float, default=None,
                    help="override the reference wall-time (use on a different machine)")
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="match the reference run's value or the wall-times are not comparable")
    ap.add_argument("--threads-per-cell", type=int, default=0,
                    help="0 divides the machine's cores evenly among the parallel cells")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the calibration and the cells, run nothing")
    ap.add_argument("--plot", action="store_true",
                    help="score the new tree and plot it against the reference "
                         f"{'--match'!r} method; runs no training")
    ap.add_argument("--include-reference-pseudo", action="store_true",
                    help="also draw the reference (eval-matched) spg/jpspg curves")
    ap.add_argument("--grid", type=int, default=801, help="deviation grid for scoring")
    ap.add_argument("--rescore", action="store_true", help="rescore runs that have scores.json")
    args = ap.parse_args()

    out_root, reference = Path(args.out), Path(args.reference)
    game_tags = [Path(g).stem for g in args.games]

    if args.plot:
        plot(out_root, reference, game_tags, args.methods, args.match, args.grid,
             args.rescore, args.include_reference_pseudo)
        return

    if not reference.exists():
        raise SystemExit(f"no reference tree at {reference}")

    print(f"reference: {reference}   target: {args.match} wall-time x {args.headroom}")
    rows = calibrate(reference, args.games, args.methods, args.match,
                     args.headroom, args.target_seconds)
    if not rows:
        raise SystemExit("nothing to calibrate -- check --games / --methods / --reference")
    print("\n" + calibration_table(rows))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "calibration.json").write_text(json.dumps({
        "reference": str(reference), "match": args.match, "headroom": args.headroom,
        "target_seconds_override": args.target_seconds,
        "max_parallel": args.max_parallel, "seeds": args.seeds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows,
    }, indent=2))
    (out_root / "calibration.md").write_text(calibration_table(rows))

    cells = len(rows) * len(args.seeds)
    total = sum(r["target_seconds"] * r["headroom"] for r in rows) * len(args.seeds)
    print(f"{cells} cells, {args.max_parallel} at a time; expect roughly "
          f"{total / max(args.max_parallel, 1) / 3600:.1f} h of wall-time")
    if args.dry_run:
        for row in rows:
            for seed in args.seeds:
                print(f"  {cell_name(row['config'], row['method'], seed)} "
                      f"budget={row['budget']:.4g}")
        return

    import os
    threads = (args.threads_per_cell
               or max(1, (os.cpu_count() or 4) // max(args.max_parallel, 1)))
    finished = launch(rows, args.seeds, out_root, args.max_parallel, threads, args.overwrite)
    check_achieved(rows, args.seeds, out_root)

    failed = [name for name, code in finished if code != 0]
    print(f"\n{len(finished)} cells{f', {len(failed)} failed: {failed}' if failed else ''}")
    print(f"logs -> {out_root / 'logs'}")
    print(f"next: python {Path(__file__).name} --plot --out {out_root}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
