"""Print the Kuhn strategy of a sequential-sweep checkpoint and where it is exploited.

For each requested step: the same strategy tables `train.py` logs, then the
exact best-response values broken down per infoset -- so a jump in `br_0`
(player 0 exploiting player 1) or `br_1` can be read off as "which card, which
action, which bet size".

    python rci_scripts/inspect_kuhn_checkpoints.py \\
        data/sequential_sweep/kuhn_solvers__self_play__seed0 --steps 30 90 130

All values are counterfactual and chance-weighted, so per-card rows sum to the
headline numbers: sum(cur) = game value (player 0's), sum(br) = br_0; for
player 1, sum(cur) = -value, sum(br) = br_1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp  # noqa: E402

from games.kuhn_best_response import _deal, bet_grid, game_value  # noqa: E402
from training.checkpoint import load_checkpoint_step_multi, target_entry  # noqa: E402
from training.config import MixturePPOHyperparams  # noqa: E402
from training.kuhn_evaluation import build_kuhn_strategy_log_fn, strategy_from_network  # noqa: E402
from training.mixture import build_mixture_network  # noqa: E402
from training.run_config import load_run_config  # noqa: E402

LABELS = "JQKA23456789"
TOP_SIZES = 3


def fmt_sizes(grid, values, k=TOP_SIZES):
    order = jnp.argsort(-values)[:k]
    return "  ".join(f"b={float(grid[j]):.2f}:{float(values[j]):+.4f}" for j in order)


def breakdown_first(grid, s0, s1, num_cards):
    """br_0: player 0 best-responding to player 1."""
    pair, showdown = _deal(num_cards)
    ws = pair * showdown
    scale = 1.0 + grid

    fold = -(pair @ s1.open_bet)                 # (c0, M) checked, P1 bet b
    call = (ws @ s1.open_bet) * scale
    faced_cur = fold * (1.0 - s0.call) + call * s0.call
    faced_br = jnp.maximum(fold, call)

    showdown_after_check = ws @ s1.open_check    # (c0,)
    check_cur = showdown_after_check + faced_cur.sum(-1)
    check_br = showdown_after_check + faced_br.sum(-1)
    bet = pair @ (1.0 - s1.call) + (ws @ s1.call) * scale   # (c0, M)

    lines = ["  br_0 breakdown (P0 exploits P1): per P0 card"]
    lines.append("    c | cur      br       gain    | BR action           | gain inside 'checked, P1 bet b'")
    for c in range(num_cards):
        cur = s0.open_check[c] * check_cur[c] + jnp.sum(s0.open_bet[c] * bet[c])
        best_bet = jnp.max(bet[c])
        br = jnp.maximum(check_br[c], best_bet)
        action = "check" if check_br[c] >= best_bet else f"bet {float(grid[jnp.argmax(bet[c])]):.2f}"
        faced_gain = float(jnp.sum(faced_br[c] - faced_cur[c]))
        lines.append(f"    {LABELS[c]} | {float(cur):+.4f}  {float(br):+.4f}  {float(br - cur):+.4f} | "
                     f"{action:<19} | {faced_gain:+.4f}")
        lines.append(f"        value(check)={float(check_br[c]):+.4f}  best bets: {fmt_sizes(grid, bet[c])}")
        faced_regret = faced_br[c] - faced_cur[c]
        if float(jnp.max(faced_regret)) > 1e-4:
            lines.append(f"        facing P1's bet, biggest per-size regret: {fmt_sizes(grid, faced_regret)}")
    return lines


def breakdown_second(grid, s0, s1, num_cards):
    """br_1: player 1 best-responding to player 0."""
    pair, showdown = _deal(num_cards)
    ws = pair * showdown
    scale = 1.0 + grid

    fold = -(pair @ s0.open_bet)                 # (c1, M) facing P0's bet b
    call = (ws @ s0.open_bet) * scale
    faced_cur = fold * (1.0 - s1.call) + call * s1.call
    faced_br = jnp.maximum(fold, call)

    reached = s0.open_check
    check = ws @ reached                         # (c1,)
    bet = pair @ ((1.0 - s0.call) * reached[:, None]) + (ws @ (s0.call * reached[:, None])) * scale

    lines = ["  br_1 breakdown (P1 exploits P0): per P1 card"]
    lines.append("    c | facing P0 bet: cur / br / gain  | after P0 check: cur / br / gain | BR after check")
    for c in range(num_cards):
        f_cur, f_br = jnp.sum(faced_cur[c]), jnp.sum(faced_br[c])
        a_cur = s1.open_check[c] * check[c] + jnp.sum(s1.open_bet[c] * bet[c])
        a_br = jnp.maximum(check[c], jnp.max(bet[c]))
        action = "check" if check[c] >= jnp.max(bet[c]) else f"bet {float(grid[jnp.argmax(bet[c])]):.2f}"
        lines.append(f"    {LABELS[c]} | {float(f_cur):+.4f} {float(f_br):+.4f} {float(f_br - f_cur):+.4f} | "
                     f"{float(a_cur):+.4f} {float(a_br):+.4f} {float(a_br - a_cur):+.4f} | {action}")
        faced_regret = faced_br[c] - faced_cur[c]
        if float(jnp.max(faced_regret)) > 1e-4:
            lines.append(f"        facing P0's bet, biggest per-size regret: {fmt_sizes(grid, faced_regret)}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a sweep run directory (holds meta.json and checkpoints/)")
    ap.add_argument("--steps", nargs="+", type=int, required=True)
    ap.add_argument("--config", default=None,
                    help="run config; defaults to configs/sequential_sweep/<run name>.yaml")
    ap.add_argument("--target", action="store_true", help="inspect the Polyak (target) iterate")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    config_path = args.config or REPO_ROOT / "configs" / "sequential_sweep" / f"{run_dir.name}.yaml"
    config = load_run_config(config_path)
    game = config.game.build()
    grid = bet_grid(game, config.game.exploitability_grid_points)
    strategy_log_fn = build_kuhn_strategy_log_fn(game)

    for step in args.steps:
        entries = load_checkpoint_step_multi(run_dir / "checkpoints", step,
                                             hyperparams_cls=MixturePPOHyperparams)
        networks, params = [], []
        for player in (0, 1):
            name = f"player_{player}"
            hyperparams, p = entries[target_entry(name) if args.target else name]
            networks.append(build_mixture_network(hyperparams))
            params.append(p)

        s0 = strategy_from_network(game, networks[0], params[0], 0, grid)
        s1 = strategy_from_network(game, networks[1], params[1], 1, grid)
        first = breakdown_first(grid, s0, s1, game.num_cards)
        second = breakdown_second(grid, s0, s1, game.num_cards)

        print(f"===== step {step}{' (target)' if args.target else ''} "
              f"| value {float(game_value(game, grid, s0, s1)):+.4f}")
        print(strategy_log_fn(SimpleNamespace(networks=networks, params=params)))
        print("\n".join(first))
        print("\n".join(second))
        print(flush=True)


if __name__ == "__main__":
    main()
