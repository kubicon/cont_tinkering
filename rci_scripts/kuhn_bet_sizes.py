"""Read a Kuhn run's strategies out of its checkpoints, and where they are exploitable.

Kuhn has exactly one continuous decision per player -- how much to bet -- at one
node each: player 0 opening, and player 1 after player 0 checked. Each of those
is one infoset per card. The other decision each player makes is binary --
fold or call a bet -- at an infoset per (card, observed bet size).

For every checkpoint in ``{run}/checkpoints`` this writes three CSVs:

``bet_sizes.csv`` -- the bet-size distribution, per (player, card):

  effective   the distribution the *game* sees, read off the same
              ``KuhnStrategy`` table the exact exploitability is computed from
              (``baselines.neural.sequential_scoring.kuhn_strategy_of``):
              ``p_bet``, and the mean / std of the size *given* a bet. Covers
              clipping to ``[min_bet, max_bet]``, discretized runs and PSRO
              populations alike. ``p_min`` / ``p_max`` are the share of bets
              landing in the first / last grid cell -- where clipping piles the
              mass of components whose means sit outside the box.
  component   the raw head, one row per Gaussian component: its joint weight
              (probability of picking that component at all), its mean, and its
              marginal std along the bet-size coordinate -- *before* clipping.
              Only for runs whose checkpoint holds a single continuous policy
              per player; PSRO populations and discretized runs have no
              meaningful single head and get effective rows only.

``br_breakdown.csv`` -- exploitability, split exactly. ``expl = br_0 + br_1``
and the game value cancels between the two, so ``expl = regret_0 + regret_1``,
where ``regret_p`` is what player ``p`` would gain by switching to a best
response against the other's current strategy. Each regret is split per card
and per decision (all terms are >= 0 and sum to ``expl``):

  open      the gain from choosing check / bet size differently at the
            betting infoset (with an optimal continuation behind a check).
            ``br_action`` / ``br_size`` say what the best response does there.
  respond   the gain from calling / folding differently against the
            opponent's bets, summed over bet sizes. ``worst_*`` names the
            size bin where most of it sits, with the call probability there
            and how often the opponent actually bets in that bin
            (``opp_bet_mass``). Gain at sizes the opponent never uses is zero:
            a mis-response only costs where it is reached.

``call_bins.csv`` -- the fold/call half of each strategy, on ``--bins`` equal
bet-size bins, per (player, card): mean call probability over the bin, the
opponent's probability of betting into it, and the response gain there. This is
where to look for untrained behaviour at off-path sizes: a best response *can*
bet sizes the policy never does, so the relevant cost of a bad call function
at such a size shows up in the *opponent's* ``open`` gain (its ``br_size``),
while ``call_bins`` shows the call function that invited it.

Folder is any ``train_sequential.py`` run directory -- a sweep run or a scratch
one. The config is found as ``--configs/{run}.yaml``, else ``meta.json``'s
``config_path``, else the config ``meta.json`` embeds.

Prints a compact block per checkpoint (with that step's ``expl`` from
``metrics.jsonl``, which the recomputed sum should match) and writes the CSVs
to ``--out-dir`` (default: the run directory; ``_target`` suffix with
``--target``).

    python rci_scripts/kuhn_bet_sizes.py data/sequential_sweep/kuhn_solvers__self_play__seed1
    python rci_scripts/kuhn_bet_sizes.py data/scratch/kuhn_sp_try1 --every 5 --target
    python rci_scripts/kuhn_bet_sizes.py RUN --steps 57 60 64 --brief
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from baselines.neural.sequential_scoring import (  # noqa: E402
    has_exact_exploitability,
    kuhn_strategy_of,
)
from games.discretized import DiscretizedSequentialGame, base_game  # noqa: E402
from games.kuhn_best_response import KuhnStrategy, bet_grid  # noqa: E402
from score_sequential_sweep import (  # noqa: E402
    CHECKPOINT_DIRNAME,
    DEFAULT_CONFIGS,
    history_by_step,
    list_steps,
    load_mixtures,
)
from training.actor_critic import masked_log_softmax  # noqa: E402
from training.gaussian import marginal_std  # noqa: E402
from training.mixture import expand_kind_mask  # noqa: E402
from training.run_config import load_run_config, run_config_from_dict  # noqa: E402
from train_sequential import prepare_game  # noqa: E402

CARD_LABELS = "JQKA23456789"
OPEN_INFOSETS = ("p0 open", "p1 after check")
RESPOND_INFOSETS = ("p0 facing bet after check", "p1 facing bet")
DEFAULT_BINS = 16
TOP_CONTRIBUTIONS = 4

BET_FIELDS = [
    "step", "params", "wall_time", "episodes", "expl",
    "player", "card", "infoset", "source", "component",
    "prob", "mean", "std", "p_min", "p_max",
]
BREAKDOWN_FIELDS = [
    "step", "params", "wall_time", "expl_logged", "expl",
    "player", "card", "decision", "infoset", "gain",
    "br_action", "br_size", "cur_p_bet", "cur_mean_size",
    "worst_bin_lo", "worst_bin_hi", "worst_gain", "worst_call", "worst_opp_bet_mass",
]
CALL_FIELDS = [
    "step", "params", "player", "card", "infoset",
    "bin_lo", "bin_hi", "call_mean", "opp_bet_mass", "gain",
]


def load_config(run_dir: Path, configs_dir: Path):
    candidate = configs_dir / f"{run_dir.name}.yaml"
    if candidate.exists():
        return load_run_config(candidate)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"no config for {run_dir}: neither {candidate} nor {meta_path} exists")
    meta = json.loads(meta_path.read_text())
    config_path = meta.get("config_path")
    if config_path:
        path = Path(config_path)
        path = path if path.is_absolute() else REPO_ROOT / path
        if path.exists():
            return load_run_config(path)
    if "config" in meta:
        return run_config_from_dict(meta["config"])
    raise SystemExit(f"{meta_path} names no readable config_path and embeds no config")


def select_steps(all_steps: list[int], steps: list[int] | None, every: int) -> list[int]:
    if steps is not None:
        missing = sorted(set(steps) - set(all_steps))
        if missing:
            raise SystemExit(f"no checkpoint for step(s) {missing}; have {all_steps}")
        return sorted(set(steps))
    chosen = all_steps[::every]
    if chosen[-1] != all_steps[-1]:
        chosen.append(all_steps[-1])
    return chosen


def as_numpy(strategy: KuhnStrategy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.asarray(strategy.open_check, dtype=np.float64),
            np.asarray(strategy.open_bet, dtype=np.float64),
            np.asarray(strategy.call, dtype=np.float64))


def effective_rows(strategy: KuhnStrategy, sizes: np.ndarray) -> list[dict]:
    """The clipped, realization-mixed bet distribution the best response is computed on."""
    _, open_bet, _ = as_numpy(strategy)  # (cards, M), joint
    rows = []
    for card in range(open_bet.shape[0]):
        p_bet = float(open_bet[card].sum())
        if p_bet > 1e-12:
            q = open_bet[card] / p_bet
            mean = float(q @ sizes)
            std = math.sqrt(max(float(q @ sizes**2) - mean**2, 0.0))
            p_min, p_max = float(q[0]), float(q[-1])
        else:
            mean = std = p_min = p_max = float("nan")
        rows.append({"card": card, "source": "effective", "component": -1, "prob": p_bet,
                     "mean": mean, "std": std, "p_min": p_min, "p_max": p_max})
    return rows


def component_rows(game, network, params, player: int) -> list[dict]:
    """Each Gaussian component's joint weight, mean and marginal std on the bet coordinate."""
    open_node, _ = game.decision_nodes(player)
    mask = expand_kind_mask(game.infoset_action_mask(open_node), network.num_components)
    rows = []
    for card in range(game.num_cards):
        logits, means, scale_trils, _ = network.apply(
            params, game.infoset_observation(jnp.asarray(card), open_node, 0.0))
        probs = np.asarray(jnp.exp(masked_log_softmax(logits, mask)))
        weights = probs[network.num_atoms:]
        mus = np.asarray(means[:, 0])
        sds = np.asarray(marginal_std(scale_trils)[:, 0])
        for k in range(network.num_components):
            rows.append({"card": card, "source": "component", "component": k,
                         "prob": float(weights[k]), "mean": float(mus[k]), "std": float(sds[k]),
                         "p_min": float("nan"), "p_max": float("nan")})
    return rows


def _deal(num_cards: int) -> tuple[np.ndarray, np.ndarray]:
    """Same "my card, their card" deal / showdown tensors as ``games.kuhn_best_response``."""
    cards = np.arange(num_cards)
    distinct = (cards[:, None] != cards[None, :]).astype(np.float64)
    pair = distinct / (num_cards * (num_cards - 1))
    showdown = np.where(cards[:, None] > cards[None, :], 1.0, -1.0)
    return pair, showdown


def regret_breakdown(sizes: np.ndarray, strategy_0: KuhnStrategy,
                     strategy_1: KuhnStrategy) -> list[dict]:
    """Each player's regret per card, split into its opening and responding decisions.

    Mirrors ``best_response_value_first`` / ``_second`` term for term, and
    additionally evaluates each player's *current* strategy on the same
    counterfactual values, so ``br - current`` splits exactly:

      player 0: br - cur = [br - (check * V_check_opt + sum bet_j * V_bet_j)]
                          + check * (V_check_opt - V_check_cur)
                -- the second term is the response gain, weighted by player 0's
                own reach into the node facing a bet (it must have checked).
      player 1: acts once per line, so opening and responding simply add.

    ``sum(current_0) == value == -sum(current_1)``, hence the per-card gains of
    both players sum to ``br_0 + br_1 = expl``.
    """
    check_0, bet_0, call_0 = as_numpy(strategy_0)
    check_1, bet_1, call_1 = as_numpy(strategy_1)
    pair, showdown = _deal(check_0.shape[0])
    weighted = pair * showdown
    scale = 1.0 + sizes

    # Player 0. Facing player 1's bet after checking: one infoset per (card, size).
    fold = -(pair @ bet_1)
    call = (weighted @ bet_1) * scale
    faced_opt = np.maximum(fold, call)
    faced_cur = (1.0 - call_0) * fold + call_0 * call
    check_opt = weighted @ check_1 + faced_opt.sum(-1)
    bet_value_0 = pair @ (1.0 - call_1) + (weighted @ call_1) * scale  # (c0, M)
    br_card_0 = np.maximum(check_opt, bet_value_0.max(-1))
    open_gain_0 = br_card_0 - (check_0 * check_opt + (bet_0 * bet_value_0).sum(-1))
    respond_0 = check_0[:, None] * (faced_opt - faced_cur)  # (c0, M)

    # Player 1. Facing player 0's opening bet: terminal after, no own reach.
    fold = -(pair @ bet_0)
    call = (weighted @ bet_0) * scale
    respond_1 = np.maximum(fold, call) - ((1.0 - call_1) * fold + call_1 * call)
    check_value_1 = weighted @ check_0
    bet_value_1 = (pair @ ((1.0 - call_0) * check_0[:, None])
                   + (weighted @ (call_0 * check_0[:, None])) * scale)
    open_gain_1 = (np.maximum(check_value_1, bet_value_1.max(-1))
                   - (check_1 * check_value_1 + (bet_1 * bet_value_1).sum(-1)))

    # Opponent's bet mass per size cell at each player's responding node,
    # averaged over the opponent's cards (for context, not part of the sum).
    opp_mass = (bet_1.mean(0), bet_0.mean(0))

    out = []
    per_player = (
        (open_gain_0, respond_0, check_opt, bet_value_0, bet_0, call_0),
        (open_gain_1, respond_1, check_value_1, bet_value_1, bet_1, call_1),
    )
    for player, (open_gain, respond, check_value, bet_value, bet, own_call) in enumerate(per_player):
        for card in range(check_0.shape[0]):
            best_bet = int(np.argmax(bet_value[card]))
            bets = bet_value[card, best_bet] > check_value[card]
            out.append({
                "player": player, "card": card,
                "open_gain": float(max(open_gain[card], 0.0)),
                "respond_gain": float(max(respond[card].sum(), 0.0)),
                "respond_by_size": np.maximum(respond[card], 0.0),
                "call": own_call[card],
                "opp_mass": opp_mass[player],
                "br_action": "bet" if bets else "check",
                "br_size": float(sizes[best_bet]) if bets else float("nan"),
                "cur_p_bet": float(bet[card].sum()),
                "cur_mean_size": (float(bet[card] @ sizes / bet[card].sum())
                                  if bet[card].sum() > 1e-12 else float("nan")),
            })
    return out


def bin_edges(num_grid: int, num_bins: int) -> np.ndarray:
    edges = np.unique(np.linspace(0, num_grid, min(num_bins, num_grid) + 1).astype(int))
    return edges


def bin_rows(entry: dict, sizes: np.ndarray, edges: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows.append({
            "bin_lo": float(sizes[lo]), "bin_hi": float(sizes[hi - 1]),
            "call_mean": float(entry["call"][lo:hi].mean()),
            "opp_bet_mass": float(entry["opp_mass"][lo:hi].sum()),
            "gain": float(entry["respond_by_size"][lo:hi].sum()),
        })
    return rows


def fmt_bets(per_player: list[list[dict]], num_cards: int) -> list[str]:
    lines = []
    for player, rows in enumerate(per_player):
        for card in range(num_cards):
            eff = next(r for r in rows if r["card"] == card and r["source"] == "effective")
            text = (f"  {OPEN_INFOSETS[player]:<14} {CARD_LABELS[card]}  "
                    f"bet {eff['prob']:.3f}  size {eff['mean']:.3f} ± {eff['std']:.3f}  "
                    f"[min {eff['p_min']:.2f} max {eff['p_max']:.2f}]")
            comps = [r for r in rows if r["card"] == card and r["source"] == "component"]
            if comps:
                text += "  | " + "  ".join(
                    f"w{r['prob']:.2f} μ{r['mean']:.3f} σ{r['std']:.3f}" for r in comps)
            lines.append(text)
    return lines


def fmt_regret(breakdown: list[dict], bins: dict[tuple[int, int], list[dict]],
               expl_logged) -> list[str]:
    totals = []
    for player in (0, 1):
        rows = [b for b in breakdown if b["player"] == player]
        opened = sum(b["open_gain"] for b in rows)
        responded = sum(b["respond_gain"] for b in rows)
        totals.append(f"p{player} {opened + responded:.4f} "
                      f"(open {opened:.4f}, respond {responded:.4f})")
    total = sum(b["open_gain"] + b["respond_gain"] for b in breakdown)
    logged = f"{expl_logged:.4f}" if isinstance(expl_logged, float) else "n/a"
    lines = [f"  regret  {'   '.join(totals)}   sum {total:.4f} (logged {logged})"]

    items = []
    for b in breakdown:
        card = CARD_LABELS[b["card"]]
        if b["open_gain"] > 0:
            action = (f"BR bets {b['br_size']:.2f}" if b["br_action"] == "bet" else "BR checks")
            items.append((b["open_gain"],
                          f"p{b['player']} {card} open     +{b['open_gain']:.4f}  {action}  "
                          f"(plays bet {b['cur_p_bet']:.2f} @ {b['cur_mean_size']:.2f})"))
        if b["respond_gain"] > 0:
            worst = max(bins[(b["player"], b["card"])], key=lambda r: r["gain"])
            items.append((b["respond_gain"],
                          f"p{b['player']} {card} respond  +{b['respond_gain']:.4f}  "
                          f"worst sizes {worst['bin_lo']:.2f}-{worst['bin_hi']:.2f}: "
                          f"+{worst['gain']:.4f}, call {worst['call_mean']:.2f}, "
                          f"opp bets there {worst['opp_bet_mass']:.3f}"))
    for _, text in sorted(items, reverse=True)[:TOP_CONTRIBUTIONS]:
        lines.append(f"    {text}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("run", type=Path, help="run directory holding checkpoints/ and meta.json")
    ap.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS,
                    help="directory searched for {run}.yaml before meta.json")
    ap.add_argument("--steps", type=int, nargs="+", default=None,
                    help="only these checkpoint steps (default: all, thinned by --every)")
    ap.add_argument("--every", type=int, default=1,
                    help="read every N-th checkpoint; the last one is always included")
    ap.add_argument("--target", action="store_true",
                    help="read the Polyak-averaged params (self_play checkpoints carry them)")
    ap.add_argument("--grid", type=int, default=None,
                    help="bet-grid points for the tables "
                         "(default: scoring.exact_grid, else the game's own)")
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS,
                    help=f"bet-size bins for call_bins.csv (default: {DEFAULT_BINS})")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the CSVs go (default: the run directory)")
    ap.add_argument("--brief", action="store_true",
                    help="print only the regret block per checkpoint, not the bet lines")
    ap.add_argument("--quiet", action="store_true", help="write the CSVs, skip the printout")
    args = ap.parse_args()
    if args.every < 1:
        raise SystemExit(f"--every must be >= 1, got {args.every}")
    if args.bins < 1:
        raise SystemExit(f"--bins must be >= 1, got {args.bins}")

    run_dir = args.run if args.run.is_absolute() else Path.cwd() / args.run
    configs_dir = args.configs if args.configs.is_absolute() else REPO_ROOT / args.configs
    checkpoint_dir = run_dir / CHECKPOINT_DIRNAME
    all_steps = list_steps(checkpoint_dir) if checkpoint_dir.is_dir() else []
    if not all_steps:
        raise SystemExit(f"no {{step}}.pkl checkpoints in {checkpoint_dir}")

    config = load_config(run_dir, configs_dir)
    game = prepare_game(config.game.build(), config)
    if not has_exact_exploitability(game):
        raise SystemExit(f"{run_dir.name} is not a Kuhn run; bet-size infosets are Kuhn-only")
    base = base_game(game)
    grid_points = args.grid if args.grid is not None else config.scoring.exact_grid
    grid = bet_grid(base) if grid_points is None else bet_grid(base, grid_points)
    sizes = np.asarray(grid, dtype=np.float64)
    edges = bin_edges(sizes.shape[0], args.bins)
    discretized = isinstance(game, DiscretizedSequentialGame)

    steps = select_steps(all_steps, args.steps, args.every)
    by_t = history_by_step(run_dir)
    expl_key = "target_expl" if args.target else "expl"
    params_label = "target" if args.target else "live"
    suffix = "_target" if args.target else ""
    out_dir = args.out_dir or run_dir
    out_dir = out_dir if out_dir.is_absolute() else Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / f"{name}{suffix}.csv"
             for name in ("bet_sizes", "br_breakdown", "call_bins")}

    print(f"{run_dir.name}: {len(steps)}/{len(all_steps)} checkpoints, {params_label} params, "
          f"bet range [{base.min_bet}, {base.max_bet}], grid {sizes.shape[0]}"
          + ("  (discretized: effective rows only)" if discretized else ""))

    with paths["bet_sizes"].open("w", newline="") as f_bets, \
            paths["br_breakdown"].open("w", newline="") as f_br, \
            paths["call_bins"].open("w", newline="") as f_calls:
        bets_writer = csv.DictWriter(f_bets, fieldnames=BET_FIELDS)
        br_writer = csv.DictWriter(f_br, fieldnames=BREAKDOWN_FIELDS)
        calls_writer = csv.DictWriter(f_calls, fieldnames=CALL_FIELDS)
        for writer in (bets_writer, br_writer, calls_writer):
            writer.writeheader()

        for step in steps:
            try:
                mixtures = load_mixtures(checkpoint_dir, step, target=args.target)
            except KeyError as err:
                raise SystemExit(f"step {step}: {err}") from err
            history_row = by_t.get(step, {})
            wall_time = history_row.get("wall_time", 0.0 if step == 0 else "")
            expl_logged = history_row.get(expl_key, "")
            strategies = [kuhn_strategy_of(game, mixtures[p], p, grid) for p in (0, 1)]

            per_player = []
            for player in (0, 1):
                mixture = mixtures[player]
                rows = effective_rows(strategies[player], sizes)
                if not discretized and len(mixture.params) == 1:
                    rows += component_rows(base, mixture.network, mixture.params[0], player)
                per_player.append(rows)
                for row in rows:
                    bets_writer.writerow({
                        "step": step, "params": params_label, "wall_time": wall_time,
                        "episodes": history_row.get("episodes", 0 if step == 0 else ""),
                        "expl": expl_logged, "player": player,
                        "card": CARD_LABELS[row["card"]], "infoset": OPEN_INFOSETS[player],
                        **{k: row[k] for k in ("source", "component", "prob", "mean",
                                               "std", "p_min", "p_max")},
                    })

            breakdown = regret_breakdown(sizes, strategies[0], strategies[1])
            expl = sum(b["open_gain"] + b["respond_gain"] for b in breakdown)
            bins = {}
            for b in breakdown:
                player, card = b["player"], b["card"]
                bins[(player, card)] = bin_rows(b, sizes, edges)
                for row in bins[(player, card)]:
                    calls_writer.writerow({
                        "step": step, "params": params_label, "player": player,
                        "card": CARD_LABELS[card], "infoset": RESPOND_INFOSETS[player], **row,
                    })
                worst = max(bins[(player, card)], key=lambda r: r["gain"])
                common = {"step": step, "params": params_label, "wall_time": wall_time,
                          "expl_logged": expl_logged, "expl": expl, "player": player,
                          "card": CARD_LABELS[card]}
                br_writer.writerow({
                    **common, "decision": "open", "infoset": OPEN_INFOSETS[player],
                    "gain": b["open_gain"], "br_action": b["br_action"],
                    "br_size": b["br_size"], "cur_p_bet": b["cur_p_bet"],
                    "cur_mean_size": b["cur_mean_size"],
                })
                br_writer.writerow({
                    **common, "decision": "respond", "infoset": RESPOND_INFOSETS[player],
                    "gain": b["respond_gain"],
                    "worst_bin_lo": worst["bin_lo"], "worst_bin_hi": worst["bin_hi"],
                    "worst_gain": worst["gain"], "worst_call": worst["call_mean"],
                    "worst_opp_bet_mass": worst["opp_bet_mass"],
                })

            for f in (f_bets, f_br, f_calls):
                f.flush()
            if not args.quiet:
                context = [f"step {step:4d}"]
                if "wall_time" in history_row:
                    context.append(f"t={history_row['wall_time'] / 3600:.2f}h")
                if isinstance(expl_logged, float):
                    context.append(f"{expl_key}={expl_logged:.4f}")
                lines = ["  ".join(context)]
                if not args.brief:
                    lines += fmt_bets(per_player, base.num_cards)
                lines += fmt_regret(breakdown, bins, expl_logged)
                print("\n".join(lines), flush=True)

    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
