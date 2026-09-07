"""Double oracle: grow a finite support one best response at a time, solve the
restricted game exactly by LP.

The canonical algorithm for a continuous game whose equilibrium is finitely
supported, and therefore the baseline that a Gaussian-mixture head has to be
compared against honestly: on a 1-D game with `K` peaks it typically needs about
`K` iterations, ends with an *exact* rational equilibrium of the restricted game,
and its exploitability is limited only by the best-response oracle. What it needs
that we do not is a global best-response oracle at every iteration (here: a grid
argmax, optionally polished by projected gradient ascent) and an LP solve on the
restricted game -- neither of which survives contact with a large action space or
a game whose payoff is only known through samples.

One iteration:

  1. solve the restricted matrix game on the current supports  -> (x, y), value v
  2. best-respond to each side's mix over the *whole* action box -> br0, br1
  3. the gap `v0 + v1` (deviation values, see `GridOracle.exploitability`) is the
     exploitability of the current iterate; stop when it is below `--tol`
  4. add br0/br1 to the supports (unless a support point is already within
     `--dedup` of them, which is what stops the support from filling up with
     copies of the same peak) and repeat.

Because the restricted game is solved exactly, this converges monotonically in the
support size, not in "iterations" in the gradient-method sense -- so the history
here has one row per *oracle call*, and comparing it to a gradient baseline is a
comparison of oracle calls against gradient steps, which is the honest framing:
each of its iterations costs a global optimization over the action space.

Usage:
    python -m baselines.double_oracle configs/two_point.yaml --iters 20 --polish
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from scipy.optimize import linprog

from games.base import ZeroSumGame

from .common import CheckpointWriter, GridOracle, StrategyPair, base_parser, load_game, \
    print_history, save_history, top_atoms


@dataclasses.dataclass
class DoubleOracleResult:
    support_0: np.ndarray   # (N0, d0) actions in player 0's final restricted strategy set
    support_1: np.ndarray   # (N1, d1)
    weights_0: np.ndarray   # (N0,) the restricted game's equilibrium mix
    weights_1: np.ndarray   # (N1,)
    value: float            # value of the final restricted game
    history: list[dict]
    converged: bool
    # "tolerance" (gap below --tol), "duplicate" (both oracles returned points the
    # support already holds -- the grid cannot resolve the remaining gap), or
    # "budget" (ran out of rounds).
    stop_reason: str


def solve_matrix_game(payoff: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Nash equilibrium `(x, y, value)` of a zero-sum matrix game, by LP.

    `payoff[i, j]` is the row player's (player 0's) utility; it maximizes, the
    column player minimizes. Two LPs are solved rather than one plus its dual
    multipliers: HiGHS reports duals with a sign convention that has changed
    between releases, and at these sizes (a handful of rows) a second solve is free.

        row:    max_x  v   s.t.  sum_i x_i payoff[i, j] >= v for all j,  x in simplex
        column: min_y  w   s.t.  sum_j payoff[i, j] y_j <= w for all i,  y in simplex

    The two optimal values agree (minimax), which is asserted -- a mismatch means
    the LP did not actually solve, and a silently wrong restricted equilibrium would
    poison every later best response.
    """
    m, n = payoff.shape

    c = np.zeros(m + 1)
    c[-1] = -1.0                                   # maximize v
    a_ub = np.hstack([-payoff.T, np.ones((n, 1))])  # -x^T P + v <= 0, per column j
    a_eq = np.zeros((1, m + 1))
    a_eq[0, :m] = 1.0
    row = linprog(c, a_ub, np.zeros(n), a_eq, [1.0],
                  bounds=[(0.0, None)] * m + [(None, None)], method="highs")

    c = np.zeros(n + 1)
    c[-1] = 1.0                                    # minimize w
    a_ub = np.hstack([payoff, -np.ones((m, 1))])   # P y - w <= 0, per row i
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    col = linprog(c, a_ub, np.zeros(m), a_eq, [1.0],
                  bounds=[(0.0, None)] * n + [(None, None)], method="highs")

    if not (row.success and col.success):
        raise RuntimeError(f"restricted-game LP failed: {row.message!r} / {col.message!r}")
    value_row, value_col = float(row.x[-1]), float(col.x[-1])
    if not np.isclose(value_row, value_col, atol=1e-6, rtol=1e-6):
        raise RuntimeError(
            f"minimax mismatch in the restricted game: {value_row} vs {value_col}")

    x = np.clip(row.x[:m], 0.0, None)
    y = np.clip(col.x[:n], 0.0, None)
    return x / x.sum(), y / y.sum(), 0.5 * (value_row + value_col)


def _append_if_new(support: np.ndarray, action: np.ndarray, dedup: float) -> tuple[np.ndarray, bool]:
    """`(support, added)`: `action` appended unless the support already carries a
    point within `dedup` of it.

    Without this the support grows by a near-duplicate every iteration once the
    oracle starts returning the same peak repeatedly (the grid argmax is
    deterministic, but a *polished* response lands a hair off the previous one), and
    the LP quietly becomes degenerate.
    """
    action = np.asarray(action, dtype=np.float64).reshape(1, -1)
    if support.shape[0] and np.min(np.linalg.norm(support - action, axis=-1)) <= dedup:
        return support, False
    return np.concatenate([support, action], axis=0), True


def run_double_oracle(
    game: ZeroSumGame,
    oracle: GridOracle,
    iters: int = 30,
    tol: float = 1e-3,
    dedup: float = 1e-2,
    polish: bool = False,
    polish_steps: int = 200,
    polish_lr: float = 1e-2,
    init: str = "center",
    seed: int = 0,
    checkpoint_fn: Callable[[StrategyPair], None] | None = None,
) -> DoubleOracleResult:
    """Double oracle for at most `iters` oracle rounds.

    `init` seeds each support with a single action: `"center"` (the midpoint of the
    box -- deterministic, and deliberately *not* on a peak) or `"random"`.

    `checkpoint_fn` is called once per round with the restricted equilibrium of that
    round, and once more on the final re-solve. `t` is the round number, so a
    checkpoint here costs a global best response per player, not a gradient step.
    """
    rng = np.random.default_rng(seed)
    if init == "center":
        s0 = (0.5 * (oracle.lo0 + oracle.hi0)).reshape(1, -1)
        s1 = (0.5 * (oracle.lo1 + oracle.hi1)).reshape(1, -1)
    elif init == "random":
        s0 = rng.uniform(oracle.lo0, oracle.hi0).reshape(1, -1)
        s1 = rng.uniform(oracle.lo1, oracle.hi1).reshape(1, -1)
    else:
        raise ValueError(f"unknown init {init!r} (choices: center, random)")

    history: list[dict] = []
    converged = False
    stop_reason = "budget"
    last_emitted: tuple[int, int] | None = None
    x = y = None
    value = 0.0

    for t in range(1, iters + 1):
        payoff = oracle.payoff(s0, s1)
        x, y, value = solve_matrix_game(payoff)

        br0, v0 = oracle.best_response(0, s1, y, polish, polish_steps, polish_lr)
        br1, v1 = oracle.best_response(1, s0, x, polish, polish_steps, polish_lr)
        gap = v0 + v1     # `(br0 - U) + (U - br1)`, as in `GridOracle.exploitability`

        history.append({
            "t": t, "expl": float(gap), "value": float(value),
            "support_0": int(s0.shape[0]), "support_1": int(s1.shape[0]),
        })

        if checkpoint_fn is not None:
            checkpoint_fn(StrategyPair(t=t, support_0=s0.copy(), weights_0=x.copy(),
                                       support_1=s1.copy(), weights_1=y.copy()))
            last_emitted = (s0.shape[0], s1.shape[0])

        if gap < tol:
            converged, stop_reason = True, "tolerance"
            break

        s0, added_0 = _append_if_new(s0, br0, dedup)
        s1, added_1 = _append_if_new(s1, br1, dedup)
        if not (added_0 or added_1):
            # Both oracles returned points the support already contains, yet the gap is
            # still above tolerance: the restricted equilibrium is a fixed point of the
            # whole procedure, so no further iteration can change anything. In exact
            # arithmetic this cannot happen below tolerance; in practice it means the
            # grid oracle is too coarse (or `--dedup` too wide) to resolve the deviation
            # it is reporting.
            stop_reason = "duplicate"
            break

    # The supports grew after the last LP, so the reported mix is re-solved on them.
    payoff = oracle.payoff(s0, s1)
    x, y, value = solve_matrix_game(payoff)
    # Only a *new* checkpoint: when the loop stopped on tolerance the supports have not
    # grown since the last round's LP, and re-emitting would just overwrite that file
    # (checkpoints are named by `t`) with an identical one.
    if checkpoint_fn is not None and (s0.shape[0], s1.shape[0]) != last_emitted:
        checkpoint_fn(StrategyPair(t=len(history) + 1, support_0=s0.copy(), weights_0=x.copy(),
                                   support_1=s1.copy(), weights_1=y.copy()))
    return DoubleOracleResult(s0, s1, x, y, float(value), history, converged, stop_reason)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--iters", type=int, default=30, help="maximum oracle rounds")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="stop once the restricted equilibrium's exploitability is below this")
    ap.add_argument("--dedup", type=float, default=1e-2,
                    help="a best response nearer than this to an existing support point is dropped")
    ap.add_argument("--polish", action="store_true",
                    help="refine each grid best response with projected gradient ascent")
    ap.add_argument("--polish-steps", type=int, default=200)
    ap.add_argument("--polish-lr", type=float, default=1e-2)
    ap.add_argument("--init", choices=("center", "random"), default="center")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    game, game_config = load_game(args.config)
    oracle = GridOracle(game, points=args.grid)
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"oracle  : grid argmax over {oracle.grid0.shape[0]} / {oracle.grid1.shape[0]} actions"
          f"{' + gradient polish' if args.polish else ''}")
    print(f"solver  : double oracle  <= {args.iters} rounds  tol={args.tol}  dedup={args.dedup}\n")

    meta = {
        "algorithm": "double_oracle", "config": args.config, "grid": args.grid,
        "iters": args.iters, "tol": args.tol, "dedup": args.dedup, "polish": args.polish,
        "init": args.init, "seed": args.seed,
    }
    writer = CheckpointWriter(args.checkpoint_dir) if args.checkpoint_dir else None
    result = run_double_oracle(
        game, oracle, iters=args.iters, tol=args.tol, dedup=args.dedup, polish=args.polish,
        polish_steps=args.polish_steps, polish_lr=args.polish_lr, init=args.init, seed=args.seed,
        checkpoint_fn=writer,
    )

    print_history(result.history, rows=min(args.log_rows, len(result.history)),
                  columns=("value", "support_0", "support_1"))
    last = result.history[-1]
    r = oracle.cluster_radius()
    print(f"\n{'converged' if result.converged else 'stopped'} after {last['t']} rounds "
          f"({result.stop_reason})  |  exploitability {last['expl']:+.5f}  |  "
          f"value {result.value:+.5f}")
    print(f"  P0 {top_atoms(result.support_0, result.weights_0, top=8, radius=r)}")
    print(f"  P1 {top_atoms(result.support_1, result.weights_1, top=8, radius=r)}")
    meta |= {"converged": result.converged, "stop_reason": result.stop_reason,
             "value": result.value}
    save_history(result.history, args.out, meta)
    if writer is not None:
        print(f"saved {len(writer.entries)} checkpoints -> {writer.write_index(meta)}")


if __name__ == "__main__":
    main()
