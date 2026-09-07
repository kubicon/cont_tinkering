"""Tabular magnetic mirror descent on the finely discretized matrix game.

The representation baseline that gives up on parametrization entirely: both action
boxes are discretized into a tensor-product grid, the payoff becomes an `N0 x N1`
matrix, and each player carries a full probability vector over its grid. The update
is the *same* one `run_idealized.categorical_mirror_update` applies to the mixture's
categorical head --

    log pi_{t+1} ~ [eta*q + eta*tau*log rho + log pi_t] / (1 + eta*tau + eta*tau_ent)

the exact argmax of `<pi, q> - tau*KL(pi||magnet) - tau_ent*KL(pi||uniform) - (1/eta)*KL(pi||pi_t)`
-- with `q` the vector of pure-action utilities against the opponent's current mix
(`A @ y` for player 0, `-(x @ A)` for player 1) and the magnet `rho` a snapshot of
the player's own iterate every `--magnet-interval` steps.

What this baseline is *for*: it is the strongest thing you can do in one dimension,
and it converges for a reason that has nothing to do with our geometry -- on the
simplex the game is bilinear, so MMD's last iterate converges without any of the
mixture solver's difficulties (no means to move, no modes to be trapped between, no
non-monotone lifted field). It is therefore both the reference solution (its final
exploitability is the discretization error, nothing else) and the baseline whose
cost is `N^d` per player: the number to point at when arguing that the parametric
mixture is what buys the dimension scaling. Set `--tau 0 --tau-ent 0` to recover
plain (unregularized) mirror ascent / multiplicative weights, whose *last* iterate
famously does not converge while its average does -- both are reported.

Usage:
    python -m baselines.grid_mmd configs/two_point.yaml --iters 20000
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from games.base import ZeroSumGame

from .common import CheckpointWriter, GridOracle, StrategyPair, base_parser, effective_support, \
    load_game, print_history, save_history, simplex_mirror_update, top_atoms


@dataclasses.dataclass
class GridMMDResult:
    grid_0: np.ndarray        # (N0, d0) the discretized action set of player 0
    grid_1: np.ndarray        # (N1, d1) ... of player 1
    weights_0: np.ndarray     # (N0,) final iterate
    weights_1: np.ndarray     # (N1,)
    avg_weights_0: np.ndarray  # (N0,) uniform average of the iterates
    avg_weights_1: np.ndarray  # (N1,)
    history: list[dict]


def _matrix_exploitability(A, x, y):
    """`max_i (A y)_i - min_j (x^T A)_j`, the exact exploitability of `(x, y)` in the
    discretized game.

    Identical to `GridOracle.exploitability` here -- the strategy grid and the
    deviation grid are the same set -- so it is computed from `A` directly and stays
    inside the jitted loop. Against the *continuous* game it is exact up to the grid
    spacing, and that residual is the honest error bar on this baseline.
    """
    return jnp.max(A @ y) - jnp.min(x @ A)


def run_grid_mmd(
    game: ZeroSumGame,
    oracle: GridOracle,
    iters: int = 20_000,
    lr: float = 1.0,
    tau: float = 0.1,
    tau_ent: float = 0.0,
    magnet_interval: int = 500,
    alternating: bool = False,
    log_every: int | None = None,
    checkpoint_fn: Callable[[StrategyPair], None] | None = None,
) -> GridMMDResult:
    """Self-play tabular MMD on `oracle`'s grid for `iters` iterations.

    `lr` is the mirror step `eta`, `tau` the magnet KL weight, `tau_ent` the pull
    toward uniform. `alternating` feeds player 0's *updated* mix into player 1's
    `q` (Gauss-Seidel) instead of the simultaneous (Jacobi) update.
    """
    A = jnp.asarray(oracle.matrix())
    n0, n1 = A.shape
    log_every = log_every or max(iters // 100, 1)
    radius = oracle.cluster_radius()

    def step(carry, _):
        xl, yl, mx, my, ax, ay, t = carry
        x, y = jax.nn.softmax(xl), jax.nn.softmax(yl)

        nxl = simplex_mirror_update(xl, A @ y, mx, lr, tau, tau_ent)
        x_for_y = jax.nn.softmax(nxl) if alternating else x
        nyl = simplex_mirror_update(yl, -(x_for_y @ A), my, lr, tau, tau_ent)

        t = t + 1
        snapshot = (t % magnet_interval) == 0
        mx = jnp.where(snapshot, nxl, mx)
        my = jnp.where(snapshot, nyl, my)
        # Uniform (Cesaro) average of the iterates -- the object classic mirror-descent
        # regret bounds actually control, and the one that converges when the last
        # iterate only cycles.
        scale = 1.0 / t.astype(jnp.float64)
        ax = ax + (jax.nn.softmax(nxl) - ax) * scale
        ay = ay + (jax.nn.softmax(nyl) - ay) * scale
        return (nxl, nyl, mx, my, ax, ay, t), None

    # One jitted scan per distinct chunk length (there are at most two: `log_every`
    # and whatever is left over), as `run_idealized.run` does -- `scan`'s length must
    # be static, so it cannot be an argument.
    chunk_fns: dict[int, object] = {}

    def chunk(carry, length: int):
        if length not in chunk_fns:
            chunk_fns[length] = jax.jit(
                lambda c, n=length: jax.lax.scan(step, c, None, length=n)[0]
            )
        return chunk_fns[length](carry)

    zeros0 = jnp.zeros(n0, dtype=jnp.float64)
    zeros1 = jnp.zeros(n1, dtype=jnp.float64)
    carry = (zeros0, zeros1, zeros0, zeros1,
             jnp.full(n0, 1.0 / n0), jnp.full(n1, 1.0 / n1),
             jnp.zeros((), dtype=jnp.int64))

    def record(t: int, c) -> dict:
        xl, yl, _, _, ax, ay, _ = c
        x, y = jax.nn.softmax(xl), jax.nn.softmax(yl)
        return {
            "t": int(t),
            "expl": float(_matrix_exploitability(A, x, y)),
            "avg_expl": float(_matrix_exploitability(A, ax, ay)),
            "support_0": effective_support(np.asarray(x), support=oracle.grid0, radius=radius),
            "support_1": effective_support(np.asarray(y), support=oracle.grid1, radius=radius),
        }

    def checkpoint(t: int, c) -> None:
        if checkpoint_fn is None:
            return
        xl, yl, _, _, ax, ay, _ = c
        checkpoint_fn(StrategyPair(
            t=t,
            support_0=oracle.grid0, weights_0=np.asarray(jax.nn.softmax(xl)),
            support_1=oracle.grid1, weights_1=np.asarray(jax.nn.softmax(yl)),
            extra={"avg_weights_0": np.asarray(ax), "avg_weights_1": np.asarray(ay)},
        ))

    history = [record(0, carry)]
    checkpoint(0, carry)
    done = 0
    while done < iters:
        length = min(log_every, iters - done)
        carry = chunk(carry, length)
        done += length
        history.append(record(done, carry))
        checkpoint(done, carry)

    xl, yl, _, _, ax, ay, _ = carry
    return GridMMDResult(
        grid_0=oracle.grid0,
        grid_1=oracle.grid1,
        weights_0=np.asarray(jax.nn.softmax(xl)),
        weights_1=np.asarray(jax.nn.softmax(yl)),
        avg_weights_0=np.asarray(ax),
        avg_weights_1=np.asarray(ay),
        history=history,
    )


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--iters", type=int, default=20_000, help="mirror-descent iterations")
    ap.add_argument("--lr", type=float, default=1.0, help="mirror step size eta")
    ap.add_argument("--tau", type=float, default=0.1, help="magnet KL weight (0 = plain mirror ascent)")
    ap.add_argument("--tau-ent", type=float, default=0.0, help="KL-to-uniform weight")
    ap.add_argument("--magnet-interval", type=int, default=500,
                    help="iterations between magnet snapshots")
    ap.add_argument("--alternating", action="store_true",
                    help="update player 1 against player 0's already-updated mix")
    args = ap.parse_args()

    game, game_config = load_game(args.config)
    oracle = GridOracle(game, points=args.grid)
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"grid    : {oracle.grid0.shape[0]} x {oracle.grid1.shape[0]} actions "
          f"({args.grid} per axis)")
    print(f"solver  : tabular MMD  eta={args.lr}  tau={args.tau}  tau_ent={args.tau_ent}  "
          f"magnet every {args.magnet_interval}  "
          f"{'alternating' if args.alternating else 'simultaneous'}\n")

    meta = {
        "algorithm": "grid_mmd", "config": args.config, "grid": args.grid,
        "iters": args.iters, "lr": args.lr, "tau": args.tau, "tau_ent": args.tau_ent,
        "magnet_interval": args.magnet_interval, "alternating": args.alternating,
    }
    writer = CheckpointWriter(args.checkpoint_dir) if args.checkpoint_dir else None
    result = run_grid_mmd(
        game, oracle, iters=args.iters, lr=args.lr, tau=args.tau, tau_ent=args.tau_ent,
        magnet_interval=args.magnet_interval, alternating=args.alternating,
        checkpoint_fn=writer,
    )

    print_history(result.history, rows=args.log_rows, columns=("avg_expl", "support_0", "support_1"))
    last = result.history[-1]
    r = oracle.cluster_radius()
    print(f"\nfinal exploitability {last['expl']:+.5f}  |  average iterate {last['avg_expl']:+.5f}")
    print(f"  (atoms are grid cells merged within {r:.3g} of each other -- see common.cluster_atoms)")
    print(f"  P0 {top_atoms(result.grid_0, result.weights_0, radius=r)}")
    print(f"  P1 {top_atoms(result.grid_1, result.weights_1, radius=r)}")
    print(f"  P0 (avg) {top_atoms(result.grid_0, result.avg_weights_0, radius=r)}")
    print(f"  P1 (avg) {top_atoms(result.grid_1, result.avg_weights_1, radius=r)}")
    save_history(result.history, args.out, meta)
    if writer is not None:
        print(f"saved {len(writer.entries)} checkpoints -> {writer.write_index(meta)}")


if __name__ == "__main__":
    main()
