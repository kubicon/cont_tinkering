"""Simultaneous incremental support adjustment and metagame solving (SISAMS).

Martin & Sandholm, arXiv:2406.08683 -- double oracle with both of its expensive parts
removed. A strategy is a **fixed-cardinality** support: `n` actions per player plus a
weight vector over them, so memory is constant rather than growing by one pure strategy
per iteration. Each iteration does two things at once, both incrementally:

  * **support adjustment** -- every atom takes a gradient step on *its own* deviation
    utility against the opponent's current mixture, so a support point that is a poor
    pure strategy improves instead of waiting for a global best-response oracle to
    replace it;
  * **metagame solving** -- the weights take a subgradient step on the exploitability of
    the current profile *within the metagame*,

        Phi(w) = max_j (U w_1)_j  -  min_k (w_0^T U)_k,

    whose subgradients are the best-responding row and column, instead of solving the
    restricted game exactly by LP.

That makes it the closest published relative of the mixture head this repo is about:
`n` atoms whose positions move and whose weights are updated, differing from a Gaussian
mixture in that the atoms are points rather than smoothed components, and from
`double_oracle.py` in that nothing is ever solved or best-responded to exactly. It
belongs here rather than in `baselines/neural/` because it has no network: in a one-shot
game a pure strategy *is* an action, and the gradients of these games' payoffs are exact.

Two metrics are reported and they answer different questions. `expl` is the true
exploitability against a fine deviation grid -- how good the strategy actually is.
`metagame_expl` is `Phi` above, what the algorithm itself minimizes -- how good it is
*within its own support*. `metagame_expl` near zero while `expl` stays high is the
signature of a support that has converged to the wrong points, which is exactly the
failure mode a fixed cardinality can have and an oracle-based method cannot.

**Fidelity note.** The paper's "rank-based mixing operator", which weights atoms by the
ordinal ranking of their utilities to encourage specialization, is reconstructed here from
a secondary description rather than the paper's own text, so it is off by default and
`--rank-weighting` enables the reconstruction (per-atom step sizes scaled by their
utility rank). The support/weight updates above are the paper's as stated; the multi-player
generalization is not implemented -- these games are two-player zero-sum.

Usage:
    python -m baselines.sisa configs/two_point.yaml --atoms 4 --iters 20000
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from games.base import ZeroSumGame
from games.spaces import simplex

from .common import ChunkRunner, CheckpointWriter, GridOracle, StrategyPair, action_bounds, \
    base_parser, effective_support, load_game, print_history, save_history, top_atoms


@dataclasses.dataclass
class SISAResult:
    support_0: np.ndarray
    weights_0: np.ndarray
    support_1: np.ndarray
    weights_1: np.ndarray
    history: list[dict]


def _init_support(lo: np.ndarray, hi: np.ndarray, atoms: int, mode: str,
                  rng: np.random.Generator) -> np.ndarray:
    if mode == "random":
        return rng.uniform(lo, hi, size=(atoms, lo.shape[0]))
    if mode != "spread":
        raise ValueError(f"unknown init {mode!r} (choices: random, spread)")
    if lo.shape[0] == 1:
        return np.linspace(lo[0], hi[0], atoms).reshape(-1, 1)
    return np.stack([rng.permutation(np.linspace(l, h, atoms)) for l, h in zip(lo, hi)], axis=-1)


def build_iteration(game: ZeroSumGame, lo, hi, rank_weighting: bool,
                    lr_support: float, lr_weight: float, weight_schedule: str = "sqrt"):
    """One SISAMS iteration, jitted: adjust the support, then the weights.

    `weight_schedule` is not a detail. `Phi` is convex but *piecewise linear* in the
    weights -- its subgradient is a single best-responding row or column -- so a constant
    step drives the weights straight to a vertex of the simplex and keeps them there: the
    profile becomes a pure strategy, unexploitable within its own (now single-point)
    support and badly exploitable in the game. The paper's `alpha_t` schedule is what
    makes the subgradient method converge to the metagame equilibrium instead; `"sqrt"`
    is the textbook `alpha_0 / sqrt(1 + t)` for a nonsmooth convex objective, and
    `"constant"` reproduces the collapse if you want to see it.
    """
    lo_0, hi_0 = jnp.asarray(lo[0]), jnp.asarray(hi[0])
    lo_1, hi_1 = jnp.asarray(lo[1]), jnp.asarray(hi[1])

    def deviation_utilities(x0, x1, w0, w1):
        """`(q0, q1, U)`: each atom's utility against the opponent's mixture, and the
        metagame's payoff matrix (which the weight step needs anyway)."""
        payoff = jax.vmap(lambda a: jax.vmap(lambda b: game.payoff(a, b))(x1))(x0)  # (n0, n1)
        return payoff @ w1, -(w0 @ payoff), payoff

    def _rank_scale(values):
        """Per-atom step scaling by the ordinal rank of the atom's utility, normalized to
        mean 1. See the module docstring's fidelity note."""
        order = jnp.argsort(jnp.argsort(values))            # 0 = worst
        ranks = (order + 1).astype(values.dtype)
        return ranks / jnp.mean(ranks)

    def iteration(carry, _):
        x0, x1, w0, w1, step = carry

        # --- support: every atom ascends its own deviation utility --------------
        q0, q1, _ = deviation_utilities(x0, x1, w0, w1)
        grad_0 = jax.jacrev(lambda x: deviation_utilities(x, x1, w0, w1)[0])(x0)
        grad_1 = jax.jacrev(lambda x: deviation_utilities(x0, x, w0, w1)[1])(x1)
        # `jacrev` of an (n,) output w.r.t. an (n, d) input is (n, n, d); only the
        # diagonal is nonzero -- atom j's utility depends on atom j alone -- and taking it
        # is what makes this `n` independent ascents rather than one coupled step.
        step_0 = jnp.diagonal(grad_0, axis1=0, axis2=1).T
        step_1 = jnp.diagonal(grad_1, axis1=0, axis2=1).T
        if rank_weighting:
            step_0 = step_0 * _rank_scale(q0)[:, None]
            step_1 = step_1 * _rank_scale(q1)[:, None]
        x0 = jnp.clip(x0 + lr_support * step_0, lo_0, hi_0)
        x1 = jnp.clip(x1 + lr_support * step_1, lo_1, hi_1)

        # --- weights: subgradient descent on the metagame's exploitability ------
        _, _, payoff = deviation_utilities(x0, x1, w0, w1)
        best_row = jnp.argmax(payoff @ w1)          # player 0's best pure reply in the metagame
        best_col = jnp.argmin(w0 @ payoff)          # player 1's
        alpha = lr_weight if weight_schedule == "constant" else \
            lr_weight / jnp.sqrt(1.0 + step.astype(jnp.float64))
        # d Phi / d w1 = payoff[best_row, :] (descend it); d Phi / d w0 = -payoff[:, best_col].
        w0 = simplex(w0.shape[0]).clip(w0 + alpha * payoff[:, best_col])
        w1 = simplex(w1.shape[0]).clip(w1 - alpha * payoff[best_row, :])
        return (x0, x1, w0, w1, step + 1), None

    return iteration


def collapse_warning(entry: dict, threshold: float = 0.05) -> str | None:
    """The message to print when the run ended unexploitable *within its support* but not
    in the game -- i.e. every atom of some player drifted into one basin.

    This is the characteristic failure of a fixed-cardinality support with no
    specialization mechanism: once a player's atoms are co-located they share a gradient
    and can never separate again, so the metagame has no mixture left to find and its
    exploitability goes to zero while the real one does not.
    """
    if entry["metagame_expl"] < threshold <= entry["expl"]:
        return (f"NOTE: metagame exploitability {entry['metagame_expl']:+.5f} but true "
                f"exploitability {entry['expl']:+.5f} -- the support has collapsed into "
                "too few distinct actions. Try --init spread, more --atoms, or another --seed.")
    return None


def metagame_exploitability(payoff: np.ndarray, w0: np.ndarray, w1: np.ndarray) -> float:
    """`Phi`: exploitability of the profile *restricted to its own support*."""
    return float(np.max(payoff @ w1) - np.min(w0 @ payoff))


def run_sisa(
    game: ZeroSumGame,
    oracle: GridOracle,
    atoms: int = 4,
    iters: int = 20_000,
    lr_support: float = 1e-2,
    lr_weight: float = 1e-2,
    rank_weighting: bool = False,
    weight_schedule: str = "sqrt",
    init: str = "random",
    seed: int = 0,
    log_every: int | None = None,
    checkpoint_fn=None,
    init_support: tuple[np.ndarray, np.ndarray] | None = None,
    score: bool = True,
) -> SISAResult:
    """`iters` SISAMS iterations on a fixed support of `atoms` actions per player.

    `init_support` pins both players' starting atoms (overriding `atoms`/`init`) -- for
    tests, and for warm-starting from another baseline's checkpoint, since a
    `StrategyPair`'s supports are exactly this shape.
    """
    log_every = log_every or max(iters // 100, 1)
    rng = np.random.default_rng(seed)
    lo = (oracle.lo0, oracle.lo1)
    hi = (oracle.hi0, oracle.hi1)
    radius = oracle.cluster_radius()

    if init_support is None:
        x0 = jnp.asarray(_init_support(lo[0], hi[0], atoms, init, rng))
        x1 = jnp.asarray(_init_support(lo[1], hi[1], atoms, init, rng))
    else:
        x0, x1 = (jnp.asarray(np.asarray(s, dtype=np.float64)) for s in init_support)
        if x0.shape[0] != x1.shape[0]:
            raise ValueError("init_support must give both players the same number of atoms")
        atoms = x0.shape[0]
    weights = jnp.full(atoms, 1.0 / atoms)
    carry = (x0, x1, weights, weights, jnp.zeros((), dtype=jnp.int64))

    iteration = build_iteration(game, lo, hi, rank_weighting, lr_support, lr_weight,
                                weight_schedule)
    runner = ChunkRunner(
        lambda n: lambda c, _: jax.lax.scan(iteration, c, None, length=n)[0])

    def record(t: int, carry) -> dict:
        x0, x1, w0, w1 = (np.asarray(v) for v in carry[:4])
        payoff = oracle.payoff(x0, x1)          # n0 x n1 -- the metagame, cheap
        entry = {
            "t": int(t),
            "wall_time": runner.run_seconds,
            "compile_time": runner.compile_seconds,
            # Two payoff matrices (one per half-step) plus two Jacobian passes.
            "payoff_evals": int(t) * 4 * x0.shape[0] * x1.shape[0],
            "metagame_expl": metagame_exploitability(payoff, w0, w1),
            "value": float(w0 @ payoff @ w1),
            "support_0": effective_support(w0, support=x0, radius=radius),
            "support_1": effective_support(w1, support=x1, radius=radius),
        }
        if score:
            entry["expl"] = float(oracle.exploitability(x0, w0, x1, w1))
        if checkpoint_fn is not None:
            checkpoint_fn(StrategyPair(t=int(t), support_0=x0, weights_0=w0,
                                       support_1=x1, weights_1=w1))
        return entry

    history = [record(0, carry)]
    done = 0
    while done < iters:
        length = min(log_every, iters - done)
        carry = runner(carry, None, length)
        done += length
        history.append(record(done, carry))

    x0, x1, w0, w1 = (np.asarray(v) for v in carry[:4])
    return SISAResult(x0, w0, x1, w1, history)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--atoms", type=int, default=4, help="support points per player (fixed)")
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--lr-support", type=float, default=1e-2, help="step on the atom positions")
    ap.add_argument("--lr-weight", type=float, default=1e-2, help="step on the metagame weights")
    ap.add_argument("--weight-schedule", choices=("sqrt", "constant"), default="sqrt",
                    help="subgradient stepsize schedule for the weights; see build_iteration")
    ap.add_argument("--rank-weighting", action="store_true",
                    help="reconstructed rank-based mixing operator (see the fidelity note)")
    ap.add_argument("--init", choices=("random", "spread"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    game, game_config = load_game(args.config)
    oracle = GridOracle(game, points=args.grid)
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"solver  : SISAMS  {args.atoms} atoms/player  lr_support={args.lr_support}  "
          f"lr_weight={args.lr_weight} ({args.weight_schedule})  init={args.init}"
          f"{'  rank-weighted' if args.rank_weighting else ''}")
    print(f"metric  : `expl` against a {oracle.grid0.shape[0]}-point grid; "
          f"`metagame_expl` within the support\n")

    meta = {"algorithm": "sisa", "config": args.config,
            **{k: v for k, v in vars(args).items() if k != "config"}}
    writer = CheckpointWriter(args.checkpoint_dir) if args.checkpoint_dir else None
    result = run_sisa(game, oracle, atoms=args.atoms, iters=args.iters,
                      lr_support=args.lr_support, lr_weight=args.lr_weight,
                      rank_weighting=args.rank_weighting, weight_schedule=args.weight_schedule,
                      init=args.init, seed=args.seed, checkpoint_fn=writer)

    print_history(result.history, rows=args.log_rows,
                  columns=("metagame_expl", "support_0", "support_1"))
    last = result.history[-1]
    r = oracle.cluster_radius()
    print(f"\nfinal exploitability {last['expl']:+.5f}  |  "
          f"within its own support {last['metagame_expl']:+.5f}")
    print(f"  P0 {top_atoms(result.support_0, result.weights_0, top=8, radius=r)}")
    print(f"  P1 {top_atoms(result.support_1, result.weights_1, top=8, radius=r)}")
    warning = collapse_warning(last)
    if warning:
        print(f"\n{warning}")
    save_history(result.history, args.out, meta)
    if writer is not None:
        print(f"saved {len(writer.entries)} checkpoints -> {writer.write_index(meta)}")


if __name__ == "__main__":
    main()
