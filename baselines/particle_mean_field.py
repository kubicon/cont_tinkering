"""Particle mean-field dynamics: a Wasserstein--Fisher--Rao flow on measures.

The representation baseline closest to what the mixture head actually does. Each
player carries `M` weighted particles, and the measure moves by the two components
of the WFR (a.k.a. Hellinger--Kantorovich) geometry:

  * **Wasserstein / transport**: every particle ascends the first variation of its
    own utility, `dz/dt = +grad_a U(a; opponent measure)` -- the same gradient a
    mixture component's *mean* follows, minus the Gaussian smoothing;
  * **Fisher--Rao / birth--death**: the weights follow mirror ascent on the same
    per-particle utilities, `w_m ~ w_m exp(eta q_m)` (replicator), which is exactly
    the update the mixture head's categorical logits get. `--tau` turns it into the
    magnetized version, so the weight half of this baseline is literally
    `common.simplex_mirror_update` -- the same step `grid_mmd` runs on grid cells.

Why this is the interesting comparison and not just another optimizer: the
mean-field analyses of two-player zero-sum games (noisy WFR flows over measures)
prove global convergence in the *infinite-particle, infinite-time* limit, and the
birth--death term is what lets mass teleport between separated modes rather than
having to be transported through the valley between them. A `K`-component Gaussian
mixture is a finite-particle version of exactly this object, with the particles
smoothed into Gaussians. So the questions this baseline answers are: how many
particles does the non-parametric version need to reach the same exploitability,
and does it get stuck in the same places?

Three ablations worth running, all one flag:
  `--freeze-weights`  pure Wasserstein flow (particle GDA): no birth--death, so mass
                      can only move by transporting particles through the landscape.
  `--freeze-positions` pure Fisher--Rao flow: a fixed random support with only the
                      weights moving -- i.e. `grid_mmd` on `M` random cells.
  `--temperature`     noisy (entropic) WFR: adds Langevin diffusion to the positions,
                      which is what the mean-field convergence results actually assume.

Usage:
    python -m baselines.particle_mean_field configs/two_point.yaml --particles 64 --iters 20000
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
class ParticleResult:
    support_0: np.ndarray   # (M, d0) final particle positions of player 0
    support_1: np.ndarray   # (M, d1)
    weights_0: np.ndarray   # (M,)
    weights_1: np.ndarray   # (M,)
    history: list[dict]


def _init_positions(lo: np.ndarray, hi: np.ndarray, particles: int, mode: str,
                    rng: np.random.Generator) -> np.ndarray:
    """`(M, d)` starting positions: `"random"` uniform in the box, or `"spread"`,
    an even sweep of the box (a linspace in 1-D, a random-but-stratified draw above
    it, since an even lattice needs `M` to be a `d`-th power)."""
    if mode == "random":
        return rng.uniform(lo, hi, size=(particles, lo.shape[0]))
    if mode != "spread":
        raise ValueError(f"unknown init {mode!r} (choices: random, spread)")
    if lo.shape[0] == 1:
        return np.linspace(lo[0], hi[0], particles).reshape(-1, 1)
    # Latin-hypercube-ish: each axis swept evenly, the axes shuffled independently.
    axes = [rng.permutation(np.linspace(l, h, particles)) for l, h in zip(lo, hi)]
    return np.stack(axes, axis=-1)


def run_particle_mean_field(
    game: ZeroSumGame,
    oracle: GridOracle,
    particles: int = 64,
    iters: int = 20_000,
    lr_position: float = 1e-2,
    lr_weight: float = 1.0,
    tau: float = 0.1,
    tau_ent: float = 0.0,
    magnet_interval: int = 500,
    temperature: float = 0.0,
    freeze_weights: bool = False,
    freeze_positions: bool = False,
    init: str = "random",
    seed: int = 0,
    log_every: int | None = None,
    checkpoint_fn: Callable[[StrategyPair], None] | None = None,
) -> ParticleResult:
    """Self-play WFR particle flow for `iters` steps.

    `lr_position` is the transport step, `lr_weight` the mirror step `eta` on the
    weights, `tau`/`tau_ent`/`magnet_interval` the same magnet and entropy knobs the
    other baselines take. `temperature` > 0 adds Langevin noise of scale
    `sqrt(2 * temperature * lr_position)` to the positions (the entropic/noisy WFR
    flow); positions are clipped back into the action box every step, exactly as
    `run_idealized`'s mixture means are.

    `checkpoint_fn` is called with a `StrategyPair` at every logged iteration -- the
    particle cloud *is* the checkpoint, positions and weights together.
    """
    rng = np.random.default_rng(seed)
    log_every = log_every or max(iters // 100, 1)
    lo0, hi0 = jnp.asarray(oracle.lo0), jnp.asarray(oracle.hi0)
    lo1, hi1 = jnp.asarray(oracle.lo1), jnp.asarray(oracle.hi1)
    radius = oracle.cluster_radius()

    def utility_0(a, z1, w1):
        """`E_{b~nu}[u(a, b)]` -- player 0's utility of the pure action `a`."""
        return jnp.sum(w1 * jax.vmap(lambda b: game.payoff(a, b))(z1))

    def utility_1(b, z0, w0):
        """`E_{a~mu}[-u(a, b)]` -- player 1's utility of the pure action `b`."""
        return -jnp.sum(w0 * jax.vmap(lambda a: game.payoff(a, b))(z0))

    def step(carry, _):
        z0, z1, l0, l1, m0, m1, t, key = carry
        w0, w1 = jax.nn.softmax(l0), jax.nn.softmax(l1)

        # Per-particle utility (the Fisher-Rao driving term) and its gradient in the
        # action (the Wasserstein one), both against the opponent's *current* measure:
        # a simultaneous (Jacobi) update, matching the mixture solver's.
        q0 = jax.vmap(lambda a: utility_0(a, z1, w1))(z0)
        q1 = jax.vmap(lambda b: utility_1(b, z0, w0))(z1)
        g0 = jax.vmap(lambda a: jax.grad(utility_0)(a, z1, w1))(z0)
        g1 = jax.vmap(lambda b: jax.grad(utility_1)(b, z0, w0))(z1)

        t = t + 1
        if freeze_positions:
            nz0, nz1 = z0, z1
        else:
            nz0, nz1 = z0 + lr_position * g0, z1 + lr_position * g1
            if temperature > 0.0:
                key, k0, k1 = jax.random.split(key, 3)
                scale = jnp.sqrt(2.0 * temperature * lr_position)
                nz0 = nz0 + scale * jax.random.normal(k0, nz0.shape, dtype=nz0.dtype)
                nz1 = nz1 + scale * jax.random.normal(k1, nz1.shape, dtype=nz1.dtype)
            nz0, nz1 = jnp.clip(nz0, lo0, hi0), jnp.clip(nz1, lo1, hi1)

        if freeze_weights:
            nl0, nl1 = l0, l1
        else:
            nl0 = simplex_mirror_update(l0, q0, m0, lr_weight, tau, tau_ent)
            nl1 = simplex_mirror_update(l1, q1, m1, lr_weight, tau, tau_ent)

        snapshot = (t % magnet_interval) == 0
        m0 = jnp.where(snapshot, nl0, m0)
        m1 = jnp.where(snapshot, nl1, m1)
        return (nz0, nz1, nl0, nl1, m0, m1, t, key), None

    # One jitted scan per distinct chunk length; `scan`'s length must be static.
    chunk_fns: dict[int, object] = {}

    def chunk(carry, length: int):
        if length not in chunk_fns:
            chunk_fns[length] = jax.jit(
                lambda c, n=length: jax.lax.scan(step, c, None, length=n)[0]
            )
        return chunk_fns[length](carry)

    z0 = jnp.asarray(_init_positions(oracle.lo0, oracle.hi0, particles, init, rng))
    z1 = jnp.asarray(_init_positions(oracle.lo1, oracle.hi1, particles, init, rng))
    logits = jnp.zeros(particles, dtype=jnp.float64)      # uniform weights
    carry = (z0, z1, logits, logits, logits, logits,
             jnp.zeros((), dtype=jnp.int64), jax.random.PRNGKey(seed))

    def record(t: int, c) -> dict:
        z0, z1, l0, l1 = c[0], c[1], c[2], c[3]
        s0, s1 = np.asarray(z0), np.asarray(z1)
        w0, w1 = np.asarray(jax.nn.softmax(l0)), np.asarray(jax.nn.softmax(l1))
        return {
            "t": int(t),
            # Against the *continuous* game: the deviation grid is the oracle's, which
            # the particles are free to sit off, so this is not exact-by-construction
            # the way `grid_mmd`'s metric is.
            "expl": float(oracle.exploitability(s0, w0, s1, w1)),
            "value": float(oracle.value(s0, w0, s1, w1)),
            "support_0": effective_support(w0, support=s0, radius=radius),
            "support_1": effective_support(w1, support=s1, radius=radius),
        }

    def checkpoint(t: int, c) -> None:
        if checkpoint_fn is None:
            return
        checkpoint_fn(StrategyPair(
            t=t,
            support_0=np.asarray(c[0]), weights_0=np.asarray(jax.nn.softmax(c[2])),
            support_1=np.asarray(c[1]), weights_1=np.asarray(jax.nn.softmax(c[3])),
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

    z0, z1, l0, l1 = carry[0], carry[1], carry[2], carry[3]
    return ParticleResult(
        support_0=np.asarray(z0), support_1=np.asarray(z1),
        weights_0=np.asarray(jax.nn.softmax(l0)), weights_1=np.asarray(jax.nn.softmax(l1)),
        history=history,
    )


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--particles", type=int, default=64, help="particles per player")
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--lr-position", type=float, default=1e-2, help="transport (Wasserstein) step")
    ap.add_argument("--lr-weight", type=float, default=1.0, help="mirror (Fisher-Rao) step eta")
    ap.add_argument("--tau", type=float, default=0.1, help="magnet KL weight on the weights")
    ap.add_argument("--tau-ent", type=float, default=0.0, help="KL-to-uniform weight")
    ap.add_argument("--magnet-interval", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Langevin noise on the positions (noisy/entropic WFR flow)")
    ap.add_argument("--freeze-weights", action="store_true",
                    help="pure Wasserstein flow: uniform weights, positions only")
    ap.add_argument("--freeze-positions", action="store_true",
                    help="pure Fisher-Rao flow: fixed support, weights only")
    ap.add_argument("--init", choices=("random", "spread"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    game, game_config = load_game(args.config)
    oracle = GridOracle(game, points=args.grid)
    flow = ("Wasserstein only" if args.freeze_weights else
            "Fisher-Rao only" if args.freeze_positions else "Wasserstein-Fisher-Rao")
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"oracle  : grid argmax over {oracle.grid0.shape[0]} / {oracle.grid1.shape[0]} actions")
    print(f"solver  : {flow}  M={args.particles}  lr_pos={args.lr_position}  "
          f"eta={args.lr_weight}  tau={args.tau}  temp={args.temperature}\n")

    meta = {
        "algorithm": "particle_mean_field", "config": args.config, "grid": args.grid,
        "particles": args.particles, "iters": args.iters, "lr_position": args.lr_position,
        "lr_weight": args.lr_weight, "tau": args.tau, "tau_ent": args.tau_ent,
        "magnet_interval": args.magnet_interval, "temperature": args.temperature,
        "freeze_weights": args.freeze_weights, "freeze_positions": args.freeze_positions,
        "init": args.init, "seed": args.seed,
    }
    writer = CheckpointWriter(args.checkpoint_dir) if args.checkpoint_dir else None
    result = run_particle_mean_field(
        game, oracle, particles=args.particles, iters=args.iters,
        lr_position=args.lr_position, lr_weight=args.lr_weight, tau=args.tau,
        tau_ent=args.tau_ent, magnet_interval=args.magnet_interval,
        temperature=args.temperature, freeze_weights=args.freeze_weights,
        freeze_positions=args.freeze_positions, init=args.init, seed=args.seed,
        checkpoint_fn=writer,
    )

    print_history(result.history, rows=args.log_rows, columns=("value", "support_0", "support_1"))
    last = result.history[-1]
    r = oracle.cluster_radius()
    print(f"\nfinal exploitability {last['expl']:+.5f}  |  value {last['value']:+.5f}")
    print(f"  (particles merged within {r:.3g} of each other -- see common.cluster_atoms)")
    print(f"  P0 {top_atoms(result.support_0, result.weights_0, radius=r)}")
    print(f"  P1 {top_atoms(result.support_1, result.weights_1, radius=r)}")
    save_history(result.history, args.out, meta)
    if writer is not None:
        print(f"saved {len(writer.entries)} checkpoints -> {writer.write_index(meta)}")


if __name__ == "__main__":
    main()
