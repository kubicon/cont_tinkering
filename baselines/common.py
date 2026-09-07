"""Shared machinery for the representation baselines: game loading, the action
grid, payoff blocks, the best-response oracle, and the exploitability metric.

Nothing here is an algorithm -- it is the part every baseline needs and that has
to be *identical* across them for their numbers to be comparable:

  * one exploitability definition, `br0 - br1`, matching
    `run_idealized.QuadratureBackend.exploitability`, so a curve produced here can
    be plotted next to one produced by the mixture solver;
  * one best-response oracle (`GridOracle`), a global grid argmax optionally
    polished by projected gradient ascent. A grid argmax is used rather than
    `ZeroSumGame.best_response`'s multi-start ascent because these games are
    deliberately multi-modal -- a decoy well is exactly the landscape a local
    ascent falls into -- and a baseline that under-reports its own best response
    would report an exploitability that is too *low*, which is the one failure
    mode that would flatter the baseline instead of us.

The solvers here work with a mixed strategy in one common form: a `support`
`(N, d)` of actions and a `weights` `(N,)` probability vector over them. A grid
distribution is that with `support == grid`; a double-oracle iterate is that with
`N` growing; a particle cloud is that with `N == M` particles.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from games.base import ZeroSumGame
from games.configs import GAME_CONFIGS
from games.sequential import SequentialZeroSumGame
from games.spaces import BoxSpace

# The baselines integrate/compare tiny differences in exploitability, exactly as
# `run_idealized.py` does, and for the same reason (a `1e-3`-level residual is the
# difference between "reached the Nash" and "stalled"). Note this is process-wide.
jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- config


def load_game(path: str | Path) -> tuple[ZeroSumGame, Any]:
    """`(game, game_config)` from the `game:` section of any config in `configs/`.

    Both config schemas `run_idealized.load_config` accepts (the shared `train.py`
    one and the legacy standalone one) carry the same `game:` section, and that is
    the only section a baseline reads -- every other knob here belongs to the
    baseline's own algorithm and comes from its CLI, since none of them has a
    Gaussian mixture, a learning rate on a mean, or a magnet on a Gaussian head.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    game_raw = dict(raw.get("game", {}))
    name = game_raw.pop("name", None)
    if name is None:
        raise ValueError(f"{path}: config.game.name is required")
    if name not in GAME_CONFIGS:
        raise ValueError(f"{path}: unknown game {name!r}, choices: {sorted(GAME_CONFIGS)}")

    cls = GAME_CONFIGS[name]
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(game_raw) - set(fields)
    if unknown:
        raise ValueError(f"{path}: unknown game.{name} field(s) {sorted(unknown)}")
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in game_raw.items()}
    game_config = cls(**kwargs)

    game = game_config.build()
    if isinstance(game, SequentialZeroSumGame):
        raise ValueError(
            f"{type(game).__name__} is a sequential game; these baselines discretize a "
            "one-shot action space and have no notion of a game tree. Use train.py."
        )
    return game, game_config


def action_bounds(game: ZeroSumGame, player: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis `(low, high)` of `player`'s action box, or a clear error.

    Same restriction as `run_idealized._bounds`, for the same reason: a simplex
    action space (`blotto`) has no tensor-product grid, so a game with one can only
    be run by `train.py`.
    """
    space = game.action_space(player)
    if not isinstance(space, BoxSpace):
        raise ValueError(
            f"{type(game).__name__} gives player {player} a {type(space).__name__}; the "
            "baselines here discretize a box action space only. Run this game with train.py."
        )
    return (np.asarray(space.low, dtype=np.float64).reshape(-1),
            np.asarray(space.high, dtype=np.float64).reshape(-1))


def tensor_grid(lo: np.ndarray, hi: np.ndarray, points: int) -> np.ndarray:
    """`(points**d, d)` cartesian product of `d` per-axis grids, in C order.

    Endpoints included: the equilibria of these games sit *on* peaks that are often
    placed symmetrically about the box, and an open grid would miss them by half a
    cell for no reason.
    """
    axes = [np.linspace(l, h, points) for l, h in zip(lo, hi)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=-1)


# --------------------------------------------------------------------------- payoffs


# One jitted payoff kernel per game, keyed by identity. Without this cache every call to
# `payoff_block` built a *new* jitted function, so JAX's own compilation cache was never
# hit and each block paid a fresh trace-and-compile -- which does not matter once in a
# solver's setup but dominates `experiments/one_shot_neural/score.py`, where the same two
# shapes are evaluated for every checkpoint of every run. The game is kept alive in the
# value so its `id` cannot be recycled underneath the key.
_PAYOFF_KERNELS: dict[int, tuple[Any, Any]] = {}


def payoff_kernel(game: ZeroSumGame):
    """`(actions_0, actions_1) -> (N0, N1)` payoffs, jitted once per game."""
    cached = _PAYOFF_KERNELS.get(id(game))
    if cached is None:
        kernel = jax.jit(
            lambda a0, a1: jax.vmap(lambda x: jax.vmap(lambda y: game.payoff(x, y))(a1))(a0))
        _PAYOFF_KERNELS[id(game)] = cached = (game, kernel)
    return cached[1]


def payoff_block(game: ZeroSumGame, actions_0: np.ndarray, actions_1: np.ndarray,
                 row_chunk: int = 4096) -> np.ndarray:
    """`U[i, j] = game.payoff(actions_0[i], actions_1[j])`, as a `(N0, N1)` array.

    Rows are evaluated `row_chunk` at a time: the whole thing is one `vmap` over
    `N0 * N1` payoff evaluations, and at the grid sizes these baselines want
    (`N = 2001` in 1-D) tracing that in one shot allocates several GiB of
    intermediates for no benefit. Chunking also keeps the number of *distinct shapes*
    small, which is what lets `payoff_kernel`'s compilation cache be hit.
    """
    a0 = jnp.asarray(actions_0, dtype=jnp.float64)
    a1 = jnp.asarray(actions_1, dtype=jnp.float64)
    kernel = payoff_kernel(game)
    out = [np.asarray(kernel(a0[i:i + row_chunk], a1))
           for i in range(0, a0.shape[0], row_chunk)]
    return np.concatenate(out, axis=0).astype(np.float64)


def polish_action(objective: Callable[[jnp.ndarray], jnp.ndarray], init: np.ndarray,
                  lo: np.ndarray, hi: np.ndarray, steps: int = 200,
                  learning_rate: float = 1e-2) -> np.ndarray:
    """Projected Adam *ascent* on `objective`, started at `init` and clipped to the box.

    Used to refine a grid argmax off the grid. It is a local ascent, so it is only
    ever run from the global grid maximizer -- which is what keeps the multi-modality
    of these games from mattering.
    """
    opt = optax.adam(learning_rate)
    lo_j, hi_j = jnp.asarray(lo), jnp.asarray(hi)

    def step(carry, _):
        a, state = carry
        grad = jax.grad(lambda x: -objective(x))(a)
        updates, state = opt.update(grad, state, a)
        return (jnp.clip(optax.apply_updates(a, updates), lo_j, hi_j), state), None

    a0 = jnp.asarray(init, dtype=jnp.float64)
    (final, _), _ = jax.lax.scan(step, (a0, opt.init(a0)), xs=None, length=steps)
    # An ascent that overshot is worse than not having run: keep whichever is better.
    return np.asarray(final if objective(final) > objective(a0) else a0)


# --------------------------------------------------------------------------- mirror step


def simplex_mirror_update(logits, q, magnet_logits, eta, tau, tau_ent):
    """One closed-form MMD step on the simplex -> new logits.

    The exact argmax of

        <pi, q> - tau*KL(pi||magnet) - tau_ent*KL(pi||uniform) - (1/eta)*KL(pi||pi_t),

    i.e. `run_idealized.categorical_mirror_update` without the PPO trust-region term
    (there is no rollout here to be off-policy with respect to). Shared by the two
    baselines that carry an explicit weight vector -- `grid_mmd` over grid cells and
    `particle_mean_field` over particles -- so that the *only* difference between them
    is what the weights sit on. `tau = tau_ent = 0` recovers plain mirror ascent
    (multiplicative weights).
    """
    lp = jax.nn.log_softmax(logits)
    lm = jax.nn.log_softmax(magnet_logits)
    return (eta * q + eta * tau * lm + lp) / (1.0 + eta * tau + eta * tau_ent)


# --------------------------------------------------------------------------- oracle


class GridOracle:
    """Best responses and exploitability against a fine grid of pure deviations.

    Holds one grid per player (they may have different boxes, and different
    dimensions -- `quadratic_asymmetric` does). A best response to *any* mixed
    strategy is pure, so maximizing over this grid is a genuine best-response
    oracle up to the grid spacing, which `polish` removes to gradient-ascent
    accuracy.

    The exploitability it reports,

        max_a E_{b~y}[u(a, b)]  -  min_b E_{a~x}[u(a, b)],

    is exactly `run_idealized.QuadratureBackend.exploitability`'s `(br0 - U) + (U - br1)`,
    with the mixture densities replaced by the finitely supported strategies these
    baselines carry. It is 0 at a Nash and positive otherwise, whatever the support.
    """

    def __init__(self, game: ZeroSumGame, points: int = 1001, row_chunk: int = 256):
        self.game = game
        self.points = points
        self.row_chunk = row_chunk
        self.lo0, self.hi0 = action_bounds(game, 0)
        self.lo1, self.hi1 = action_bounds(game, 1)
        self.grid0 = tensor_grid(self.lo0, self.hi0, points)
        self.grid1 = tensor_grid(self.lo1, self.hi1, points)

    def cluster_radius(self, fraction: float = 0.02) -> float:
        """Default merge radius for `cluster_atoms`: `fraction` of the widest axis of
        either box. Wide enough to swallow a few grid cells, narrow enough that two
        peaks of one of these games (separated by order 1) never merge."""
        widths = np.concatenate([self.hi0 - self.lo0, self.hi1 - self.lo1])
        return float(fraction * np.max(widths))

    # -- payoff blocks -----------------------------------------------------
    def payoff(self, actions_0: np.ndarray, actions_1: np.ndarray) -> np.ndarray:
        return payoff_block(self.game, actions_0, actions_1, self.row_chunk)

    def matrix(self) -> np.ndarray:
        """The full `grid0 x grid1` payoff matrix -- the discretized matrix game.

        `grid_mmd` is the only baseline that wants this; the others only ever need
        one grid against a small support.
        """
        n0, n1 = self.grid0.shape[0], self.grid1.shape[0]
        gib = n0 * n1 * 8 / 2**30
        if gib > 8.0:
            raise ValueError(
                f"a {n0} x {n1} payoff matrix needs {gib:.1f} GiB. Lower --grid "
                f"(currently {self.points} points per axis, dim "
                f"{self.grid0.shape[1]}/{self.grid1.shape[1]})."
            )
        return self.payoff(self.grid0, self.grid1)

    # -- best responses ----------------------------------------------------
    def best_response(self, player: int, support: np.ndarray, weights: np.ndarray,
                      polish: bool = False, polish_steps: int = 200,
                      polish_lr: float = 1e-2) -> tuple[np.ndarray, float]:
        """`(action, value)`: `player`'s best pure reply to the opponent's mixture.

        `support`/`weights` describe the *opponent's* strategy. Player 0 maximizes
        `E[u(a, b)]`, player 1 maximizes `E[-u(a, b)]`.
        """
        weights = np.asarray(weights, dtype=np.float64)
        if player == 0:
            values = self.payoff(self.grid0, support) @ weights   # (M0,)
            grid, lo, hi = self.grid0, self.lo0, self.hi0
            def objective(a):
                w = jnp.asarray(weights)
                s = jnp.asarray(support, dtype=jnp.float64)
                return jnp.sum(w * jax.vmap(lambda b: self.game.payoff(a, b))(s))
        elif player == 1:
            values = -(weights @ self.payoff(support, self.grid1))  # (M1,)
            grid, lo, hi = self.grid1, self.lo1, self.hi1
            def objective(a):
                w = jnp.asarray(weights)
                s = jnp.asarray(support, dtype=jnp.float64)
                return -jnp.sum(w * jax.vmap(lambda b: self.game.payoff(b, a))(s))
        else:
            raise ValueError(f"player must be 0 or 1, got {player}")

        idx = int(np.argmax(values))
        action = grid[idx]
        if polish:
            action = polish_action(objective, action, lo, hi, polish_steps, polish_lr)
            return action, float(objective(jnp.asarray(action)))
        return action, float(values[idx])

    # -- metrics -----------------------------------------------------------
    def value(self, support_0, weights_0, support_1, weights_1) -> float:
        """`E[u]` under the two mixed strategies."""
        u = self.payoff(support_0, support_1)
        return float(np.asarray(weights_0) @ u @ np.asarray(weights_1))

    def exploitability(self, support_0, weights_0, support_1, weights_1,
                       polish: bool = False) -> float:
        _, v0 = self.best_response(0, support_1, weights_1, polish=polish)
        _, v1 = self.best_response(1, support_0, weights_0, polish=polish)
        # `v1` is already player 1's utility, i.e. `-min_b E[u]`, so the sum of the
        # two deviation values *is* `br0 - br1`.
        return v0 + v1


# --------------------------------------------------------------------------- chunked runs


class ChunkRunner:
    """One compiled `lax.scan` per chunk length, with compilation timed separately.

    Every solver here runs in chunks so it can log between them, which means a jitted
    function per distinct chunk length. Two things about that matter for a wall-time
    comparison and are easy to get wrong:

      * **compilation is not training.** Ahead-of-time `lower(...).compile()` is used so
        the compile cost lands in `compile_seconds` instead of being smeared into the
        first chunk's runtime -- where it would look like a slow method rather than a
        slow compiler.
      * **JAX is asynchronous.** `block_until_ready` is what makes the timer measure the
        computation rather than the dispatch; without it a fast-looking method is just
        one that queued its work and returned.

    `build(length) -> (carry, xs) -> result` supplies the scan for a given chunk length.
    """

    def __init__(self, build: Callable[[int], Callable]):
        self.build = build
        self.compiled: dict[int, Any] = {}
        self.compile_seconds = 0.0
        self.run_seconds = 0.0

    def __call__(self, carry, xs, length: int):
        if length not in self.compiled:
            started = time.monotonic()
            self.compiled[length] = jax.jit(self.build(length)).lower(carry, xs).compile()
            self.compile_seconds += time.monotonic() - started
        started = time.monotonic()
        result = jax.block_until_ready(self.compiled[length](carry, xs))
        self.run_seconds += time.monotonic() - started
        return result


# --------------------------------------------------------------------------- checkpoints


@dataclasses.dataclass
class StrategyPair:
    """Both players' strategies at one point in a run -- the checkpoint format.

    Every baseline here represents a mixed strategy the same way (a support of
    actions and a probability vector over it), so one format serves all three, and
    a downstream consumer -- a plot, an exploitability re-evaluation on a finer
    grid, a warm start -- never has to know which algorithm produced it. `extra`
    carries whatever else an algorithm wants kept (`grid_mmd` stores its averaged
    iterate there); it is written into the same file and read back by name.

    Note the supports are *not* the same shape across a `double_oracle` run: they
    grow by one action per round. One file per checkpoint, rather than one stacked
    array per run, is what keeps that from mattering.
    """

    t: int
    support_0: np.ndarray
    weights_0: np.ndarray
    support_1: np.ndarray
    weights_1: np.ndarray
    extra: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t=np.asarray(self.t),
            support_0=np.asarray(self.support_0), weights_0=np.asarray(self.weights_0),
            support_1=np.asarray(self.support_1), weights_1=np.asarray(self.weights_1),
            **{f"extra__{k}": np.asarray(v) for k, v in self.extra.items()},
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StrategyPair":
        with np.load(path) as z:
            extra = {k[len("extra__"):]: z[k] for k in z.files if k.startswith("extra__")}
            return cls(int(z["t"]), z["support_0"], z["weights_0"],
                       z["support_1"], z["weights_1"], extra)


class CheckpointWriter:
    """`checkpoint_fn` for the baselines: writes one `.npz` per call into `directory`.

    Files are named by iteration (`step_000020000.npz`), so they sort chronologically
    and a run that was interrupted still leaves a usable prefix behind. `write_index`
    drops an `index.json` next to them listing what was written and under which
    settings -- the thing to read first when picking a checkpoint up later.
    """

    def __init__(self, directory: str | Path, prefix: str = "step"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.entries: list[dict] = []

    def __call__(self, pair: StrategyPair) -> None:
        path = self.directory / f"{self.prefix}_{pair.t:09d}.npz"
        pair.save(path)
        self.entries.append({
            "t": int(pair.t), "file": path.name,
            "support_0": int(np.asarray(pair.support_0).shape[0]),
            "support_1": int(np.asarray(pair.support_1).shape[0]),
        })

    def write_index(self, meta: dict | None = None) -> Path:
        path = self.directory / "index.json"
        path.write_text(json.dumps({"meta": meta or {}, "checkpoints": self.entries}, indent=2))
        return path


def load_checkpoints(directory: str | Path) -> list[StrategyPair]:
    """Every checkpoint in `directory`, oldest first. Reads the files, not the index,
    so a run whose index was never written (interrupted) still loads."""
    return [StrategyPair.load(p) for p in sorted(Path(directory).glob("*.npz"))]


def latest_checkpoint(directory: str | Path) -> StrategyPair:
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no checkpoints in {directory}")
    return StrategyPair.load(files[-1])


# --------------------------------------------------------------------------- reporting


def cluster_atoms(support: np.ndarray, weights: np.ndarray, radius: float):
    """Merge atoms lying within `radius` of each other into one weighted atom.

    A discretized strategy spreads mass over several adjacent cells -- the payoff
    difference between neighbouring grid points is tiny next to the magnet's pull,
    so MMD's fixed point is a narrow *bump* around a peak rather than a single cell,
    and the same is true of a particle cloud that has collapsed onto a peak. Reading
    that as "support size 20" would be an artifact of the representation, not a fact
    about the strategy, so support sizes and printed atoms are reported after this
    merge. `(centers, masses)`, heaviest first; the center of a cluster is the
    weight-weighted mean of its members, i.e. where the bump actually sits.
    """
    support = np.asarray(support, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(-weights)
    taken = np.zeros(weights.shape[0], dtype=bool)
    centers, masses = [], []
    for i in order:
        if taken[i] or weights[i] <= 0.0:
            continue
        near = (~taken) & (np.linalg.norm(support - support[i], axis=-1) <= radius)
        mass = float(np.sum(weights[near]))
        centers.append(np.sum(weights[near, None] * support[near], axis=0) / max(mass, 1e-300))
        masses.append(mass)
        taken |= near
    return np.asarray(centers), np.asarray(masses)


def top_atoms(support: np.ndarray, weights: np.ndarray, top: int = 6,
              threshold: float = 1e-3, radius: float | None = None) -> str:
    """The heaviest atoms of a finitely supported strategy, as `w@action` fields.

    Deliberately the same shape of line as `run_idealized._fmt`, so a baseline's
    output can be eyeballed against the mixture solver's without re-reading either.
    """
    if radius is not None:
        support, weights = cluster_atoms(support, weights, radius)
    weights = np.asarray(weights, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    order = np.argsort(-weights)[:top]
    parts = []
    for k in order:
        if weights[k] < threshold:
            continue
        a = np.atleast_1d(support[k])
        body = f"{a[0]:+.3f}" if a.size == 1 else "(" + ",".join(f"{x:+.2f}" for x in a) + ")"
        parts.append(f"{weights[k]:.3f}@{body}")
    mass = float(np.sum(np.sort(weights)[::-1][:top]))
    return "[" + " ".join(parts) + "]" + (f" ({mass:.1%} of mass)" if mass < 0.999 else "")


def effective_support(weights: np.ndarray, threshold: float = 1e-3,
                      support: np.ndarray | None = None,
                      radius: float | None = None) -> int:
    """How many atoms carry more than `threshold` weight -- the support size a
    baseline actually converged to, which is the quantity a mixture head with `K`
    components has to match. With `support`/`radius`, adjacent cells are merged
    first (see `cluster_atoms`), which is what makes the count comparable across
    grid resolutions."""
    weights = np.asarray(weights)
    if radius is not None and support is not None:
        _, weights = cluster_atoms(support, weights, radius)
    return int(np.sum(weights > threshold))


def save_history(history: Sequence[dict], out: str | Path | None, extra: dict | None = None) -> None:
    if not out:
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": extra or {}, "history": list(history)}, indent=2))
    print(f"saved history -> {path}")


def print_history(history: Sequence[dict], rows: int = 12, columns: Sequence[str] = ()) -> None:
    """`rows` evenly spaced entries of `history`, always including the last one."""
    n = len(history)
    idx = sorted({round(i * (n - 1) / max(rows - 1, 1)) for i in range(rows)})
    for i in idx:
        e = history[i]
        extra = "  ".join(
            f"{c}={e[c]:+9.5f}" if isinstance(e[c], float) else f"{c}={e[c]}"
            for c in columns if c in e
        )
        # `expl` is absent when a run was scored offline instead of during training
        # (see `experiments/one_shot_neural/`), which is the fast path.
        head = f"  t={e['t']:7d}"
        if "expl" in e:
            head += f"  expl={e['expl']:+9.5f}"
        if "wall_time" in e:
            head += f"  {e['wall_time']:7.1f}s"
        print(f"{head}   {extra}")


def base_parser(description: str) -> argparse.ArgumentParser:
    """The CLI flags every baseline shares: which game, how fine the oracle grid is,
    and where to write the history."""
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to any config in configs/ (only its `game:` section is read)")
    ap.add_argument("--grid", type=int, default=1001,
                    help="points per axis in the best-response / discretization grid (default 1001)")
    ap.add_argument("--log-rows", type=int, default=12, help="history rows to print (default 12)")
    ap.add_argument("--out", default=None, help="write the full history to this JSON path")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="write one StrategyPair .npz per logged iteration into this directory")
    return ap
