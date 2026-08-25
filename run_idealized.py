"""CLI entry point for the idealized (noise-free, exact-gradient) MMD solver.

The counterpart of `train.py`, but for `idealized_mmd.py` instead of the PPO
trainers: same YAML-config style, same `games.configs.GAME_CONFIGS` registry, but
no sampling, no critic, no PPO clipping -- just the MMD vector field on a
parametric Gaussian mixture. Use it to ask whether a failure is intrinsic to the
game geometry or an artifact of PPO/noise.

Two payoff backends, auto-selected (override with `mmd.backend`):

  closed_form -- exact Gaussian convolutions. Requires the game to expose the
                 `MultiPointGame` structure (`.peaks`, `._target_moments`, ...);
                 covers `multi_point` and `decoy_well`.
  quadrature  -- discretizes the action interval and integrates numerically.
                 Works for ANY 1-D box game (`forsaken`, `matching_pennies`, ...),
                 at the cost of grid error. This is what lets the idealized solver
                 run games the closed form was never written for.

Example:
  python run_idealized.py configs/idealized_two_point.yaml
  python run_idealized.py configs/idealized_decoy_well.yaml
  python run_idealized.py configs/idealized_forsaken.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from games.base import ZeroSumGame
from games.configs import GAME_CONFIGS
from games.spaces import BoxSpace
from idealized_mmd import (
    Params,
    categorical_mirror_update,
    component_q as closed_form_component_q,
    expected_payoff as closed_form_expected_payoff,
    exploitability as closed_form_exploitability,
    gaussian_kl,
)


# --------------------------------------------------------------------------- config


@dataclasses.dataclass
class MMDSection:
    """MMD hyperparameters. Mirrors `idealized_mmd.MMDConfig`, plus the knobs the
    convergence experiments showed to matter."""

    lr: float = 0.05
    steps: int = 20000
    magnet_interval: int = 200
    magnet_coef: float = 0.2          # tau: proximal weight on KL(. || magnet)
    entropy_coef: float = 0.0         # pull of the categorical head toward uniform
    num_components: int = 2
    train_means: bool = True
    train_std: bool = True

    # Graduated optimization: hold every component's std >= this early, then relax
    # the floor to `std_min` over training. 0 disables.
    anneal_std_from: float = 0.0
    std_min: float = 1e-3
    std_max: float = 1.0

    # Mean-repulsion sweep (see convergence_experiments/counterexample/COUNTEREXAMPLE.md
    # section 4c): adds `coef * |mu_i - mu_j|` to each player's objective, with the
    # coefficient ramped 0 -> coef -> 0. Set `repulsion_coef` to 0 to disable.
    repulsion_coef: float = 0.0
    repulsion_ramp: float = 0.2       # fraction of training spent ramping up
    repulsion_hold: float = 0.5       # fraction spent held at `repulsion_coef`

    # Freeze the categorical head at uniform weights (isolates the Gaussian head).
    freeze_weights: bool = False

    backend: str = "auto"             # "auto" | "closed_form" | "quadrature"
    grid_points: int = 801            # quadrature only


@dataclasses.dataclass
class InitSection:
    """Initial mixture for both players. `means: spread` reproduces the trainer's
    `training.mixture._spread_bias_init` (component k at fraction (k+0.5)/K of the box)."""

    means: Any = "spread"             # "spread" | list[float] | list[list[float]] (per player)
    weights: Any = None               # None (uniform) | list[float]
    log_std: Any = None               # None -> (high-low)/(2K), as the trainer does


@dataclasses.dataclass
class LogSection:
    every: int = 500
    rows: int = 16
    out: str | None = None            # optional path to dump the full history as JSON


@dataclasses.dataclass
class IdealizedRunConfig:
    game: Any
    mmd: MMDSection = dataclasses.field(default_factory=MMDSection)
    init: InitSection = dataclasses.field(default_factory=InitSection)
    log: LogSection = dataclasses.field(default_factory=LogSection)


def _build(cls: type, data: dict) -> Any:
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown field(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> IdealizedRunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"game", "mmd", "init", "log"}
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")

    game_raw = dict(raw.get("game", {}))
    name = game_raw.pop("name", None)
    if name is None:
        raise ValueError("config.game.name is required")
    if name not in GAME_CONFIGS:
        raise ValueError(f"unknown game {name!r}, choices: {sorted(GAME_CONFIGS)}")

    return IdealizedRunConfig(
        game=_build(GAME_CONFIGS[name], game_raw),
        mmd=_build(MMDSection, raw.get("mmd", {}) or {}),
        init=_build(InitSection, raw.get("init", {}) or {}),
        log=_build(LogSection, raw.get("log", {}) or {}),
    )


# --------------------------------------------------------------------------- backends


class ClosedFormBackend:
    """Exact Gaussian convolutions -- only for the `MultiPointGame` family."""

    name = "closed_form"

    def __init__(self, game: ZeroSumGame):
        self.game = game

    def expected_payoff(self, p0: Params, p1: Params):
        return closed_form_expected_payoff(p0, p1, self.game)

    def component_q(self, p: Params, opp: Params, sign: float):
        return closed_form_component_q(p, opp, self.game, sign)

    def exploitability(self, p0: Params, p1: Params):
        return closed_form_exploitability(p0, p1, self.game)


class QuadratureBackend:
    """Numeric integration on a 1-D grid. Works for any 1-D box game.

    The payoff matrix `R[i, j] = payoff(a_i, a_j)` is built once. Expected payoff is
    `p0^T R p1 * dx^2`; best responses are optimized over grid points inside the box
    (actions must be feasible), while the mixture densities are integrated over a
    padded grid so the Gaussian tails are not truncated.
    """

    name = "quadrature"

    def __init__(self, game: ZeroSumGame, n: int, std_max: float):
        space = game.action_space(0)
        if not isinstance(space, BoxSpace) or space.shape != (1,):
            raise ValueError("quadrature backend supports 1-D BoxSpace games only")
        lo, hi = float(space.low[0]), float(space.high[0])
        pad = 4.0 * std_max
        self.grid = jnp.linspace(lo - pad, hi + pad, n, dtype=jnp.float64)
        self.dx = float(self.grid[1] - self.grid[0])
        self.in_box = (self.grid >= lo) & (self.grid <= hi)

        a = self.grid[:, None]
        pay = jax.vmap(lambda x: jax.vmap(lambda y: game.payoff(x, y))(a))(a)
        self.R = jnp.asarray(pay, dtype=jnp.float64)  # (n, n)

    def _density(self, p: Params):
        w = jax.nn.softmax(p.logits)
        s = jnp.exp(p.log_std)
        comp = jnp.exp(-((self.grid[None, :] - p.means[:, None]) ** 2) / (2 * s[:, None] ** 2))
        comp = comp / (jnp.sqrt(2 * jnp.pi) * s[:, None])
        return jnp.sum(w[:, None] * comp, axis=0)  # (n,)

    def expected_payoff(self, p0: Params, p1: Params):
        d0, d1 = self._density(p0), self._density(p1)
        return (d0 @ self.R @ d1) * self.dx**2

    def component_q(self, p: Params, opp: Params, sign: float):
        """Per-component expected utility q_k = E_{a~N(mu_k,s_k)}[ this player's utility ]."""
        d_opp = self._density(opp)
        # utility of a *pure* action a for this player, against the opponent's mixture
        u = sign * (self.R @ d_opp) * self.dx            # (n,) -- player 0 maximizes payoff
        s = jnp.exp(p.log_std)
        comp = jnp.exp(-((self.grid[None, :] - p.means[:, None]) ** 2) / (2 * s[:, None] ** 2))
        comp = comp / (jnp.sqrt(2 * jnp.pi) * s[:, None])
        return (comp @ u) * self.dx                      # (K,)

    def exploitability(self, p0: Params, p1: Params):
        d0, d1 = self._density(p0), self._density(p1)
        U = (d0 @ self.R @ d1) * self.dx**2
        v0 = (self.R @ d1) * self.dx                     # player 0's value of each pure action
        v1 = (d0 @ self.R) * self.dx                     # player 1's cost of each pure action
        br0 = jnp.max(jnp.where(self.in_box, v0, -jnp.inf))
        br1 = jnp.min(jnp.where(self.in_box, v1, jnp.inf))
        return (br0 - U) + (U - br1)


def build_backend(game: ZeroSumGame, mmd: MMDSection):
    supports_closed_form = hasattr(game, "peaks") and hasattr(game, "_target_moments")
    choice = mmd.backend
    if choice == "auto":
        choice = "closed_form" if supports_closed_form else "quadrature"
    if choice == "closed_form":
        if not supports_closed_form:
            raise ValueError(
                f"{type(game).__name__} has no `.peaks`/`._target_moments`, so the closed-form "
                "backend does not apply. Use `mmd.backend: quadrature`."
            )
        return ClosedFormBackend(game)
    if choice == "quadrature":
        return QuadratureBackend(game, mmd.grid_points, mmd.std_max)
    raise ValueError(f"unknown mmd.backend {mmd.backend!r}")


# --------------------------------------------------------------------------- init


def build_init(game: ZeroSumGame, mmd: MMDSection, init: InitSection) -> tuple[Params, Params]:
    space = game.action_space(0)
    lo, hi = float(space.low[0]), float(space.high[0])
    k = mmd.num_components

    def means_for(player: int) -> jnp.ndarray:
        m = init.means
        if m == "spread":  # mirrors training.mixture._spread_bias_init
            frac = (jnp.arange(k, dtype=jnp.float64) + 0.5) / k
            return lo + frac * (hi - lo)
        arr = jnp.asarray(m, dtype=jnp.float64)
        if arr.ndim == 2:  # per-player means
            arr = arr[player]
        if arr.shape[0] != k:
            raise ValueError(f"init.means has {arr.shape[0]} entries, expected num_components={k}")
        return arr

    if init.log_std is None:
        log_std_val = float(np.log((hi - lo) / (2 * k)))  # mirrors _std_bias_init
    else:
        log_std_val = float(np.log(init.log_std)) if init.log_std > 0 else float(init.log_std)

    if init.weights is None:
        logits = jnp.zeros(k, dtype=jnp.float64)
    else:
        w = jnp.asarray(init.weights, dtype=jnp.float64)
        if w.shape[0] != k:
            raise ValueError(f"init.weights has {w.shape[0]} entries, expected num_components={k}")
        logits = jnp.log(w)

    log_std = jnp.full((k,), log_std_val, dtype=jnp.float64)
    log_std = jnp.clip(log_std, jnp.log(mmd.std_min), jnp.log(mmd.std_max))
    return (Params(logits=logits, means=means_for(0), log_std=log_std),
            Params(logits=logits, means=means_for(1), log_std=log_std))


# --------------------------------------------------------------------------- run


def repulsion_coef_at(mmd: MMDSection, frac: float) -> float:
    """Ramp 0 -> coef over `repulsion_ramp`, hold for `repulsion_hold`, then anneal to 0."""
    if mmd.repulsion_coef == 0.0:
        return 0.0
    ramp, hold = mmd.repulsion_ramp, mmd.repulsion_hold
    if frac < ramp:
        return mmd.repulsion_coef * (frac / max(ramp, 1e-9))
    if frac < ramp + hold:
        return mmd.repulsion_coef
    tail = max(1.0 - ramp - hold, 1e-9)
    return mmd.repulsion_coef * max(0.0, (1.0 - frac) / tail)


def std_floor_at(mmd: MMDSection, frac: float) -> float:
    """Anneal the log-std floor from `anneal_std_from` down to `std_min`."""
    base = float(np.log(mmd.std_min))
    if mmd.anneal_std_from <= 0.0:
        return base
    return (1.0 - frac) * float(np.log(mmd.anneal_std_from)) + frac * base


def run(game: ZeroSumGame, mmd: MMDSection, p0: Params, p1: Params, log: LogSection):
    backend = build_backend(game, mmd)
    space = game.action_space(0)
    lo, hi = float(space.low[0]), float(space.high[0])
    log_std_hi = float(np.log(mmd.std_max))

    def step(p0, p1, m0, m1, lam, floor):
        def apply(p, opp, magnet, sign):
            if mmd.freeze_weights:
                logits = p.logits
            else:
                q = backend.component_q(p, opp, sign)
                logits = categorical_mirror_update(
                    p.logits, q, magnet.logits,
                    _CfgShim(mmd.lr, mmd.magnet_coef, mmd.entropy_coef),
                )

            def obj(pp):
                pay = (backend.expected_payoff(pp, opp) if sign > 0
                       else -backend.expected_payoff(opp, pp))
                ent = mmd.entropy_coef * jnp.sum(pp.log_std)
                mag = mmd.magnet_coef * gaussian_kl(pp.means, pp.log_std,
                                                    magnet.means, magnet.log_std)
                rep = lam * jnp.sum(jnp.abs(pp.means[:, None] - pp.means[None, :])) / 2.0
                return pay + ent + rep - mag

            g = jax.grad(obj)(p)
            means = p.means + (mmd.lr * jnp.exp(2 * p.log_std) * g.means if mmd.train_means else 0.0)
            log_std = p.log_std + (mmd.lr * 0.5 * g.log_std if mmd.train_std else 0.0)
            means = jnp.clip(means, lo, hi)
            log_std = jnp.clip(log_std, floor, log_std_hi)
            return Params(logits=logits, means=means, log_std=log_std)

        return apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)

    jstep = jax.jit(step)
    m0, m1 = p0, p1
    history = []
    for t in range(mmd.steps):
        frac = t / max(mmd.steps - 1, 1)
        p0, p1 = jstep(p0, p1, m0, m1,
                       repulsion_coef_at(mmd, frac), std_floor_at(mmd, frac))
        if (t + 1) % mmd.magnet_interval == 0:
            m0, m1 = p0, p1
        if t % log.every == 0 or t == mmd.steps - 1:
            history.append({
                "t": t,
                "expl": float(backend.exploitability(p0, p1)),
                "w0": [float(x) for x in jax.nn.softmax(p0.logits)],
                "means0": [float(x) for x in p0.means],
                "std0": [float(x) for x in jnp.exp(p0.log_std)],
                "w1": [float(x) for x in jax.nn.softmax(p1.logits)],
                "means1": [float(x) for x in p1.means],
                "std1": [float(x) for x in jnp.exp(p1.log_std)],
            })
    return p0, p1, history, backend


def _fmt(w, mu, sd) -> str:
    return "[" + " ".join(f"{a:.2f}@{b:+.2f}(sd{c:.3f})" for a, b, c in zip(w, mu, sd)) + "]"


class _CfgShim:
    """`categorical_mirror_update` reads `.lr`, `.magnet_coef`, `.entropy_coef`."""

    def __init__(self, lr, magnet_coef, entropy_coef):
        self.lr, self.magnet_coef, self.entropy_coef = lr, magnet_coef, entropy_coef


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to a YAML config (see configs/idealized_*.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    game = cfg.game.build()
    p0, p1 = build_init(game, cfg.mmd, cfg.init)

    print(f"game    : {type(game).__name__}  {dataclasses.asdict(cfg.game)}")
    print(f"mmd     : {dataclasses.asdict(cfg.mmd)}")
    p0f, p1f, history, backend = run(game, cfg.mmd, p0, p1, cfg.log)
    print(f"backend : {backend.name}\n")

    n = len(history)
    idx = sorted({round(i * (n - 1) / max(cfg.log.rows - 1, 1)) for i in range(cfg.log.rows)})
    for i in idx:
        e = history[i]
        print(f"  t={e['t']:6d}  expl={e['expl']:+8.4f}   "
              f"P0 {_fmt(e['w0'], e['means0'], e['std0'])}   "
              f"P1 {_fmt(e['w1'], e['means1'], e['std1'])}")

    tail = float(np.mean([h["expl"] for h in history[int(n * 0.7):]]))
    best = min(h["expl"] for h in history)
    print(f"\nfinal exploitability {history[-1]['expl']:+.4f} | "
          f"tail(30%) {tail:+.4f} | best {best:+.4f}")

    if cfg.log.out:
        Path(cfg.log.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.log.out).write_text(json.dumps(history, indent=2))
        print(f"saved history -> {cfg.log.out}")


if __name__ == "__main__":
    main()
