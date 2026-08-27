"""CLI entry point for the idealized (noise-free, exact-gradient) MMD solver.

The counterpart of `train.py`, and it takes **the same YAML run config**: the
`training.run_config.RunConfig` schema (`game` / `network` / `optimizer` /
`ppo` / `train`). Same game, same policy class (a K-component Gaussian
mixture per player), same MMD hyperparameters -- but with every PPO
approximation stripped out:

  train.py                                run_idealized.py
  ------------------------------------    ------------------------------------
  `ppo.batch_size` sampled actions        exact payoff integral, no sampling
  learned critic baseline + advantage     exact per-component q-values
    normalization
  clipped importance ratios               true mirror step: closed form on the
                                            simplex, natural gradient (Fisher
                                            metric) on the Gaussians
  Adam/AdamW on network weights           the KL geometry itself
  policy = MLP(obs) -> four heads         policy = the mixture parameters

Because every game in `games/` hands both players a constant observation, the
network in `train.py` is only an overparameterized container for
`(logits, means, log_std, value)` -- so the two scripts optimize the *same*
strategy space and a difference between their runs is attributable to
sampling noise and the optimizer/parameterization, nothing else.

  python run_idealized.py configs/quadratic.yaml    # same file train.py takes

Solver-only knobs (grid resolution, std bounds, custom init, the annealed
mean-repulsion sweep) live in an optional extra `idealized:` section, which
`train.py` ignores -- so one file drives both scripts.

The older standalone schema (`mmd:` / `init:` / `log:` -- see
`configs/idealized_*.yaml`) is still accepted; it is selected automatically by
the presence of an `mmd:` section.

Two payoff backends, auto-selected (override with `idealized.backend`):

  closed_form -- exact Gaussian convolutions. Requires the game to expose the
                 `MultiPointGame` structure (`.peaks`, `._target_moments`, ...);
                 covers `multi_point` and `decoy_well`. Self-play only.
  quadrature  -- discretizes the action interval and integrates numerically.
                 Works for ANY 1-D box game (`quadratic`, `forsaken`,
                 `matching_pennies`, ...), at the cost of grid error, and is
                 the only backend that supports `train.mode: fixed_opponent`.

Example:
  python run_idealized.py configs/quadratic.yaml
  python run_idealized.py configs/idealized_two_point.yaml
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
    component_q as closed_form_component_q,
    expected_payoff as closed_form_expected_payoff,
    exploitability as closed_form_exploitability,
)
from training.run_config import RunConfig, run_config_from_dict


# --------------------------------------------------------------------------- config


@dataclasses.dataclass
class IdealizedSection:
    """Solver-only knobs: the `idealized:` section of a shared run config.

    None of these have a PPO counterpart, which is exactly why they are kept
    out of the shared schema -- `train.py` accepts (and ignores) this section
    so a single YAML file drives both scripts.
    """

    # --- payoff backend ---
    backend: str = "auto"                  # "auto" | "closed_form" | "quadrature"
    grid_points: int = 801                 # quadrature backend; also the grid the
                                           #   closed-form backend integrates its
                                           #   `gaussian_entropy: marginal` term on
    normalize_density: bool = True         # renormalize each mixture on the grid, so a
                                           #   component narrower than `dx` keeps its mass

    # --- std bounds. `std_max: null` mirrors `MixtureActorCritic`'s own ceiling,
    #     `log(high - low)`; `std_min` mirrors its `LOG_STD_MIN` floor. ---
    std_min: float = 1e-3
    std_max: float | None = None

    # --- graduated optimization: hold every component's std >= this early, then
    #     relax the floor to `std_min` over training. 0 disables. ---
    anneal_std_from: float = 0.0

    # --- annealed mean-repulsion sweep (see
    #     convergence_experiments/counterexample/COUNTEREXAMPLE.md section 4c):
    #     adds `coef * |mu_i - mu_j|` to each player's objective, ramped
    #     0 -> coef -> 0. 0 disables. ---
    repulsion_coef: float = 0.0
    repulsion_ramp: float = 0.2            # fraction of training spent ramping up
    repulsion_hold: float = 0.5            # fraction spent held at `repulsion_coef`

    # --- which heads train ---
    train_means: bool = True
    train_std: bool = True
    freeze_weights: bool = False           # pin the categorical head (isolates the Gaussian one)

    # --- how faithfully to imitate the sampled objective ---
    gaussian_entropy: str = "marginal"     # "marginal": exact differential entropy of the
                                           #   mixture, what `mixture_ppo_loss` estimates.
                                           #   "component": sum(log_std), the legacy term.
    kl_weighting: str = "policy"           # "policy": weight each component's Gaussian KL by
                                           #   its probability, as sampling does.
                                           #   "uniform": plain sum over components (legacy).

    # --- initial mixture. Defaults reproduce the trainer's own init exactly
    #     (`training.mixture._spread_bias_init` / `_std_bias_init`). ---
    init_means: Any = "spread"             # "spread" | list[float] | list[list[float]] (per player)
    init_weights: Any = None               # None (uniform) | list[float]
    init_log_std: Any = None               # None -> (high-low)/(2K), as the trainer does

    # --- logging ---
    log_rows: int = 16                     # rows in the end-of-run summary table
    log_out: str | None = None             # path for the history JSON (defaults to
                                           #   `<train.checkpoint_dir>/idealized_history.json`)
    verbose: bool = True                   # print one line per outer step, like train.py


@dataclasses.dataclass
class SolverConfig:
    """Everything `run` needs: the shared config's MMD hyperparameters, flattened,
    plus `IdealizedSection`. Built by `load_config` from either schema."""

    # --- loop shape ---
    chunks: tuple[int, ...] = ()   # iterations per logged chunk; sum = total iterations
    inner_steps: int = 1           # gradient steps per iteration (`ppo.ppo_epochs`)

    # --- core MMD ---
    lr: float = 1e-3               # eta (`optimizer.learning_rate`)
    num_components: int = 2        # K  (`network.num_components`)
    magnet_interval: int = 500
    target_tau: float = 0.001      # EMA of the mixture params, reported alongside

    # --- regularizers, per head (`ppo.*`) ---
    category_entropy_coef: float = 0.0
    gaussian_entropy_coef: float = 0.0
    trpo_category_kl_coef: float = 0.0
    trpo_gaussian_kl_coef: float = 0.0
    magnet_category_kl_coef: float = 0.0
    magnet_gaussian_kl_coef: float = 0.0

    # --- who trains (`train.*`) ---
    mode: str = "self_play"        # "self_play" | "fixed_opponent"
    perspective: int = 0
    opponent: str = "random"       # "random" | "static"

    idealized: IdealizedSection = dataclasses.field(default_factory=IdealizedSection)

    @property
    def total_iters(self) -> int:
        return sum(self.chunks)


def _build(cls: type, data: dict) -> Any:
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown field(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


# Fields of the shared schema with no idealized counterpart, and why. Printed at
# startup so a run is never silently ignoring half its config.
_UNUSED_SHARED_FIELDS = {
    ("network", "hidden_dims"): "no network -- the policy IS the mixture parameters",
    ("network", "activation"): "no network",
    ("network", "normalization"): "no network",
    ("network", "clip_means"): "the solver always projects means onto the box (`player_step`)",
    ("optimizer", "max_grad_norm"): "clips gradients w.r.t. network weights, which do not exist here",
    ("optimizer", "optimizer"): "the update is mirror/natural-gradient, not Adam",
    ("optimizer", "weight_decay"): "no network weights to decay",
    ("ppo", "clip_eps"): "no importance ratios without sampling",
    ("ppo", "value_coef"): "no critic -- the payoff integral is exact",
    ("ppo", "batch_size"): "the payoff integral is exact (this is the noise knob PPO needs)",
    ("train", "seed"): "the dynamics are deterministic",
}


def _shared_to_solver(cfg: RunConfig, idealized: IdealizedSection) -> SolverConfig:
    """Map a `train.py` run config onto the idealized solver.

    The correspondence is exact where it exists:
      `train.steps` x `train.epochs`  -> MMD iterations (one per PPO rollout),
                                         logged once per outer step, as train.py does
      `ppo.ppo_epochs`                -> gradient steps per iteration against the
                                         frozen "rollout" policy, which is what makes
                                         the `trpo_*_kl_coef` terms bite
      `optimizer.learning_rate`       -> eta, the mirror/natural step size
      `ppo.magnet_interval`           -> iterations between hard magnet snapshots
                                         (same units: `magnet_step` ticks once per rollout)
      `ppo.target_tau`                -> EMA over the mixture params; like train.py this
                                         only affects the *reported* target strategy
    """
    return SolverConfig(
        chunks=(cfg.train.epochs,) * cfg.train.steps,
        inner_steps=cfg.ppo.ppo_epochs,
        lr=cfg.optimizer.learning_rate,
        num_components=cfg.network.num_components,
        magnet_interval=cfg.ppo.magnet_interval,
        target_tau=cfg.ppo.target_tau,
        category_entropy_coef=cfg.ppo.category_entropy_coef,
        gaussian_entropy_coef=cfg.ppo.gaussian_entropy_coef,
        trpo_category_kl_coef=cfg.ppo.trpo_category_kl_coef,
        trpo_gaussian_kl_coef=cfg.ppo.trpo_gaussian_kl_coef,
        magnet_category_kl_coef=cfg.ppo.magnet_category_kl_coef,
        magnet_gaussian_kl_coef=cfg.ppo.magnet_gaussian_kl_coef,
        mode=cfg.train.mode,
        perspective=cfg.train.perspective,
        opponent=cfg.train.opponent,
        idealized=idealized,
    )


@dataclasses.dataclass
class _LegacyMMDSection:
    """The pre-shared-schema `mmd:` section, kept so `configs/idealized_*.yaml` still run."""

    lr: float = 0.05
    steps: int = 20000
    magnet_interval: int = 200
    magnet_coef: float = 0.2
    entropy_coef: float = 0.0
    num_components: int = 2
    train_means: bool = True
    train_std: bool = True
    anneal_std_from: float = 0.0
    std_min: float = 1e-3
    std_max: float = 1.0
    repulsion_coef: float = 0.0
    repulsion_ramp: float = 0.2
    repulsion_hold: float = 0.5
    freeze_weights: bool = False
    backend: str = "auto"
    grid_points: int = 801


@dataclasses.dataclass
class _LegacyInitSection:
    means: Any = "spread"
    weights: Any = None
    log_std: Any = None


@dataclasses.dataclass
class _LegacyLogSection:
    every: int = 500
    rows: int = 16
    out: str | None = None


def _legacy_to_solver(raw: dict) -> SolverConfig:
    mmd = _build(_LegacyMMDSection, raw.get("mmd", {}) or {})
    init = _build(_LegacyInitSection, raw.get("init", {}) or {})
    log = _build(_LegacyLogSection, raw.get("log", {}) or {})

    every = max(int(log.every), 1)
    chunks = [every] * (mmd.steps // every)
    if mmd.steps % every:
        chunks.append(mmd.steps % every)

    return SolverConfig(
        chunks=tuple(chunks),
        inner_steps=1,
        lr=mmd.lr,
        num_components=mmd.num_components,
        magnet_interval=mmd.magnet_interval,
        target_tau=0.0,
        # the legacy schema has one coefficient per *term*, shared by both heads
        category_entropy_coef=mmd.entropy_coef,
        gaussian_entropy_coef=mmd.entropy_coef,
        magnet_category_kl_coef=mmd.magnet_coef,
        magnet_gaussian_kl_coef=mmd.magnet_coef,
        mode="self_play",
        idealized=IdealizedSection(
            backend=mmd.backend,
            grid_points=mmd.grid_points,
            std_min=mmd.std_min,
            std_max=mmd.std_max,
            anneal_std_from=mmd.anneal_std_from,
            repulsion_coef=mmd.repulsion_coef,
            repulsion_ramp=mmd.repulsion_ramp,
            repulsion_hold=mmd.repulsion_hold,
            train_means=mmd.train_means,
            train_std=mmd.train_std,
            freeze_weights=mmd.freeze_weights,
            gaussian_entropy="component",   # legacy term: sum(log_std)
            kl_weighting="uniform",         # legacy term: plain sum over components
            init_means=init.means,
            init_weights=init.weights,
            init_log_std=init.log_std,
            log_rows=log.rows,
            log_out=log.out,
            verbose=False,                  # legacy runs print only the summary table
        ),
    )


def load_config(path: str | Path) -> tuple[Any, SolverConfig, RunConfig | None]:
    """`(game_config, solver_config, run_config_or_None)` from either schema.

    The shared (`train.py`) schema is used unless the file has an `mmd:` section,
    which selects the legacy standalone schema.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if "mmd" in raw:
        unknown = set(raw) - {"game", "mmd", "init", "log"}
        if unknown:
            raise ValueError(f"unknown top-level section(s) for the legacy schema: {sorted(unknown)}")
        game_raw = dict(raw.get("game", {}))
        name = game_raw.pop("name", None)
        if name is None:
            raise ValueError("config.game.name is required")
        if name not in GAME_CONFIGS:
            raise ValueError(f"unknown game {name!r}, choices: {sorted(GAME_CONFIGS)}")
        return _build(GAME_CONFIGS[name], game_raw), _legacy_to_solver(raw), None

    shared = dict(raw)
    idealized = _build(IdealizedSection, shared.pop("idealized", {}) or {})
    run_config = run_config_from_dict(shared)
    return run_config.game, _shared_to_solver(run_config, idealized), run_config


# --------------------------------------------------------------------------- backends
#
# A backend turns mixture `Params` into a *handle* -- whatever representation it
# integrates against -- and answers three questions about handles: expected payoff,
# per-component q-values, and exploitability. The quadrature backend's handle is a
# density on a grid, which is what lets a non-mixture opponent (uniform, or a point
# mass) be plugged in for `train.mode: fixed_opponent`.


def _density_grid(lo: float, hi: float, n: int, std_max: float):
    """The padded integration grid both backends integrate densities on.

    Padded by `4 * std_max` on each side so a wide component's tails are not
    truncated; the payoff is only ever evaluated (and best responses only ever
    taken) on the `in_box` part.
    """
    grid = jnp.linspace(lo - 4.0 * std_max, hi + 4.0 * std_max, n, dtype=jnp.float64)
    return grid, float(grid[1] - grid[0])


def _component_densities(p: Params, grid, dx: float, normalize: bool):
    """Per-component density on `grid`, each renormalized to unit mass if `normalize`
    -- otherwise a component narrower than `dx` silently loses its mass between grid
    points."""
    s = jnp.exp(p.log_std)
    comp = jnp.exp(-((grid[None, :] - p.means[:, None]) ** 2) / (2 * s[:, None] ** 2))
    comp = comp / (jnp.sqrt(2 * jnp.pi) * s[:, None])
    if normalize:
        comp = comp / (jnp.sum(comp, axis=-1, keepdims=True) * dx + 1e-300)
    return comp  # (K, n)


def _marginal_entropy(p: Params, grid, dx: float, normalize: bool):
    """Exact differential entropy of the mixture, `-int p log p` on `grid` -- what
    `mixture_ppo_loss` estimates with `-mixture_marginal_log_prob(...)`."""
    w = jax.nn.softmax(p.logits)
    d = jnp.sum(w[:, None] * _component_densities(p, grid, dx, normalize), axis=0)
    return -jnp.sum(d * jnp.log(d + 1e-300)) * dx


class ClosedFormBackend:
    """Exact Gaussian convolutions -- only for the `MultiPointGame` family."""

    name = "closed_form"
    supports_fixed_opponent = False

    def __init__(self, game: ZeroSumGame, grid, dx: float, normalize: bool):
        self.game = game
        # payoffs are closed-form, but a mixture's differential entropy is not, so
        # `entropy` still needs a grid -- see the note on `entropy` below.
        self._grid, self._dx, self._normalize = grid, dx, normalize

    def handle(self, p: Params) -> Params:
        return p

    def expected_payoff(self, h0, h1):
        return closed_form_expected_payoff(h0, h1, self.game)

    def component_q(self, p: Params, h_opp, sign: float):
        return closed_form_component_q(p, h_opp, self.game, sign)

    def exploitability(self, h0, h1):
        return closed_form_exploitability(h0, h1, self.game)

    def entropy(self, p: Params):
        """Exact differential entropy of the mixture, integrated on a grid.

        There is no closed form for it, but the per-component surrogate
        `sum(log_std)` is NOT a stand-in: it grows without bound in every
        component's std at full weight (gradient `1` per component, regardless of
        that component's probability, and with no reward for separating the
        components), so with a nonzero `gaussian_entropy_coef` it pins every std at
        `std_max` and the mixture can never localize onto narrow payoff peaks.
        Integrating costs one `(K, grid_points)` array per call, which is cheap
        next to the closed-form payoff this backend exists for.
        """
        return _marginal_entropy(p, self._grid, self._dx, self._normalize)


class QuadratureBackend:
    """Numeric integration on a 1-D grid. Works for any 1-D box game.

    The payoff matrix `R[i, j] = payoff(a_i, a_j)` is built once. Expected payoff is
    `d0^T R d1 * dx^2`; best responses are optimized over grid points inside the box
    (actions must be feasible), while the mixture densities are integrated over a
    padded grid so the Gaussian tails are not truncated.
    """

    name = "quadrature"
    supports_fixed_opponent = True

    def __init__(self, game: ZeroSumGame, n: int, std_max: float, normalize: bool):
        space = game.action_space(0)
        if not isinstance(space, BoxSpace) or space.shape != (1,):
            raise ValueError(
                "quadrature backend supports 1-D BoxSpace games only "
                f"(got {type(space).__name__} with shape {getattr(space, 'shape', None)}); "
                "set the game's `dim: 1`, or use `idealized.backend: closed_form`."
            )
        self.lo, self.hi = float(space.low[0]), float(space.high[0])
        self.grid, self.dx = _density_grid(self.lo, self.hi, n, std_max)
        self.in_box = (self.grid >= self.lo) & (self.grid <= self.hi)
        self.normalize = normalize

        a = self.grid[:, None]
        pay = jax.vmap(lambda x: jax.vmap(lambda y: game.payoff(x, y))(a))(a)
        self.R = jnp.asarray(pay, dtype=jnp.float64)  # (n, n)

    # -- handles -----------------------------------------------------------
    def handle(self, p: Params):
        w = jax.nn.softmax(p.logits)
        return jnp.sum(w[:, None] * self.component_densities(p), axis=0)  # (n,)

    def component_densities(self, p: Params):
        return _component_densities(p, self.grid, self.dx, self.normalize)  # (K, n)

    def uniform_handle(self):
        d = jnp.where(self.in_box, 1.0, 0.0)
        return d / (jnp.sum(d) * self.dx)

    def point_handle(self, x: float):
        idx = int(jnp.argmin(jnp.abs(self.grid - x)))
        return jnp.zeros_like(self.grid).at[idx].set(1.0 / self.dx)

    # -- questions ---------------------------------------------------------
    def expected_payoff(self, h0, h1):
        return (h0 @ self.R @ h1) * self.dx**2

    def component_q(self, p: Params, h_opp, sign: float):
        """Per-component expected utility `q_k = E_{a~N(mu_k, s_k)}[ this player's utility ]`."""
        u = sign * (self.R @ h_opp) * self.dx           # utility of a *pure* action a
        return (self.component_densities(p) @ u) * self.dx

    def exploitability(self, h0, h1):
        U = (h0 @ self.R @ h1) * self.dx**2
        v0 = (self.R @ h1) * self.dx                    # player 0's value of each pure action
        v1 = (h0 @ self.R) * self.dx                    # player 1's cost of each pure action
        br0 = jnp.max(jnp.where(self.in_box, v0, -jnp.inf))
        br1 = jnp.min(jnp.where(self.in_box, v1, jnp.inf))
        return (br0 - U) + (U - br1)

    def entropy(self, p: Params):
        """Exact differential entropy of the mixture -- what `mixture_ppo_loss`
        estimates with `-mixture_marginal_log_prob(...)` at the sampled action."""
        return _marginal_entropy(p, self.grid, self.dx, self.normalize)


def build_backend(game: ZeroSumGame, cfg: SolverConfig):
    supports_closed_form = hasattr(game, "peaks") and hasattr(game, "_target_moments")
    choice = cfg.idealized.backend
    if choice == "auto":
        choice = "closed_form" if supports_closed_form else "quadrature"
    if choice == "closed_form":
        if not supports_closed_form:
            raise ValueError(
                f"{type(game).__name__} has no `.peaks`/`._target_moments`, so the closed-form "
                "backend does not apply. Use `idealized.backend: quadrature`."
            )
        lo, hi = _bounds(game)
        grid, dx = _density_grid(lo, hi, cfg.idealized.grid_points, _std_max(game, cfg))
        return ClosedFormBackend(game, grid, dx, cfg.idealized.normalize_density)
    if choice == "quadrature":
        return QuadratureBackend(game, cfg.idealized.grid_points, _std_max(game, cfg),
                                 cfg.idealized.normalize_density)
    raise ValueError(f"unknown idealized.backend {cfg.idealized.backend!r}")


# --------------------------------------------------------------------------- init


def _bounds(game: ZeroSumGame) -> tuple[float, float]:
    """`(low, high)` of the 1-D action box, or a clear error.

    Both backends parameterize the policy as a Gaussian mixture over a scalar
    action, so a game whose players act on a simplex (`blotto`) has no idealized
    counterpart -- only `train.py` can run it.
    """
    space = game.action_space(0)
    if not isinstance(space, BoxSpace):
        raise ValueError(
            f"{type(game).__name__} gives players a {type(space).__name__}; the idealized "
            "solver only handles 1-D box action spaces. Run this game with train.py."
        )
    if space.shape != (1,):
        raise ValueError(
            f"{type(game).__name__}'s action space has shape {space.shape}; the idealized "
            "solver is 1-D. Set the game's `dim: 1`, or use idealized_mmd_multidim.py."
        )
    return float(space.low[0]), float(space.high[0])


def _std_max(game: ZeroSumGame, cfg: SolverConfig) -> float:
    """`idealized.std_max`, defaulting to `MixtureActorCritic`'s own ceiling `high - low`."""
    if cfg.idealized.std_max is not None:
        return float(cfg.idealized.std_max)
    lo, hi = _bounds(game)
    return hi - lo


def build_init(game: ZeroSumGame, cfg: SolverConfig) -> tuple[Params, Params]:
    """The trainer's own init by default: means spread over the box, uniform weights,
    `std = (high - low) / (2K)` -- see `training.mixture._spread_bias_init` /
    `_std_bias_init`, and note that `logits_head` outputs zeros at a zero observation."""
    lo, hi = _bounds(game)
    k = cfg.num_components
    init = cfg.idealized

    def means_for(player: int) -> jnp.ndarray:
        m = init.init_means
        if isinstance(m, str):
            if m != "spread":
                raise ValueError(f"unknown idealized.init_means {m!r}")
            frac = (jnp.arange(k, dtype=jnp.float64) + 0.5) / k
            return lo + frac * (hi - lo)
        arr = jnp.asarray(m, dtype=jnp.float64)
        if arr.ndim == 2:  # per-player means
            arr = arr[player]
        if arr.shape[0] != k:
            raise ValueError(f"idealized.init_means has {arr.shape[0]} entries, expected num_components={k}")
        return arr

    if init.init_log_std is None:
        log_std_val = float(np.log((hi - lo) / (2 * k)))
    else:
        log_std_val = float(np.log(init.init_log_std)) if init.init_log_std > 0 else float(init.init_log_std)

    if init.init_weights is None:
        logits = jnp.zeros(k, dtype=jnp.float64)
    else:
        w = jnp.asarray(init.init_weights, dtype=jnp.float64)
        if w.shape[0] != k:
            raise ValueError(f"idealized.init_weights has {w.shape[0]} entries, expected num_components={k}")
        logits = jnp.log(w)

    log_std = jnp.full((k,), log_std_val, dtype=jnp.float64)
    log_std = jnp.clip(log_std, jnp.log(init.std_min), jnp.log(_std_max(game, cfg)))
    return (Params(logits=logits, means=means_for(0), log_std=log_std),
            Params(logits=logits, means=means_for(1), log_std=log_std))


# --------------------------------------------------------------------------- update


def categorical_mirror_update(logits, q, magnet_logits, old_logits, eta, tau, tau_ent, tau_trpo):
    """Closed-form MMD simplex update -> new logits.

    Generalizes `idealized_mmd.categorical_mirror_update` with the PPO-side
    trust-region term. It solves, exactly,

      argmax_pi  <pi, q> - tau*KL(pi||magnet) - tau_ent*KL(pi||uniform)
                         - tau_trpo*KL(pi||pi_old) - (1/eta)*KL(pi||pi_t)

    whose solution is

      log pi ~ [eta*q + eta*tau*log rho + eta*tau_trpo*log pi_old + log pi_t]
               / (1 + eta*tau + eta*tau_ent + eta*tau_trpo).

    Note the direction: the trust-region term here is `KL(new || old)`, the mirror
    -descent direction, while `mixture_ppo_loss` penalizes `KL(old || new)`. The two
    agree to second order, and only the former has a closed-form argmax. `tau_trpo`
    only bites when `ppo.ppo_epochs > 1`, since `pi_old == pi_t` on the first
    gradient step of an iteration.
    """
    lp = jax.nn.log_softmax(logits)
    lm = jax.nn.log_softmax(magnet_logits)
    lo = jax.nn.log_softmax(old_logits)
    num = eta * q + eta * tau * lm + eta * tau_trpo * lo + lp  # uniform log-prob is constant
    return num / (1.0 + eta * tau + eta * tau_ent + eta * tau_trpo)


def gaussian_kl_per_component(m_p, ls_p, m_q, ls_q):
    """`KL(N(m_p, s_p) || N(m_q, s_q))` per component, shape (K,)."""
    vp, vq = jnp.exp(2 * ls_p), jnp.exp(2 * ls_q)
    return ls_q - ls_p + (vp + (m_p - m_q) ** 2) / (2 * vq) - 0.5


def repulsion_coef_at(cfg: SolverConfig, frac):
    """Ramp 0 -> coef over `repulsion_ramp`, hold for `repulsion_hold`, then anneal to 0."""
    i = cfg.idealized
    if i.repulsion_coef == 0.0:
        return jnp.zeros((), dtype=jnp.float64)
    ramp, hold = i.repulsion_ramp, i.repulsion_hold
    tail = max(1.0 - ramp - hold, 1e-9)
    up = i.repulsion_coef * (frac / max(ramp, 1e-9))
    down = i.repulsion_coef * jnp.maximum(0.0, (1.0 - frac) / tail)
    return jnp.where(frac < ramp, up, jnp.where(frac < ramp + hold, i.repulsion_coef, down))


def std_floor_at(cfg: SolverConfig, frac):
    """Anneal the log-std floor from `anneal_std_from` down to `std_min`."""
    base = float(np.log(cfg.idealized.std_min))
    if cfg.idealized.anneal_std_from <= 0.0:
        return jnp.full((), base, dtype=jnp.float64)
    return (1.0 - frac) * float(np.log(cfg.idealized.anneal_std_from)) + frac * base


# --------------------------------------------------------------------------- run


def _tree_where(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


def _ema(new, old, tau: float):
    if tau <= 0.0:
        return new
    return jax.tree_util.tree_map(lambda n, o: tau * n + (1.0 - tau) * o, new, old)


def build_iteration(game: ZeroSumGame, cfg: SolverConfig, backend):
    """One MMD iteration = `cfg.inner_steps` gradient steps against a frozen policy.

    Mirrors `train.py`'s `_build_train_step`: one rollout (here: one exact payoff
    integral) defines the "old" policy, `ppo.ppo_epochs` gradient steps are taken
    against it, then the target EMA and the magnet snapshot tick once.
    """
    lo, hi = _bounds(game)
    log_std_hi = float(np.log(_std_max(game, cfg)))
    i = cfg.idealized
    total = max(cfg.total_iters - 1, 1)

    if cfg.magnet_interval < 1:
        raise ValueError("magnet_interval must be >= 1")

    trains = {0: True, 1: True}
    if cfg.mode == "fixed_opponent":
        trains[1 - cfg.perspective] = False

    if i.gaussian_entropy == "marginal":
        entropy_term = backend.entropy           # exact mixture differential entropy
    elif i.gaussian_entropy == "component":
        entropy_term = lambda pp: jnp.sum(pp.log_std)  # noqa: E731 -- legacy per-component term
    else:
        raise ValueError(f"unknown idealized.gaussian_entropy {i.gaussian_entropy!r}")

    def player_step(p: Params, h_opp, old: Params, magnet: Params, sign: float, lam, floor):
        # --- categorical head: exact mirror step on the simplex ---
        if i.freeze_weights:
            logits = p.logits
        else:
            q = backend.component_q(p, h_opp, sign)
            logits = categorical_mirror_update(
                p.logits, q, magnet.logits, old.logits,
                cfg.lr, cfg.magnet_category_kl_coef,
                cfg.category_entropy_coef, cfg.trpo_category_kl_coef,
            )

        # --- Gaussian head: natural-gradient step on the same objective ---
        if i.kl_weighting == "policy":
            # weight each component's KL by how often it is sampled, as PPO does
            kl_w = jax.lax.stop_gradient(jax.nn.softmax(old.logits))
        elif i.kl_weighting == "uniform":
            kl_w = jnp.ones_like(p.logits)
        else:
            raise ValueError(f"unknown idealized.kl_weighting {i.kl_weighting!r}")

        def obj(pp: Params):
            h = backend.handle(pp)
            pay = (backend.expected_payoff(h, h_opp) if sign > 0
                   else -backend.expected_payoff(h_opp, h))
            ent = cfg.gaussian_entropy_coef * entropy_term(pp)
            mag = cfg.magnet_gaussian_kl_coef * jnp.sum(
                kl_w * gaussian_kl_per_component(pp.means, pp.log_std, magnet.means, magnet.log_std)
            )
            # `mixture_ppo_loss` penalizes KL(old || new) for the Gaussian head; kept
            # in that direction here since a gradient step needs no closed-form argmax.
            trpo = cfg.trpo_gaussian_kl_coef * jnp.sum(
                kl_w * gaussian_kl_per_component(old.means, old.log_std, pp.means, pp.log_std)
            )
            rep = lam * jnp.sum(jnp.abs(pp.means[:, None] - pp.means[None, :])) / 2.0
            return pay + ent + rep - mag - trpo

        g = jax.grad(obj)(p)
        # Fisher metric of N(mu, s) in (mu, rho = log s): I_mu = 1/s^2, I_rho = 2.
        means = p.means + (cfg.lr * jnp.exp(2 * p.log_std) * g.means if i.train_means else 0.0)
        log_std = p.log_std + (cfg.lr * 0.5 * g.log_std if i.train_std else 0.0)
        return Params(logits=logits,
                      means=jnp.clip(means, lo, hi),
                      log_std=jnp.clip(log_std, floor, log_std_hi))

    def iteration(carry, _):
        p0, p1, m0, m1, e0, e1, fixed_handle, it = carry
        frac = it.astype(jnp.float64) / total
        lam, floor = repulsion_coef_at(cfg, frac), std_floor_at(cfg, frac)
        old0, old1 = p0, p1

        def grad_step(pair, _):
            a, b = pair
            # a non-training player is not a mixture at all (it may be uniform or a
            # point mass), so its handle is the fixed density built in `run`.
            h0 = backend.handle(a) if trains[0] else fixed_handle
            h1 = backend.handle(b) if trains[1] else fixed_handle
            na = player_step(a, h1, old0, m0, +1.0, lam, floor) if trains[0] else a
            nb = player_step(b, h0, old1, m1, -1.0, lam, floor) if trains[1] else b
            return (na, nb), None

        (p0, p1), _ = jax.lax.scan(grad_step, (p0, p1), None, length=cfg.inner_steps)

        it = it + 1
        snapshot = (it % cfg.magnet_interval) == 0
        m0, m1 = _tree_where(snapshot, p0, m0), _tree_where(snapshot, p1, m1)
        e0, e1 = _ema(p0, e0, cfg.target_tau), _ema(p1, e1, cfg.target_tau)
        return (p0, p1, m0, m1, e0, e1, fixed_handle, it), None

    return iteration


def run(game: ZeroSumGame, cfg: SolverConfig, p0: Params, p1: Params):
    backend = build_backend(game, cfg)

    fixed_handle = jnp.zeros_like(backend.handle(p0)) if backend.name == "quadrature" else None
    if cfg.mode == "fixed_opponent":
        if not backend.supports_fixed_opponent:
            raise ValueError(
                f"train.mode: fixed_opponent needs the quadrature backend "
                f"(the {backend.name} backend can only integrate two Gaussian mixtures)."
            )
        if cfg.opponent == "random":
            fixed_handle = backend.uniform_handle()
        elif cfg.opponent == "static":
            fixed_handle = backend.point_handle(0.5 * (backend.lo + backend.hi))
        else:
            raise ValueError(f"unknown train.opponent {cfg.opponent!r}")
    elif cfg.mode != "self_play":
        raise ValueError(f"unknown train.mode {cfg.mode!r}")

    iteration = build_iteration(game, cfg, backend)
    chunk_fns: dict[int, Any] = {}

    def handles_of(a: Params, b: Params):
        h0 = fixed_handle if (cfg.mode == "fixed_opponent" and cfg.perspective == 1) else backend.handle(a)
        h1 = fixed_handle if (cfg.mode == "fixed_opponent" and cfg.perspective == 0) else backend.handle(b)
        return h0, h1

    # Not jitted: `idealized_mmd.exploitability` builds its best-response grid with
    # Python floats off the game object, so it cannot be traced. It runs once per
    # logged chunk, so eager execution costs nothing.
    def expl_fn(a, b):
        return backend.exploitability(*handles_of(a, b))

    def record(t: int, a: Params, b: Params, ea: Params, eb: Params) -> dict:
        return {
            "t": t,
            "expl": float(expl_fn(a, b)),
            "target_expl": float(expl_fn(ea, eb)),
            "w0": [float(x) for x in jax.nn.softmax(a.logits)],
            "means0": [float(x) for x in a.means],
            "std0": [float(x) for x in jnp.exp(a.log_std)],
            "w1": [float(x) for x in jax.nn.softmax(b.logits)],
            "means1": [float(x) for x in b.means],
            "std1": [float(x) for x in jnp.exp(b.log_std)],
        }

    carry = (p0, p1, p0, p1, p0, p1, fixed_handle, jnp.zeros((), dtype=jnp.int64))
    history = [record(0, p0, p1, p0, p1)]
    done = 0
    for step, length in enumerate(cfg.chunks, start=1):
        if length not in chunk_fns:
            chunk_fns[length] = jax.jit(
                lambda c, n=length: jax.lax.scan(iteration, c, None, length=n)[0]
            )
        carry = chunk_fns[length](carry)
        done += length
        entry = record(done, carry[0], carry[1], carry[4], carry[5])
        history.append(entry)
        if cfg.idealized.verbose:
            print(f"  step {step:4d}/{len(cfg.chunks)} | iteration {done:7d} "
                  f"| expl {entry['expl']:+.4f} | target expl {entry['target_expl']:+.4f}")
    return carry[0], carry[1], history, backend


# --------------------------------------------------------------------------- cli


def _fmt(w, mu, sd) -> str:
    return "[" + " ".join(f"{a:.2f}@{b:+.2f}(sd{c:.3f})" for a, b, c in zip(w, mu, sd)) + "]"


def _report_unused(run_config: RunConfig) -> None:
    print("ignored (no idealized counterpart):")
    for (section, field), why in _UNUSED_SHARED_FIELDS.items():
        value = getattr(getattr(run_config, section), field)
        print(f"  {section}.{field} = {value!r}  -- {why}")
    print("  ppo advantage normalization  -- PPO rescales its gradient by the batch's own "
          "advantage std; the exact q-values are used unscaled here")


def _warn_grid(backend, history: list[dict]) -> None:
    """Warn if any component actually got narrower than the quadrature grid can see.

    Checked against the stds the run *reached*, not the configured floor: a run whose
    components stay wide is unaffected by a coarse grid, and only the ones that
    collapse turn their payoff integral into a point mass at the nearest node.
    """
    if backend.name != "quadrature":
        return
    reached = min(min(h["std0"] + h["std1"]) for h in history)
    if reached >= 2 * backend.dx:
        return
    span = float(backend.grid[-1] - backend.grid[0])
    needed = int(span / (reached / 3.0)) + 1
    print(f"WARNING: a component reached std={reached:.4g}, below the grid spacing "
          f"dx={backend.dx:.4g}. A component that narrow falls between grid points, so its "
          f"payoff integral is only as accurate as a point mass at the nearest node --\n"
          f"         treat the tail of this run as unresolved. Raise `idealized.grid_points` "
          f"to ~{needed}, raise `idealized.std_min`, or lower `idealized.std_max` "
          f"(which sets the grid padding).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to a YAML config (train.py's, or a legacy configs/idealized_*.yaml)")
    args = ap.parse_args()

    game_config, cfg, run_config = load_config(args.config)
    game = game_config.build()
    p0, p1 = build_init(game, cfg)

    print(f"config  : {args.config}  ({'shared with train.py' if run_config else 'legacy standalone'})")
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"solver  : {dataclasses.asdict(cfg)}")
    if run_config is not None:
        _report_unused(run_config)
    print()

    p0f, p1f, history, backend = run(game, cfg, p0, p1)
    _warn_grid(backend, history)
    print(f"\nbackend : {backend.name}  |  mode: {cfg.mode}"
          f"{'' if cfg.mode == 'self_play' else f' (player {cfg.perspective} vs {cfg.opponent})'}"
          f"  |  {cfg.total_iters} iterations x {cfg.inner_steps} gradient step(s)\n")

    n = len(history)
    idx = sorted({round(i * (n - 1) / max(cfg.idealized.log_rows - 1, 1))
                  for i in range(cfg.idealized.log_rows)})
    frozen = (1 - cfg.perspective) if cfg.mode == "fixed_opponent" else None
    for i in idx:
        e = history[i]
        cols = [_fmt(e["w0"], e["means0"], e["std0"]), _fmt(e["w1"], e["means1"], e["std1"])]
        if frozen is not None:  # not a mixture -- its params never moved and mean nothing
            cols[frozen] = f"fixed({cfg.opponent})"
        print(f"  t={e['t']:7d}  expl={e['expl']:+8.4f}   P0 {cols[0]}   P1 {cols[1]}")

    tail = float(np.mean([h["expl"] for h in history[int(n * 0.7):]]))
    best = min(h["expl"] for h in history)
    print(f"\nfinal exploitability {history[-1]['expl']:+.4f} | "
          f"tail(30%) {tail:+.4f} | best {best:+.4f}")

    out = cfg.idealized.log_out
    if out is None and run_config is not None and run_config.train.checkpoint_dir is not None:
        out = str(Path(run_config.train.checkpoint_dir) / "idealized_history.json")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(history, indent=2))
        print(f"saved history -> {out}")


if __name__ == "__main__":
    main()
