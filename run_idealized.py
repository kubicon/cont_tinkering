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
`(logits, means, scale_tril, value)` -- so the two scripts optimize the *same*
strategy space and a difference between their runs is attributable to
sampling noise and the optimizer/parameterization, nothing else.

  python run_idealized.py configs/quadratic.yaml    # same file train.py takes

Solver-only knobs (grid resolution, std bounds, custom init, the annealed
mean-repulsion sweep) live in an optional extra `idealized:` section, which
`train.py` ignores -- so one file drives both scripts.

The older standalone schema (`mmd:` / `init:` / `log:` -- see
`configs/idealized_*.yaml`) is still accepted; it is selected automatically by
the presence of an `mmd:` section.

The policy is a K-component multivariate Gaussian mixture over a `d`-dimensional
box, `d` read off the game -- means `(K, d)` and a Cholesky factor `(K, d, d)` of
each component's covariance, the same parametrization `train.py`'s policy uses
(`training/gaussian.py`). `idealized.full_covariance` picks a full factor over a
diagonal one, inheriting `network.full_covariance` when a shared config sets it. The
Gaussian head's update is the exact natural gradient in those coordinates
(`gaussian.natural_gradient`), which on a diagonal factor is the
`(sigma^2 grad_mu, 1/2 grad_log_sigma)` preconditioner this solver has always used.

Four payoff backends; the first three are auto-selected (override with
`idealized.backend`):

  closed_form  -- exact Gaussian convolutions, 1-D. Requires the game to expose the
                  `MultiPointGame` structure (`.peaks`, `._target_moments`, ...);
                  covers `multi_point` and `decoy_well`. Self-play only.
  closed_form_multidim
               -- the same, lifted to the separable `multidim_decoy_well` game
                  (`idealized_mmd_multidim.py`): per-axis convolutions summed over
                  coordinates, best responses by `d` independent 1-D grid searches.
                  Self-play only.
  quadrature   -- discretizes the action box and integrates numerically. Works for
                  ANY box game (`quadratic`, `forsaken`, `matching_pennies`, ...),
                  at the cost of grid error -- and, in `d > 1`, of a payoff matrix
                  with `grid_points^(2d)` entries, so it is practical only on a
                  coarse grid in 2-D and refuses outright when the matrix will not
                  fit (`idealized.max_quadrature_gib`). Prefer `sampled` above 1-D.
  sampled      -- the middle ground with `train.py`: the tabular mirror step is
                  kept, but the payoff integral is replaced by `ppo.batch_size`
                  Monte-Carlo action draws per iteration, exactly as a rollout
                  does. Its cost is independent of `d`, which makes it the backend
                  for a genuinely multidimensional game. Never auto-selected -- ask
                  for it with `idealized.backend: sampled`. `idealized.samples ->
                  inf` recovers the quadrature run, which makes the sample count a
                  dial between the two scripts with nothing else changing.

Example:
  python run_idealized.py configs/quadratic.yaml
  python run_idealized.py configs/idealized_two_point.yaml
  python run_idealized.py configs/multidim/exp4_2d_2peak_decoy.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from games.base import ZeroSumGame
from games.configs import GAME_CONFIGS
from games.examples import MultiDimDecoyWellGame
from games.sequential import SequentialZeroSumGame
from games.spaces import BoxSpace
from idealized_mmd import (
    Params as Params1D,
    component_q as closed_form_component_q,
    expected_payoff as closed_form_expected_payoff,
    exploitability as closed_form_exploitability,
)
import idealized_mmd_multidim as multidim
from training.gaussian import (
    clamp_scale_tril,
    diagonal_slots,
    gaussian_kl,
    gaussian_log_prob,
    gaussian_sample,
    log_scale_det,
    marginal_std,
    natural_gradient,
    pack_scale_tril,
    scale_param_size,
    tril_positions,
)
from training.run_config import RunConfig, run_config_from_dict


class Params(NamedTuple):
    """One player's mixture policy -- the solver's entire parameter vector.

    The same shape as `training.mixture.MixtureActorCritic`'s four heads minus the
    critic: a categorical over `K` components, each a multivariate Gaussian given by
    its mean and a lower-triangular Cholesky factor `A` of its covariance
    (`Sigma = A A^T`). `d == 1` is the scalar-action case the solver started life as,
    where `A` is the `(1, 1)` matrix holding the standard deviation.

    The factor, rather than a `log`-standard-deviation vector, is what makes the
    parametrization work in `d > 1`: see `training/gaussian.py` for why it keeps the
    KL regularizer uniformly strongly convex, and `natural_gradient` for the step
    this solver takes in these coordinates.
    """

    logits: jnp.ndarray      # (K,)       categorical logits over the K components
    means: jnp.ndarray       # (K, d)     each component's Gaussian mean
    scale_tril: jnp.ndarray  # (K, d, d)  each component's Cholesky covariance factor


def _to_1d(p: Params) -> Params1D:
    """View a `d == 1` policy as `idealized_mmd`'s scalar one, differentiably.

    The 1-D closed-form backend predates the factor parametrization and speaks
    `(means, log_std)`; this is the only place the two meet. Gradients flow back
    through `log` into `scale_tril` unchanged, so the caller never sees the seam.
    """
    return Params1D(logits=p.logits,
                    means=p.means[:, 0],
                    log_std=jnp.log(marginal_std(p.scale_tril))[:, 0])


# --------------------------------------------------------------------------- config


@dataclasses.dataclass
class IdealizedSection:
    """Solver-only knobs: the `idealized:` section of a shared run config.

    None of these have a PPO counterpart, which is exactly why they are kept
    out of the shared schema -- `train.py` accepts (and ignores) this section
    so a single YAML file drives both scripts.
    """

    # --- payoff backend ---
    backend: str = "auto"                  # "auto" | "closed_form" | "closed_form_multidim"
                                           #   | "quadrature" | "sampled"
    grid_points: int = 801                 # quadrature backend, *per axis*; also the grid
                                           #   the closed-form backends integrate their
                                           #   `gaussian_entropy: marginal` term on. The
                                           #   quadrature grid has `grid_points^d` nodes and
                                           #   its payoff matrix `grid_points^(2d)` entries,
                                           #   so this must come down sharply above 1-D.
    max_quadrature_gib: float = 2.0        # refuse to build a payoff matrix larger than
                                           #   this, rather than dying in the allocator
    normalize_density: bool = True         # renormalize each mixture on the grid, so a
                                           #   component narrower than `dx` keeps its mass

    # --- covariance shape. Mirrors `network.full_covariance`: a full lower-triangular
    #     Cholesky factor per component instead of a diagonal one. Inert on a game
    #     whose payoff separates over a player's own coordinates (every game in
    #     `games/examples.py` except `curvature_pump` / `asymmetric_well`): such a
    #     payoff's gradient on an off-diagonal entry vanishes wherever that entry is
    #     zero, and the init is uncorrelated, so the extra parameters never move.
    #     `null` (the default) inherits `network.full_covariance`, so one field in a
    #     shared config sets the covariance shape for both scripts at once. ---
    full_covariance: bool | None = None

    # --- `backend: sampled` only. The payoff integral becomes an average over
    #     `samples` joint action draws, redrawn once per iteration (and reused
    #     across the `ppo_epochs` gradient steps, as a PPO batch is). ---
    samples: int | None = None             # draws per iteration; None -> `ppo.batch_size`
    sample_seed: int | None = None         # None -> `train.seed`
    q_estimator: str = "responsibility"    # how the per-component q_k are estimated:
                                           #   "responsibility": self-normalized importance
                                           #     weights w_k N(a|k)/p_mix(a) over the on-policy
                                           #     batch -- every draw informs every component,
                                           #     and the batch costs exactly `samples` payoffs
                                           #   "per_component": draw a separate batch from each
                                           #     component. Unbiased and low-variance, but costs
                                           #     K x `samples` payoffs and is off-policy.
                                           #   "onpolicy": average over the draws that actually
                                           #     picked k, which is what PPO's advantage sees.
    grad_estimator: str = "pathwise"       # Gaussian-head gradient: "pathwise" reparameterizes
                                           #   (needs a payoff differentiable in the action, and
                                           #   is far lower variance); "score" is REINFORCE with
                                           #   a batch-mean baseline, as PPO's surrogate is.
    entropy_source: str = "sampled"        # "sampled": -log p_mix at the sampled action, exactly
                                           #   what `mixture_ppo_loss` adds. "exact": the grid
                                           #   integral. See `SampledBackend.entropy`.

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
    init_means: Any = "spread"             # "spread" (component k at (k+0.5)/K of the box in
                                           #   every axis -- the diagonal) | a nested list
                                           #   shaped (K,), (K, d), or (2, K, d) per player
    init_weights: Any = None               # None (uniform) | list[float]
    init_log_std: Any = None               # None -> (high-low)/(2K), as the trainer does.
                                           #   A *standard deviation* despite the name, put
                                           #   on the factor's diagonal (the init is always
                                           #   uncorrelated, `full_covariance` or not)

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
    batch_size: int = 256          # sampled backend only (`ppo.batch_size`)
    seed: int = 0                  # sampled backend only (`train.seed`)
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
    ("network", "mean_box_penalty_coef"): "nothing to pull back -- the solver's projection is exact, "
                                          "not a straight-through one",
    ("optimizer", "max_grad_norm"): "clips gradients w.r.t. network weights, which do not exist here",
    ("optimizer", "optimizer"): "the update is mirror/natural-gradient, not Adam",
    ("optimizer", "weight_decay"): "no network weights to decay",
    ("ppo", "clip_eps"): "the update is a mirror step at the current policy, not a "
                         "ratio-clipped surrogate",
    ("ppo", "value_coef"): "no critic -- the baseline is the exact payoff, or (sampled "
                           "backend) the batch mean",
    ("ppo", "batch_size"): "the payoff integral is exact (this is the noise knob PPO needs)",
    ("train", "seed"): "the dynamics are deterministic",
}

# ... except under `backend: sampled`, which is exactly the backend that needs them.
_SAMPLED_SHARED_FIELDS = {
    ("ppo", "batch_size"): "action draws per iteration (unless `idealized.samples` overrides it)",
    ("train", "seed"): "seeds those draws (unless `idealized.sample_seed` overrides it)",
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
        batch_size=cfg.ppo.batch_size,
        seed=cfg.train.seed,
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
        idealized=dataclasses.replace(
            idealized,
            full_covariance=(cfg.network.full_covariance if idealized.full_covariance is None
                             else idealized.full_covariance),
        ),
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
    full_covariance: bool = False
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
            full_covariance=mmd.full_covariance,
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
#
# Every method also takes the iteration's `noise` and the acting `player`, which only
# a stochastic backend (`SampledBackend`, whose handle is a batch of sampled actions)
# reads: the deterministic ones ignore both. `stochastic` says which kind it is, and
# a stochastic backend carries a deterministic `.metrics` backend that the reported
# exploitability is computed with -- a metric as noisy as the thing it measures would
# say nothing.


def _density_grid(lo: np.ndarray, hi: np.ndarray, n: int, std_max: np.ndarray):
    """The padded integration grid the grid backends integrate densities on.

    Returns `(nodes, dv, per_axis)`: `nodes` is `(n**d, d)`, the cartesian product of
    `d` per-axis grids of `n` points each, in C order; `dv` is the volume element
    `prod_j dx_j`; `per_axis` is the list of 1-D grids, which the exploitability
    search and the grid-resolution warning still want separately.

    Padded by `4 * std_max` on each side so a wide component's tails are not
    truncated; the payoff is only ever evaluated (and best responses only ever
    taken) on the `in_box` part. Note the node count is `n**d`: this is the grid
    backends' whole scaling problem, and the reason `sampled` exists.
    """
    axes = [jnp.linspace(lo[j] - 4.0 * std_max[j], hi[j] + 4.0 * std_max[j], n, dtype=jnp.float64)
            for j in range(len(lo))]
    dv = float(np.prod([float(a[1] - a[0]) for a in axes]))
    nodes = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))
    return nodes, dv, axes


def _log_prob_at(nodes, mean, scale_tril):
    """`log N(node | mean, A A^T)` for a batch of nodes against ONE component -> `(M,)`.

    `gaussian.gaussian_log_prob` broadcasts its leading axes elementwise (its
    triangular solve needs the action and the factor to agree on them), so a batch of
    actions against a single component is a `vmap`, not a broadcast.
    """
    return jax.vmap(gaussian_log_prob, in_axes=(0, None, None))(nodes, mean, scale_tril)


def _component_densities(p: Params, nodes, dv: float, normalize: bool):
    """Per-component density at every grid node, `(K, n**d)`.

    Each renormalized to unit mass if `normalize` -- otherwise a component narrower
    than the grid spacing silently loses its mass between nodes. In `d > 1` the
    density is the full multivariate one, so a correlated component is represented
    exactly rather than by its marginals.
    """
    log_p = jax.vmap(_log_prob_at, in_axes=(None, 0, 0))(nodes, p.means, p.scale_tril)
    comp = jnp.exp(log_p)                                        # (K, n**d)
    if normalize:
        comp = comp / (jnp.sum(comp, axis=-1, keepdims=True) * dv + 1e-300)
    return comp


def _marginal_entropy(p: Params, nodes, dv: float, normalize: bool):
    """Exact differential entropy of the mixture, `-int p log p` on the grid -- what
    `mixture_ppo_loss` estimates with `-mixture_marginal_log_prob(...)`."""
    w = jax.nn.softmax(p.logits)
    d = jnp.sum(w[:, None] * _component_densities(p, nodes, dv, normalize), axis=0)
    return -jnp.sum(d * jnp.log(d + 1e-300)) * dv


class ClosedFormBackend:
    """Exact Gaussian convolutions -- 1-D, and only for the `MultiPointGame` family.

    The `handle` a payoff question is asked about is the policy itself: this backend
    integrates two mixtures against each other analytically, with no grid in between
    (the grid it does hold is only for the entropy term below).
    """

    name = "closed_form"
    supports_fixed_opponent = False
    stochastic = False

    def __init__(self, game: ZeroSumGame, nodes, dv: float, normalize: bool):
        self.game = game
        # payoffs are closed-form, but a mixture's differential entropy is not, so
        # `entropy` still needs a grid -- see the note on `entropy` below.
        self._nodes, self._dv, self._normalize = nodes, dv, normalize

    def handle(self, p: Params, noise=None, player: int = 0) -> Params:
        return p

    def expected_payoff(self, h0, h1):
        return closed_form_expected_payoff(_to_1d(h0), _to_1d(h1), self.game)

    def component_q(self, p: Params, h_opp, sign: float, noise=None):
        return closed_form_component_q(_to_1d(p), _to_1d(h_opp), self.game, sign)

    def exploitability(self, h0, h1):
        return closed_form_exploitability(_to_1d(h0), _to_1d(h1), self.game)

    def entropy(self, p: Params, noise=None, player: int = 0):
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
        return _marginal_entropy(p, self._nodes, self._dv, self._normalize)


class ClosedFormMultiDimBackend:
    """`ClosedFormBackend` lifted to the separable `multidim_decoy_well` game.

    Same contract, same exactness, `d` dimensions: the integrals live in
    `idealized_mmd_multidim.py` and reduce to per-axis Gaussian convolutions summed
    over coordinates, because the game's payoff is a sum of per-coordinate terms.
    The exploitability's best responses separate the same way, into `d` independent
    1-D grid searches -- which is what keeps this backend free of the `grid^d`
    blow-up the quadrature one suffers.

    That separability is also the backend's one caveat: only the *marginal* variances
    enter, and an off-diagonal entry `A_ij` reaches those only through its own square
    (`Sigma_ii = sum_j A_ij^2`), so the payoff's gradient on it *vanishes at zero*.
    Since `build_init` always starts uncorrelated, `idealized.full_covariance` is
    inert here by construction -- nothing in the objective can push a component off
    the diagonal, the magnet KL least of all (it pulls back toward the snapshot's
    correlations, which are zero too). The flag is not useless in general, just on
    this game; a payoff with genuine cross-coordinate structure (`curvature_pump`,
    `asymmetric_well`) needs the `sampled` backend, whose expectations this module
    does not have to know.
    """

    name = "closed_form_multidim"
    supports_fixed_opponent = False
    stochastic = False

    def __init__(self, game: MultiDimDecoyWellGame, nodes, dv: float, normalize: bool):
        self.game = game
        self.geom = multidim.geometry(game)
        self._nodes, self._dv, self._normalize = nodes, dv, normalize

    def handle(self, p: Params, noise=None, player: int = 0) -> Params:
        return p

    def expected_payoff(self, h0, h1):
        return multidim.expected_payoff(h0, h1, self.geom)

    def component_q(self, p: Params, h_opp, sign: float, noise=None):
        return multidim.component_q(p, h_opp, self.geom, sign)

    def exploitability(self, h0, h1):
        return multidim.exploitability(h0, h1, self.game, self.geom)

    def entropy(self, p: Params, noise=None, player: int = 0):
        """As `ClosedFormBackend.entropy`, on the `d`-dimensional grid.

        The only part of this backend that pays the `grid_points^d` cost, and the
        reason `gaussian_entropy: component` (the cheap per-component surrogate)
        exists -- the multidim experiments in `MultiDim.md` all use it.
        """
        return _marginal_entropy(p, self._nodes, self._dv, self._normalize)


class QuadratureBackend:
    """Numeric integration on a tensor-product grid. Works for any box game.

    The payoff matrix `R[i, j] = payoff(a_i, a_j)` over the `M = grid_points^d` nodes
    is built once. Expected payoff is `d0^T R d1 * dv^2`; best responses are optimized
    over nodes inside the box (actions must be feasible), while the mixture densities
    are integrated over a padded grid so the Gaussian tails are not truncated.

    `R` has `M^2 = grid_points^(2d)` entries, which is the hard limit on this backend:
    the 1-D default (801 points) is 5 MB, the same grid in 2-D would be 3 TB. Above
    1-D either drop `grid_points` to a few dozen and accept the grid error, or use the
    `sampled` backend, whose cost does not depend on `d` at all. `max_quadrature_gib`
    turns the difference between the two into an error message instead of an OOM.
    """

    name = "quadrature"
    supports_fixed_opponent = True
    stochastic = False

    def __init__(self, game: ZeroSumGame, n: int, std_max: np.ndarray, normalize: bool,
                 max_gib: float = 2.0):
        space = game.action_space(0)
        if not isinstance(space, BoxSpace):
            raise ValueError(
                "quadrature backend supports BoxSpace games only "
                f"(got {type(space).__name__}); use `idealized.backend: sampled`."
            )
        self.lo, self.hi = _bounds(game)
        self.dim = len(self.lo)
        nodes = n**self.dim
        gib = nodes**2 * 8 / 2**30
        if gib > max_gib:
            raise ValueError(
                f"a {self.dim}-D quadrature grid of {n} points per axis needs a "
                f"{nodes} x {nodes} payoff matrix ({gib:.1f} GiB > "
                f"idealized.max_quadrature_gib={max_gib}). Lower `idealized.grid_points` "
                f"(to about {int((max_gib * 2**30 / 8) ** (0.5 / self.dim))} for this "
                f"dimension), raise the cap, or use `idealized.backend: sampled`, whose "
                f"cost does not grow with the dimension."
            )
        self.grid, self.dv, self.axes = _density_grid(self.lo, self.hi, n, std_max)
        self.dx = float(self.axes[0][1] - self.axes[0][0])
        self.in_box = jnp.all((self.grid >= jnp.asarray(self.lo)) & (self.grid <= jnp.asarray(self.hi)),
                              axis=-1)
        self.normalize = normalize

        a = self.grid
        pay = jax.vmap(lambda x: jax.vmap(lambda y: game.payoff(x, y))(a))(a)
        self.R = jnp.asarray(pay, dtype=jnp.float64)  # (M, M)

    # -- handles -----------------------------------------------------------
    def handle(self, p: Params, noise=None, player: int = 0):
        w = jax.nn.softmax(p.logits)
        return jnp.sum(w[:, None] * self.component_densities(p), axis=0)  # (M,)

    def component_densities(self, p: Params):
        return _component_densities(p, self.grid, self.dv, self.normalize)  # (K, M)

    def uniform_handle(self):
        d = jnp.where(self.in_box, 1.0, 0.0)
        return d / (jnp.sum(d) * self.dv)

    def point_handle(self, x):
        """A point mass at the grid node nearest `x` (a scalar or a `(d,)` action)."""
        x = jnp.broadcast_to(jnp.asarray(x, dtype=jnp.float64), (self.dim,))
        idx = int(jnp.argmin(jnp.sum(jnp.square(self.grid - x), axis=-1)))
        return jnp.zeros(self.grid.shape[0], dtype=jnp.float64).at[idx].set(1.0 / self.dv)

    # -- questions ---------------------------------------------------------
    def expected_payoff(self, h0, h1):
        return (h0 @ self.R @ h1) * self.dv**2

    def component_q(self, p: Params, h_opp, sign: float, noise=None):
        """Per-component expected utility `q_k = E_{a~N(mu_k, Sigma_k)}[ this player's utility ]`.

        `R` is indexed `[player 0's action, player 1's action]`, so which side of it the
        opponent's density is contracted against depends on who is asking: player 0 gets
        `R @ d1`, player 1 gets `-(d0 @ R)`. The two agree only for a payoff that is
        antisymmetric under swapping the players (`multi_point`, `matching_pennies`);
        for `quadratic` with a nonzero coupling, or any other game whose two players do
        not share one landscape, they do not.
        """
        u = (self.R @ h_opp if sign > 0 else -(h_opp @ self.R)) * self.dv  # utility of a *pure* action
        return (self.component_densities(p) @ u) * self.dv

    def exploitability(self, h0, h1):
        U = (h0 @ self.R @ h1) * self.dv**2
        v0 = (self.R @ h1) * self.dv                    # player 0's value of each pure action
        v1 = (h0 @ self.R) * self.dv                    # player 1's cost of each pure action
        br0 = jnp.max(jnp.where(self.in_box, v0, -jnp.inf))
        br1 = jnp.min(jnp.where(self.in_box, v1, jnp.inf))
        return (br0 - U) + (U - br1)

    def entropy(self, p: Params, noise=None, player: int = 0):
        """Exact differential entropy of the mixture -- what `mixture_ppo_loss`
        estimates with `-mixture_marginal_log_prob(...)` at the sampled action."""
        return _marginal_entropy(p, self.grid, self.dv, self.normalize)


class SampledNoise(NamedTuple):
    """One iteration's rollout randomness, drawn once and reused by every gradient step.

    Reused, not redrawn, because that is what `train.py` does: a rollout is collected
    once and `ppo.ppo_epochs` gradient steps are taken against it. Freezing `eps` and
    `comp` (rather than the actions themselves) is what lets the same batch stay
    differentiable as the means and stds move within the iteration -- common random
    numbers, the sampled analogue of the exact solver re-integrating at every step.
    """

    eps: jnp.ndarray = None      # (2, N, d) standard normals, one row per player
    comp: jnp.ndarray = None     # (2, N)    component indices drawn from the rollout policy
    unif: jnp.ndarray = None     # (N, d)    uniform draws in the box, for a `random` opponent


class SampledHandle(NamedTuple):
    """A batch of actions a player is playing, plus their log-probs for `score`."""

    actions: jnp.ndarray = None   # (N, d)
    log_prob: jnp.ndarray = None  # (N,), differentiable w.r.t. this player's params


def _per_component_log_prob(actions, p: Params):
    """`log N(a_n | mu_k, Sigma_k)` for every draw against every component -> `(N, K)`."""
    return jax.vmap(_log_prob_at, in_axes=(None, 0, 0), out_axes=1)(
        actions, p.means, p.scale_tril)


def _mixture_log_prob(actions, p: Params):
    """`log p_mix(a_n)` -> `(N,)`."""
    return jax.nn.logsumexp(jax.nn.log_softmax(p.logits)[None, :]
                            + _per_component_log_prob(actions, p), axis=-1)


class SampledBackend:
    """Monte-Carlo payoffs: the middle ground between this solver and `train.py`.

    The policy, the mirror step on the simplex and the natural-gradient step on the
    Gaussians are the exact ones -- only the two quantities that touch the game are
    estimated from `samples` joint action draws per iteration:

      q_k = E_{a~N(mu_k,s_k)}[u(a, b)]     -> `component_q`, see `q_estimator`
      grad_(mu, A) E[u]                    -> `expected_payoff`, see `grad_estimator`

    Everything else in the objective (the magnet KL, the trust-region KL, the
    repulsion term) is closed-form *in the parameters*, so it stays exact and the
    noise enters at exactly one place. That is the point of the backend: `train.py`
    differs from the exact solver in four ways at once (sampling, a learned critic
    baseline, PPO's clipped ratios, and an MLP trained with Adam), and this isolates
    the first.

    The batch costs `samples` payoff evaluations per iteration, the same budget a
    `ppo.batch_size` rollout spends -- except under `q_estimator: per_component`,
    which spends K times that.
    """

    name = "sampled"
    supports_fixed_opponent = True
    stochastic = True

    _Q_ESTIMATORS = ("responsibility", "per_component", "onpolicy")
    _GRAD_ESTIMATORS = ("pathwise", "score")
    _ENTROPY_SOURCES = ("sampled", "exact")

    def __init__(self, game: ZeroSumGame, cfg: "SolverConfig", metrics: QuadratureBackend):
        i = cfg.idealized
        self.game = game
        self.metrics = metrics          # exact grid; only the reported metrics use it
        self.lo, self.hi = metrics.lo, metrics.hi
        self.dim = len(self.lo)
        self.n = int(i.samples) if i.samples is not None else int(cfg.batch_size)
        if self.n < 1:
            raise ValueError(f"idealized.samples must be >= 1, got {self.n}")
        self.seed = int(i.sample_seed) if i.sample_seed is not None else int(cfg.seed)
        for value, allowed, field in ((i.q_estimator, self._Q_ESTIMATORS, "q_estimator"),
                                      (i.grad_estimator, self._GRAD_ESTIMATORS, "grad_estimator"),
                                      (i.entropy_source, self._ENTROPY_SOURCES, "entropy_source")):
            if value not in allowed:
                raise ValueError(f"unknown idealized.{field} {value!r}, choices: {list(allowed)}")
        self.q_estimator, self.grad_estimator = i.q_estimator, i.grad_estimator
        self.entropy_source = i.entropy_source
        self._payoff = jax.vmap(game.payoff)     # (N, d), (N, d) -> (N,)
        self.detail = (f"{self.n} draws/iteration, seed {self.seed}, q={self.q_estimator}, "
                       f"grad={self.grad_estimator}, entropy={self.entropy_source}")

    # -- the batch ---------------------------------------------------------
    def draw_noise(self, key: jnp.ndarray, p0: Params, p1: Params) -> SampledNoise:
        """The iteration's rollout draws. Components come from the policy at the *start*
        of the iteration, which is the policy `train.py` would have rolled out."""
        k_eps, k_c0, k_c1, k_u = jax.random.split(key, 4)
        return SampledNoise(
            eps=jax.random.normal(k_eps, (2, self.n, self.dim), dtype=jnp.float64),
            comp=jnp.stack([jax.random.categorical(k_c0, p0.logits, shape=(self.n,)),
                            jax.random.categorical(k_c1, p1.logits, shape=(self.n,))]),
            unif=jax.random.uniform(k_u, (self.n, self.dim), dtype=jnp.float64,
                                    minval=jnp.asarray(self.lo), maxval=jnp.asarray(self.hi)),
        )

    def _actions(self, p: Params, noise: SampledNoise, player: int):
        """Reparameterized draws `a_n = mu_{k_n} + A_{k_n} eps_n`, `(N, d)`.

        The triangular matvec is the one place a correlated component differs from a
        diagonal one in the sampler -- see `gaussian.gaussian_sample`.
        """
        eps, comp = noise.eps[player], noise.comp[player]
        mean, scale_tril = p.means[comp], p.scale_tril[comp]
        return gaussian_sample(mean, scale_tril, eps), mean, scale_tril

    def handle(self, p: Params, noise: SampledNoise = None, player: int = 0) -> SampledHandle:
        actions, mean, scale_tril = self._actions(p, noise, player)
        if self.grad_estimator == "score":
            # REINFORCE differentiates the log-prob, not the action, so the action is
            # cut off the graph exactly as a PPO rollout's stored `raw_action` is.
            actions = jax.lax.stop_gradient(actions)
            log_prob = gaussian_log_prob(actions, mean, scale_tril)
        else:
            log_prob = jnp.zeros(actions.shape[0], dtype=actions.dtype)
        return SampledHandle(actions=actions, log_prob=log_prob)

    def fixed_handle(self, kind: str, noise: SampledNoise) -> SampledHandle:
        """The non-training opponent's batch, for `train.mode: fixed_opponent`.

        `random` draws uniformly from the box each iteration, which is exactly what
        `train.py`'s `space.sample` opponent does -- unlike the quadrature backend,
        which integrates against the uniform *density* and so never sees that noise.
        """
        if kind == "random":
            actions = noise.unif
        elif kind == "static":
            mid = jnp.asarray(0.5 * (self.lo + self.hi), dtype=jnp.float64)
            actions = jnp.broadcast_to(mid, (self.n, self.dim))
        else:
            raise ValueError(f"unknown train.opponent {kind!r}")
        return SampledHandle(actions=actions,
                             log_prob=jnp.zeros(self.n, dtype=jnp.float64))

    # -- questions ---------------------------------------------------------
    def expected_payoff(self, h0, h1):
        """MC estimate of `E[payoff]`, differentiable w.r.t. whichever side is traced.

        `pathwise` differentiates the payoff through the reparameterized action:
        `d/dmu_k E[u] = E[ 1{k_n = k} du/da ]`, which is the exact gradient in
        expectation and has a fraction of REINFORCE's variance -- but it needs a
        payoff that is differentiable in the action (every game in `games/` is; a
        step-shaped payoff would silently return a zero gradient).

        `score` is REINFORCE with the batch mean as its baseline, wrapped in the DiCE
        trick so the returned *value* is still the plain MC mean while the gradient is
        `E[(u - b) grad log pi(a)]`. That is the estimator behind PPO's surrogate,
        minus the clipped importance ratio.
        """
        u = self._payoff(h0.actions, h1.actions)
        if self.grad_estimator == "pathwise":
            return jnp.mean(u)
        u = jax.lax.stop_gradient(u)
        baseline = jnp.mean(u)
        lp = h0.log_prob + h1.log_prob                    # the untraced side contributes a constant
        dice = jnp.exp(lp - jax.lax.stop_gradient(lp))    # == 1 in value, grad log pi in gradient
        return jnp.mean((u - baseline) * dice) + baseline

    def component_q(self, p: Params, h_opp, sign: float, noise: SampledNoise = None):
        """Estimate `q_k = E_{a~N(mu_k,s_k)}[ this player's utility ]` for every component.

        No gradient is taken through `q` -- it feeds the closed-form simplex update --
        so the estimators are free to be non-differentiable.

        `responsibility` reweights the one on-policy batch by each component's
        responsibility `r_nk = w_k N(a_n|k) / p_mix(a_n)`: self-normalized importance
        sampling with the mixture as its proposal. Every draw informs every component,
        which is strictly more than `onpolicy` gets out of the same batch -- though as
        the components separate the responsibilities go to 0/1 and the two coincide,
        so a well-separated mixture cannot be rescued from the low-weight-component
        starvation that `onpolicy` (and PPO) suffer.

        `per_component` is the honest unbiased estimator -- a fresh batch per component,
        sharing `eps` as common random numbers -- at K times the payoff budget.
        """
        player = 0 if sign > 0 else 1
        b = h_opp.actions

        def utility(a):                                   # a: (N, d) -> (N,)
            return self._payoff(a, b) if sign > 0 else -self._payoff(b, a)

        if self.q_estimator == "per_component":
            eps = noise.eps[player]                       # (N, d), shared across components
            a = jax.vmap(gaussian_sample, in_axes=(0, 0, None))(
                p.means, p.scale_tril, eps)               # (K, N, d)
            return jax.vmap(lambda a_k: jnp.mean(utility(a_k)))(a)

        actions, _, _ = self._actions(p, noise, player)
        u = utility(actions)                              # (N,)

        if self.q_estimator == "onpolicy":
            onehot = jax.nn.one_hot(noise.comp[player], p.means.shape[0], dtype=jnp.float64)
            count = jnp.sum(onehot, axis=0)               # (K,)
            total = onehot.T @ u
            # a component nobody sampled gets the batch mean: no information means no
            # relative advantage, and a constant drops out of the softmax update.
            return jnp.where(count > 0, total / jnp.maximum(count, 1.0), jnp.mean(u))

        log_r = jax.nn.log_softmax(p.logits)[None, :] + _per_component_log_prob(actions, p)
        r = jax.nn.softmax(log_r, axis=-1)                # (N, K), rows sum to 1
        return (r.T @ u) / jnp.maximum(jnp.sum(r, axis=0), 1e-300)

    def entropy(self, p: Params, noise: SampledNoise = None, player: int = 0):
        """`entropy_source: sampled` is `-log p_mix(a_n)` at the sampled action, averaged
        -- character for character what `mixture_ppo_loss` adds to its loss.

        Worth knowing what that term does here: the action is detached, so the estimate
        differentiates only the density, and `E_{a~p}[grad -log p(a)]` is `-grad int p`,
        which is *zero*. The sampled entropy bonus is therefore a mean-zero gradient --
        variance with no drift -- while the exact solver's is a real force pushing the
        stds up. Set `entropy_source: exact` to compare against that force; the gap
        between the two runs is how much `gaussian_entropy_coef` is actually buying
        `train.py`.
        """
        if self.entropy_source == "exact":
            return self.metrics.entropy(p)
        actions, _, _ = self._actions(p, noise, player)
        return -jnp.mean(_mixture_log_prob(jax.lax.stop_gradient(actions), p))

    def exploitability(self, h0, h1):
        raise AssertionError("the sampled backend reports metrics through `.metrics`")


def build_backend(game: ZeroSumGame, cfg: SolverConfig):
    """Pick and construct the payoff backend named by `idealized.backend`.

    `auto` takes the most exact backend the game supports: a closed form where one
    exists (the 1-D `MultiPointGame` family, or the separable `multidim_decoy_well`),
    otherwise the grid. It never picks `sampled` -- that backend exists to *add* noise
    back, which is a thing to ask for, not a fallback.
    """
    i = cfg.idealized
    dim = len(_bounds(game)[0])
    multi = isinstance(game, MultiDimDecoyWellGame)
    supports_closed_form = hasattr(game, "peaks") and hasattr(game, "_target_moments")
    exact_1d = supports_closed_form and dim == 1
    exact_nd = supports_closed_form and multi

    def grid():
        lo, hi = _bounds(game)
        return _density_grid(lo, hi, i.grid_points, _std_max(game, cfg))

    choice = i.backend
    if choice == "auto":
        if exact_nd and dim > 1:
            choice = "closed_form_multidim"
        elif exact_1d:
            choice = "closed_form"
        else:
            choice = "quadrature"
    if choice == "closed_form":
        if not exact_1d:
            raise ValueError(
                f"{type(game).__name__} (dim {dim}) has no 1-D closed form -- it needs "
                "`.peaks`/`._target_moments` and a scalar action. Use "
                f"`idealized.backend: {'closed_form_multidim' if exact_nd else 'quadrature'}`."
            )
        nodes, dv, _ = grid()
        return ClosedFormBackend(game, nodes, dv, i.normalize_density)
    if choice == "closed_form_multidim":
        if not exact_nd:
            raise ValueError(
                f"the multidim closed form is specialised to MultiDimDecoyWellGame's "
                f"separable payoff, and {type(game).__name__} is not one. Use "
                "`idealized.backend: quadrature` (1-D or a coarse 2-D grid) or `sampled`."
            )
        nodes, dv, _ = grid()
        return ClosedFormMultiDimBackend(game, nodes, dv, i.normalize_density)
    if choice in ("quadrature", "sampled"):
        quadrature = QuadratureBackend(game, i.grid_points, _std_max(game, cfg),
                                       i.normalize_density, i.max_quadrature_gib)
        if choice == "quadrature":
            return quadrature
        # the grid is kept as the *metrics* backend: exploitability estimated off the
        # same noisy batch it is meant to judge would be uninformative.
        return SampledBackend(game, cfg, quadrature)
    raise ValueError(f"unknown idealized.backend {cfg.idealized.backend!r}")


# --------------------------------------------------------------------------- init


def _bounds(game: ZeroSumGame) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis `(low, high)` of the action box, or a clear error.

    Every backend parameterizes the policy as a Gaussian mixture over a box, so a
    game whose players act on a simplex (`blotto`) has no idealized counterpart --
    only `train.py` can run it. The dimension is read off the space: `(1,)` is the
    scalar case, anything larger is the multivariate one.
    """
    space = game.action_space(0)
    if not isinstance(space, BoxSpace):
        raise ValueError(
            f"{type(game).__name__} gives players a {type(space).__name__}; the idealized "
            "solver only handles box action spaces. Run this game with train.py."
        )
    return (np.asarray(space.low, dtype=np.float64).reshape(-1),
            np.asarray(space.high, dtype=np.float64).reshape(-1))


def _std_max(game: ZeroSumGame, cfg: SolverConfig) -> np.ndarray:
    """Per-axis `idealized.std_max`, defaulting to `MixtureActorCritic`'s own ceiling
    `high - low`. A scalar in the config is broadcast over the axes."""
    lo, hi = _bounds(game)
    if cfg.idealized.std_max is not None:
        return np.broadcast_to(np.asarray(cfg.idealized.std_max, dtype=np.float64), lo.shape).copy()
    return hi - lo


def build_init(game: ZeroSumGame, cfg: SolverConfig) -> tuple[Params, Params]:
    """The trainer's own init by default: means spread over the box, uniform weights,
    `std = (high - low) / (2K)` -- see `training.mixture._spread_bias_init` /
    `_std_bias_init`, and note that `logits_head` outputs zeros at a zero observation.

    In `d > 1` the spread runs along the box diagonal: component `k` sits at fraction
    `(k+0.5)/K` of the box in *every* axis, which is what the trainer's per-axis bias
    init also produces. The factor always starts diagonal, so `full_covariance` never
    moves the starting point -- only what the dynamics may do to it afterwards.
    """
    lo, hi = _bounds(game)
    dim = len(lo)
    k = cfg.num_components
    init = cfg.idealized

    def means_for(player: int) -> jnp.ndarray:
        m = init.init_means
        if isinstance(m, str):
            if m != "spread":
                raise ValueError(f"unknown idealized.init_means {m!r}")
            frac = (jnp.arange(k, dtype=jnp.float64) + 0.5) / k
            return jnp.asarray(lo) + frac[:, None] * jnp.asarray(hi - lo)
        arr = jnp.asarray(m, dtype=jnp.float64)
        if arr.ndim == 3:               # (2, K, d): one block per player
            arr = arr[player]
        elif arr.ndim == 2 and dim == 1 and arr.shape == (2, k):
            arr = arr[player][:, None]  # (2, K) per-player scalar means
        elif arr.ndim == 1:             # (K,) scalar means, shared by both players
            arr = arr[:, None]
        if arr.shape != (k, dim):
            raise ValueError(
                f"idealized.init_means has shape {tuple(arr.shape)}, expected "
                f"(num_components, dim) = ({k}, {dim})"
            )
        return arr

    std_max = _std_max(game, cfg)
    if init.init_log_std is None:
        std_val = (hi - lo) / (2 * k)
    else:
        # a *standard deviation*, despite the name it has carried since the parameter
        # itself was a log-std; a non-positive value was the log then, so it still is
        value = float(init.init_log_std)
        std_val = np.broadcast_to(np.exp(value) if value <= 0 else value, lo.shape)
    std_val = np.clip(std_val, init.std_min, std_max)   # (d,)

    # Only the diagonal is populated: the policy starts uncorrelated either way, and
    # under `full_covariance: false` the off-diagonal entries are not even allocated.
    full_covariance = bool(init.full_covariance)
    flat = jnp.zeros((k, scale_param_size(dim, full_covariance)), dtype=jnp.float64)
    flat = flat.at[:, diagonal_slots(dim, full_covariance)].set(jnp.asarray(std_val))
    scale_tril = pack_scale_tril(flat, dim, full_covariance)

    if init.init_weights is None:
        logits = jnp.zeros(k, dtype=jnp.float64)
    else:
        w = jnp.asarray(init.init_weights, dtype=jnp.float64)
        if w.shape[0] != k:
            raise ValueError(f"idealized.init_weights has {w.shape[0]} entries, expected num_components={k}")
        logits = jnp.log(w)

    return (Params(logits=logits, means=means_for(0), scale_tril=scale_tril),
            Params(logits=logits, means=means_for(1), scale_tril=scale_tril))


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
    """Anneal the floor on the scale factor's diagonal from `anneal_std_from` to `std_min`.

    Geometric interpolation -- linear in `log sigma`, as it was when the parameter
    itself was `log sigma` -- so the schedule is unchanged and only its units differ.
    """
    base = float(cfg.idealized.std_min)
    if cfg.idealized.anneal_std_from <= 0.0:
        return jnp.full((), base, dtype=jnp.float64)
    return jnp.exp((1.0 - frac) * float(np.log(cfg.idealized.anneal_std_from))
                   + frac * float(np.log(base)))


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
    lo, hi = jnp.asarray(_bounds(game)[0]), jnp.asarray(_bounds(game)[1])
    dim = lo.shape[0]
    std_hi = jnp.asarray(_std_max(game, cfg))
    i = cfg.idealized
    total = max(cfg.total_iters - 1, 1)

    if cfg.magnet_interval < 1:
        raise ValueError("magnet_interval must be >= 1")

    # Which entries of the factor are parameters at all. Under `full_covariance:
    # false` the off-diagonal ones are not, and masking the step is what enforces
    # that -- `training/mixture.py` gets it for free by never allocating them, but
    # here the factor *is* the parameter, and a term the payoff does not separate
    # over (the mixture's differential entropy, say) has a nonzero gradient on an
    # off-diagonal entry even at a factor that is exactly diagonal.
    scale_mask = jnp.zeros((dim, dim), dtype=jnp.float64).at[
        tril_positions(dim, i.full_covariance)].set(1.0)

    trains = {0: True, 1: True}
    if cfg.mode == "fixed_opponent":
        trains[1 - cfg.perspective] = False

    if i.gaussian_entropy == "marginal":
        entropy_term = backend.entropy           # mixture differential entropy
    elif i.gaussian_entropy == "component":
        # noqa: E731 -- legacy per-component term; `sum_j log A_jj == 1/2 log det Sigma`
        # is its multivariate form, and reduces to `sum(log_std)` on a diagonal factor
        entropy_term = lambda pp, noise=None, player=0: jnp.sum(log_scale_det(pp.scale_tril))
    else:
        raise ValueError(f"unknown idealized.gaussian_entropy {i.gaussian_entropy!r}")

    def player_step(p: Params, h_opp, old: Params, magnet: Params, sign: float, lam, floor,
                    noise=None):
        player = 0 if sign > 0 else 1
        # --- categorical head: exact mirror step on the simplex ---
        if i.freeze_weights:
            logits = p.logits
        else:
            q = backend.component_q(p, h_opp, sign, noise)
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
            h = backend.handle(pp, noise, player)
            pay = (backend.expected_payoff(h, h_opp) if sign > 0
                   else -backend.expected_payoff(h_opp, h))
            ent = cfg.gaussian_entropy_coef * entropy_term(pp, noise, player)
            mag = cfg.magnet_gaussian_kl_coef * jnp.sum(
                kl_w * gaussian_kl(pp.means, pp.scale_tril, magnet.means, magnet.scale_tril)
            )
            # `mixture_ppo_loss` penalizes KL(old || new) for the Gaussian head; kept
            # in that direction here since a gradient step needs no closed-form argmax.
            trpo = cfg.trpo_gaussian_kl_coef * jnp.sum(
                kl_w * gaussian_kl(old.means, old.scale_tril, pp.means, pp.scale_tril)
            )
            # per-axis L1 repulsion between component means, summed over pairs and axes
            rep = lam * jnp.sum(jnp.abs(pp.means[:, None, :] - pp.means[None, :, :])) / 2.0
            return pay + ent + rep - mag - trpo

        g = jax.grad(obj)(p)
        # `F^-1 grad` for the Gaussian Fisher metric in `(mu, A)` coordinates. On a
        # diagonal factor this is exactly the `(sigma^2 grad_mu, 1/2 sigma^2 grad_sigma)`
        # preconditioner the scalar solver used, written in `sigma` rather than
        # `log sigma`; the natural gradient is parametrization-invariant, so the two
        # agree in the continuous limit and differ only in the discretization.
        nat_mean, nat_scale = natural_gradient(p.scale_tril, g.means, g.scale_tril * scale_mask)
        nat_scale = nat_scale * scale_mask
        means = p.means + (cfg.lr * nat_mean if i.train_means else 0.0)
        scale_tril = p.scale_tril + (cfg.lr * nat_scale if i.train_std else 0.0)
        # `A E` with both factors lower triangular is lower triangular, so the step
        # preserves the structure exactly and only the diagonal needs a projection.
        return Params(logits=logits,
                      means=jnp.clip(means, lo, hi),
                      scale_tril=clamp_scale_tril(scale_tril, floor, std_hi))

    def iteration(carry, _):
        p0, p1, m0, m1, e0, e1, fixed_handle, it, key = carry
        frac = it.astype(jnp.float64) / total
        lam, floor = repulsion_coef_at(cfg, frac), std_floor_at(cfg, frac)
        old0, old1 = p0, p1

        # One batch per iteration, reused by every gradient step below -- a PPO rollout.
        noise, fixed_h = None, fixed_handle
        if backend.stochastic:
            key, rollout_key = jax.random.split(key)
            noise = backend.draw_noise(rollout_key, old0, old1)
            if cfg.mode == "fixed_opponent":
                fixed_h = backend.fixed_handle(cfg.opponent, noise)

        def grad_step(pair, _):
            a, b = pair
            # a non-training player is not a mixture at all (it may be uniform or a
            # point mass), so its handle is the fixed one built above / in `run`.
            h0 = backend.handle(a, noise, 0) if trains[0] else fixed_h
            h1 = backend.handle(b, noise, 1) if trains[1] else fixed_h
            na = player_step(a, h1, old0, m0, +1.0, lam, floor, noise) if trains[0] else a
            nb = player_step(b, h0, old1, m1, -1.0, lam, floor, noise) if trains[1] else b
            return (na, nb), None

        (p0, p1), _ = jax.lax.scan(grad_step, (p0, p1), None, length=cfg.inner_steps)

        it = it + 1
        snapshot = (it % cfg.magnet_interval) == 0
        m0, m1 = _tree_where(snapshot, p0, m0), _tree_where(snapshot, p1, m1)
        e0, e1 = _ema(p0, e0, cfg.target_tau), _ema(p1, e1, cfg.target_tau)
        # a stochastic backend redraws its fixed opponent above, from this iteration's
        # noise, so what it carries here is the `None` `run` handed it.
        return (p0, p1, m0, m1, e0, e1, fixed_handle, it, key), None

    return iteration


def run(game: ZeroSumGame, cfg: SolverConfig, p0: Params, p1: Params):
    backend = build_backend(game, cfg)

    # the exact backend the reported metrics are computed with: itself, unless it is
    # stochastic, in which case its grid.
    metrics = backend.metrics if backend.stochastic else backend

    fixed_handle = jnp.zeros_like(metrics.handle(p0)) if metrics.name == "quadrature" else None
    metric_fixed = fixed_handle
    if cfg.mode == "fixed_opponent":
        if not backend.supports_fixed_opponent:
            raise ValueError(
                f"train.mode: fixed_opponent needs the quadrature or sampled backend "
                f"(the {backend.name} backend can only integrate two Gaussian mixtures)."
            )
        if cfg.opponent == "random":
            metric_fixed = metrics.uniform_handle()
        elif cfg.opponent == "static":
            metric_fixed = metrics.point_handle(0.5 * (metrics.lo + metrics.hi))
        else:
            raise ValueError(f"unknown train.opponent {cfg.opponent!r}")
        # the sampled backend redraws its opponent batch every iteration instead
        fixed_handle = None if backend.stochastic else metric_fixed
    elif cfg.mode != "self_play":
        raise ValueError(f"unknown train.mode {cfg.mode!r}")

    iteration = build_iteration(game, cfg, backend)
    chunk_fns: dict[int, Any] = {}

    def handles_of(a: Params, b: Params):
        h0 = metric_fixed if (cfg.mode == "fixed_opponent" and cfg.perspective == 1) else metrics.handle(a)
        h1 = metric_fixed if (cfg.mode == "fixed_opponent" and cfg.perspective == 0) else metrics.handle(b)
        return h0, h1

    # Not jitted: `idealized_mmd.exploitability` builds its best-response grid with
    # Python floats off the game object, so it cannot be traced. It runs once per
    # logged chunk, so eager execution costs nothing.
    def expl_fn(a, b):
        return metrics.exploitability(*handles_of(a, b))

    def player_record(p: Params) -> dict:
        """One player's strategy, per component: weight, mean, and *marginal* std.

        `marginal_std` rather than the factor's diagonal: the diagonal is the
        conditional std given the earlier coordinates, and only coincides with the
        per-axis spread when the component is uncorrelated (`training/gaussian.py`).
        `corr` is the largest off-diagonal correlation each component carries, which
        is the only place a `full_covariance` run's extra parameters are visible at
        all -- it is identically zero for a diagonal factor.
        """
        return {
            "w": [float(x) for x in jax.nn.softmax(p.logits)],
            "means": np.asarray(p.means).tolist(),
            "std": np.asarray(marginal_std(p.scale_tril)).tolist(),
            "corr": [float(x) for x in _max_abs_correlation(p.scale_tril)],
        }

    def record(t: int, a: Params, b: Params, ea: Params, eb: Params) -> dict:
        entry = {"t": t, "expl": float(expl_fn(a, b)), "target_expl": float(expl_fn(ea, eb))}
        for player, p in ((0, a), (1, b)):
            for key, value in player_record(p).items():
                entry[f"{key}{player}"] = value
        return entry

    carry = (p0, p1, p0, p1, p0, p1, fixed_handle, jnp.zeros((), dtype=jnp.int64),
             jax.random.PRNGKey(backend.seed if backend.stochastic else 0))
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


def _max_abs_correlation(scale_tril):
    """Largest `|Sigma_ij| / sqrt(Sigma_ii Sigma_jj)`, `i != j`, per component -> `(K,)`.

    Zero for a diagonal factor and for `d == 1`. Reported because it is the only
    thing a `full_covariance` run does that a diagonal one cannot, and without it a
    correlation that never moved off zero is indistinguishable from one that did.
    """
    if scale_tril.shape[-1] == 1:
        return jnp.zeros(scale_tril.shape[0], dtype=scale_tril.dtype)
    cov = jnp.einsum("...ij,...kj->...ik", scale_tril, scale_tril)
    sd = marginal_std(scale_tril)
    corr = cov / (sd[..., :, None] * sd[..., None, :])
    off = corr * (1.0 - jnp.eye(corr.shape[-1], dtype=corr.dtype))
    return jnp.max(jnp.abs(off), axis=(-2, -1))


def _fmt(w, mu, sd, corr=None) -> str:
    """One player's mixture as a line: `weight@mean(sd)` per component.

    A `d`-dimensional mean prints as a parenthesized tuple and the std alongside it
    is the per-axis marginal one; a nonzero correlation is appended as `r<value>` so
    that a `full_covariance` run's off-diagonals are visible in the run's own output.
    """
    parts = []
    for k, (wk, mk, sk) in enumerate(zip(w, mu, sd)):
        mk, sk = np.atleast_1d(mk), np.atleast_1d(sk)
        if mk.size == 1:      # a scalar action reads better without the brackets
            body = f"{mk[0]:+.2f}(sd{sk[0]:.3f})"
        else:
            body = ("(" + ",".join(f"{x:+.2f}" for x in mk) + ")"
                    + "(sd" + ",".join(f"{x:.3f}" for x in sk) + ")")
        if corr is not None and abs(corr[k]) > 5e-3:
            body += f"r{corr[k]:+.2f}"
        parts.append(f"{wk:.2f}@{body}")
    return "[" + " ".join(parts) + "]"


def _report_unused(run_config: RunConfig, cfg: SolverConfig) -> None:
    sampled = cfg.idealized.backend == "sampled"
    used = _SAMPLED_SHARED_FIELDS if sampled else {}
    if used:
        print("used by the sampled backend:")
        for (section, field), why in used.items():
            value = getattr(getattr(run_config, section), field)
            print(f"  {section}.{field} = {value!r}  -- {why}")
    print("ignored (no idealized counterpart):")
    for (section, field), why in _UNUSED_SHARED_FIELDS.items():
        if (section, field) in used:
            continue
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
    backend = backend.metrics if backend.stochastic else backend
    if backend.name != "quadrature":
        return
    reached = float(min(np.min(np.asarray(h["std0"] + h["std1"])) for h in history))
    if reached >= 2 * backend.dx:
        return
    span = float(backend.axes[0][-1] - backend.axes[0][0])
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
    if isinstance(game, SequentialZeroSumGame):
        raise ValueError(
            f"{type(game).__name__} is a sequential game, and this solver integrates a one-shot "
            "payoff over the action space -- there is no such integral for a game tree. "
            "Run it with train.py instead."
        )
    p0, p1 = build_init(game, cfg)

    print(f"config  : {args.config}  ({'shared with train.py' if run_config else 'legacy standalone'})")
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"solver  : {dataclasses.asdict(cfg)}")
    if run_config is not None:
        _report_unused(run_config, cfg)
    print()

    p0f, p1f, history, backend = run(game, cfg, p0, p1)
    _warn_grid(backend, history)
    detail = getattr(backend, "detail", None)
    print(f"\nbackend : {backend.name}{f' ({detail})' if detail else ''}  |  mode: {cfg.mode}"
          f"{'' if cfg.mode == 'self_play' else f' (player {cfg.perspective} vs {cfg.opponent})'}"
          f"  |  {cfg.total_iters} iterations x {cfg.inner_steps} gradient step(s)\n")

    n = len(history)
    idx = sorted({round(i * (n - 1) / max(cfg.idealized.log_rows - 1, 1))
                  for i in range(cfg.idealized.log_rows)})
    frozen = (1 - cfg.perspective) if cfg.mode == "fixed_opponent" else None
    for i in idx:
        e = history[i]
        cols = [_fmt(e["w0"], e["means0"], e["std0"], e["corr0"]),
                _fmt(e["w1"], e["means1"], e["std1"], e["corr1"])]
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
