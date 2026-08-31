"""Idealized (noise-free, exact-gradient) MMD for the *multidimensional* decoy-well game.

`idealized_mmd.py` is 1-D: a policy is a K-component Gaussian mixture over a scalar
action. This module is its `dim`-dimensional counterpart, specialised to
`games.examples.MultiDimDecoyWellGame` -- the separable product game whose payoff
is a sum of `dim` independent 1-D `DecoyWellGame`s.

Policy model: a **joint** K-component diagonal-Gaussian mixture (the same shape as
the PPO `MixtureActorCritic`), i.e. pick component `k ~ Cat(w)` then draw
`a ~ N(means_k, diag(std_k^2))` with `means_k, std_k in R^dim`. This is *not* a
per-axis product mixture -- the categorical head is shared across coordinates, so
the components' cross-coordinate assignment matters.

Why K components can still represent a Nash despite the product Nash needing `K^dim`
grid corners: the game separates, so the expected payoff depends only on each
player's *per-coordinate marginals*. A joint mixture that puts component k at
`(peak_k, peak_k, ..., peak_k)` (the diagonal) with weight `weights_k` has, in every
coordinate, the marginal `weights` over `peaks` -- the 1-D Nash marginal -- so it is
an exact Nash. The interesting question is whether the MMD *vector field* reaches
such a configuration, and whether the per-coordinate decoy traps block it.

All expectations are closed form (Gaussian convolution of the per-axis well, plus
Gaussian moments of the `[-1,1]`-normalised coupling feature), summed over
coordinates. Best responses for the exploitability separate per coordinate and are
done by 1-D grid search. `dim == 1` reproduces `idealized_mmd.py`.

Run: `python idealized_mmd_multidim.py configs/multidim/<name>.yaml`
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

from games.configs import GAME_CONFIGS
from games.examples import MultiDimDecoyWellGame
from training.gaussian import (
    clamp_scale_tril,
    diagonal_slots,
    gaussian_kl,
    log_scale_det,
    marginal_std,
    natural_gradient,
    pack_scale_tril,
    scale_param_size,
)


class Params(NamedTuple):
    logits: jnp.ndarray      # (K,)            categorical logits over components
    means: jnp.ndarray       # (K, dim)        each component's Gaussian means
    scale_tril: jnp.ndarray  # (K, dim, dim)   each component's Cholesky covariance factor


# --------------------------------------------------------------------------- geometry


class _Geom(NamedTuple):
    dim: int
    centers: jnp.ndarray     # (B,)  per-axis bump centers (peaks + decoys)
    heights: jnp.ndarray     # (B,)
    widths: jnp.ndarray      # (B,)
    mid: float
    half_range: float
    max_order: int
    target: jnp.ndarray      # (max_order,)
    coupling: float


def geometry(game: MultiDimDecoyWellGame) -> _Geom:
    return _Geom(
        dim=int(game.dim),
        centers=jnp.asarray(game.bump_centers, dtype=jnp.float64),
        heights=jnp.asarray(game.bump_heights, dtype=jnp.float64),
        widths=jnp.asarray(game.bump_widths, dtype=jnp.float64),
        mid=float(game._mid),
        half_range=float(game._half_range),
        max_order=int(game.peaks.shape[0]) - 1,
        target=jnp.asarray(game._target_moments, dtype=jnp.float64),
        coupling=float(getattr(game, "coupling", 1.0)),
    )


def well_expectation(means, scale_tril, g: _Geom):
    """E_{a~N(mu_k, Sigma_k)}[ sum_d D(a_d) ] per component -> (K,).

    Same closed form as `idealized_mmd.well_expectation` (a bump of height h, width w
    convolved with N(0, s^2) has height h*w/sqrt(w^2+s^2)), summed over coordinates.

    Only the *marginal* variances `Sigma_dd` appear, and that is not an
    approximation: `sum_d D(a_d)` is a sum of functions of one coordinate each,
    so its expectation is determined by the one-dimensional marginals alone. It
    is also the reason a full covariance buys nothing on this game -- see the
    note in `run`.
    """
    var = jnp.square(marginal_std(scale_tril))[..., None] + g.widths**2   # (K, dim, B)
    amp = g.heights * g.widths / jnp.sqrt(var)
    contrib = amp * jnp.exp(-(means[..., None] - g.centers) ** 2 / (2 * var))
    return jnp.sum(contrib, axis=(-2, -1))                        # (K,)


def _central_moment(i: int, std):
    if i % 2 == 1:
        return jnp.zeros_like(std)
    k = i // 2
    dfact = math.factorial(i) / (2**k * math.factorial(k))
    return dfact * std**i


def component_feat_moments(means, scale_tril, g: _Geom):
    """E_{a~N(mu,Sigma)}[u(a)^j], j=1..max_order, elementwise over (K, dim) -> (K, dim, max_order).

    Per-coordinate powers again, so again the marginals are the whole story.
    """
    std = marginal_std(scale_tril)
    shift = means - g.mid
    moments = [
        sum(math.comb(j, i) * shift ** (j - i) * _central_moment(i, std) for i in range(j + 1))
        / g.half_range**j
        for j in range(1, g.max_order + 1)
    ]
    return jnp.stack(moments, axis=-1)                           # (K, dim, max_order)


def mixture_stats(p: Params, g: _Geom):
    """`(w, e_well, e_feat, feat_centered, well_comp)`.

    e_feat is the per-coordinate expected feature vector, shape (dim, max_order);
    feat_centered is the per-component centered moment, shape (K, dim, max_order).
    """
    w = jax.nn.softmax(p.logits)                                 # (K,)
    well_comp = well_expectation(p.means, p.scale_tril, g)       # (K,)
    e_well = jnp.sum(w * well_comp)
    feat_centered = component_feat_moments(p.means, p.scale_tril, g) - g.target  # (K, dim, max_order)
    e_feat = jnp.sum(w[:, None, None] * feat_centered, axis=0)   # (dim, max_order)
    return w, e_well, e_feat, feat_centered, well_comp


def expected_payoff(px: Params, py: Params, g: _Geom):
    _, well_x, feat_x, _, _ = mixture_stats(px, g)
    _, well_y, feat_y, _, _ = mixture_stats(py, g)
    return well_x - well_y + g.coupling * jnp.sum(feat_x * feat_y)


def component_q(p: Params, opp: Params, g: _Geom, sign: float):
    """Per-component expected utility q_k for the categorical mirror update -> (K,)."""
    _, _, feat_opp, _, _ = mixture_stats(opp, g)
    _, _, _, feat_centered, well_comp = mixture_stats(p, g)
    coupling_term = sign * g.coupling * jnp.einsum("dj,kdj->k", feat_opp, feat_centered)
    return well_comp + coupling_term


# --------------------------------------------------------------------------- exploitability


def _grid(game: MultiDimDecoyWellGame, n: int = 2001):
    space = game.action_space(0)
    return jnp.linspace(float(space.low[0]), float(space.high[0]), n, dtype=jnp.float64)


def _well_grid(a, g: _Geom):  # (N,) -> per-axis 1-D well D(a) at each grid point
    return jnp.sum(
        g.heights * jnp.exp(-(a[:, None] - g.centers) ** 2 / (2 * g.widths**2)), axis=-1
    )


def _feat_grid(a, g: _Geom):  # (N,) -> (N, max_order): u(a)^j - target_j
    u = (a - g.mid) / g.half_range
    orders = jnp.arange(1, g.max_order + 1, dtype=jnp.float64)
    return u[:, None] ** orders - g.target


def exploitability(px: Params, py: Params, game: MultiDimDecoyWellGame, g: _Geom):
    """NashConv. Best responses separate per coordinate -> `dim` independent 1-D grid searches."""
    _, well_x, feat_x, _, _ = mixture_stats(px, g)
    _, well_y, feat_y, _, _ = mixture_stats(py, g)
    grid = _grid(game)
    Dg = _well_grid(grid, g)                 # (N,)
    Fg = _feat_grid(grid, g)                 # (N, max_order)
    # player 0 best response value: sum_d max_a [ D(a) + coupling * feat(a) . feat_y[d] ] - E[well_y]
    per_dim0 = jnp.max(Dg[:, None] + g.coupling * (Fg @ feat_y.T), axis=0)   # (dim,)
    br0 = jnp.sum(per_dim0) - well_y
    # player 1 best response value: E[well_x] + sum_d min_b [ -D(b) + coupling * feat_x[d] . feat(b) ]
    per_dim1 = jnp.min(-Dg[:, None] + g.coupling * (Fg @ feat_x.T), axis=0)  # (dim,)
    br1 = well_x + jnp.sum(per_dim1)
    return br0 - br1


# --------------------------------------------------------------------------- MMD update


def categorical_mirror_update(logits, q, magnet_logits, lr, magnet_coef, entropy_coef):
    lp = jax.nn.log_softmax(logits)
    lm = jax.nn.log_softmax(magnet_logits)
    num = lr * q + lr * magnet_coef * lm + lp
    return num / (1.0 + lr * magnet_coef + lr * entropy_coef)


@dataclasses.dataclass
class MMDSection:
    lr: float = 0.05
    steps: int = 20000
    magnet_interval: int = 200
    magnet_coef: float = 0.2
    entropy_coef: float = 0.0
    num_components: int = 2
    train_means: bool = True
    train_std: bool = True
    freeze_weights: bool = False
    anneal_std_from: float = 0.0
    std_min: float = 1e-3
    std_max: float = 1.0
    # Give each component a full covariance (a lower-triangular Cholesky factor)
    # rather than a diagonal one. Inert on this game -- see the note in `run` --
    # and off by default for that reason.
    full_covariance: bool = False
    # Annealed mean-repulsion sweep (the one gradient-only escape from the decoy trap in 1-D):
    # add `coef * sum_{i<j} ||mu_i - mu_j||_1` to each player's objective, ramped 0 -> coef -> 0.
    repulsion_coef: float = 0.0
    repulsion_ramp: float = 0.2       # fraction of training spent ramping up
    repulsion_hold: float = 0.5       # fraction held at `repulsion_coef`


@dataclasses.dataclass
class InitSection:
    means: Any = "spread"     # "spread" | "diagonal" | list (per component, per dim)
    weights: Any = None
    log_std: Any = None


@dataclasses.dataclass
class LogSection:
    every: int = 250
    rows: int = 12
    out: str | None = None


def repulsion_coef_at(mmd: MMDSection, frac: float) -> float:
    """Ramp 0 -> coef over `repulsion_ramp`, hold for `repulsion_hold`, then anneal back to 0."""
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
    """The floor on the scale factor's diagonal at training fraction `frac`.

    Geometric interpolation from `anneal_std_from` down to `std_min` -- i.e.
    linear in `log sigma`, as it was when the parameter itself was `log sigma`,
    so the schedule is unchanged; only the units it is expressed in are.
    """
    if mmd.anneal_std_from <= 0.0:
        return mmd.std_min
    return float(np.exp((1.0 - frac) * np.log(mmd.anneal_std_from) + frac * np.log(mmd.std_min)))


def run(game: MultiDimDecoyWellGame, mmd: MMDSection, p0: Params, p1: Params, log: LogSection):
    """MMD (magnetic mirror descent) on the Gaussian-mixture parameters.

    The continuous update is the *natural* gradient -- `F^-1 grad` for the
    Gaussian Fisher metric `F` (see `training.gaussian.natural_gradient`) --
    which is what makes it MMD rather than plain projected gradient ascent, and
    which is also why the choice between `sigma` and `log sigma` coordinates is
    immaterial here: the natural gradient is parametrization-invariant, and the
    diagonal floor clips to the same set either way. That is *not* true of the
    sampled trainer in `training/`, which takes Euclidean steps.

    A warning about `mmd.full_covariance` on this game: `MultiDimDecoyWellGame`
    is a sum of per-coordinate payoffs, so the expected payoff depends only on
    the per-axis marginals (see `well_expectation`) and its gradient w.r.t.
    every off-diagonal entry of the factor is *identically zero*. The only force
    on those entries is the magnet KL, which pulls them to the magnet's
    correlations -- zero, since the init is diagonal. Turning it on here is a
    no-op by construction; it is here so the machinery exists for a payoff with
    genuine cross-coordinate structure (`CurvaturePumpGame`,
    `AsymmetricWellGame`), whose expectations this runner does not implement.
    """
    g = geometry(game)
    space = game.action_space(0)
    lo, hi = float(space.low[0]), float(space.high[0])

    def step(p0, p1, m0, m1, floor, lam):
        def apply(p, opp, magnet, sign):
            if mmd.freeze_weights:
                logits = p.logits
            else:
                q = component_q(p, opp, g, sign)
                logits = categorical_mirror_update(
                    p.logits, q, magnet.logits, mmd.lr, mmd.magnet_coef, mmd.entropy_coef
                )

            def obj(pp):
                pay = expected_payoff(pp, opp, g) if sign > 0 else -expected_payoff(opp, pp, g)
                # Gaussian differential entropy up to the additive constant that
                # `entropy_coef` cannot see: `sum_i log A_ii == 1/2 log det Sigma`.
                ent = mmd.entropy_coef * jnp.sum(log_scale_det(pp.scale_tril))
                mag = mmd.magnet_coef * jnp.sum(
                    gaussian_kl(pp.means, pp.scale_tril, magnet.means, magnet.scale_tril)
                )
                # per-axis L1 repulsion between component means, summed over pairs and dims
                diff = jnp.abs(pp.means[:, None, :] - pp.means[None, :, :])
                rep = lam * jnp.sum(diff) / 2.0
                return pay + ent + rep - mag

            grad = jax.grad(obj)(p)
            nat_mean, nat_scale = natural_gradient(p.scale_tril, grad.means, grad.scale_tril)
            means = p.means + (mmd.lr * nat_mean if mmd.train_means else 0.0)
            scale_tril = p.scale_tril + (mmd.lr * nat_scale if mmd.train_std else 0.0)
            means = jnp.clip(means, lo, hi)
            # `A E` with both factors lower triangular is lower triangular, so the
            # step preserves the structure exactly and only the diagonal needs a
            # projection back into `[floor, std_max]`.
            scale_tril = clamp_scale_tril(scale_tril, floor, mmd.std_max)
            return Params(logits=logits, means=means, scale_tril=scale_tril)

        return apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)

    jstep = jax.jit(step)
    m0, m1 = p0, p1
    history = []
    for t in range(mmd.steps):
        frac = t / max(mmd.steps - 1, 1)
        p0, p1 = jstep(p0, p1, m0, m1, std_floor_at(mmd, frac), repulsion_coef_at(mmd, frac))
        if (t + 1) % mmd.magnet_interval == 0:
            m0, m1 = p0, p1
        if t % log.every == 0 or t == mmd.steps - 1:
            history.append({
                "t": t,
                "expl": float(exploitability(p0, p1, game, g)),
                "w0": [float(x) for x in jax.nn.softmax(p0.logits)],
                "means0": np.asarray(p0.means).tolist(),
                "std0": np.asarray(marginal_std(p0.scale_tril)).tolist(),
                "w1": [float(x) for x in jax.nn.softmax(p1.logits)],
                "means1": np.asarray(p1.means).tolist(),
                "std1": np.asarray(marginal_std(p1.scale_tril)).tolist(),
            })
    return p0, p1, history


# --------------------------------------------------------------------------- init


def build_init(game: MultiDimDecoyWellGame, mmd: MMDSection, init: InitSection) -> tuple[Params, Params]:
    space = game.action_space(0)
    lo, hi = float(space.low[0]), float(space.high[0])
    dim = int(game.dim)
    k = mmd.num_components

    def means_for(kind) -> jnp.ndarray:
        if isinstance(kind, str) and kind in ("spread", "diagonal"):
            # component k at fraction (k+0.5)/K of the box, identically in every axis
            # -> all components sit on the diagonal (mirrors the trainer's per-axis spread init)
            frac = (jnp.arange(k, dtype=jnp.float64) + 0.5) / k
            col = lo + frac * (hi - lo)
            return jnp.broadcast_to(col[:, None], (k, dim))
        arr = jnp.asarray(kind, dtype=jnp.float64)
        if arr.shape != (k, dim):
            raise ValueError(f"init.means must be shape ({k}, {dim}), got {arr.shape}")
        return arr

    # `init.log_std` is (and always was) a *standard deviation*, not its log.
    std_val = (hi - lo) / (2 * k) if init.log_std is None else float(init.log_std)
    std_val = float(np.clip(std_val, mmd.std_min, mmd.std_max))
    # Diagonal factor: the policy starts uncorrelated whether or not
    # `full_covariance` is on, so the flag never moves the starting point.
    flat = jnp.zeros((k, scale_param_size(dim, mmd.full_covariance)), dtype=jnp.float64)
    flat = flat.at[:, diagonal_slots(dim, mmd.full_covariance)].set(std_val)
    scale_tril = pack_scale_tril(flat, dim, mmd.full_covariance)

    if init.weights is None:
        logits = jnp.zeros(k, dtype=jnp.float64)
    else:
        w = jnp.asarray(init.weights, dtype=jnp.float64)
        if w.shape[0] != k:
            raise ValueError(f"init.weights has {w.shape[0]} entries, expected num_components={k}")
        logits = jnp.log(w)

    m = init.means
    m0 = m[0] if isinstance(m, list) and len(m) == 2 and isinstance(m[0], list) and isinstance(m[0][0], list) else m
    m1 = m[1] if isinstance(m, list) and len(m) == 2 and isinstance(m[0], list) and isinstance(m[0][0], list) else m
    return (Params(logits=logits, means=means_for(m0), scale_tril=scale_tril),
            Params(logits=logits, means=means_for(m1), scale_tril=scale_tril))


# --------------------------------------------------------------------------- CLI


def _build(cls: type, data: dict) -> Any:
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown field(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path):
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"game", "mmd", "init", "log"}
    if unknown:
        raise ValueError(f"unknown top-level section(s): {sorted(unknown)}")
    game_raw = dict(raw.get("game", {}))
    name = game_raw.pop("name", None)
    if name != "multidim_decoy_well":
        raise ValueError("this runner only supports game.name == 'multidim_decoy_well'")
    game = _build(GAME_CONFIGS[name], game_raw).build()
    return (game,
            _build(MMDSection, raw.get("mmd", {}) or {}),
            _build(InitSection, raw.get("init", {}) or {}),
            _build(LogSection, raw.get("log", {}) or {}))


def _fmt(w, mu, sd) -> str:
    parts = []
    for wi, mi, si in zip(w, mu, sd):
        pos = ",".join(f"{x:+.2f}" for x in mi)
        parts.append(f"{wi:.2f}@({pos})")
    return "[" + " ".join(parts) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    args = ap.parse_args()

    game, mmd, init, log = load_config(args.config)
    p0, p1 = build_init(game, mmd, init)

    print(f"game : MultiDimDecoyWellGame dim={game.dim} peaks={list(np.asarray(game.peaks))} "
          f"decoys={game.decoys}")
    print(f"mmd  : {dataclasses.asdict(mmd)}")
    p0f, p1f, history = run(game, mmd, p0, p1, log)

    n = len(history)
    idx = sorted({round(i * (n - 1) / max(log.rows - 1, 1)) for i in range(log.rows)})
    for i in idx:
        e = history[i]
        print(f"  t={e['t']:6d}  expl={e['expl']:+8.4f}   "
              f"P0 {_fmt(e['w0'], e['means0'], e['std0'])}")

    tail = float(np.mean([h["expl"] for h in history[int(n * 0.7):]]))
    best = min(h["expl"] for h in history)
    print(f"\nfinal expl {history[-1]['expl']:+.4f} | tail(30%) {tail:+.4f} | best {best:+.4f}")

    if log.out:
        Path(log.out).parent.mkdir(parents=True, exist_ok=True)
        Path(log.out).write_text(json.dumps(history, indent=2))
        print(f"saved history -> {log.out}")


if __name__ == "__main__":
    main()
