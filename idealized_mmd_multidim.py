"""Closed-form payoffs for the *multidimensional* separable well game.

The `dim`-dimensional counterpart of `idealized_mmd.py`'s math half, and the
library behind `run_idealized.py`'s `closed_form_multidim` backend. It is
specialised to `games.examples.MultiDimDecoyWellGame` -- the separable product
game whose payoff is a sum of `dim` independent 1-D `DecoyWellGame`s.

The policy it integrates is the solver's own: a **joint** K-component Gaussian
mixture (`run_idealized.Params`, the same shape as the PPO
`MixtureActorCritic`), i.e. pick component `k ~ Cat(w)` then draw
`a ~ N(means_k, A_k A_k^T)` with `means_k in R^dim` and `A_k` a lower-triangular
Cholesky factor. This is *not* a per-axis product mixture -- the categorical head
is shared across coordinates, so the components' cross-coordinate assignment
matters.

Why K components can still represent a Nash despite the product Nash needing
`K^dim` grid corners: the game separates, so the expected payoff depends only on
each player's *per-coordinate marginals*. A joint mixture that puts component k at
`(peak_k, peak_k, ..., peak_k)` (the diagonal) with weight `weights_k` has, in every
coordinate, the marginal `weights` over `peaks` -- the 1-D Nash marginal -- so it is
an exact Nash. The interesting question is whether the MMD *vector field* reaches
such a configuration, and whether the per-coordinate decoy traps block it; that
question is asked by `run_idealized.py`, which owns the MMD loop, the config schema
and the CLI. This module owns only the integrals.

All expectations are closed form (Gaussian convolution of the per-axis well, plus
Gaussian moments of the `[-1,1]`-normalised coupling feature), summed over
coordinates. Best responses for the exploitability separate per coordinate and are
done by 1-D grid search.

Because the payoff separates over coordinates, only the *marginal* variances
`Sigma_dd` ever appear -- which is exactly why `full_covariance` is inert on this
game, and why the off-diagonal entries of `A` get an identically zero payoff
gradient here. See `run_idealized.ClosedFormMultiDimBackend`.

Run: `python run_idealized.py configs/multidim/<name>.yaml`
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from games.examples import MultiDimDecoyWellGame
from training.gaussian import marginal_std

if TYPE_CHECKING:  # the mixture parameters this module integrates; see run_idealized.py
    from run_idealized import Params


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
