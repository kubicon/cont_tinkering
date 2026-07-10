"""Concrete `ZeroSumGame`s, mainly to exercise/demonstrate the base class."""

from __future__ import annotations

import chex
import jax.numpy as jnp

from .base import ZeroSumGame
from .spaces import ActionSpace, box, simplex


class QuadraticZeroSumGame(ZeroSumGame):
    """payoff(a1, a2) = -||a1||^2 + ||a2||^2 + a1^T C a2, concave in a1, convex in a2.

    Has a unique interior saddle point; with `coupling` small enough relative
    to `bound` it sits at the origin, which makes this a convenient sanity
    check for best-response / exploitability computations.
    """

    def __init__(self, coupling: chex.Array, bound: float = 3.0):
        dim_1, dim_2 = coupling.shape
        self.coupling = coupling
        self._spaces = {
            0: box(-bound * jnp.ones(dim_1), bound * jnp.ones(dim_1)),
            1: box(-bound * jnp.ones(dim_2), bound * jnp.ones(dim_2)),
        }

    def action_space(self, player: int) -> ActionSpace:
        return self._spaces[player]

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        quadratic = -jnp.sum(action_1**2) + jnp.sum(action_2**2)
        bilinear = action_1 @ self.coupling @ action_2
        return quadratic + bilinear


class ContinuousBlottoGame(ZeroSumGame):
    """Colonel-Blotto-style resource allocation game.

    Each player distributes a fixed budget over `front_values.shape[0]`
    fronts (a `SimplexSpace`); whoever commits more to a front wins its
    value there. The win indicator is smoothed with `tanh` so the payoff is
    differentiable and gradient-based best responses apply.
    """

    def __init__(self, front_values: chex.Array, budget: float = 1.0, sharpness: float = 10.0):
        self.front_values = front_values
        self.sharpness = sharpness
        self._space = simplex(front_values.shape[0], total=budget)

    def action_space(self, player: int) -> ActionSpace:
        return self._space

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        margin = action_1 - action_2
        return jnp.sum(self.front_values * jnp.tanh(self.sharpness * margin))


class MultiPointGame(ZeroSumGame):
    """Matching pennies whose unique Nash has *isolated* K-point support, K>=2.

    `ContinuousMatchingPennies` has a degenerate equilibrium: its payoff is
    bilinear, so the expected payoff depends only on each player's *mean*
    action and every strategy with `E[a] = 1/2` is an equally-good Nash. The
    Gaussian mixture head therefore has no reason to place its modes at any
    particular points -- the mode locations are unidentifiable and drift.

    This game removes that degeneracy. The payoff adds a double-well shaping
    term `D` (a sum of Gaussian bumps peaked at `peaks`) that each player
    *individually* wants to sit on top of, and keeps a matching-pennies-style
    coupling only as a tie-breaker between the equally-tall peaks:

    payoff(a1, a2) = D(a1) - D(a2) + coupling * sum_{j=1}^{K-1} feat_j(a1) * feat_j(a2)

        D(a)      = sum_k exp(-(a - peak_k)^2 / (2*width^2))
        u(a)      = (a - mid) / half_range,  mid/half_range from min/max(peaks)
                    (so `u` maps the peak range onto exactly `[-1, 1]`)
        feat_j(a) = u(a)^j - E_{k~weights}[u(peak_k)^j],   j = 1, ..., K-1

    With K peaks, a categorical distribution over them has K-1 free
    parameters, so matching one feature isn't enough to pin down an
    arbitrary target -- this uses all K-1 raw moments of the peaks (in the
    same normalized coordinate `u`). Because the peaks are distinct, the
    K-1 moments determine a distribution over K support points uniquely
    (a Vandermonde system in `u(peak_k)`), so at the unique Nash each
    player's marginal distribution over the peaks equals `weights` exactly:

      - matching all K-1 moments makes every `feat_j` average to zero under
        the opponent's Nash mixture, so the coupling term vanishes for
        *every* action, leaving `u(a) = D(a) - const`, strictly maximized
        only at the (equally tall) peaks -- pinning the support to `peaks`;
      - any other distribution over the peaks -- including any pure peak --
        leaves at least one moment mismatched, so the coupling term is
        nonzero somewhere and a player benefits from shifting weight toward
        (or away from) specific peaks, exactly the matching-pennies
        incentive to not settle on the wrong mix.

    Raw powers of `u` get correlated/ill-conditioned for many peaks (large
    K); this is fine for the small K (a handful of peaks) this toy is meant
    for, but isn't a numerically robust design for large K.
    """

    def __init__(
        self,
        peaks: tuple[float, ...],
        weights: tuple[float, ...] | None = None,
        width: float = 0.1,
        coupling: float = 1.0,
        action_margin: float = 1.0,
    ):
        num_peaks = len(peaks)
        if num_peaks < 2:
            raise ValueError(f"need at least 2 peaks, got {num_peaks}")
        if weights is None:
            weights = (1.0 / num_peaks,) * num_peaks
        if len(weights) != num_peaks:
            raise ValueError(f"weights has {len(weights)} entries, expected {num_peaks} (one per peak)")
        weights_arr = jnp.asarray(weights, dtype=jnp.float32)
        if not jnp.allclose(jnp.sum(weights_arr), 1.0, atol=1e-5):
            raise ValueError(f"weights must sum to 1, got {float(jnp.sum(weights_arr))}")

        self.peaks = jnp.asarray(peaks, dtype=jnp.float32)
        self.weights = weights_arr
        self.width = width
        self.coupling = coupling

        peak_lo, peak_hi = jnp.min(self.peaks), jnp.max(self.peaks)
        self._mid = (peak_lo + peak_hi) / 2
        self._half_range = (peak_hi - peak_lo) / 2
        self._orders = jnp.arange(1, num_peaks, dtype=jnp.float32)  # (K-1,), exponents 1..K-1

        u_peaks = self._u(self.peaks)
        peak_moments = u_peaks[:, None] ** self._orders[None, :]              # (K, K-1)
        self._target_moments = jnp.sum(self.weights[:, None] * peak_moments, axis=0)  # (K-1,)

        self._space = box(
            (peak_lo - action_margin) * jnp.ones(1), (peak_hi + action_margin) * jnp.ones(1)
        )

    def action_space(self, player: int) -> ActionSpace:
        return self._space

    def _u(self, a: chex.Array) -> chex.Array:
        return (a - self._mid) / self._half_range

    def _well(self, action: chex.Array) -> chex.Array:
        x = action[..., 0]
        bumps = jnp.exp(-jnp.square(x[..., None] - self.peaks) / (2 * self.width**2))
        return jnp.sum(bumps, axis=-1)

    def _feat(self, action: chex.Array) -> chex.Array:
        """`(feat_1(a), ..., feat_{K-1}(a))`."""
        u = self._u(action[..., 0])
        powers = u[..., None] ** self._orders  # (..., K-1)
        return powers - self._target_moments

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        coupling = jnp.sum(self._feat(action_1) * self._feat(action_2), axis=-1)
        return self._well(action_1) - self._well(action_2) + self.coupling * coupling


class ContinuousMatchingPennies(ZeroSumGame):
    """Matching pennies embedded in a continuous action: `a1, a2 in [0, 1]`.

    payoff(a1, a2) = (2*a1 - 1) * (2*a2 - 1), bilinear in a1 and a2. Because
    it's linear in each action for any fixed opponent action, every best
    response sits at an endpoint (0 or 1) -- never in the interior -- so
    there's no pure Nash equilibrium, exactly as in matching pennies. The
    unique equilibrium is each player playing 0 and 1 with probability 1/2
    each (value 0), i.e. a mixed strategy with finite (2-point) support
    despite the action space being a continuum. A unimodal policy (e.g. a
    diagonal Gaussian) cannot represent this equilibrium exactly: it can
    only spread mass around the midpoint, which is precisely where the true
    equilibrium places zero mass.
    """

    def __init__(self):
        self._space = box(jnp.zeros(1), jnp.ones(1))

    def action_space(self, player: int) -> ActionSpace:
        return self._space

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        return jnp.sum((2 * action_1 - 1) * (2 * action_2 - 1))
