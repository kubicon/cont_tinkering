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


class AsymmetricWellGame(ZeroSumGame):
    """Unique Nash, but the two players do *not* share one landscape `phi`.

    payoff(a1, a2) = -||a1||^2 + ||a2||^4 + coupling * a1 . a2

    Read against the decomposition `U = W(pi_0) - W(pi_1) + C`: the self-terms
    are `-||a1||^2` for player 0 and `-(-||a2||^4)` for player 1, i.e. player 0
    sits in a quadratic well and player 1 in a quartic one. The cross partial
    is the constant `coupling * I`, which pins any bilinear `C` to exactly
    `coupling * a1 . a2`, leaving `-||a1||^2 + ||a2||^4 = phi(a1) - phi(a2)` --
    impossible, since it would need `phi(z) = -||z||^2` and `phi(z) = -||z||^4`
    at once. So this game is outside the "one shared well + bilinear coupling"
    class that `MultiPointGame` lives in.

    It is still concave in `a1` and convex in `a2`, hence its pseudo-gradient
    *is* monotone (take `phi = 0`, `C = payoff`). That is the point: the
    restrictive clause is the shared well, not monotonicity.

    Unique Nash at the origin: stationarity gives `a1 = coupling*a2/2` and
    `4||a2||^2 a2 + coupling*a1 = 0`, so `a2 (4||a2||^2 + coupling^2/2) = 0`,
    forcing `a2 = a1 = 0`. Strict concavity/convexity rules out mixing, so the
    unique Nash of the mixed extension is the pair of point masses at 0.
    """

    def __init__(self, dim: int = 1, coupling: float = 1.0, bound: float = 3.0):
        self.coupling = coupling
        self._space = box(-bound * jnp.ones(dim), bound * jnp.ones(dim))

    def action_space(self, player: int) -> ActionSpace:
        return self._space

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        wells = -jnp.sum(action_1**2) + jnp.sum(action_2**2) ** 2
        return wells + self.coupling * jnp.dot(action_1, action_2)


class CurvaturePumpGame(ZeroSumGame):
    """Unique Nash, genuinely non-monotone: the opponent controls your curvature.

    payoff(a1, a2) = -||a1||^4 + pump * ||a1||^2 ||a2||^2 + ||a2||^4

    Player 0's own Hessian is `-(4||a1||^2 I + 8 a1 a1^T) + 2*pump*||a2||^2 I`,
    which is *positive* definite near `a1 = 0` for any `a2 != 0`: player 0's
    payoff is convex (not concave) in its own action wherever the opponent
    moves off the origin, so the pseudo-gradient is not monotone. Nor can a
    well fix it: `C = payoff - phi(a1) + phi(a2)` is concave in `a1` only if
    `phi` dominates `pump*||a2||^2 ||a1||^2` for *every* `a2`, so on an
    unbounded action space no finite `phi` exists (midpoint convexity at
    `{-e, 0, e}` fails as `||a2|| -> inf`); on the box here it exists only by
    making the well strongly convex with modulus `>= 2*pump*bound^2`, i.e. a
    landscape that dwarfs the game itself.

    Uniqueness survives anyway, because player 1 never goes where the
    non-concavity lives: against *any* mixture `mu` of player 0, player 1
    minimizes `E||a2||^4 + pump * E_mu||a1||^2 * E||a2||^2`, a sum of two
    nonnegative terms, uniquely minimized by the point mass at 0; and against
    that, player 0 uniquely maximizes `-E||a1||^4` at the point mass at 0. So
    the mixed extension has exactly one Nash: `(delta_0, delta_0)`.
    """

    def __init__(self, dim: int = 1, pump: float = 4.0, bound: float = 2.0):
        self.pump = pump
        self._space = box(-bound * jnp.ones(dim), bound * jnp.ones(dim))

    def action_space(self, player: int) -> ActionSpace:
        return self._space

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        sq_1, sq_2 = jnp.sum(action_1**2), jnp.sum(action_2**2)
        return -sq_1**2 + self.pump * sq_1 * sq_2 + sq_2**2


class ForsakenGame(ZeroSumGame):
    """The "Forsaken" saddle (Hsieh et al., 2021), as the control case.

    payoff(a1, a2) = a2 * (a1 - 0.45) + phi(a2) - phi(a1),
        phi(z) = z^2/4 - z^4/2 + z^6/6

    This one *is* inside the `W(pi_0) - W(pi_1) + C` class -- shared landscape
    `-phi`, coupling `a1*a2 - 0.45*a2`, which is bilinear-plus-linear and hence
    skew, hence monotone. It is included as the control that shows class
    membership buys nothing dynamically: `phi` is non-convex, so the *game* is
    still non-monotone, and it is the standard example of gradient-based
    min-max cycling around a limit cycle instead of converging.

    On this box the unique Nash (checked by LP on a fine grid) is *mixed*, with
    both players supported on the two well minima `a ~= +-1.31`; the interior
    stationary point near `(0.08, 0.40)` from the paper is only a local saddle.
    Tabular MMD on it is step-size sensitive -- it cycles at `eta=0.1,
    alpha=0.05` (exploitability stuck around 2.3) and converges at `eta=0.05,
    alpha=0.1` -- which is exactly the sensitivity this game exists to expose.
    """

    def __init__(self, bound: float = 1.5):
        self._space = box(-bound * jnp.ones(1), bound * jnp.ones(1))

    def action_space(self, player: int) -> ActionSpace:
        return self._space

    @staticmethod
    def _phi(z: chex.Array) -> chex.Array:
        return z**2 / 4 - z**4 / 2 + z**6 / 6

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        a1, a2 = action_1[..., 0], action_2[..., 0]
        return a2 * (a1 - 0.45) + self._phi(a2) - self._phi(a1)


class DecoyWellGame(ZeroSumGame):
    """`num_components == |Nash support|` is enough capacity -- and MMD still cannot get there.

    Same skeleton as `MultiPointGame` (shared well + moment-matching coupling,
    so it is squarely inside the C1/C2/C4 class of `ROUTE_A.md`), with one
    addition: the well carries extra **decoy** bumps that are *strictly lower*
    than the true peaks but far **broader**.

    payoff(a1, a2) = D(a1) - D(a2) + coupling * sum_j feat_j(a1) * feat_j(a2)

        D(a) = sum_k peak_height * exp(-(a - peak_k)^2 / (2 * peak_width^2))
             + sum_d height_d   * exp(-(a - center_d)^2 / (2 * width_d^2))

    The coupling features are built from the **true peaks only** (identical to
    `MultiPointGame`), so the equilibrium analysis is unchanged: at a Nash the
    opponent matches all K-1 moments, the coupling term vanishes for *every*
    action, and each player is left maximizing `D` alone. Since every decoy is
    strictly lower than the peaks, the argmax of `D` is exactly the peaks, so
    the unique Nash is still the K-point mixture on `peaks` with `weights`.
    **The decoys carry zero equilibrium mass.** `num_components = K` suffices to
    represent it exactly.

    Why it is nevertheless a counterexample -- *mass beats height under
    smoothing*. A Gaussian bump of height `h` and width `w`, convolved with
    `N(0, s^2)`, has height `h * w / sqrt(w^2 + s^2)`: for `s >> w` that is
    `~ (h*w)/s`, i.e. it decays in proportion to the bump's **mass** `h*w`, not
    its height. With the defaults (peaks: h=1.0, w=0.05, mass 0.05; decoy:
    h=0.7, w=0.45, mass 0.315) the decoy has ~6x the mass of a peak, so:

      - at `s = 0`   the global maxima of `D` are the true peaks (1.06 vs 0.70);
      - at `s >~ 0.1` the global maximum of the *smoothed* `D` is the **decoy**;
      - at `s >~ 0.5` the smoothed `D` is **unimodal** -- one basin, the decoy's.

    This inverts the ranking exactly where graduated optimization relies on it.
    A broad-initialized (or std-annealed) component feels the smoothed
    landscape, so it is pulled to the *decoy*; as the std anneals down, the
    decoy remains a strict local maximum of `D`, and the peaks are so narrow
    that the region between them is numerically flat (a component at `a = 0.5`
    feels a peak-gradient of order `exp(-50)`). So the component has no gradient
    with which to leave, and no gradient telling it where to go. The std-anneal
    escape hatch that rescues `idealized_mmd.py`'s other traps
    (`MMDConfig.anneal_std_from`) here does the opposite: it *delivers* the
    components into the trap and then locks the door.

    The trap is an exact, stable fixed point, not a slow drift. With the default
    symmetric setup (`peaks=(-1,1)`, `weights=(0.5,0.5)`) the target first moment
    is 0, so a player sitting entirely on the decoy at `a = 0` has feature
    `feat(0) = u(0) - 0 = 0`: the coupling term vanishes identically, each player's
    effective landscape collapses to `D` alone, and `a = 0` is a strict local
    maximum of `D` with zero gradient. Both players collapsed onto the decoy is
    therefore a fixed point of the MMD vector field -- and it is *not* a Nash:
    each player could deviate to a peak and gain `1.06 - 0.70`, for a total
    exploitability of ~0.72.

    The default `action_margin=2.0` puts the box at `[-3, 3]`, which is
    deliberate: `training.mixture._spread_bias_init` starts `K=2` components at
    `+-1.5` -- inside the dead zone, where the only gradient they can feel is the
    decoy's. The natural initialization walks straight in.

    Knobs to make the trap milder/harsher: raise `peak_width` (more peak mass ->
    smoothing eventually finds them), lower the decoy `height`/`width` (less
    mass), or shrink `action_margin`.
    """

    def __init__(
        self,
        peaks: tuple[float, ...] = (-1.0, 1.0),
        weights: tuple[float, ...] | None = None,
        peak_width: float = 0.05,
        peak_height: float = 1.0,
        decoys: tuple[tuple[float, float, float], ...] = ((0.0, 0.7, 0.45),),
        coupling: float = 1.0,
        action_margin: float = 2.0,
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
        for center, height, width in decoys:
            if height >= peak_height:
                raise ValueError(
                    f"decoy at {center} has height {height} >= peak_height {peak_height}; decoys must be "
                    "strictly lower than the peaks or they would carry equilibrium mass"
                )
            if width <= 0.0:
                raise ValueError(f"decoy at {center} has non-positive width {width}")

        self.peaks = jnp.asarray(peaks, dtype=jnp.float32)
        self.weights = weights_arr
        self.peak_width = peak_width
        self.peak_height = peak_height
        self.decoys = tuple(decoys)
        self.coupling = coupling

        # Full bump set (true peaks + decoys) defining the well `D`. Exposed as
        # arrays so the closed-form Gaussian convolution in `idealized_mmd` can
        # handle per-bump heights/widths.
        self.bump_centers = jnp.asarray(
            [float(p) for p in peaks] + [float(c) for c, _, _ in decoys], dtype=jnp.float32
        )
        self.bump_heights = jnp.asarray(
            [peak_height] * num_peaks + [float(h) for _, h, _ in decoys], dtype=jnp.float32
        )
        self.bump_widths = jnp.asarray(
            [peak_width] * num_peaks + [float(w) for _, _, w in decoys], dtype=jnp.float32
        )

        # Coupling geometry: built from the TRUE PEAKS ONLY, exactly as in
        # `MultiPointGame` -- the decoys are invisible to the feature map, so the
        # equilibrium is unchanged.
        peak_lo, peak_hi = jnp.min(self.peaks), jnp.max(self.peaks)
        self._mid = (peak_lo + peak_hi) / 2
        self._half_range = (peak_hi - peak_lo) / 2
        self._orders = jnp.arange(1, num_peaks, dtype=jnp.float32)

        u_peaks = self._u(self.peaks)
        peak_moments = u_peaks[:, None] ** self._orders[None, :]
        self._target_moments = jnp.sum(self.weights[:, None] * peak_moments, axis=0)

        self._space = box(
            (peak_lo - action_margin) * jnp.ones(1), (peak_hi + action_margin) * jnp.ones(1)
        )

    def action_space(self, player: int) -> ActionSpace:
        return self._space

    def _u(self, a: chex.Array) -> chex.Array:
        return (a - self._mid) / self._half_range

    def _well(self, action: chex.Array) -> chex.Array:
        x = action[..., 0]
        bumps = self.bump_heights * jnp.exp(
            -jnp.square(x[..., None] - self.bump_centers) / (2 * self.bump_widths**2)
        )
        return jnp.sum(bumps, axis=-1)

    def _feat(self, action: chex.Array) -> chex.Array:
        u = self._u(action[..., 0])
        powers = u[..., None] ** self._orders
        return powers - self._target_moments

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        coupling = jnp.sum(self._feat(action_1) * self._feat(action_2), axis=-1)
        return self._well(action_1) - self._well(action_2) + self.coupling * coupling


class MultiDimDecoyWellGame(ZeroSumGame):
    """`DecoyWellGame` lifted to `dim`-dimensional actions as a separable product.

    The action is now a vector `a in R^dim` rather than a scalar, and the
    payoff is the *sum* of `dim` independent copies of the 1-D `DecoyWellGame`,
    one per coordinate (all sharing the same peaks/decoys/coupling):

        payoff(a1, a2) = sum_{d=1}^{dim} [ D(a1_d) - D(a2_d)
                                           + coupling * sum_j feat_j(a1_d) feat_j(a2_d) ]

        D(x)      = sum_k peak_height * exp(-(x - peak_k)^2 / (2 peak_width^2))
                  + sum_m height_m   * exp(-(x - center_m)^2 / (2 width_m^2))
        feat_j(x) = u(x)^j - E_{k~weights}[u(peak_k)^j],   u(x) = (x - mid)/half_range

    Because the payoff is a sum over coordinates and the two players' coordinates
    are chosen independently, the game **separates**: it is `dim` non-interacting
    1-D `DecoyWellGame`s stacked together. Everything the 1-D analysis establishes
    therefore holds coordinate-wise:

      - each coordinate has the same unique 1-D Nash -- the K-point mixture on
        `peaks` with `weights`, with the decoys carrying zero mass -- and the
        joint Nash is the **product** of these per-coordinate mixtures;
      - the decoy trap (mass beats height under smoothing; see `DecoyWellGame`)
        fires independently in every coordinate.

    This is what "N components in each dimension" means for the policy. The joint
    equilibrium is a product distribution whose *marginal in each coordinate* is
    the K-point peak mixture. A policy that factorizes across coordinates (an
    independent K-component mixture per axis) represents it exactly with `K`
    components per dimension; a single joint categorical mixture over R^dim would
    instead need one component per *grid corner*, i.e. `K^dim` of them, to place
    mass on the product of the per-axis supports. The separable design is
    deliberately the case where per-axis capacity `K` is enough and joint-mixture
    capacity blows up combinatorially.

    `dim == 1` reproduces `DecoyWellGame` exactly (same defaults, same box).
    """

    def __init__(
        self,
        dim: int = 2,
        peaks: tuple[float, ...] = (-1.0, 1.0),
        weights: tuple[float, ...] | None = None,
        peak_width: float = 0.05,
        peak_height: float = 1.0,
        decoys: tuple[tuple[float, float, float], ...] = ((0.0, 0.7, 0.45),),
        coupling: float = 1.0,
        action_margin: float = 2.0,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
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
        for center, height, width in decoys:
            if height >= peak_height:
                raise ValueError(
                    f"decoy at {center} has height {height} >= peak_height {peak_height}; decoys must be "
                    "strictly lower than the peaks or they would carry equilibrium mass"
                )
            if width <= 0.0:
                raise ValueError(f"decoy at {center} has non-positive width {width}")

        self.dim = dim
        self.peaks = jnp.asarray(peaks, dtype=jnp.float32)
        self.weights = weights_arr
        self.peak_width = peak_width
        self.peak_height = peak_height
        self.decoys = tuple(decoys)
        self.coupling = coupling

        # Per-axis bump set (identical in every coordinate). Same layout as
        # `DecoyWellGame`: true peaks first, then decoys.
        self.bump_centers = jnp.asarray(
            [float(p) for p in peaks] + [float(c) for c, _, _ in decoys], dtype=jnp.float32
        )
        self.bump_heights = jnp.asarray(
            [peak_height] * num_peaks + [float(h) for _, h, _ in decoys], dtype=jnp.float32
        )
        self.bump_widths = jnp.asarray(
            [peak_width] * num_peaks + [float(w) for _, _, w in decoys], dtype=jnp.float32
        )

        # Coupling geometry: true peaks only, exactly as in `MultiPointGame` /
        # `DecoyWellGame`, applied identically to every coordinate.
        peak_lo, peak_hi = jnp.min(self.peaks), jnp.max(self.peaks)
        self._mid = (peak_lo + peak_hi) / 2
        self._half_range = (peak_hi - peak_lo) / 2
        self._orders = jnp.arange(1, num_peaks, dtype=jnp.float32)

        u_peaks = self._u(self.peaks)
        peak_moments = u_peaks[:, None] ** self._orders[None, :]
        self._target_moments = jnp.sum(self.weights[:, None] * peak_moments, axis=0)

        self._space = box(
            (peak_lo - action_margin) * jnp.ones(dim), (peak_hi + action_margin) * jnp.ones(dim)
        )

    def action_space(self, player: int) -> ActionSpace:
        return self._space

    def _u(self, a: chex.Array) -> chex.Array:
        return (a - self._mid) / self._half_range

    def _well(self, action: chex.Array) -> chex.Array:
        """`sum_d D(a_d)` -- the per-axis wells summed over coordinates."""
        # action: (..., dim); broadcast against (num_bumps,) on a new trailing axis.
        bumps = self.bump_heights * jnp.exp(
            -jnp.square(action[..., None] - self.bump_centers) / (2 * self.bump_widths**2)
        )  # (..., dim, num_bumps)
        return jnp.sum(bumps, axis=(-2, -1))

    def _feat(self, action: chex.Array) -> chex.Array:
        """Per-coordinate feature vector, shape `(..., dim, K-1)`."""
        u = self._u(action)  # (..., dim)
        powers = u[..., None] ** self._orders  # (..., dim, K-1)
        return powers - self._target_moments

    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        coupling = jnp.sum(self._feat(action_1) * self._feat(action_2), axis=(-2, -1))
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
