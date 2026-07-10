"""Idealized (noise-free, exact-gradient) MMD on a 2-component Gaussian mixture.

Removes every PPO/sampling confound from train.py and keeps only the game
geometry + the MMD update structure (true mirror ascent on expected payoff +
entropy bonus + periodic magnet proximal term).

The game itself is a `games.base.ZeroSumGame` from the `games/` package (e.g.
`games.examples.MultiPointGame`) -- the same game definition used by
train.py's PPO trainers. Expected payoffs of a Gaussian mixture against such a
game are available in closed form (Gaussian convolution of the double-well
shaping term the game exposes via its `.peaks`/`.width`/`.coupling` attributes,
plus closed-form Gaussian moments of its `[-1, 1]`-normalized coupling
feature), so the gradients here are *exact* -- no Monte-Carlo noise, no
critic, no PPO clipping. This isolates the question: does the MMD *vector
field* on the parametric mixture converge to the Nash, or is non-convergence
intrinsic to the geometry?

Player 0 maximizes the payoff, player 1 minimizes it (maximizes its negation).
Action is 1-D; policies are raw (unclipped) 2-Gaussian mixtures with the means
projected onto the game's `action_space` box and the log-std clipped to
[log 1e-3, 0].
"""
from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from games.base import ZeroSumGame
from games.examples import MultiPointGame

jax.config.update("jax_enable_x64", True)


class Params(NamedTuple):
    logits: jnp.ndarray   # (K,) categorical logits over the K mixture components
    means: jnp.ndarray    # (K,) each component's Gaussian mean
    log_std: jnp.ndarray  # (K,) each component's log-std


def well_expectation(means, log_std, game: ZeroSumGame):
    """E_{a~N(mu,s)}[ sum_j exp(-(a-p_j)^2/(2 w^2)) ] per component (Gaussian conv).

    Games without a double-well shaping term (e.g. `ContinuousMatchingPennies`,
    which has no `.peaks`) contribute zero here -- their payoff is the
    coupling/bilinear term alone.
    """
    peaks = getattr(game, "peaks", None)
    if peaks is None:
        return jnp.zeros_like(means)
    peaks = jnp.asarray(peaks, dtype=jnp.float64)
    width = game.width
    var = jnp.exp(2 * log_std)[:, None] + width**2  # (K, 1)
    amp = width / jnp.sqrt(var)
    bumps = amp * jnp.exp(-(means[:, None] - peaks[None, :]) ** 2 / (2 * var))
    return jnp.sum(bumps, axis=-1)  # (K,)


def _feature_geometry(game: MultiPointGame):
    """`(mid, half_range, max_order, target_moments)` for the game's coupling feature.

    `MultiPointGame`'s K-1 raw-power coupling features are `u(a)^j -
    target_j` polynomials in `u(a) = (a-mid)/half_range` (which maps the peak
    range onto exactly `[-1, 1]`) -- this extracts that parameterization so
    the mixture expectation of the feature vector can be computed once, in
    closed form.
    """
    max_order = int(game.peaks.shape[0]) - 1
    return (float(game._mid), float(game._half_range), max_order,
            jnp.asarray(game._target_moments, dtype=jnp.float64))


def _central_moment(i: int, std):
    """E[(a-mean)^i] for a~N(mean,std): 0 for odd i, std^i*(i-1)!! for even i."""
    if i % 2 == 1:
        return jnp.zeros_like(std)
    k = i // 2
    dfact = math.factorial(i) / (2**k * math.factorial(k))
    return dfact * std**i


def _component_feat_moments(mean, log_std, mid, half_range, max_order):
    """`E_{a~N(mean,std)}[u(a)^j]` for `j=1..max_order`, `u(a)=(a-mid)/half_range`."""
    std = jnp.exp(log_std)
    shift = mean - mid
    moments = [
        sum(math.comb(j, i) * shift ** (j - i) * _central_moment(i, std) for i in range(j + 1))
        / half_range**j
        for j in range(1, max_order + 1)
    ]
    return jnp.stack(moments)  # (max_order,)


def mixture_stats(p: Params, game: MultiPointGame):
    mid, half_range, max_order, target = _feature_geometry(game)
    w = jax.nn.softmax(p.logits)                                       # (K,)
    e_well = jnp.sum(w * well_expectation(p.means, p.log_std, game))   # E[D(a)]
    comp_feat = jax.vmap(_component_feat_moments, in_axes=(0, 0, None, None, None))(
        p.means, p.log_std, mid, half_range, max_order)                # (K, max_order)
    e_feat = jnp.sum(w[:, None] * comp_feat, axis=0) - target           # E[feat(a)], (max_order,)
    mean_action = jnp.sum(w * p.means)                                  # E[a]
    return w, e_well, e_feat, comp_feat


def expected_payoff(px: Params, py: Params, game: MultiPointGame):
    """E[payoff] for player 0 (px) vs player 1 (py)."""
    _, well_x, feat_x, _ = mixture_stats(px, game)
    _, well_y, feat_y, _ = mixture_stats(py, game)
    coupling = getattr(game, "coupling", 1.0)
    return well_x - well_y + coupling * jnp.sum(feat_x * feat_y)


# ---- Exploitability (NashConv), exact 1-D grid best responses -------------
def _action_grid(game: ZeroSumGame, n: int = 4001):
    space = game.action_space(0)
    return jnp.linspace(float(space.low[0]), float(space.high[0]), n)


def _well_grid(a, game: ZeroSumGame):
    peaks = getattr(game, "peaks", None)
    if peaks is None:
        return jnp.zeros_like(a)
    peaks = jnp.asarray(peaks, dtype=jnp.float64)
    return jnp.sum(jnp.exp(-(a[:, None] - peaks[None, :]) ** 2 / (2 * game.width**2)), axis=-1)


def _feat_grid(a, mid, half_range, max_order, target):
    """`u(a)^j - target_j` at every grid point `a`, shape `(len(a), max_order)`."""
    u = (a - mid) / half_range
    orders = jnp.arange(1, max_order + 1, dtype=jnp.float64)
    powers = u[:, None] ** orders[None, :]
    return powers - target[None, :]


def exploitability(px: Params, py: Params, game: MultiPointGame):
    """NashConv: how much each player could gain by best-responding. 0 at a Nash."""
    coupling = getattr(game, "coupling", 1.0)
    mid, half_range, max_order, target = _feature_geometry(game)
    _, well_x, feat_x, _ = mixture_stats(px, game)
    _, well_y, feat_y, _ = mixture_stats(py, game)
    U = expected_payoff(px, py, game)
    grid = _action_grid(game)
    Dg = _well_grid(grid, game)
    feat_g = _feat_grid(grid, mid, half_range, max_order, target)  # (N, max_order)
    # player 0 best response: max_a  D(a) - well_y + coupling*feat(a).feat_y
    br0 = jnp.max(Dg - well_y + coupling * jnp.sum(feat_g * feat_y[None, :], axis=-1))
    # player 1 best response: min_b  well_x - D(b) + coupling*feat_x.feat(b)
    br1 = jnp.min(well_x - Dg + coupling * jnp.sum(feat_x[None, :] * feat_g, axis=-1))
    return (br0 - U) + (U - br1)


# ---- MMD update -----------------------------------------------------------
def gaussian_kl(m_p, ls_p, m_q, ls_q):
    vp, vq = jnp.exp(2 * ls_p), jnp.exp(2 * ls_q)
    return jnp.sum(ls_q - ls_p + (vp + (m_p - m_q) ** 2) / (2 * vq) - 0.5)


@dataclasses.dataclass(frozen=True)
class MMDConfig:
    lr: float = 0.05                # step size (eta)
    magnet_interval: int = 200      # iterations between hard magnet snapshots
    magnet_coef: float = 0.2        # magnet temperature tau: weight on KL(. || magnet)
    entropy_coef: float = 0.0       # extra pull of the categorical head toward uniform
    steps: int = 20000
    train_std: bool = True          # let each component's std adapt
    train_means: bool = True        # if False, means are frozen at their init
    anneal_std_from: float = 0.0    # >0 enables graduated optimization: hold every
                                    # component's std >= this value early, then relax
                                    # the floor to 1e-3 over training. Starting broad
                                    # lets a mode 'feel' the far peak across the dead
                                    # plateau, which escapes the structural traps.


# ---- True MMD mirror step -------------------------------------------------
# The policy is hierarchical: pick component k ~ Cat(w), then a ~ N(mu_k, s_k).
#   * Categorical head: the exact closed-form entropic MMD proximal update on
#     the simplex (Sokota et al. 2023), which needs each component's Q-value.
#   * Gaussian head: mirror descent in the Gaussians' KL (Fisher-Rao) geometry,
#     i.e. natural gradient -- the small-step limit of the KL-proximal update.


def component_q(p: Params, opp: Params, game: MultiPointGame, sign: float):
    """Per-component expected utility q_k = E_{a~N(mu_k,s_k)}[ this player's utility ].

    For player 0 (sign +1): u(a) = D(a) + coupling*f(a)*feat_opp  (+ const).
    For player 1 (sign -1): u(b) = D(b) - coupling*f(b)*feat_opp  (+ const).
    Constants (the opponent's well) drop out under softmax; omitted.
    """
    coupling = getattr(game, "coupling", 1.0)
    mid, half_range, max_order, target = _feature_geometry(game)
    _, _, feat_opp, comp_feat = mixture_stats(opp, game)
    f_comp = jax.vmap(_component_feat_moments, in_axes=(0, 0, None, None, None))(
        p.means, p.log_std, mid, half_range, max_order) - target[None, :]  # (K, max_order)
    coupling_term = sign * coupling * jnp.einsum("j,kj->k", feat_opp, f_comp)
    return well_expectation(p.means, p.log_std, game) + coupling_term


def categorical_mirror_update(logits, q, magnet_logits, cfg: MMDConfig):
    """Closed-form MMD simplex update -> new logits.

    argmax_pi  <pi,q> - tau*KL(pi||magnet) - tau_ent*KL(pi||uniform) - (1/eta)*KL(pi||pi_t)
    has solution  log pi ~ (eta*q + eta*tau*log rho + log pi_t) / (1 + eta*tau + eta*tau_ent).
    """
    lp = jax.nn.log_softmax(logits)
    lm = jax.nn.log_softmax(magnet_logits)
    eta, tau, tau_ent = cfg.lr, cfg.magnet_coef, cfg.entropy_coef
    num = eta * q + eta * tau * lm + lp          # uniform log-prob is constant -> drops in softmax
    return num / (1.0 + eta * tau + eta * tau_ent)


def gaussian_natural_step(p: Params, opp: Params, magnet: Params, game, cfg, sign):
    """Natural-gradient (KL-geometry) step on the Gaussian means and log-stds.

    Fisher metric of N(mu, s) in (mu, rho=log s): I_mu = 1/s^2, I_rho = 2.
    So natural grad = (s^2 * d/dmu,  0.5 * d/drho).
    """
    def gauss_obj(pp):
        pay = expected_payoff(pp, opp, game) if sign > 0 else -expected_payoff(opp, pp, game)
        ent = jnp.sum(pp.log_std)  # per-component Gaussian entropy, up to a constant
        mag = gaussian_kl(pp.means, pp.log_std, magnet.means, magnet.log_std)
        return pay + cfg.entropy_coef * ent - cfg.magnet_coef * mag

    g = jax.grad(gauss_obj)(p)
    nat_means = jnp.exp(2 * p.log_std) * g.means
    nat_log_std = 0.5 * g.log_std
    return nat_means, nat_log_std


def run(game: ZeroSumGame, cfg: MMDConfig, init0: Params, init1: Params, log_every: int = 200):
    """Simultaneous MMD ascent for both players. Returns (final0, final1, history)."""
    p0, p1 = init0, init1
    m0, m1 = init0, init1  # magnet snapshots

    action_low = float(game.action_space(0).low[0])
    action_high = float(game.action_space(0).high[0])

    def _std_floor(t):
        """log-std lower bound: anneals from log(anneal_std_from) to log(1e-3) over training."""
        base = jnp.log(1e-3)
        if cfg.anneal_std_from <= 0.0:
            return base
        frac = t / max(cfg.steps - 1, 1)
        return (1.0 - frac) * jnp.log(cfg.anneal_std_from) + frac * base

    def _clip(means, log_std, t):
        return (jnp.clip(means, action_low, action_high),
                jnp.clip(log_std, _std_floor(t), jnp.log(1.0)))

    @jax.jit
    def step(p0, p1, m0, m1, t):
        def apply(p, opp, magnet, sign):
            logits = categorical_mirror_update(p.logits, component_q(p, opp, game, sign),
                                               magnet.logits, cfg)
            nat_means, nat_log_std = gaussian_natural_step(p, opp, magnet, game, cfg, sign)
            means = p.means + (cfg.lr * nat_means if cfg.train_means else 0.0)
            log_std = p.log_std + (cfg.lr * nat_log_std if cfg.train_std else 0.0)
            means, log_std = _clip(means, log_std, t)
            return Params(logits=logits, means=means, log_std=log_std)
        return apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)

    history = []
    for t in range(cfg.steps):
        p0, p1 = step(p0, p1, m0, m1, jnp.asarray(t, dtype=jnp.float64))
        if (t + 1) % cfg.magnet_interval == 0:
            m0, m1 = p0, p1
        if t % log_every == 0 or t == cfg.steps - 1:
            w0 = jax.nn.softmax(p0.logits)
            w1 = jax.nn.softmax(p1.logits)
            history.append({
                "t": t,
                "expl": float(exploitability(p0, p1, game)),
                "means0": [float(x) for x in p0.means],
                "means1": [float(x) for x in p1.means],
                "std0": [float(x) for x in jnp.exp(p0.log_std)],
                "std1": [float(x) for x in jnp.exp(p1.log_std)],
                "w0": [float(x) for x in w0],
                "w1": [float(x) for x in w1],
                "mean_action0": float(jnp.sum(w0 * p0.means)),
                "spread0": float(jnp.max(p0.means) - jnp.min(p0.means)),
            })
    return p0, p1, history


def make_init(means, logits=None, log_std: float = float(jnp.log(0.1))) -> Params:
    """`means`/`logits`/`log_std` may be scalars-per-component or a fixed scalar log_std for all K."""
    means = jnp.asarray(means, dtype=jnp.float64)
    k = means.shape[0]
    if logits is None:
        logits = jnp.zeros(k, dtype=jnp.float64)
    else:
        logits = jnp.asarray(logits, dtype=jnp.float64)
    return Params(logits=logits, means=means, log_std=jnp.full((k,), log_std, dtype=jnp.float64))


def _tail_expl(history, frac: float = 0.3) -> float:
    import numpy as np
    e = np.array([h["expl"] for h in history])
    return float(e[int(len(e) * (1 - frac)):].mean())

TWO_POINT = MultiPointGame(peaks=(0.0, 1.0), width=0.1, coupling=1.0)
THREE_POINT = MultiPointGame(peaks=(-1.0, 0.0, 1.0), width=0.08, coupling=1.0)


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    game: MultiPointGame
    init0: Params
    init1: Params
    note: str


SCENARIOS: list[Scenario] = [
    Scenario("both modes in the LEFT basin", TWO_POINT,
             make_init([-0.05, 0.15]), make_init([-0.05, 0.15]),
             "both players cover only peak 0; peak 1 is never reached"),
    Scenario("players stuck one-sided", TWO_POINT,
             make_init([0.05, 0.2]), make_init([0.8, 0.95]),
             "p0 covers peak 0, p1 covers peak 1; neither can reach the other"),
    # -- Counterexamples with num_components > |Nash support| ---------------
    Scenario("exact symmetric center, K=3 > support=2", TWO_POINT,
             make_init([0.5, 0.5, 0.5]), make_init([0.5, 0.5, 0.5]),
             "all 3 components start identical at the exact midpoint between the "
             "two peaks: this is a genuine fixed point of the vector field (means, "
             "logits, and the opponent's feature are all exactly symmetric, so every "
             "gradient the update touches is exactly zero) -- extra capacity can't "
             "help because nothing ever breaks the tie between components. Robust "
             "to any lr/magnet/entropy_coef (it's an algebraic identity, not a "
             "tuning issue); only broken by literally perturbing the init."),
    Scenario("weight-starvation freeze, K=3 > support=2", TWO_POINT,
             make_init([-0.08, -0.02, 0.1]), make_init([-0.05, 0.0, 0.12]),
             "3 components all start near the SAME peak (0.0); one one grabs "
             "nearly all the weight, the other two get starved toward w~0 and then "
             "freeze in place (their mean's natural gradient is itself weighted by "
             "w, so low weight suppresses its own escape) -- peak 1.0 is never "
             "found despite 50% spare capacity. Immune to lr/magnet_coef retuning; "
             "only fixed by MMDConfig(anneal_std_from=...), the deliberate escape "
             "hatch already in this file."),
    Scenario("missing-middle-peak trap, K=4 > support=3", THREE_POINT,
             make_init([-0.95, -1.05, -0.9, -1.1]), make_init([-1.0, -0.98, -1.02, -0.9]),
             "3-peak MultiPointGame, 4 components all initialized near the SAME "
             "(leftmost) peak. Both players end up splitting mass over only 2 of "
             "the 3 peaks (the middle peak at 0.0 is never colonized) and plateau "
             "at NashConv ~1.3-1.6. Unlike the K=3 (2-peak) trap above, this "
             "one survives anneal_std_from up to 1.0, lr up to 0.1, and magnet_coef "
             "down to 0.05, over 15k steps -- i.e. it is NOT rescued by the "
             "existing escape mechanism. Structural: once >=2 components fall into "
             "the same basin, the deterministic (noise-free) dynamics have no "
             "mechanism to redistribute one of them to an uncolonized peak."),
]

_PASS = 0.1  # tail-exploitability below this counts as reaching the Nash


def _fmt_player(w, mu, sd) -> str:
    comps = " ".join(f"{wi:.2f}@{mi:+.2f}(sd{si:.2f})" for wi, mi, si in zip(w, mu, sd))
    return f"[{comps}]"


def trace(cfg: MMDConfig, scenario: Scenario, rows: int = 16) -> None:
    """Run a scenario and print how both players' strategies evolve over iterations."""
    print(f"\n### {scenario.name} -- {scenario.note}")
    print(f"CONFIG: lr={cfg.lr} magnet_coef={cfg.magnet_coef} "
          f"entropy_coef={cfg.entropy_coef} steps={cfg.steps} anneal_std_from={cfg.anneal_std_from}")
    _, _, h = run(scenario.game, cfg, scenario.init0, scenario.init1)
    n = len(h)
    idx = sorted({round(i * (n - 1) / (rows - 1)) for i in range(rows)})
    for i in idx:
        e = h[i]
        print(f"  t={e['t']:6d}  expl={e['expl']:6.3f}   "
              f"P0 {_fmt_player(e['w0'], e['means0'], e['std0'])}   "
              f"P1 {_fmt_player(e['w1'], e['means1'], e['std1'])}")
    tail = _tail_expl(h)
    print(f"  --> tail_expl={tail:.3f}  ({'PASS (Nash)' if tail < _PASS else 'FAIL -- stuck off equilibrium'})")

