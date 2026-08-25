"""Isolated Gaussian-head MMD: does the magnet (KL proximal) term ever help?

Single-component (K=1) diagonal-Gaussian policies, so the categorical head is
trivial (softmax of one logit = 1) and the *only* moving parts are the Gaussian
means/log-stds. This strips away the confound that `theory/exp6` always carried:
its games all had a curvature-dominated well, so the Gaussian head was already
strongly concave in its means and the magnet was (correctly) found redundant.

Here we control that curvature directly. Game family (1-D, box [-B, B]):

    U(mu0, s0, mu1, s1) = W(mu0, s0) - W(mu1, s1) + c * mu0 * mu1
    W(mu, s)            = -0.5 * kappa * (mu^2 + s^2)     # E[-0.5 kappa a^2]

  * kappa large  -> each player's own payoff is strongly CONCAVE in its mean
                    (Hessian -kappa << 0): the exp6 regime, curvature carries
                    local monotonicity, magnet predicted redundant.
  * kappa = 0    -> pure bilinear "matching pennies in the mean": interior
                    saddle at the origin, payoff std-INDEPENDENT, own-Hessian
                    exactly 0. Discrete natural-gradient descent-ascent has NO
                    own curvature to contract with and rotates outward.

The MMD update is the same as `idealized_mmd.gaussian_natural_step`: a
Fisher-Rao (natural-gradient) ascent on `pay + entropy_coef*ent - tau*KL(.||magnet)`,
means scaled by s^2, log-stds by 1/2, with a periodic hard magnet snapshot.
`tau` is the Gaussian-head magnet weight -- the knob under test.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class G(NamedTuple):
    """One diagonal Gaussian (single component)."""
    mean: jnp.ndarray      # scalar
    log_std: jnp.ndarray   # scalar


@dataclasses.dataclass(frozen=True)
class Cfg:
    kappa: float = 0.0        # own-well curvature (0 = pure bilinear)
    coupling: float = 1.0     # c
    lr: float = 0.05
    tau: float = 0.0          # Gaussian-head magnet weight  <-- the knob
    entropy_coef: float = 0.0
    magnet_interval: int = 200
    steps: int = 20000
    bound: float = 3.0
    train_std: bool = True
    std_floor: float = 1e-3
    std_ceil: float = 3.0


def W(mean, log_std, kappa):
    return -0.5 * kappa * (mean ** 2 + jnp.exp(2 * log_std))


def expected_payoff(p0: G, p1: G, cfg: Cfg):
    """E[payoff] for player 0 (maximizer) vs player 1 (minimizer)."""
    return (W(p0.mean, p0.log_std, cfg.kappa)
            - W(p1.mean, p1.log_std, cfg.kappa)
            + cfg.coupling * p0.mean * p1.mean)


def gaussian_kl(m_p, ls_p, m_q, ls_q):
    vp, vq = jnp.exp(2 * ls_p), jnp.exp(2 * ls_q)
    return ls_q - ls_p + (vp + (m_p - m_q) ** 2) / (2 * vq) - 0.5


def exploitability(p0: G, p1: G, cfg: Cfg):
    """NashConv over the box. Nash is the pair of point masses at 0 (value 0).

    br0 = max_a  W(a) - W(mu1) + c*a*mu1 ; br1 = min_b W(mu0) - W(b) + c*mu0*b.
    W here is evaluated at std->0 (a best response is a pure action).
    """
    B = cfg.bound
    grid = jnp.linspace(-B, B, 4001)
    Wg = -0.5 * cfg.kappa * grid ** 2
    W0 = -0.5 * cfg.kappa * p0.mean ** 2
    W1 = -0.5 * cfg.kappa * p1.mean ** 2
    U = expected_payoff(p0, p1, cfg)
    br0 = jnp.max(Wg - W1 + cfg.coupling * grid * p1.mean)
    br1 = jnp.min(W0 - Wg + cfg.coupling * p0.mean * grid)
    return float((br0 - U) + (U - br1))


def _step(p0, p1, m0, m1, cfg: Cfg):
    def apply(p, opp, magnet, sign):
        def obj(pp):
            pay = expected_payoff(pp, opp, cfg) if sign > 0 else -expected_payoff(opp, pp, cfg)
            ent = cfg.entropy_coef * pp.log_std
            mag = cfg.tau * gaussian_kl(pp.mean, pp.log_std, magnet.mean, magnet.log_std)
            return pay + ent - mag

        g = jax.grad(obj)(p)
        mean = jnp.clip(p.mean + cfg.lr * jnp.exp(2 * p.log_std) * g.mean, -cfg.bound, cfg.bound)
        if cfg.train_std:
            log_std = jnp.clip(p.log_std + cfg.lr * 0.5 * g.log_std,
                               jnp.log(cfg.std_floor), jnp.log(cfg.std_ceil))
        else:
            log_std = p.log_std
        return G(mean, log_std)

    return apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)


def run(cfg: Cfg, init0: G, init1: G, log_every: int = 50):
    jstep = jax.jit(lambda a, b, c, d: _step(a, b, c, d, cfg))
    p0, p1 = init0, init1
    m0, m1 = init0, init1
    hist = {"t": [], "expl": [], "mu0": [], "mu1": [], "s0": [], "s1": []}
    for t in range(cfg.steps):
        p0, p1 = jstep(p0, p1, m0, m1)
        if (t + 1) % cfg.magnet_interval == 0:
            m0, m1 = p0, p1
        if t % log_every == 0 or t == cfg.steps - 1:
            hist["t"].append(t)
            hist["expl"].append(exploitability(p0, p1, cfg))
            hist["mu0"].append(float(p0.mean)); hist["mu1"].append(float(p1.mean))
            hist["s0"].append(float(jnp.exp(p0.log_std))); hist["s1"].append(float(jnp.exp(p1.log_std)))
    e = np.array(hist["expl"])
    summary = {
        "final": float(e[-1]),
        "tail": float(e[int(len(e) * 0.7):].mean()),   # last-iterate quality (mean over tail)
        "tail_max": float(e[int(len(e) * 0.7):].max()),  # cycling shows up here
        "best": float(e.min()),
        "iterate_radius": float(np.hypot(hist["mu0"][-1], hist["mu1"][-1])),
    }
    return summary, hist, (p0, p1)


def init(mu, std=0.5):
    return G(jnp.asarray(float(mu)), jnp.asarray(float(np.log(std))))
