"""Check the scale-decay law predicted in theory2/game_class.tex (Cor. 2).

Predictions under test (TWO_POINT: peaks (0,1), h=1, w=0.1, coupling=1):

  P1  sigma_dot = -c sigma^3  =>  sigma^-2(t) = sigma_0^-2 + 2 c t   (affine!)
      with the free (no Gaussian magnet) constant
          c = (eta/2) * w_k * h*w/(w^2+s^2)^{3/2}  ->  (eta/2)*w_k*h/w^2 = 1.25
  P2  expl(t) = 2h(1 - w/sqrt(w^2+sigma^2)) ~ 100 sigma^2 = Theta(1/t)
  P3  the Gaussian magnet rescales c but preserves the power law, by the factor
          c_eff / c_free = (1 - exp(-eta*tau*T)) / (eta*tau*T)
      derived as follows. Within one refresh cycle sigma is nearly constant; with
      x = rho - rho_bar the magnet's log-std gradient is (eta/2)*tau*(1 - e^{2x}),
      so  dx/dk = -g + B(1 - e^{2x}),  g = c_free*sigma^2, B = (eta/2)*tau.
      In u = e^{2x} this is logistic, du/dk = 2u(A - Bu) with A = B - g; solving
      from u(0)=1 gives the decay over a cycle |x(T)| = (g/2B)(1 - e^{-2BT}),
      hence the factor above (2BT = eta*tau*T). Note the brake SATURATES inside
      the cycle -- a linear-response estimate gives 0.625 and is 15% wrong.
  P4  the floor 1e-3 was INACTIVE in theory/exp2: free decay from sigma_0=0.1
      predicts sigma(2e4) = (100 + 2*c_eff*2e4)^{-1/2}, and exp2 reports 0.0069105
      (=> c_eff = 0.521), i.e. a value well above the floor.

The run loop below is a transcription of `idealized_mmd.run` with two knobs the
original hard-codes: the std floor, and a separate magnet temperature for the
Gaussian head. Run A verifies the transcription reproduces `run` exactly.

Usage:  .venv/bin/python theory2/check_scale_law.py
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from idealized_mmd import (MMDConfig, Params, TWO_POINT, _feat_grid,
                           _feature_geometry, _well_grid, categorical_mirror_update,
                           component_q, expected_payoff, exploitability,
                           gaussian_natural_step, make_init, mixture_stats, run)

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

GAME = TWO_POINT
LOW, HIGH = -1.0, 2.0
H, W = 1.0, 0.1          # bump height / width of TWO_POINT
BASE = MMDConfig(lr=0.05, magnet_coef=0.2, magnet_interval=200)
C_FREE = (BASE.lr / 2) * 0.5 * H / W**2      # = 1.25


def brake(tau: float, T: int, eta: float = BASE.lr) -> float:
    """c_eff / c_free for the Gaussian magnet: (1 - e^{-eta tau T}) / (eta tau T)."""
    z = eta * tau * T
    return 1.0 if z == 0 else float((1 - np.exp(-z)) / z)


# ---- fine-grid exploitability (the repo's default n=4001 grid is too coarse) --
def expl_fine(px: Params, py: Params, game=GAME, n: int = 300_001) -> float:
    """`idealized_mmd.exploitability` with a configurable grid.

    At n=4001 the grid spacing is 7.5e-4 and neither peak is a grid point, so the
    best-response value is short of the true max by up to ~0.5*|D''|*(dx/2)^2
    ~ 7e-6 per player -- comparable to the exploitabilities we are measuring.
    n=300001 gives spacing 1e-5, i.e. an error ~1e-9.
    """
    coupling = getattr(game, "coupling", 1.0)
    mid, half_range, max_order, target = _feature_geometry(game)
    _, well_x, feat_x, _ = mixture_stats(px, game)
    _, well_y, feat_y, _ = mixture_stats(py, game)
    U = expected_payoff(px, py, game)
    space = game.action_space(0)
    grid = jnp.linspace(float(space.low[0]), float(space.high[0]), n)
    Dg = _well_grid(grid, game)
    feat_g = _feat_grid(grid, mid, half_range, max_order, target)
    br0 = jnp.max(Dg - well_y + coupling * jnp.sum(feat_g * feat_y[None, :], axis=-1))
    br1 = jnp.min(well_x - Dg + coupling * jnp.sum(feat_x[None, :] * feat_g, axis=-1))
    return float((br0 - U) + (U - br1))


# ---- run loop: `idealized_mmd.run` + configurable floor and tau_gauss ---------
def make_stepper(cfg: MMDConfig, tau_gauss: float, floor: float):
    cfg_g = dataclasses.replace(cfg, magnet_coef=tau_gauss)
    log_floor, log_ceil = jnp.log(floor), jnp.log(1.0)

    def one(carry, _):
        p0, p1, m0, m1, t = carry

        def apply(p, opp, magnet, sign):
            logits = categorical_mirror_update(
                p.logits, component_q(p, opp, GAME, sign), magnet.logits, cfg)
            nat_means, nat_log_std = gaussian_natural_step(p, opp, magnet, GAME, cfg_g, sign)
            means = jnp.clip(p.means + cfg.lr * nat_means, LOW, HIGH)
            log_std = jnp.clip(p.log_std + cfg.lr * nat_log_std, log_floor, log_ceil)
            return Params(logits=logits, means=means, log_std=log_std)

        q0, q1 = apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)
        t = t + 1
        refresh = (t % cfg.magnet_interval) == 0
        pick = lambda new, old: jnp.where(refresh, new, old)
        n0 = Params(*(pick(a, b) for a, b in zip(q0, m0)))
        n1 = Params(*(pick(a, b) for a, b in zip(q1, m1)))
        return (q0, q1, n0, n1, t), None

    @jax.jit
    def chunk(carry, n_steps):
        # fori_loop takes a traced bound, so distinct chunk lengths do not recompile
        return jax.lax.fori_loop(0, n_steps, lambda _, c: one(c, None)[0], carry)

    return chunk


def trajectory(init0, init1, cfg, tau_gauss, floor, checkpoints):
    """Run to each checkpoint, logging (t, sigma, expl)."""
    chunk = make_stepper(cfg, tau_gauss, floor)
    carry = (init0, init1, init0, init1, jnp.asarray(0, dtype=jnp.int64))
    rows, prev = [], 0
    for t in checkpoints:
        carry = chunk(carry, int(t - prev))
        prev = t
        p0, p1 = carry[0], carry[1]
        rows.append({
            "t": int(t),
            "sigma": float(jnp.exp(p0.log_std)[0]),
            "sigma_all": [float(x) for x in jnp.exp(p0.log_std)],
            "expl": expl_fine(p0, p1),
            "means": [float(x) for x in p0.means],
            "w": [float(x) for x in jax.nn.softmax(p0.logits)],
        })
    return rows, carry


def logspace_checkpoints(t_max, per_decade=12, t_min=200):
    pts = np.unique(np.round(np.logspace(np.log10(t_min), np.log10(t_max),
                                         int(per_decade * np.log10(t_max / t_min)) + 1)))
    return [int(p) for p in pts]


def fit_affine(t, sig):
    """sigma^-2 = a + b t  ->  returns (b, c=b/2, R^2)."""
    y = 1.0 / np.asarray(sig) ** 2
    t = np.asarray(t, dtype=float)
    b, a = np.polyfit(t, y, 1)
    resid = y - (a + b * t)
    r2 = 1 - resid.var() / y.var()
    return b, b / 2, r2, a


def loglog_slope(t, y, last_decades=2.0):
    t, y = np.asarray(t, float), np.asarray(y, float)
    keep = t >= t.max() / 10**last_decades
    return float(np.polyfit(np.log10(t[keep]), np.log10(y[keep]), 1)[0])


def report(name, rows, c_pred=None):
    t = [r["t"] for r in rows]
    sig = [r["sigma"] for r in rows]
    ex = [r["expl"] for r in rows]
    tail = [i for i, tt in enumerate(t) if tt >= t[-1] / 100]  # last 2 decades
    b, c, r2, a0 = fit_affine([t[i] for i in tail], [sig[i] for i in tail])
    s_sl = loglog_slope(t, sig)
    e_sl = loglog_slope(t, ex)
    print(f"\n--- {name} ---")
    print(f"  {'t':>9} {'sigma':>12} {'expl':>12} {'100*sig^2':>12} {'sig^-2':>14}")
    for r in rows[:: max(1, len(rows) // 10)] + [rows[-1]]:
        print(f"  {r['t']:>9d} {r['sigma']:>12.6e} {r['expl']:>12.6e} "
              f"{100 * r['sigma'] ** 2:>12.6e} {1 / r['sigma'] ** 2:>14.2f}")
    print(f"  fit sigma^-2 = {a0:.1f} + {b:.6f} t   (R^2 = {r2:.8f})  ->  c = {c:.6f}")
    if c_pred is not None:
        print(f"  predicted c = {c_pred:.6f}   ratio measured/predicted = {c / c_pred:.4f}")
    print(f"  log-log slope  sigma vs t = {s_sl:+.4f}  (predicted -0.5)")
    print(f"  log-log slope  expl  vs t = {e_sl:+.4f}  (predicted -1.0)")
    return {"c": c, "b": b, "r2": r2, "slope_sigma": s_sl, "slope_expl": e_sl, "rows": rows}


def main():
    results = {}
    init_atoms = make_init([0.05, 0.95])          # exp2's init: sigma_0 = 0.1

    # ---- A. transcription check: reproduce theory/exp2 exactly ---------------
    print("=== A. transcription check (floor 1e-3, tau_gauss=0.2, 20k steps) ===")
    cfgA = dataclasses.replace(BASE, steps=20000)
    p0r, p1r, _ = run(GAME, cfgA, init_atoms, init_atoms, log_every=10**9)
    rowsA, carryA = trajectory(init_atoms, init_atoms, BASE, 0.2, 1e-3, [20000])
    mine = carryA[0]
    print(f"  repo run():  sigma={float(jnp.exp(p0r.log_std)[0]):.10f}  "
          f"expl={float(exploitability(p0r, p1r, GAME)):.8f}")
    print(f"  this loop:   sigma={rowsA[0]['sigma']:.10f}  "
          f"expl(coarse)={float(exploitability(mine, carryA[1], GAME)):.8f}")
    print(f"  exp2.json:   sigma=0.0069105343  expl=0.0047573967")
    dsig = abs(rowsA[0]["sigma"] - float(jnp.exp(p0r.log_std)[0]))
    print(f"  |delta sigma| vs repo run() = {dsig:.3e}   "
          f"{'MATCH' if dsig < 1e-12 else 'MISMATCH'}")
    print(f"  fine-grid expl at that point = {rowsA[0]['expl']:.8e} "
          f"(coarse grid understates by the grid error)")
    results["A_transcription"] = {
        "sigma_repo": float(jnp.exp(p0r.log_std)[0]), "sigma_mine": rowsA[0]["sigma"],
        "expl_repo_coarse": float(exploitability(p0r, p1r, GAME)),
        "expl_fine": rowsA[0]["expl"], "delta_sigma": dsig}

    # P4: was the floor active in exp2?
    c_implied = (1 / rowsA[0]["sigma"] ** 2 - 100) / (2 * 20000)
    print(f"\n  [P4] floor = 1e-3, observed sigma(2e4) = {rowsA[0]['sigma']:.6f} "
          f"= {rowsA[0]['sigma'] / 1e-3:.1f}x the floor -> floor INACTIVE")
    print(f"       implied c_eff = {c_implied:.4f}  (free c = {C_FREE}, "
          f"predicted magnet-braked c = {C_FREE / 2})")
    results["P4"] = {"sigma_at_2e4": rowsA[0]["sigma"], "floor": 1e-3,
                     "c_eff_implied": c_implied}

    T_MAX = 2_000_000
    ckpt = logspace_checkpoints(T_MAX)

    # ---- B. free decay: Gaussian magnet OFF, floor released ------------------
    rowsB, _ = trajectory(init_atoms, init_atoms, BASE, 0.0, 1e-12, ckpt)
    results["B_free"] = report("B. tau_gauss = 0, floor 1e-12 (free decay)",
                               rowsB, c_pred=C_FREE)

    # ---- C. full magnet, floor released -------------------------------------
    rowsC, _ = trajectory(init_atoms, init_atoms, BASE, 0.2, 1e-12, ckpt)
    results["C_magnet"] = report("C. tau_gauss = 0.2, floor 1e-12 (magnet brake)",
                                 rowsC, c_pred=brake(0.2, BASE.magnet_interval) * C_FREE)

    # ---- D. same law after a transport phase --------------------------------
    init_spread = make_init([-0.30, 1.30], log_std=float(np.log(0.2)))
    rowsD, _ = trajectory(init_spread, init_spread, BASE, 0.2, 1e-12, ckpt)
    results["D_transport"] = report("D. spread init (-0.3, 1.3), sigma_0=0.2, "
                                    "tau_gauss=0.2, floor 1e-12",
                                    rowsD, c_pred=brake(0.2, BASE.magnet_interval) * C_FREE)

    # ---- E. the magnet brake factor, swept over tau and T -------------------
    print("\n--- E. magnet brake: c_eff/c_free = (1-exp(-eta*tau*T))/(eta*tau*T) ---")
    print(f"  {'tau':>6} {'T':>6} {'eta*tau*T':>10} {'factor':>9} {'c_pred':>9} "
          f"{'c_meas':>9} {'ratio':>7}  {'R2':>10}")
    ckpt_e = [int(x) for x in np.unique(np.round(np.logspace(4, 6, 25)))]
    sweep = []
    for tau, T in [(0.2, 50), (0.2, 200), (0.2, 800), (0.05, 200), (0.8, 200), (0.0, 200)]:
        rows, _ = trajectory(init_atoms, init_atoms,
                             dataclasses.replace(cfg_T := BASE, magnet_interval=T),
                             tau, 1e-12, ckpt_e)
        _, c_meas, r2, _ = fit_affine([r["t"] for r in rows], [r["sigma"] for r in rows])
        f = brake(tau, T)
        print(f"  {tau:>6} {T:>6} {BASE.lr * tau * T:>10.3f} {f:>9.5f} "
              f"{C_FREE * f:>9.5f} {c_meas:>9.5f} {c_meas / (C_FREE * f):>7.4f}  {r2:>10.7f}")
        sweep.append({"tau": tau, "T": T, "factor": f, "c_pred": C_FREE * f,
                      "c_meas": c_meas, "ratio": c_meas / (C_FREE * f), "r2": r2})
    results["E_brake_sweep"] = sweep

    # ---- F. the same factor rescales MEAN transport --------------------------
    # Mean update inside a refresh cycle: x = m - m_bar obeys x' = g - eta*tau*x,
    # so the cycle-averaged velocity is g*(1-e^{-eta tau T})/(eta tau T) -- the
    # same beta as the log-std. If both coordinates share one factor the whole
    # (m, sigma) flow is time-rescaled: asymptotic basins unchanged, finite-horizon
    # reach shrinks by beta. Test: transport time should scale as 1/beta.
    print("\n--- F. transport time vs the brake factor (spread init) ---")
    print(f"  {'tau_g':>7} {'beta':>9} {'t_arrive':>10} {'ratio':>8} {'1/beta':>8}")
    ck_f = [int(x) for x in np.unique(np.round(np.logspace(1, 6.3, 200)))]
    base_t, transport = None, []
    for tau in [0.0, 0.2, 2.0]:
        rows, _ = trajectory(init_spread, init_spread, BASE, tau, 1e-12, ck_f)
        arrive = next(r["t"] for r in rows if abs(r["means"][0]) < 1e-3)
        base_t = base_t or arrive
        f = brake(tau, BASE.magnet_interval)
        print(f"  {tau:>7} {f:>9.5f} {arrive:>10d} {arrive / base_t:>8.3f} {1 / f:>8.3f}")
        transport.append({"tau_g": tau, "beta": f, "t_arrive": arrive,
                          "ratio": arrive / base_t, "inv_beta": 1 / f})
    results["F_transport"] = transport

    (OUT / "scale_law.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'scale_law.json'}")


if __name__ == "__main__":
    main()
