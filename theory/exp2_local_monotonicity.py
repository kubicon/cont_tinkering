"""Exp 2 — Layer 3: the Nash-realizing configuration is a locally-monotone,
locally-stable fixed point of the parametric MMD dynamics (prediction P2).

Four measurements on TWO_POINT (MultiPointGame, peaks (0,1), c=1):

(A) **Fixed point**: run the idealized dynamics to convergence, report the final
    configuration and the one-step residual ||step(z, magnet=z) - z||.

(B) **Jacobian spectrum**: spectral radius of the linearized step map
    z -> step(z; magnet=z*) at z*. rho < 1 <=> local exponential stability.

(C) **Own-Hessian blocks** at z*: eigenvalues of d^2 U / d mu_0^2 (should be
    negative: strong concavity in own means, curvature ~ -w_k * D_s''(peak_k)),
    and the within-player cross term d^2 U / d pi_k d mu_k = q_k'(mu_k) ~ 0.
    Compare against the closed-form prediction from the smoothed well.
    By the block-skew lemma this certifies local monotonicity of the field.

(D) **Basin structure**: the theory predicts convergence iff the initial means
    put one component in each peak's basin (distinct-basin assignment) — the
    monotone island around z* certifies local convergence, and the effective
    landscape's basins delimit the funnel. We sample random mean inits over the
    box, run the dynamics, and score the predictor "exactly one component per
    basin (split at the midpoint 0.5)" against actual convergence.

Run:  .venv/bin/python theory/exp2_local_monotonicity.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from idealized_mmd import (MMDConfig, Params, TWO_POINT, _tail_expl,
                           categorical_mirror_update, component_q,
                           exploitability, expected_payoff,
                           gaussian_natural_step, make_init, run,
                           well_expectation)

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

GAME = TWO_POINT
CFG = MMDConfig(lr=0.05, magnet_coef=0.2, magnet_interval=200, steps=20000)
LOW, HIGH = -1.0, 2.0  # TWO_POINT box
STD_FLOOR = 1e-3


def flatten(p0: Params, p1: Params):
    return jnp.concatenate([p0.logits, p0.means, p0.log_std,
                            p1.logits, p1.means, p1.log_std])


def unflatten(z, k):
    parts = jnp.split(z, 6)
    return (Params(parts[0], parts[1], parts[2]),
            Params(parts[3], parts[4], parts[5]))


def mmd_step(z, zmag, cfg=CFG, game=GAME):
    """One simultaneous MMD step (magnet fixed), the map whose Jacobian we take."""
    k = z.shape[0] // 6
    p0, p1 = unflatten(z, k)
    m0, m1 = unflatten(zmag, k)

    def apply(p, opp, magnet, sign):
        logits = categorical_mirror_update(p.logits, component_q(p, opp, game, sign),
                                           magnet.logits, cfg)
        nat_means, nat_log_std = gaussian_natural_step(p, opp, magnet, game, cfg, sign)
        means = jnp.clip(p.means + cfg.lr * nat_means, LOW, HIGH)
        log_std = jnp.clip(p.log_std + cfg.lr * nat_log_std,
                           jnp.log(STD_FLOOR), jnp.log(1.0))
        return Params(logits, means, log_std)

    q0, q1 = apply(p0, p1, m0, +1), apply(p1, p0, m1, -1)
    return flatten(q0, q1)


def main():
    out = {}

    # ---- (A) find the fixed point ------------------------------------------
    init = make_init([0.05, 0.95])
    pf0, pf1, hist = run(GAME, CFG, init, init)
    zstar = flatten(pf0, pf1)
    resid = float(jnp.linalg.norm(mmd_step(zstar, zstar) - zstar))
    expl = float(exploitability(pf0, pf1, GAME))
    print("=== (A) fixed point ===")
    print(f"  means0={np.array(pf0.means)}  w0={np.array(jax.nn.softmax(pf0.logits))}"
          f"  std0={np.array(jnp.exp(pf0.log_std))}")
    print(f"  exploitability at z* = {expl:.5f}")
    print(f"  one-step residual ||step(z*)-z*|| = {resid:.3e}")
    out["fixed_point"] = {"expl": expl, "residual": resid,
                          "means0": np.array(pf0.means).tolist(),
                          "w0": np.array(jax.nn.softmax(pf0.logits)).tolist(),
                          "std0": np.array(jnp.exp(pf0.log_std)).tolist()}

    # ---- (B) Jacobian spectrum ---------------------------------------------
    J = np.asarray(jax.jacobian(lambda z: mmd_step(z, zstar))(zstar))
    eig = np.linalg.eigvals(J)
    rho = float(np.max(np.abs(eig)))
    print("\n=== (B) Jacobian of the step map at z* (magnet at z*) ===")
    print(f"  spectral radius rho = {rho:.6f}  ({'STABLE' if rho < 1 else 'UNSTABLE'})")
    print(f"  |eigenvalues| = {np.sort(np.abs(eig))[::-1].round(6)}")
    out["jacobian"] = {"rho": rho, "abs_eigs": np.abs(eig).tolist()}

    # ---- (C) own-Hessian blocks (local monotonicity certificate) -----------
    print("\n=== (C) own-player Hessian blocks at z* ===")
    hess0 = np.asarray(jax.hessian(
        lambda m: expected_payoff(pf0._replace(means=m), pf1, GAME))(pf0.means))
    hess1 = np.asarray(jax.hessian(
        lambda m: -expected_payoff(pf0, pf1._replace(means=m), GAME))(pf1.means))
    e0, e1 = np.linalg.eigvalsh(hess0), np.linalg.eigvalsh(hess1)
    # closed-form prediction: d2/dmu2 of the smoothed well at a peak, weighted by w_k
    w0 = np.array(jax.nn.softmax(pf0.logits))
    s2 = np.exp(2 * np.array(pf0.log_std))
    var = s2 + GAME.width ** 2
    pred = -w0 * (GAME.width / np.sqrt(var)) / var  # w_k * D_s''(peak_k), height 1 bumps
    print(f"  eig(H_mu0) = {e0.round(4)}   eig(H_mu1) = {e1.round(4)}"
          f"   ({'both concave' if e0.max() < 0 and e1.max() < 0 else 'NOT concave'})")
    print(f"  diag(H_mu0) = {np.diag(hess0).round(4)}  vs predicted w_k*D_s''(p_k) = {pred.round(4)}")
    # within-player cross term: dq_k/dmu_k (should vanish at the peaks)
    qgrad = np.asarray(jax.jacobian(
        lambda m: component_q(pf0._replace(means=m), pf1, GAME, +1.0))(pf0.means))
    print(f"  cross term q_k'(mu_k) = {np.diag(qgrad).round(5)}  (theory: ~0 at peaks)")
    out["hessians"] = {"eig_h0": e0.tolist(), "eig_h1": e1.tolist(),
                       "diag_h0": np.diag(hess0).tolist(), "pred": pred.tolist(),
                       "cross": np.diag(qgrad).tolist()}

    # ---- (D) basin structure: distinct-basin assignment predicts convergence
    print("\n=== (D) distinct-basin predictor over random mean inits ===")
    rng = np.random.default_rng(0)
    n_samples = 60
    midpoint = 0.5  # basin boundary between peaks 0 and 1 (equal weights)
    records = []
    for i in range(n_samples):
        means = rng.uniform(LOW, HIGH, size=2)
        p_init = make_init(means.tolist())  # std 0.1, uniform logits
        _, _, h = run(GAME, CFG, p_init, p_init, log_every=500)
        tail = _tail_expl(h)
        converged = bool(tail < 0.1)
        distinct = bool((means.min() < midpoint) and (means.max() >= midpoint))
        records.append({"means": means.tolist(), "tail_expl": tail,
                        "converged": converged, "distinct_basin": distinct})
    acc = np.mean([r["converged"] == r["distinct_basin"] for r in records])
    n_conv = sum(r["converged"] for r in records)
    n_dist = sum(r["distinct_basin"] for r in records)
    mism = [r for r in records if r["converged"] != r["distinct_basin"]]
    print(f"  {n_samples} random inits: {n_conv} converged, {n_dist} predicted "
          f"(distinct-basin), predictor accuracy = {acc:.2%}")
    for r in mism[:10]:
        print(f"    mismatch: means={np.round(r['means'],3)} tail={r['tail_expl']:.3f} "
              f"conv={r['converged']} pred={r['distinct_basin']}")
    out["basin"] = {"accuracy": float(acc), "records": records}

    with open(RESULTS / "exp2.json", "w") as f:
        json.dump(out, f)
    print(f"\nsaved -> {RESULTS / 'exp2.json'}")


if __name__ == "__main__":
    main()
