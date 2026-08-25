# theory3 — the simple convergence theory

`simple_convergence.tex` (16pp, compiles with `pdflatex`).

Scope: tabular, exact gradients, no approximation error, fixed `K` components with
`K >= ` number of landscape cells. The Gaussian head is treated only as a
preconditioner + a smoothing, never as a state variable.

## The one idea

Exploitability splits **exactly**, no slack, no cross term:

```
Expl(mu, nu) = Expl_M(w, v) + Delta_x + Delta_y
```

- `M(X,Y)_{kl} = u(x_k, y_l)` — the finite matrix game on the current atom locations
- `Delta_x = max_a u(a,nu) - max_k u(x_k,nu)` — the support gap (is the best response in my support?)

Then: **MMD kills the first term** (contraction to QRE of the restricted matrix game — the
magnet is what makes it a contraction). **Location ascent kills the other two** (each atom
climbs to its cell's maximum; global max = best cell max, so covering the cells suffices).

Careful with the second half: it carries no equilibrium structure of its own, but that is
not the same as being single-agent. At `K=1` we get `Expl_M ≡ 0` and the whole game sits
in the support gaps, where GDA on `u=xy` cycles forever. The two location blocks are
ascent/descent on one saddle function `U(X,Y) = Σ w_k v_l u(x_k,y_l)`; what makes them
tractable is that (C1) renders that saddle problem *strongly monotone*, not that it is an
optimization problem. See §4 and §7.

## The result

The proof has the same shape as the fixed-matrix MMD proof: Brouwer gives a fixed point of
the joint one-step map, and one iteration contracts towards it. Nothing tracks a moving
target — no cell-maximiser sensitivity, no QRE sensitivity. Small-gain coupling of the two
loops gives convergence iff

```
kappa  <~  m_star * tau        (coupling  <~  curvature x temperature)
m_star = w_min * m             (low-weight atoms move slowly)
```

and that is the *only* non-step-size condition — final `Expl <= 2 tau log K`. In
particular the cross-curvature `L_ab = ||grad^2_ab u||` reaches only the step-size bound
`eta_x <~ m_star p_min / (p_max^2 (L+L_ab)^2)`, never the threshold: the two location
blocks influence each other antisymmetrically and the `grad^2_ab` terms cancel in the
monotonicity inner product.

## Three things worth arguing about

1. **Both step sizes cancel out of the small-gain condition.** Two-timescale separation
   changes the transient and the rate, never the stability threshold. The threshold is a
   property of the game and of `tau`.
   Two corollaries about bookkeeping, each worth a factor. (a) Measuring the errors
   against *moving* targets — the current cell maximiser, the current QRE — costs
   `kappa/m` and `1/tau` and yields the strictly worse `kappa <~ m^2 tau^2`. (b) Treating
   each player's atoms as an exogenous disturbance to the other's landscape manufactures
   a spurious second threshold `p_max L_ab <~ p_min m_star`. Both vanish under the
   fixed-point / joint-saddle formulation.
2. **The magnet keeps atoms alive, not just the weights stable.** `w_k >= rho_k e^{-2U/tau}`,
   and since the location step is scaled by `w_k`, zero temperature ⇒ frozen atoms ⇒
   self-reinforcing starvation trap.
3. **Actionable:** dropping the `w_k` factor from the mean gradient in
   `gaussian_natural_step` is free (same fixed points, Lemma 3.1) and removes the
   starvation mode entirely, upgrading the tau-annealing rate from `1/log t` to polynomial.
   It also replaces `m_star = w_min * m` by `m`, i.e. it moves the **stability
   threshold**, not just the rate — by `1/w_min`, the largest constant in the theory.

## Falsifiable prediction on the current defaults

`MultiPointGame(peaks=±1, width=0.1, coupling=1)`, using `omega_sigma^2 ≈ 0.02` (the
sigma-smoothed bump width) and the observed `w_min ≈ 1/2`: threshold `tau >~ 0.04`.
Default `magnet_coef = 0.2` is 5x above it. Sweeping `magnet_coef` down through ~4e-2
should show onset of oscillation in exploitability. Cheap to run with `run_idealized.py`.

Second, sharper experiment: with `p_k = sigma_k^2` (item 3 above) the threshold drops to
`tau >~ 0.02` and stops depending on the equilibrium mixture. Running both sweeps
separates the `w_min` factor from everything else.

## Not proved (stated as such in §13)

- exact Nash when `kappa > 0` (a floor `O(kappa log K / m_star)` remains)
- sigma-annealing (cells move; conjecture only, §10)
- any exploration mechanism across cells — Assumption B just asserts coverage
- extensive form
