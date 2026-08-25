# Why does MMD on a Gaussian-mixture policy converge to Nash in continuous games?

Working log of a theory for the empirically observed convergence of Magnetic Mirror
Descent (MMD) on the K-component Gaussian-mixture parameterization
(`idealized_mmd.py`, `training/mixture*.py`), despite MMD's convergence theory
requiring monotonicity that the *parametric* game does not have.

Formal statements and proofs live in [`mmd_mixture_theory.tex`](mmd_mixture_theory.tex).
Every experiment referenced here is a script in this folder.

---

## The puzzle

- The games (`MultiPointGame`, `DecoyWellGame`, `ForsakenGame`, ...) have mixed
  Nash equilibria with finite support; the policy is a K-component Gaussian
  mixture (categorical head + Gaussian components).
- MMD's convergence guarantee (Sokota et al. 2023) needs the game's pseudo-gradient
  operator to be **monotone**. In the mixture *parameters* (logits, means, log-stds)
  the expected payoff is wildly non-concave (multi-well landscape), so the parametric
  game is **not** monotone — yet MMD converges to the Nash from reasonable inits.

## The proposed theory (three-layer explanation)

**Layer 1 — Lifted monotonicity (global, exact).**
The mixed extension of *any* two-player zero-sum game over probability measures is
*linear in each player's measure*, hence concave–convex, hence its pseudo-gradient
is monotone (in fact skew: ⟨v(z)−v(z′), z−z′⟩ = 0 exactly). So in *measure space*
MMD is squarely inside its convergence theory: the entropy-regularized fixed point
exists and MMD converges to it; magnet updates walk it to Nash. Non-monotonicity is
purely an artifact of the *parameterization*, not of the game.
→ Lemma 1 in the LaTeX. Verified by `exp1_lifted_monotonicity.py`.

**Layer 2 — The parametric update is a Fisher–Rao restriction of the lifted flow.**
The implemented update is mirror descent in the KL geometry of the hierarchical
policy: the categorical head does the *exact* closed-form simplex MMD step on the
component Q-values, and the Gaussian head does a natural-gradient (Fisher–Rao) step
— the small-step limit of the KL-proximal update. So the parametric dynamics are
the *projection* of the monotone measure-space MMD field onto the mixture
submanifold. Convergence survives the projection exactly where the projection
preserves monotonicity.
→ Lemma 2 (block-skew structure of zero-sum parametric games): the symmetrized
Jacobian of the pseudo-gradient field is **block-diagonal** — cross-player blocks
are exactly skew and drop out. Hence the parametric field is monotone on a convex
region S iff U is concave in each player's own parameters on S.

**Layer 3 — Local monotonicity islands (the actual explanation).**
At a Nash-realizing configuration z* (one component per support atom, correct
weights, std at floor):

- the opponent's feature expectation is matched (F(ν*) = 0), so the coupling term
  vanishes *identically* — each player's effective landscape collapses to the well D;
- each mean sits at a strict local max of the smoothed D → the payoff is strongly
  concave in each mean (curvature ≈ −w_k·h/w² < 0);
- the payoff is exactly *linear* in the categorical π (the head works in π-space);
- own-block cross terms ∂²U/∂w_k∂μ_k = Q′(μ_k) = 0 at the peak.

So U is locally concave in own-parameters for both players ⇒ by Lemma 2 the field
is monotone on a neighborhood of z*, strongly monotone once the magnet τ > 0 is
added ⇒ MMD converges linearly inside this **monotone island**.

The observed non-global behavior is then *predicted*, not anomalous: other
configurations (all mass on the decoy; symmetric ties) are *also* locally-monotone
islands with their own basins, and the simplex boundary (weight starvation) freezes
coordinates because the Fisher–Rao mean-gradient is weighted by w_k.
→ Theorem 3 + Proposition 4 in the LaTeX.

## Falsifiable predictions

- **P1**: Tabular (measure-space) MMD converges *globally* on the very games whose
  parametric mixture version has traps (decoy well, forsaken) — the traps are
  parameterization artifacts. (Layer 1)
- **P2**: The Jacobian of the parametric MMD map at the Nash-realizing fixed point
  is stable (spectral radius < 1), and the own-player Hessian blocks of U are
  negative semidefinite there. (Layer 3)
- **P3**: The decoy trap configuration is *also* a stable fixed point with locally
  concave own-blocks — a second monotone island. (Layer 3)
- **P4**: The size of the Nash island shrinks as the coupling strength c grows
  (the coupling term c·F(ν)·f″ can flip the mean's concavity once F(ν) ≠ 0, and
  ‖F(ν)‖ scales with distance from Nash ⇒ monotone radius ~ 1/c). Empirical basin
  radius should track the measured monotone radius. (Layer 3)
- **P5**: Residual exploitability at convergence scales as O(s²) in the std floor s
  (smoothing bias of representing atoms by narrow Gaussians).

## Experiment log

(updated as results come in)

| # | Script | Tests | Status |
|-|-|-|-|
| 1 | `exp1_lifted_monotonicity.py` | P1 + exact skewness of lifted field | ✅ confirmed |
| 2 | `exp2_local_monotonicity.py` | P2 (Jacobian spectrum, Hessian blocks, monotone radius) | ✅ confirmed |
| 3 | `exp3_coupling_sweep.py` | P4 (basin radius vs coupling, vs monotone radius) | ✅ confirmed |
| 4 | `exp4_decoy_island.py` | P3 (decoy trap = second monotone island) | ✅ confirmed |
| 5 | `exp5_std_floor_bias.py` | P5 (residual expl ~ s²) | ✅ confirmed |
| 6 | `exp6_gaussian_head_magnet.py` | Is the magnet term on the *Gaussian* head useful? | ✅ answered: no |
| 7 | `exp7_ebm_regression_measure.py` (+`exp7b`) | Does the EBM-regression proposal inherit the tabular escape? | ⚠️ only conditionally |
| 8 | `exp8_scale_dominance.py` (+`8b`–`8e`) | Global assumptions: the scale-space window | ✅ assumption set found & validated |

### Exp 1 — lifted monotonicity & P1: **confirmed**

- **Skewness exact**: worst `|<F(z)-F(z'), z-z'>|` over 100 random measure pairs is
  ~1.7e-16 (machine precision) on both `two_point` (n=401) and `decoy_well` (n=601)
  grids. The lifted game is monotone *with equality* — pure skew, as proved in Lemma 1.
- **Tabular MMD converges globally** (lr=0.05, τ=0.2, magnet every 200):
  - two_point: uniform + 2 random inits → tail expl ≈ 0.002–0.003. PASS.
  - decoy_well: uniform + random inits → expl ≈ 0.0001. PASS.
  - decoy_well **initialized ON the decoy** (sharp Gaussian logits at 0, ~200
    log-units below the peaks): expl 0.7205 at 20k steps (looks trapped!) but
    → **0.0000 by ~76k steps**. Escape is slow, not blocked: mirror dynamics gain
    ~q-gap/τ log-units of peak mass per magnet cycle, and the init dug a 200
    log-unit hole. In measure space the support never dies, so the persistent
    Q-advantage of the peaks (~0.36) always wins eventually. The *parametric*
    mixture at the same configuration is a **stable fixed point forever** — the
    sharpest possible confirmation that the trap is a parameterization artifact.
- **Forsaken step-size sensitivity reproduced in measure space**: (lr=0.05, τ=0.1)
  and (0.02, 0.2) converge (expl ≈ 0.002–0.004); (lr=0.1, τ=0.05) cycles
  (tail expl 2.34). Monotone-but-not-strongly games need the MMD step-size
  condition; violating it cycles even in the exact lifted space. This cleanly
  separates *step-size* failures from *parameterization* failures.

### Exp 2 — local monotonicity at the Nash configuration (P2): **confirmed**

On TWO_POINT (peaks (0,1), c=1), lr=0.05, τ=0.2:

- **(A) Fixed point**: dynamics settle at means (0.0002, 0.9998), weights
  (0.5, 0.5), std 0.0069; exploitability 0.0048; one-step residual 1.2e-4.
- **(B) Jacobian spectrum**: spectral radius of the linearized step map (magnet
  frozen at z*) is **ρ = 0.9913 < 1** — locally exponentially stable. The
  contraction rate 1−ρ ≈ 0.0087 ≈ lr·τ/(1+lr·τ) ≈ 0.0099: exactly the strong
  monotonicity injected by the magnet, as the theory says (the unregularized
  skew part alone would give ρ ≈ 1).
- **(C) Own-Hessian blocks**: eig(∂²U/∂μ₀²) = (−49.64, −49.64) — strongly
  concave — and matches the closed-form prediction −w_k·D_s″(p_k) = −49.644 to
  four digits. Within-player cross term q_k′(μ_k) = ∓0.023 ≈ 0. Both players. By
  the block-skew lemma this certifies the field is monotone near z*: **the
  monotone island exists and z* sits inside it.**
- **(D) Basin structure**: 60 random mean-inits over the box. Convergence is
  predicted by "sort the means, assign to sorted peaks, require every
  |μ_k − p_k| < r*" with **96.7% accuracy for any r* ∈ [0.45, 0.55]** — the
  basin boundary is the **Voronoi boundary of the peaks** (half separation
  0.5), not the naive "one mean each side of the midpoint" (61.7%). All 12
  inits with max-distance < 0.43 converge; all 44 with > 0.56 fail; the 3
  boundary cases sit in 0.47–0.56. Interpretation: the certified monotone
  island (radius ~√(w²+s²) ≈ 0.14, where the well cap is concave) sits inside a
  larger funnel bounded by the peaks' Voronoi cells — inside a cell the
  smoothed-well pull plus the moment-matching coupling force never eject a
  component; once a component starts outside its cell, the dynamics freeze in
  the flat zone or lock onto a wrong-assignment island.

### Exp 3 — island radius vs coupling strength (P4): **confirmed**

Game: MultiPointGame(peaks (−1,0,1), width 0.08), K=3, curved feature (u²) so
coupling *can* bend the own-Hessian. Two measurements:

**(a) Certified monotone radius** (fraction of r-perturbed configs with both
own-mean Hessians NSD, 100 samples):

| c \ r |0.02|0.04|0.06|0.08|0.10|0.12|0.16|0.20|
|-|-|-|-|-|-|-|-|-|
| 0.5–8 |1.00|1.00|1.00|1.00|0.09|0.00|0.00|0.00|
| 32 |1.00|1.00|1.00|0.97|0.08|0.00|0.00|0.00|
| 100 |1.00|1.00|1.00|0.89|0.08|0.00|0.00|0.00|
| 300 |1.00|1.00|1.00|0.69|0.15|0.05|0.03|0.03|

The certified island radius is **exactly flat in c through c=8** (rows
identical), pinned at the well concavity cap √(w²+s²) ≈ 0.08, and only starts
degrading at c = 32–300 = O(h/w²) ≈ O(156) — the two-regime prediction
r_mono ~ min{√(w²+s²), O(h/(c·w²·‖f″‖))} verified on both sides.

**(b) Empirical basin** (fraction of runs converged, lr=0.05, τ=0.2):

| c \ r |0.05|0.10|0.15|0.20|0.30|0.40|0.50|
|-|-|-|-|-|-|-|-|
| 0.5 |1.00|0.17|0.00|0.00|0.00|0.00|0.00|
| 1.0 |1.00|1.00|0.83|0.17|0.00|0.00|0.00|
| 2.0 |1.00|1.00|1.00|1.00|0.50|0.00|0.17|
| 4.0 |1.00|1.00|1.00|1.00|0.83|0.50|0.17|
| 8.0 |0.00|0.00|0.00|0.00|0.00|0.00|0.00|

Two effects, both theory-consistent:
- The **basin grows with c** (0.5→4): outside the well cap the only inward
  force is the moment-matching coupling force c·F(ν)·f′(μ), so a stronger
  coupling shepherds stray components through the dead zone. At c=0.5 the
  basin barely exceeds the certified island; at c=4 it reaches r≈0.4.
- The **c=8 wipeout is a step-size violation, not an island failure**: the
  monotone island at c=8 is identical to c=0.5 (table a), but MMD's step-size
  condition η ≲ τ/L² tightens as the field's Lipschitz constant grows ∝ c.
  Verified: at c=8, r=0.05, lr=0.05 diverges (tail expl ≈ 17.9 on 4/4 seeds)
  while **lr=0.01 converges on 4/4 seeds (tail expl 0.02)**. Same failure
  signature as tabular Forsaken in exp1 — step-size, not geometry.

### Exp 4 — the decoy trap is a second monotone island (P3): **confirmed**

DecoyWellGame (peaks ±1 h=1 w=0.05; decoy at 0 h=0.7 w=0.45), lr=0.05, τ=0.2:

- From the trainer's spread init (means ±1.5, **std 1.5**) the dynamics collapse
  onto the decoy: means (±0.001), expl **0.7238** — the documented trap.
- At the trap endpoint: **F(ν) = 0 exactly** (u(0) = target moment = 0, so the
  coupling term vanishes identically), own-mean Hessians = −1.71 < 0 (the decoy
  is a strict local max of D_s), one-step residual 1.3e-4, Jacobian spectral
  radius **0.9902 < 1**. It is a *bona fide* locally-monotone, locally-stable
  fixed point — indistinguishable, locally, from the Nash island.
- The Nash island of the same game (init on the peaks): expl 0.0048, Hessians
  −197.9, ρ = 0.9913. **Two coexisting monotone islands of one vector field**;
  which one you get is decided by the init.
- Bonus finding: with the *same* spread means ±1.5 but init **std 0.1**, the
  run converges to the **Nash** (expl 0.014). The initial smoothing scale
  selects the island: broad components feel the mass-dominated smoothed
  landscape (decoy wins); narrow components feel the height-dominated one
  (peaks win). This sharpens the repo's "annealing delivers you into the trap"
  observation into a mechanism.

### Exp 5 — residual exploitability = O(s²) smoothing bias (P5): **confirmed**

At the exact Nash configuration with common std s on TWO_POINT, measured
exploitability matches the closed form 2h(1 − w/√(w²+s²)) to ~1e-5 across
s ∈ [1e-3, 0.3]; log-log slope over s ≤ 0.03 is **2.002** (theory: 2). The
"convergence not quite to zero" seen in training is quantified representation
bias, not failure to converge.

### Exp 6 — the magnet term on the Gaussian head is **not** helpful

The two heads use the magnet asymmetrically in theory: the **categorical** head's
payoff is linear in π (no own curvature), so its magnet `τ·KL(π‖magnet)` is what
injects the strong convexity MMD needs; the **Gaussian** head is already strongly
*concave* in its means at a Nash-realizing config (Hessian ≈ −w·h/w² ≪ 0, exp2),
so the added proximal `−τ·KL(current‖magnet)` sits on top of curvature that
already provides local monotonicity. Prediction: the Gaussian-head magnet is
redundant for convergence and only a transport brake. Separating the two
coefficients (cat_τ, gauss_τ) and running matched A/B/C variants confirms it:

| scenario | A both (final / reach) | B gauss-magnet OFF | C cat-magnet OFF |
|-|-|-|-|
| two_point Nash basin | 0.0048 / 1100 | **0.0020 / 600** | 0.0049 / 1100 |
| three_point Nash basin | 0.0073 / 2100 | **0.0031 / 1300** | **0.55 / never** |
| decoy on-peaks (clean) | 0.0048 / 1300 | **0.0020 / 700** | 0.0048 / 1300 |
| weight-starvation trap | 4.034 / never | 4.033 / never | 4.034 / never |
| decoy spread (the trap) | 0.724 / never | 0.721 / never | 0.724 / never |

- **Dropping the Gaussian magnet never hurts and mildly helps.** Every case that
  converges with it converges *faster and to a lower residual* without it
  (two_point 0.0020 vs 0.0048, reaches 0.1 in 600 vs 1100 steps) — the magnet is
  a brake plus a small pull toward the stale snapshot's std.
- **Dropping the *categorical* magnet, by contrast, breaks convergence** on the
  harder game (three_point: 0.55, never converges) — confirming the value of the
  MMD proximal term lives entirely in the categorical head, exactly where the
  payoff is linear and has no curvature of its own. (K=2 two_point survives it
  only because curvature alone suffices there.)
- **On the traps it is irrelevant** (transport barriers, not selection failures)
  — consistent with the repo's "magnet fixes equilibrium *selection*, not
  transport" and MultiDim exp6 (a *stronger* magnet is worse).
- **Local-stability subtlety.** The Gaussian magnet *does* pull the step map's
  spectral radius at the Nash from ρ = 0.99988 down to 0.9913 (1−ρ ≈ η·τ). But
  that near-unit mode is **99.7% in the log-std direction** (verified via the
  dominant eigenvector: |μ|,|logits| components ≈ 0.02, |log-std| = 1.0), i.e. it
  is the exploitability-irrelevant O(s²) smoothing-bias direction (std drifting to
  the floor). The means/weights that set exploitability contract fast through
  curvature regardless — which is why B converges *faster* in exploitability
  despite the worse ρ. The magnet's extra "stability" buys nothing that matters.

**Verdict:** running the MMD-like proximal (magnet) loss on the Gaussian head is
not helpful — it is mildly counterproductive on convergent cases and inert on
the traps. Keep the magnet on the categorical head (essential); the Gaussian head
should rely on its own curvature (natural-gradient step, entropy floor), not a
magnet. Practical implication for the PPO trainer: `magnet_gaussian_kl_coef` can
be set to 0 (or much smaller than `magnet_category_kl_coef`) without losing
convergence, and likely gains a little speed and a little less residual.

### Exp 7 — does the EBM-regression proposal inherit the tabular escape? **only conditionally**

Context. A natural fix for the parametric traps is to drop the Gaussian mixture
and represent `g = log π` directly as an energy model `π_θ ∝ exp(g_θ)`, trained
each step by *regression* onto the closed-form mirror target
`g⁺(a) = (η·q(a) + ητ·ḡ(a) + g(a)) / (1+ητ)` (Sokota et al.'s neural-MMD recipe).
Convergence then lives at the functional level (monotone, Lemma 1) and the network
is a per-step approximation error ε ⇒ perturbed-VI gives an O(ε/τ) neighborhood.
It *should* inherit the measure-space escape (exp1) because an EBM's support never
dies. exp7 stress-tests that "for free" by isolating, with no MCMC and no SGD, the
two things that can inflate ε: the **representable sharpness** of `g_θ`, and the
**regression measure** μ (a real sampler weights the loss by where it draws
actions). The network fit is modelled exactly as the μ-weighted least-squares
projection of `g⁺` onto a fixed smooth basis Φ (RBFs + const), so the only moving
parts are capacity (m centres) and μ. Testbed: the decoy trap, both players
initialised *on* the decoy (the exp1 escape case, start expl ≈ 0.72).

Two findings, both cautionary.

- **(1) Escape is capacity-gated.** Below an RBF-sharpness threshold *no* measure
  escapes: at m ≤ 60 (n=401) every measure sticks at ≈0.706, indistinguishable
  from the init. Escape appears only once the basis can sharpen the width-0.05
  peaks — m=120 (n=401) reaches 0.018, and at n=601 uniform μ fully escapes
  (expl 7e-4 at 40k; 1e-4 at m=180). This sharpens exp1's "support never dies":
  the tabular escape needs not just surviving support but *arbitrarily peaked*
  mass at the atoms (escape accrues a pointwise log-mass advantage there, the
  ~0.36 Q-gap of exp1). A smooth EBM caps that peakedness, and the cap — not the
  sampler — is the first thing that blocks escape, tying the escape rate to
  `g_θ`'s bandwidth/Lipschitz constant near the atoms. Attribution control: the
  two_point game (no transport trap) escapes under every measure at m=60.

- **(2) At adequate capacity the regression measure is load-bearing — and
  non-monotone.** Pinning capacity where uniform robustly escapes (n=601,
  m ∈ {120,150,180}) and sweeping μ on the decoy trap:

  | μ | final expl | verdict |
  |-|-|-|
  | uniform | 0.0007 | escaped (slow crawl) |
  | on-policy π | 0.0122 | escaped |
  | tempered π^0.5 | 0.0064 | escaped |
  | mix (1-ρ)π + ρ·unif, ρ=0.02 | 0.7198 | **stuck** |
  | mix ρ=0.05 | 0.7198 | **stuck** |
  | mix ρ=0.10 | 0.7198 | **stuck** |

  Both *endpoints* of the mix family escape (ρ=0 is on-policy, ρ=1 is uniform)
  yet every interior mix sticks — robust across ρ and across m=120/150. Plausible
  mechanism: on-policy leaves the (unweighted) peak region free to grow its
  log-density, and uniform lets exp1's slow crawl operate; a small uniform floor
  instead pins the peak region to fit the *early* target there, where the smoothed
  peak barely exceeds the decoy — actively holding the peaks down. The effect is
  genuine (the iterate moves, then stalls at a fixed point — not a NaN/freeze),
  but the exact non-monotone shape is specific to this weighted-LS projection
  model; what transfers is the qualitative lesson, not the ρ-response curve.

**Verdict.** The EBM proposal does **not** inherit the tabular escape for free.
Two independent knobs gate it — the network's representable sharpness near the
support atoms, and the action-sampling/regression measure — and the measure
dependence is strong and non-monotone, exactly the failure mode a naive on-policy
amortised sampler would stumble into. For the practical MMD trainer this says an
EBM policy needs (a) enough local capacity/sharpness at the support atoms and
(b) a deliberately chosen sampling measure (pure exploration *or* pure on-policy
both worked here; half-measures did not) — sampling design is a first-class
correctness concern, not an implementation detail. Scripts:
`exp7_ebm_regression_measure.py`, `exp7b_measure_at_capacity.py`.

### Exp 8 — global convergence assumptions: the scale-space window

Exp 1–4 explain convergence *island-by-island*; exp8 asked the global question:
**which games (and inits) does the transit reach the Nash island from?** The
answer is a checkable assumption set with a quantitative constant for each
edge, a closed-form predictor, and a matched counterexample per assumption.
Scripts: `exp8_scale_dominance.py`, `exp8b_boundaries.py`,
`exp8c_coupling_threshold.py`, `exp8d_window.py`, `exp8e_flow_oracle.py`.

**The reduced model that decides everything.** Along a symmetric transit the
coupling term vanishes *identically* (verified brutally in exp8c-F: runs with
c = 0.5, 1, 2 are bit-identical), so each component's fate is decided by pure
smoothed-well ascent with a self-annealing scale — the natural-gradient flow
of a **single Gaussian** (μ, s) on the smoothed well amplitude
D_s(μ) = Σ_b h_b w_b/√(w_b²+s²) · exp(−(μ−c_b)²/(2(w_b²+s²))):

    dμ/dt = s²·∂D_s/∂μ,   d(log s)/dt = ½·∂D_s/∂log s

(`exp8e_flow_oracle.py`; microseconds per query; knows nothing about MMD, the
magnet, the categorical head, or the opponent). This flow predicts every
measured threshold to within ~10%:

| edge | flow prediction | measured (full MMD) |
|-|-|-|
| decoy-mass trap threshold (s0=1.5) | h_d* = 0.220 (mass ×1.98) | trap at 0.25; 0.22 converges (slowly, 150k) |
| dominance scale s* (h_d=0.7, init ±1.5) | 0.272 | flips in (0.28, 0.30) |
| heavy decoy, init ±2.5 | **no s0 converges** | no s0 converges (H1) |
| light decoy (mass ×0.9), init ±2.5 | s0 ≳ 0.3 converges | s0 ≥ 0.5 converges (H2) |

**The result: convergence ⇔ a nonempty init-std window.** From component
means at distance d from their atoms, annealed MMD converges iff the init
std s0 lies in

    s_reach(d)  ≲  s0  ≲  min( s_dom(game), s_ej(game) )

with three separately measured edges, each with its own counterexample:

- **s_reach — the transport (dead-zone) edge.** Below it the means freeze:
  the smoothed gradient at distance d is ~exp(−d²/2s0²), so finite-horizon
  convergence needs d ≲ κ·s0 (κ ≈ 5 at lr=0.05, ≤150k steps). Counterexample:
  original decoy game, init ±1.5, s0 = 0.05 — frozen at ±1.47 after 40k steps
  while s0 = 0.10 converges (slowly) and s0 ∈ [0.15, 0.28] converges fast.
  The flow has no finite-time cutoff, which is its only systematic error
  (it calls s0=0.3 from ±2.5 "reachable"; MMD needs 0.5).
- **s_dom — the scale-dominance (decoy) edge.** The scale at which spurious
  smoothed mass overtakes the Nash atoms' modes in D_s. Above it, components
  anneal into the decoy island (exp4) and lock. Sharp: s0=0.28 → Nash,
  s0=0.30 → decoy, against the flow's 0.272. Equivalently, at fixed s0=1.5
  the decoy *mass* threshold is ratio ≈ 2 (converge ×1.8 fast, ×1.98 slowly,
  locked from ×2.25 — h_d=0.25 still on the decoy at 150k with std collapsed).
  **Convergence time diverges as the boundary is approached** (0.20: 40k;
  0.22: 150k) — the bifurcation is real, not a horizon artifact.
- **s_ej — the ejection (moment-compensation) edge.** New failure mode found
  by the random-game sweep (exp8-C, exp8b-C'): with **asymmetric equilibrium
  weights**, a too-broad init lets the coupling's moment-matching force expel
  one component to the box edge (means → ±3, std → 1, weight starved ~0.1)
  while the other keeps both peaks' mass — an ejection trap that occurs even
  with NO decoy in the game. Quantified (weights (0.65, 0.35), no decoy,
  init ±1.5): s0 ∈ [0.2, 0.5] converges *to the correct asymmetric weights*,
  s0 ≥ 0.8 ejects, s0 = 0.1 freezes. Symmetric weights: s_ej = ∞ (all of
  exp8-A converged at s0 = 1.5). This edge is invisible to the flow oracle
  (it lives in the coupling, which the symmetric transit never feels).

**The game-class assumption (the broad, init-free statement).** A game is
*solvable by annealed mixture-MMD* iff the window is nonempty for the given
init distance: s_dom, s_ej ≳ d_max/κ. Clean sufficient conditions (each
checkable from the game definition in closed form):

- **(A1) scale dominance**: every spurious local max of D is dominated by the
  Nash atoms at every scale — for Gaussian bumps, **height dominance h_d < h_p
  AND mass dominance h_d·w_d < h_p·w_p** (then the smoothed ranking can never
  invert, s_dom = ∞; the amplitude ratio moves monotonically from the height
  ratio at s=0 to the mass ratio at s=∞). With a margin: mass ratio ≤ 1 sits
  ×2 below the measured trap threshold.
- **(A2) bounded weight asymmetry** (or an init std below the measured
  ejection edge): keeps s_ej above the window.
- **(A3) the standing caps**: step size η ≲ τ/L² and coupling below the
  curvature cap (exp1/exp3 — otherwise cycling/divergence even in measure
  space).
- **(A4) capacity match & genericity**: K = |support|, distinct init means,
  no exact symmetric ties (exp0 scenarios / `idealized_mmd.py`).

Violating any single assumption has a demonstrated counterexample **while all
others hold**: A1 → decoy capture (H1: *no* init std works — not a tuning
failure); A2 → ejection (H3, and 6/8 random asymmetric games from broad init);
A3 → exp3's c=8 divergence / tabular Forsaken cycling; A4 → weight-starvation
and symmetric-tie freezes. Conversely, when all hold, every game tested
converged: the full exp8-A sweep (decoy masses ×0.45–×1.8 from broad init),
7/8 random asymmetric-weight decoy games from s0 = 0.3 (the eighth, the most
asymmetric (0.31, 0.69), fails A2's margin), and the light-decoy far-init
control (H2).

**Two cautionary notes.**
- The moment-mismatch weakening FAILS: a mass-dominant decoy off the
  moment-matched point is *not* rescued by the coupling (exp8-D: still traps,
  just at an uglier configuration, one component ejected). A1 cannot be
  weakened to "no *moment-matched* dominant decoy".
- Niceness is non-monotone in the well: one random game (exp8b-C' game 1)
  fails from broad init *without* its decoy and converges *with* it — a
  low-mass decoy can act as a transport stepping stone. The window statement
  absorbs this (the decoy raises s_ej's effective edge by re-routing the
  transit), but bump-counting intuition does not.

---

## The σ-nice connection: theorem statement, checklist, and usage guide

### Background: graduated optimization and σ-niceness

Hazan, Levy & Shalev-Shwartz, *On Graduated Optimization for Stochastic
Non-Convex Problems* (ICML 2016, [arXiv:1503.03712](https://arxiv.org/abs/1503.03712))
give the only general convergence theory for the anneal-the-smoothing strategy
our std schedule implements. Their objects map onto ours exactly:

| Hazan et al. (minimization) | here (maximization) |
|-|-|
| smoothed objective f_δ(x) = E_u f(x + δu) | smoothed well D_s(μ) = E_{a~N(μ,s²)} D(a) |
| smoothing radius δ, shrunk on a ladder | component std s, annealed (or self-annealed by the natural-gradient std dynamics) |
| GradOpt: optimize at scale δ, halve δ, re-optimize | the (μ, s) scale-space flow of exp8e |
| **σ-nice** f | **scale-dominant** D (exp8's A1) |

**Definition (σ-nice, adapted to maximization).** D is σ-nice on [s_floor, s0]
if for every s in that range: (i) *path property* — the maximizer of D_{s/2}
within distance s/2 of the maximizer of D_s is unique (the smoothed maximizers
form one continuous branch down to the true maximizer, no branch-jumping); and
(ii) *local strong concavity* — D_s is σ(s)-strongly concave on a ball of
radius ~3s around that maximizer. Their theorem: under σ-niceness, graduated
optimization finds the *global* optimum of a non-convex function in O(1/ε²)
first-order steps, also with noisy gradients.

For our multi-atom games the definition is applied *per atom*: the top-K modes
of D_s must each satisfy (i)+(ii) on their own branch. The decoy game is a
certified **non**-σ-nice instance: at s ≈ 0.27 the global-maximizer branch
jumps from the peak branch to the decoy branch (the flow oracle locates the
jump at 0.272; full MMD flips in (0.28, 0.30)). Two caveats on fidelity:
Hazan et al. smooth with a uniform ball and re-solve to completion at every
scale, while we smooth with a Gaussian (exactly — the mixture policy computes
E_{N(μ,s)}[D] in closed form, the smoothing is *free*) and let s move
continuously with μ; neither difference matters for the geometry, but it means
we inherit their result as a template, not verbatim.

### Theorem statement (target form; status of each piece marked)

**Setting.** Two-player zero-sum game on a box, U(a, b) = D(a) − D(b) +
c·⟨f(a), f(b)⟩, with D ∈ C², ‖f′‖ bounded, and a unique mixed Nash
μ* = Σ_k w*_k δ_{p_k} with K atoms at the strict global maxima of D, weights
identified by the moment features (Vandermonde condition of `MultiPointGame`).
Each player runs a K-component Gaussian mixture; the update is MMD: exact
entropic mirror step with magnet τ > 0 on the categorical head, Fisher–Rao
natural gradient on means/log-stds (Gaussian-head magnet = 0; see below),
std floor s_floor, init std s0, step size η.

**Assumptions.**
- **(A1) σ-nice well / scale dominance**: D is per-atom σ-nice on
  [s_floor, s0]. *Closed-form sufficient condition for Gaussian-bump wells*:
  every spurious bump (c_d, h_d, w_d) satisfies h_d < min_k h_k (height
  dominance) **and** h_d·w_d < min_k h_k·w_k (mass dominance) — then the
  smoothed amplitude h·w/√(w²+s²), which interpolates monotonically between
  height (s→0) and mass (s→∞), never lets a spurious mode overtake an atom,
  so no branch-jump exists at any scale.
- **(A2) reach**: each init mean is within the atom's basin at scale s0 and
  |μ_k(0) − p_k| ≤ κ·s0 (κ ≈ 5 at lr = 0.05 over ≤150k steps; a finite-time,
  not asymptotic, condition).
- **(A3) coupling caps**: (a) curvature cap — c·sup‖f″‖·diam(F) strictly below
  the atom curvature min_k h_k/w_k² (exp3's two-regime bound), and (b)
  ejection cap — s0 < s_ej, where s_ej = ∞ for symmetric equilibrium weights
  and decreases with weight asymmetry (measured: s_ej ∈ (0.5, 0.8) at weights
  (0.65, 0.35); no analytic form yet).
- **(A4) step size**: η ≲ τ/L², L the Lipschitz constant of the pseudo-gradient
  field (L grows with c and with min_k h_k/w_k²).
- **(A5) capacity & genericity**: K = |supp(μ*)|, distinct init means, init
  not on a symmetry-invariant set.

**Conclusion.** The last iterate converges to the Nash-realizing monotone
island at linear rate ≈ ητ once inside it, and the converged point has
NashConv = Σ_k 2h_k(1 − w_k/√(w_k² + s_floor²)) = O(s_floor²). With magnet
updates, the island fixed point walks to the (smoothing-biased) Nash.

**Status of the pieces.**
- Proved (LaTeX Lemmas 1–2, Theorem 4): lifted skewness; block-skew structure
  (only own-player concavity matters); local monotonicity + linear rate on the
  island; O(s_floor²) bias (Lemma 3, exp5: exponent measured 2.002).
- Empirically established at ~10% accuracy (exp8): the transit is governed by
  the single-component scale-space flow; A1's thresholds (mass ratio ≈ 2 at
  the trap boundary, so ratio ≤ 1 has a 2× margin); A2's κ; A3(b)'s window.
- Open: the shadowing bound (two-player symmetric MMD transit tracks the flow
  — the F(ν) ≡ 0 symmetry argument plus perturbation off exact symmetry),
  an analytic s_ej, and importing Hazan et al.'s Gaussian-smoothing variant
  verbatim for the per-atom statement.

### How to check the assumptions on a given game

1. **Get the well.** If the payoff is built as `W(a) − W(b) + coupling`, read
  D off directly (bump list for our games). Otherwise estimate D(a) =
  E_b~ν[U(a, b)] at the current opponent and grid it.
2. **A1, closed form** (Gaussian bumps): compute every bump's height and mass
  h·w; require every spurious bump below every atom bump in *both*. Margin =
  the worst spurious/atom mass ratio; ≤ 1 is safe (measured trap onset ≈ 2).
3. **A1, general D**: build the scale ladder D_s for s on a grid of
  [s_floor, s0] (one FFT convolution each), track the local maxima: the top-K
  modes must neither disappear, merge with a spurious mode, nor be overtaken
  in height along the ladder. This is the general σ-nice check (mode-tree
  persistence), no bump structure needed.
4. **A2 + A1 jointly**: run `exp8e_flow_oracle.py`'s `flow()` from each
  intended init mean (microseconds): it returns the landing point; require
  all K atoms hit, distinct. Sweep s0 to print the usable window; empty
  window ⇒ change the init means (transport), not the hyperparameters.
5. **A3(a)**: compare c·‖f″‖·diam(F) against min h_k/w_k² (for `MultiPointGame`
  features, ‖f″‖ and diam(F) are explicit polynomials in the u-coordinate).
6. **A3(b)**: if the target weights are asymmetric, either start s0 ≤ ~0.5·
  (atom spacing) or verify with one cheap 2-component `idealized_mmd` run —
  ejection is visible within 2k steps (a mean racing to the box edge with
  growing std).
7. **A4**: if a run oscillates/diverges at lr η, halve η before touching
  anything else (exp1 Forsaken and exp3 c=8 both converge again at smaller η).
8. **A5**: set K = |support| when known. If unknown, note K > support gives
  starvation freezes and K < support cannot represent μ* — prefer counting
  modes of D at the floor scale (step 3 gives this for free).

### Is the magnet beneficial? (theory answer, from exp6 + exp8)

The magnet's role is asymmetric across the two heads, and exp8 does not change
exp6's verdict — it sharpens *why*:

- **Categorical head: essential.** The payoff is exactly linear in π, so this
  head has zero own-curvature anywhere; the τ·KL(π‖magnet) term is the *only*
  source of the strong convexity MMD's rate needs. Removing it breaks
  convergence on any game where curvature alone can't carry selection
  (three_point: 0.55 exploitability forever). All of exp8's convergent runs
  used it. Keep `magnet_category_kl_coef` > 0 always.
- **Gaussian head: not beneficial — set it to 0.** At every configuration that
  matters the means sit in a strongly concave smoothed well (−w_k·h/w² at the
  atoms), so local monotonicity is already paid for by curvature; the Gaussian
  magnet only (a) brakes the transit — exactly the phase exp8 shows is the
  hard part — and (b) pulls stds toward a stale snapshot. A/B: dropping it
  converges ~2× faster to ~2× lower residual, never hurts. Its one measurable
  "benefit" (spectral radius 0.99988 → 0.9913) lives 99.7% in the log-std
  direction, i.e. the exploitability-irrelevant O(s²) bias mode.
- **No magnet fixes transport.** The magnet is an equilibrium-*selection*
  device inside a monotone island; every exp8 failure (freeze, decoy,
  ejection) is a *transport* failure that happens before any island is
  reached, and is provably magnet-invariant (the trap endpoints are fixed
  points for all τ). Interventions must act on the transit: init means inside
  basins, s0 inside the window, repulsion sweeps.

### When to use the Gaussian-mixture policy (usage guide)

Use it when all of the following hold:
- the equilibrium is expected to be a **finite mixture over isolated atoms**
  (matching-pennies-like tie-breaking over a landscape) and a sane bound on
  the support size is available — capacity must be matched, not overshot;
- the payoff's self-term D is smooth and **scale-dominant** (A1) — or, if it
  is not, the init means can be placed inside the atoms' basins directly
  (narrow-init mode: A1 is only needed on [s_floor, s0], so small s0
  weakens it — at the price of needing better init means, the A2/A1 trade);
- action dimension is small, or the game is separable so an independent
  mixture per axis represents the product equilibrium (`MultiDim.md`; a joint
  mixture would need K^dim components).

Reach for a different tool when:
- the support size is unknown/large or atoms are not isolated → the K-match
  assumption has no safe value; the EBM route (exp7) removes the capacity
  trap but introduces its own two gates (representable sharpness at atoms,
  and a non-monotone dependence on the sampling measure — pure on-policy or
  pure uniform, no half-measures);
- the well has heavy spurious mass **and** inits far from atoms (the
  exp8d-H1 regime): the window is empty, no std schedule works — this needs
  transport machinery (restarts / repulsion across Voronoi cells), which no
  MMD hyperparameter provides;
- the game is not of the shared-well + bounded-coupling form and its
  pseudo-gradient is heavily non-monotone in ways the block-skew lemma does
  not cover (e.g. `CurvaturePumpGame` at large pump): the local island theory
  itself may not apply.

Defaults that the theory endorses: init std in the window
[atom-spacing/5, s_dom) (compute s_dom with the flow oracle; use the *lower*
half of the window when weights are asymmetric); magnet on the categorical
head only; std floor set by the target exploitability via ε ≈ Σ 2h_k·s²/2w_k²
(invert the exp5 law); anneal (`anneal_std_from`) only after A1 is verified —
annealing is exactly the mechanism that walks you into a non-σ-nice well's
decoy.

### Dimension dependence: what survives at action dimension M > 1

(Empirical anchors: `MultiDim.md` / `idealized_mmd_multidim.py`, the separable
`MultiDimDecoyWellGame` with a joint diagonal-Gaussian mixture.)

**Dimension-free — hold verbatim for any M.**
- Layers 1–2 (lifted skewness; block-skew ⇒ only own-player concavity
  matters): neither argument mentions the action dimension.
- The local island theorem: the own-mean Hessian becomes an M×M block per
  component; a strict local max of D_s is still strongly concave, so islands,
  the ρ ≈ 1 − ητ rate, and the magnet verdicts (categorical essential,
  Gaussian magnet off) carry over unchanged.
- The flow oracle: μ ∈ R^M with the same (μ, s) natural-gradient flow.
- Confirmed: no-decoy games converge at dim 2 and 3 exactly like 1-D
  (MultiDim exp1–2).

**Quantitative degradation — the M-scaling laws.**
- **Mass dominance exponentiates.** An isotropic bump (h, w) smoothed at
  scale s has amplitude h·(w²/(w²+s²))^{M/2} ≈ h·(w/s)^M: the effective mass
  is **h·w^M**, so A1's closed-form condition becomes h_d·w_d^M < h_p·w_p^M.
  The default decoy (9× wider than a peak) has mass advantage 6.3 in 1-D,
  ≈57 in 2-D, ≈510 in 3-D — wide-low traps get *exponentially* worse with
  dimension (why no knob rescued MultiDim exp4–8, and only the repulsion
  sweep, a transport intervention, escaped in exp9). Equivalently the height
  headroom h_p/h_d enters the dominance-scale equation with exponent 1/M, so
  **s_dom shrinks as M grows**. Untested sharp prediction: the 1-D trap
  threshold (mass ratio ≈ 2) should appear in 2-D at per-axis
  h_d ≈ 2·h_p·(w_p/w_d)² ≈ 0.025 — an ~9× lower height threshold.
- **The window closes from both sides.** Non-separable init distances scale
  like √M (s_reach grows) while s_dom shrinks ⇒ for a fixed game family there
  is a finite critical dimension beyond which the window is empty for natural
  inits. Separable games with per-axis-symmetric inits keep the 1-D s_reach
  per axis; only the s_dom shrinkage bites.
- **The bias law gains a factor M**: residual exploitability ≈ M·h·s²/w²
  (trace over axes) ⇒ the std floor must shrink like 1/√M for the same
  target.
- **σ-nice certification weakens.** 1-D Gaussian smoothing can never create a
  local max (scale-space causality), so bump-wise dominance certifies the
  whole mode tree. In M ≥ 2 smoothing *can* create modes (and sums of ≥3
  Gaussians can have more modes than components): the bump-wise inequality
  still rules out bump-vs-bump overtakes but no longer excludes *combination*
  modes between close atoms. The numerical mode-ladder check (checklist step
  3) remains valid at any M and is the honest certificate there.

**New at M ≥ 2 — the shared categorical head couples the axes** (MultiDim
exp3). A component sits on its peak in *every* coordinate, so its per-axis
Q-values add, and one axis's weight preference drags the categorical weights
of all axes at once: weight starvation is dimension-*amplified* (expl 0.38 in
1-D → ~1.6 *per axis* in 2-D; entropy and repulsion both fail to fix it).
This is a policy-architecture failure, not a game failure — a per-axis
factorized categorical would decouple it, at the price of representing only
product distributions over the per-axis supports.

**Capacity dichotomy, sharpened.** Product equilibrium (separable game):
a factorized K-per-axis policy represents it exactly and the 1-D theory
applies per axis — the recommended regime; a joint mixture would need K^M
components with K^M-way starvation risk. Genuinely non-product equilibrium
(N isolated atoms in R^M): the joint N-component mixture is the right shape
and the island theory holds, but transport crosses N! permutation islands and
M-dimensional basins — the un-analyzed regime.

**Net.** For M > 1 the theorem's assumption set survives with: h·w → h·w^M in
A1, s_dom = s_dom(M) decreasing, bias O(M·s²) in the conclusion, and one
added assumption — *either the game is separable and the policy factorized
per axis, or the shared-categorical axis-coupling amplification must be
absorbed into A5's capacity condition*. The closed-form A1 check downgrades
from "certifies σ-nice" to "necessary bump-wise condition"; certification at
M ≥ 2 is the numerical mode ladder.

---

## Conclusions

**The resolution of the puzzle.** MMD's convergence on these games is not
outside its theory — it is the theory, applied in the right space. The games are
monotone (exactly skew) in the space of mixed strategies; the mixture policy's
KL-geometry update is a Fisher–Rao chart of that space; and the chart preserves
monotonicity locally around any configuration that is (a) feature/moment-matched
(coupling term vanishes) and (b) curvature-dominated (each component mean at a
strict local max of the smoothed landscape). The Nash-realizing configuration
satisfies (a)+(b), so MMD converges linearly to it — at rate ≈ ητ, set by the
magnet, exactly as measured.

**Why it is "surprising but not supported by theory" no longer.** The missing
piece was never monotonicity of the parametric game (it is genuinely
non-monotone); it is that for zero-sum games the symmetrized Jacobian of the
pseudo-gradient field is block-diagonal (Lemma 2), so *only own-player
concavity matters*, and own-player concavity holds exactly on the islands
described above. Player interaction — however strong — cannot break
monotonicity by itself.

**The global picture (exp8).** The local island theory composes into a global
criterion: convergence from a natural init is decided by the scale-space flow
of a single Gaussian component on the smoothed well, and the assumption set —
scale dominance (mass + height dominance of the Nash atoms over every spurious
bump), bounded weight asymmetry (or moderate init std), the step-size/coupling
caps, and capacity match — carves out the convergent game class. Each
assumption has a quantified threshold, a closed-form predictor accurate to
~10% (`exp8e_flow_oracle.py`), and a counterexample that breaks the algorithm
by violating it alone. The strongest form: for a mass-dominant decoy at
sufficient init distance the init-std window is *empty* — no tuning of the
smoothing scale converges (exp8d-H1) — while the same init with the decoy's
mass reduced below the peaks' converges across the whole upper std range.

**What the theory buys practically.**
- Convergence is *island selection*: guaranteed once each component starts in
  the Voronoi/basin cell of a distinct support atom with a smoothing scale that
  keeps that atom's peak dominant. Init means *and* init std both select.
- The init std is a first-class hyperparameter with a *computable* safe
  window: run the microsecond flow oracle from the intended init against the
  game's bump description before training (exp8e). Broad-and-anneal is only
  safe under mass dominance AND weight symmetry; the moderate window
  (s0 ≈ atom spacing / 5 … dominance scale) is the robust default.
- The failures are classified, not mysterious: second islands (decoy —
  moment-matched + mass-dominant under smoothing), symmetry-invariant sets
  (exact ties), and simplex-boundary freezes (the w_k factor in the Fisher–Rao
  mean update). Each corresponds to a specific violated hypothesis of the local
  theorem, which tells you which intervention can work: transport
  (repulsion sweeps) for wrong-island inits — not magnets (equilibrium
  *selection*), not annealing (which changes *which* island is selected, in the
  decoy's favor), not capacity (spare components starve at w→0).
- Residual exploitability is the O(s²) smoothing bias; drive the std floor
  down and it vanishes quadratically.

Formal statements and proofs: [`mmd_mixture_theory.tex`](mmd_mixture_theory.tex)
(compiled: `mmd_mixture_theory.pdf`). Lemma 1 (lifted skewness), Lemma 2
(block-skew structure), Lemma 3 (O(s²) bias), Theorem 4 (local monotonicity +
linear convergence), Proposition 5 + Corollary (coexisting islands, decoy).

**Open items / future work.**
- Prove the funnel extends from the certified monotone cap (radius ~√(w²+s²))
  to the full Voronoi cell for the two-peak game (exp2's 96.7% empirical
  boundary at half-separation) — likely via a Lyapunov argument on the
  assignment-preserving region rather than monotonicity.
- ~~Quantify the island-selection threshold in init std~~ → done in exp8: the
  dominance scale s* is computable (flow oracle: 0.272 for the default decoy;
  MMD flips in (0.28, 0.30)). Remaining: turn the flow-oracle reduction into a
  theorem — a shadowing/consistency bound showing the two-player symmetric MMD
  transit tracks the single-component flow (the F(ν)≡0 symmetry argument plus
  a perturbation bound off exact symmetry), and derive the ejection edge s_ej
  analytically from the coupling force at broad scale (it is the one edge the
  flow cannot see).
- The stochastic (PPO, sampled-gradient) case: noise can kick trajectories
  between islands; the measure-space picture suggests noise helps escape
  shallow islands (cf. tabular escape in exp1) but the parametric w→0 freeze is
  noise-robust.
- The EBM-regression path (exp7): (a) explain the non-monotone measure flip —
  why both pure on-policy and pure uniform escape while every interior mixture
  sticks — and check whether it survives a *real* MCMC/SGD EBM rather than the
  idealized weighted-LS projection; (b) quantify the capacity threshold (basis
  bandwidth vs peak width w) at which escape becomes possible, analogous to the
  O(s²) bias law of exp5.
