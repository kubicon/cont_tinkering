# theory2 — research notes: the broadest game class on which mixture-MMD converges

Working log for the question: *define as broad a class of continuous-action
zero-sum games as possible such that MMD on a Gaussian-mixture policy converges
to a Nash equilibrium*, under the standing assumptions

- the Nash equilibrium has **finite support**, of size `M` (per player), and
- the policy uses **`K ≥ M` Gaussian components**,
- multi-dimensional actions wherever possible.

Everything here is analysis, not experiment. Source material: `theory/THEORY.md`,
`theory/mmd_mixture_theory.tex`, `gaussian_magnet/THEORY.md`, `MultiDim.md`, and
the implementation `idealized_mmd.py` (the update these theories describe).
Claims from those files are re-derived here before being used; the ones that did
not survive re-derivation are listed in §2.

The output of this log is the class defined in
[`game_class.tex`](game_class.tex).

---

## 0. What the algorithm actually is (re-derived from `idealized_mmd.py`)

Player 0 maximizes, player 1 minimizes `u(a,b)` on compact `A ⊂ R^{n_0}`,
`B ⊂ R^{n_1}`. Policy of player 0: `μ_θ = Σ_k w_k N(m_k, diag(σ_k²))`,
`w = softmax(ℓ)`.

**Categorical head** (`categorical_mirror_update`, verified against the code):

```
ℓ⁺ = ( η q + η τ log w̄ + log w ) / (1 + η τ + η τ_ent)
```

which is exactly `argmax_p ⟨p,q⟩ − τ KL(p‖w̄) − τ_ent KL(p‖unif) − (1/η) KL(p‖w)`
— the closed-form MMD proximal step on the simplex. ✔

**Gaussian head** (`gaussian_natural_step`):

```
m_k⁺ = m_k + η σ_k² ∇_{m_k} Φ ,      ρ_k⁺ = ρ_k + (η/2) ∇_{ρ_k} Φ
Φ = U(θ, opp) + τ_ent Σ ρ − τ KL(π_θ ‖ π_θ̄)
```

with `σ` clipped to `[s_floor, 1]` (`_clip`: `jnp.clip(log_std, floor, log 1.0)`).
So a **hard scale ceiling `s_max = 1` already exists in the code** — the theory
below needs it, and it is not an extra requirement.

### 0.1 The one structural fact that organises everything

Write `Q_ν(a) = ∫u(a,b)dν(b)` (player 0's *effective landscape*) and
`Λ_s = Λ * N(0, s²)` for Gaussian smoothing. Then

```
U(θ_0, θ_1) = Σ_k w_k (Q_ν)_{σ_k}(m_k) = Σ_{k,l} w_k v_l Ā_{kl},
Ā_{kl} := E_{a~N(m_k,σ_k²), b~N(n_l,ς_l²)} [ u(a,b) ].
```

Two consequences, both exact and both requiring **no structure on `u`**:

1. **The categorical heads play a finite matrix game exactly.** `q_k = Σ_l v_l Ā_kl`,
   and the update above is *verbatim* tabular MMD on the `K_0 × K_1` matrix game
   `Ā(m, σ)`. Tabular MMD on a matrix game converges last-iterate **globally**
   (Sokota et al. 2023 — the lifted game is bilinear hence monotone). There is no
   basin structure, no init dependence, nothing to assume.
2. **The Gaussian heads do natural-gradient ascent on the smoothed effective
   landscape**, at the component's own scale:
   `ṁ_k ∝ η σ_k² w_k ∇(Q_ν)_{σ_k}(m_k)`, `ρ̇_k ∝ (η/2) ∂_{ρ_k}(Q_ν)_{σ_k}(m_k) w_k`.

**Therefore the entire non-global behaviour of mixture-MMD lives in the mean
dynamics.** Whatever class we define, its hypotheses only have to say: *the
component means find the right places*. That is the design principle of the class
in `game_class.tex`, and it is why the class can be stated as a condition on the
family of effective landscapes `{Q_ν}` rather than on the algebraic form of `u`.

### 0.2 A discrepancy in the implemented metric worth recording

The mean step carries a factor `w_k`, because `∇_{m_k} U = w_k ∇(Q_ν)_σ(m_k)`
and the preconditioner used is the *per-component* inverse Fisher metric `σ_k²`.
But the Fisher metric of the **hierarchical** policy `p(k,a) = w_k N(a|m_k,σ_k)`
has `m_k`-block `E[∂_{m_k}log p ⊗ ∂_{m_k}log p] = w_k/σ_k²`, whose inverse is
`σ_k²/w_k`. The exact hierarchical natural gradient is therefore

```
ṁ_k = η σ_k² ∇(Q_ν)_σ(m_k)        (no w_k)
```

i.e. **the implemented update is not the natural gradient of the policy it
claims; it is the natural gradient of each component's conditional law, times
`w_k`.** This single factor is the entire "weight-starvation freeze" pathology
(`theory/THEORY.md` A5, `MultiDim.md` exp3): a component whose weight is being
competed away loses the ability to move and freezes at a non-atom, permanently.
With the correct metric the freeze is impossible — a starved component keeps
climbing and stays available.

This matters for the class definition: with the implemented metric we must
*assume* `w_k ≥ w_min > 0` along the trajectory (an assumption about the
dynamics, which is ugly); with the corrected metric the assumption is not needed
at all. I state the class with a `w_min` hypothesis and flag that it is an
artifact removable by a one-line change to the preconditioner.

---

## 1. Re-derivations of the inherited results (what survives)

| claim | status after re-derivation |
|-|-|
| Lifted game is exactly skew (`Lemma 1`) | ✔ correct, needs nothing but zero-sum |
| Block-skew: `sym(J)` is block diagonal | ✔ correct; equality of mixed partials, rectangular blocks fine |
| "monotone iff each player's payoff concave in own params" | ✔ correct, **for the Euclidean field**; see §2.1 for the metric gap |
| Categorical head has no own curvature (payoff linear in `w`) | ✔ |
| Bias `= Σ_k w_k h_k(1 − w_k/√(w_k²+s²)) = O(s²)` | ✔ Gaussian bump convolution, direct |
| Gaussian-bump smoothed amplitude `h·w/√(w²+s²)` monotone between height and mass | ✔ |
| Concavity cap of a smoothed bump at radius `√(w²+s²)` | ✔ |
| Local rate `1 − ητ/(1+ητ)` | ✔ from the closed-form simplex step |
| Moment reduction for finite-rank games (`gaussian_magnet` Lemma 1) | ✔ |
| Nash support `≤ rank(C)+1` (Carathéodory) | ✔ |
| `M ≥ 2`: smoothing can create maxima; 1-D it cannot | ✔ (scale-space causality is 1-D only) |

### 1.1 A lemma the inherited theory needs but does not state: scales shrink

The `σ`-dynamics are usually described empirically ("self-annealing"). They are
provable at any configuration where a mean sits at a maximiser of its smoothed
landscape. With `Λ_s = Λ * N(0,s²I)` the heat equation gives
`∂_{(s²/2)} Λ_s = Δ Λ_s`. If `ξ(s) = argmax_{cell} Λ_s` then by the envelope
theorem

```
d/d(s²/2) [ Λ_s(ξ(s)) ] = ΔΛ_s(ξ(s)) ≤ 0,   with < 0 at a nondegenerate maximum.
```

So **the value at a branch maximum is strictly decreasing in the scale**: once a
component is at (or near) its branch, the log-std natural gradient is strictly
negative and `σ` falls monotonically to the floor. No annealing schedule is
needed to get `σ → s_floor`; a schedule only changes *which branch* you are on
while you get there. This is the honest version of "annealing delivers you into
the decoy" and it holds in any dimension (`Δ` is the Laplacian).

---

## 2. Where the inherited theory is wrong, loose, or incomplete

### 2.1 Monotonicity is asserted for a field that is then run in a different metric

`theory/mmd_mixture_theory.tex` Lemma 2 proves the *Euclidean* pseudo-gradient
field `F` has block-diagonal symmetric part, then concludes monotonicity, and
then the update actually applied is `G^{-1}F` with `G` the (block-diagonal,
per-player, state-dependent) Fisher metric. Monotonicity of `F` does **not**
imply monotonicity of `G^{-1}F` in the Euclidean sense.

The gap is repairable *locally* and I use the repair in the class:

> **Fact.** If `sym(J) ≺ 0` at a fixed point and `G ≻ 0`, then `G^{-1}J` is
> Hurwitz. Proof: `G(G^{-1}J) + (G^{-1}J)^⊤G = J + J^⊤ ≺ 0`, so `V(z)=z^⊤Gz` is a
> Lyapunov function.

So local exponential stability survives any positive-definite preconditioner,
including the Fisher metric — the linear-rate claims are safe. But **the global
statements must not be phrased as "the field is monotone"**, because the flow
that is actually run is not the flow of a monotone field in the metric in which
monotonicity was proved. The class below therefore avoids global monotonicity
entirely and uses Lyapunov/ISS arguments that are invariant to positive-definite
per-player preconditioning.

### 2.2 "Scale dominance (A1)" is neither necessary nor sufficient

`theory/THEORY.md` A1 says: every spurious bump is dominated by the Nash atoms in
both height and mass ⇒ the smoothed ranking never inverts ⇒ `s_dom = ∞`. The
computation is right (the amplitude ratio
`(h_d w_d)/(h_p w_p) · √(w_p²+s²)/√(w_d²+s²)` moves monotonically from the height
ratio to the mass ratio), but the *conclusion drawn from it is not the thing that
governs the dynamics*:

- **Not sufficient.** A dominated spurious bump is still a *local* maximum with a
  nonempty basin. A component initialised in that basin at small `σ` is captured
  forever, whatever the global ranking says. The dynamics do local ascent; they
  never compare global amplitudes.
- **Not necessary.** A spurious bump that is globally dominant is harmless if no
  component ever enters its basin along the scale path.

What is actually load-bearing is A2 (each mean in the *basin* of its atom at the
init scale) **plus** the branch-persistence half of σ-niceness (the basin
assignment is not handed off to another branch as `σ` anneals). A1 is a proxy for
"the global-max branch doesn't jump", which is the special case of
branch-persistence relevant to the specific symmetric broad init used in exp8.
The class below is stated directly in terms of basins/branches, which is both
more correct and strictly broader.

### 2.3 The failure modes are three symptoms of one condition

`theory/THEORY.md` lists three separately-measured window edges — `s_reach`
(dead zone), `s_dom` (decoy), `s_ej` (coupling ejection at asymmetric weights) —
as independent assumptions with independent counterexamples. They are not
independent. All three are the statement

> *the smoothed effective landscape `(Q_ν)_s` fails to attract the component to
> its own atom, for some opponent policy `ν` in play and some scale `s` in the
> range the run visits.*

- `s_dom`: at `s > s_dom` the decoy branch swallows the peak branch — the cell's
  attractor is the wrong point.
- `s_ej`: at large `s` the well is flattened by smoothing, the coupling term
  `⟨φ_s(a), E_ν ψ⟩` dominates, and its maximiser is at the box boundary — again
  the cell's attractor is the wrong point, this time because of the opponent, not
  the well. (This is why the flow oracle, which ignores the coupling, cannot see
  `s_ej`.)
- `s_reach`: the attractor is the *right* point but the modulus of attraction is
  `~exp(−d²/2s²)`, i.e. a rate statement, not a qualitative one.

So a single condition — *uniform attraction to one point per cell, uniformly over
opponent policies and over scales in `[s_floor, s_max]`* — subsumes all three,
with `s_reach` demoted from an assumption to a rate. That unification is the main
theoretical content of `game_class.tex`.

### 2.4 `gaussian_magnet/THEORY.md` §6 dichotomy: correct, but it is not a
dichotomy of *games*

The claim "magnet necessary on a head iff that head has no own curvature at
equilibrium" is right, and §6.1's NSD-at-interior-Nash argument is right. But the
useful object is the **per-direction** statement, not the two endpoint classes:
the own-Hessian at an interior Nash is NSD, and each of its null directions needs
a non-curvature stabiliser (magnet, or the box constraint). The class below is
stated so that both stabilisers are allowed, by imposing the attraction condition
on the *magnet-regularised* landscape `Q_ν(a) − (τ_g/2σ̄²)‖a − ā‖²` rather than on
`Q_ν` alone. With `τ_g = 0` this is the well class; with `Q` flat it is the
pure-coupling class; the mixed case is generic and is covered without a case
split.

One caveat that must be carried: in the flat case the *parameters* do not
converge (the atoms are unidentified — any moment-matched configuration is a
fixed point), only the **induced measure / moment vector** does. So the
convergence conclusion must be stated as convergence of exploitability, not of
`θ`. That is the right statement anyway.

### 2.5 The multi-dim "mass dominance `h·w^M`" law

`theory/THEORY.md` states that an isotropic bump `(h,w)` smoothed at scale `s`
has amplitude `h (w²/(w²+s²))^{M/2}`, hence effective mass `h w^M`. ✔ correct.
The conclusion drawn ("wide-low traps get exponentially worse with dimension") is
correct but again is a statement about *global ranking*, so per §2.2 it is a
statement about which trap you fall into, not about whether a trap exists. In the
basin formulation the dimension dependence enters twice and both are real:
(i) basins in `R^n` are harder to hit by an init (the covering condition costs
more components), and (ii) Gaussian smoothing in `n ≥ 2` can *create* maxima, so
"exactly `M` maxima at scale `0`" no longer certifies "exactly `M` at every
scale" — the condition has to be imposed on the whole scale range, not derived
from the bottom of it.

---

## 3. Deriving the class

### 3.1 What must be assumed, by counterexample

By §0.1 the only problem is the mean dynamics. Two hard facts bound how broad any
class can be:

**(a) Spurious attractors are fatal and unavoidable.** If `â` is a strict local
max of `(Q_ν̂)_s` and `b̂` a strict local max of `(W_μ̂)_s` where `μ̂, ν̂` are the
policies with all components at `â`, `b̂`, then that configuration is a fixed
point with `sym(J) ≺ 0` on the active face, hence exponentially stable
(`theory/THEORY.md` Prop. 5, plus §2.1's Fact for the metric). It has an open
basin of initialisations. So **no class that permits a self-consistent spurious
local-max pair can be init-robust**. Conversely, the only way to be robust to
*all* inits is to have no spurious local maxima at all. This is what forces the
"exactly `M` maxima" shape of the condition; it is not a convenience.

**(b) Attraction must be uniform in the opponent.** The landscape a component
climbs is `Q_ν`, and `ν` moves. Basins of a *time-varying* flow are not forward
invariant — a component in the right basin at time `t` can be in the wrong one at
`t+1`. So "init in the right basin" is not a closable hypothesis; what is needed
is a Lyapunov function for the mean that is **the same function for every
opponent policy**. The weakest standard condition that provides one is
star-monotonicity toward the target (quasar-concavity):

```
⟨ ∇(Q_ν)_s(a), ξ − a ⟩ ≥ ρ(‖ξ − a‖) ‖ξ − a‖ > 0     for all a ≠ ξ in the cell,
```

which makes `V(a) = ½‖a − ξ‖²` decrease **for every `ν`, every `s`**. This is
strictly weaker than concavity on the cell (concavity is false for the repo's
games: a Gaussian bump is concave only within `√(w²+s²)` of its peak, yet the
two-point game converges from far outside that radius), and strictly stronger
than "the basin contains `ξ`" (which is what fails under time variation).

**What the target is, and what it is not.** `ξ_j` is *one atom*, not a Nash — no
single action is an equilibrium when `M > 1`, and the condition never claims
otherwise. The class splits the problem in two and star-concavity owns only the
first half: *where are the atoms* (Gaussian heads, `M` parallel single-target
problems, one per cell) versus *how much mass on each* (categorical heads, one
matrix game, §0.1). That is why the condition is imposed cellwise on a
partition — globally the landscape has `M` maxima so no global star-concavity can
hold, while on each cell there is a single target and the borrowed machinery
applies verbatim.

The two halves meet at equilibrium: every atom attains `max Q_{ν*}` (support
characterisation), so the `M` targets are *tied* there, and it is the opponent's
weights that tilt the landscape until they are — the indifference condition. Each
`ξ_j` is a global maximiser at equilibrium, of a landscape the opponent has
flattened across the cells.

This also places the borrowed inequality. Quasar-convexity (Hardt–Ma–Recht,
Hinder–Sidford–Sohoni) is single-target, static, game-free — I take the
inequality and its Lyapunov argument and use them `M` times on a moving
landscape. Variational stability (Mertikopoulos–Zhou) *does* cover mixed
equilibria, but only because its strategy set is convex, so `μ*` is a single
point of it — the measure-space picture, where a one-point condition is
immediately available and no partition is needed. The parametric update does not
live in that space, and the cellwise form is the repair. (Citations from memory;
verify statements before citing formally.)

### 3.2 The tracking argument

Let `ξ_j(ν,s)` be the branch attractor of cell `j`, `r = ‖m_k − ξ_j‖`. With
`ṁ_k = η σ² w_k ∇(Q_ν)_σ(m_k)`,

```
ṙ ≤ − η σ² w_k ρ(r) + ‖ξ̇_j‖ .
```

This is an ISS estimate: the mean tracks its moving target with an error set by
the target's speed, and `r → 0` whenever `ξ̇_j → 0`. `ξ_j` moves because (i) the
opponent moves and (ii) `σ` moves. Both are *converging inputs*, so the argument
closes provided the loop gain is < 1 — which is exactly a coupling cap:

```
‖∂ξ_j/∂y‖ ≤ ‖∇φ‖ / α_0        (implicit function theorem at the branch,
                                α_0 the curvature there)
‖∂y/∂(opponent means)‖ ≤ ‖∇ψ‖
```

so the round-trip gain is `≈ ‖∇φ‖‖∇ψ‖ / (α_0 α_1)`, and the small-gain condition
is **curvature dominates coupling**, `α_0 α_1 > ‖∇φ‖ ‖∇ψ‖`. For the repo's
`c·⟨f(a), f(b)⟩` with bump curvature `h/w²` this reads `c ‖f'‖² < (h/w²)²`, i.e.
the same `O(h/w²)` cap that exp3 measures. Consistency check passed, and the
statement is now global rather than local.

**Proof architecture that this suggests** (three ISS blocks + small gain):

| block | contraction rate | input | gain |
|-|-|-|-|
| means (both players) | `η σ² w_min ρ(r)` | opponent moment `y`, own scale `σ` | `‖∇φ‖/α` |
| weights (both players) | `η τ_cat` | drift of the matrix `Ā` | `‖∇Ā‖` |
| scales | `η·|ΔQ_s(ξ)|` (§1.1, one-signed near the branch) | mean error `r` | bounded |

Each block is separately provable; the composition is a standard small-gain
theorem. This is a genuinely closable route, unlike "prove the parametric field
is globally monotone" (false) or "prove a shadowing bound against the
measure-space flow" (`theory/THEORY.md`'s open item — much harder, because the
mixture manifold is not convex and the moment map is not injective).

### 3.3 Why `K ≥ M` is the right capacity statement, and what it buys

- `K < M`: the mixture cannot realise `μ*`. Excluded.
- `K = M`: works, but requires the init to place exactly one mean per cell — a
  measure-zero-boundary condition that random inits satisfy with probability < 1.
- `K > M`: **strictly better under the class's hypotheses**, contrary to the
  repo's "extra capacity doesn't help". The repo's observation was made on the
  decoy game, i.e. *outside* the class: there, spare components are captured by
  the spurious attractor. Inside the class there is no spurious attractor to be
  captured by, so:
  - every spare component converges to *some* cell's branch attractor, i.e. to a
    genuine Nash atom;
  - duplicates in one cell are harmless: they converge to the same point, the
    categorical head splits weight between them arbitrarily, and the induced
    measure is unchanged (the induced matrix game has duplicated rows — its
    equilibrium is non-unique in `w` but unique as a measure on atoms);
  - and `K ≫ M` with a space-filling init makes the covering condition
    ("every cell gets at least one mean") automatic instead of assumed.

So the honest statement is: **`K ≥ M` plus a covering init**, and overcapacity is
the cheap way to buy the covering. The one cost is the `w_k` factor of §0.2 —
spare components starve and freeze — which is why I flag the metric fix: with the
hierarchical metric, overcapacity is free.

### 3.4 Multi-dimensional actions

Everything above is dimension-free: cells are subsets of `R^n`, `ρ` is a
one-dimensional modulus of a vector inner product, the scale range is a box
`[s_floor,s_max]^n` for the diagonal-covariance policy (the smoothing is
anisotropic and the condition is imposed for every anisotropic scale in the box),
and §1.1 uses the Laplacian. What is *not* dimension-free is **verification**:

- in 1-D, scale-space causality (the number of extrema of `Λ_s` is non-increasing
  in `s`) means "exactly `M` maxima at `s = s_floor`, none merging below `s_max`"
  certifies the whole ladder;
- in `n ≥ 2`, smoothing can create maxima, so the condition must be checked (or
  assumed) on the whole scale range.

Separable games `u = Σ_d u_d(a_d,b_d)` reduce to per-axis verification of the 1-D
condition, which is why `MultiDim.md`'s no-decoy runs behave exactly like 1-D. But
the *policy* must then be factorised per axis: with a shared categorical head over
joint components, the per-axis `q`-values add, and one axis's weight preference
drags every axis (`MultiDim.md` exp3 — the 1-D plateau at 0.38 becomes 3.29 in
2-D). This is a **policy-class** condition, not a game condition, and the class
must carry it: either the game is genuinely `M`-atom in `R^n` (joint mixture is
right), or it is separable (factorised per-axis mixture is right, joint mixture
needs `M^n` components).

### 3.5 What the class deliberately does *not* assume

- no symmetry between players (different `A`,`B`, `n_0 ≠ n_1`, `M_0 ≠ M_1`,
  different `K`, `η`, `τ`, floors);
- no finite rank of the coupling — finite rank is offered only as a *checkable
  sufficient condition* for the "uniformly over opponent policies" quantifier,
  because it collapses `ν` to `m` numbers;
- no Gaussian-bump structure — that is offered only for closed-form constants;
- no strict concavity of the landscape (§3.1(b));
- no requirement that the atoms be maxima of the *own* term `D_0`. At equilibrium
  `supp(μ*) ⊆ argmax Q_{ν*}` is a *theorem* (the support characterisation of a
  best response), not an assumption. The assumption is only that these maxima are
  nondegenerate and that there are no others.

---

## 4. The class, in one paragraph

A **bounded measurable** zero-sum game `u` on compact `A ⊂ R^{n_0}`,
`B ⊂ R^{n_1}` is
**`(M, Σ, N)`-attracting** if there is a scale range `Σ = [s_floor, s_max]` with
`s_floor > 0` strictly (see §6 — the floor is a hypothesis, and it is what lets
`u` be merely bounded measurable), a
forward-invariant set `N` of policy pairs containing the initialisation and the
equilibrium, and for each player a partition of the action set into `M_i` cells
with branch maps `ξ_j(ν, s)` such that, *for every opponent policy in `N` and
every scale in `Σ`*, the magnet-regularised smoothed effective landscape is
star-concave on cell `j` toward `ξ_j(ν,s)`, with the branch maps Lipschitz and
the round-trip branch-drift gain below 1. Plus: the branch endpoints at
equilibrium are the Nash atoms; the induced `M_1 × M_0` matrix game has a unique
equilibrium as a measure; `K_i ≥ M_i` with a covering init; and the standard
step-size/magnet caps. Formal version, with the sufficiency argument and the
checkable specialisations, in [`game_class.tex`](game_class.tex).

**Why this is "as broad as possible":** by §3.1(a) any class permitting a
self-consistent spurious local maximum has an open set of failing inits, so
"exactly `M` attractors" is forced if we want init-robustness; by §3.1(b) any
class whose attraction condition is not uniform in the opponent has no Lyapunov
function for the mean and cannot close; and the three separate window edges of
the inherited theory are exactly the three ways this one condition fails (§2.3).
What remains genuinely restrictive — and, I believe, irreducibly so — is the
covering condition on the initialisation, which is a statement about the
algorithm's exploration, not about the game.

---

## 5. Open items / things I could not settle from theory alone

1. **The small-gain constant.** I have the *form* of the gain
   (`‖∇φ‖‖∇ψ‖/(α_0α_1)`) from the implicit function theorem at the branch, but
   the composition with the categorical block's gain (drift of `Ā` induced by the
   means) needs the matrix-game MMD sensitivity `∂(QRE)/∂Ā`, which is
   `O(1/τ_cat)`. Small gain then wants `τ_cat` *large*, while low bias wants the
   magnet refresh to drive `τ`-regularisation away. This tension is real and I
   have not resolved the ordering.
2. **`ρ(r)` for the repo games is exponentially small at long range**, so the
   class's asymptotic convergence is compatible with the observed finite-horizon
   freezes. Turning `ρ` into an explicit time-to-converge bound (the honest
   version of `s_reach`) is arithmetic I have not done.
3. **Whether star-concavity per cell can be weakened to "the cell is forward
   invariant and contains one critical point"** while keeping uniformity in `ν`.
   I suspect yes with an extra transversality condition on `∂A_j`, but the
   Lyapunov function then has to be built, not written down.
4. **Flat (pure-coupling) directions**: the class covers them via the
   magnet-regularised landscape, but the conclusion degrades from parameter
   convergence to moment convergence, and the magnet-refresh outer loop then
   needs its own argument (the fixed point is a set, not a point). Stated but not
   proved.
5. **The scale collapse throttles transport** (added in §6). The mean update
   carries the `σ²` preconditioner, so Step 1 (transport) and Step 2 (scales
   fall) are not sequential the way the theorem is written — the second one
   slows the first by four orders of magnitude between `σ=0.1` and `σ=1e-3`.
   Floored, this is a rate issue only. Unfloored, contraction and drift both
   vanish and the ISS estimate needs their *ratio*, which I have not checked.
6. The `w_k`-in-the-metric issue (§0.2) has a PPO analogue. `training/mixture.py`
   has no explicit preconditioner — it is a network trained by the PPO ratio loss
   — but component `k` receives gradient signal only on the steps where it was
   *sampled*, i.e. at rate `w_k`. So the same `w_k` factor appears, now as a
   sampling frequency rather than a metric, and the same starvation-freeze
   follows. The fix there is different in form (importance-weight the
   per-component loss by `1/w_k`, or sample components from a fixed exploration
   distribution) and I have not checked it is stable in the PPO objective.

---

## 6. Decision: the variance floor is a hypothesis, not an approximation

Settled after working through what `s → 0` versus `s ≥ s_floor > 0` actually
changes. **The floor is kept**, and it is promoted from an implementation detail
to a stated hypothesis of the class.

### 6.1 Why: the floor makes the class *broader*, not narrower

This is the non-obvious part and it is the whole reason for the decision.

Condition (C) is quantified over `s ∈ Σ`. With `Σ = [s_floor, s_max]` it is a
condition on `Q_{s_floor}` and above; with `Σ = (0, s_max]` it is, in the limit,
a condition on the **raw** landscape `Q`. In 1-D, scale-space causality says
smoothing only ever *removes* extrema, so the raw landscape has at least as many
local maxima as any smoothing of it. Concretely:

```
D  =  M clean bumps  +  ripple of wavelength ≪ s_floor
   →  thousands of local maxima at s = 0        (fails (C) unfloored)
   →  exactly M local maxima at s = s_floor     (passes (C) floored)
```

So every landscape whose spurious structure is finer than the floor is inside the
class with a floor and outside without. This matters practically whenever `D` is
an estimate rather than a formula.

Second broadening, from §2 of these notes: with `s ≥ s_floor` every object the
algorithm touches is `(Q_ν)_s`, which is `C^∞` with `‖∇^k(Q_ν)_s‖ ≲ ‖u‖_∞/s_floor^k`
— finite, uniform in `ν` and `s`. Since (C), (D), `prop:sharp` and `prop:cap` are
*all* stated on smoothed objects, `u` only needs to be **bounded measurable**.
That admits `|a−b|`, `min(a,b)`, hard-threshold Blotto, indicators — the
infinite-rank kernels `gaussian_magnet/THEORY.md` §7.1 lists as outside the
separable class.

Third: uniform constants ⇒ a compact scale range to check ⇒ a linear local rate
rather than a degrading one.

### 6.2 What it costs: one new condition (R)

Smoothing at `s_floor` can *merge* modes. Atoms closer than the merge scale are
invisible to the cell structure, so the class needs a resolution requirement —
for Gaussian bumps, `√(w² + s_floor²) < ½ min_{j≠j'} ‖p_j − p_{j'}‖`. This has
no analogue in an `s → 0` formulation.

The inequality is exactly the classical bimodality threshold: a bump of width `w`
smoothed at `s` has width `W = √(w²+s²)`, and two equal Gaussians at separation
`d` are bimodal iff `d > 2W`.

**It is not an independent hypothesis** — I first wrote it as a numbered
condition (R) and that was wrong. (C) already gives `M` cells with distinct
branch points at every `s ∈ Σ`, including `s_floor`, and (M) identifies those
with the atoms there; together they entail it. It is now `lem:resolution` in the
tex. Its value is one of *role*, not logical content: every other condition
constrains the game given the scale range, while this one read backwards
constrains a **hyperparameter** given the game — don't set the floor above half
the atom spacing — and it is the only part of the class with a one-line check.

Note also which term binds: at `w = 0.1`, `s_floor = 1e-3`, `d = 1`, the floor
contributes `1e-6` against the bump width's `1e-2`. So in practice it is a
statement about the *landscape* (peaks narrower than half their spacing, i.e.
visibly separate peaks at all), true with 5× margin, and the floor binds only if
pushed up toward the atom spacing.

Also forced: **(M) must be anchored at `s_floor`, not at scale 0** — the run
never reaches scale 0, so the branch endpoint is `p_j + O(s_floor²)`.

### 6.3 Exactness is not sacrificed — it moves to a corollary

The floor is not what drives the bias down; the `σ`-dynamics do. At a branch max,
`∂_{log σ}(Q)_σ(ξ) = σ²ΔQ(ξ) + O(σ⁴)` with `ΔQ(ξ) < 0`, so with
`c = (η/2)w_k|ΔQ|`:

```
σ̇ = −c σ³   ⇒   σ(t) = (σ₀⁻² + 2ct)^{−1/2} = Θ(t^{−1/2})
             ⇒   ε(t) = Θ(1/t)              (via ε ∝ σ²)
```

Transport survives the collapse but only just: the mean contraction carries `σ²`,
so accumulated contraction is `∫σ²dt = Θ(log t)` — divergent, hence still
convergent, logarithmically. That borderline is now open item 5.

Without `C²` at the atoms the exponent degrades rather than the conclusion
failing: a corner max gives deficit `E|N(0,s)| = s√(2/π) = O(s)`, hence
`ε = O(t^{−1/2})`. So **the measured `s²` law is a certificate of `C²` at the
atoms**, not a universal law. This is why smoothness survives in the class as an
isolated condition (B), used nowhere except the bias exponent.

### 6.4 Cross-check against existing measurements

The `σ ~ t^{−1/2}` law is testable against numbers already in
`theory/THEORY.md`. Two-peak game, `η=0.05`, `w_k=0.5`, `|D_s''(p)| = 99.3`
(exp2's own-mean Hessian `−49.64 = −0.5 × 99.3`) gives `c = 1.241`, so at
`t = 2e4` free decay predicts `σ ≈ 0.0045`.

- exp2 reports `σ = 0.0069` with the floor at `1e-3` ⇒ **the floor was never
  active**; that run's scale was still falling.
- The gap (0.0069 vs 0.0045) is the Gaussian magnet's brake, and exp6
  independently finds the near-unit eigenmode is 99.7% log-std — same
  coordinate, right direction.
- exp2's exploitability `0.0048` matches `2h(1 − w/√(w²+σ²)) = 0.0047` at
  `σ = 0.0069`.

So the residual exploitabilities quoted throughout the previous theory are
snapshots of a decaying quantity, not floor-set constants — and `ε ∝ 1/t` is a
sharp prediction for a long unfloored run.

### 6.5 Verification (`check_scale_law.py`, run 2026-07-31)

All four predictions confirmed. `theory2/results/scale_law.{json,log}`.
The run loop is a transcription of `idealized_mmd.run` with two knobs the
original hard-codes (floor, separate `tau_gauss`); run A confirms it reproduces
`run()` bit-for-bit (`|Δσ| = 0`) and matches `theory/results/exp2.json` to all
printed digits. Exploitability is measured on a 3e5-point grid — at the repo's
default 4001 points neither peak is a grid point and the BR value is short of
the true max by ~7e-6 per player, i.e. the same order as what we're measuring.

| run (2e6 steps, floor 1e-12) | fitted `c` | predicted | ratio |
|-|-|-|-|
| B: `τ_g=0` free decay | 1.249835 | 1.25 | 0.9999 |
| C: `τ_g=0.2`, atom init | 0.541092 | 0.540415 | 1.0013 |
| D: `τ_g=0.2`, spread init (−0.3,1.3) | 0.541092 | 0.540415 | 1.0013 |

`σ⁻²` vs `t` is affine with `R² ≥ 1 − 3e−8` in every run; log-log slopes are
`σ`: −0.502…−0.506 (pred −0.5), `ε`: −1.003…−1.011 (pred −1); and
`ε = 100σ²` holds to 5 significant figures throughout, so the residual is
provably pure representation bias. **D differs from C only in the fitted
intercept** — a transport phase changes where the law starts, not the law.

**P4 confirmed**: at 2e4 steps with the floor at 1e-3 the run sits at
σ = 0.006911 = 6.9× the floor. The floor was never active in exp2.

**New result (not predicted in advance).** My first estimate of the magnet
brake — linear response, `c_eff = c/2 = 0.625` — was 15% off (measured 0.541).
The reason is that the brake *saturates* within a refresh cycle. Doing it
properly: within a cycle σ is constant to O(1/t), and with `x = ρ − ρ̄`,

```
ẋ = −g + B(1 − e^{2x}),   g = c·σ², B = (η/2)τ_g
```

which in `u = e^{2x}` is logistic, `u̇ = 2u(A − Bu)`, `A = B − g`. From
`u(0)=1`, decay per cycle is `|x(T)| = (g/2B)(1 − e^{−2BT})`, so

```
c_eff / c = (1 − e^{−η τ_g T}) / (η τ_g T)
```

Swept over a 16× range in `η τ_g T`, ratios measured/predicted 0.998–1.004; the
two configs sharing `ητT = 0.5` (`τ=0.2,T=50` and `τ=0.05,T=200`) collapse onto
the same `c`, confirming dependence on the product alone. The defaults sit at
`ητT = 2`, squarely in the saturated regime — which is why linear response
fails there.


---

## 7. What the Gaussian magnet does to the class

Condition (C) is imposed on the *regularised* landscape
`Q^reg = (Q_ν)_s − (τ_g/2σ̄²)‖a − ā‖²`, so `τ_g` is inside the class definition.
Whether it broadens the class is directional.

### 7.1 Flat directions: strictly broader

Pure-coupling game (`u = ⟨φ(a), Cψ(b)⟩`, no self-term) ⇒ at equilibrium
`Q_{ν*} ≡ const` ⇒ `∇Q ≡ 0` ⇒ star-concavity fails everywhere at `τ_g = 0`.
These games are **outside** the class without the magnet and **inside** with it
(one cell, `ρ(r) = (τ_g/σ̄²)r`). So the magnet genuinely enlarges the class — by
exactly the games with no landscape curvature.

Price: the branch point is `ā`, the previous iterate, so (M) has no content and
the atoms are unidentified. Conclusion degrades to convergence of the induced
measure, not the parameters. Real convergence there happens in moment space,
where the game is bilinear and the magnet supplies the strong monotonicity —
that is `gaussian_magnet/THEORY.md` Prop 5, and it is a different theorem.

### 7.2 Curved directions: no broadening, and it deepens traps

The obstruction to (C) in the curved regime is a spurious local max, and the
magnet does not remove one. At the spurious configuration the magnet is
refreshed to itself, so `ā = â` and the magnet's mean gradient vanishes
identically: **the trap is a fixed point for every `τ_g`**, and the magnet adds
`−ητ_g I` to the Jacobian, i.e. makes it *more* stable. This is the analytic
form of the repo's "no magnet fixes transport" and of MultiDim exp6
(`magnet_coef=2.0` → 2.09 vs 1.45, worse).

### 7.3 The cost is exactly a time-rescaling

The mean update inside a refresh cycle obeys `ẋ = g − ητ_g x` for `x = m − m̄`
(linear), giving cycle-averaged velocity `βg` with

```
β = (1 − e^{−η τ_g T}) / (η τ_g T)
```

— **the same β** as the log-std (§6.5), because the log-std's linearised decay
rate `2B = ητ_g` is the same exponent. Both Gaussian-head coordinates are slowed
by one common factor, so the `(m,σ)` flow is time-reparameterised `t → βt`.

Verified two ways:

| check | predicted | measured |
|-|-|-|
| `σ⁻²` slope, `τ_g=0.2` vs `τ_g=0` at rescaled time | match | 0.14% apart |
| transport time ratio, `τ_g=2.0` | 20.0 | 20.18 |
| transport time ratio, `τ_g=0.2` | 2.313 | 2.667 |

(`τ_g=0.2` is looser: early transport happens at large σ, before the per-cycle
displacement is small enough for the leading-order rescaling.)

Consequence for the hypotheses: **asymptotic basins unchanged** (a time
reparameterisation cannot change which inits converge), **finite-horizon reach
shrinks by β**, and the bias at fixed step count is worse by `1/β`. That is the
only sense in which the magnet narrows anything.

### 7.4 The dichotomy is per direction, not per game

At an interior equilibrium the own-Hessian is NSD, so each own-direction is
either curved (magnet redundant + costly) or flat (magnet necessary). The
generic game mixes them, and "magnet on the flat null-space only" is not
expressible with a scalar `τ_g` — if any direction is flat you pay β in all of
them. A projector-weighted KL magnet is the obvious fix and, as far as I can
tell, untried in the repo.


---

## 8. (I) restated: nondegeneracy, not uniqueness of the game's equilibrium

(I) constrains the **`M_0 × M_1` matrix game on the atoms**, `A_{jl} = u(p_j,r_l)`
— not the continuous game's equilibrium structure. Working through it, two of its
three clauses were free and the third was overstated:

- *`(w*,v*)` is a Nash of `A`* — automatic (a Nash of the continuous game admits
  no profitable reshuffling among the atoms);
- *fully mixed* — already declared in the setup, since the atoms **are** the
  support (`w*, v* > 0`);
- *unique* — the only real content, and stronger than needed.

**Not needed for the conclusion.** Zero-sum equilibria are interchangeable: the
optimal-strategy sets are convex, any pair of optimal strategies is a Nash, all
share the value. So landing on *some* equilibrium of `A` is, in exploitability,
identical to landing on a designated one. And the weight block needs only
monotonicity of the matrix game, which holds regardless of multiplicity (the
magnet's QRE path selects a point). Dropping (I) weakens the theorem from
"converges to `(w*,v*)`" to "converges to a Nash of `A`" — the same statement
about `ε`.

**Needed for well-posedness in `s`.** The algorithm faces `Ā = A + O(s²)`, never
`A`, because components are Gaussians of width `s` rather than point masses.
Nondegeneracy is an *open* condition, so `Ā`'s equilibrium is unique and `O(s²)`
away and the weights are continuous in the floor. With a continuum of equilibria
an arbitrarily small perturbation can slide the solution to an extreme point:
the weights become `O(1)`-sensitive to the smoothing scale and "the limit is near
`w*`" becomes unsupportable. **(I) is a conditioning requirement, not a
convergence requirement.**

Holds by construction in the repo's games: `MultiPointGame`'s `K−1` moment
features make `w ↦ Σ_j w_j φ(p_j)` injective on distributions over the peaks
(Vandermonde in the normalised peak coordinates, nonsingular since the peaks are
distinct), so exactly one weight vector matches the target moments.

Renamed **(I) Identifiability → (I) Nondegeneracy** in the tex, with
`rem:nondeg` carrying the argument and Step 3 of the theorem softened to state
what survives without it.
