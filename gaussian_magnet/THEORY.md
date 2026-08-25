# When does the magnet help the Gaussian head? A separable-games theory

Companion theory note to the experiments in this folder (`exp1`–`exp7`, `README.md`).
It isolates the game class in which the MMD magnet on the *Gaussian* head is not
merely helpful but **necessary** for last-iterate convergence, reduces those games
to a finite monotone (bilinear) game via a sufficient statistic, and sketches how
the discrete-action MMD last-iterate proof lifts to the Gaussian-mixture policy.
It also states the unifying **dichotomy** that recovers the repo's existing verdict
(`theory/THEORY.md`, exp6: "Gaussian magnet redundant") as the complementary case.

Statements are marked **[rigorous]**, **[sketch]** (proof idea complete, details
routine), or **[open]** (genuine gap). Cross-references to the repo's main theory
file are written `theory/THEORY.md §…`.

---

## 0. Setup and notation

Two-player zero-sum game on a compact action set `A ⊂ R^d` (a box). Mixed
strategies `μ, ν ∈ Δ(A)`. Player 0 maximizes, player 1 minimizes the payoff to
player 0,

```
U(a, b)  with expected payoff  J(μ, ν) = E_{a~μ, b~ν}[ U(a, b) ].
```

**Policy class.** A `K`-component diagonal-Gaussian mixture with parameters
`θ = (ℓ, m, ρ)`: categorical logits `ℓ ∈ R^K`, means `m ∈ R^{K×d}`, log-stds
`ρ ∈ R^{K×d}`, weights `w = softmax(ℓ)`,

```
μ_θ = Σ_{k=1}^K w_k · N(m_k, diag(exp(2ρ_k))).
```

**MMD update** (as implemented in `idealized_mmd.py` / `exp6_nowell_twopoint.py`):
per player, per step,

- **categorical head** — the exact closed-form entropic simplex step on the
  component Q-values, with magnet weight `τ_cat` (Sokota et al. 2023);
- **Gaussian head** — Fisher–Rao natural-gradient ascent on
  `J − τ_gauss · KL(μ_θ ‖ magnet)`, means scaled by `s²`, log-stds by ½;
- a hard **magnet** snapshot every `T` steps (the anchor toward which the KL pulls).

`τ_gauss` is the knob under study. All logs base `e`; `⟨·,·⟩` Euclidean.

---

## 1. The game class: separable (finite-rank) zero-sum games

**Definition (separable game).** `U` is **separable of rank ≤ m** if there are
feature maps `f : A → R^m`, `g : A → R^n` and a matrix `C ∈ R^{m×n}` with

```
U(a, b) = f(a)ᵀ C g(b) = Σ_{i,j} C_{ij} f_i(a) g_j(b).                     (S)
```

(Dresher–Karlin; Stein, Ozdaglar & Parrilo, *Separable and low-rank continuous
games*, IJGT 2008.) The **symmetric self-play** case used in our experiments is
`f = g`, `C = diag(c)`:

```
U(a, b) = Σ_{i=1}^m c_i f_i(a) f_i(b).                                     (S′)
```

Our examples, all with centered features so that "moments matched" ⇔ `E[f] = 0`:

| game | features `f` | m | Nash | needs |
|-|-|-|-|-|
| `exp1` bilinear (K=1) | `(a)` | 1 | `δ_0` | K=1 |
| `exp6` Part 1 (no well) | `(a, a²−1)` | 2 | 50/50 on `{−1,+1}` (or `N(0,1)`) | K=1* |
| `exp6` Part 2 (no well) | `(a, a²−1, a⁴−1)` | 3 | 50/50 on `{−1,+1}` | **K=2** |

*Part 1's Nash is representable by a single Gaussian, so it does not *strictly*
need two components (`exp7`); Part 2 does (Lemma 2 below).

**What is NOT in the class.** A shared *potential/well* term `D(a) − D(b)` (as in
`MultiPointGame`, `theory/THEORY.md`) is **not** of the form (S): it is not a
finite bilinear form in features of the two players. It is exactly this term that
supplies curvature and flips the verdict (§6).

---

## 2. The sufficient-statistic reduction

**Lemma 1 (moment reduction). [rigorous]**
Let `x = F(μ) := E_{a~μ}[f(a)] ∈ R^m` and `y = G(ν) := E_{b~ν}[g(b)] ∈ R^n`.
Then for all `μ, ν`,

```
J(μ, ν) = F(μ)ᵀ C G(ν) = xᵀ C y.
```

Define the **moment bodies** `M_f = F(Δ(A)) = conv f(A) ⊂ R^m`, similarly `M_g`.
Both are compact convex. The measure game `(Δ(A), Δ(A), J)` is strategically
equivalent to the **finite-dimensional bilinear game**
`(M_f, M_g, (x,y) ↦ xᵀ C y)`:

1. `μ*` is a Nash strategy of the measure game **iff** `x* = F(μ*)` is a Nash of
   the bilinear game on `M_f × M_g`, and the values coincide.
2. Every `x ∈ M_f` is realized by some `μ` (by definition of `M_f`).

*Proof.* `J(μ,ν) = E_{μ,ν}[f(a)ᵀ C g(b)] = E_μ[f]ᵀ C E_ν[g]` by linearity of
expectation and independence of `a, b`. `F` is affine in `μ` (an expectation), so
its image of the convex set `Δ(A)` is convex, and `conv f(A)` by Carathéodory.
Because `J` depends on `μ` only through `x = F(μ)` and `F` is onto `M_f`, best
responses, Nash, and value transfer between the two games. ∎

This is the engine of everything below: **payoff-relevant information about the
policy is the finite moment vector `x`**, and in those coordinates the game is a
plain matrix game. (It is the finite-`m` special case of the "lifted" measure game
in `theory/THEORY.md §Layer 1`; here the lift collapses to `R^m`.)

---

## 3. Finite-support equilibria ⇒ why `K` Gaussians

**Lemma 2 (finite support). [rigorous]**
The bilinear game on `M_f × M_g` has a Nash `(x*, y*)`, and there is a Nash
strategy `μ* = Σ_{k=1}^{r} w_k δ_{a_k}` of the original game with

```
r ≤ rank(C) + 1  ≤  min(m, n) + 1.
```

*Proof.* Existence of `(x*, y*)` on the compact convex `M_f × M_g` for the
continuous bilinear payoff is von Neumann's minimax theorem. By Carathéodory,
`x* ∈ M_f = conv f(A)` is a convex combination of at most `m+1` extreme points
`f(a_k)`, giving `μ* = Σ w_k δ_{a_k}` with `≤ m+1` atoms and `F(μ*) = x*`. The
sharper `rank(C)+1` is Stein–Ozdaglar–Parrilo: only the projection of `x` onto the
row space of `C` (dimension `rank C`) affects the payoff, so the reduction runs in
`R^{rank C}`. ∎

**Corollary (capacity). [rigorous for point masses; O(s²) for Gaussians]**
A Gaussian mixture with `K ≥ r` components represents `μ*` exactly in the
`s → 0` limit; at std floor `s` it matches the moments up to `O(s²)` (the bias law
of `theory/THEORY.md exp5`). Conversely `K < r` **cannot** realize `x*`.

**Worked infeasibility (Part 2).** A single Gaussian `N(μ, s²)` has moment vector
`(μ, μ²+s²−1, μ⁴+6μ²s²+3s⁴−1)`. Requiring the first two to vanish forces `μ=0,
s²=1`, whence the third is `3 − 1 = 2 ≠ 0`. So `x* = 0` is **not** in the
one-Gaussian image: `K=1` is infeasible, `K=2` (atoms `±1`) is feasible. This is
`exp6`'s single-Gaussian NashConv `= 3.2 > 0` and `exp7`'s "K=1 works for Part 1
but not Part 2." **"Needs two Gaussians" is Lemma 2.**

---

## 4. Monotonicity: the game is skew in moment space

**Lemma 3 (skewness). [rigorous]**
On `Z = M_f × M_g` define the pseudo-gradient operator of the bilinear game,

```
Φ(x, y) = ( −∇_x J , +∇_y J ) = ( −C y , Cᵀ x ).
```

Then `Φ` is **skew-affine**: `⟨Φ(z) − Φ(z′), z − z′⟩ = 0` for all `z, z′ ∈ Z`.
In particular `Φ` is monotone (with equality), and the VI `⟨Φ(z*), z − z*⟩ ≥ 0`
characterizes the Nash.

*Proof.* With `z = (x,y)`, `Φ(z)−Φ(z′) = (−C(y−y′), Cᵀ(x−x′))`. Then
`⟨Φ(z)−Φ(z′), z−z′⟩ = −(x−x′)ᵀC(y−y′) + (y−y′)ᵀCᵀ(x−x′) = 0`. ∎

This is `theory/THEORY.md` Lemma 1 (lifted skewness) specialized to the finite
moment coordinates. **Key qualitative point:** the operator is monotone but has
**no strongly-monotone part** — its symmetric part is exactly `0`. Bilinear games
have no curvature; that is the whole reason a proximal/magnet term is needed for
last-iterate behavior, and why plain descent–ascent cycles (`exp1`, `exp2`).

---

## 5. The magnet and last-iterate convergence

### 5.1 In moment space (the idealization) — **[rigorous, via measure-space MMD]**

Run MMD directly on `Δ(A)` (tabular), i.e. mirror descent with a strongly convex
mirror map `ψ` (negative entropy) and the magnet proximal `τ·KL(· ‖ anchor)`:

```
μ_{t+1} = argmin_{μ}  −η⟨∇_μ J(μ_t, ν_t), μ⟩ + η τ KL(μ ‖ μ̄_t) + KL(μ ‖ μ_t),
```

symmetrically for `ν`, with anchors `μ̄, ν̄` refreshed every `T` steps.

**Proposition 4 (magnet last-iterate, tabular). [rigorous — Sokota et al. 2023]**
For a monotone game, the τ-regularized operator `Φ + τ∇²ψ` is `τ`-strongly
monotone, the MMD step is a contraction, and the last iterate converges linearly
(rate `1 − Θ(ητ)`) to the `τ`-regularized Nash `μ_τ*`; refreshing the magnet
drives `μ_τ* → μ*` (the true Nash) as the anchor tracks the iterate.

Because payoff sees only `x = F(μ)`, the induced trajectory `x_t = F(μ_t)`
converges last-iterate to `x*`. This is verified in the repo as
`theory/THEORY.md exp1` ("tabular MMD converges globally" on exactly the games
whose parametric versions have traps). **So piece (ii) of the target theorem is
settled at the measure/moment level.**

### 5.2 On the Gaussian mixture (what we actually run) — local statement

The parametric update is the Fisher–Rao restriction of the measure-space flow to
the mixture manifold `{μ_θ}`. Write the joint parameter `Θ = (θ⁰, θ¹)` (both
players) and the one-step magnet map `Θ_{t+1} = S(Θ_t; Θ̄)` with frozen anchor `Θ̄`.

**Proposition 5 (local last-iterate on the mixture; magnet-driven). [sketch]**
Let `θ*` be a Nash-realizing configuration of a separable game (S′) with **centered
features and `0 ∈ relint M_f`**, so the Nash sits at `x* = y* = 0` and hence
`C y* = 0`. Assume capacity `K ≥ r` and genericity (distinct atoms, no exact
symmetry ties; `theory/THEORY.md` A5). Then the step map at the fixed point has
spectral radius

```
ρ( DS(Θ*) ) = 1 − η τ_gauss + o(η τ_gauss)   in the mean/log-std directions,
```

so `Θ_t → Θ*` locally at linear rate `Θ(η τ_gauss)`, and this contraction is
**entirely** due to the magnet: at `τ_gauss = 0`, `ρ = 1` (marginal, cycling).

*Proof sketch.* Three ingredients.

**(a) Own-curvature vanishes at `θ*`.** For fixed opponent, `J(μ_θ, ν) =
x(θ)ᵀ C y` is *linear in the moments* `x(θ) = F(μ_θ)`. Its own-parameter Hessian is

```
∂²J/∂θ² = Σ_i (C y)_i · ∂² x_i(θ)/∂θ².
```

At the Nash `C y* = 0`, so `∂²J/∂θ²|_{θ*} = 0`. The Gaussian head has **no own
curvature** at the equilibrium — the payoff is flat in its own parameters there.
(This is the analytic version of `exp6`'s "best-response curvature
`2c₂E[a²−1]_opp + …` flips sign with the opponent and is 0 at Nash.")

**(b) Block-skew structure.** By `theory/THEORY.md` Lemma 2, for a zero-sum game
the symmetrized Jacobian of the parametric pseudo-gradient field is
**block-diagonal**: the cross-player blocks are exactly skew and cancel in
`v + vᵀ`. So the linearization of the *unregularized* natural-gradient field is
`A = J_metric · (own-Hessian ⊕ own-Hessian) + (skew cross terms)`. By (a) the
own-Hessian is `0`, leaving `A` purely skew: `A = −Aᵀ`. Its eigenvalues are
imaginary, `±iβ_j` (the rotation rate `β` set by `η · s² · C` — the coupling).
Thus `D(I + ηA) = I + ηA` has eigenvalues `1 ± iηβ_j`, modulus
`√(1 + η²β_j²) ≥ 1`: **no contraction without the magnet** (cycles/marginal,
matching `exp1`/`exp2` at `τ=0` and `exp6` OFF).

**(c) The magnet contributes `−ητ·I`.** The Gaussian-head magnet is
`−τ_gauss·KL(μ_θ ‖ μ_{Θ̄})`. Near the anchor, `KL(θ ‖ θ̄) = ½(θ−θ̄)ᵀ G(θ̄)(θ−θ̄) +
o(‖·‖²)` with `G` the Fisher metric, so its **natural** gradient (metric-inverse
times gradient) is `G^{-1} G (θ−θ̄) = (θ−θ̄)`. The magnet therefore adds
`−η τ_gauss (θ − θ̄)` to the update, i.e. `−η τ_gauss · I` to `DS`. At the magnet
fixed point `θ̄ = θ*`, the linearization becomes `I + ηA − ητ_gauss I`, with
eigenvalues `(1 − ητ_gauss) ± iηβ_j`, modulus

```
|λ_j|² = (1 − ητ_gauss)² + η²β_j²  <  1   ⟺   η < 2 τ_gauss / (τ_gauss² + β_j²),
```

a nonempty step-size window. Hence `ρ(DS(θ*)) = 1 − ητ_gauss + O(η²)` and last
iterate converges locally. ∎(sketch)

Numerically this is `exp6` (ON converges, OFF cycles) and matches the repo's
measured `ρ = 0.9913 = 1 − ητ` with `ητ = 0.05·0.2 = 0.01`
(`theory/THEORY.md exp2/exp6`). Note the categorical head still needs its **own**
magnet `τ_cat > 0` (its payoff is linear in `w` — same flatness, no curvature —
so `τ_cat` is essential independently; kept ON in all `exp6` runs).

**Remarks / where the hypotheses bite.**
- `0 ∈ relint M_f` gives `x* = 0 ⇒ Cy* = 0 ⇒` exact flatness. If the Nash sits on
  a face of `M_f` (boundary), `Cy*` may be a nonzero normal, giving partial own
  curvature; the magnet is then helpful but not strictly necessary in those
  directions. The clean "necessary" statement is the interior/centered case.
- Degenerate directions (the many-to-one moment map: several `θ` give the same
  `x`, e.g. Part 1's non-load-bearing second component) are exploitability-null;
  the magnet also *selects* a representative in that fiber (why `exp6` ON lands at
  a specific 2-point config), the continuous analogue of MMD selecting the QRE.

### 5.3 Global convergence (arbitrary init) — **[open]**

Proposition 5 is local. The global claim "`x_t → x*` from a reasonable init"
requires a **shadowing lemma**: the parametric trajectory `x_t = F(μ_{θ_t})` tracks
the measure-space MMD trajectory of §5.1, which converges globally (Prop 4). The
gap is exactly the repo's open item (`theory/THEORY.md`, "Open items"), but **more
tractable in the separable class** for two reasons:

1. *No non-separable potential to fight.* Transport happens in the convex moment
   body `M_f` (Lemma 1); there is no dead-zone/decoy geometry (those are potential
   phenomena, `theory/THEORY.md exp4/exp8`).
2. *A candidate Lyapunov function.* Both flows decrease the `τ`-regularized duality
   gap `G_τ(x,y)` of the bilinear game (strongly convex–concave after
   regularization). A bound of the form `d/dt G_τ(x_t,y_t) ≤ −c·G_τ + ε(θ_t)`,
   where `ε` measures the mismatch between the natural-gradient lift and the ideal
   mirror step on `x`, would close it — provided `ε` is controlled away from the
   **weight-starvation** boundary (a component's `w_k → 0` freezes its mean, since
   the Fisher–Rao mean update carries a `w_k` factor; `theory/THEORY.md` A5). This
   is the parametric obstruction and the crux of the open problem.

**Conjecture (target theorem).** For a separable zero-sum game (S′) with centered
features, `0 ∈ relint M_f`, capacity `K ≥ r`, and an init whose moment image lies
in the basin of the measure-space MMD flow, parametric MMD (categorical magnet
`τ_cat > 0`, Gaussian magnet `τ_gauss > 0`) converges **last-iterate** to a
Nash-realizing configuration at rate `Θ(η·min(τ_cat, τ_gauss))`. Both magnets are
necessary: with either at `0` there is a separable game on which the last iterate
cycles.

### 5.4 Finite-rank is not sufficient: the transport sub-classification

Everything in §§2–4 is a property of the *game*; §5.1 is measure-space; only §5.2
is parametric, and it is **local**. It must be stressed that **finite rank
(separability) does not imply the Gaussian-mixture dynamics converge.** It buys the
finite reduction (Lemma 1), a finite-support Nash (Lemma 2), and — via monotonicity
(Lemma 3) — global convergence of *measure-space* MMD (Prop 4). It does **not** buy
global convergence of the *parametric* update, because the moment map
`θ ↦ x = F(μ_θ)` is non-injective and the mixture manifold is non-convex, so the
parametric field can have spurious fixed points that are stable but **not Nash**.

**This is a parameterization phenomenon, not a game phenomenon.** On a separable
game the measure-space flow *always* reaches the Nash (Prop 4); the mixture
restriction may not. The same game can converge tabularly while the parametric
mixture sits at a stable non-Nash fixed point forever — the sharpest form of which
is `theory/THEORY.md exp1` (measure-space MMD escapes a config at which the
parametric mixture is a permanent stable fixed point).

**So the finite-rank games split into two sub-classes (for a given init):**

- **(1) parametrically convergent** — the transport succeeds: each component reaches
  the basin of a distinct Nash atom, after which Prop 5 contracts locally;
- **(2) parametrically trapped** — a transport trap (a stable non-Nash fixed point
  of the parametric field) intervenes, even though measure-space MMD on the
  identical game converges.

The dividing line is a **transport-niceness** condition layered on top of finite
rank — the σ-niceness family of `theory/THEORY.md` (A1 scale-dominance/σ-niceness,
A2 reach, A3 caps, A4 step-size, A5 capacity/genericity). Sub-class (1) is where
that assumption set holds; each assumption has a matched counterexample that
produces a sub-class-(2) trap while all others hold. So the target theorem's
hypothesis "*init whose moment image lies in the basin of the measure-space flow*"
(§5.3 Conjecture) is exactly this transport-niceness condition, and it is **not
implied by** separability — it is the extra, genuinely restrictive assumption.

**The niceness condition differs across the two halves of §7.** For the
**potential-curved (well) subclass** transport is graduated optimization on the
smoothed well `D_s`, so the condition is *literally* σ-niceness of `D` (mode-tree
persistence / scale dominance), and the traps are the potential-induced ones:
decoy islands, dead zones, ejection (`theory/THEORY.md exp4/exp8`). For the
**flat / pure-coupling subclass** (the magnet-necessary games of this note) there
is no well, so σ-niceness of `D` is vacuous; transport is instead driven by the
**coupling-induced landscape**

```
L_opp(a) = Σ_i c_i f_i(a) · E[f_i]_opp ,
```

which is time-varying and *vanishes identically at the Nash* (`E[f_i]_opp = 0`).
The analogue of σ-niceness here is a **niceness condition on the feature map**: the
maxima of `L_opp` must sit at, or shepherd components toward, the Nash atoms rather
than spurious points — for *every* opponent moment vector met along the transit.
This can fail:

- **Boundary escape (feature-landscape trap).** A convex, unbounded feature makes
  `L_opp` maximized at the box boundary rather than at the interior atoms. This is
  exactly `exp6` Part 2: the `a⁴` feature drives components to `±B` during transit
  (a sub-class-(2) trap), which is why the quartic run is numerically delicate even
  though the game is finite-rank and its Nash is the clean `±1` mixture. It is the
  flat-class counterpart of the decoy trap — same status (a stable non-Nash
  parametric fixed point), different mechanism (feature convexity, not spurious
  smoothed mass).
- **Weight starvation / symmetric ties** (A5) — parameterization-generic, hit both
  halves: with `K > support` a component's weight `→ 0` freezes its mean (the
  Fisher–Rao mean update carries a `w_k` factor), and exact symmetry ties are
  invariant fixed points.

So §5.3's remark that the separable class has "no dead-zone/decoy geometry" is
correct but narrow — those are *potential*-induced traps, absent when `D ≡ 0`; the
flat class simply has its **own** trap zoo (boundary escape, starvation) governed
by the feature-landscape niceness above. **Finite rank is the umbrella; transport
niceness — σ-nice `D`, or the feature-landscape analogue — is what carves out the
parametrically convergent sub-class.**

---

## 6. The unifying dichotomy: potential curvature vs. the magnet

Now re-attach a shared potential (well) `D`, spanning both classes:

```
U(a, b) = D(a) − D(b) + Σ_i c_i f_i(a) f_i(b).                            (SP)
```

This is *still* separable (a self-term `D(a) − D(b)` pairs `D` with the opponent's
constant feature — see §7), but unlike the pure coupling it contributes genuine
own-parameter curvature `∇²E_μ[D]` that survives at the Nash.

**Theorem 6 (dichotomy). [sketch]**
At a Nash-realizing config of (SP) with matched (centered) features (so the
coupling's own-Hessian vanishes, §5.2(a)), the Gaussian head's own-mean Hessian
equals the **smoothed-well Hessian alone**,

```
∂²J / ∂m_k²  =  ∇² D_s(atom_k)  =  −w_k · h_k / w_k²  <  0   (strict, for a peak),
```

with `D_s` the `N(0,s²)`-smoothed well. Consequently, in the mean/log-std
directions the local step map has

```
ρ(DS(θ*)) = 1 − η · Θ( max( κ , τ_gauss ) ),   κ := ‖∇²D_s(atom)‖ ≥ 0.
```

Two regimes:

- **(i) Curved potential (`κ > 0`; the well games).** The field is strongly
  monotone from curvature alone; last-iterate converges at rate `Θ(ηκ)` even with
  `τ_gauss = 0`. The Gaussian magnet is **redundant** (and, being a brake plus a
  pull toward a stale std snapshot, mildly counterproductive). This is the repo's
  `theory/THEORY.md exp6` verdict and this folder's `exp4`/`exp5` obstruction:
  resolving two isolated interior atoms *forces* `κ ~ h/w² > 0` (a bump narrow
  enough to be a distinct peak is steep), so no member of the well class exposes a
  Gaussian-magnet benefit.
- **(ii) Flat potential (`κ = 0`; the separable class).** Own-Hessian is `0`;
  monotone but not strongly (Lemma 3). Last-iterate convergence holds **iff**
  `τ_gauss > 0`, at rate exactly `Θ(η τ_gauss)` (Prop 5). This note's `exp1`–`exp3`
  (K=1) and `exp6`–`exp7` (K=2).

*Proof sketch.* Redo §5.2(a) with the potential: `∂²J/∂m² = ∇²(E_μ[D]) +
Σ_i(Cy*)_i∂²x_i/∂m²`. The second sum is `0` at matched Nash (`Cy* = 0`); the first
is `∇²D_s`, the closed-form smoothed-well curvature (`theory/THEORY.md exp2`,
matching `−w_k h/w²` to four digits). Insert this own-block in place of `0` in the
§5.2(b)–(c) linearization; the eigenvalues become `(1 − η(κ + τ_gauss)) ± iηβ`,
giving `ρ = 1 − ηΘ(max(κ,τ_gauss))`. Whichever of curvature `κ` or magnet
`τ_gauss` is larger sets the rate; `κ > 0` makes `τ_gauss` inessential, `κ = 0`
makes it essential. ∎(sketch)

**One-line statement.**

> The MMD magnet on a head is necessary for last-iterate convergence **iff that
> head's own payoff supplies no curvature at the equilibrium** — i.e. iff the game
> is (in that head's coordinates) separable/bilinear rather than potential-curved.
> The categorical head is always in the flat case (payoff linear in `w`), so its
> magnet is always essential; the Gaussian head is flat exactly for separable
> games, and curved whenever a well pins its atoms.

### 6.1 The full curvature taxonomy (why potential-curved and pure-coupling are endpoints)

Theorem 6 splits by "curvature `κ > 0`" vs. "`κ = 0`", which invites the question:
are *potential-curved* and *pure-coupling* the only two types of separable game? No
— **they are the two endpoints of a spectrum**, and there are two further types
that the negative/zero dichotomy misses. The organizing variable is the **signature
of the own-parameter Hessian** at the equilibrium.

**What the sign means (intuition).** The own-Hessian is the player's *self-restoring
force*: freeze the opponent, wiggle your own action, and watch your own payoff.
You are the maximizer, so read it as a ball seeking the high ground of *its own*
payoff landscape:

- **negative (concave — you sit on a hilltop):** your action is locally best *on
  its own merits*; deviating costs you, so you are self-anchored. The action has
  **intrinsic value** — a reason to be *here* regardless of the opponent (a
  well/potential). The magnet has nothing to add.
- **zero (flat — you sit on a table):** you are *indifferent* to where you sit; only
  the opponent's coupling moves you (a rotation). You have **no intrinsic
  preference** — pure matching-pennies, here only to balance the opponent. The
  magnet installs the *artificial* anchor "stay near where you just were", exactly
  the intrinsic preference the game withheld. The magnet is load-bearing.
- **positive (convex — you sit in a pit):** your spot is your own *worst* point;
  either deviation *improves* you, so you flee. The action is intrinsically *bad* —
  you were pushed into it. The magnet can hold you only if `τ` outmuscles the
  repulsion; it is fighting the game.

The opponent *modulates* this curvature: the coupling contributes `∝ f''(a)·E[f]_opp`,
so a mismatched opponent bends your self-landscape either way; at the Nash the
opponent is matched (`E[f]_opp = 0`) and this vanishes, leaving only the self-term's
intrinsic curvature — which is why the taxonomy is clean exactly at equilibrium.

**Nash optimality forces NSD at an interior equilibrium.** In moment space the game
is always skew-bilinear (Lemma 3): *zero* own curvature in `x`. All `θ`-curvature is
the moment map `x(θ)` bending, weighted by `(Cy*)_i`. At a genuine interior
Nash-realizing config `θ*` is a best response over the (representable, `K ≥ r`)
parametric class, hence a local *max* of `J(·, ν*)` in `θ`, hence

```
∂²J/∂θ²|_{θ*}  ⪯  0     (negative semidefinite).
```

A strictly positive eigenvalue would let the player improve by deviating — not a
Nash. So *at an interior equilibrium* only NSD signatures occur, classified by rank:

| own-Hessian at Nash | class | what pins each own-direction | magnet |
|-|-|-|-|
| negative **definite** (no null space) | potential-curved (§6(i)) | well curvature | redundant |
| NSD, **mixed** (some `<0`, some `=0`) | *partially curved* — **generic** | curvature ⊕ magnet, per direction | needed on the null-space only |
| **fully flat** (all-zero) | pure-coupling (§6(ii)) | magnet | necessary everywhere |

So `§6(i)` and `§6(ii)` are the extremes `rank = full` and `rank = 0`; the **generic
separable game is the mixed middle** (e.g. a 2-D action with a well in one
coordinate and matching-pennies in the other), and the magnet is required precisely
on the flat null-space. The clean two-way split is a statement about the two *pure*
regimes, not an exhaustive partition.

**Two further types, off the interior-Nash NSD constraint.** Strictly positive
curvature is excluded *at* an interior Nash but is real elsewhere, and it brings two
mechanisms neither §6 class covers:

- **(C) Non-monotone transport regions (`∂²J/∂θ² ≻ 0` off-Nash).** Nothing forbids
  positive own-curvature *away* from equilibrium. `CurvaturePumpGame`
  (`games/examples.py`) is separable (`pump·‖a₁‖²‖a₂‖²` is `f(a)g(b)`) yet gives
  player 0 own-Hessian `2·pump·‖a₂‖² I ≻ 0` whenever the opponent is off the origin
  — "the opponent controls your curvature." The Nash is fine (the pump Hessian is 0
  at the origin), but the *surrounding field is non-monotone*, so the block-skew
  argument (which needs own-concavity, `theory/THEORY.md` Lemma 2) fails nearby and
  the dynamics are pushed *away* — a transport trap the magnet cannot fix (it would
  have to overpower a positive curvature). `theory/THEORY.md` flags exactly this:
  "the local island theory itself may not apply."
- **(D) Boundary equilibria (constraint-pinned).** If the Nash atoms sit on the box
  boundary (matching-pennies pushing mass to `{0,1}`, `ContinuousMatchingPennies`),
  the second-order condition is only *one-sided*: the outward direction is held by
  the **box constraint / projection**, not by curvature and not by the magnet. The
  free-direction curvature can even be positive there. A third stabilization
  mechanism — the constraint — orthogonal to both well and magnet.

**Complete taxonomy — by what stabilizes each own-direction:**

- **curvature** (concave self-feature) → potential-curved direction; magnet redundant;
- **magnet** (flat/skew direction, interior) → pure-coupling direction; magnet necessary;
- **constraint** (boundary Nash) → constraint-pinned direction; neither well nor magnet;
- **nothing / repelling** (`∂²J ≻ 0` off-Nash) → non-monotone transit (CurvaturePump);
  a trap no magnet fixes.

`§6(i)`/`§6(ii)` are the first two; the generic game mixes them per direction; the
last two are the extra types. "`Hessian > 0`" is the right thing to point at — it
simply cannot sit *at* an interior Nash (optimality), so it surfaces as boundary
equilibria (D) and non-monotone transport obstructions (C) instead.

---

## 7. Relation to the well class (`theory/THEORY.md`)

How does the separable class here sit against the "shared-well + bounded-coupling"
class the repo's main theory file is built on? Not disjoint, and not the same: **the
well class is a *proper subclass* of the separable class**, and the distinguishing
axis is not separability at all but *which feature is active at the Nash*.

### 7.1 A shared potential is itself separable

The well payoff `U = D(a) − D(b) + c⟨f(a), f(b)⟩` looks like it carries an extra
ingredient `D` outside form (S). It does not. A one-player function is trivially
separable — pair it with the **opponent's constant feature**:

```
D(a) − D(b) = D(a)·1 + 1·(−D(b)).
```

So the well game is separable with augmented feature maps
`F(a) = [D(a), 1, f₁(a), …]`, `G(b) = [1, D(b), f₁(b), …]` and a suitable `C`. `D`
counts as **one feature** however complicated it is (a rank-1 block), so
`D ∈ C²` "arbitrary" is still finite rank. Consequences:

- **Every game in the repo is separable.** `MultiPointGame`, `DecoyWellGame`,
  `ForsakenGame` (`a₁a₂ − 0.45a₂ + φ(a₂) − φ(a₁)`), `CurvaturePumpGame`
  (`pump·‖a₁‖²‖a₂‖²` is `f(a)g(b)`), `AsymmetricWellGame`, `QuadraticZeroSumGame`
  — all finite sums of products. Lemmas 1–3 (moment reduction, finite support,
  skewness) therefore apply to all of them.
- **What is *outside* separable:** genuinely infinite-rank kernels — `exp(a·b)`,
  `|a−b|`, `min(a,b)` — whose feature expansion never terminates. None occur in the
  repo, but they are where the finite moment reduction (Lemma 1) fails and only the
  universal (infinite-dimensional) lifted monotonicity of `theory/THEORY.md`
  Layer 1 survives.

### 7.2 Two feature types — the real distinction

Since separability is shared, it cannot be what separates the two classes. The
distinction is **which kind of feature is active at the equilibrium**, i.e. exactly
the `Cy*` computation of Prop 5. In the separable representation features come in
two types:

- **Self-features** — a one-player function paired with the opponent's *constant*
  feature, `D(a)·1`. The opponent's factor is the constant `1`, which is **never
  zero**, so this term contributes persistent own-curvature `∇²E_μ[D]` at the Nash.
  Its `(Cy*)`-coefficient is the constant, *not* a matched-to-zero moment.
- **Coupling features** — `f_i(a)` paired with the opponent's *centered,
  non-constant* `f_i(b)`. At the Nash the opponent's moment is matched,
  `E[f_i] = 0`, so `(Cy*)_i = 0` and these contribute **zero** own-curvature (the
  `Cy* = 0` step in Prop 5).

The magnet is load-bearing exactly when the *active* curvature — from self-features
only — vanishes. That is the clean split **inside** the separable class:

- **well class = separable + a curving self-feature** (`D` strictly concave at the
  atoms) ⇒ own-Hessian `< 0` ⇒ magnet redundant (Thm 6(i); `theory/THEORY.md`);
- **pure-coupling class = separable with no curving self-feature** ⇒ own-Hessian
  `= 0` ⇒ magnet necessary (Thm 6(ii); this note's `exp1`, `exp6`).

The two are **complementary within separable**, neither nested in the other:
`exp6` has no potential at all (only `D ≡ 0` would force it into the well *form*,
but then the atoms are not strict maxima of `D`, violating the well theorem's
hypothesis), and the well games have a genuinely concave `D`. Both are proper
subsets of separable, which is a proper subset of all zero-sum games.

### 7.3 The containment picture

```
all zero-sum games            — lifted (measure-space) game is skew/monotone
                                (theory/THEORY.md Lemma 1); universal, no finiteness
  ⊃ separable / finite-rank   — lift COLLAPSES to a finite bilinear game on the
                                moment body M_f ⊂ R^m (Lemma 1); finite support
                                (Lemma 2). ← the umbrella of THIS note
      ⊃ potential-curved      — a self-feature D is strictly concave at the atoms;
        (= the well class)      own-Hessian < 0 ⇒ MAGNET REDUNDANT.
                                ← theory/THEORY.md's convergence theorem
      ⊃ flat / pure-coupling  — no curving self-feature; own-Hessian = 0,
                                only skew ⇒ MAGNET NECESSARY.
                                ← exp1–exp3 (K=1), exp6–exp7 (K=2)
```

**One-line comparison.** `theory/THEORY.md` studies *separable games whose
self-term supplies curvature* — so its narrative is transport into
curvature-pinned islands and the magnet only does equilibrium *selection*. This
note studies *separable games whose self-term supplies no curvature* — so the
magnet must supply the strong monotonicity itself. The object unifying both, new
relative to either file, is the **moment reduction**: every separable game is a
finite bilinear game in `x = E[f]`, and whether the magnet is load-bearing is
decided by whether any *self*-feature is active-and-concave at `x*`.

---

## 8. Summary of the proof status

| piece | statement | status |
|-|-|-|
| Lemma 1 | moment reduction to a finite bilinear game | **rigorous** |
| Lemma 2 | finite-support Nash (`≤ rank C + 1`) ⇒ capacity `K` | **rigorous** (SOP 2008) |
| Lemma 3 | skew-monotone in moment space | **rigorous** |
| Prop 4 | magnet last-iterate in measure/moment space | **rigorous** (Sokota 2023; repo exp1) |
| Prop 5 | local parametric last-iterate, `ρ = 1 − ητ_gauss` | **sketch** (linearization complete) |
| §5.4 | finite rank ⇏ parametric convergence; transport sub-classification | **rigorous** (framing) + open constants |
| §5.3 | global shadowing (parametric tracks measure-space) | **open** |
| Thm 6 | dichotomy: curvature `κ` vs. magnet `τ_gauss` | **sketch** |
| §6.1 | full curvature taxonomy (NSD at Nash; endpoints + constraint/non-monotone types) | **rigorous** (NSD argument) + framing |

**The one genuinely open problem** is the global shadowing bound (§5.3): show the
Fisher–Rao natural-gradient lift on the mixture manifold tracks the (globally
convergent) measure-space MMD flow in moment coordinates, with the mismatch
controlled away from the weight-starvation boundary. Everything else is either
standard (Lemmas 1–3, Prop 4) or a completed linearization (Props 5, Thm 6).

## References

- M. Sokota, R. D'Orazio, J. Z. Kolter, et al. *A Unified Approach to RL, QRE, and
  Two-Player Zero-Sum Games* (Magnetic Mirror Descent), ICLR 2023. — the magnet
  last-iterate proof (Prop 4) to lift.
- N. Stein, A. Ozdaglar, P. Parrilo. *Separable and low-rank continuous games*,
  Int. J. Game Theory, 2008. — the class (Def §1), Lemma 2.
- S. Karlin; M. Dresher. Classical polynomial/separable game theory — finite
  support.
- P. Mertikopoulos, Z. Zhou; N. Golowich, S. Pattathil, C. Daskalakis. Last-iterate
  of OMD/OGDA in monotone/bilinear games — the moment-space engine.
- Y.-P. Hsieh, P. Mertikopoulos, et al.; C. Domingo-Enrich, et al. Fisher–Rao /
  Wasserstein gradient flows for games over measures — the natural language for the
  lift (§5.3).
- `theory/THEORY.md` (this repo) — lifted skewness (Lemma 1), block-skew structure
  (Lemma 2), local islands, exp1/exp2/exp5/exp6 used above.
```
