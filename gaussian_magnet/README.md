# Gaussian-magnet experiments — when is a KL magnet on the Gaussian head beneficial?

**Question.** Can we construct a game where keeping the MMD magnet (KL proximal)
term on the **Gaussian component** of the mixture policy is *beneficial*?

**Context / prior result.** `theory/exp6_gaussian_head_magnet.py` and
`theory/THEORY.md` answer "no": across `two_point`, `three_point`, and the decoy
well, dropping the Gaussian-head magnet converges ~2× faster to ~2× lower
residual and never hurts. The magnet's value lives entirely in the *categorical*
head (whose payoff is linear in π, so it has no own curvature and needs the
magnet's strong convexity), while the Gaussian head is already strongly
**concave** in its means at every configuration that matters — the well provides
curvature `≈ −w·h/w² ≪ 0`, so the magnet is redundant there.

**The gap we exploit.** That "no" is conditional on a hidden assumption: *every
game exp6 tested has a curvature-dominated well.* MMD's magnet exists precisely
to handle the opposite case — a payoff that is **linear/skew (bilinear)** in the
strategy, where descent–ascent cycles and only a proximal term gives last-iterate
convergence. On the categorical head that case is *always* present. On the
Gaussian head it appears the moment the own-player payoff stops being strongly
concave in the mean. So: remove the well, and the Gaussian head inherits exactly
the situation that makes the magnet essential.

**Theory.** [`THEORY.md`](THEORY.md) formalizes the game class (separable /
finite-rank zero-sum games), the moment-reduction lemma, the local last-iterate
proposition (`ρ = 1 − ητ`), and the unifying dichotomy (magnet necessary iff the
head supplies no curvature) — marking what is rigorous vs. the one open piece
(global shadowing).

## Testbed (`magnet_core.py`)

Single-component (**K = 1**) diagonal Gaussians, so the categorical head is
trivial and the Gaussian head is fully isolated. 1-D box game family:

    U(μ₀,s₀,μ₁,s₁) = W(μ₀,s₀) − W(μ₁,s₁) + c·μ₀·μ₁,   W(μ,s) = −½·κ·(μ² + s²)

- **κ (own curvature)** is the knob. κ large → strongly concave own-payoff (the
  exp6 regime). **κ = 0 → pure bilinear "matching pennies in the mean"**: an
  interior saddle at the origin, own-Hessian exactly 0, payoff std-independent
  (so std cannot confound). This is the affine image of the repo's own
  `ContinuousMatchingPennies` (`games/examples.py`).
- The update is identical to `idealized_mmd.gaussian_natural_step`: Fisher–Rao
  natural-gradient ascent on `payoff + entropy − τ·KL(·‖magnet)`, means scaled by
  s², log-stds by ½, with a periodic hard magnet snapshot. **τ is the knob under
  test** (0 = magnet off; 0.2 = on).

## Results

### Exp 1 — the counterexample and the crossover (`exp1_curvature_sweep.py`)

Magnet OFF (τ=0) vs ON (τ=0.2) from an off-Nash init, sweeping κ (c=1, lr=0.05,
20k steps). "|z|" is the final iterate radius (Nash = 0; box edge = 3).

| κ | OFF final | OFF tail-max | OFF \|z\| | ON final | ON tail-max | ON \|z\| | verdict |
|-|-|-|-|-|-|-|-|
| 4.0 | 0.0000 | 0.0000 | 0.002 | 0.0000 | 0.0001 | 0.004 | both ok |
| 2.0 | 0.0000 | 0.0000 | 0.004 | 0.0001 | 0.0001 | 0.007 | both ok |
| 1.0 | 0.0001 | 0.0001 | 0.007 | 0.0002 | 0.0003 | 0.013 | both ok |
| 0.5 | 0.0003 | 0.0005 | 0.014 | 0.0004 | 0.0008 | 0.018 | both ok |
| 0.2 | 0.0034 | 0.0069 | 0.036 | 0.0008 | 0.0017 | 0.018 | both ok |
| 0.1 | 0.0273 | 0.0539 | 0.074 | 0.0003 | 0.0008 | 0.008 | both ok |
| 0.05 | 0.2252 | 0.4242 | 0.150 | 0.0000 | 0.0001 | 0.001 | **MAGNET HELPS** |
| 0.0 | **12.7688** | 12.79 | **3.014** | **0.0000** | 0.0000 | **0.000** | **MAGNET HELPS** |

At κ=0 the magnet-free head diverges to the box edge; the magnet lands exactly on
the Nash. Large κ reproduces exp6 (magnet redundant, even a hair worse). The
crossover is near κ ≈ 0.1.

### Exp 2 — mechanism: not a tuning fix, and it's *last-iterate* (`exp2_mechanism.py`)

**(A) lr-invariance (κ=0).** Plain descent–ascent on a bilinear saddle is the map
`z → z + lr·Rot(z)` with eigenvalues `1 ± i·lr·c`, modulus `√(1+(lr·c)²) > 1` for
*every* lr>0 — so it diverges at *all* step sizes. Confirmed:

| lr | OFF final | OFF \|z\| | ON final | ON \|z\| |
|-|-|-|-|-|
| 0.20 | 12.30 | 3.04 | 0.0000 | 0.000 |
| 0.10 | 11.60 | 3.01 | 0.0000 | 0.000 |
| 0.05 | 12.77 | 3.01 | 0.0000 | 0.000 |
| 0.02 | 7.32 | 2.32 | 0.0158 | 0.004 |
| 0.01 | 7.28 | 1.92 | 2.47 | 0.609 |
| 0.005 | 7.35 | 1.83 | 5.77 | 1.539 |

OFF diverges at every lr. ON converges wherever the magnet contraction (rate
`~lr·τ`) has enough horizon; at very small lr it just needs more than 20k steps.

**(B) last-iterate vs time-average (κ=0, lr=0.05).** Classic GDA on a bilinear
game has a *converging time-average* even as the last iterate blows up. The
magnet's specific contribution is **last-iterate** convergence — the thing a
deployed policy actually uses.

| | last-iterate expl | time-avg expl |
|-|-|-|
| magnet OFF | **12.77** | 0.036 |
| magnet ON | **0.0000** | 0.067 |

**(C) trajectory radius.** OFF spirals outward (1.80 → 3.01, hits box), ON spirals
inward (1.80 → 0.0000).

### Exp 3 — structural criterion (`exp3_phase_diagram.py`)

Sweep over (κ, c): the magnet helps whenever curvature loses to coupling. The
crossover κ* grows with the coupling strength c — consistent with the discrete
rotation rate `~lr·c` overwhelming the curvature contraction `~lr·κ`.

`H` = magnet helps (ON reaches Nash, OFF diverges); `.` = both converge
(curvature regime, magnet redundant → exp6); rows = κ, cols = coupling c:

| κ \ c | 0.5 | 1.0 | 2.0 | 4.0 |
|-|-|-|-|-|
| 2.0 | . | . | . | . |
| 1.0 | . | . | . | . |
| 0.5 | . | . | . | . |
| 0.3 | . | . | . | . |
| 0.2 | . | . | . | **H** |
| 0.1 | . | . | **H** | **H** |
| 0.05 | **H** | **H** | **H** | **H** |
| 0.02 | **H** | **H** | **H** | **H** |
| 0.0 | **H** | **H** | **H** | **H** |

Crossover κ* (largest κ where the magnet still helps): 0.05, 0.05, 0.10, 0.20 for
c = 0.5, 1, 2, 4 — it **grows with coupling**, as predicted: the magnet becomes
necessary once the coupling-driven rotation `~lr·c` overwhelms the curvature
contraction `~lr·κ`.

## Verdict

**Yes — such a game exists, and there is a clean structural criterion.** The
Gaussian-head magnet is beneficial exactly when the own-player expected payoff is
**not strongly concave in the mean** — canonically, a bilinear/skew mean-game
(interior saddle), where natural-gradient descent–ascent diverges in the last
iterate for every step size and the magnet's proximal strong-convexity is the
*only* thing that restores last-iterate convergence.

This does not contradict `theory/exp6` — it delimits it. exp6's "the Gaussian
magnet never helps" is true **on curvature-dominated wells** (its entire game
set), where own concavity already pays for local monotonicity. Remove that well
(κ → 0, coupling-dominated) and the Gaussian head reduces to exactly the
linear-payoff situation that makes the magnet essential on the categorical head.
The magnet's usefulness on *either* head is governed by one thing: whether that
head's own payoff supplies its own curvature.

**Practical implication.** `magnet_gaussian_kl_coef = 0` is safe on shared-well +
bounded-coupling games (the repo's `MultiPointGame`/`DecoyWell` family, where the
Nash atoms sit at strict maxima of D). It is **not** safe on games whose
equilibrium the Gaussian head must reach through a coupling-dominated /
bilinear-in-the-mean region (e.g. matching-pennies-type interior saddles, weak or
absent self-term) — there the Gaussian magnet is what converges the last iterate.

## The K=2 case — a mixed Nash over *two* actions

Exp 1–3 use K=1 (single-point Nash). Does the magnet also help when the Nash is a
genuine **two-point mixed strategy** (needs two Gaussian components)? Yes, but only
in a specific class — and the search that gets there is itself the lesson.

### Exp 4/5 — the obstruction: a well-pinned two-atom Nash does NOT benefit (`exp4_two_point.py`, `exp5_search_window.py`)

Natural first try: `MultiPointGame(peaks=(−1,+1))`, K=2, whose unique Nash is the
50/50 mix over {−1,+1}. Weaken the well curvature and crank the coupling to try to
make the means cycle. It fails to produce a Gaussian-magnet win *anywhere* in the
(width, coupling) plane:

- In every run the two **means land at ±1** regardless of the magnet — the
  Gaussian head is never the bottleneck. Failures at strong coupling are the
  **categorical weights** collapsing (0.50/0.50 → 0/1), a step-size failure the
  Gaussian magnet can't touch (and ON is slightly worse on it).

Why: a mixed Nash over two **isolated interior atoms** requires those atoms to be
strict maxima of the well, which *forces* a strong mean-curvature (~h/w²) — and
you can't make it weak without widening the bumps until the two peaks merge into
one basin (destroying the two-point Nash). The means are curvature-pinned by
necessity. This is the two-atom face of the same coin: **curvature-dominated
head ⇒ magnet redundant** (it re-confirms `theory/exp6`).

### Exp 6 — the fix: remove the well; hold two atoms with coupling alone (`exp6_nowell_twopoint.py`)

To get the K=1 flat-mean situation *with* a two-point Nash, drop the well and pin
the atoms with **moment-matching coupling only** — a linear and a quadratic
feature:

    U(a,b) = c₁·(E[a]₀)(E[a]₁) + c₂·(E[a²]−1)₀·(E[a²]−1)₁          (no well)

A player is unexploitable iff `E[a]=0` and `E[a²]=1`, i.e. the **50/50 mix over
{−1,+1}**. The best-response curvature in the mean is `2·c₂·E[a²−1]_opp`, which
*flips sign* with the opponent's spread error and is **exactly 0 at the Nash** —
so the mean has no own curvature, exactly the K=1 bilinear situation. Categorical
magnet ON in both; only the Gaussian magnet varies:

| variant | final | tail | tail-max | final means | weights |
|-|-|-|-|-|-|
| gauss magnet OFF | 1.02 | 1.45 | **5.75** | [−0.50, +1.57] | 0.27 / 0.73 |
| gauss magnet ON | **0.0000** | 0.0000 | 0.0000 | [−0.54, +1.47] | 0.27 / 0.73 |

Both reach a genuine **two-component** configuration, but magnet-OFF **cycles**
(exploitability oscillates, tail-max 5.75) while magnet-ON converges to the
two-point Nash in the last iterate. Same mechanism as exp1–3, now on a mixed
strategy over two actions.

**Caveat (value-degeneracy) — Part 1 actually works at K=1 (`exp7_k1_check.py`).**
Like `ContinuousMatchingPennies`, this no-well game admits a *single* Gaussian
N(0,1) as a value-Nash (E[a]=0, E[a²]=1), so it does **not** strictly require two
components. Run with one Gaussian and the dynamics split into two independent
bilinear matching-pennies games — one in the mean μ (Nash 0), one in the spread s²
(Nash 1), both flat — so magnet-OFF cycles (tail-max 107) and magnet-ON converges
exactly to N(0,1). Part 1's two components are real in the run but not
load-bearing. Adding a matched **quartic** feature `a⁴−1`
(a single Gaussian has `E[a⁴]=3 ≠ 1`) makes one Gaussian strictly exploitable
(single-Gaussian NashConv 3.2), forcing K=2. There, magnet-OFF diverges (tail 84)
while magnet-ON still reaches a genuine two-point near-Nash `[−1.94, +0.37]`,
w `0.16/0.84` (tail 0.014). The quartic's unbounded growth makes the magnitudes
numerically delicate (hence kept as a secondary Part 2), but the qualitative
OFF-diverges / ON-converges split is intact.

### K=2 takeaway

A two-point-mixed-Nash game benefits from the Gaussian magnet **iff the two atoms
are held by coupling (flat mean-curvature), not by a well (strong mean-curvature).**
The well that most naturally creates two isolated atoms is exactly what removes the
benefit — so the counterexample lives in the *no-well, moment-matched* class, the
two-component sibling of the K=1 bilinear game.

## Reproduce

```
python gaussian_magnet/exp1_curvature_sweep.py     # K=1: counterexample + crossover
python gaussian_magnet/exp2_mechanism.py           # K=1: mechanism (last-iterate, lr-invariance)
python gaussian_magnet/exp3_phase_diagram.py       # K=1: curvature-vs-coupling criterion
python gaussian_magnet/exp4_two_point.py           # K=2: well-pinned atoms -> no benefit
python gaussian_magnet/exp5_search_window.py       # K=2: obstruction is robust
python gaussian_magnet/exp6_nowell_twopoint.py     # K=2: no-well -> magnet essential
python gaussian_magnet/exp7_k1_check.py            # confirms exp6 Part 1 works at K=1
```
