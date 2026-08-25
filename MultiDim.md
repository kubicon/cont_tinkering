# Idealized MMD on the multidimensional decoy-well game

Experiments running the **idealized** (noise-free, exact-gradient) MMD vector field on
`games.examples.MultiDimDecoyWellGame` — the separable product game whose payoff is a
sum of `dim` independent 1-D `DecoyWellGame`s. The question, as in 1-D: does the MMD
vector field on a parametric Gaussian mixture reach the Nash, or does the geometry trap it?

**Engine.** `idealized_mmd_multidim.py` (the `dim`-dimensional counterpart of
`idealized_mmd.py`). The policy is a **joint** `K`-component diagonal-Gaussian mixture
(means/log-stds of shape `(K, dim)`, one shared categorical head) — the same shape as the
PPO `MixtureActorCritic`. All expectations are closed form (per-axis Gaussian convolution
of the well + Gaussian moments of the coupling feature, summed over axes); best responses
for the exploitability separate per axis and are done by 1-D grid search. Verified: `dim=1`
reproduces `idealized_mmd.py` exactly, and exploitability is ≈0 at a diagonal Nash.

**Runner.** `python idealized_mmd_multidim.py configs/multidim/<name>.yaml`. One config per
experiment is kept under `configs/multidim/`.

**Why `K` components suffice.** The game separates, so the expected payoff depends only on
each player's *per-coordinate marginals*. A joint mixture with component `k` at
`(peak_k, …, peak_k)` (the diagonal) and weight `weights_k` has the 1-D Nash marginal in
every coordinate, so it is an exact Nash. `K` = per-axis peak count is therefore enough
capacity; a joint categorical over grid corners would need `K^dim`. The experiments ask
whether the *dynamics* find such a configuration.

Exploitability threshold for "reached the Nash": **< 0.1** (as in 1-D). Residual ~0.05–0.10
on the passing runs is the magnet proximal term + grid resolution, not a trap.

---

## Results ladder

| # | Config | Setup | Final expl | Verdict |
|-|-|-|-|-|
| 1 | `exp1_2d_2peak_nodecoy.yaml` | 2-D, 2 peaks/axis, no decoy, K=2 | **0.049** | ✅ converges |
| 2 | `exp2_3d_2peak_nodecoy.yaml` | 3-D, 2 peaks/axis, no decoy, K=2 | **0.074** | ✅ converges |
| 3 | `exp3_2d_3peak_nodecoy.yaml` | 2-D, 3 peaks/axis, no decoy, K=3 | **3.29** | ❌ weight-starvation |
| 4 | `exp4_2d_2peak_decoy.yaml` | 2-D, 2 peaks/axis + decoy, K=2 | **1.45** | ❌ decoy trap |
| 5 | `exp5_decoy_anneal.yaml` | exp4 + `anneal_std_from=1.0` | 1.46 | ❌ no help |
| 6 | `exp6_decoy_magnet.yaml` | exp4 + `magnet_coef=2.0` | 2.09 | ❌ worse |
| 7 | `exp7_decoy_k4.yaml` | exp4 + `num_components=4` | 1.46 | ❌ no help |
| 8 | `exp8_decoy_lrhi.yaml` | exp4 + `lr=0.2` | 1.45 | ❌ no help |
| 9 | `exp9_decoy_repulsion.yaml` | exp4 + annealed mean-repulsion sweep | **0.13** | ✅ escapes |

---

## Exp 1–2 — no decoys, two peaks per axis: MMD converges (any dim)

Starting from the diagonal spread init (both components on the diagonal at ±1.5), MMD pulls
the two components onto the peaks at `±1` in every coordinate and settles at the diagonal
Nash. `dim=2` reaches expl 0.049, `dim=3` reaches 0.074 (the small residual is the magnet
term). **Adding dimensions alone does not break MMD** — the separable structure means each
axis converges like the plain 1-D two-point game.

## Exp 3 — three peaks per axis, no decoy: a *new*, dimension-amplified failure

With three equal peaks per axis and K=3 (spread init covers all three initially), the
categorical head collapses almost all weight onto the **center** peak; the two outer
components starve (`w → 0`), and — because each mean's natural gradient is itself weighted
by its own `w` — a starved component can no longer move. Final expl **3.29**.

This is the "weight-starvation freeze" from `idealized_mmd.py`'s `SCENARIOS`, but **worse in
multi-D**: the same setup at `dim=1` plateaus at **0.38**, i.e. ~1.6 *per axis* in 2-D vs
0.38 in 1-D. The mechanism is the **shared categorical head**: a component sits on the same
peak in every coordinate (diagonal), so its per-axis `q`-values *add*, and the axis in which
the center peak is favored drags the weight for *all* axes at once. Extra dimensions compound
the collapse instead of averaging it out.

**Hyperparameters tried (none fixed it):**
- `entropy_coef=0.1` → weights equalize but all three means collapse onto the origin (expl 3.5).
- `entropy_coef=0.5` → uniform weights, but means freeze at `±0.72` (not the `±1` peaks), expl 4.2.
- frozen-uniform-weights + repulsion → outer components overshoot to the box walls `±2`, expl ↑.

Verdict: structural, as in 1-D. Not the decoy mechanism — flagged here because the shared
categorical head makes it a genuinely *multidimensional* pathology.

## Exp 4 — two peaks per axis **+ decoy**: the decoy trap reproduces in multi-D

The headline. Per axis: two tall/narrow peaks (`±1`, height 1, width 0.05) plus one
broad/low decoy at the origin (height 0.7, width 0.45 → ~6× a peak's mass). Box `[-3,3]`,
spread init lands at `±1.5` — inside the dead zone. Exactly as in the 1-D counterexample,
both components are pulled onto the decoy (origin) in **both** axes and freeze at a stable
**non-Nash** fixed point: final expl **1.45 ≈ 2 × 0.72** (the per-axis 1-D trap value,
summed over axes). `K=2` is enough capacity and MMD still cannot get there.

The trap fires **independently and identically in every coordinate** — the multi-D game
inherits the 1-D counterexample per axis, and the shared categorical head does not rescue it
(both components want the decoy in both axes, so weights stay 0.5/0.5 on two trapped modes).

## Exp 5–8 — the standard knobs do **not** rescue the decoy trap

Same conclusions as the 1-D `configs/idealized_decoy_well.yaml` header, now confirmed in 2-D:

- **Graduated optimization** (`anneal_std_from=1.0`, exp5): no help (1.46). Smoothing favors
  **mass over height**, and the broad decoy dominates the smoothed landscape — annealing
  *delivers* the components into the trap.
- **Strong magnet** (`magnet_coef=2.0`, exp6): **worse** (2.09). The magnet fixes equilibrium
  *selection*; this is a *transport* barrier, so pinning to the snapshot only slows escape.
- **Extra capacity** (`K=4`, exp7): no help (1.46). The two spare components starve to `w≈0`
  at the box edges; the two active ones sit on the decoy.
- **Larger step** (`lr=0.2`, exp8): no help (1.45). The dead zone has ~zero gradient; a bigger
  multiplier on ~0 is still ~0.

## Exp 9 — the mean-repulsion sweep escapes (the one knob that works)

The annealed **mean-repulsion sweep** — the single gradient-only escape in 1-D
(`configs/idealized_decoy_sweep.yaml`) — lifted to multi-D with a per-axis L1 repulsion
`coef · Σ_{i<j} ‖μ_i − μ_j‖₁`, ramped `0 → coef → 0`. Tuning mirrors the 1-D sweep: frozen
uniform weights, narrow std ceiling (`std_max=0.10`, so components stay sighted and never
widen into the decoy), light magnet (`0.05`, its pull on the mean is a brake), start *on* the
decoy. The repulsion pushes the two components apart until each clears the dead zone and locks
onto a peak: expl falls **1.47 → 0.13** (still annealing down at 40k steps — the razor-thin
peaks make the final settling slow, the same `~1/width²` cost as in 1-D).

This is the added multidimensional wrinkle working *for* us: a component must be carried out
of the decoy in every coordinate at once, and the per-axis repulsion does exactly that,
independently per axis.

---

## Takeaways

1. **Dimension alone is harmless.** The separable game converges per axis; 2-D/3-D two-peak
   cases reach the Nash just like 1-D.
2. **The decoy counterexample survives the lift to multi-D**, firing identically per axis
   (`expl → dim × 0.72`). None of anneal-std / magnet / extra-capacity / larger-lr help; only
   the annealed mean-repulsion sweep escapes — the same verdict as 1-D.
3. **The shared categorical head adds a genuinely multidimensional failure** (Exp 3): with
   ≥3 peaks per axis the weight-starvation collapse is *amplified* by dimension (0.38 in 1-D →
   3.29 in 2-D) and resisted every knob tried here. A per-axis-factored categorical (or a
   policy that decouples the heads across coordinates) is the natural thing to test next.

## Reproduce

```
python idealized_mmd_multidim.py configs/multidim/exp1_2d_2peak_nodecoy.yaml
python idealized_mmd_multidim.py configs/multidim/exp4_2d_2peak_decoy.yaml     # the trap
python idealized_mmd_multidim.py configs/multidim/exp9_decoy_repulsion.yaml    # the escape
```
