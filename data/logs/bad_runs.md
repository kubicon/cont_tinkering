# two_point_peaks sweep: runs that failed to converge

Analysis of the 9 sweep logs in `logs/two_point_peaks_*`, each containing 15 runs
(`network.num_components` ∈ {2,3,4} × `train.seed` ∈ {0..4}, 50k iterations).
A run is judged "bad" if, at the final checkpoint, the target strategy of either
player does not have its probability mass split across both true peaks with
roughly the right location and weight (tolerance: weight ±0.08, location ±0.15).

**28 / 135 runs (21%) failed.** All 28 are `num_components=2` (24) or `num_components=3` (4).
Every `num_components=4` run converged correctly (0/45).

## The common failure trajectory

The important pattern is **not** that these runs never found the two peaks — almost all of
them do, early on. What breaks them is a **slow drift away from a good solution during the
back half of training**. For every single bad run, the *minimum* exploitability reached at
some earlier checkpoint is much lower than the *final* exploitability at iter 50000:

| example | min exploitability (iter) | final exploitability (iter 50000) |
|---|---|---|
| `-1.0-1.0 weights_0.7-0.3` nc=2 seed=2 | 0.054 (@19500) | 0.367 |
| `-1.5-0.0 weights_0.2-0.8` nc=3 seed=0 | -0.300 (@9500) | 0.953 |
| `0.5-2.0 weights_0.2-0.8` nc=2 seed=1 | 0.052 (@33000) | 0.933 |
| `-1.5-0.0 weights_0.7-0.3` nc=2 seed=4 | 0.036 (@30500) | 0.901 |

So training typically reaches a near-optimal mixture (both peaks located correctly,
weights close to target) somewhere between iter 5,000–30,000, and then **regresses**
instead of staying there or continuing to improve. Two distinct drift patterns show up:

### Pattern A — weight-ratio drift (locations stay correct, mixture weights don't)
Most common with skewed target weights (0.2/0.8 or 0.7/0.3) and `num_components=2`,
where there's no redundant component to reallocate mass into. Example,
`-1.0-1.0_weights_0.7-0.3`, nc=2, seed=0, p1 target strategy:

```
iter  5500: 0.71x(-1.00) 0.29x(0.98)   <- correct (target 0.70/0.30)
iter 20500: 0.60x(-0.99) 0.40x(0.99)
iter 50000: 0.60x(-1.00) 0.40x(1.03)   <- drifted to ~0.60/0.40
```
The peak locations stay pinned throughout — only the relative weighting drifts, and it
never recovers. Since one of only two components carries a minority weight, it gets a
comparatively weak/noisy PPO gradient signal and the ratio wanders under continued
policy updates instead of staying locked at the optimum found around iter 5–10k.

### Pattern B — a minority-weight component drifts off the valid domain entirely
The most severe failures. When a component's assigned weight is small (~0.05–0.20), the
signal pulling its *location* toward the true peak is weak enough that it can wander
outside the game's actual domain and never come back — effectively "losing" that peak.
Example, `-1.0-1.0_weights_0.2-0.8`, nc=2, seed=0, p2 target strategy:

```
iter   500: 0.30x(-0.87) 0.70x(1.09)   <- both peaks found correctly
iter  5500: 0.17x(-2.67) 0.83x(0.99)   <- minority component already drifting
iter 50000: 0.19x(-2.91) 0.81x(1.01)   <- ends up locked ~3 units away from the true peak at -1.0
```
This same pattern (minority component escaping to a wildly wrong location) recurs in
several other seeds/configs below.

### Extra components in num_components=3 runs
In the 4 failing `num_components=3` runs, the 3rd component simply collapses to ~0.00
weight (a harmless "dead" component) — the run degenerates into an effective
`num_components=2` case and inherits Pattern A. Example, `-1.0-1.0_weights_0.7-0.3` nc=3
seed=0, p1: `0.70x(-1.00) 0.00x(-0.30)±wide 0.30x(1.01)` at iter 10500, drifting to
`0.58x(-0.99) 0.00x(-0.05) 0.42x(1.08)` by iter 50000 — same weight-ratio drift as nc=2,
just with a spectator component along for the ride.

`num_components=4` never showed this failure mode in the sweep — with two spare/dead
components available, there's apparently enough redundancy that the live components'
weight ratio stays anchored near the optimum instead of drifting.

## Full list of failed runs

### `two_point_peaks_-1.0-1.0_weights_0.2-0.8` (peaks -1.0/1.0, weights 0.2/0.8) — 3/15 bad
| run | issue | min→final exploitability |
|---|---|---|
| nc=2 seed=0 | p1 weight drifts to 0.05/0.95 (want 0.20/0.80); p2's -1.0 component drifts to **-2.91** (Pattern B) | 0.444@42000 → 0.943 |
| nc=2 seed=1 | same weight drift on p1; p2's -1.0 component drifts to **-2.99** | 0.453@8000 → 1.012 |
| nc=2 seed=4 | same weight drift on p1; p2's -1.0 component drifts to **-2.78** | 0.324@16000 → 0.957 |

### `two_point_peaks_-1.0-1.0_weights_0.7-0.3` — 6/15 bad
| run | issue | min→final exploitability |
|---|---|---|
| nc=2 seed=0 | p1 weight drifts to 0.60/0.40 (want 0.70/0.30) — Pattern A | 0.099@10000 → 0.374 |
| nc=2 seed=1 | p1 weight drifts to 0.60/0.40 | 0.063@33000 → 0.399 |
| nc=2 seed=2 | p1 weight drifts to 0.61/0.39 | 0.054@19500 → 0.367 |
| nc=2 seed=4 | p1 weight drifts to 0.54/0.46 | 0.144@10500 → 0.489 |
| nc=3 seed=0 | 3rd component dies to ~0; p1 weight drifts to 0.58/0.42 | 0.050@10500 → 0.714 |
| nc=3 seed=3 | 3rd component dies to ~0; p1 weight drifts to 0.56/0.44 | 0.060@4500 → 0.481 |

### `two_point_peaks_-1.5-0.0_weights_0.2-0.8` (peaks -1.5/0.0) — 5/15 bad
| run | issue | min→final exploitability |
|---|---|---|
| nc=2 seed=0 | p1 weight drifts to 0.30/0.70 (want 0.20/0.80) | -0.051@19000 → 0.402 |
| nc=2 seed=1 | p1 weight drifts to 0.06/0.94; p2's -1.5 component drifts to **-3.46** (Pattern B) | 0.239@15000 → 0.733 |
| nc=2 seed=2 | p1 weight drifts to 0.07/0.93; p2's -1.5 component drifts to **-3.57** | 0.150@21000 → 0.927 |
| nc=2 seed=3 | p1 weight drifts to 0.07/0.93; p2 weight off 0.12/0.88 and -1.5 component drifts to **-3.43** | 0.135@8000 → 0.693 |
| nc=3 seed=0 | both players' weight ratio drifts to ~0.28-0.29 / 0.72 | -0.300@9500 → 0.953 |

### `two_point_peaks_-1.5-0.0_weights_0.7-0.3` — 5/15 bad
| run | issue | min→final exploitability |
|---|---|---|
| nc=2 seed=0 | p1 weight drifts to 0.59/0.41 (want 0.70/0.30) | 0.035@14500 → 0.619 |
| nc=2 seed=1 | both players' weight drifts to 0.83/0.17; p2's 0.0-peak component also drifts to **1.82** | 0.364@38000 → 0.532 |
| nc=2 seed=2 | p1 weight drifts to 0.57/0.43 | -0.129@7500 → 0.449 |
| nc=2 seed=3 | p1 weight drifts to 0.61/0.39 | -0.115@6500 → 0.443 |
| nc=2 seed=4 | p1 weight drifts to 0.48/0.52 (essentially flips priority) | 0.036@30500 → 0.901 |

### `two_point_peaks_0.5-2.0_weights_0.2-0.8` — 5/15 bad
| run | issue | min→final exploitability |
|---|---|---|
| nc=2 seed=0 | p1 weight drifts to 0.28/0.72 (want 0.20/0.80) | -0.159@15000 → 0.273 |
| nc=2 seed=1 | p1 weight drifts to 0.06/0.94; p2's 0.5-peak component drifts to **-1.34** (Pattern B) | 0.052@33000 → 0.933 |
| nc=2 seed=2 | p1 weight drifts to 0.07/0.93; p2's 0.5-peak component drifts to **-1.39** | 0.150@12000 → 0.528 |
| nc=2 seed=3 | p1 weight drifts to 0.07/0.93; p2's 0.5-peak component drifts to **-1.38** | 0.041@41500 → 0.500 |
| nc=3 seed=0 | p1's 0.5 component location drifts to 0.34; p2 weight drifts to 0.32/0.68 | 0.075@12500 → 0.952 |

### `two_point_peaks_0.5-2.0_weights_0.5-0.5` — 1/15 bad
| run | issue | min→final exploitability |
|---|---|---|
| nc=2 seed=2 | p1 weight drifts to 0.58/0.42 (want 0.50/0.50, mild) | -0.175@7500 → 0.452 |

### `two_point_peaks_0.5-2.0_weights_0.7-0.3` — 3/15 bad
| run | issue | min→final exploitability |
|---|---|---|
| nc=2 seed=0 | p1 weight drifts to 0.56/0.44 (want 0.70/0.30) | 0.035@23500 → 0.497 |
| nc=2 seed=2 | both players' weight drifts to 0.82-0.83/0.17-0.18; p2's 2.0-peak component drifts to **4.31** (Pattern B) | 0.096@29500 → 0.517 |
| nc=2 seed=4 | both players' weight drifts to 0.83/0.17; p2's 2.0-peak component drifts to **4.06** | 0.318@4000 → 0.945 |

### Fully converged configs (0 bad runs)
- `two_point_peaks_-1.0-1.0_weights_0.5-0.5` (0/15)
- `two_point_peaks_-1.5-0.0_weights_0.5-0.5` (0/15)

Balanced target weights (0.5/0.5) are far more robust — with no minority component to
starve of gradient signal, there's nothing to drift away from.

## Takeaways
- Use `network.num_components >= 3`, ideally `4`, for reliable peak recovery — the extra
  redundancy prevents the weight-ratio drift that plagues `num_components=2`.
- The instability is a **late-training** phenomenon, not a discovery problem: checkpointing
  and early-stopping near the actual exploitability minimum (often iter 10k-30k, well before
  the scheduled 50k) would salvage the vast majority of these runs.
- Skewed target weights (0.2/0.8, 0.7/0.3) are much more fragile than balanced (0.5/0.5)
  splits, presumably because the minority-weight component gets a weak/noisy training signal
  under PPO.
