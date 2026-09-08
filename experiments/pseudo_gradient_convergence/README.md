# Pseudo-gradient convergence (SPG / JPSPG)

SPG and JPSPG did not converge in [`one_shot_neural`](../one_shot_neural/): on every
game the final exploitability stayed near the start of the run (see
`data/one_shot_neural/summary.md`). A first follow-up,
[`ablate_jpspg.py`](../one_shot_neural/ablate_jpspg.py), swept **dynamics / lr /
optimizer** on `two_point` and still finished around expl ≈ 2.

This experiment asks a narrower question: *are the paper defaults for the
zeroth-order estimator itself the problem?* It sweeps the knobs that control
gradient quality — **σ** (smoothing radius) and **utility_samples** — for both
`spg` and `jpspg`, then re-checks dynamics at the best setting.

| stage | what | why |
|-|-|-|
| 1 baseline | paper defaults × both methods | reproduce the failure on a clean game |
| 2 sigma | `{0.01, 0.05, 0.1, 0.3, 1.0}` | ZO bias–variance; paper uses `0.1` |
| 3 samples | `{64, 256, 1024, 4096}` | noise in each utility evaluation |
| 4 dynamics | simultaneous / optimistic / extragradient | only after σ and samples look sane |

Default interactive game is **`matching_pennies`**: bilinear, unique Nash at the
origin. If last-iterate exploitability does not fall here, the estimator or
ascent is broken for the simplest continuous game we have — fix that before
returning to `two_point` / continuum games. The cluster generator also runs
`two_point`, `circle`, `glicksberg_gross`, and `all_pay_auction` (the same set
that failed in `one_shot_neural`). Scoring is last-iterate only (these methods
do not checkpoint a Polyak average); if curves oscillate around a low mean, that
is a separate code change to try next.

## Run

```bash
# smoke (stage 1, cheap)
python experiments/pseudo_gradient_convergence/run.py \
    --stages 1 --budget 500000 --seed 0

# full staged search; score between stages and carry the best σ / samples forward
python experiments/pseudo_gradient_convergence/run.py \
    --budget 20000000 --seed 0 --score-between

# pin knobs and only sweep dynamics
python experiments/pseudo_gradient_convergence/run.py \
    --stages 4 --sigma 0.05 --utility-samples 1024 --budget 20000000

# same failure mode as one_shot_neural
python experiments/pseudo_gradient_convergence/run.py \
    --game configs/two_point.yaml --budget 20000000 --seed 0 --score-between

# score + rank + plot
python experiments/pseudo_gradient_convergence/run.py --report \
    --out data/pseudo_gradient_convergence
```

Cluster: generate one SLURM job per `(game, stage)` with

```bash
python rci_scripts/generate_pseudo_gradient_convergence.py
# games: matching_pennies, two_point, circle, glicksberg_gross, all_pay_auction
bash scripts/pseudo_gradient_convergence/run_all.sh
```

Or only the continuum games that failed in `one_shot_neural`:

```bash
python rci_scripts/generate_pseudo_gradient_convergence.py \
    --games configs/circle.yaml configs/glicksberg_gross.yaml \
           configs/all_pay_auction.yaml
```

## Output

```
data/pseudo_gradient_convergence/
  summary.json
  <game>_ranking.md          # from --report
  <game>_curves.png
  <game>/<variant_tag>/seed<k>/   # same layout as one_shot_neural
```

`--report` ranks by **final** exploitability. Treat a run as "converged" if that
lands below ~0.1 on matching pennies (or ~0.2 on two_point). Anything still near
the random baseline means these knobs are not enough — next suspects are
average-iterate logging, `noise_dim` / network capacity, or a bug in the
estimator path.

## Tuning knobs by hand (`tune.py`)

`run.py` runs the fixed staged sweep. [`tune.py`](tune.py) is the interactive
counterpart: every hyperparameter is a flag taking **one or more** values, the
cartesian product is run, and each run is scored live.

```bash
# one run, paper defaults, ~4 s at the default 2e7 budget
python experiments/pseudo_gradient_convergence/tune.py

# the recommended setting, on the game that never converged
python experiments/pseudo_gradient_convergence/tune.py \
    --game configs/two_point.yaml --sigma 1.0 --utility-samples 1024

# sweep two knobs (4 runs) and rank them
python experiments/pseudo_gradient_convergence/tune.py --sigma 0.1 1.0 --utility-samples 256 1024

# how many iterations does a budget buy? (runs nothing)
python experiments/pseudo_gradient_convergence/tune.py --sigma 0.01 1.0 --dry-run
```

Sweepable: `--method --sigma --utility-samples --dynamics --lr --optimizer
--noise-dim --max-grad-norm --seed`. Single-valued: `--hidden-dims --activation
--no-antithetic`. The run tag carries the method plus whatever was swept or set
away from its default, so directories never collide; finished runs are skipped
unless `--overwrite`.

**Why it reports more than exploitability.** The dominant failure of this
estimator does not look like a bad number, it looks like a *frozen* one. Small
`sigma` drives the policy into tanh saturation: every sampled action pins to the
box edge, the policy becomes a point mass, and exploitability sits at a constant
for the rest of the run. On `two_point` that collapse *out-ranks* every healthy
run, so `run.py`'s table recommends it. Each run therefore also reports

| column | meaning | collapsed run |
|-|-|-|
| `spread` | std of sampled actions / box width | `< 0.01` |
| `pinned` | fraction of coordinates within 0.5% of a box edge | `> 0.9` |
| `tail range` | expl movement over the last quarter of the run | `~0` |

and is flagged `COLLAPSED` whatever its exploitability says. The live
`support_0` / `support_1` columns show the same thing while the run is going:
they fall from ~28 effective atoms to 1–2.

Output goes to `data/pseudo_gradient_tuning/<game>/<tag>/seed<k>/` in the same
layout as `run.py`, so `score.py` and `run.py --report` work on this tree too.
