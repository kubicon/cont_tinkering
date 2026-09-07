# One-shot games: neural methods, compared

Runs the neural methods on the one-shot games and stores everything needed to compare
them afterwards. Three scripts:

| script | what it does |
|-|-|
| `run_cell.py` | one `(game, method, seed)` cell — the unit of work, runnable on its own |
| `run_all.py` | the grid of cells, several at a time, one process each |
| `score.py` | computes exploitability for every checkpoint of every run, afterwards |

```bash
# one method at a time, which is the normal way to use this
python experiments/one_shot_neural/run_all.py --methods psro --seeds 0 1 2 --max-parallel 3

# or the whole grid
python experiments/one_shot_neural/run_all.py --max-parallel 4 --budget 2000000

# then measure everything with one instrument
python experiments/one_shot_neural/score.py --out data/one_shot_neural --grid 801
```

Methods: `mixture` (this repo's Gaussian-mixture MMD — the method under test),
`mmd_discrete`, `nfsp`, `psro`, `spg`, `jpspg`, `sisa`.

## The two decisions that make the comparison mean something

**Budgets are in payoff evaluations, not iterations.** An iteration of PSRO and an
iteration of a Gaussian mixture have nothing in common; a scored action pair does. Each
method converts `--budget` into its own iteration or round count through the cost formula
in `run_cell.plan_units` — including PSRO's empirical payoff matrix, whose every new
population pair costs an outer product of samples and which is easy to forget. Sampling
done *for checkpoints or for the metric* is excluded: instrumentation is not the
algorithm, and counting it would penalize whichever method logs most.

**Nothing is scored during training.** One in-training exploitability call best-responds
over the whole deviation grid against a few thousand sampled actions — on the cheaper
methods that costs more than the training between two checkpoints, and it would land
inside the wall-time being compared. So runs write strategies and costs; `score.py`
measures them all afterwards with one oracle on one grid. Rerun it with a finer `--grid`
and every curve is recomputed without retraining. (`--score` puts in-run scoring back for
interactive use.)

## What a run writes

```
data/one_shot_neural/
  logs/<game>__<method>__seed<k>.log
  curves.json, summary.md                       written by score.py
  <game>/<method>/seed<k>/
    meta.json          plan, settings, timings, access model, git commit
    history.json       per checkpoint: t, wall_time, compile_time, payoff_evals, cheap metrics
    checkpoints/*.npz  StrategyPair — the format every baseline in this repo shares
    params/<name>/     network weights, in training.checkpoint's format
    scores.json        written by score.py: expl per checkpoint, beside the cost columns
```

Because every method checkpoints the same object — both players' support and weights —
the curves are directly comparable, and any later analysis (a different metric, a finer
grid, a support-recovery measure) reads the checkpoints rather than rerunning anything.

## Timing

`wall_time` is **training only**: compilation is measured separately into `compile_time`
(via ahead-of-time `lower().compile()`), and logging, checkpointing and scoring sit
outside the timer. `jax.block_until_ready` is used before stopping the clock — without it
a "fast" method is just one that queued its work and returned.

Two caveats to state whenever wall-time is reported:

* NFSP and PSRO build a fresh best-response trainer per round, so they pay a **new XLA
  compilation each round**, and for them that compile cost is inside `wall_time` (it is
  not separable there the way it is for the single-scan methods). That is a property of
  how those algorithms are structured, but the constant is this repo's.
* Everything runs in float64, because `baselines.common` turns on x64 process-wide for
  the exact solvers. It is the same for every method, so comparisons are fair, but the
  absolute numbers are not what a float32 implementation would give.

`run_all.py` caps each cell's thread pool (`--threads-per-cell`, by default the core count
divided by `--max-parallel`) so that parallel cells do not measure contention with each
other.

## Reporting

Plot exploitability against `payoff_evals` (the primary axis) and against `wall_time` (the
secondary one) from the same `curves.json`. Keep the `access_model` column in any table:
`sisa` uses exact payoff gradients, `spg`/`jpspg` assume only a black-box payoff, and the
PPO-based methods sit in between — putting them on one axis without saying so implies a
fairness that is not there. Use several seeds and show median with an IQR band; these are
stochastic and single-seed curves do not reproduce.
