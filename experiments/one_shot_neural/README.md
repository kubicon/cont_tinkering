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

Methods in the default grid: `mixture` (this repo's Gaussian-mixture MMD — the method
under test), `mmd_discrete`, `nfsp`, `psro`, `rpn_pathwise`, `sisa`.

`spg` and `jpspg` are still implemented and still runnable, but are **opt-in** — ask for
them by name (`run_all.py --methods spg jpspg`). `rpn_pathwise` carries the randomized
policy network in the grid instead: the same implicit policy `a = f(o, z)` those two train
by a zeroth-order pseudo-gradient, trained by the *exact* pathwise gradient
(`baselines/neural/randomized_policy_pathwise.py`). Read the swap for what it is — a
change of question, not an upgrade. `rpn_pathwise` differentiates the payoff and so
assumes a strictly stronger access model; `spg`/`jpspg` assume only that the payoff can be
*evaluated*. Run them alongside it whenever the cost of that assumption is the thing being
measured, which is what the `access_model` column exists to keep visible.

## The games

Two groups, and the second one is what makes a comparison of *representations* possible.

|game|Nash|what a residual there means|
|-|-|-|
|`two_point`, `multi_point`|finitely many points, known in closed form|the optimizer: a `K`-component mixture can represent this exactly|
|`idealized_decoy_well`|finite support, exactly as large as `K`|the counterexample: enough capacity and MMD still cannot reach it|
|`all_pay_auction`|**uniform on an interval**|the representation: no finite mixture is a Nash here at all|
|`glicksberg_gross`|**density `~1/sqrt(t)`, unbounded at 0**|the representation, at its hardest; value is exactly `4/pi`|
|`circle`|uniform, and any `M > harmonics` evenly spaced atoms|cycling, not capacity — the last iterate rotates while the average converges|

The three continuum-support games carry their analytic equilibrium as
`game.nash_strategy(num_atoms)`, in the same `(support, weights)` form `GridOracle`
consumes, so "how far is this run from the true Nash" is answerable directly and not
only through exploitability. `tests/test_continuum_games.py` pins those equilibria.

One warning that shows up immediately: `all_pay_auction` uses the **hard** win rule by
default (`sharpness: null`), under which the payoff has no useful gradient — the
exact-gradient method (`sisa`) sees only the `-a1 + a2` term and sits at `expl ~ 1`.
That is the game being honest, not a bug; set `game.sharpness` to soften the contest,
at the cost of the analytic equilibrium no longer being exactly uniform.

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

## The pseudo-gradient defaults are this repo's, not the paper's

`run_cell.Settings` matches each baseline's own defaults, with one deliberate exception:
the `spg` / `jpspg` block.

| knob | paper | here | why |
|-|-|-|-|
| `sigma` | 0.1 | **2.0** | the one that matters — see below |
| `pseudo_lr` | 1e-4 | **1e-3** | |
| `noise_dim` | 8 | **16** | latent noise fed to the implicit policy |
| `pseudo_max_grad_norm` | 0 (off) | **1.0** | |
| `utility_samples` | 256 | 256 | unchanged |
| `dynamics` | simultaneous | simultaneous | unchanged |

Under the paper's values both methods drive the `tanh` squash into **saturation** on
`circle` and `all_pay_auction`: the pre-tanh output drifts to +22 (circle) or +56
(all-pay) while the parameter norm barely moves, `tanh` returns `1 - 1e-5` with a
derivative to match, and the policy becomes a point mass insensitive to its own noise
input. At that point the two antithetic evaluations `u(theta + sigma z)` and
`u(theta - sigma z)` return the *identical* number, so

```python
coefficient = (u_plus - u_minus) / (2.0 * sigma)   # exactly 0.0
```

and the run is dead — `grad_norm` reaches literally `0.0`, and no later move by the
opponent can restart it. A saturated squash is an absorbing state for a parameter-space
smoothed gradient, which is the same pathology the `randomized_policy` docstring notes
for a hard clip, arrived at asymptotically instead of immediately. A large `sigma` is
what keeps the estimator alive: a perturbation of that size still moves the action after
the squash has saturated. `experiments/pseudo_gradient_convergence/tune.py` reports
`spread` / `pinned` per run and flags a collapsed one, because its exploitability is a
frozen number rather than a converged one.

These are settings, not code: every one is recorded in each run's `meta.json`, so what a
stored run used is always recoverable from the run itself. `pseudo_gradient_convergence/run.py`
and `ablate_jpspg.py` pin the paper's values explicitly rather than inheriting these, since
measuring the paper's defaults is the whole point of those two.

**`samples` moved too, and it is not method-specific.** It is now 16384 (was 4096) —
the actions drawn from a continuous policy per checkpoint, i.e. what a checkpointed
strategy *is*. The exploitability scored from it inherits that sampling error, and
because the best-response term is a max over a grid of noisy empirical means it is biased
*upward* by roughly `1/sqrt(n)`. Measured on a stored run by resampling: mean expl
+0.0535 at n=256, +0.0432 at n=1024, +0.0415 at n=4096 on `glicksberg_gross`. Raising it
to 16384 shrinks that bias and the seed-to-seed noise, at the cost of a slower `score.py`
pass (the best-response block scales linearly in `n`). It applies to *every* method that
carries a continuous policy, so it changes the metric for all of them equally.

**Runs made before this change used the old values.** `run_all.py` skips a cell that
already has a `meta.json`, so re-running will not overwrite them — but it also means a
tree can end up holding both, indistinguishable in `summary.md`. Either re-run the two
methods with `--overwrite`:

```bash
python experiments/one_shot_neural/run_all.py --methods spg jpspg --overwrite
```

or send the new runs to a fresh `--out` tree. The already-stored `data/one_shot_neural`
spg/jpspg runs are paper-default runs; the collapse described above is what they show.
Those runs stay reproducible: both methods remain registered in `run_cell.py`
(`OPTIONAL_METHODS`) with this block of settings intact, so dropping them from the default
grid changed what runs by default and nothing about what they do when asked for.

`rpn_pathwise` does **not** inherit this block. Its defaults are its own
(`Settings.rpn_*`): OGD at `lr = 1e-2` with `optimism = 1.0` rather than AdaBelief, and a
`batch` that is the only sampling in the update, because the gradient itself is exact and
there is no perturbation batch to average on top of it. `sigma` in particular does not
transfer — that variant squashes with a `sigmoid` rather than a scaled `tanh`, and the
`2.0` tuned for the latter is badly wrong for the former.

`run_all.py` caps each cell's thread pool (`--threads-per-cell`, by default the core count
divided by `--max-parallel`) so that parallel cells do not measure contention with each
other.

## Reporting

Plot exploitability against `payoff_evals` (the primary axis) and against `wall_time` (the
secondary one) from the same `curves.json`. Keep the `access_model` column in any table:
`sisa` uses exact payoff gradients, `rpn_pathwise` differentiates the payoff through the
policy, `spg`/`jpspg` assume only a black-box payoff, and the PPO-based methods sit in
between — putting them on one axis without saying so implies a fairness that is not there.
`rpn_pathwise` beside `spg`/`jpspg` is a *controlled* pair — same policy, same games, only
the gradient differs — so a gap between them prices the black-box assumption and nothing
else; `rpn_pathwise` beside `mixture` is not, and should be read as an upper bound on what
the implicit representation can do rather than as a like-for-like result. Use several seeds and show median with an IQR band; these are
stochastic and single-seed curves do not reproduce.

## Wall-time-matched re-run of SPG / JPSPG (`run_walltime_matched.py`)

Equal payoff evaluations is the right common currency, but it is not the only question a
reader has. The zeroth-order methods are an order of magnitude cheaper *per evaluation*
than everything else — at 2e7 evals on this tree,

| method | mixture | mmd_discrete | psro | nfsp | spg | jpspg |
|-|-|-|-|-|-|-|
| train s (mean over games/seeds) | ~730 | ~430 | ~440 | ~1200 | ~38 | ~50 |

so the eval-matched grid also hands the mixture ~20× the compute, and "SPG loses" invites
the objection that it was never given the same *time*. [`run_walltime_matched.py`](run_walltime_matched.py)
removes that objection: it measures each pseudo-gradient method's throughput
(`payoff_evals / train_seconds`) and the mixture's wall-time from this tree, then picks
the per-`(game, method)` budget that lands the re-run at the same wall-time × `--headroom`
(1.1 by default, so the answer is "more time, still behind" rather than a near-miss).

```bash
# what budgets, and why — runs nothing
python experiments/one_shot_neural/run_walltime_matched.py --dry-run

# run them (~24 cells; matching the reference's --max-parallel keeps wall-times comparable)
python experiments/one_shot_neural/run_walltime_matched.py --max-parallel 4

# score the new tree and plot it against the reference mixture
python experiments/one_shot_neural/run_walltime_matched.py --plot
```

The budget differs per cell, which is what `run_all.py`'s single `--budget` cannot
express — hence a separate script and a **separate tree**, `data/one_shot_neural_walltime`.
`data/one_shot_neural` is only ever read; the calibration it derived is written to
`calibration.json` / `calibration.md` next to the new runs, and `--target-seconds`
overrides it when re-running on a different machine.

`--plot` writes `<game>_walltime_curves.png` with exactly three curves: the two re-runs,
relabelled `spg (wall-matched)` / `jpspg (wall-matched)`, and `mixture` read from the
reference tree's `curves.json`. The relabelling is not cosmetic — the two trees hold
different budgets under the same method name, and the merged plot's *budget* panel is
therefore no longer a like-for-like comparison. Read the wall-time panel; that is the one
this experiment is for. `--include-reference-pseudo` adds the original eval-matched
spg/jpspg curves for contrast, and `--match` targets a method other than `mixture`.

Achieved wall-times are printed against their targets when the runs finish, because
throughput is measured rather than guaranteed — check that column before quoting the
result.
