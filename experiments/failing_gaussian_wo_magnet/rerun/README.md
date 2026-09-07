# The magnet grid: 3 domains x {magnet, no magnet} x 3 losses

A rerun of `experiments/failing_gaussian_wo_magnet` as a full factorial, with every
iterate kept on disk so the statistics can be computed afterwards rather than being
baked into the run.

```
bash experiments/failing_gaussian_wo_magnet/rerun/run_all.sh [max_parallel]   # default 4
```

18 cells, `<domain>_<magnet|nomagnet>_<engine>.yaml`. All of them run
`train.steps: 200` x `train.epochs: 500` = **100 000 iterations**, logging and
checkpointing once per step (201 records per cell, `t = 0, 500, ..., 100000`).

## The three factors

**Domain.**

| prefix | game | dim | Nash | notes |
|-|-|-|-|-|
| `mp` | `matching_pennies` (`ContinuousMatchingPennies`) | 1 | any point mass at the box midpoint | bilinear, so only the mean matters; `init_means: [0.2]` starts off it |
| `rot2` | `coupled_rotation` (`CoupledRotationGame`) | 2 | the origin, uniquely | the documented reference case (`CoupledRotation.md`) |
| `rot3` | `coupled_rotation` | 3 | the origin, uniquely | same game, one dimension up |

`CoupledRotationGame` has no `dim = 1`: its coupling is `g(a1)^T A g(a2)` with
`A = S - S^T` the skew tridiagonal shift, which is the zero matrix at `dim = 1`.
`games/examples.py` rejects it rather than silently running a game with no coupling.

**Magnet.** `ppo.magnet_gaussian_kl_coef`, and nothing else -- a `_magnet` file and its
`_nomagnet` sibling differ in that one line plus the `checkpoint_dir` they write to.
`tau = 0.2` for `mp`, `0.5` for `rot2`/`rot3`; `0.0` for all six no-magnet cells.

**Loss.**

| suffix | entry point | what it is |
|-|-|-|
| `idealized` | `run_idealized.py` | the exact tabular mirror step, payoffs by quadrature |
| `sampled` | `run_idealized.py` | the same exact step, payoffs by Monte Carlo -- the one PPO approximation, in isolation |
| `ppo` | `train.py` | the neural network: `MixtureActorCritic` + PPO |

## Checkpoints

Everything lands under `data/failing_gaussian_rerun/<cell>/`, with per-cell stdout in
`data/failing_gaussian_rerun/logs/<cell>.log` and a completion ledger in
`logs/_done.txt`.

**`idealized` and `sampled` cells** write `idealized_history.json`: a list of 201
records, one per logged step, each holding the complete solver state --

```
{"t": 500, "expl": ..., "target_expl": ...,
 "w0": [...], "means0": [[...]], "std0": [[...]], "corr0": [...],
 "w1": ..., "means1": ..., "std1": ..., "corr1": ...}
```

`expl` is exploitability of the live iterate, `target_expl` of the Polyak average.
`std` is the per-axis marginal standard deviation and `corr` the largest absolute
off-diagonal correlation per component (identically zero in `mp`, where `d = 1`).
There is nothing else to checkpoint -- for the exact solvers the mixture *is* the state.

**`ppo` cells** write `0.pkl .. 200.pkl` via `MixtureSelfPlayPPOTrainer.save`, each
holding the hyperparameters, the live network parameters and the target parameters.
Reload one with `MixtureSelfPlayPPOTrainer.load(checkpoint_dir, step, game)`.

## Two caveats worth carrying into the analysis

**3-D quadrature is resolution-limited.** The quadrature backend materializes a
`grid_points^(2d)` payoff matrix, so `rot3_*_idealized` runs at 21 points per axis
(a 0.64 GiB matrix); 41 points per axis would need 38 GiB. `CoupledRotation.md`
records that a 3-D quadrature run at this resolution can be uninformative in absolute
terms. Read `rot3_*_sampled` -- 4096 Monte-Carlo draws per iteration, with
exploitability still scored on the grid rather than on the noisy batch -- as the
trustworthy 3-D cell, and `rot3_*_idealized` for direction only.

**The learning rates are not comparable across losses.** `optimizer.learning_rate` is
the exact solver's mirror step `eta` in the `idealized`/`sampled` cells (0.05 for `mp`,
0.02 for the rotation games) and the network's Adam step in the `ppo` cells (0.001).
The magnet comparison is within a cell pair; the loss comparison is qualitative.
