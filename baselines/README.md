# Representation baselines

`run_idealized.py` fixes the representation (a `K`-component Gaussian mixture) and varies
the update rule. These three fix the game and vary the **representation** of a mixed
strategy. They read only the `game:` section of any config in `configs/`, so the game is
literally the same object the mixture solver runs on, and they all report the same
exploitability, `max_a E_{b~y}[u(a,b)] - min_b E_{a~x}[u(a,b)]`, computed by
`common.GridOracle` — the same quantity as `run_idealized.QuadratureBackend.exploitability`.

| file | representation | update | cost per iteration |
|-|-|-|-|
| `grid_mmd.py` | probability vector over an `N^d` grid | tabular MMD (the mixture solver's categorical step) | one `N^d x N^d` mat-vec |
| `double_oracle.py` | finite support, grown by best responses | exact LP on the restricted game | one global argmax per player + an LP |
| `particle_mean_field.py` | `M` weighted particles | Wasserstein–Fisher–Rao flow (gradient on positions, mirror ascent on weights) | `M^2` payoff evaluations |
| `sisa.py` | fixed-cardinality support (`n` atoms + weights) | SISAMS (Martin & Sandholm): atoms ascend their own deviation utility, weights take a subgradient step on metagame exploitability | `n^2` payoff evaluations |

```bash
python -m baselines.grid_mmd             configs/two_point.yaml --grid 401 --iters 20000 --lr 10 --tau 1.0 --magnet-interval 200
python -m baselines.double_oracle        configs/two_point.yaml --grid 1001 --polish --dedup 1e-3
python -m baselines.particle_mean_field  configs/two_point.yaml --grid 401 --particles 32 --iters 4000 --lr-weight 10 --tau 1.0 --magnet-interval 200
python -m baselines.sisa                 configs/two_point.yaml --grid 401 --atoms 4 --iters 20000 --init spread
```

Each takes `--out history.json` to dump its per-iteration exploitability for plotting
against `idealized_history.json`, and `--checkpoint-dir DIR` to write one
`common.StrategyPair` npz per logged iteration (both players' support and weights, one
format for all three algorithms — reload with `common.load_checkpoints`).

To run the whole grid of games x algorithms and keep the checkpoints, use
[`experiments/one_shot_tabular/run_baselines.py`](../experiments/one_shot_tabular/) rather
than driving the three CLIs by hand.

## Notes that matter when reading their output

* **Atoms are clustered before printing.** A discretized or particle strategy spreads mass
  over adjacent cells, since the payoff gap between neighbours is tiny next to the magnet's
  pull. `common.cluster_atoms` merges anything within 2% of the box width, so a converged
  run prints `[0.700@+1.000 0.300@-1.000]` rather than twenty cells; `support_0/1` in the
  history is that merged count.
* **`grid_mmd`'s metric is exact for the discretized game** (its strategy grid and its
  deviation grid are the same set), so its residual is the discretization error. The other
  two sit off the grid and are measured against it, so their residual includes the oracle's.
* **Double oracle counts oracle calls, not gradient steps.** Each of its rounds is a global
  optimization over the whole action space plus an LP — that is the cost being compared, and
  the reason it stops in a handful of rounds.
* `--tau 0` in `grid_mmd`/`particle_mean_field` removes the magnet and leaves plain mirror
  ascent, whose last iterate cycles while its average converges; `grid_mmd` reports both.

Sanity checks run during implementation, on `configs/two_point.yaml` (peaks ±1, Nash
weights 0.3/0.7, which `MultiPointGame`'s docstring derives in closed form): double oracle
converges in 6 rounds to exploitability `0.00000`, tabular MMD to `0.0101`, and the particle
flow to `0.0036` — all at `[0.700@+1.000 0.300@-1.000]`. `tests/test_baselines.py` pins this.

## Neural baselines

The three solvers above use exact gradients and no network. Their neural counterparts —
MMD on a discretized head, NFSP, and PSRO, all sharing one RL best-response oracle built
on the repo's own `MixturePPOTrainer` — live in [`neural/`](neural/) and report the same
exploitability through the same `GridOracle`, so a sampled curve and an exact one are the
same measurement.

### SISAMS (`sisa.py`)

The closest published relative of the mixture head: a *fixed* number of atoms whose
positions move and whose weights are updated, with neither a best-response oracle nor an
exact metagame solve — double oracle with both expensive parts replaced by incremental
steps. It reports two numbers: `expl` (true, against the deviation grid) and
`metagame_expl` (`Phi`, what the algorithm minimizes, i.e. exploitability *within its own
support*). The gap between them is the diagnostic.

Observed on `configs/two_point.yaml` (peaks ±1, Nash 0.3/0.7): with `--init spread` and 4
atoms, or 8 random atoms, it reaches `expl 0.001` with the atoms exactly on the peaks and
the correct mix. With 4 *random* atoms it can end at `expl 1.21` and `metagame_expl 0.000`
— every atom of one player having drifted into the same basin, after which they share a
gradient and can never separate, so the metagame has no mixture left to find. That is a
real property of a fixed-cardinality support with no specialization mechanism (the paper's
rank-based mixing operator targets it; see the fidelity note in the module docstring), and
the run prints a warning when it happens rather than reporting a converged-looking
`metagame_expl`.
