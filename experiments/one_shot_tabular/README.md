# One-shot games: tabular / non-parametric baselines

Runs the three representation baselines in [`baselines/`](../../baselines) over the
one-shot games with mixed equilibria, and keeps every iterate on disk.

```bash
python experiments/one_shot_tabular/run_baselines.py                 # the default grid
python experiments/one_shot_tabular/run_baselines.py --dry-run       # what it would run
python experiments/one_shot_tabular/run_baselines.py \
    --configs configs/two_point_peaks_*.yaml --seeds 0 1 2 --out data/one_shot_tabular
```

Each cell is `(game x algorithm x seed)`; a cell whose `meta.json` exists is skipped, so
the script is restartable and adding a game later costs only that game (`--overwrite`
re-runs). A cell that fails is recorded and the sweep continues — its traceback lands in
`error.txt` next to the run.

## What is written

```
data/one_shot_tabular/
  summary.json, summary.md              every cell's final numbers, one row each
  <game>/<algorithm>/seed<k>/
    meta.json                           settings, game config, git commit, timings, final metrics
    history.json                        per-logged-iteration exploitability (and value, supports)
    checkpoints/index.json              what was written, under which settings
    checkpoints/step_*.npz              a StrategyPair per logged iteration
```

A checkpoint is a `baselines.common.StrategyPair`: **both players' support and weights**,
in the same format for all three algorithms even though one carries a grid, one a growing
finite support and one a particle cloud. That is the point of the format — a later
analysis re-scores all of them with one measure:

```python
from baselines.common import GridOracle, load_checkpoints, latest_checkpoint
from baselines.common import load_game

game, _ = load_game("configs/two_point.yaml")
oracle = GridOracle(game, points=4001)          # a finer grid than the run used
for cp in load_checkpoints("data/one_shot_tabular/two_point/grid_mmd/seed0/checkpoints"):
    expl = oracle.exploitability(cp.support_0, cp.weights_0, cp.support_1, cp.weights_1)
    print(cp.t, expl)
```

`grid_mmd` also stores its averaged iterate under `cp.extra["avg_weights_0"/"_1"]`, so the
last-iterate and average-iterate curves can both be recovered from the same files.

## Reading the numbers

* **Iteration counts are not comparable across algorithms.** A `double_oracle` "iteration"
  is a global best response per player plus an LP; a `grid_mmd` or `particle_mean_field`
  iteration is one mat-vec / one gradient. Compare final exploitability and wall time, and
  compare iteration counts only within an algorithm.
* **Seeds.** Only `particle_mean_field` uses one (random initial particle positions).
  `grid_mmd` and `double_oracle` are deterministic as configured, so they run once however
  many seeds are given; `--seed-all` overrides that (useful with `--do-init random`).
* **Support sizes** in `summary.md` are counted after merging atoms within 2% of the box
  width (`baselines.common.cluster_atoms`) — otherwise a converged grid run reads as
  "support 40" when it is two peaks smeared over neighbouring cells.
* The defaults (`--iters 20000 --lr-weight 10 --tau 1.0 --magnet-interval 200 --grid 401`)
  are the settings the two-peak game was verified with; they are not tuned per game, and a
  game where a baseline stalls should be checked against a larger `--iters` before it is
  reported as a failure of that baseline.
