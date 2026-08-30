# Failing Gaussian without the magnet

This experiment shows that even in the purely Gaussian setting, where the categorical head is trivial (`network.num_components: 1`), the magnet term is what makes MMD converge.

The game is `matching_pennies` = `ContinuousMatchingPennies`, `payoff(a0, a1) = a0 * a1`
on `[-1, 1]`. It is bilinear, so only the *mean* of each policy matters, every best response sits at a box corner, and the unique Nash has both players' mean at the box
midpoint `0`.

* `wo_magnet_{idealized,ppo}.yaml` -- `magnet_gaussian_kl_coef: 0.0`. Fails: the pair of
  means spirals *away* from the Nash (the classic bilinear rotation, which gradient
  ascent-descent turns into an outward spiral) until both means clip at the box
  boundary, then orbits there forever. Exploitability flat at ~1.0-1.2.
* `with_magnet_{idealized,ppo}.yaml` -- identical except `magnet_gaussian_kl_coef: 0.2`.
  The orbit becomes an inward spiral; exploitability decays geometrically to ~1e-8.

Each setting has an `_idealized.yaml`, a `_sampled.yaml` and a `_ppo.yaml` file. The
idealized/sampled pair differ only in `idealized.backend` (`quadrature` vs `sampled`
plus its sample knobs); the `_ppo.yaml` differs from those in `optimizer.learning_rate`
(0.001 vs 0.05) -- that one field feeds both the idealized/sampled solver's step size
and the PPO network's Adam learning rate, and the engines need different values (see
`plot_dynamics.py`'s docstring).

In *average* strategies (the `target_tau` EMA, dashed in the plot) the no-magnet run
improves and then plateaus around 1e-1. That plateau is EMA ripple, not a second
failure: the plain time-average of the last 5000 iterations is `(+0.0005, -0.0059)`,
i.e. the orbit's center *is* the Nash, but `target_tau: 0.001` gives a ~1000-iteration
window against a ~500-iteration orbit, so the EMA still swings +-0.08 around it. Only
the current strategies fail to converge.

Run:

```
python run_idealized.py experiments/failing_gaussian_wo_magnet/wo_magnet_idealized.yaml
python run_idealized.py experiments/failing_gaussian_wo_magnet/wo_magnet_sampled.yaml
python train.py         experiments/failing_gaussian_wo_magnet/wo_magnet_ppo.yaml
```

## Plot

`plot_dynamics.py` runs both configs and draws the dynamics. `--engine` picks who does
the running:

| engine | driver | policy |
|-|-|-|
| `idealized` | `run_idealized.py`'s solver, exact payoff integral | the mixture parameters themselves, exact gradients |
| `sampled` | `run_idealized.py` with `idealized.backend: sampled` | same tabular mirror step, payoff from `ppo.batch_size` draws/iteration (the one PPO approximation, isolated); reads `*_sampled.yaml` |
| `ppo` | `train.py`'s `MixtureSelfPlayPPOTrainer` | the network's policy head, sampled rollouts + critic + clipped ratios |
| `both` (default) | all three, overlaid | PPO dashed, sampled dash-dotted |

```
# exact solver, both configs as written -> dynamics.png (~3 min)
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py

# the same configs through train.py's PPO -> dynamics_ppo.png (~4 min)
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --engine ppo

# the idealized solver with a Monte-Carlo payoff -> dynamics_sampled.png
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --engine sampled

# overlay all three -> dynamics_both.png
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --engine both

# matched inits: the clean magnet ablation
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --init 0.2 \
    --out experiments/failing_gaussian_wo_magnet/dynamics_matched_init.png \
    --cache experiments/failing_gaussian_wo_magnet/dynamics_runs_matched_init.pkl

# replot from the cached traces, no re-run
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --engine both --reuse
```

It needs `matplotlib` (added to `requirements.txt`).

Panels: the `(E[a_0], E[a_1])` phase plane (color darkens with iteration, circle =
start, star = end), NashConv for current and average strategies, and the distance to
the Nash together with the policy std.

### What the PPO engine does

Same networks, hyperparameters and self-play train step as `train.py` -- it builds a
`MixtureSelfPlayPPOTrainer` from the config's `_ppo.yaml` -- but instead of `trainer.train()`'s
chunked loop it scans the train step itself, reading the policy head out at the game's
(constant) observation every iteration. Two deliberate differences from a bare
`python train.py`:

* The run starts from `idealized.init_means` / `init_log_std`, written into the
  `means_head` / `log_std_head` biases (both heads have a zero kernel, so at init their
  output is exactly the bias). Left to its own `_spread_bias_init`, a one-component
  network starts at the box midpoint -- which on this game *is* the Nash, so the run
  would sit at a fixed point from step 0 and "converge" without taking a real step.
  `--network-init` keeps the trainer's own init if you want to see that.
* Importing `run_idealized` enables JAX x64 process-wide, so the numbers are not
  bit-identical to a bare `python train.py`.

### Results

| run | engine | final NashConv (current) | final NashConv (average) |
|-|-|-|-|
| no magnet | idealized | 1.24 | 0.094 |
| magnet | idealized | ~0 (2e-8) | ~0 |
| no magnet | ppo | 2.00 | 2.00 |
| magnet | ppo | 0.46 | 0.32 |

The two PPO rows, and `dynamics_ppo.png` / `dynamics_both.png`, were measured **before**
`network.clip_means` existed, i.e. with the mean head unconstrained; both configs now set
`clip_means: true`, so a fresh `--engine ppo` run will not reproduce them. Rerun to
refresh.

The PPO picture is the idealized one plus a noise floor, and it is harsher. Without the
magnet (and without the clip) the means do not merely orbit: Adam plus the sampled
advantage drives them to `|E[a]| ~ 1e3`, far outside the box, where both policies are
point masses on opposite corners and NashConv saturates at its maximum, 2.0. With the
magnet the pair stays in a cloud around the Nash, NashConv fluctuating in `[1e-2, 5e-1]`
(average strategies lower) instead of decaying geometrically as in the exact solver --
that floor is sampling noise (`ppo.batch_size: 256`), not a failure to converge.

Notes on the method, several of which cost a first attempt at the figure:

* Both engines are traced **every iteration**, not once per outer step. One orbit takes
  ~500 iterations here (`2*pi / (lr * std^2)`, since the Fisher metric scales the mean
  step by `std^2` -- `lr: 0.05` with std pinned at the `0.5` ceiling), so logging once
  per outer step (200 iterations) aliases the circle into a polygon that reads as a
  slow drift.
* Both engines are scored with the **same** quadrature NashConv from
  `run_idealized.py`, so the curves are comparable. Exploitability needs an eager
  best-response sweep, so it is subsampled (`--expl-points`) rather than computed at
  every traced iteration.
* The PPO policy is scored on the law of the **clipped** action (each component's
  out-of-box tail mass put back as a spike on the boundary grid point), because
  `train.py` clips every sampled action into the box. Without that, a policy whose mean
  has run to `-50` would have no mass on the padded grid at all and would score as
  *unexploitable*.
* The phase plane is fixed to the action box plus a 5% margin (`[-1.1, 1.1]` here). With
  `network.clip_means` on that is where the strategies live; a run whose means do leave
  the box runs out of frame, and the distance-to-Nash panel is what shows how far it
  went.

### Why the PPO means reach 1e3

Nothing in the network path constrains them. `train.py` clips the sampled *action*
(`space.clip(raw_action)`, `training/mixture.py`) so the payoff stays well-defined, but
both PPO ratios are taken on `raw_action`, and `means_head` is unbounded -- only
`log_std` is clipped. So once a mean is outside the box, every sample clips to the same
corner, the payoff stops depending on this player's action, and the only thing left in
the raw advantage is opponent-sampling noise and critic error -- which
`training/ppo.py`'s advantage normalization rescales straight back to unit variance.
Adam then turns that into a step of about `lr` per iteration regardless of gradient
magnitude: `20000 * 0.05 = 1000`, which is the observed scale. The idealized solver has
no such problem because it projects after every step (`jnp.clip(means, lo, hi)` in
`run_idealized.player_step`).

`network.clip_means: true` applies the same projection to the network's mean head, which
discards any gradient pushing a mean out of the box (`clip` has zero derivative there).
Both configs here set it; it is off by default repo-wide. The one thing to watch: the
derivative is zero on *both* sides once out, so a mean that overshoots the bound by one
step is pinned there and can only be moved by whatever also moves the torso.

### The run cache

Every run is pickled to `--cache` (default `dynamics_runs[_<engine>].pkl`), keyed
`"<engine>:<config>.yaml"`. Each record is three plain dicts of numpy arrays, so a
finished run can be re-analyzed without re-running it:

| key | contents |
|-|-|
| `hyperparameters` | engine, game, action bounds, Nash point, iteration count, `lr`, the magnet coefficient, the realized init means, plus the whole resolved config: `solver` (`SolverConfig`, `idealized:` section included) and `run_config` (the `train.py` schema) |
| `stats` | `t`, `x`/`y` (each player's `E[a]`), `x_avg`/`y_avg` (average strategies), `std`, `dist_nash`, `dist_nash_avg`, and `expl`/`expl_avg` on their `expl_t` subsample |
| `params` | the raw traced mixture per player (`player0`, `player1`, `player0_avg`, `player1_avg`), each `weights` / `means` / `std` / `logits` of shape `(iterations, num_components)` |

Everything in `stats` is derived from `params`, so questions the plot does not answer
(component weights, per-component spread) can be answered from the cache alone.

```python
import pickle
runs = pickle.load(open("experiments/failing_gaussian_wo_magnet/dynamics_runs_both.pkl", "rb"))
r = runs["ppo:wo_magnet_ppo.yaml"]
r["hyperparameters"]["run_config"]["ppo"]["magnet_gaussian_kl_coef"]  # 0.0
r["stats"]["x"][-1]                                                   # final E[a] of player 0
```

### Files

| file | what |
|-|-|
| `wo_magnet_{idealized,sampled,ppo}.yaml` | no magnet on the Gaussian head; starts at mean +1.0; `lr` 0.05 (idealized/sampled) / 0.001 (ppo) |
| `with_magnet_{idealized,sampled,ppo}.yaml` | same, `magnet_gaussian_kl_coef: 0.2`; starts at mean +0.2 |
| `*_sampled.yaml` | `*_idealized.yaml` + `idealized.backend: sampled` (payoff from 256 draws/iteration) |
| `plot_dynamics.py` | runs the matching engine's pair of configs, writes the figure and a trace cache |
| `dynamics.png` | idealized engine, configs as written (inits differ, see below) |
| `dynamics_matched_init.png` | idealized, `--init 0.2`: the clean ablation, magnet is the only difference |
| `dynamics_ppo.png` | `train.py`'s PPO on the same two configs |
| `dynamics_sampled.png` | the idealized solver with the payoff estimated from `ppo.batch_size` draws/iteration |
| `dynamics_both.png` | all three engines overlaid |
| `dynamics_runs*.pkl` | pickled run records: hyperparameters + per-iteration numpy arrays |

The two YAMLs do **not** share `init_means` (+1.0 vs +0.2), so `dynamics.png` is not a
pure ablation; `dynamics_matched_init.png` is. The conclusion is the same either way. The
matched-init figure is the more informative one: from the *same* start, the no-magnet
run spirals outward to the boundary while the magnet run spirals inward to the Nash --
the magnet is what turns the rotation's outward drift into a contraction.

