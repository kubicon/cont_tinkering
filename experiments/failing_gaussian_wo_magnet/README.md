# Failing Gaussian without the magnet

This experiment shows that even in the purely Gaussian setting, where the categorical head is trivial (`network.num_components: 1`), the magnet term is what makes MMD converge.

The game is `matching_pennies` = `ContinuousMatchingPennies`, `payoff(a0, a1) = a0 * a1`
on `[-1, 1]`. It is bilinear, so only the *mean* of each policy matters, every best response sits at a box corner, and the unique Nash has both players' mean at the box
midpoint `0`.

* `wo_magnet.yaml` -- `magnet_gaussian_kl_coef: 0.0`. Fails: the pair of means spirals
  *away* from the Nash (the classic bilinear rotation, which gradient ascent-descent
  turns into an outward spiral) until both means clip at the box boundary, then orbits
  there forever. Exploitability flat at ~1.0-1.2.
* `with_magnet.yaml` -- identical except `magnet_gaussian_kl_coef: 0.2`. The orbit
  becomes an inward spiral; exploitability decays geometrically to ~1e-8.

In *average* strategies (the `target_tau` EMA, dashed in the plot) the no-magnet run
improves and then plateaus around 1e-1. That plateau is EMA ripple, not a second
failure: the plain time-average of the last 5000 iterations is `(+0.0005, -0.0059)`,
i.e. the orbit's center *is* the Nash, but `target_tau: 0.001` gives a ~1000-iteration
window against a ~500-iteration orbit, so the EMA still swings +-0.08 around it. Only
the current strategies fail to converge.

Run:

```
python run_idealized.py experiments/failing_gaussian_wo_magnet/wo_magnet.yaml
python train.py         experiments/failing_gaussian_wo_magnet/wo_magnet.yaml
```

## Plot

`plot_dynamics.py` runs both configs and draws the dynamics:

```
# both configs as written -> dynamics.png (~3 min, exact-gradient runs)
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py

# matched inits: the clean magnet ablation
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --init 0.2 \
    --out experiments/failing_gaussian_wo_magnet/dynamics_matched_init.png \
    --cache experiments/failing_gaussian_wo_magnet/dynamics_history_matched_init.json.gz

# replot from the cached traces, no re-run
python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --reuse
```

It needs `matplotlib` (added to `requirements.txt`).

Panels: the `(E[a_0], E[a_1])` phase plane (color darkens with iteration, circle =
start, star = end), NashConv for current and average strategies, and the distance to
the Nash together with the policy std.

Two notes on the method, both of which cost a first attempt at this figure:

* The trajectory is traced **every iteration**, not once per outer step. One orbit
  takes ~500 iterations here (`2*pi / (lr * std^2)`, since the Fisher metric scales the
  mean step by `std^2` -- `lr: 0.05` with std pinned at the `0.5` ceiling), so logging
  once per outer step (200 iterations) aliases the circle into a polygon that reads as
  a slow drift.
* Exploitability is evaluated eagerly on a best-response grid, so it is subsampled
  (`--expl-points`) rather than computed at every traced iteration.

### Files

| file | what |
|-|-|
| `wo_magnet.yaml` | no magnet on the Gaussian head; starts at mean +1.0 |
| `with_magnet.yaml` | same, `magnet_gaussian_kl_coef: 0.2`; starts at mean +0.2 |
| `plot_dynamics.py` | runs both, writes the figure and a history cache |
| `dynamics.png` | the two configs exactly as written (inits differ, see below) |
| `dynamics_matched_init.png` | `--init 0.2`: the clean ablation, magnet is the only difference |
| `dynamics_history*.json.gz` | cached per-iteration traces, for `--reuse` |

The two YAMLs do **not** share `init_means` (+1.0 vs +0.2), so `dynamics.png` is not a
pure ablation; `dynamics_matched_init.png` is. The conclusion is the same either way. The
matched-init figure is the more informative one: from the *same* start, the no-magnet
run spirals outward to the boundary while the magnet run spirals inward to the Nash --
the magnet is what turns the rotation's outward drift into a contraction.

