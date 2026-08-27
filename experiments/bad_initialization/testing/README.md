# Why the PPO backend fails on `bad_initialization`, and what fixes it

Every YAML here is a variant of the committed `../bad_init_ppo.yaml` /
`../good_init_ppo.yaml` pair, one hypothesis per prefix, `_bad`/`_good` per
initialization. Each file's header states the hypothesis it tests. Nothing under
`training/`, `games/` or `run_idealized.py` was modified to produce these numbers.

Run one with the same engine `plot_dynamics.py` uses:

    python experiments/bad_initialization/plot_dynamics.py --engine ppo   # committed pair
    # or, for a single file here, call plot_dynamics.simulate(path, "ppo", 40)

Metric is `W1(policy, Nash)` from `plot_dynamics._w1_series` -- 0 only when the
component means sit on the peaks, the categorical weights match `game.weights`,
and the stds have collapsed. `avg` is the target-EMA (average) strategy, which is
the one the algorithm actually claims to converge; the current iterate jitters
around the Nash even in the runs that succeed.

## Results (45k iterations, seed 0)

| config | change vs committed | good: W1 end (min) / avg | bad: W1 end (min) / avg | verdict |
|-|-|-|-|-|
| *(committed baseline)* | -- | 0.705 (0.703) / 0.706 | 1.012 (0.992) / 1.005 | fails both |
| `t1_wide_init_std` | `init_log_std: 0.1 -> 0.0` | 0.088 (0.009) / **0.013** | 0.208 (0.023) / **0.037** | **converges both** |
| `t2_mid_init_std` | `init_log_std: 0.1 -> -1.0` | 0.193 (0.049) | 0.596 (0.411) | partial |
| `t3_big_batch` | `batch_size: 256 -> 2048` | 0.891 (0.874) | 1.493 (1.485) | fails both |
| `t4_higher_lr` | `learning_rate: 0.001 -> 0.01` | 0.728 (0.208) | 1.012 (0.972) | fails both |
| `t5_free_std` | `gaussian_entropy_coef: 0.5`, both Gaussian KLs off | 1.932 (0.557) | 1.493 (0.303) | **diverges** |
| `t6_wide_peaks` | `game.width: 0.1 -> 0.4` | 0.180 (0.027) | 1.072 (1.042) | good only |
| `t7_combo` | t1 + `batch_size: 1024` + `learning_rate: 0.003` | 0.099 (0.007) | 0.537 (0.014) | converges both |
| `t8_combo_wide_peaks` | t7 + `game.width: 0.4` | 0.103 (0.017) | 0.160 (0.027) | converges both |
| `t9_no_regularizers` | all entropy + KL coefficients 0 | 1.500 (0.514) | 1.988 (0.989) | fails both |
| `t10_strong_magnet` | t7 + magnet coef 1.0, interval 5000 | 0.519 (0.517) | 0.732 (0.671) | fails both |

## What was applied upstream

`../good_init_ppo.yaml` and `../bad_init_ppo.yaml` now carry the `t1` fix
(`init_log_std: 0.1 -> 0.0`), plus `ppo_epochs: 2` on both (the committed pair had
1 for bad and 2 for good -- an unintended asymmetry). Verified end state:

| config | W1 end (min) | W1 avg (min) | expl | expl avg |
|-|-|-|-|-|
| `good_init_ppo.yaml` | 0.088 (0.009) | **0.013** (0.009) | +0.164 | **+0.017** |
| `bad_init_ppo.yaml` | 0.249 (0.006) | **0.013** (0.005) | +0.521 | **+0.035** |

Both players end at ~0.5 x -0.99 / 0.5 x +1.00 with std ~0.006. With
`ppo_epochs: 2` the bad init does better than the `t1` row below, which kept the
committed `ppo_epochs: 1`.

Note that `idealized.init_log_std` is only read on the `plot_dynamics.py --engine ppo`
path, which seeds the network heads from it; `train.py` ignores the whole
`idealized:` block, and its own `_spread_bias_init`/`_std_bias_init` start this
game at means -1.0/+1.0 with std 1.0 -- already on the Nash support, so under
`train.py` neither initialization is being tested. The configs' headers say so.

## What is actually wrong

**1. The reward is unreachable by sampling.** The peaks are Gaussian bumps of
`width: 0.1` at +-1, and both configs start at `init_log_std: 0.1` (std = 0.1).
Over 20k samples drawn from the *good* init (mu = +-0.1, std = 0.1):
`E[D] = 1.6e-10`, and not one sample has `D > 0.01`. The peak term is ~9 std away.
All PPO can see is the bilinear coupling `a1*a2`, whose equilibrium is `E[a] = 0`
-- exactly where the good-init run sits for all 45k iterations.

`t3` is the control: 8x the batch does not help, because 8x a signal that is
identically zero is still zero. This is not a variance problem.

**2. The Gaussian entropy bonus cannot inflate the std -- it has zero expected
gradient.** The idealized solver escapes (1) on its own: its exact
`gaussian_entropy: marginal` gradient inflates std 0.1 -> 0.33 *before* the means
move, which is enough overlap for the peak gradient to bite (visible in
`../dynamics_runs.pkl`). The PPO loss cannot do this.
`training/mixture.py:mixture_ppo_loss` estimates the Gaussian entropy as
`-log p(raw_action)` at the *sampled, detached* action. Differentiating that
w.r.t. `log_std` gives an estimator whose expectation is
`-E_{a~p}[grad log p(a)] = 0` for any parameters. Measured for the good init's
mixture: `0.0015 +- 0.0017`, against a true `dH/dlog_std` of `0.2755`. The term
contributes noise and no expected pressure on the std at all.

`t5` confirms it: raising the coefficient to 0.5 and removing the two Gaussian KL
terms that would otherwise oppose it does not widen the policy -- the std pins at
its floor and the *means* random-walk to -1120 and -20672. That also explains the
module comment in `mixture.py` reporting that this coefficient NaNs above ~0.5.

**3. Once (1) and (2) hold, the trust region freezes the run in place.** With no
signal, every gradient is noise, and the magnet/TRPO KL pins that noise-driven
walk at the initialization -- which is why the good-init run's means never leave
+-0.1. `t9` (all regularizers off) shows the other side: the mean head then
random-walks freely and *does* stumble onto both peaks around 20% in
(-1.00/+0.97, W1 0.514), but has nothing to hold it there and ends worse than
baseline. `t10` shows the opposite extreme -- a magnet strong and slow enough to
prevent collapse also prevents convergence, freezing std at 1.0 forever.

**4. The bad init has a second, terminal failure.** By ~11k iterations the std
hits the hard floor (`training/mixture.py:LOG_STD_MIN` = log 1e-3) and the
categorical head collapses to a single component (`w = [0, 1]`). After that the
policy is a pair of point masses and the two-point Nash is unrepresentable no
matter how long it trains -- W1 pins at 1.005. Note that `idealized.std_min` is
**not** read by the PPO path; that floor is hard-coded, so there is no config-level
brake on this.

## Recommendation

`init_log_std: 0.0` on both configs (`t1`) -- one line, no code change. std = 1.0
spreads the components over the whole `[-2, 2]` action box, ~15% of samples land
on a peak, and both initializations then reach the two-point Nash: average-strategy
W1 0.013 (good) and 0.037 (bad), exploitability +0.017 and +0.058, down from
+1.88 and +2.01. The magnet must stay at its committed strength for this to hold
(see `t9`/`t10`).

`t7` adds `batch_size: 1024` and `learning_rate: 0.003`; it reaches a slightly
better minimum (W1 0.007) but is not more stable at the end and costs ~4x the
compute, so `t1` is the better default. `t8` is the most robust of all but only
because it widens the payoff peaks, i.e. it changes the game.

Structural fix, if the entropy bonus is meant to work as advertised: replace the
single-sample `-log p(a)` estimate with a closed-form entropy for the Gaussian
components (exact per component, or the same grid integral the idealized solver
uses for the marginal). That would give the PPO backend the same std-inflation
mechanism that lets the idealized solver solve `good_init` unaided, rather than
requiring the initialization to be wide enough by hand.
