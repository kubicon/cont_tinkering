# Neural baselines

Same games, same metric, but the setting the method is actually meant for: a policy
network trained from sampled rollouts.

|file|what it is|policy|what converges|
|-|-|-|-|
|`mmd_discrete.py`|our algorithm on a discretized action space|categorical over `bins^d` cells|last iterate (magnet)|
|`nfsp.py`|neural fictitious self-play|RL best response + supervised average|the *average* policy|
|`psro.py`|policy-space response oracles|population + LP meta-solver|the meta-strategy|
|`randomized_policy.py`|Martin & Sandholm's randomized policy networks + simultaneous pseudo-gradient (IJCAI 2023)|implicit `a = f(o, z)`, no density|last iterate|
|`jpspg.py`|joint-perturbation pseudo-gradient (IJCAI 2025) — same policy, cheaper estimator|implicit `a = f(o, z)`|last iterate|
|`br_oracle.py`|the shared RL best-response oracle (not a baseline)|—|—|

```bash
python -m baselines.neural.mmd_discrete configs/two_point.yaml --bins 51 --iters 4000
python -m baselines.neural.nfsp         configs/two_point.yaml --rounds 30
python -m baselines.neural.psro         configs/two_point.yaml --rounds 12
python -m baselines.neural.randomized_policy configs/two_point.yaml --iters 20000
python -m baselines.neural.jpspg        configs/two_point.yaml --iters 20000   # --estimator separate for the IJCAI'23 one
```

Every one takes `--checkpoint-dir DIR` and writes the same run layout the tabular sweep
writes (`history.json`, `meta.json`, `checkpoints/step_*.npz`), plus `params/<name>/` with
the network weights in `training.checkpoint`'s format.

## What is reused

The best-response oracle is `training.mixture_trainer.MixturePPOTrainer` — the repo's own
PPO trainer against a fixed `opponent_action_fn` already *is* an RL best-response oracle,
so NFSP and PSRO best-respond with the same code the method under test trains with.
`br_oracle.py` adds only the opponent side of that: `Opponent` (a jit-safe sampler paired
with a finitely supported snapshot) plus constructors for uniform, single-policy,
population, finite-support and mixed opponents, and `train_best_response` to drive the
trainer quietly. `MixturePPOTrainer.train` gained `verbose=` / `measure_exploitability=`
flags (defaulting to the old behaviour) so it can serve as an inner loop.

The network configuration comes from the *same YAML* as the method under test
(`network:` / `optimizer:` / `ppo:` sections, read through `training.hyperparams`), and
`mmd_discrete` inherits the mixture method's **categorical-head** coefficients
(`ppo.category_entropy_coef`, `ppo.trpo_category_kl_coef`, `ppo.magnet_category_kl_coef`)
as its own — a policy that is nothing but a categorical head.

## Metrics

`expl` always means the same thing: hand a finitely supported strategy to
`baselines.common.GridOracle` and maximize the deviation over a fine grid. A discretized
policy supplies that exactly; a continuous one supplies it by sampling `--samples`
actions. This is stricter than `ZeroSumGame.mixture_exploitability` (gradient ascent from
random restarts), so these numbers are directly comparable to the tabular baselines' and
never under-report a deviation.

Read the algorithm-specific columns as follows:

* `mmd_discrete`: `expl` is the live iterate, `target_expl` the Polyak-averaged one.
* `nfsp`: `expl` is the **average** policies' — the object NFSP's guarantee is about.
  `br_expl` is the best-response pair's, which is not expected to converge; a falling
  `expl` beside a non-falling `br_expl` is what a working run looks like.
* `psro`: `expl` is the meta-strategy's true exploitability; `rl_gap` is what PSRO itself
  believes is left, from its RL responses' realized payoffs. `expl` ≫ `rl_gap` means the
  RL oracle, not the population, is the bottleneck.

## Budget the best-response oracle

Everything NFSP and PSRO claim rests on their inner best response actually being one.
Measured on `configs/two_point.yaml` against the exact Nash — where the exact best-response
value is `0.000`, because no deviation is worth anything at an equilibrium — the PPO oracle
reaches:

|iterations|value of the trained policy|
|-|-|
|100|-0.879|
|400|-0.722|
|1000|-0.019|
|2500|-0.005|

So the defaults are 1000 iterations per round for NFSP and 2000 for PSRO. A too-cheap
oracle is visible in the output: `rl_gap` collapses toward zero while `expl` stays high.
(`tests/test_neural_baselines.py` pins the converged end of this.)

## Deviations worth knowing

* **NFSP uses PPO, not DQN**, for its best response (this repo has PPO; the paper's
  structure — best response to the opponents' average, supervised imitation of one's own
  best responses — is unchanged). Its average policy is a Gaussian mixture fit by maximum
  likelihood to a reservoir; `--average-head reservoir` replaces the fit with the exact
  empirical average, separating "fictitious play did not converge" from "the density
  estimator did not fit".
* **The average net gets its own capacity** (`--average-components`, default 8) rather
  than the config's `network.num_components`: it is a density estimator, not the object
  under test.
* **Best responses are unregularized.** `br_hyperparams` zeroes the magnet/TRPO/entropy
  coefficients the config carries — a best response pulled toward its own past understates
  the opponent's exploitability, which would flatter whatever is calling the oracle.
* **PSRO's payoff matrix is Monte-Carlo.** Each member's action sample is drawn once and
  kept, so entries do not drift between rounds and the LP is not chasing resampling noise.

## The zeroth-order pair (`randomized_policy.py`, `jpspg.py`)

These two are one method with two estimators, so they share the policy, the utility
estimate, the optimizer and the metric; only the perturbation differs.

The policy is *implicit*: `a = f(o, z)` with `z ~ N(0, I)`, squashed into the action box.
It represents arbitrary distributions with a fixed parameter count — and it has **no
tractable density**, which is why the update is zeroth-order rather than a policy
gradient. Nothing that needs `log pi` is available for it: no PPO ratio, no trust region,
and no magnet. That is the trade against the mixture head, stated as a fact of the
representation rather than a tuning choice: unrestricted expressiveness, no
regularizability.

The estimator is the Gaussian-smoothed central difference. `randomized_policy.py`
perturbs one player at a time (`2n` utility evaluations per iteration); `jpspg.py`
perturbs all players jointly and reads both pseudo-gradients off one evaluation pair (2,
regardless of `n`). For two players that is 4 vs 2 — a factor of two here, not the
order-of-magnitude the paper reports for many-player games. `utility_evaluations` is
logged for that reason: comparing these two by iteration count flatters the joint one.

Defaults follow the papers: AdaBelief, `alpha = 1e-4`, `sigma = 0.1` — which means these
runs need *many* iterations (a few hundred barely move the exploitability). The dynamics
their appendix runs are all available: `--dynamics simultaneous|extragradient|optimistic`.
