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

And the same two algorithms on a **game tree** (Kuhn, Leduc, sequential Blotto):

|file|what it is|policy|what converges|
|-|-|-|-|
|`sequential_nfsp.py`|NFSP on a tree|RL best response + supervised average, both over infosets|the *average* policy|
|(`games/discretized.py`)|the discretization baseline: our algorithm, categorical head only|categorical over `bins` actions per node|last iterate (magnet)|
|`sequential_rpn.py`|randomized policy networks by zeroth-order pseudo-gradient, on a tree|implicit `(kind, size) = f(o, z)`, no density|last iterate|
|`sequential_psro.py`|PSRO on a tree|population of policies + LP meta-solver|the meta-strategy|
|`sequential_oracle.py`|the tree best-response oracle and `PolicyMixture` (not a baseline)|—|—|
|`sequential_scoring.py`|what "exploitability" means per game (not a baseline)|—|—|
|`sequential_run.py`|what a run cost — wall time, episodes, env steps — and the run log (not a baseline)|—|—|

All five sequential solvers — including the repo's own self-play PPO and the
discretization baseline — also run from `train_sequential.py`, which is the entry point to
use for a comparison:

```bash
python train_sequential.py configs/kuhn_solvers.yaml --solver self_play    --checkpoint-dir data/kuhn_sp
python train_sequential.py configs/kuhn_solvers.yaml --solver discrete_mmd --checkpoint-dir data/kuhn_disc
python train_sequential.py configs/kuhn_solvers.yaml --solver nfsp         --checkpoint-dir data/kuhn_nfsp
python train_sequential.py configs/kuhn_solvers.yaml --solver psro         --checkpoint-dir data/kuhn_psro
python train_sequential.py configs/kuhn_solvers.yaml --solver rpn          --checkpoint-dir data/kuhn_rpn
```

`discrete_mmd` is the repo's own algorithm with **only the categorical head**: the game's
continuous action is split into `discrete.bins` evenly spaced actions
([`games/discretized.py`](../../games/discretized.py)) and the continuous branch is masked
illegal everywhere, so the mixture policy's atoms carry the whole strategy. It runs the
*same* trainer, rollout, loss and schedule as `self_play` — `SOLVER_RUNNERS` maps both to
the same function — which is what makes the pair a controlled comparison of the
representation rather than of two implementations. (`atom_frac` in the log is exactly
`1.00` on such a run; that is the assertion, not a coincidence.) One caveat on the metric:
`expl` on Kuhn best-responds over the fine evaluation grid whatever the players can play,
so it charges the grid its full price, while `expl_lb` on Leduc/Blotto trains a responder
*inside* the discretized game and therefore cannot see off-grid deviations.

`rpn` is Martin & Sandholm's randomized policy network trained by zeroth-order
pseudo-gradient ([`sequential_rpn.py`](sequential_rpn.py)), the tree counterpart of
[`randomized_policy.py`](randomized_policy.py). It is the one solver here whose *policy
class* differs: `(kind, size) = f(o, z)` with the kind read off a masked argmax, implicit
and with no density, so it shares the torso shape with the others and nothing else — no
PPO ratio, no trust region, no magnet, none of which can be written without a `log pi`.
Three consequences show up in its log and are worth expecting:

* **No `loss` column**, because the method never forms one. What it ascends is the
  utility, which is the `h2h` column; `grad_norm_*` are the pseudo-gradients' norms, and a
  norm that will not shrink while `expl` stalls means the estimator is returning noise —
  raise `rpn.perturbation_batch` before touching the learning rate.
* **It is expensive per iteration.** One iteration plays
  `perturbation_batch * utility_episodes` hands (twice that under the `separate`
  estimator, which is the IJCAI'23 method; `joint` is the IJCAI'25 JPSPG that halves it).
  In the one-shot version a utility evaluation is a vectorized payoff and nearly free; on
  a tree it is a batch of rollouts. Compare this solver against the others on `env_steps`,
  never on iteration counts — which is precisely why those columns exist.
* **Its `expl` is estimated, not exact.** With no density, the behavioral strategy has to
  be *sampled* out of the policy (`rpn.strategy_samples` noise draws per infoset) before
  Kuhn's exact best response can be run against it, so the number carries
  `1/sqrt(samples)` error. Its `expl_lb` off Kuhn is a best response trained by the same
  zeroth-order ascent — the method's own oracle, weaker than the PPO responder the other
  solvers get, and warm-started from the run's own policy.

It reads `train.solver` plus the `discrete:` / `nfsp:` / `psro:` / `rpn:` / `scoring:`
sections of the shared YAML schema, scores every solver through the *same* metric where
the metric exists, and writes one run directory per
run: `history.json` and a streamed `metrics.jsonl` (every log point carries `wall_time`,
`total_wall_time`, `episodes`, `env_steps`, `iterations`, the interval's mean `loss` and
`grad_norm`, and whatever exploitability the game supports), plus `checkpoints/{t}.pkl`.
Environment steps are exact for training batches and estimated at the measured mean
episode length for evaluation rollouts; measurement is charged to `total_wall_time` and to
neither step counter, so a run scored every round stays comparable with one scored twice.
Its own docstring has the details.

The module CLIs below stay as they are — one algorithm, its own flags, no config sections
needed:

```bash
python -m baselines.neural.mmd_discrete configs/two_point.yaml --bins 51 --iters 4000
python -m baselines.neural.nfsp         configs/two_point.yaml --rounds 30
python -m baselines.neural.psro         configs/two_point.yaml --rounds 12
python -m baselines.neural.randomized_policy configs/two_point.yaml --iters 20000
python -m baselines.neural.jpspg        configs/two_point.yaml --iters 20000   # --estimator separate for the IJCAI'23 one

python -m baselines.neural.sequential_psro configs/kuhn.yaml  --rounds 12
python -m baselines.neural.sequential_nfsp configs/leduc.yaml --rounds 30 --score-every 5
```

Every one takes `--checkpoint-dir DIR` and writes the same run layout the tabular sweep
writes (`history.json`, `meta.json`, `checkpoints/step_*.npz`), plus `params/<name>/` with
the network weights in `training.checkpoint`'s format. The two sequential runs write that
layout minus the per-iteration `checkpoints/`: a `StrategyPair` is a support and a weight
vector, which a policy over infosets has none of. They save weights instead — every
population member for PSRO (plus `meta.npz`, the meta-weights and the empirical payoff
matrix), the average and best-response nets for NFSP.

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

## The sequential pair

Same algorithms, different objects. A strategy in a tree is a policy over infosets, so
there is no support to carry and no action grid to best-respond over; the meta-strategy
and the anticipatory mixture are mixtures over *policies*, drawn once per hand and held
(`sequential_oracle.PolicyMixture`). That is a mixed strategy, not a behavioral one, and
the difference is measurable: `games.kuhn_best_response.mix_strategies` converts one into
the realization-equivalent behavioral strategy, and the entrywise average of the members'
tables -- the thing it is easy to mistake it for -- gives a different value
(`tests/test_sequential_baselines.py` pins that).

Everything above the round loop is shared with the one-shot pair: the meta-solver LP, the
config schema, `training.hyperparams`, `br_hyperparams` (regularizers zeroed, same
reasoning), the run directory. The oracle underneath is the repo's own sequential PPO
(`training/sequential_rollout.py`), i.e. again the code the method under test trains with
-- `collect_sequential_batch` gained a `param_axes` argument so one player's parameters
can vary per episode, which is all a mixture opponent needs.

Two constraints are worth knowing before reading a run:

* **One architecture per run.** A batched tree rollout evaluates both players' heads and
  selects between them, and plays a mixture by gathering one member's parameters per
  episode out of one stacked pytree — so every policy in the run has the same shape.
  `network.num_components` in the config therefore also sizes NFSP's average net, which
  the one-shot version gives its own `--average-components`.
* **`--average-head reservoir` has no counterpart.** A one-shot reservoir of actions *is*
  a strategy; a reservoir of infoset rows is not, so the supervised fit is the only
  average policy available.
* **The discretization baseline is a wrapped game, not a second policy.** `mmd_discrete.py`
  is a separate implementation for the one-shot case (its own categorical actor-critic, its
  own rollout); on a tree, `games/discretized.py` turns the continuous branch into atoms
  instead, so the existing policy, rollout and loss run unchanged and the only difference
  from a `self_play` run is the action set. Run it with `train_sequential.py --solver
  discrete_mmd`; `discrete.bins` is its hyperparameter.

### Validated against classic Kuhn

`configs/kuhn_classic.yaml` fixes the bet size, so the bet grid holds one point, `expl`
carries no approximation at all, and the equilibrium value is known to be `-1/18 =
-0.0556`. Both solvers were run there at the default oracle budget (1000 PPO iterations
per best response, seed 0):

|solver|rounds|`expl` start → end|final `meta_value` / `h2h`|wall time|
|-|-|-|-|-|
|`sequential_psro`|10|`+0.804` → `+0.007`|`-0.0552`|402s|
|`sequential_nfsp`|20|`+0.661` → `+0.03`–`+0.12`|`-0.043`|735s|

PSRO's meta-game value landing on the known `-1/18` is the check worth having: it goes
through the population, the LP, the Monte-Carlo payoff matrix *and* the mixed→behavioral
conversion, so it is not a number any one of those could produce alone. NFSP's average is
the noisier of the two here (its round-to-round `expl` oscillates by a factor of three
late in the run) while its `br_expl` stays up at `0.33`–`0.83` — the shape a working NFSP
run has, since the best responses are not supposed to converge.

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

In a tree the grid oracle does not exist, so the sequential pair reports:

* `expl` — **Kuhn only**, exact on the bet grid (`games/kuhn_best_response.py`), computed
  every round because it is a tree traversal rather than a training run.
* `expl_lb` — Leduc and sequential Blotto have no finite infoset enumeration, so the only
  available number is a *trained* best response's value, exactly what `best_response.py`
  reports for a self-play checkpoint. It costs as much as a round of the algorithm, so it
  is off by default: `--score-every N`. It is a lower bound of a different quality from
  Kuhn's — it is only as good as the PPO run inside it.
* `h2h` — the two strategies' own value against each other, on every game every round.
* `rl_gap` (PSRO) and `br_expl` (NFSP) mean what they do above.

A sequential NFSP run also writes its average policies as a `player_0`/`player_1`
checkpoint, so a finished run can be scored offline by `best_response.py` with no
NFSP-aware loader in the picture:

```bash
python -m baselines.neural.sequential_nfsp configs/leduc.yaml --rounds 30 --checkpoint-dir data/leduc_nfsp
python best_response.py configs/leduc_br.yaml --checkpoint-dir data/leduc_nfsp/average_checkpoint --responder both
```

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

The table above is a *cold-started* oracle, and only a cold-started one behaves this way.
**Neither solver warm-starts its best-response net from the previous round**, and neither
should: with the regularizers zeroed (below), the Gaussian scale is the policy's only
exploration, so a warm start from last round's converged — hence near-deterministic —
policy makes each round local ascent around wherever it last stopped, and no iteration
budget fixes it. Measured on `configs/circle.yaml` against a fixed opponent at the default
1000 iterations: cold start reaches `+0.700` against an exact best response of `+0.703`;
warm start reaches `+0.356`. NFSP warm-started until it was removed, which is why its
`circle` exploitability sat near `1.4` while PSRO — the same oracle, cold-started — reached
`0.000`.

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
  the opponent's exploitability, which would flatter whatever is calling the oracle. The
  cost is that the oracle has no exploration beyond its own scale, hence no warm starts
  (above).
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

Defaults follow the papers: AdaBelief, `alpha = 1e-4`, `sigma = 0.1`,
`--perturbation-batch 256` — which means these runs need *many* iterations (a few hundred
barely move the exploitability). The dynamics their appendix runs are all available:
`--dynamics simultaneous|extragradient|optimistic`.

**`--perturbation-batch` is the one that decides whether these converge at all.** It is
the papers' `batch_size`: how many perturbations are drawn and averaged per iteration.
A single draw is a random direction in `R^d`, so at the `d ~ 5e3` of a 64x64 policy it
has cosine `~0.02` with the pseudo-gradient it estimates — the step is ~99% noise.
IJCAI'25's experiments use 256; the authors' published reference implementation
*defaults* to 2, and this repo inherited that default, which is why both methods sat at
their starting exploitability across `experiments/one_shot_neural/`. The budget also has
to go on the right axis: at a fixed payoff-evaluation budget, moving samples out of
`--utility-samples` into `--perturbation-batch` improves the gradient's alignment ~4-5x,
because the utility estimate was already far more precise than the direction was.
