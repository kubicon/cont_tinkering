"""Neural baselines: the same games, sampled gradients, and a network policy.

`baselines/` one level up varies the *representation* of a mixed strategy with exact
gradients. This package asks the same question in the setting the method is actually
meant for -- a policy network trained from sampled rollouts:

  * `mmd_discrete`  -- our own algorithm (magnetic mirror descent, magnet and all) on a
                       policy that discretizes the action space into a categorical head
                       instead of parametrizing a Gaussian mixture. The sharpest
                       baseline: same update, same rollouts, same regularizers, only the
                       head differs, so whatever separates it from the mixture policy is
                       the head.
  * `nfsp`          -- neural fictitious self-play: an RL best response plus a supervised
                       average policy, the established deep baseline whose guarantee is
                       about the *average* iterate.
  * `psro`          -- policy-space response oracles: a population grown by RL best
                       responses, mixed by an exact meta-solver. The deep counterpart of
                       `baselines/double_oracle.py`.

`sequential_psro`, `sequential_nfsp` and `sequential_rpn` are three of those algorithms
on a *game tree* (Kuhn, Leduc, sequential Blotto) -- PSRO, NFSP, and the randomized policy
network trained by zeroth-order pseudo-gradient. They are separate modules rather than a branch inside
the two above because almost nothing below the round loop survives the move: a strategy
is a policy over infosets instead of a distribution over one action, so there is no
support to carry, no action grid to best-respond over, and the meta-strategy is a
mixture over *policies*. What they do share is everything above that -- the meta-solver
LP, the config loading, the hyperparameters, the run directory -- plus
`sequential_oracle`, the tree counterpart of `br_oracle`; `sequential_scoring`, which
owns the choice of metric (exact on Kuhn, an RL best-response bound elsewhere); and
`sequential_run`, which counts what a run cost in the units they can all be compared in
(wall time, episodes, environment steps). `train_sequential.py` at the repo root drives
them from one config schema, alongside the self-play trainer they are baselines for and
the discretization of it that `games/discretized.py` makes possible.

`br_oracle` holds the shared best-response machinery both NFSP and PSRO need. It is a
thin wrapper over `training.mixture_trainer.MixturePPOTrainer` -- the repo's own PPO
trainer against a fixed opponent already *is* an RL best-response oracle -- so the
oracle inside these baselines is the same code the method under test uses.

All the one-shot baselines report exploitability through `baselines.common.GridOracle`
and checkpoint through `baselines.common.StrategyPair`, exactly as the tabular baselines
do, so a neural curve and a tabular one are the same measurement. The sequential pair
cannot: a deviation in a tree is a policy, not a point of an action grid. They report
`expl` from `games.kuhn_best_response` where an exact tree best response exists and an
`expl_lb` from a trained best response where it does not -- the same number
`best_response.py` reports for a self-play checkpoint.
"""
