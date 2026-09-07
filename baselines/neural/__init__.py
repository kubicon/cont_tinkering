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

`br_oracle` holds the shared best-response machinery both NFSP and PSRO need. It is a
thin wrapper over `training.mixture_trainer.MixturePPOTrainer` -- the repo's own PPO
trainer against a fixed opponent already *is* an RL best-response oracle -- so the
oracle inside these baselines is the same code the method under test uses.

All four report exploitability through `baselines.common.GridOracle` and checkpoint
through `baselines.common.StrategyPair`, exactly as the tabular baselines do, so a
neural curve and a tabular one are the same measurement.
"""
