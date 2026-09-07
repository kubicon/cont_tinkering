"""Non-parametric baselines for the one-shot continuous games in `games/`.

Everything in `run_idealized.py` represents a mixed strategy the same way: as a
`K`-component Gaussian mixture, and varies only the *update rule* acting on that
parametrization. The baselines here vary the other axis -- they keep the game
fixed and change the *representation* of a mixed strategy:

  * `grid_mmd`          -- a probability vector over a fine action grid, moved by
                           the same magnetic mirror descent update the mixture
                           solver uses on its categorical head (tabular MMD).
  * `double_oracle`     -- a finite support grown one best response at a time, with
                           the restricted game solved exactly by LP.
  * `particle_mean_field` -- `M` weighted particles moved by a Wasserstein--Fisher--Rao
                           flow: gradient ascent on the positions, mirror ascent on
                           the weights.

All three report the *same* exploitability number, computed by `common.GridOracle`
against a fine grid of pure deviations, so their curves can be plotted against
`run_idealized.py`'s directly.
"""
