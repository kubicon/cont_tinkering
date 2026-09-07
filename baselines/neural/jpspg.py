"""Joint-perturbation simultaneous pseudo-gradient (JPSPG).

Martin & Sandholm, IJCAI 2025 (arXiv:2408.09306). The follow-up to the zeroth-order
method in `randomized_policy.py`, and a one-line change to it: instead of perturbing one
player's parameters at a time, perturb **all players at once** with a single draw and
read every player's pseudo-gradient off that same evaluation,

    g = (1/sigma) * u(theta + sigma z) (*) z          [(*) = elementwise, per player block]

which is exact in expectation for the same smoothed objective because the diagonal of
the pseudo-Jacobian is `a (*) b` when the full object is `a (x) b`. The cost per
iteration drops from linear in the number of players to **constant**: with the
central-difference estimator this repo uses, 2 utility evaluations instead of `2n`.

For the two-player zero-sum games here that is 2 instead of 4 -- a factor of two, not the
order-of-magnitude the paper reports for many-player games, so do not expect this to look
dramatic on `configs/`. What it is here is the correct, cheaper estimator of the same
quantity, and the honest thing to run when comparing against that line of work. Both
estimators are available behind `--estimator`, sharing the policy, the utility estimate,
the optimizer and the metric with `randomized_policy.py`, so a comparison between them is
a comparison of estimators and nothing else. `utility_evaluations` is logged for exactly
that reason: iteration counts flatter the joint estimator, evaluation counts do not.

Usage:
    python -m baselines.neural.jpspg configs/two_point.yaml --iters 20000
    python -m baselines.neural.jpspg configs/two_point.yaml --estimator separate   # the IJCAI'23 one
"""

from __future__ import annotations

import dataclasses

import chex
import jax

from ..common import GridOracle
from .common import RunWriter, load_run, neural_parser
from .randomized_policy import add_arguments, hyperparams_from_config, report, \
    run_pseudo_gradient, spg_pseudo_gradients, tree_add_scaled, tree_normal, tree_scale


def jpspg_pseudo_gradients(utility, params, key: chex.PRNGKey, sigma: float,
                           antithetic: bool = True) -> tuple[tuple, int]:
    """Both players' pseudo-gradients from **one** joint perturbation.

    In a two-player zero-sum game the two players' utilities are `+u` and `-u` at the
    *same* perturbed profile, so a single (central-differenced) evaluation of `u` gives
    both coefficients -- which is the whole saving. Note the noise blocks stay per-player:
    each player's gradient is its own perturbation scaled by its own utility difference,
    not the other's.
    """
    noise_key_0, noise_key_1, utility_key = jax.random.split(key, 3)
    noise = (tree_normal(noise_key_0, params[0]), tree_normal(noise_key_1, params[1]))

    plus = tuple(tree_add_scaled(params[i], noise[i], sigma) for i in (0, 1))
    u_plus = utility(plus, utility_key)
    if antithetic:
        minus = tuple(tree_add_scaled(params[i], noise[i], -sigma) for i in (0, 1))
        u_minus = utility(minus, utility_key)
        coefficient = (u_plus - u_minus) / (2.0 * sigma)
        evaluations = 2
    else:
        coefficient = u_plus / sigma
        evaluations = 1

    # Player 1 maximizes the negated payoff, so its coefficient is the negated one.
    return (tree_scale(noise[0], coefficient), tree_scale(noise[1], -coefficient)), evaluations


ESTIMATORS = {"joint": jpspg_pseudo_gradients, "separate": spg_pseudo_gradients}


def main() -> None:
    ap = neural_parser(__doc__)
    add_arguments(ap)
    ap.add_argument("--estimator", choices=tuple(ESTIMATORS), default="joint",
                    help="'joint' is JPSPG; 'separate' is the per-player SPG it replaces")
    args = ap.parse_args()

    game, game_config, config = load_run(args.config)
    oracle = GridOracle(game, points=args.grid)
    hyperparams = hyperparams_from_config(game, config, args)
    per_iteration = (2 if args.estimator == "joint" else 4) // (1 if not args.no_antithetic else 2)

    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"policy  : randomized network a=f(o,z), noise {hyperparams.noise_dim}, "
          f"hidden {hyperparams.hidden_dims}")
    print(f"solver  : {args.estimator} pseudo-gradient  sigma={hyperparams.sigma}  "
          f"lr={hyperparams.learning_rate}  {hyperparams.optimizer}  "
          f"dynamics={hyperparams.dynamics}  "
          f"({per_iteration} utility evaluations/iteration)\n")

    meta = {"algorithm": f"{args.estimator}_pseudo_gradient", "config": args.config,
            "hyperparams": hyperparams.to_dict(),
            **{k: v for k, v in vars(args).items() if k != "config"}}
    writer = RunWriter(args.checkpoint_dir, meta) if args.checkpoint_dir else None
    result = run_pseudo_gradient(game, oracle, hyperparams, ESTIMATORS[args.estimator],
                                 iterations=args.iters, log_every=args.log_every,
                                 samples=args.samples, seed=args.seed, writer=writer,
                                 estimator_name=args.estimator)
    report(args, game, game_config, oracle, hyperparams, result, writer, meta)


if __name__ == "__main__":
    main()
