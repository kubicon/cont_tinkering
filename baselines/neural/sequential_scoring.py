"""Measuring a sequential strategy pair: exact where the tree allows it, RL where it does not.

`baselines.common.GridOracle` -- the one metric every one-shot baseline reports
-- does not survive the move to a tree: it maximizes a deviation over a grid of
*actions*, and a deviation here is a whole policy over infosets. What replaces
it depends on the game, so the choice is made here rather than inside either
algorithm:

  * **Kuhn** has an exact best response (`games.kuhn_best_response`), cheap
    enough to run every round. It is exact given the bet grid, hence a lower
    bound in the same benign sense the trainer's own column is: refine the grid
    until it stops moving.
  * **Leduc and sequential Blotto** have none -- their public state carries real
    accumulated bet sizes, so there is no finite infoset enumeration to traverse
    (see `games/leduc.py`). The only available number is what
    `best_response.py` reports for a trained checkpoint: train an approximate
    best response and read its value. That is a lower bound of an entirely
    different quality -- it is only as good as the PPO run inside it -- so it is
    reported under its own name (`expl_lb`) and is off by default.

**A discretized run is measured differently by the two.** `expl` on Kuhn
best-responds over the fine evaluation grid regardless of what the players can
play, so it charges a `games.discretized` run the full price of its action set,
off-grid deviations included. `expl_lb` cannot: a trained responder plays the
same game its opponent does, so on a discretized Leduc it only searches the same
grid and the bound understates by whatever lies between the points. Compare
`expl_lb` between a discretized run and a continuous one with that in mind.

Both take `PolicyMixture`s, not networks: PSRO's meta-strategy is a mixture, and
a mixture is a genuinely different object from the policy you would get by
averaging its members (`games.kuhn_best_response.mix_strategies`).
"""

from __future__ import annotations

from typing import Callable

import chex
import jax

from games.kuhn_best_response import (
    KuhnStrategy,
    best_response_value_first,
    best_response_value_second,
    bet_grid,
    game_value,
    mix_strategies,
)
from games.discretized import base_game
from games.sequential import SequentialZeroSumGame
from games.sequential_examples import ContinuousKuhnPoker
from training.kuhn_evaluation import strategy_from_policy
from training.mixture import build_mixture_network

from . import sequential_oracle as so
from .sequential_oracle import MixtureEvaluator, PolicyMixture


def has_exact_exploitability(game: SequentialZeroSumGame) -> bool:
    """Is there an exact tree best response for this game? (Kuhn: yes. The rest: no.)

    Read through a discretization (`games.discretized`): the tree is what decides
    this, and wrapping a game's action set into a grid does not change its tree.
    """
    return isinstance(base_game(game), ContinuousKuhnPoker)


def kuhn_strategy_of(game: SequentialZeroSumGame, mixture: PolicyMixture, player: int,
                     grid: chex.Array) -> KuhnStrategy:
    """The mixture as one `KuhnStrategy`: every member tabulated, then realization-mixed.

    `game` is the game the *policies play* -- discretized or not -- because that
    is what says how to read a policy's bet-size distribution; the arithmetic
    downstream runs on the tree underneath either way.
    """
    strategies = [
        strategy_from_policy(game, mixture.network, params, player, grid)
        for params in mixture.params
    ]
    return mix_strategies(strategies, list(mixture.probs), player)


def exact_kuhn_exploitability(
    game: SequentialZeroSumGame,
    mixtures: tuple[PolicyMixture, PolicyMixture],
    grid_points: int | None = None,
) -> dict[str, float]:
    """`expl`, both best-response values and the pair's value -- exact on the bet grid.

    The best response is over the *evaluation* grid, which is fine (1025 points by
    default) whatever the players' own action set is. For a discretized run that
    is the point: the responder may bet between the grid points its opponent is
    restricted to, so `expl` charges the discretization for exactly what it gave up.
    """
    base = base_game(game)
    grid = bet_grid(base) if grid_points is None else bet_grid(base, grid_points)
    strategy_0 = kuhn_strategy_of(game, mixtures[0], 0, grid)
    strategy_1 = kuhn_strategy_of(game, mixtures[1], 1, grid)
    br_0 = float(best_response_value_first(base, grid, strategy_1))
    br_1 = float(best_response_value_second(base, grid, strategy_0))
    return {"expl": br_0 + br_1, "br_0": br_0, "br_1": br_1,
            "value": float(game_value(base, grid, strategy_0, strategy_1))}


def rl_exploitability_bound(
    game: SequentialZeroSumGame,
    mixtures: tuple[PolicyMixture, PolicyMixture],
    hyperparams: tuple,
    steps: int = 50,
    epochs: int = 20,
    seed: int = 0,
    episodes: int = 20_000,
    key: chex.PRNGKey | None = None,
    evaluator: MixtureEvaluator | None = None,
    on_response: Callable[[int, so.SequentialBestResponse, float], None] | None = None,
) -> dict[str, float]:
    """Train a best response to each side and add up what they win: a lower bound on `expl`.

    `on_response(player, response, value)`, if given, receives each trained
    response and its measured value (to the responder) once it has been played --
    the hook a caller saves the responses through; they are discarded otherwise.

    The two responses are measured by *playing* them (`MixtureEvaluator`) rather
    than by reading the last training chunk's mean payoff: the chunk average is
    taken over parameters that were still moving, and over the exploratory
    rollouts that moved them, so it both lags and under-reports.

    Costs two best responses -- as much as a round of PSRO or NFSP itself -- so
    callers run it every `--score-every` rounds, not every round.
    """
    key = jax.random.PRNGKey(seed) if key is None else key
    values = []
    for player in (0, 1):
        response = so.train_sequential_best_response(
            game, player, mixtures[1 - player], hyperparams[player],
            steps=steps, epochs=epochs, seed=seed + player,
        )
        responder = response.as_mixture(f"scoring_br_{player}")
        pair = (responder, mixtures[1]) if player == 0 else (mixtures[0], responder)
        scorer = MixtureEvaluator(game, (pair[0].network, pair[1].network)) if evaluator is None else evaluator
        key, eval_key = jax.random.split(key)
        mean, _ = scorer.evaluate(pair[0], pair[1], eval_key, num_episodes=episodes)
        values.append(mean if player == 0 else -mean)
        if on_response is not None:
            on_response(player, response, float(values[-1]))
    return {"expl_lb": values[0] + values[1], "br_lb_0": values[0], "br_lb_1": values[1]}


def build_scorer(
    game: SequentialZeroSumGame,
    hyperparams: tuple,
    exact_grid: int | None = None,
    score_every: int = 0,
    br_steps: int = 50,
    br_epochs: int = 20,
    episodes: int = 20_000,
    seed: int = 0,
) -> Callable[..., dict[str, float]]:
    """`(mixtures, round_index, full=True) -> metrics`: what this game supports measuring.

    Always reports the pair's own value `h2h` (the two strategies played against
    each other), which costs one batch of episodes and is the only column
    available on every game every round. `expl` joins it on Kuhn; `expl_lb` joins
    it every `score_every` rounds when asked for.

    `force=True` runs the RL bound regardless of the round count, so a run can
    always score the strategy it finishes on rather than whichever round last
    happened to be a multiple of `score_every`.

    `full=False` asks for only the free measurements -- the exact ones, where they
    exist. That is for a *secondary* strategy pair, such as NFSP's best responses
    beside its averages: worth a column when it costs a tree traversal, not worth
    two more PPO runs and a batch of hands.
    """
    evaluator = MixtureEvaluator(
        game, (build_mixture_network(hyperparams[0]), build_mixture_network(hyperparams[1])))
    exact = has_exact_exploitability(game)
    key = jax.random.PRNGKey(seed + 7919)

    def score(mixtures: tuple[PolicyMixture, PolicyMixture], round_index: int,
              full: bool = True, force: bool = False) -> dict[str, float]:
        nonlocal key
        row = {}
        if full:
            key, h2h_key = jax.random.split(key)
            mean, stderr = evaluator.evaluate(mixtures[0], mixtures[1], h2h_key,
                                              num_episodes=min(episodes, 20_000))
            row.update({"h2h": mean, "h2h_stderr": stderr})
        if exact:
            row.update(exact_kuhn_exploitability(game, mixtures, exact_grid))
        if full and score_every and (force or round_index % score_every == 0):
            key, br_key = jax.random.split(key)
            row.update(rl_exploitability_bound(
                game, mixtures, hyperparams, steps=br_steps, epochs=br_epochs,
                seed=seed + 1000 * round_index, episodes=episodes, key=br_key,
                evaluator=evaluator,
            ))
        return row

    return score
