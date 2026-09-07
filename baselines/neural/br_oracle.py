"""The reusable RL best-response oracle, and the opponents one can be trained against.

Both NFSP and PSRO are built out of the same primitive: *train a policy against a fixed
opponent strategy*. The repo already has that -- `training.mixture_trainer.MixturePPOTrainer`
trains one player's `MixtureActorCritic` against an `opponent_action_fn`, which is
exactly an opponent strategy expressed as a sampler -- so nothing here re-implements
PPO, rollouts, or the policy. What this module adds is:

  * `Opponent`, a strategy in the two forms the callers need at once: a *sampler* (what
    the rollout consumes, so it has to be jit-safe) and a finitely supported *snapshot*
    (what the exploitability metric and the checkpoints consume). Keeping them together
    is what stops a run from measuring a different strategy than it trained against.
  * constructors for the opponents these baselines need: uniform, a single policy, a
    weighted population (PSRO's meta-strategy), a finite support (a reservoir, or a
    strategy loaded from a tabular baseline), and a mixture of opponents (NFSP's
    anticipatory play).
  * `train_best_response`, which drives the trainer quietly and hands back the trained
    policy.

**Why the best response is unregularized by default.** `build_hyperparams` fills in
whatever magnet/TRPO/entropy coefficients the config carries -- they are the method
under test. A best response with a magnet term is not a best response: it is pulled
towards its own past, which *understates* the opponent's exploitability and so flatters
whatever algorithm is calling this oracle. `br_hyperparams` therefore zeroes them, and
says so in the returned object; pass `keep_regularizers=True` to override deliberately.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.base import ZeroSumGame
from training.config import MixturePPOHyperparams
from training.hyperparams import build_hyperparams
from training.mixture import MixtureActorCritic, sample_mixture_actions
from training.mixture_trainer import MixturePPOTrainer
from training.run_config import RunConfig

# The observation of a one-shot game is a constant, so the key it is drawn with is
# irrelevant; fixed here so every readout in these baselines is the same observation.
OBSERVATION_KEY = jax.random.PRNGKey(0)

# Coefficients that would bias a best response towards something other than the best
# response. See the module docstring.
BR_REGULARIZERS = (
    "category_entropy_coef", "gaussian_entropy_coef",
    "trpo_category_kl_coef", "trpo_gaussian_kl_coef",
    "magnet_category_kl_coef", "magnet_gaussian_kl_coef",
)

SampleFn = Callable[[chex.PRNGKey, int], chex.Array]


@dataclasses.dataclass(frozen=True)
class Opponent:
    """A fixed strategy to best-respond to, in both forms the callers need.

    `sample_fn(key, n) -> (n, d)` is consumed inside a jitted rollout; `support`
    `(N, d)` / `weights` `(N,)` is the same strategy as a finitely supported snapshot,
    for `GridOracle` and for checkpoints. For a continuous policy the snapshot is a
    sample of it (see `baselines.neural.common.empirical_strategy`), so it carries
    Monte-Carlo error while the sampler does not.
    """

    sample_fn: SampleFn
    support: np.ndarray
    weights: np.ndarray
    label: str

    def __call__(self, key: chex.PRNGKey, num_samples: int) -> chex.Array:
        return self.sample_fn(key, num_samples)


def _snapshot(sample_fn: SampleFn, key: chex.PRNGKey, samples: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(sample_fn(key, samples), dtype=np.float64)
    return actions, np.full(actions.shape[0], 1.0 / actions.shape[0])


def uniform_opponent(game: ZeroSumGame, player: int, samples: int = 4096,
                     key: chex.PRNGKey | None = None) -> Opponent:
    """Uniform random play -- PSRO's usual seed policy, and a sanity baseline."""
    space = game.action_space(player)

    def sample_fn(k: chex.PRNGKey, n: int) -> chex.Array:
        return space.sample(k, (n,))

    support, weights = _snapshot(sample_fn, key if key is not None else OBSERVATION_KEY, samples)
    return Opponent(sample_fn, support, weights, f"uniform(player {player})")


def policy_opponent(game: ZeroSumGame, player: int, network: MixtureActorCritic, params,
                    samples: int = 4096, key: chex.PRNGKey | None = None,
                    label: str = "policy") -> Opponent:
    """One `MixtureActorCritic` policy, frozen."""
    space = game.action_space(player)
    obs = game.observation(player, OBSERVATION_KEY)

    def sample_fn(k: chex.PRNGKey, n: int) -> chex.Array:
        return sample_mixture_actions(network, params, obs, space, k, n)

    support, weights = _snapshot(sample_fn, key if key is not None else OBSERVATION_KEY, samples)
    return Opponent(sample_fn, support, weights, f"{label}(player {player})")


def population_opponent(game: ZeroSumGame, player: int, network: MixtureActorCritic,
                        params_list: Sequence, meta_weights, samples: int = 4096,
                        key: chex.PRNGKey | None = None) -> Opponent:
    """A weighted population of policies -- PSRO's meta-strategy.

    Every member is sampled and the draw is then *selected* per row, rather than
    gathering each row's parameters: the members share an architecture, so one `vmap`
    over the stacked parameters costs `len(params_list)` times the sampling and nothing
    in gathers or control flow, which is the cheaper trade at the population sizes PSRO
    reaches here.
    """
    if len(params_list) != len(meta_weights):
        raise ValueError(f"{len(params_list)} policies but {len(meta_weights)} weights")
    space = game.action_space(player)
    obs = game.observation(player, OBSERVATION_KEY)
    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *params_list)
    log_weights = jnp.log(jnp.clip(jnp.asarray(np.asarray(meta_weights, dtype=np.float64)), 1e-12, None))
    members = len(params_list)

    def sample_fn(k: chex.PRNGKey, n: int) -> chex.Array:
        select_key, action_key = jax.random.split(k)
        per_member = jax.vmap(
            lambda p, mk: sample_mixture_actions(network, p, obs, space, mk, n)
        )(stacked, jax.random.split(action_key, members))          # (M, n, d)
        index = jax.random.categorical(select_key, log_weights, shape=(n,))
        return per_member[index, jnp.arange(n)]

    support, weights = _snapshot(sample_fn, key if key is not None else OBSERVATION_KEY, samples)
    return Opponent(sample_fn, support, weights, f"population[{members}](player {player})")


def finite_opponent(support, weights, label: str = "finite") -> Opponent:
    """A finitely supported strategy: a reservoir of actions, a tabular baseline's
    checkpoint, or any `StrategyPair` side. Sampled by drawing rows."""
    support_j = jnp.asarray(np.asarray(support, dtype=np.float64))
    log_weights = jnp.log(jnp.clip(jnp.asarray(np.asarray(weights, dtype=np.float64)), 1e-12, None))

    def sample_fn(k: chex.PRNGKey, n: int) -> chex.Array:
        return support_j[jax.random.categorical(k, log_weights, shape=(n,))]

    return Opponent(sample_fn, np.asarray(support, dtype=np.float64),
                    np.asarray(weights, dtype=np.float64), label)


def mix_opponents(opponents: Sequence[Opponent], probs: Sequence[float],
                  samples: int = 4096, key: chex.PRNGKey | None = None) -> Opponent:
    """Play `opponents[i]` with probability `probs[i]` -- NFSP's anticipatory mixture
    of a player's best-response and average policies."""
    if len(opponents) != len(probs):
        raise ValueError(f"{len(opponents)} opponents but {len(probs)} probabilities")
    log_probs = jnp.log(jnp.clip(jnp.asarray(np.asarray(probs, dtype=np.float64)), 1e-12, None))

    def sample_fn(k: chex.PRNGKey, n: int) -> chex.Array:
        keys = jax.random.split(k, len(opponents) + 1)
        drawn = jnp.stack([o.sample_fn(keys[i + 1], n) for i, o in enumerate(opponents)])  # (M, n, d)
        index = jax.random.categorical(keys[0], log_probs, shape=(n,))
        return drawn[index, jnp.arange(n)]

    support, weights = _snapshot(sample_fn, key if key is not None else OBSERVATION_KEY, samples)
    label = " + ".join(f"{p:.2f}*{o.label}" for o, p in zip(opponents, probs))
    return Opponent(sample_fn, support, weights, label)


# --------------------------------------------------------------------------- oracle


@dataclasses.dataclass
class BestResponse:
    """A trained best response: the policy, and what it is worth against its opponent."""

    network: MixtureActorCritic
    params: chex.ArrayTree
    hyperparams: MixturePPOHyperparams
    history: list[dict]
    mean_reward: float          # realized payoff over the last training chunk
    opponent_label: str

    def opponent(self, game: ZeroSumGame, player: int, samples: int = 4096,
                 key: chex.PRNGKey | None = None) -> Opponent:
        """This policy as an `Opponent` -- i.e. frozen, for the other side to respond to."""
        return policy_opponent(game, player, self.network, self.params, samples, key, "br")


def br_hyperparams(game: ZeroSumGame, player: int, config: RunConfig,
                   keep_regularizers: bool = False, **overrides) -> MixturePPOHyperparams:
    """`training.hyperparams.build_hyperparams`, with the regularizers stripped.

    See the module docstring for why. `overrides` are applied last, so a caller can put
    a specific coefficient back (a small entropy bonus to keep exploration alive, say)
    without keeping all six.
    """
    hyperparams = build_hyperparams(game, player, config)
    if not keep_regularizers:
        hyperparams = dataclasses.replace(hyperparams, **{name: 0.0 for name in BR_REGULARIZERS})
    return dataclasses.replace(hyperparams, **overrides) if overrides else hyperparams


def train_best_response(
    game: ZeroSumGame,
    player: int,
    opponent: Opponent,
    hyperparams: MixturePPOHyperparams,
    steps: int = 20,
    epochs: int = 10,
    seed: int = 0,
    init_params: chex.ArrayTree | None = None,
    verbose: bool = False,
) -> BestResponse:
    """Train `player`'s policy against `opponent` with the repo's own PPO trainer.

    `init_params` warm-starts from a previous best response -- NFSP's best-response net
    is meant to be *continuously* improved rather than restarted every round. Note that
    only the parameters carry over: `MixturePPOTrainer` builds its own optimizer state,
    so Adam's moments restart with each call (the same limitation
    `MixturePPOTrainer.load` documents).
    """
    trainer = MixturePPOTrainer(game, hyperparams, opponent.sample_fn, perspective=player, seed=seed)
    if init_params is not None:
        trainer.state = trainer.state.replace(
            params=init_params, target_params=init_params, magnet_params=init_params
        )
    history = trainer.train(steps, epochs, verbose=verbose, measure_exploitability=False)
    return BestResponse(
        network=trainer.network,
        params=trainer.state.params,
        hyperparams=hyperparams,
        history=history,
        mean_reward=float(history[-1]["mean_reward"]) if history else float("nan"),
        opponent_label=opponent.label,
    )


def payoff_estimate(game: ZeroSumGame, actions_0, actions_1) -> float:
    """`E[u]` for two independent samples of the players' strategies.

    The outer product, not the diagonal: `actions_0[i]` and `actions_1[i]` are two
    independent draws, and pairing them row-wise would estimate the same expectation
    with `n` samples instead of `n^2` -- which matters, because PSRO's meta-game is
    built out of these numbers and its LP is sensitive to their noise.
    """
    a0 = jnp.asarray(actions_0, dtype=jnp.float64)
    a1 = jnp.asarray(actions_1, dtype=jnp.float64)
    payoffs = jax.vmap(lambda x: jax.vmap(lambda y: game.payoff(x, y))(a1))(a0)
    return float(jnp.mean(payoffs))
