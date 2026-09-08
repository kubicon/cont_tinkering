"""The RL best-response oracle for *game trees*, and the opponents one runs against.

The one-shot counterpart is `baselines.neural.br_oracle`, and this module is the
same idea one level up: NFSP and PSRO are both "train a policy against a fixed
opponent strategy", so the oracle is the repo's own PPO -- here
`training.sequential_rollout` + `training.ppo`, the very code
`SequentialSelfPlayPPOTrainer` trains with -- and nothing below is a new
algorithm.

**What a strategy is here.** In a one-shot game a mixed strategy is a
distribution over actions, so `br_oracle.Opponent` can carry one as a sampler
plus a finite support. A tree has no such object: a strategy is a policy over
infosets, and the thing PSRO's meta-solver and NFSP's anticipatory rule mix is
whole *policies*. `PolicyMixture` is therefore a list of parameter sets with
probabilities -- a mixed strategy over behavioral ones, drawn once per episode
and held for the hand. That distinction is not pedantic: drawing a fresh member
at every decision would be a different strategy (see `mix_strategies` in
`games.kuhn_best_response` for the conversion between the two, and why the
naive entry-by-entry average is the wrong object).

**One architecture per run.** Every member of a mixture, and both players, share
one `MixtureActorCritic` shape. That is forced by the batched rollout rather
than chosen: a step evaluates both players' networks and selects the acting one
with `jnp.where` (`training.sequential_rollout._validate_players_match`), and a
mixture is played by *gathering* one member's parameters per episode, which
needs the members stacked into one pytree. The practical consequence is that
`network.num_components` in the config applies to everything in the run --
including NFSP's average net, which the one-shot version gives its own capacity.

**Why the best response is unregularized**, and why it is never warm-started:
exactly as in `br_oracle`, whose docstring has the measurements. `br_hyperparams`
is reused verbatim for both.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.sequential import TERMINAL, SequentialZeroSumGame
from training.best_response import policy_action_fn
from training.config import MixturePPOHyperparams
from training.mixture import MixtureActorCritic, build_mixture_network
from training.ppo import ppo_update
from training.sequential_rollout import build_episode_sampler, collect_sequential_batch
from training.trainer_common import (
    append_chunk_records,
    build_loss_fn,
    create_mixture_train_state,
    reject_batch_norm,
    update_target_and_magnet,
)

from .br_oracle import br_hyperparams  # noqa: F401  (re-exported: the sequential runs use it unchanged)

# Architecture fields that have to agree for two parameter sets to be stackable
# -- i.e. for one to be substitutable for the other inside a compiled rollout.
ARCHITECTURE_FIELDS = (
    "action_dim", "hidden_dims", "activation", "normalization", "num_components",
    "num_atoms", "full_covariance", "scale_parameterization", "clip_means",
)


def _architecture(hyperparams: MixturePPOHyperparams) -> tuple:
    return tuple(getattr(hyperparams, name) for name in ARCHITECTURE_FIELDS)


def _stack(params_list: Sequence) -> chex.ArrayTree:
    """The members' parameters as one pytree with a leading member axis."""
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *params_list)


def _gather(stacked: chex.ArrayTree, index: chex.Array) -> chex.ArrayTree:
    """Row `index` of a stacked pytree -- `index` may itself be a batch of rows."""
    return jax.tree_util.tree_map(lambda leaf: leaf[index], stacked)


@dataclasses.dataclass(frozen=True)
class PolicyMixture:
    """A mixed strategy over policies: `params[k]` played with probability `weights[k]`.

    Every member shares `network`/`hyperparams` (see the module docstring), which
    is what lets `stacked` exist and hence what lets a batch of episodes each
    play a different member. A single policy is the one-member case and is what
    most callers pass around -- `single` builds it.
    """

    network: MixtureActorCritic
    hyperparams: MixturePPOHyperparams
    params: tuple
    weights: np.ndarray
    label: str

    def __post_init__(self) -> None:
        if len(self.params) != len(self.weights):
            raise ValueError(f"{len(self.params)} members but {len(self.weights)} weights")
        if len(self.params) == 0:
            raise ValueError("a mixture needs at least one member")
        total = float(np.sum(self.weights))
        if not total > 0:
            raise ValueError(f"weights must sum to something positive, got {total}")

    def __len__(self) -> int:
        return len(self.params)

    @property
    def probs(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.float64) / float(np.sum(self.weights))

    @property
    def stacked(self) -> chex.ArrayTree:
        return _stack(self.params)

    @property
    def log_probs(self) -> chex.Array:
        return jnp.log(jnp.clip(jnp.asarray(self.probs), 1e-12, None))

    def draw(self, key: chex.PRNGKey, num_episodes: int) -> chex.Array:
        """One member index per episode -- the draw that makes this a *mixed* strategy."""
        return jax.random.categorical(key, self.log_probs, shape=(num_episodes,))

    def per_episode_params(self, key: chex.PRNGKey, num_episodes: int) -> chex.ArrayTree:
        """`num_episodes` parameter sets, drawn from the mixture; leading axis `num_episodes`."""
        return _gather(self.stacked, self.draw(key, num_episodes))

    def member(self, index: int) -> "PolicyMixture":
        """Member `index` on its own, as a one-member mixture."""
        return dataclasses.replace(
            self, params=(self.params[index],), weights=np.ones(1),
            label=f"{self.label}[{index}]",
        )

    def with_weights(self, weights) -> "PolicyMixture":
        return dataclasses.replace(self, weights=np.asarray(weights, dtype=np.float64))


def single(network: MixtureActorCritic, hyperparams: MixturePPOHyperparams, params,
           label: str = "policy") -> PolicyMixture:
    """One policy, as a (degenerate) mixture -- the form every consumer here takes."""
    return PolicyMixture(network, hyperparams, (params,), np.ones(1), label)


def population(hyperparams: MixturePPOHyperparams, params_list: Sequence, weights,
               network: MixtureActorCritic | None = None,
               label: str = "population") -> PolicyMixture:
    """A weighted population of policies -- PSRO's meta-strategy."""
    network = build_mixture_network(hyperparams) if network is None else network
    return PolicyMixture(network, hyperparams, tuple(params_list),
                         np.asarray(weights, dtype=np.float64), f"{label}[{len(params_list)}]")


def mix(mixtures: Sequence[PolicyMixture], probs: Sequence[float]) -> PolicyMixture:
    """Play `mixtures[i]` with probability `probs[i]`: NFSP's anticipatory mixture.

    Flattened into one member list rather than kept as a mixture of mixtures --
    drawing a member of a member is the same draw, and one flat list is what the
    rollout gathers from.
    """
    if len(mixtures) != len(probs):
        raise ValueError(f"{len(mixtures)} mixtures but {len(probs)} probabilities")
    reference = mixtures[0]
    for other in mixtures[1:]:
        if _architecture(other.hyperparams) != _architecture(reference.hyperparams):
            raise ValueError(
                "every member of a mixture must share one architecture (a batched rollout "
                f"gathers one member's parameters per episode): {other.label} differs from "
                f"{reference.label}. See baselines/neural/sequential_oracle.py's docstring."
            )
    params, weights = [], []
    for mixture, prob in zip(mixtures, probs):
        params.extend(mixture.params)
        weights.extend(prob * mixture.probs)
    label = " + ".join(f"{p:.2f}*{m.label}" for m, p in zip(mixtures, probs))
    return PolicyMixture(reference.network, reference.hyperparams, tuple(params),
                         np.asarray(weights, dtype=np.float64), label)


def initial_policy(game: SequentialZeroSumGame, player: int,
                   hyperparams: MixturePPOHyperparams, seed: int = 0) -> PolicyMixture:
    """An untrained network: the seed strategy both algorithms need before round 1.

    PSRO's population and NFSP's average both have to start *somewhere*, and a
    freshly initialized policy is the tree's analogue of the one-shot baselines'
    uniform random play -- not uniform (the categorical head starts near-uniform
    over legal kinds, but the bet-size mixture starts spread across the box by
    `_spread_bias_init`, not flat on it), which is why it is named for what it is.
    """
    network = build_mixture_network(hyperparams)
    init_key, state_key = jax.random.split(jax.random.PRNGKey(seed))
    dummy_state = game.initial_state(state_key)
    params = network.init(init_key, game.observation(player, dummy_state))
    return single(network, hyperparams, params, f"initial(player {player})")


# --------------------------------------------------------------------------- evaluation


class MixtureEvaluator:
    """Payoffs between two `PolicyMixture`s, by playing them.

    Holds the compiled rollout across calls the way
    `training.best_response.PairEvaluator` does, and for the same reason -- the
    parameters are arguments, not constants, so moving weights does not throw the
    executable away. One program per *pair of member counts*, since those are
    shapes: PSRO's growing population would recompile every round, which is why
    its payoff matrix uses `PairEvaluator` on single policies instead and this
    class is for scoring a meta-strategy as a whole.
    """

    def __init__(self, game: SequentialZeroSumGame,
                 networks: tuple[MixtureActorCritic, MixtureActorCritic]):
        self.game = game
        self.networks = networks
        self._spaces = (game.action_space(0), game.action_space(1))
        self._compiled: dict[tuple[int, int], Callable] = {}

    def _build(self, sizes: tuple[int, int]) -> Callable:
        def batch_sums(stacked_0, log_probs_0, stacked_1, log_probs_1, keys):
            def one(key):
                key_0, key_1, play_key = jax.random.split(key, 3)
                index = (jax.random.categorical(key_0, log_probs_0),
                         jax.random.categorical(key_1, log_probs_1))
                params = (_gather(stacked_0, index[0]), _gather(stacked_1, index[1]))
                action_fns = tuple(
                    policy_action_fn(self.networks[p], params[p], self._spaces[p])
                    for p in (0, 1)
                )
                return self.game.play_episode(action_fns, play_key)[1]

            payoff = jax.vmap(one)(keys)
            return jnp.sum(payoff), jnp.sum(payoff ** 2)

        del sizes  # only used as the cache key; shapes come in with the arguments
        return jax.jit(batch_sums)

    def evaluate(self, mixture_0: PolicyMixture, mixture_1: PolicyMixture, key: chex.PRNGKey,
                 num_episodes: int = 20_000, batch_size: int = 20_000) -> tuple[float, float]:
        """`(mean payoff to player 0, standard error)` over `num_episodes` hands."""
        if num_episodes < 2:
            raise ValueError(f"num_episodes must be at least 2, got {num_episodes}")
        sizes = (len(mixture_0), len(mixture_1))
        if sizes not in self._compiled:
            self._compiled[sizes] = self._build(sizes)
        batch_sums = self._compiled[sizes]

        stacked = (mixture_0.stacked, mixture_1.stacked)
        log_probs = (mixture_0.log_probs, mixture_1.log_probs)
        sums, remaining = [], num_episodes
        while remaining > 0:
            key, batch_key = jax.random.split(key)
            size = min(batch_size, remaining)
            sums.append(batch_sums(stacked[0], log_probs[0], stacked[1], log_probs[1],
                                   jax.random.split(batch_key, size)))
            remaining -= size

        total, total_squares = (float(v) for v in jax.tree_util.tree_map(
            lambda *batch: sum(batch), *jax.device_get(sums)))
        mean = total / num_episodes
        variance = max(total_squares / num_episodes - mean ** 2, 0.0)
        return mean, (variance / num_episodes) ** 0.5


# --------------------------------------------------------------------------- the oracle


@dataclasses.dataclass
class SequentialBestResponse:
    """A trained best response: the policy, and what it is worth against its opponent."""

    network: MixtureActorCritic
    params: chex.ArrayTree
    hyperparams: MixturePPOHyperparams
    history: list[dict]
    br_value: float             # realized payoff to the responder over the last training chunk
    opponent_label: str

    def as_mixture(self, label: str = "br") -> PolicyMixture:
        return single(self.network, self.hyperparams, self.params, label)


def _build_train_step(
    game: SequentialZeroSumGame,
    networks: tuple[MixtureActorCritic, MixtureActorCritic],
    hyperparams: MixturePPOHyperparams,
    responder: int,
    opponent: PolicyMixture,
):
    """One PPO iteration against a mixture opponent: draw members, roll out, update.

    The draw is *inside* the scanned step, so every iteration sees a fresh sample
    of the opponent's mixture rather than one fixed assignment reused for the
    whole chunk -- with a peaked meta-strategy the latter would train several
    hundred iterations against whichever members happened to come up.
    """
    sample_episode = build_episode_sampler(game, networks[0], networks[1])
    loss_fn = build_loss_fn(responder, hyperparams)
    sign = 1.0 if responder == 0 else -1.0
    stacked, log_probs = opponent.stacked, opponent.log_probs
    num_envs = hyperparams.num_envs
    # The responder's parameters are shared by the batch; the opponent's are one
    # per episode. A player's magnet entry rides on the same axis as its params:
    # the frozen side's magnet is simply its params (its magnet outputs are
    # masked out of the responder's loss, so nothing they hold reaches a gradient).
    param_axes = (None, None, 0, 0) if responder == 0 else (0, 0, None, None)

    def step(state, key: chex.PRNGKey):
        select_key, rollout_key = jax.random.split(key)
        index = jax.random.categorical(select_key, log_probs, shape=(num_envs,))
        frozen = _gather(stacked, index)
        live = (state.params, state.magnet_params)
        (params_0, magnet_0), (params_1, magnet_1) = (
            (live, (frozen, frozen)) if responder == 0 else ((frozen, frozen), live)
        )
        batch, payoff = collect_sequential_batch(
            sample_episode, params_0, magnet_0, params_1, magnet_1, rollout_key, num_envs,
            param_axes=param_axes,
        )
        state, metrics = ppo_update(state, networks[responder], batch, hyperparams, loss_fn=loss_fn)
        state = update_target_and_magnet(state, hyperparams)
        return state, {
            "br_value": sign * jnp.mean(payoff),
            "episode_length": jnp.mean(jnp.sum((batch.actor != TERMINAL).astype(jnp.float32), axis=-1)),
            **metrics,
        }

    return step


class SequentialMixtureBestResponseTrainer:
    """PPO against a fixed `PolicyMixture`: trains `responder`, leaves the mixture alone.

    `training.best_response.SequentialBestResponseTrainer` is the same trainer
    against a *single* frozen policy. This one exists because both PSRO and NFSP
    best-respond to a mixture -- a meta-strategy, an anticipatory mixture -- and
    a mixture is not a policy: it cannot be collapsed into one network, and
    responding to any one member is a different (easier) problem.
    """

    def __init__(self, game: SequentialZeroSumGame, responder: int, opponent: PolicyMixture,
                 hyperparams: MixturePPOHyperparams, seed: int = 0):
        if responder not in (0, 1):
            raise ValueError(f"responder must be 0 or 1, got {responder}")
        reject_batch_norm("hyperparams", hyperparams)
        if _architecture(hyperparams) != _architecture(opponent.hyperparams):
            raise ValueError(
                "the responder and the opponent mixture must share one architecture (a batched "
                "rollout evaluates both players' heads and selects between them): "
                f"{_architecture(hyperparams)} vs {_architecture(opponent.hyperparams)}"
            )

        self.game = game
        self.responder = responder
        self.opponent = opponent
        self.hyperparams = hyperparams
        self.network = build_mixture_network(hyperparams)
        self.networks = (
            (self.network, opponent.network) if responder == 0 else (opponent.network, self.network)
        )

        key = jax.random.PRNGKey(seed)
        init_key, state_key, self.key = jax.random.split(key, 3)
        dummy_state = game.initial_state(state_key)
        params = self.network.init(init_key, game.observation(responder, dummy_state))
        self.state = create_mixture_train_state(self.network, params, hyperparams)

        train_step = _build_train_step(game, self.networks, hyperparams, responder, opponent)
        self._run_chunk = jax.jit(lambda state, keys: jax.lax.scan(train_step, state, keys))
        self.history: list[dict] = []

    @property
    def params(self):
        return self.state.params

    def policy(self, target: bool = False, label: str = "br") -> PolicyMixture:
        params = self.state.target_params if target else self.state.params
        return single(self.network, self.hyperparams, params, label)

    def train(self, steps: int, epochs: int = 10, verbose: bool = False) -> list[dict]:
        """`steps` chunks of `epochs` scanned PPO iterations each.

        The chunk loop rather than `training.trainer_common.run_training_chunks`
        only because this one runs as an *inner* loop, hundreds of times per run,
        and must be able to keep quiet.
        """
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        for chunk in range(steps):
            self.key, chunk_key = jax.random.split(self.key)
            self.state, metrics = self._run_chunk(self.state, jax.random.split(chunk_key, epochs))
            record = append_chunk_records(self.history, metrics, chunk, epochs)
            if verbose:
                print(f"    br iter {record['iteration']:5d} | br_value {record['br_value']:+.4f} "
                      f"| len {record['episode_length']:.2f}")
        return self.history


def train_sequential_best_response(
    game: SequentialZeroSumGame,
    player: int,
    opponent: PolicyMixture,
    hyperparams: MixturePPOHyperparams,
    steps: int = 50,
    epochs: int = 20,
    seed: int = 0,
    verbose: bool = False,
) -> SequentialBestResponse:
    """Train `player`'s policy against `opponent` with the repo's own sequential PPO.

    Cold-started every time, deliberately -- see `baselines.neural.nfsp`'s
    docstring for the measurement behind that; the argument is about the
    oracle's exploration and does not change in a tree.
    """
    trainer = SequentialMixtureBestResponseTrainer(game, player, opponent, hyperparams, seed=seed)
    history = trainer.train(steps, epochs, verbose=verbose)
    return SequentialBestResponse(
        network=trainer.network,
        params=trainer.state.params,
        hyperparams=hyperparams,
        history=history,
        br_value=float(history[-1]["br_value"]) if history else float("nan"),
        opponent_label=opponent.label,
    )


# --------------------------------------------------------------------------- SL data


def sample_decision_rows(
    game: SequentialZeroSumGame,
    player: int,
    own: PolicyMixture,
    opponent: PolicyMixture,
    key: chex.PRNGKey,
    num_episodes: int,
    own_member: int | None = None,
) -> dict[str, np.ndarray]:
    """`player`'s decisions over `num_episodes` hands, as flat rows -- NFSP's SL data.

    Returns `obs`, `action_mask` (the *expanded* per-logit mask the policy
    sampled under), `action_kind` and `raw_action`, one row per decision `player`
    actually made: padding steps and the opponent's steps are dropped, so the
    rows are already the empirical distribution of "infosets this player reached,
    and what it did there".

    `own_member` keeps only the episodes in which `own` drew that member. That is
    what makes this NFSP's memory rather than something looser: the *trajectory*
    distribution must come from both players' anticipatory mixtures (which is why
    `own` is a mixture at all), while the *actions stored* must be the best
    response's alone.
    """
    sample_episode = build_episode_sampler(
        game, own.network if player == 0 else opponent.network,
        opponent.network if player == 0 else own.network,
    )
    own_key, opponent_key, rollout_key = jax.random.split(key, 3)
    own_index = own.draw(own_key, num_episodes)
    own_params = _gather(own.stacked, own_index)
    opponent_params = opponent.per_episode_params(opponent_key, num_episodes)
    params_0, params_1 = (
        (own_params, opponent_params) if player == 0 else (opponent_params, own_params)
    )
    batch, _ = collect_sequential_batch(
        sample_episode, params_0, params_0, params_1, params_1, rollout_key, num_episodes,
        param_axes=(0, 0, 0, 0),
    )

    keep = np.asarray(batch.actor) == player                       # (episodes, max_steps)
    if own_member is not None:
        keep &= (np.asarray(own_index) == own_member)[:, None]
    return {
        "obs": np.asarray(batch.obs)[keep],
        "action_mask": np.asarray(batch.action_mask)[keep],
        "action_kind": np.asarray(batch.action_kind)[keep],
        "raw_action": np.asarray(batch.raw_action)[keep],
    }
