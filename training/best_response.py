"""Training an approximate best response to a *fixed* strategy in a sequential game.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp

from games.sequential import TERMINAL, SequentialZeroSumGame
from games.spaces import MASKED_LOGIT, HybridAction, HybridSpace

from .checkpoint import load_checkpoint_step_multi, target_entry
from .config import MixturePPOHyperparams
from .mixture import (
    MixtureActorCritic,
    build_mixture_network,
    component_to_kind,
    expand_kind_mask,
    gaussian_component_index,
    sample_mixture_component,
)
from .ppo import ppo_update
from .sequential_rollout import build_episode_sampler, collect_sequential_batch
from .trainer_common import (
    build_loss_fn,
    create_mixture_train_state,
    reject_batch_norm,
    run_training_chunks,
    update_target_and_magnet,
)

# The name `SequentialSelfPlayPPOTrainer.save` files each player's entry under.
CHECKPOINT_ENTRY = "player_{}"


@dataclasses.dataclass(frozen=True)
class FrozenPolicy:
    """A strategy to respond to: an architecture, its weights, and the hyperparams behind it.
    """

    network: MixtureActorCritic
    params: Any
    hyperparams: MixturePPOHyperparams


def load_frozen_policy(
    checkpoint_dir: str | Path, step: int, player: int, target: bool = False
) -> FrozenPolicy:
    """Read one player's strategy out of a `SequentialSelfPlayPPOTrainer` checkpoint.
    """
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")
    entries = load_checkpoint_step_multi(
        checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams
    )
    name = CHECKPOINT_ENTRY.format(player)
    if target:
        name = target_entry(name)
    if name not in entries:
        detail = (
            " -- it predates target params being checkpointed, so its averaged iterate is not "
            "recoverable; re-run the training, or measure the live one"
            if target
            else ""
        )
        raise KeyError(
            f"checkpoint {Path(checkpoint_dir) / f'{step}.pkl'} has no entry {name!r}; "
            f"it holds {sorted(entries)}{detail}"
        )
    hyperparams, params = entries[name]
    return FrozenPolicy(
        network=build_mixture_network(hyperparams), params=params, hyperparams=hyperparams
    )


def latest_checkpoint_step(checkpoint_dir: str | Path) -> int:
    """The highest `{step}.pkl` in `checkpoint_dir` -- the end of a finished run."""
    directory = Path(checkpoint_dir)
    steps = [int(path.stem) for path in directory.glob("*.pkl") if path.stem.isdigit()]
    if not steps:
        raise FileNotFoundError(f"no `{{step}}.pkl` checkpoints in {directory}")
    return max(steps)


def validate_responder(
    responder: MixturePPOHyperparams, opponent: FrozenPolicy
) -> None:
    """Fail early on the shape agreement `build_episode_sampler` requires.

    """
    if responder.num_components != opponent.hyperparams.num_components:
        raise ValueError(
            f"network.num_components must match the opponent checkpoint's "
            f"({opponent.hyperparams.num_components}), got {responder.num_components}: a "
            "batched rollout selects between the two players' categorical heads, so they "
            "must be the same width"
        )
    if responder.num_atoms != opponent.hyperparams.num_atoms:
        raise ValueError(
            f"num_atoms ({responder.num_atoms}) must match the opponent checkpoint's "
            f"({opponent.hyperparams.num_atoms}); both come from the game's action space, so "
            "this usually means the checkpoint is from a different game"
        )


REGULARIZERS = (
    "trpo_category_kl_coef",
    "trpo_gaussian_kl_coef",
    "magnet_category_kl_coef",
    "magnet_gaussian_kl_coef",
)


def warn_if_regularized(hyperparams: MixturePPOHyperparams) -> list[str]:
    """Name the KL coefficients that bias a best response *downwards*, if any are set.
    """
    return [name for name in REGULARIZERS if getattr(hyperparams, name) != 0.0]


# ---- evaluating a policy pair -------------------------------------------


def policy_action_fn(
    network: MixtureActorCritic,
    params,
    space: HybridSpace,
    greedy: bool = False,
) -> Callable[[chex.Array, chex.Array, chex.PRNGKey], HybridAction]:
    """A `SequentialZeroSumGame.play_episode` action function backed by a network.
    """
    num_atoms = network.num_atoms

    def action_fn(obs: chex.Array, mask: chex.Array, key: chex.PRNGKey) -> HybridAction:
        logits, means, scale_trils, _ = network.apply(params, obs)
        component_mask = expand_kind_mask(mask, network.num_components)
        if greedy:
            component = jnp.argmax(jnp.where(component_mask, logits, MASKED_LOGIT))
            raw_action = means[gaussian_component_index(component, num_atoms)]
        else:
            component, raw_action = sample_mixture_component(
                logits, means, scale_trils, component_mask, num_atoms, key
            )
        return HybridAction(
            kind=component_to_kind(component, num_atoms), value=space.box.clip(raw_action)
        )

    return action_fn


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """A payoff estimate to one player, with the sampling error that bounds it."""

    value: float
    stderr: float
    episodes: int

    def __str__(self) -> str:
        return f"{self.value:+.4f} +- {1.96 * self.stderr:.4f}"


class PairEvaluator:
    """A compiled `play_episode` rollout for a *fixed pair of architectures*.

    Only the weights change between measurements -- the networks, the game and
    the greedy/sampled choice do not -- so the weights are passed to the
    compiled function as **arguments**. Closing over them instead bakes them
    into the executable as constants, which throws the compilation away every
    time the params move: a per-chunk progress evaluation then spends more time
    in XLA than the training chunk it is reporting on.

    Hold one of these across a run. `evaluate_pair` is the one-off wrapper.
    """

    def __init__(
        self,
        game: SequentialZeroSumGame,
        networks: tuple[MixtureActorCritic, MixtureActorCritic],
    ):
        self.game = game
        self.networks = networks
        self._spaces = (game.action_space(0), game.action_space(1))
        # One entry per `(greedy_0, greedy_1)`: those four are different
        # programs, but each is compiled once for the life of the evaluator.
        self._batch_sums: dict[tuple[bool, bool], Callable] = {}

    def _build_batch_sums(self, greedy: tuple[bool, bool]) -> Callable:
        """`(params_0, params_1, keys) -> (sum payoff, sum payoff^2)` over the batch.

        The two sums are reduced on device, so a batch returns two scalars
        rather than one payoff per episode.
        """
        def batch_sums(params_0, params_1, keys: chex.PRNGKey):
            params = (params_0, params_1)
            action_fns = tuple(
                policy_action_fn(self.networks[p], params[p], self._spaces[p], greedy[p])
                for p in (0, 1)
            )
            payoff = jax.vmap(lambda k: self.game.play_episode(action_fns, k)[1])(keys)
            return jnp.sum(payoff), jnp.sum(payoff ** 2)

        return jax.jit(batch_sums)

    def evaluate(
        self,
        params: tuple,
        player: int,
        key: chex.PRNGKey,
        num_episodes: int = 100_000,
        greedy: tuple[bool, bool] = (False, False),
        batch_size: int = 20_000,
    ) -> Evaluation:
        """Play `params` against each other and report the payoff to `player`."""
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player}")
        if isinstance(greedy, bool):
            raise TypeError("greedy is per player: pass a (bool, bool), not a single bool")
        if num_episodes < 2:
            raise ValueError(f"num_episodes must be at least 2, got {num_episodes}")

        greedy = (bool(greedy[0]), bool(greedy[1]))
        if greedy not in self._batch_sums:
            self._batch_sums[greedy] = self._build_batch_sums(greedy)
        batch_sums = self._batch_sums[greedy]

        # Dispatch every batch before reading any of them back: JAX is
        # asynchronous, so blocking on each batch's sum in turn would serialize
        # the queue against the host for no reason.
        sums = []
        remaining = num_episodes
        while remaining > 0:
            key, batch_key = jax.random.split(key)
            size = min(batch_size, remaining)
            sums.append(batch_sums(*params, jax.random.split(batch_key, size)))
            remaining -= size

        total, total_squares = (float(value) for value in jax.tree_util.tree_map(
            lambda *batch: sum(batch), *jax.device_get(sums)
        ))

        # Player 0's payoff is what the game reports; the game is zero-sum.
        sign = 1.0 if player == 0 else -1.0
        mean = total / num_episodes
        variance = max(total_squares / num_episodes - mean ** 2, 0.0)
        return Evaluation(
            value=sign * mean,
            stderr=(variance / num_episodes) ** 0.5,
            episodes=num_episodes,
        )


def evaluate_pair(
    game: SequentialZeroSumGame,
    policies: tuple[FrozenPolicy, FrozenPolicy],
    player: int,
    key: chex.PRNGKey,
    num_episodes: int = 100_000,
    greedy: tuple[bool, bool] = (False, False),
    batch_size: int = 20_000,
) -> Evaluation:
    """Play `policies` against each other and report the payoff to `player`.

    A one-off measurement: it compiles the rollout, uses it once and drops it.
    Anything measuring repeatedly (a training loop's progress metric) should
    hold a `PairEvaluator` instead and call its `evaluate`.
    """
    evaluator = PairEvaluator(game, (policies[0].network, policies[1].network))
    return evaluator.evaluate(
        (policies[0].params, policies[1].params),
        player,
        key,
        num_episodes=num_episodes,
        greedy=greedy,
        batch_size=batch_size,
    )


# ---- the trainer ---------------------------------------------------------


def _build_best_response_train_step(
    game: SequentialZeroSumGame,
    networks: tuple[MixtureActorCritic, MixtureActorCritic],
    hyperparams: MixturePPOHyperparams,
    responder: int,
    opponent_params,
):
    """The self-play step minus the opponent's update; see this module's docstring."""
    sample_episode = build_episode_sampler(game, networks[0], networks[1])
    loss_fn = build_loss_fn(responder, hyperparams)
    sign = 1.0 if responder == 0 else -1.0

    def step(state, key: chex.PRNGKey):
        # The frozen player's `magnet_params` are simply its params: the magnet
        # outputs recorded on its rows are masked out of the responder's loss,
        # so what they hold cannot reach a gradient.
        live = (state.params, state.magnet_params)
        frozen = (opponent_params, opponent_params)
        (params_0, magnet_0), (params_1, magnet_1) = (
            (live, frozen) if responder == 0 else (frozen, live)
        )
        batch, payoff = collect_sequential_batch(
            sample_episode, params_0, magnet_0, params_1, magnet_1, key, hyperparams.num_envs
        )
        state, metrics = ppo_update(state, networks[responder], batch, hyperparams, loss_fn=loss_fn)
        state = update_target_and_magnet(state, hyperparams)

        return state, {
            # The headline: the payoff *to the responder*, which is the quantity
            # a best response maximizes and the one exploitability sums.
            "br_value": sign * jnp.mean(payoff),
            "episode_length": jnp.mean(jnp.sum((batch.actor != TERMINAL).astype(jnp.float32), axis=-1)),
            **metrics,
        }

    return step


class SequentialBestResponseTrainer:
    """PPO against a frozen opponent: trains `responder`, leaves the other player alone.

    """

    def __init__(
        self,
        game: SequentialZeroSumGame,
        opponent: FrozenPolicy,
        hyperparams: MixturePPOHyperparams,
        responder: int = 0,
        seed: int = 0,
    ):
        if responder not in (0, 1):
            raise ValueError(f"responder must be 0 or 1, got {responder}")
        reject_batch_norm("hyperparams", hyperparams)
        validate_responder(hyperparams, opponent)

        self.game = game
        self.opponent = opponent
        self.responder = responder
        self.opponent_player = 1 - responder
        self.hyperparams = hyperparams

        network = build_mixture_network(hyperparams)
        self.network = network
        self.networks = (
            (network, opponent.network) if responder == 0 else (opponent.network, network)
        )
        self._evaluator = PairEvaluator(game, self.networks)

        key = jax.random.PRNGKey(seed)
        init_key, state_key, self.key = jax.random.split(key, 3)
        dummy_state = game.initial_state(state_key)
        params = network.init(init_key, game.observation(responder, dummy_state))
        self.state = create_mixture_train_state(network, params, hyperparams)

        train_step = _build_best_response_train_step(
            game, self.networks, hyperparams, responder, opponent.params
        )

        def scan_body(state, key):
            state, metrics = train_step(state, key)
            return state, metrics

        self._run_chunk = jax.jit(lambda state, keys: jax.lax.scan(scan_body, state, keys))
        self.history: list[dict] = []

    @property
    def params(self):
        return self.state.params

    def policy(self, target: bool = False) -> FrozenPolicy:
        """The responder as a `FrozenPolicy`, ready for `evaluate_pair`.
        """
        params = self.state.target_params if target else self.state.params
        return FrozenPolicy(network=self.network, params=params, hyperparams=self.hyperparams)

    def evaluate(
        self,
        key: chex.PRNGKey,
        num_episodes: int = 100_000,
        greedy: bool = False,
        target: bool = False,
        **kwargs,
    ) -> Evaluation:
        """The responder's payoff against the frozen opponent -- one best-response bound.
        """
        responder_params = self.state.target_params if target else self.state.params
        params = (
            (responder_params, self.opponent.params)
            if self.responder == 0
            else (self.opponent.params, responder_params)
        )
        # `self._evaluator`, not a fresh `evaluate_pair`: this runs once per
        # training chunk, and the rollout is only compiled the first time.
        return self._evaluator.evaluate(
            params,
            self.responder,
            key,
            num_episodes=num_episodes,
            greedy=(greedy, False) if self.responder == 0 else (False, greedy),
            **kwargs,
        )

    def train(
        self,
        steps: int,
        epochs: int = 10,
        metric_fn: Callable[["SequentialBestResponseTrainer"], dict[str, float]] | None = None,
    ) -> list[dict]:
        """Trains for `steps` chunks of `epochs` `lax.scan`-ned iterations each.
        """
        def commit(state) -> None:
            self.state = state

        def format_record(record: dict) -> str:
            return (
                f"iter {record['iteration']:5d} | br_value {record['br_value']:+.4f} "
                f"| len {record['episode_length']:.2f} "
                f"| policy {record['policy_loss']:+.4f} value {record['value_loss']:.4f} "
                f"cat_H {record['category_entropy']:.4f} atom {record['atom_frac']:.2f}"
            )

        self.key = run_training_chunks(
            steps=steps,
            epochs=epochs,
            key=self.key,
            states=self.state,
            run_chunk=self._run_chunk,
            commit=commit,
            history=self.history,
            format_record=format_record,
            metric_fn=(lambda: metric_fn(self)) if metric_fn is not None else None,
        )
        return self.history
