"""Checkpointing: hyperparameters as JSON, params as msgpack bytes.

The architecture itself is never stored — the hyperparams object carries
enough (`hidden_dims`, `activation`, `normalization`, and whatever else a
given actor-critic needs) to rebuild the exact network used at training
time. Observation shape is *not* a hyperparameter — it comes from the game
(`ZeroSumGame.obs_dim`), so the caller passes a `dummy_obs` (typically
`game.observation(player, key)`) to re-init a correctly-shaped target
pytree, which the saved bytes are then deserialized into. A checkpoint is
just two small, human-inspectable files rather than a pickled model object.

Defaults to `ActorCritic`/`PPOHyperparams`; pass `build_network` and
`hyperparams_cls` to checkpoint a different actor-critic (e.g.
`MixtureActorCritic`/`MixturePPOHyperparams` in `training/mixture.py`).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable, Optional

import chex
import flax.serialization
import jax

from .actor_critic import ActorCritic
from .config import PPOHyperparams

HYPERPARAMS_FILENAME = "hyperparams.json"
PARAMS_FILENAME = "params.msgpack"


def _default_build_network(hyperparams: PPOHyperparams) -> ActorCritic:
    return ActorCritic(
        action_dim=hyperparams.action_dim,
        hidden_dims=hyperparams.hidden_dims,
        activation=hyperparams.activation,
        normalization=hyperparams.normalization,
    )


def save_checkpoint(path: str | Path, hyperparams: PPOHyperparams, params) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / HYPERPARAMS_FILENAME).write_text(json.dumps(hyperparams.to_dict(), indent=2))
    (path / PARAMS_FILENAME).write_bytes(flax.serialization.to_bytes(params))


def load_checkpoint(
    path: str | Path,
    dummy_obs: chex.Array,
    build_network: Optional[Callable[[PPOHyperparams], object]] = None,
    hyperparams_cls: type = PPOHyperparams,
) -> tuple[PPOHyperparams, dict]:
    path = Path(path)
    hyperparams = hyperparams_cls.from_dict(json.loads((path / HYPERPARAMS_FILENAME).read_text()))

    network = (build_network or _default_build_network)(hyperparams)
    target = network.init(jax.random.PRNGKey(0), dummy_obs)
    params = flax.serialization.from_bytes(target, (path / PARAMS_FILENAME).read_bytes())
    return hyperparams, params


def save_checkpoint_step(directory: str | Path, step: int, hyperparams: PPOHyperparams, params) -> None:
    """Single-file checkpoint `{step}.pkl` (hyperparams + params) inside `directory`.

    Used for periodic training checkpoints, one file per step (step 0 is the
    freshly initialized, pre-training params).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{step}.pkl").open("wb") as f:
        pickle.dump({"hyperparams": hyperparams.to_dict(), "params": params}, f)


def load_checkpoint_step(
    directory: str | Path, step: int, hyperparams_cls: type = PPOHyperparams
) -> tuple[PPOHyperparams, dict]:
    directory = Path(directory)
    with (directory / f"{step}.pkl").open("rb") as f:
        data = pickle.load(f)
    return hyperparams_cls.from_dict(data["hyperparams"]), data["params"]


def save_checkpoint_step_multi(
    directory: str | Path, step: int, entries: dict[str, tuple[PPOHyperparams, object]]
) -> None:
    """Like `save_checkpoint_step`, but bundles several named (hyperparams, params)
    pairs -- e.g. both players of a self-play run -- into one `{step}.pkl`.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = {name: {"hyperparams": hp.to_dict(), "params": params} for name, (hp, params) in entries.items()}
    with (directory / f"{step}.pkl").open("wb") as f:
        pickle.dump(data, f)


def load_checkpoint_step_multi(
    directory: str | Path, step: int, hyperparams_cls: type = PPOHyperparams
) -> dict[str, tuple[PPOHyperparams, object]]:
    directory = Path(directory)
    with (directory / f"{step}.pkl").open("rb") as f:
        data = pickle.load(f)
    return {name: (hyperparams_cls.from_dict(v["hyperparams"]), v["params"]) for name, v in data.items()}
