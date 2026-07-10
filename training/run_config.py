"""Hierarchical YAML run config: the single argument `train.py` takes.

A config file has up to five top-level sections -- `game`, `network`,
`optimizer`, `ppo`, `train` -- each optional (defaults apply if omitted).
`game.name` selects one of `games.configs.GAME_CONFIGS`; only that game's own
fields are needed there, not every game's arguments. See `configs/*.yaml` for
worked examples, one per game.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from games.configs import GAME_CONFIGS


@dataclasses.dataclass
class NetworkConfig:
    hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "gelu"
    normalization: str = "rms_norm"
    num_components: int = 2


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    max_grad_norm: float = 0.5
    optimizer: str = "adam"
    weight_decay: float = 0.0  # only supported by "adamw" and "muon"; see `training.optimizers.WEIGHT_DECAY_OPTIMIZERS`


@dataclasses.dataclass
class PPOConfig:
    clip_eps: float = 0.1
    value_coef: float = 0.5
    batch_size: int = 256
    ppo_epochs: int = 2

    # Target/magnet parameter tracking.
    target_tau: float = 0.001
    magnet_interval: int = 500

    # Entropy bonus, split per head.
    category_entropy_coef: float = 0.1
    gaussian_entropy_coef: float = 0.1

    # KL regularization, split per head (0 disables a term).
    trpo_category_kl_coef: float = 0.05
    trpo_gaussian_kl_coef: float = 0.05
    magnet_category_kl_coef: float = 0.2
    magnet_gaussian_kl_coef: float = 0.2


@dataclasses.dataclass
class TrainConfig:
    mode: str = "self_play"  # "self_play" or "fixed_opponent"
    perspective: int = 0  # fixed_opponent: which player trains
    opponent: str = "random"  # fixed_opponent: opponent policy ("random" or "static")

    # Each of `steps` outer steps runs `epochs` training iterations inside a
    # `lax.scan`, then logs and checkpoints once.
    steps: int = 200
    epochs: int = 200
    checkpoint_dir: str | None = "data/test"
    seed: int = 0


@dataclasses.dataclass
class RunConfig:
    game: Any  # one of `games.configs.GAME_CONFIGS`'s dataclasses
    network: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)


def _build_dataclass(cls: type, data: dict) -> Any:
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown field(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def run_config_from_dict(raw: dict) -> RunConfig:
    """Builds a `RunConfig` from an already-parsed config dict (e.g. `yaml.safe_load`'s
    output, or one produced by `sweep.py` for a single sweep combination).
    """
    unknown_sections = set(raw) - {"game", "network", "optimizer", "ppo", "train"}
    if unknown_sections:
        raise ValueError(f"unknown top-level config section(s): {sorted(unknown_sections)}")

    game_raw = dict(raw.get("game", {}))
    game_name = game_raw.pop("name", None)
    if game_name is None:
        raise ValueError("config.game.name is required")
    if game_name not in GAME_CONFIGS:
        raise ValueError(f"unknown game {game_name!r}, choices: {sorted(GAME_CONFIGS)}")
    game_config = _build_dataclass(GAME_CONFIGS[game_name], game_raw)

    return RunConfig(
        game=game_config,
        network=_build_dataclass(NetworkConfig, raw.get("network", {}) or {}),
        optimizer=_build_dataclass(OptimizerConfig, raw.get("optimizer", {}) or {}),
        ppo=_build_dataclass(PPOConfig, raw.get("ppo", {}) or {}),
        train=_build_dataclass(TrainConfig, raw.get("train", {}) or {}),
    )


def load_run_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return run_config_from_dict(raw)
