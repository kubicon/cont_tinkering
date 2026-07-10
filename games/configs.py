"""Per-game YAML config dataclasses: each mirrors one `ZeroSumGame` subclass's
constructor, so a run config only needs to list arguments for the chosen game.

`GAME_CONFIGS` maps a `game.name` string (as it appears in the YAML config) to
its dataclass; see `training.run_config.load_run_config` for how it's used.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from .base import ZeroSumGame
from .examples import (
    ContinuousBlottoGame,
    ContinuousMatchingPennies,
    MultiPointGame,
    QuadraticZeroSumGame,
)


@dataclasses.dataclass
class MatchingPenniesConfig:
    def build(self) -> ZeroSumGame:
        return ContinuousMatchingPennies()


@dataclasses.dataclass
class MultiPointConfig:
    peaks: tuple[float, ...] = (0.0, 1.0, 2.0)
    weights: tuple[float, ...] | None = None
    width: float = 0.1
    coupling: float = 1.0

    def build(self) -> ZeroSumGame:
        return MultiPointGame(
            peaks=tuple(self.peaks),
            weights=tuple(self.weights) if self.weights is not None else None,
            width=self.width,
            coupling=self.coupling,
        )


@dataclasses.dataclass
class QuadraticConfig:
    dim: int = 2
    coupling: float = 0.1
    bound: float = 3.0

    def build(self) -> ZeroSumGame:
        coupling = self.coupling * jnp.eye(self.dim)
        return QuadraticZeroSumGame(coupling=coupling, bound=self.bound)


@dataclasses.dataclass
class BlottoConfig:
    fronts: int = 3
    budget: float = 1.0
    sharpness: float = 10.0

    def build(self) -> ZeroSumGame:
        front_values = jnp.ones(self.fronts)
        return ContinuousBlottoGame(front_values=front_values, budget=self.budget, sharpness=self.sharpness)


GAME_CONFIGS: dict[str, type] = {
    "matching_pennies": MatchingPenniesConfig,
    "multi_point": MultiPointConfig,
    "quadratic": QuadraticConfig,
    "blotto": BlottoConfig,
}
