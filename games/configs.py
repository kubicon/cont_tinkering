"""Per-game YAML config dataclasses: each mirrors one game subclass's constructor,
so a run config only needs to list arguments for the chosen game.

`GAME_CONFIGS` maps a `game.name` string (as it appears in the YAML config) to
its dataclass; see `training.run_config.load_run_config` for how it's used.

Both kinds of game live in this one registry -- the one-shot `ZeroSumGame`s and
the sequential `SequentialZeroSumGame`s -- and `train.py` picks the matching
trainer from the type of whatever `build()` returns. A config file therefore
looks the same either way; only `game.name` decides which machinery runs.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from .base import ZeroSumGame
from .examples import (
    AsymmetricWellGame,
    ContinuousBlottoGame,
    ContinuousMatchingPennies,
    ContinuousMatchingPenniesShifted,
    CurvaturePumpGame,
    DecoyWellGame,
    ForsakenGame,
    MultiDimDecoyWellGame,
    MultiPointGame,
    QuadraticZeroSumGame,
)
from .sequential import SequentialZeroSumGame
from .sequential_examples import ContinuousKuhnPoker


@dataclasses.dataclass
class MatchingPenniesConfig:
    def build(self) -> ZeroSumGame:
        return ContinuousMatchingPennies()


@dataclasses.dataclass
class MatchingPenniesShiftedConfig:
    def build(self) -> ZeroSumGame:
        return ContinuousMatchingPenniesShifted()


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


@dataclasses.dataclass
class AsymmetricWellConfig:
    dim: int = 1
    coupling: float = 1.0
    bound: float = 3.0

    def build(self) -> ZeroSumGame:
        return AsymmetricWellGame(dim=self.dim, coupling=self.coupling, bound=self.bound)


@dataclasses.dataclass
class CurvaturePumpConfig:
    dim: int = 1
    pump: float = 4.0
    bound: float = 2.0

    def build(self) -> ZeroSumGame:
        return CurvaturePumpGame(dim=self.dim, pump=self.pump, bound=self.bound)


@dataclasses.dataclass
class ForsakenConfig:
    bound: float = 1.5

    def build(self) -> ZeroSumGame:
        return ForsakenGame(bound=self.bound)


@dataclasses.dataclass
class DecoyWellConfig:
    peaks: tuple[float, ...] = (-1.0, 1.0)
    weights: tuple[float, ...] | None = None
    peak_width: float = 0.05
    peak_height: float = 1.0
    # Each decoy is `(center, height, width)`; heights must be < `peak_height`.
    decoys: tuple[tuple[float, float, float], ...] = ((0.0, 0.7, 0.45),)
    coupling: float = 1.0
    action_margin: float = 2.0

    def build(self) -> ZeroSumGame:
        return DecoyWellGame(
            peaks=tuple(self.peaks),
            weights=tuple(self.weights) if self.weights is not None else None,
            peak_width=self.peak_width,
            peak_height=self.peak_height,
            decoys=tuple(tuple(d) for d in self.decoys),
            coupling=self.coupling,
            action_margin=self.action_margin,
        )


@dataclasses.dataclass
class MultiDimDecoyWellConfig:
    dim: int = 2
    peaks: tuple[float, ...] = (-1.0, 1.0)
    weights: tuple[float, ...] | None = None
    peak_width: float = 0.05
    peak_height: float = 1.0
    # Each decoy is `(center, height, width)`; heights must be < `peak_height`.
    decoys: tuple[tuple[float, float, float], ...] = ((0.0, 0.7, 0.45),)
    coupling: float = 1.0
    action_margin: float = 2.0

    def build(self) -> ZeroSumGame:
        return MultiDimDecoyWellGame(
            dim=self.dim,
            peaks=tuple(self.peaks),
            weights=tuple(self.weights) if self.weights is not None else None,
            peak_width=self.peak_width,
            peak_height=self.peak_height,
            decoys=tuple(tuple(d) for d in self.decoys),
            coupling=self.coupling,
            action_margin=self.action_margin,
        )


@dataclasses.dataclass
class KuhnConfig:
    """Continuous-bet Kuhn poker -- a *sequential* game, unlike everything above it.

    `min_bet == max_bet` fixes the size and recovers textbook Kuhn, whose
    equilibria and value (`-1/18`) are known in closed form; that is the setting
    to validate a run against before trusting the continuous one.
    """

    num_cards: int = 3
    min_bet: float = 0.5
    max_bet: float = 2.0
    # Bet sizes the exact best response may choose between when measuring
    # exploitability (see `games.kuhn_best_response`). Finer is a tighter lower
    # bound; the cost is one batched forward pass either way.
    exploitability_grid_points: int = 1025

    def build(self) -> SequentialZeroSumGame:
        return ContinuousKuhnPoker(
            num_cards=self.num_cards, min_bet=self.min_bet, max_bet=self.max_bet
        )


GAME_CONFIGS: dict[str, type] = {
    "matching_pennies": MatchingPenniesConfig,
    "matching_pennies_shifted": MatchingPenniesShiftedConfig,
    "multi_point": MultiPointConfig,
    "quadratic": QuadraticConfig,
    "blotto": BlottoConfig,
    "asymmetric_well": AsymmetricWellConfig,
    "curvature_pump": CurvaturePumpConfig,
    "forsaken": ForsakenConfig,
    "decoy_well": DecoyWellConfig,
    "multidim_decoy_well": MultiDimDecoyWellConfig,
    "kuhn": KuhnConfig,
}
