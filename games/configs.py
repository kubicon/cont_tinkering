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
from .disk_sumo import DiskSumo
from .disk_sumo_v2 import DiskSumoV2
from .disk_sumo_v3 import DiskSumoV3
from .mjx_sumo import MjxAntSumo, MjxBugSumo, MjxSpiderSumo, MjxSumo
from .examples import (
    AllPayAuctionGame,
    AsymmetricWellGame,
    CircleGame,
    ContinuousBlottoGame,
    ContinuousMatchingPennies,
    ContinuousMatchingPenniesShifted,
    CoupledRotationGame,
    CurvaturePumpGame,
    DecoyWellGame,
    ForsakenGame,
    GlicksbergGrossGame,
    SilentDuelGame,
    MultiDimDecoyWellGame,
    MultiPointGame,
    QuadraticAsymmetricGame,
    QuadraticZeroSumGame,
)
from .leduc import ContinuousLeducHoldem
from .sequential import SequentialZeroSumGame
from .sequential_blotto import ContinuousSequentialBlotto
from .sequential_examples import ContinuousKuhnPoker


@dataclasses.dataclass
class MatchingPenniesConfig:
    dim: int = 1

    def build(self) -> ZeroSumGame:
        return ContinuousMatchingPennies(dim=self.dim)


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
class QuadraticAsymmetricConfig:
    dim: int = 1

    def build(self) -> ZeroSumGame:
        return QuadraticAsymmetricGame(dim=self.dim)


@dataclasses.dataclass
class CoupledRotationConfig:
    dim: int = 2
    coupling: float = 20.0
    warp: float = 0.3
    damping: float = 0.25
    bound: float = 1.5

    def build(self) -> ZeroSumGame:
        return CoupledRotationGame(
            dim=self.dim,
            coupling=self.coupling,
            warp=self.warp,
            damping=self.damping,
            bound=self.bound,
        )


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
    # Measure exploitability with every Gaussian's sigma at zero -- each component
    # bets its (clipped) mean -- while check/bet, the component weights and
    # fold/call stay distributions (see `training.kuhn_evaluation`).
    exploitability_greedy_gaussians: bool = False

    def build(self) -> SequentialZeroSumGame:
        return ContinuousKuhnPoker(
            num_cards=self.num_cards, min_bet=self.min_bet, max_bet=self.max_bet
        )


@dataclasses.dataclass
class LeducConfig:
    """Continuous-raise Leduc Hold'em -- the second *sequential* game here.

    `num_ranks=3, num_suits=2, min_bet=max_bet=2, second_round_scale=2.0,
    max_raises=2` is the textbook benchmark game (fixed bets of 2 and 4), the
    setting to validate a run against before reading anything into a
    continuous-raise one. There is no exact best response for the continuous
    version -- the public state contains real bet sizes -- so unlike `kuhn` this
    trains without an exploitability metric.
    """

    num_ranks: int = 3
    num_suits: int = 2
    min_bet: float = 0.5
    max_bet: float = 2.0
    max_raises: int = 2
    # Multiplies every raise made after the board card is turned; 2.0 reproduces
    # the classic game's doubled second-round bet size.
    second_round_scale: float = 1.0

    def build(self) -> SequentialZeroSumGame:
        return ContinuousLeducHoldem(
            num_ranks=self.num_ranks,
            num_suits=self.num_suits,
            min_bet=self.min_bet,
            max_bet=self.max_bet,
            max_raises=self.max_raises,
            second_round_scale=self.second_round_scale,
        )


@dataclasses.dataclass
class SequentialBlottoConfig:
    """Colonel Blotto contested one front at a time -- the third *sequential* game.

    `sharpness: null` in the YAML means the bigger bid simply wins; any positive
    number softens the front into a logistic contest, which is what to reach for
    when a run stalls (see `games.sequential_blotto`). Like `leduc` there is no
    exact best response, so this trains without an exploitability metric --
    measure it afterwards with `best_response.py`.
    """

    num_fields: int = 3
    # One value per front; `null` weights every front equally.
    field_values: tuple[float, ...] | None = None
    budget: float = 1.0
    # Positive: a logistic contest of this steepness. `null`: hard argmax.
    sharpness: float | None = 10.0

    def build(self) -> SequentialZeroSumGame:
        return ContinuousSequentialBlotto(
            num_fields=self.num_fields,
            field_values=tuple(self.field_values) if self.field_values is not None else None,
            budget=self.budget,
            sharpness=self.sharpness,
        )


@dataclasses.dataclass
class DiskSumoConfig:
    """Two disks pushing each other out of a ring -- a sequential game with physics.

    Each control step is two decisions (see `games.disk_sumo`), so a bout is up
    to `2 * horizon` decisions long, all sharing one terminal payoff. There is no
    exact best response; measure checkpoints with `best_response.py`.
    """

    horizon: int = 50
    ring_radius: float = 1.0
    disk_radius: float = 0.15
    start_distance: float = 0.6
    start_jitter: float = 0.05
    random_orientation: bool = True
    # Place both disks uniformly in the ring every bout, instead of facing each
    # other across the centre; `start_distance`, `start_jitter` and
    # `random_orientation` are then unused.
    random_start: bool = False
    egocentric: bool = True
    dt: float = 0.1
    substeps: int = 10
    mass: float = 1.0
    max_force: float = 1.0
    # Per bout, one randomly selected disk gets multiplier 1 + a and the other
    # 1 - a. Zero keeps the original symmetric game and 11-D observation.
    force_asymmetry: float = 0.0
    # Optional resource that makes repeated full thrust progressively weaker.
    # Stamina drains/recharges per simulated second; at zero stamina the disk
    # retains `stamina_min_force` of its otherwise available thrust.
    stamina_enabled: bool = False
    stamina_drain_rate: float = 0.5
    stamina_recovery_rate: float = 0.25
    stamina_min_force: float = 0.25
    drag: float = 1.0
    stiffness: float = 100.0
    contact_damping: float = 5.0
    # Timeout tie-breaker weight on the distance-from-centre gap; 0 is pure sumo.
    margin_weight: float = 0.0
    # Dense potential-based shaping on the same distance gap; 0 is terminal-only.
    shaping_weight: float = 0.0

    def build(self) -> SequentialZeroSumGame:
        return DiskSumo(**dataclasses.asdict(self))


@dataclasses.dataclass
class DiskSumoV2Config:
    """`disk_sumo` with a random archetype per seat, optionally hidden from the opponent.

    See `games.disk_sumo_v2`. Stamina is always on. `archetypes` selects the
    enabled subset (null: all of `DEFAULT_ARCHETYPES`); `archetype_traits`
    overrides individual traits, e.g. `{quick: {force: 0.9}}`, or defines a new
    archetype.
    """

    horizon: int = 100
    ring_radius: float = 1.0
    disk_radius: float = 0.15
    start_distance: float = 0.45
    start_jitter: float = 0.04
    random_orientation: bool = True
    random_start: bool = False
    egocentric: bool = True
    dt: float = 0.1
    substeps: int = 10
    mass: float = 1.0
    max_force: float = 1.0
    stamina_drain_rate: float = 0.5
    stamina_recovery_rate: float = 0.25
    stamina_min_force: float = 0.25
    drag: float = 0.5
    stiffness: float = 400.0
    contact_damping: float = 10.0
    margin_weight: float = 0.0
    shaping_weight: float = 0.0
    archetypes: list[str] | None = None
    archetype_traits: dict[str, dict[str, float]] | None = None
    # False turns the opponent's archetype one-hot into zeros.
    observe_opponent_archetype: bool = True
    # Null follows `observe_opponent_archetype`: stamina leaks the drain rate.
    observe_opponent_stamina: bool | None = None
    # Append both disks' velocity change over the last control step, the
    # single-frame evidence a memoryless policy has about a hidden archetype.
    observe_velocity_change: bool = False

    def build(self) -> SequentialZeroSumGame:
        return DiskSumoV2(**dataclasses.asdict(self))


@dataclasses.dataclass
class DiskSumoV3Config(DiskSumoV2Config):
    """`disk_sumo_v2` with three archetypes that counter each other in a cycle.

    See `games.disk_sumo_v3`: rammer beats grinder, grinder beats anchor, anchor
    beats rammer. `archetype_traits` also accepts `top_speed`, `impact_brace`
    and `brace_speed`.
    """

    # Width of the grip cut-out around `top_speed`, as a fraction of it.
    grip_width: float = 0.05

    def build(self) -> SequentialZeroSumGame:
        return DiskSumoV3(**dataclasses.asdict(self))


@dataclasses.dataclass
class MjxSumoConfig:
    """`disk_sumo` on MuJoCo/MJX physics instead of the hand-rolled integrator.

    Same game, same 11-dimensional observation, same `[-1, 1]^2` egocentric
    force, so a `disk_sumo` config transfers by changing `name` and swapping the
    contact parameters (`stiffness` / `contact_damping` become the `solref`
    pair). See `games/mjx_sumo.py`. Like `disk_sumo` it has no exact best
    response; measure checkpoints with `best_response.py`.
    """

    horizon: int = 50
    ring_radius: float = 1.0
    disk_radius: float = 0.15
    start_distance: float = 0.6
    start_jitter: float = 0.05
    random_orientation: bool = True
    egocentric: bool = True
    dt: float = 0.1
    substeps: int = 10
    mass: float = 1.0
    max_force: float = 1.0
    drag: float = 1.0
    # MuJoCo contact softness, as (timeconst, dampratio); smaller is harder.
    solref_timeconst: float = 0.02
    solref_dampratio: float = 1.0
    # One contact converges in far fewer than MuJoCo's default 100 iterations.
    solver_iterations: int = 4
    solver_ls_iterations: int = 8
    # Timeout tie-breaker weight on the distance-from-centre gap; 0 is pure sumo.
    margin_weight: float = 0.0
    # Dense potential-based shaping on the same distance gap; 0 is terminal-only.
    shaping_weight: float = 0.0
    # "auto" selects Warp on an NVIDIA JAX backend and JAX elsewhere.
    physics_backend: str = "jax"
    # Warp contact capacity is shared by all vmapped worlds; size this for the
    # largest rollout/evaluation batch. njmax is per world.
    warp_naconmax: int | None = None
    warp_njmax: int | None = None
    # "auto" uses Warp's device default; staged modes trade copies for stable
    # pointers and fewer expensive CUDA graph recaptures.
    warp_graph_mode: str = "auto"

    def build(self) -> SequentialZeroSumGame:
        return MjxSumo(**dataclasses.asdict(self))


@dataclasses.dataclass
class MjxAntSumoConfig:
    """Two MuJoCo ants pushing each other out of a ring -- RoboSumo, in JAX.

    The articulated counterpart of `mjx_sumo`: the action is eight joint
    torques rather than a force, the observation is 65-dimensional, and an
    agent loses by leaving the ring *or* by ending up on its back
    (`knockdown_height`). See `games/mjx_sumo.py`. No exact best response;
    measure checkpoints with `best_response.py`.
    """

    horizon: int = 80
    ring_radius: float = 3.0
    start_distance: float = 2.4
    start_jitter: float = 0.1
    random_orientation: bool = True
    dt: float = 0.05
    substeps: int = 5
    gear: float = 150.0
    # The height an ant stands at, and the one below which it counts as downed.
    start_height: float = 0.55
    knockdown_height: float = 0.3
    # Pure input normalization: a walking ant has no closed-form top speed.
    velocity_scale: float = 5.0
    angular_scale: float = 10.0
    joint_velocity_scale: float = 10.0
    solver_iterations: int = 4
    solver_ls_iterations: int = 8
    # Timeout tie-breaker weight on the distance-from-centre gap; 0 is pure sumo.
    margin_weight: float = 0.0
    # Dense potential-based shaping on the same distance gap; 0 is terminal-only.
    shaping_weight: float = 0.0
    physics_backend: str = "jax"
    warp_naconmax: int | None = None
    warp_njmax: int | None = None
    warp_graph_mode: str = "auto"

    def build(self) -> SequentialZeroSumGame:
        return MjxAntSumo(**dataclasses.asdict(self))


@dataclasses.dataclass
class MjxBugSumoConfig(MjxAntSumoConfig):
    """`mjx_ant_sumo` with six legs instead of four: 12 torques, 81 observations.

    Harder to tip and slower to turn than the ant, and the most expensive of the
    three to simulate.
    """

    def build(self) -> SequentialZeroSumGame:
        return MjxBugSumo(**dataclasses.asdict(self))


@dataclasses.dataclass
class MjxSpiderSumoConfig(MjxAntSumoConfig):
    """`mjx_ant_sumo` with three legs instead of four: 6 torques, 57 observations.

    A tripod has no stability margin, so `knockdown_height` decides far more
    bouts here than it does for the ant. The cheapest of the three.
    """

    def build(self) -> SequentialZeroSumGame:
        return MjxSpiderSumo(**dataclasses.asdict(self))


@dataclasses.dataclass
class AllPayAuctionConfig:
    value: float = 0.5
    high: float = 1.0
    sharpness: float | None = None   # null keeps the hard rule, and the exact equilibrium

    def build(self) -> ZeroSumGame:
        return AllPayAuctionGame(value=self.value, high=self.high, sharpness=self.sharpness)


@dataclasses.dataclass
class CircleConfig:
    harmonics: int = 3
    coefficients: tuple[float, ...] | None = None   # null -> 2^-k, decaying

    def build(self) -> ZeroSumGame:
        return CircleGame(
            harmonics=self.harmonics,
            coefficients=tuple(self.coefficients) if self.coefficients is not None else None,
        )


@dataclasses.dataclass
class GlicksbergGrossConfig:
    """No parameters: the game, its equilibrium and its value are all fixed."""

    def build(self) -> ZeroSumGame:
        return GlicksbergGrossGame()


@dataclasses.dataclass
class SilentDuelConfig:
    exponent: float = 1.0
    sharpness: float | None = None   # null keeps the hard rule, and the exact equilibrium

    def build(self) -> ZeroSumGame:
        return SilentDuelGame(exponent=self.exponent, sharpness=self.sharpness)


GAME_CONFIGS: dict[str, type] = {
    "matching_pennies": MatchingPenniesConfig,
    "matching_pennies_shifted": MatchingPenniesShiftedConfig,
    "multi_point": MultiPointConfig,
    "quadratic": QuadraticConfig,
    "quadratic_asymmetric": QuadraticAsymmetricConfig,
    "coupled_rotation": CoupledRotationConfig,
    "blotto": BlottoConfig,
    "asymmetric_well": AsymmetricWellConfig,
    "curvature_pump": CurvaturePumpConfig,
    "forsaken": ForsakenConfig,
    "decoy_well": DecoyWellConfig,
    "all_pay_auction": AllPayAuctionConfig,
    "circle": CircleConfig,
    "glicksberg_gross": GlicksbergGrossConfig,
    "silent_duel": SilentDuelConfig,
    "multidim_decoy_well": MultiDimDecoyWellConfig,
    "kuhn": KuhnConfig,
    "leduc": LeducConfig,
    "sequential_blotto": SequentialBlottoConfig,
    "disk_sumo": DiskSumoConfig,
    "disk_sumo_v2": DiskSumoV2Config,
    "disk_sumo_v3": DiskSumoV3Config,
    "mjx_sumo": MjxSumoConfig,
    "mjx_ant_sumo": MjxAntSumoConfig,
    "mjx_bug_sumo": MjxBugSumoConfig,
    "mjx_spider_sumo": MjxSpiderSumoConfig,
}
