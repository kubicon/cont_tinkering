"""Hierarchical YAML run config: the single argument `train.py` takes.

A config file has up to eleven top-level sections -- `game`, `network`,
`optimizer`, `ppo`, `train`, `best_response`, `discrete`, `nfsp`, `psro`, `rpn`,
`scoring` -- each optional (defaults apply if omitted). The last five are read by
`train_sequential.py`: `train.solver` picks which algorithm plays the game tree,
`discrete`/`nfsp`/`psro`/`rpn` carry that algorithm's own settings (they need *vastly*
different budgets, which is the whole reason they are separate sections rather
than shared fields), and `scoring` says how the result is measured. Every entry
point ignores the sections it does not read, so one file can be handed to any of
them. `best_response` is read only by `best_response.py`, which otherwise
runs off exactly the same schema as `train.py`: a best response is trained by
the same PPO on the same networks, so it would be a mistake for it to have its
own idea of what a network or an optimizer is. A twelfth section, `idealized`, is
accepted and ignored here: it carries the solver-only knobs `run_idealized.py`
needs, so the *same* file runs under every entry point.
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


POLICIES = ("gaussian_mixture", "exp_family")

# `train.py` copies its run config into `train.checkpoint_dir` under this name,
# and `best_response.py` reads the game back from it.
CONFIG_FILENAME = "config.yaml"


@dataclasses.dataclass
class NetworkConfig:
    # Which policy parametrization the one-shot trainers build.
    #   "gaussian_mixture" -- `training.mixture.MixtureActorCritic` (the default;
    #       a categorical over `num_components` Gaussians).
    #   "exp_family"       -- `training.expfam.ExpFamilyActorCritic` (a log-linear
    #       density over a fixed basis). Its entropy and KL terms are exact
    #       rather than estimated; see that module's docstring. One-shot games
    #       only -- a sequential run rejects it.
    # The two read disjoint subsets of the fields below.
    policy: str = "gaussian_mixture"

    hidden_dims: tuple[int, ...] = (64, 64)
    activation: str = "gelu"
    normalization: str = "rms_norm"
    num_components: int = 2
    # Give each Gaussian component a full covariance (a lower-triangular
    # Cholesky factor) instead of a diagonal one. Only meaningful for a
    # multi-dimensional action, and only useful when the payoff actually couples
    # a player's own coordinates: a separable payoff's expectation depends on
    # the per-axis marginals alone, so the off-diagonal entries get exactly zero
    # payoff gradient. See `training.gaussian`.
    full_covariance: bool = False
    # "log" (the default here) or "linear" -- how the scale head's output is read
    # into the Cholesky factor. See `training.config.PPOHyperparams` for the
    # tradeoff; "log" is positive by construction and gives the spread and the
    # correlations separate gradients, at the cost of the KL's uniform strong
    # convexity.
    scale_parameterization: str = "log"
    # `scale_parameterization: "log"` + `full_covariance` only; bounds the
    # condition number of `Sigma`. 0 leaves the off-diagonal unbounded.
    max_correlation: float = 0.0
    # Bounds on each Gaussian component's standard deviation, given as plain
    # sigma in action units under either `scale_parameterization` ("log" clips
    # to their logs). A floor keeps a component from collapsing onto a point;
    # a ceiling keeps it from spreading past the box. `null` keeps the built-in
    # bounds: `training.gaussian.SIGMA_MIN` (1e-3) below, the box width above.
    # Best responses built from this config inherit the same bounds.
    sigma_min: float | None = None
    sigma_max: float | None = None
    # Cut the action box into `num_components` equal buckets (a regular grid,
    # so a perfect d-th power of components in d dims) and keep each component's
    # mean inside its own, starting at the center. Shifting bet mass across the
    # box is then a component-weight change, and every region keeps a component
    # -- and, with `ppo.explore_eps`, bucket-local exploration -- of its own.
    # Pair with a `sigma_min` of about a quarter bucket width so each component
    # keeps covering its bucket. See `training.mixture.MixtureActorCritic`.
    bucket_means: bool = False
    clip_means: bool = False  # constrain the mean head to the action box; see `MixtureActorCritic`
    # Pulls a mean that drifts out of the box back to its edge; only bites with
    # `clip_means` on. See `training.mixture.mean_box_excess`.
    mean_box_penalty_coef: float = 1.0

    # `policy: exp_family` only -- the fixed basis; see `training.expfam.build_basis`.
    grid_points: int = 256          # bins the piecewise-constant density lives on
    num_basis: int = 8              # RBF bumps per action dimension
    poly_order: int = 2             # monomials z^1..z^poly_order, z the box-normalized action
    basis_width_scale: float = 1.0  # RBF width, in units of the RBF spacing
    # Initial exponential tilt `p ~ exp(t*z)`. 0 starts the policy uniform on the
    # box, which in a bilinear game is already a Nash -- set it nonzero to start
    # off-equilibrium, as `idealized.init_means` does for the Gaussian runs.
    init_tilt: float = 0.0


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

    # Off-policy exploration, `train_sequential.py`'s `self_play` only
    # (`discrete_mmd` never samples through the mixture sampler, so it is inert
    # there). With probability `explore_eps` the action is a uniform draw
    # instead: the check/bet (fold/call) choice uniformly over the legal kinds,
    # and a bet's component uniformly with its size uniform on the action box
    # (on that component's bucket under `network.bucket_means`). Each player's loss
    # importance-weights its own exploratory actions back to its policy, while
    # the opponent's exploration is left in: that is what teaches a player how
    # to answer sizes the opponent's policy never plays. Evaluation and
    # checkpoints are the policy alone. Best responses never explore. 0 disables.
    explore_eps: float = 0.0

    # Advantage estimator. "monte_carlo" regresses on the episode's return and
    # never bootstraps; "vtrace" (sequential games only) bootstraps along each
    # player's own decisions with V-trace, importance-weighting its own
    # (possibly exploring) actions back to the current policy -- the opponent's
    # moves, exploration included, are treated as environment. `gamma` discounts
    # per own decision (1 keeps the zero-sum objective); `vtrace_lambda` < 1 is
    # what reduces the variance; `vtrace_rho_bar` / `vtrace_c_bar` truncate the
    # importance weights in the TD errors and in the traces.
    advantage: str = "monte_carlo"
    gamma: float = 1.0
    vtrace_lambda: float = 0.95
    vtrace_rho_bar: float = 2.0
    vtrace_c_bar: float = 1.0
    # What V-trace does with the opponent's exploration (`training.mixture.
    # build_mixture_ppo_loss_fn`): "none" leaves it as environment; "future"
    # importance-weights the opponent's moves after each own decision; and
    # "future_and_past" also weights each decision by the opponent's moves
    # before it, floored at `vtrace_opponent_past_floor` so states only
    # exploration reaches keep being trained on. V-trace only.
    vtrace_opponent_correction: str = "none"
    vtrace_opponent_past_floor: float = 0.05
    # "cumulative" floors the product itself; "range" smooths each opponent row's
    # continuous value towards uniform instead, so exploration-only states keep
    # the opponent's policy over kinds of action (see `training.vtrace.
    # range_smoothed_log_rhos`).
    vtrace_opponent_past_floor_mode: str = "cumulative"

    # Entropy bonus, split per head.
    category_entropy_coef: float = 0.1
    gaussian_entropy_coef: float = 0.1

    # KL regularization, split per head (0 disables a term).
    trpo_category_kl_coef: float = 0.05
    trpo_gaussian_kl_coef: float = 0.05
    magnet_category_kl_coef: float = 0.2
    magnet_gaussian_kl_coef: float = 0.2

    # Categorical head update, `policy: gaussian_mixture` only. "ppo" is the
    # clipped surrogate. "neurd" is Neural Replicator Dynamics: each logit moves
    # by its entry's advantage with no `pi` factor, so an action the policy has
    # all but abandoned recovers as fast as any other; the centered logits stay
    # within [-neurd_beta, neurd_beta], and each sampled entry's advantage over
    # its behavior probability is clipped to `neurd_clip`. Under "neurd" the
    # categorical entropy/magnet coefficients act inside the advantage, in payoff
    # units, rather than as loss terms.
    category_update: str = "ppo"
    neurd_beta: float = 2.0
    neurd_clip: float = 10.0
    # Standardize the policy-gradient advantage over each player's decisions in
    # the batch. False feeds the surrogate the raw advantage against the critic,
    # in payoff units, so the entropy/KL coefficients weigh against payoffs.
    normalize_advantage: bool = True
    # Probability floor on the categorical head, under either update: the loss
    # gains `category_floor_coef * sum_k relu(log category_floor - log pi_k)`,
    # zero above the floor and a push back up that does not shrink with `pi_k`
    # below it. "kind" floors check/bet and fold/call (a bet's components
    # summed), "entry" every component too. 0 disables.
    category_floor: float = 0.0
    category_floor_coef: float = 0.01
    category_floor_mode: str = "kind"

    # `policy: exp_family` only. A log-linear density has one head, so it has one
    # of each coefficient rather than the mixture's per-head pair. `null` (the
    # default) reuses the corresponding `*_gaussian_*` value above, so an
    # existing config runs under either policy without being rewritten.
    density_entropy_coef: float | None = None
    trpo_density_kl_coef: float | None = None
    magnet_density_kl_coef: float | None = None


SOLVERS = ("self_play", "nfsp", "psro", "discrete_mmd", "rpn")


@dataclasses.dataclass
class TrainConfig:
    mode: str = "self_play"  # "self_play" or "fixed_opponent"

    # `train_sequential.py` only: which algorithm solves the tree.
    #   "self_play" -- `SequentialSelfPlayPPOTrainer`, i.e. the method under test
    #       (both players' Gaussian-mixture policies learning simultaneously).
    #       Uses `steps`/`epochs` below.
    #   "nfsp"      -- `baselines.neural.sequential_nfsp`, scheduled by `nfsp:`.
    #   "psro"      -- `baselines.neural.sequential_psro`, scheduled by `psro:`.
    #   "discrete_mmd" -- the *same* self-play run on a game whose continuous
    #       action has been split into `discrete.bins` evenly spaced actions
    #       (`games.discretized`), so the policy uses its categorical head alone.
    #       The discretization baseline; `steps`/`epochs` schedule it too.
    #   "rpn"       -- randomized policy networks trained by zeroth-order
    #       pseudo-gradient (Martin & Sandholm), scheduled by `rpn:`. A different
    #       policy class entirely: implicit, with no density, so it reads
    #       `network.hidden_dims`/`activation` and nothing else from `network:`.
    # `train.py` ignores this field: it always runs the self-play trainer.
    solver: str = "self_play"
    perspective: int = 0  # fixed_opponent: which player trains
    opponent: str = "random"  # fixed_opponent: opponent policy ("random" or "static")

    # Each of `steps` outer steps runs `epochs` training iterations inside a
    # `lax.scan`, then logs and checkpoints once.
    steps: int = 200
    epochs: int = 200
    checkpoint_dir: str | None = "data/test"
    seed: int = 0


@dataclasses.dataclass
class BestResponseConfig:
    """`best_response.py` only: whose strategy to respond to, and how to measure it.

    """

    # The `SequentialSelfPlayPPOTrainer` run to evaluate, and which of its
    # checkpoints; `null` takes the last one written.
    checkpoint_dir: str = "data/leduc"
    checkpoint_step: int | None = None

    # 0 or 1 to respond to that player's *opponent*, or "both" to train one of
    # each and report their sum as an exploitability lower bound.
    responder: str | int = "both"

    # Which of the checkpoint's two strategies to measure: "live" (the last
    # iterate), "target" (the Polyak average), or "both".
    opponent_iterate: str = "live"
    responder_iterate: str = "live"

    # The final measurement. Sampling error scales as 1/sqrt(episodes) and the
    # per-hand variance in a poker-like game is several antes, so a number
    # quoted to three decimals needs a lot of hands.
    eval_episodes: int = 200_000
    eval_batch_size: int = 20_000
    eval_seed: int = 12345

    # Episodes per chunk for the progress line, which exists to show the curve
    # flattening. Cheap and noisy on purpose; the headline uses the fields above.
    progress_episodes: int = 20_000


@dataclasses.dataclass
class DiscreteConfig:
    """`train.solver: discrete_mmd` only -- see `games/discretized.py`."""

    # How many actions the continuous branch is split into, evenly spaced across
    # the action box with both endpoints included. *The* hyperparameter of the
    # discretization baseline: too few and no strategy on the grid is close to an
    # equilibrium, too many and the categorical head is estimating a distribution
    # over hundreds of actions from the same number of hands.
    bins: int = 16
    # Refuse a grid wider than this many actions. With a `d`-dimensional action
    # the grid is `bins**d`, and that blow-up is the cost this baseline exists to
    # show rather than something to discover as an allocation failure.
    max_actions: int = 1024


@dataclasses.dataclass
class NFSPConfig:
    """`train.solver: nfsp` only -- see `baselines/neural/sequential_nfsp.py`."""

    rounds: int = 30
    # One best response per player per round: `br_steps` chunks of `br_epochs`
    # scanned PPO iterations. Everything NFSP claims rests on this being an
    # actual best response, so it is the budget to raise first.
    br_steps: int = 50
    br_epochs: int = 20
    eta: float = 0.1  # anticipatory parameter: how often a player plays `beta` rather than `pi`
    reservoir_capacity: int = 200_000
    # Hands played per player per round to fill the supervised memory. Only the
    # ~`eta` fraction in which the learner drew `beta` contribute rows.
    reservoir_episodes: int = 4096
    sl_steps: int = 400  # supervised gradient steps per round
    sl_batch: int = 256


@dataclasses.dataclass
class PSROConfig:
    """`train.solver: psro` only -- see `baselines/neural/sequential_psro.py`."""

    rounds: int = 12  # best responses added per player
    br_steps: int = 50
    br_epochs: int = 20
    payoff_episodes: int = 20_000  # hands per population pair for the empirical payoff matrix
    meta_solver: str = "nash"  # "nash" (the LP) or "uniform" (the self-play-ish ablation)


@dataclasses.dataclass
class RPNConfig:
    """`train.solver: rpn` only -- see `baselines/neural/sequential_rpn.py`."""

    iterations: int = 2_000
    log_every: int = 50          # iterations per log point (and per checkpoint)

    # The policy. Width and activation come from `network:`; these are the parts
    # that block has no field for.
    noise_dim: int = 8           # width of the `z` that makes the policy mixed

    # The estimator. `separate` is the IJCAI'23 per-player pseudo-gradient (the
    # "randomized policy networks" paper), `joint` the IJCAI'25 JPSPG that reads
    # both players' gradients off one perturbation -- half the evaluations.
    estimator: str = "separate"
    sigma: float = 0.1           # smoothing radius of the central difference
    # Hands per utility evaluation, and perturbations averaged per iteration.
    # Their *product* (doubled, for `separate`) is what one iteration costs in
    # episodes -- which is why both are far below the papers' one-shot settings.
    # `perturbation_batch` is nonetheless the knob that decides whether the
    # estimate carries signal at all: a single random direction in R^d is nearly
    # orthogonal to the gradient it estimates.
    utility_episodes: int = 64
    perturbation_batch: int = 64
    antithetic: bool = True
    dynamics: str = "simultaneous"   # simultaneous | extragradient | optimistic

    # Optimizer. The papers' pairing: AdaBelief at 1e-4 with an averaged estimate.
    learning_rate: float = 1e-4
    optimizer: str = "adabelief"
    max_grad_norm: float = 0.0   # 0 disables clipping

    # Measurement. The policy has no density, so its strategy is *sampled*: this
    # many noise draws per infoset when reading it out for the exact Kuhn metric.
    strategy_samples: int = 128
    # Iterations of zeroth-order ascent behind each `expl_lb` bound.
    br_iterations: int = 500


@dataclasses.dataclass
class ScoringConfig:
    """How a sequential run is measured, for all three solvers alike.

    Kuhn has an exact tree best response and uses it every log point for free.
    Leduc and sequential Blotto have none, so the only available number is a
    *trained* best response's value, which costs as much as a round of the
    algorithm -- hence `score_every: 0` by default. See
    `baselines/neural/sequential_scoring.py`.
    """

    exact_grid: int | None = None  # bet-grid points for the exact Kuhn metric; null = the game's own
    score_every: int = 0  # train an RL best-response bound every N log points (0 = never)
    br_steps: int = 50
    br_epochs: int = 20
    episodes: int = 20_000  # hands per measurement (the head-to-head value, and each bound)
    # Also score the Polyak-averaged iterate where the metric is free (Kuhn), under
    # `target_` columns. Only `self_play` has one; NFSP and PSRO ignore it.
    include_target: bool = True


@dataclasses.dataclass
class RunConfig:
    game: Any  # one of `games.configs.GAME_CONFIGS`'s dataclasses
    network: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    best_response: BestResponseConfig = dataclasses.field(default_factory=BestResponseConfig)
    discrete: DiscreteConfig = dataclasses.field(default_factory=DiscreteConfig)
    nfsp: NFSPConfig = dataclasses.field(default_factory=NFSPConfig)
    psro: PSROConfig = dataclasses.field(default_factory=PSROConfig)
    rpn: RPNConfig = dataclasses.field(default_factory=RPNConfig)
    scoring: ScoringConfig = dataclasses.field(default_factory=ScoringConfig)


def _build_dataclass(cls: type, data: dict) -> Any:
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown field(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


ADVANTAGES = ("monte_carlo", "vtrace")
OPPONENT_CORRECTIONS = ("none", "future", "future_and_past")
OPPONENT_PAST_FLOOR_MODES = ("cumulative", "range")


def _check_advantage(ppo: PPOConfig) -> PPOConfig:
    """Coerce the V-trace constants to floats and reject impossible ones."""
    if ppo.advantage not in ADVANTAGES:
        raise ValueError(f"unknown ppo.advantage {ppo.advantage!r}, choices: {list(ADVANTAGES)}")
    values = {name: float(getattr(ppo, name))
              for name in ("gamma", "vtrace_lambda", "vtrace_rho_bar", "vtrace_c_bar")}
    if not 0.0 < values["gamma"] <= 1.0:
        raise ValueError(f"ppo.gamma must lie in (0, 1], got {values['gamma']}")
    if not 0.0 <= values["vtrace_lambda"] <= 1.0:
        raise ValueError(f"ppo.vtrace_lambda must lie in [0, 1], got {values['vtrace_lambda']}")
    for name in ("vtrace_rho_bar", "vtrace_c_bar"):
        if values[name] <= 0.0:
            raise ValueError(f"ppo.{name} must be positive, got {values[name]}")
    if ppo.vtrace_opponent_correction not in OPPONENT_CORRECTIONS:
        raise ValueError(
            f"unknown ppo.vtrace_opponent_correction {ppo.vtrace_opponent_correction!r}, "
            f"choices: {list(OPPONENT_CORRECTIONS)}"
        )
    if ppo.vtrace_opponent_correction != "none" and ppo.advantage != "vtrace":
        raise ValueError("ppo.vtrace_opponent_correction needs ppo.advantage: vtrace")
    values["vtrace_opponent_past_floor"] = float(ppo.vtrace_opponent_past_floor)
    if not 0.0 < values["vtrace_opponent_past_floor"] <= 1.0:
        raise ValueError(
            f"ppo.vtrace_opponent_past_floor must lie in (0, 1], got {values['vtrace_opponent_past_floor']}"
        )
    if ppo.vtrace_opponent_past_floor_mode not in OPPONENT_PAST_FLOOR_MODES:
        raise ValueError(
            f"unknown ppo.vtrace_opponent_past_floor_mode {ppo.vtrace_opponent_past_floor_mode!r}, "
            f"choices: {list(OPPONENT_PAST_FLOOR_MODES)}"
        )
    return dataclasses.replace(ppo, **values)


CATEGORY_UPDATES = ("ppo", "neurd")
CATEGORY_FLOOR_MODES = ("kind", "entry")


def _check_category_update(ppo: PPOConfig, network: NetworkConfig) -> PPOConfig:
    """Coerce the categorical-head constants to floats and reject impossible combinations."""
    if ppo.category_update not in CATEGORY_UPDATES:
        raise ValueError(
            f"unknown ppo.category_update {ppo.category_update!r}, choices: {list(CATEGORY_UPDATES)}"
        )
    if ppo.category_update != "ppo" and network.policy != "gaussian_mixture":
        raise ValueError(
            f"ppo.category_update: {ppo.category_update} needs network.policy: gaussian_mixture, "
            f"got {network.policy!r}"
        )
    values = {name: float(getattr(ppo, name)) for name in ("neurd_beta", "neurd_clip")}
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"ppo.{name} must be positive, got {value}")
    values["category_floor"] = float(ppo.category_floor)
    values["category_floor_coef"] = float(ppo.category_floor_coef)
    if not 0.0 <= values["category_floor"] < 1.0:
        raise ValueError(f"ppo.category_floor must lie in [0, 1), got {values['category_floor']}")
    if values["category_floor_coef"] < 0.0:
        raise ValueError(f"ppo.category_floor_coef must be nonnegative, got {values['category_floor_coef']}")
    if ppo.category_floor_mode not in CATEGORY_FLOOR_MODES:
        raise ValueError(
            f"unknown ppo.category_floor_mode {ppo.category_floor_mode!r}, "
            f"choices: {list(CATEGORY_FLOOR_MODES)}"
        )
    if values["category_floor"] > 0.0 and network.policy != "gaussian_mixture":
        raise ValueError(
            f"ppo.category_floor needs network.policy: gaussian_mixture, got {network.policy!r}"
        )
    return dataclasses.replace(ppo, **values)


def _check_sigma_bounds(network: NetworkConfig) -> NetworkConfig:
    """Coerce `sigma_min` / `sigma_max` to floats and reject impossible bounds.

    Coercion matters: YAML 1.1 reads an unsigned exponent like `1e-3` as a
    string, not a number.
    """
    bounds = {}
    for name in ("sigma_min", "sigma_max"):
        value = getattr(network, name)
        if value is None:
            continue
        value = float(value)
        if not value > 0.0:
            raise ValueError(f"network.{name} must be positive, got {value}")
        bounds[name] = value
    if len(bounds) == 2 and bounds["sigma_min"] > bounds["sigma_max"]:
        raise ValueError(
            f"network.sigma_min ({bounds['sigma_min']}) exceeds network.sigma_max "
            f"({bounds['sigma_max']})"
        )
    return dataclasses.replace(network, **bounds)


def run_config_from_dict(raw: dict) -> RunConfig:
    """Builds a `RunConfig` from an already-parsed config dict (e.g. `yaml.safe_load`'s
    output, or one produced by `sweep.py` for a single sweep combination).
    """
    # `idealized` holds solver-only knobs for `run_idealized.py` (grid resolution,
    # std bounds, custom init). It is accepted and ignored here so one config file
    # can drive both `train.py` and `run_idealized.py`.
    unknown_sections = set(raw) - {
        "game", "network", "optimizer", "ppo", "train", "best_response", "idealized",
        "nfsp", "psro", "scoring", "discrete", "rpn",
    }
    if unknown_sections:
        raise ValueError(f"unknown top-level config section(s): {sorted(unknown_sections)}")

    game_raw = dict(raw.get("game", {}))
    game_name = game_raw.pop("name", None)
    if game_name is None:
        raise ValueError("config.game.name is required")
    if game_name not in GAME_CONFIGS:
        raise ValueError(f"unknown game {game_name!r}, choices: {sorted(GAME_CONFIGS)}")
    game_config = _build_dataclass(GAME_CONFIGS[game_name], game_raw)

    network = _build_dataclass(NetworkConfig, raw.get("network", {}) or {})
    if network.policy not in POLICIES:
        raise ValueError(f"unknown network.policy {network.policy!r}, choices: {sorted(POLICIES)}")
    network = _check_sigma_bounds(network)
    ppo = _build_dataclass(PPOConfig, raw.get("ppo", {}) or {})
    if not 0.0 <= float(ppo.explore_eps) < 1.0:
        raise ValueError(f"ppo.explore_eps must lie in [0, 1), got {ppo.explore_eps}")
    ppo = dataclasses.replace(ppo, explore_eps=float(ppo.explore_eps))
    ppo = _check_advantage(ppo)
    ppo = _check_category_update(ppo, network)

    train = _build_dataclass(TrainConfig, raw.get("train", {}) or {})
    if train.solver not in SOLVERS:
        raise ValueError(f"unknown train.solver {train.solver!r}, choices: {sorted(SOLVERS)}")

    return RunConfig(
        game=game_config,
        network=network,
        optimizer=_build_dataclass(OptimizerConfig, raw.get("optimizer", {}) or {}),
        ppo=ppo,
        train=train,
        best_response=_build_dataclass(BestResponseConfig, raw.get("best_response", {}) or {}),
        discrete=_build_dataclass(DiscreteConfig, raw.get("discrete", {}) or {}),
        nfsp=_build_dataclass(NFSPConfig, raw.get("nfsp", {}) or {}),
        psro=_build_dataclass(PSROConfig, raw.get("psro", {}) or {}),
        rpn=_build_dataclass(RPNConfig, raw.get("rpn", {}) or {}),
        scoring=_build_dataclass(ScoringConfig, raw.get("scoring", {}) or {}),
    )


def load_run_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return run_config_from_dict(raw)
