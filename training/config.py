"""Hyperparameters for a PPO run.

Kept as a single flat, JSON-serializable dataclass: the same object is
logged, checkpointed, and used to rebuild the `ActorCritic` architecture on
load (see `training/checkpoint.py`), so nothing about the network is ever
implicit.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PPOHyperparams:
    # Architecture (enough to rebuild `ActorCritic` from scratch; obs_dim
    # comes from the game itself -- see `ZeroSumGame.obs_dim`).
    action_dim: int
    hidden_dims: tuple[int, ...]
    activation: str = "tanh"
    normalization: str = "none"

    # Optimization.
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    optimizer: str = "adam"  # one of `training.optimizers.OPTIMIZERS`
    weight_decay: float = 0.0  # only optimizers in `training.optimizers.WEIGHT_DECAY_OPTIMIZERS` support this

    # PPO.
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0

    # Rollout / update shape.
    num_envs: int = 256
    num_epochs: int = 4

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PPOHyperparams":
        data = dict(data)
        data["hidden_dims"] = tuple(data["hidden_dims"])
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class MixturePPOHyperparams(PPOHyperparams):
    """`PPOHyperparams` plus what's needed to rebuild a `MixtureActorCritic`.

    `low`/`high` are the action space bounds (per action dimension), used
    only to spread the components' initial means across the space; see
    `training.mixture._spread_bias_init`.
    """

    num_components: int = 2
    # Parameterless discrete actions in front of the Gaussian components (fold,
    # check, call). 0 for the one-shot games in `games.examples`; a sequential
    # game's own `HybridSpace` dictates it, so `train.py` reads it off the
    # action space rather than taking it from the YAML.
    num_atoms: int = 0
    low: tuple[float, ...] = (0.0,)
    high: tuple[float, ...] = (1.0,)
    clip_means: bool = False  # constrain the mean head to `[low, high]`; see `MixtureActorCritic`
    # Weight on the squared distance of the *raw* mean head from `[low, high]`
    # (see `training.mixture.mean_box_excess`). Inert unless `clip_means`, whose
    # straight-through projection is what needs the restoring force. In squared
    # action units, so its natural scale follows the width of the action box.
    mean_box_penalty_coef: float = 0.0

    # Target/magnet parameter tracking (see `training.mixture_trainer.MixtureTrainState`).
    target_tau: float = 0.005  # Polyak-averaging coefficient for `target_params`.
    magnet_interval: int = 1000  # `L`: iterations between `magnet_params` snapshots.

    # Entropy bonus and KL regularization terms (see
    # `training.mixture.mixture_ppo_loss`), split per head since the
    # categorical and Gaussian factors can need very different weights.
    # `entropy_coef` (inherited from `PPOHyperparams`) is *not* used by the
    # mixture loss -- these two replace it. All default to 0 (disabled).
    category_entropy_coef: float = 0.0
    gaussian_entropy_coef: float = 0.0
    trpo_category_kl_coef: float = 0.0  # KL(old || current), categorical head.
    trpo_gaussian_kl_coef: float = 0.0  # KL(old || current), Gaussian head (old = params at update start).
    magnet_category_kl_coef: float = 0.0  # KL(current || magnet_params), categorical head.
    magnet_gaussian_kl_coef: float = 0.0  # KL(current || magnet_params), Gaussian head.

    @classmethod
    def from_dict(cls, data: dict) -> "MixturePPOHyperparams":
        data = dict(data)
        data["hidden_dims"] = tuple(data["hidden_dims"])
        data["low"] = tuple(data["low"])
        data["high"] = tuple(data["high"])
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class ExpFamilyPPOHyperparams(PPOHyperparams):
    """`PPOHyperparams` plus what's needed to rebuild an `ExpFamilyActorCritic`.

    The basis is *fixed* (only `theta` is learned), so every field describing it
    has to be stored for a checkpoint to be reloadable: a policy is meaningless
    without the features its natural parameters are natural for.

    `low`/`high` are the action box. Unlike the mixture's, they are not merely an
    initialization detail -- they are the support of the density itself, which is
    why there is no `clip_means`/`mean_box_penalty_coef` counterpart here.
    """

    low: tuple[float, ...] = (0.0,)
    high: tuple[float, ...] = (1.0,)

    # The basis (see `training.expfam.build_basis`).
    grid_points: int = 256      # bins the density is piecewise-constant on
    num_basis: int = 8          # RBF bumps per action dimension
    poly_order: int = 2         # monomials z^1..z^poly_order in the normalized coordinate
    basis_width_scale: float = 1.0  # RBF width, in units of the RBF spacing
    init_tilt: float = 0.0      # initial exponential tilt exp(t*z); 0 starts uniform

    # Target/magnet parameter tracking (see `training.trainer_common`).
    target_tau: float = 0.005
    magnet_interval: int = 1000

    # Entropy bonus and KL regularization. One density means one of each, rather
    # than the mixture's per-head pair; `entropy_coef` (inherited) is unused.
    density_entropy_coef: float = 0.0
    trpo_density_kl_coef: float = 0.0    # KL(old || current)
    magnet_density_kl_coef: float = 0.0  # KL(current || magnet_params)

    @classmethod
    def from_dict(cls, data: dict) -> "ExpFamilyPPOHyperparams":
        data = dict(data)
        data["hidden_dims"] = tuple(data["hidden_dims"])
        data["low"] = tuple(data["low"])
        data["high"] = tuple(data["high"])
        return cls(**data)
