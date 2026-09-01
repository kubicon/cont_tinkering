"""Turning a `RunConfig` plus a game into the `MixturePPOHyperparams` a trainer needs.
"""

from __future__ import annotations

import jax.numpy as jnp

from games.base import ZeroSumGame
from games.spaces import ActionSpace, BoxSpace, HybridSpace

from .config import ExpFamilyPPOHyperparams, MixturePPOHyperparams
from .run_config import RunConfig


def action_bounds(space: ActionSpace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """`(low, high)` for spreading mixture-component means; see `MixturePPOHyperparams`.

    Exact for a `BoxSpace`, and for the `box` inside a `HybridSpace`. A
    `SimplexSpace` has no natural per-axis bounds, so it's approximated as
    `[0, total]` per dimension -- only used for initialization, the actual
    action is still projected onto the simplex.
    """
    if isinstance(space, (BoxSpace, HybridSpace)):
        return tuple(float(x) for x in space.low), tuple(float(x) for x in space.high)
    total = float(getattr(space, "total"))
    dim = space.shape[0]
    return (0.0,) * dim, (total,) * dim


def _num_atoms(space: ActionSpace) -> int:
    """How many parameterless discrete actions the policy's categorical head needs.

    Read off the action space rather than the config: the atoms are a property
    of the *game* (a `HybridSpace` says how many discrete choices its tree
    offers), not a network size to be tuned. Every purely continuous space has
    none.
    """
    return space.num_atoms if isinstance(space, HybridSpace) else 0


def build_hyperparams(game: ZeroSumGame, player: int, config: RunConfig) -> MixturePPOHyperparams:
    space = game.action_space(player)
    low, high = action_bounds(space)
    network, optimizer, ppo = config.network, config.optimizer, config.ppo
    return MixturePPOHyperparams(
        action_dim=space.shape[0],
        hidden_dims=tuple(network.hidden_dims),
        activation=network.activation,
        normalization=network.normalization,
        learning_rate=optimizer.learning_rate,
        max_grad_norm=optimizer.max_grad_norm,
        optimizer=optimizer.optimizer,
        weight_decay=optimizer.weight_decay,
        clip_eps=ppo.clip_eps,
        value_coef=ppo.value_coef,
        num_envs=ppo.batch_size,
        num_epochs=ppo.ppo_epochs,
        num_components=network.num_components,
        full_covariance=network.full_covariance,
        scale_parameterization=network.scale_parameterization,
        max_correlation=network.max_correlation,
        num_atoms=_num_atoms(space),
        clip_means=network.clip_means,
        mean_box_penalty_coef=network.mean_box_penalty_coef,
        low=low,
        high=high,
        target_tau=ppo.target_tau,
        magnet_interval=ppo.magnet_interval,
        category_entropy_coef=ppo.category_entropy_coef,
        gaussian_entropy_coef=ppo.gaussian_entropy_coef,
        trpo_category_kl_coef=ppo.trpo_category_kl_coef,
        trpo_gaussian_kl_coef=ppo.trpo_gaussian_kl_coef,
        magnet_category_kl_coef=ppo.magnet_category_kl_coef,
        magnet_gaussian_kl_coef=ppo.magnet_gaussian_kl_coef,
    )


def _or_gaussian(value: float | None, fallback: float) -> float:
    """An `exp_family` coefficient, defaulting to its Gaussian-head counterpart.

    Lets a config written for the mixture policy run under the exponential
    family unchanged (`density_*_coef: null`), while still allowing the two to
    be tuned apart -- the terms are not the same object, since the mixture's are
    estimates of quantities the log-linear family computes exactly.
    """
    return fallback if value is None else value


def build_expfam_hyperparams(
    game: ZeroSumGame, player: int, config: RunConfig
) -> ExpFamilyPPOHyperparams:
    """`build_hyperparams`'s counterpart for `network.policy: exp_family`.

    The action box is not an initialization hint here but the support of the
    density itself, so a space with no box (a `SimplexSpace`) is rejected rather
    than approximated: there is nothing sensible to lay a grid on.
    """
    space = game.action_space(player)
    if not isinstance(space, BoxSpace):
        raise ValueError(
            f"policy 'exp_family' needs a BoxSpace to lay its grid on, got {type(space).__name__}. "
            "(A HybridSpace would need the atoms put back in front of the density; see "
            "training/expfam.py's scope note.)"
        )
    low, high = action_bounds(space)
    network, optimizer, ppo = config.network, config.optimizer, config.ppo
    return ExpFamilyPPOHyperparams(
        action_dim=space.shape[0],
        hidden_dims=tuple(network.hidden_dims),
        activation=network.activation,
        normalization=network.normalization,
        learning_rate=optimizer.learning_rate,
        max_grad_norm=optimizer.max_grad_norm,
        optimizer=optimizer.optimizer,
        weight_decay=optimizer.weight_decay,
        clip_eps=ppo.clip_eps,
        value_coef=ppo.value_coef,
        num_envs=ppo.batch_size,
        num_epochs=ppo.ppo_epochs,
        low=low,
        high=high,
        grid_points=network.grid_points,
        num_basis=network.num_basis,
        poly_order=network.poly_order,
        basis_width_scale=network.basis_width_scale,
        init_tilt=network.init_tilt,
        target_tau=ppo.target_tau,
        magnet_interval=ppo.magnet_interval,
        density_entropy_coef=_or_gaussian(ppo.density_entropy_coef, ppo.gaussian_entropy_coef),
        trpo_density_kl_coef=_or_gaussian(ppo.trpo_density_kl_coef, ppo.trpo_gaussian_kl_coef),
        magnet_density_kl_coef=_or_gaussian(ppo.magnet_density_kl_coef, ppo.magnet_gaussian_kl_coef),
    )
