"""Optimizer registry, selected by name so `PPOHyperparams.optimizer` can swap freely.

"Optimistic" variants (`optimistic_sgd`, `optimistic_adam_v2`) use the
previous step's gradient as a cheap look-ahead correction; they're the
standard fix for the non-convergence/cycling that plain simultaneous
gradient descent-ascent can suffer in zero-sum self-play (see
`SelfPlayPPOTrainer`'s docstring), so they're worth trying there even
though nothing here enforces that pairing.
"""

from __future__ import annotations

import warnings
from typing import Callable

import optax
import optax.contrib as optax_contrib

OPTIMIZERS: dict[str, Callable[[float], optax.GradientTransformation]] = {
    "sgd": optax.sgd,
    "optimistic_sgd": optax.optimistic_gradient_descent,
    "adam": optax.adam,
    "adamw": optax.adamw,
    "optimistic_adam_v2": optax.optimistic_adam_v2,
    "muon": optax_contrib.muon,
}

# Optimizers whose factory accepts a `weight_decay` kwarg.
WEIGHT_DECAY_OPTIMIZERS = frozenset({"adamw", "muon"})


def build_optimizer(name: str, learning_rate: float, weight_decay: float = 0.0) -> optax.GradientTransformation:
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}, expected one of {sorted(OPTIMIZERS)}")
    if name in WEIGHT_DECAY_OPTIMIZERS:
        return OPTIMIZERS[name](learning_rate, weight_decay=weight_decay)
    if weight_decay != 0.0:
        warnings.warn(
            f"optimizer {name!r} does not support weight_decay (only {sorted(WEIGHT_DECAY_OPTIMIZERS)} do); ignoring it"
        )
    return OPTIMIZERS[name](learning_rate)
