"""Activation functions, registered by name so modules can select one via a config string."""

from __future__ import annotations

from typing import Callable

import chex
import flax.linen as nn
import jax.numpy as jnp

ACTIVATIONS: dict[str, Callable[[chex.Array], chex.Array]] = {
    "identity": lambda x: x,
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "elu": nn.elu,
    "selu": nn.selu,
    "gelu": nn.gelu,
    "silu": nn.silu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "softplus": nn.softplus,
}


class Activation(nn.Module):
    """Stateless activation selected by name; drop-in `nn.Module` for use inside `nn.compact` bodies."""

    kind: str = "relu"

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if self.kind not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {self.kind!r}, expected one of {sorted(ACTIVATIONS)}")
        return ACTIVATIONS[self.kind](x)
