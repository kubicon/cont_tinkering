"""A configurable MLP: pick activation/normalization by name and swap freely."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import chex
import flax.linen as nn

from .activations import Activation
from .normalization import Normalization


class MLP(nn.Module):
    """Dense -> normalization -> activation, repeated, then a plain output layer.

    Pass `train=True` when calling during optimization if `normalization`
    is "batch_norm" (it needs `mutable=["batch_stats"]` in that case); every
    other normalization ignores `train`.

    `output_kernel_init`/`output_bias_init` override the final layer's
    initializers (default: flax's usual `nn.Dense` initializers). Useful for
    e.g. forcing specific initial output values regardless of the input by
    setting `output_kernel_init=nn.initializers.zeros` alongside a custom
    `output_bias_init` — since the kernel contributes nothing, the initial
    output is exactly the bias.
    """

    hidden_dims: Sequence[int]
    output_dim: int
    activation: str = "relu"
    normalization: str = "none"
    output_activation: str = "identity"
    use_bias: bool = True
    output_kernel_init: Optional[Callable] = None
    output_bias_init: Optional[Callable] = None

    @nn.compact
    def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
        for dim in self.hidden_dims:
            x = nn.Dense(dim, use_bias=self.use_bias)(x)
            x = Normalization(kind=self.normalization)(x, use_running_average=not train)
            x = Activation(kind=self.activation)(x)

        output_kwargs = {}
        if self.output_kernel_init is not None:
            output_kwargs["kernel_init"] = self.output_kernel_init
        if self.output_bias_init is not None:
            output_kwargs["bias_init"] = self.output_bias_init
        x = nn.Dense(self.output_dim, use_bias=self.use_bias, **output_kwargs)(x)
        x = Activation(kind=self.output_activation)(x)
        return x
