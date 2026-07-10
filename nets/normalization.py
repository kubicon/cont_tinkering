"""Normalization layers, registered by name behind one uniform call signature.

`nn.BatchNorm` is the odd one out among flax's norm layers: it's stateful
(tracks running statistics in a `batch_stats` collection) and needs to know
whether it's training. `Normalization` hides that behind `use_running_average`
so callers can swap normalization kind without changing call sites.
"""

from __future__ import annotations

import chex
import flax.linen as nn

NORMALIZATIONS = ("none", "layer_norm", "rms_norm", "group_norm", "batch_norm")


class Normalization(nn.Module):
    """Normalization selected by name: "none", "layer_norm", "rms_norm", "group_norm", "batch_norm"."""

    kind: str = "none"
    num_groups: int = 8

    @nn.compact
    def __call__(self, x: chex.Array, use_running_average: bool = True) -> chex.Array:
        if self.kind == "none":
            return x
        elif self.kind == "layer_norm":
            return nn.LayerNorm()(x)
        elif self.kind == "rms_norm":
            return nn.RMSNorm()(x)
        elif self.kind == "group_norm":
            return nn.GroupNorm(num_groups=self.num_groups)(x)
        elif self.kind == "batch_norm":
            return nn.BatchNorm(use_running_average=use_running_average)(x)
        else:
            raise ValueError(f"Unknown normalization {self.kind!r}, expected one of {NORMALIZATIONS}")
