"""Rock-paper-scissors disk sumo: three archetypes that counter each other in a cycle.

`games.disk_sumo_v2` with a trait table designed to be *intransitive* rather
than merely balanced. v2 balances seven archetypes so that none is strong on
average; here each of three archetypes has one clear counter and one clear
victim, so the matchup -- not the archetype -- decides whether to attack or to
wait. Each counter rests on a threshold rather than a smooth curve, so a
matchup is decided by which side of the threshold the bout is played on:

  * `rammer` beats `grinder`. Light, low drag, no speed limit, but the weakest
    thrust: it loses any standing shove and wins only with a run-up. A grinder
    knocked back past its top speed has lost its grip and slides.
  * `grinder` beats `anchor`. By far the strongest stall thrust, with grip that
    cuts out just above a low top speed. Its slow push never reaches the
    anchor's brace threshold, so against it the anchor is an ordinary disk.
  * `anchor` beats `rammer`. While idle it resists motion sharply above a
    threshold speed, so a fast impact is soaked up; once the rammer has
    stopped, the anchor out-shoves it.

The favoured side of each matchup has a different job: the rammer backs off and
charges, the grinder closes slowly and leans, the anchor holds the middle and
only pushes once the collision is over.

**Three traits on top of v2's** (all traits are multipliers on shared physics):
  * `top_speed` -- grip that cuts out with speed. Every substep, thrust is
    scaled by `sigmoid((1 - |v| / top_speed) / grip_width)`, with `top_speed` in
    units of the free-running speed `max_force / drag`: full thrust below it,
    almost none above it (`grip_width` is a game parameter, the width of the
    cut-out as a fraction of `top_speed`). Because `|v|` has no direction, a
    disk knocked back faster than `top_speed` cannot brake either: it slides
    under drag alone. `inf` (neutral) is v2's constant thrust.
  * `impact_brace`, `brace_speed` -- idle drag above a threshold speed: the
    drag coefficient gains `impact_brace * (1 - effort) * max(|v| - brace_speed, 0)`,
    with `brace_speed` in units of `max_force / drag`. Below the threshold the
    disk is unbraced; above it the resisting force grows like `|v|^2`. v2's
    linear `brace` is still available.

The ring is fixed. Everything else -- observation layout, stamina, hidden types,
the payoff -- is v2's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .disk_sumo import _EPS
from .disk_sumo_v2 import TRAIT_NAMES as V2_TRAIT_NAMES
from .disk_sumo_v2 import _NEUTRAL as V2_NEUTRAL
from .disk_sumo_v2 import DiskSumoV2, DiskSumoV2State

TRAIT_NAMES = V2_TRAIT_NAMES + ("top_speed", "impact_brace", "brace_speed")
_NEUTRAL = {**V2_NEUTRAL, "top_speed": float("inf"), "impact_brace": 0.0, "brace_speed": 0.0}

# Starting point for the threshold mechanics, not yet tuned: measure with
# scripts/balance_disk_sumo_v2.py configs/disk_sumo_v3.yaml. Exhausted thrust
# `force * min_force` is equal for everyone, as in v2, so a bout between two
# spent disks is not decided by it. With the shared physics of
# configs/disk_sumo_v3.yaml (max_force 1, drag 0.5) the speed unit is 2: the
# grinder is capped at 0.8, the anchor braces above 1.0, and a free-running
# rammer reaches 0.7 / 0.3 = 2.33.
DEFAULT_ARCHETYPES: dict[str, dict[str, float]] = {
    "rammer": {"force": 0.7, "mass": 0.5, "drag": 0.6, "drain": 1.2, "min_force": 1.429},
    # Low drag matters as much as the grip: knocked loose, it slides.
    "grinder": {"force": 1.5, "mass": 0.8, "drag": 0.5, "top_speed": 0.4, "min_force": 0.667},
    # brace_speed sits above the grinder's top speed, so only a charge triggers it.
    "anchor": {"force": 1.0, "impact_brace": 8.0, "brace_speed": 0.5},
}


class DiskSumoV3(DiskSumoV2):
    """Disk sumo with three cyclically countering archetypes. See the module docstring.

    Takes every `DiskSumoV2` argument, plus `grip_width`.
    """

    TRAIT_NAMES = TRAIT_NAMES
    DEFAULT_ARCHETYPES = DEFAULT_ARCHETYPES
    NEUTRAL_TRAITS = _NEUTRAL

    def __init__(self, grip_width: float = 0.05, **kwargs):
        if not grip_width > 0.0:
            raise ValueError(f"grip_width must be positive, got {grip_width}")
        self.grip_width = float(grip_width)
        super().__init__(**kwargs)
        for name, traits in self.archetype_table.items():
            if not traits["top_speed"] > 0.0:
                raise ValueError(f"archetype {name!r}: top_speed must be positive, got {traits['top_speed']}")
            for trait in ("impact_brace", "brace_speed"):
                if traits[trait] < 0.0:
                    raise ValueError(f"archetype {name!r}: {trait} must be non-negative, got {traits[trait]}")

    def _base_traits(self) -> dict[str, float]:
        return {**super()._base_traits(), "top_speed": self._speed_scale,
                "impact_brace": 1.0, "brace_speed": self._speed_scale}

    # ---- physics ------------------------------------------------------------

    def _integrate(self, state: DiskSumoV2State, pos, vel, forces, efforts):
        """Per-disk bodies, both braces for a disk that is barely thrusting, the grip cut-out."""
        traits = self.traits(state.archetype)
        idle = 1.0 - efforts
        drag = traits["drag"] * (1.0 + traits["brace"] * idle)
        return self._simulate_motor(
            pos, vel, forces, traits["mass"], drag, traits["impact_brace"] * idle,
            traits["brace_speed"], traits["top_speed"],
        )

    def _simulate_motor(self, pos, vel, forces, mass, drag, impact, brace_speed, top_speed):
        """`DiskSumoV2._simulate_bodies` with speed-dependent thrust and drag."""
        h = self.dt / self.substeps
        never = self.substeps
        inv_mass = (1.0 / mass)[:, None]

        def body(i, carry):
            pos, vel, exit_at = carry
            delta = pos[1] - pos[0]
            distance = jnp.linalg.norm(delta)
            normal = delta / jnp.maximum(distance, _EPS)
            overlap = jnp.maximum(2.0 * self.disk_radius - distance, 0.0)
            closing = jnp.dot(vel[1] - vel[0], normal)
            push = jnp.where(
                overlap > 0.0,
                jnp.maximum(self.stiffness * overlap - self.contact_damping * closing, 0.0),
                0.0,
            )
            contact = jnp.stack([-push * normal, push * normal])
            speed = jnp.linalg.norm(vel, axis=-1)
            motor = jax.nn.sigmoid((1.0 - speed / top_speed) / self.grip_width)
            resist = drag + impact * jnp.maximum(speed - brace_speed, 0.0)
            vel = vel + h * (forces * motor[:, None] - resist[:, None] * vel + contact) * inv_mass
            pos = pos + h * vel
            out = jnp.linalg.norm(pos, axis=-1) > self.ring_radius
            exit_at = jnp.where(out & (exit_at == never), i, exit_at)
            return pos, vel, exit_at

        exit_at = jnp.full((2,), never, dtype=jnp.int32)
        pos, vel, exit_at = jax.lax.fori_loop(0, self.substeps, body, (pos, vel, exit_at))
        out_0, out_1 = exit_at[0] < never, exit_at[1] < never
        result = jnp.where(
            exit_at[0] < exit_at[1], -1.0, jnp.where(exit_at[1] < exit_at[0], 1.0, 0.0)
        ).astype(jnp.float32)
        return pos, vel, result, out_0 | out_1
