"""Rock-paper-scissors disk sumo: three archetypes that counter each other in a cycle.

`games.disk_sumo_v2` with a trait table designed to be *intransitive* rather
than merely balanced. v2 balances seven archetypes so that none is strong on
average; here each of three archetypes has one clear counter and one clear
victim, so the matchup -- not the archetype -- decides whether to attack or to
wait:

  * `rammer` beats `grinder`. Light, low drag, no speed limit: it can always
    open up the distance for a run-up, and a grinder knocked back fast has lost
    its grip and slides. Weakest in a slow shove. (Light, not heavy: the energy
    a run-up can store is `force * distance` whatever the mass, and the speed a
    hit gives the target peaks when the two masses are equal -- while a heavy
    disk is too slow to escape a grinder that simply charges it.)
  * `grinder` beats `anchor`. The strongest stall thrust, but its grip fades as
    it moves faster. A slow push barely excites the anchor's brace, which grows
    with speed, and pushing back switches that brace off.
  * `anchor` beats `rammer`. While idle it resists motion quadratically in its
    speed, so a fast impact is soaked up; once the rammer has stopped, the
    anchor out-shoves it.

The favoured side of each matchup has a different job: the rammer backs off and
charges, the grinder closes slowly and leans, the anchor holds the middle and
only pushes once the collision is over.

**Two traits on top of v2's** (all traits are multipliers on shared physics):
  * `top_speed` -- grip that fades with speed. Every substep, thrust is scaled
    by `clip(1 - |v| / top_speed, 0, 1)`, with `top_speed` in units of the
    free-running speed `max_force / drag`. It caps how fast the disk can drive
    itself, and -- because `|v|` has no direction -- it also means a disk
    knocked back faster than `top_speed` cannot brake: it slides under drag
    alone. That second half is what lets a hit beat a stronger pusher; with a
    one-sided curve (thrust kept while driven backwards) the stronger thrust
    soaks up any impact a run-up on this ring can deliver. `inf` (neutral) is
    v2's constant thrust.
  * `brace_quadratic` -- idle drag that grows with speed: the drag coefficient
    gains `brace_quadratic * (1 - effort) * |v|`, so the resisting force is
    `~ |v|^2`. v2's linear `brace` is still available.

**Shrinking ring.** With `ring_shrink = s > 0` the ring radius falls linearly
from `ring_radius` to `(1 - s) * ring_radius` over the horizon, so the side a
matchup disfavours cannot run out the clock: sooner or later the disks meet.
The observed edge margins are measured against the current ring; positions are
still in units of the initial `ring_radius`, and so is the timeout margin.

Everything else -- observation layout, stamina, hidden types, the payoff -- is
v2's.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .disk_sumo import _EPS
from .disk_sumo_v2 import TRAIT_NAMES as V2_TRAIT_NAMES
from .disk_sumo_v2 import _NEUTRAL as V2_NEUTRAL
from .disk_sumo_v2 import DiskSumoV2, DiskSumoV2State

TRAIT_NAMES = V2_TRAIT_NAMES + ("top_speed", "brace_quadratic")
_NEUTRAL = {**V2_NEUTRAL, "top_speed": float("inf"), "brace_quadratic": 0.0}

# Tuned by sweeping traits with scripts/balance_disk_sumo_v2.py's scripted
# library against configs/disk_sumo_v3.yaml (contact_damping 3, ring_shrink
# 0.7, start_distance 0.45). Measured matchup values at 64 episodes:
# rammer > grinder +0.40, grinder > anchor +0.06, anchor > rammer +0.63.
# The cycle is sensitive: grinder top_speed 0.5 or rammer mass 0.6 already
# flips rammer > grinder. Exhausted thrust `force * min_force` is equal for
# everyone, as in v2, so a bout between two spent disks is not decided by it.
DEFAULT_ARCHETYPES: dict[str, dict[str, float]] = {
    "rammer": {"force": 0.85, "mass": 0.5, "drag": 0.6, "drain": 1.2, "min_force": 1.176},
    # Low drag matters as much as the grip: knocked loose, it slides.
    "grinder": {"force": 1.35, "mass": 0.8, "drag": 0.5, "top_speed": 0.4, "min_force": 0.741},
    "anchor": {"force": 1.0, "brace_quadratic": 6.0},
}


class DiskSumoV3(DiskSumoV2):
    """Disk sumo with three cyclically countering archetypes. See the module docstring.

    Takes every `DiskSumoV2` argument, plus `ring_shrink`.
    """

    TRAIT_NAMES = TRAIT_NAMES
    DEFAULT_ARCHETYPES = DEFAULT_ARCHETYPES
    NEUTRAL_TRAITS = _NEUTRAL

    def __init__(self, ring_shrink: float = 0.0, **kwargs):
        if not 0.0 <= ring_shrink < 1.0:
            raise ValueError(f"ring_shrink must be in [0, 1), got {ring_shrink}")
        self.ring_shrink = float(ring_shrink)
        super().__init__(**kwargs)
        for name, traits in self.archetype_table.items():
            if not traits["top_speed"] > 0.0:
                raise ValueError(f"archetype {name!r}: top_speed must be positive, got {traits['top_speed']}")
            if traits["brace_quadratic"] < 0.0:
                raise ValueError(
                    f"archetype {name!r}: brace_quadratic must be non-negative, got {traits['brace_quadratic']}"
                )

    def _base_traits(self) -> dict[str, float]:
        return {**super()._base_traits(), "top_speed": self._speed_scale, "brace_quadratic": 1.0}

    def ring_radius_at(self, state: DiskSumoV2State) -> chex.Array:
        """Linear from `ring_radius` at control step 0 to `(1 - ring_shrink) * ring_radius` at the horizon."""
        step = jnp.minimum(state.turn // 2, self.horizon).astype(jnp.float32)
        return self.ring_radius * (1.0 - self.ring_shrink * step / self.horizon)

    # ---- physics ------------------------------------------------------------

    def _integrate(self, state: DiskSumoV2State, pos, vel, forces, efforts):
        """Per-disk bodies, both braces for a disk that is barely thrusting, the motor curve, the current ring."""
        traits = self.traits(state.archetype)
        idle = 1.0 - efforts
        drag = traits["drag"] * (1.0 + traits["brace"] * idle)
        return self._simulate_motor(
            pos, vel, forces, traits["mass"], drag, traits["brace_quadratic"] * idle,
            traits["top_speed"], self.ring_radius_at(state),
        )

    def _simulate_motor(self, pos, vel, forces, mass, drag, quadratic, top_speed, ring):
        """`DiskSumoV2._simulate_bodies` with speed-dependent thrust and drag, and ring radius `ring`."""
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
            motor = jnp.clip(1.0 - speed / top_speed, 0.0, 1.0)
            resist = drag + quadratic * speed
            vel = vel + h * (forces * motor[:, None] - resist[:, None] * vel + contact) * inv_mass
            pos = pos + h * vel
            out = jnp.linalg.norm(pos, axis=-1) > ring
            exit_at = jnp.where(out & (exit_at == never), i, exit_at)
            return pos, vel, exit_at

        exit_at = jnp.full((2,), never, dtype=jnp.int32)
        pos, vel, exit_at = jax.lax.fori_loop(0, self.substeps, body, (pos, vel, exit_at))
        out_0, out_1 = exit_at[0] < never, exit_at[1] < never
        result = jnp.where(
            exit_at[0] < exit_at[1], -1.0, jnp.where(exit_at[1] < exit_at[0], 1.0, 0.0)
        ).astype(jnp.float32)
        return pos, vel, result, out_0 | out_1
