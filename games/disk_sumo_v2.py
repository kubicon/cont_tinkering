"""Archetype disk sumo: `games.disk_sumo` where each disk is dealt a random body type.

The ring, the facing/random start, the egocentric frame, the two-decisions-per-
control-step encoding of the simultaneous move, the payoff and the shaping are
all exactly `DiskSumo`'s (see that module). What changes is the bodies.

**Archetypes.** At the start of every bout each seat independently draws an
archetype uniformly from `archetypes` (mirror matches included, so the game is
symmetric in expectation and one self-play policy can play both seats). An
archetype is a row of traits, all multipliers on the shared physics unless
noted:

  * `force`    -- thrust, the only thing that decides a steady shoving match;
  * `mass`     -- inertia: slower to accelerate, harder to knock back;
  * `drag`     -- top speed is `force / drag`, so lower drag is faster;
  * `drain`, `recovery` -- stamina drain / recovery rates;
  * `min_force` -- thrust retained at zero stamina (multiplies `stamina_min_force`);
  * `brace`    -- absolute, not a multiplier: the drag coefficient becomes
    `drag * (1 + brace * (1 - effort))`, so a disk that is barely thrusting is
    planted and hard to move, while a full-effort disk moves normally.

Speed and push strength are both `force / mass`-flavoured in this physics, so
"fast but weak" is expressed as low drag plus low force, never force alone.
`DEFAULT_ARCHETYPES` is tuned so that no archetype dominates: see
`scripts/balance_disk_sumo_v2.py`, which solves the archetype-vs-archetype
matrix game over a library of scripted strategies.

Stamina is always on: three of the archetypes are defined by it.

**Information.** `observe_opponent_archetype` controls whether the opponent's
archetype one-hot is visible (a hidden one reads as all zeros, the observation
width never changes). The opponent's stamina level leaks its drain rate, so
`observe_opponent_stamina` defaults to the same setting. With hidden types a
memoryless policy can only infer the opponent from a single frame, so
`observe_velocity_change` appends each disk's velocity change over the last
control step -- acceleration is what reveals mass and force. Note that with a
hidden archetype the game is no longer one where the current state is a
sufficient statistic of the history: past play carries evidence about the type.

Observation layout, `(11 + 2K + 2 [+ 4],)` for `K` enabled archetypes: the base
11 of `DiskSumo`, own archetype one-hot, opponent archetype one-hot (or zeros),
own and opponent stamina (or zero), and optionally own and opponent velocity
change in the egocentric frame.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import chex
import jax
import jax.numpy as jnp

from .disk_sumo import _EPS, DiskSumo
from .spaces import HybridAction

TRAIT_NAMES = ("force", "mass", "drag", "drain", "recovery", "min_force", "brace")
_NEUTRAL = {"force": 1.0, "mass": 1.0, "drag": 1.0, "drain": 1.0,
            "recovery": 1.0, "min_force": 1.0, "brace": 0.0}

# Tuned with scripts/balance_disk_sumo_v2.py against the physics of
# configs/disk_sumo_v2.yaml (drag 0.5, stamina 0.5 / 0.25 / 0.25).
#
# Two invariants keep long shoving matches from deciding every matchup, which
# the first measurements showed they otherwise do:
#   * exhausted thrust `force * min_force` is equal for everyone
#     (`min_force = 1 / force`), so once both disks are spent neither gains; and
#   * the ground an archetype can take before it is spent -- roughly
#     `(force - floor) * full_thrust_seconds` -- stays near the neutral 1.5
#     (0.75 * 2 s), so no stamina profile simply wins the opening shove.
#     Endurance buys a longer window with a lower peak, burst the reverse.
# What remains to differ is how a disk gets there: peak thrust, speed, inertia,
# how long full thrust lasts, and bracing.
# DEFAULT_ARCHETYPES: dict[str, dict[str, float]] = {
#     # Faster and lighter, loses a straight shove.
#     "quick": {"force": 0.948, "mass": 0.9, "drag": 0.72, "min_force": 1.056},
#     # Wins a straight shove, low top speed.
#     "strong": {"force": 1.029, "drag": 1.4, "min_force": 0.972},
#     # Weaker peak, but full thrust lasts 2.7 s instead of 2 s.
#     "endurance": {"force": 0.82, "drain": 0.75, "recovery": 0.8, "min_force": 1.219},
#     # Barely moved by collisions, sluggish to start and to turn.
#     "heavy": {"force": 0.972, "mass": 1.6, "min_force": 1.029},
#     # Changes direction almost instantly, low top speed, easily knocked back.
#     "nimble": {"force": 0.979, "mass": 0.6, "drag": 1.3, "min_force": 1.023},
#     # Hits very hard for 1.3 s, then gasses out. Its drag keeps its top speed
#     # below quick's: thrust does not also make it fast.
#     "burst": {"force": 1.39, "drag": 1.45, "drain": 1.5, "recovery": 1.3, "min_force": 0.72},
#     # Planted when not thrusting, weak on offence.
#     "brace": {"force": 0.958, "brace": 3.0, "min_force": 1.044},
# }

DEFAULT_ARCHETYPES: dict[str, dict[str, float]] = {
    # Faster and lighter, loses a straight shove.
    "quick": {"force": 0.9, "mass": 0.9, "drag": 0.72, "min_force": 1.111},
    # Wins a straight shove, low top speed.
    "strong": {"force": 1.08, "drag": 1.4, "min_force": 0.926},
    # Weaker peak, but full thrust lasts 2.7 s instead of 2 s.
    "endurance": {"force": 0.85, "drain": 0.75, "recovery": 0.8, "min_force": 1.176},
    # Barely moved by collisions, sluggish to start and to turn.
    "heavy": {"force": 0.957, "mass": 1.6, "min_force": 1.045},
    # Changes direction almost instantly, low top speed, easily knocked back.
    "nimble": {"force": 0.961, "mass": 0.6, "drag": 1.3, "min_force": 1.041},
    # Hits very hard for 1.3 s, then gasses out. Drag scales with its force so
    # its top speed stays the neutral one: thrust does not also make it fast.
    "burst": {"force": 1.45, "drag": 1.45, "drain": 1.5, "recovery": 1.3, "min_force": 0.69},
    # Planted when not thrusting, weak on offence.
    "brace": {"force": 0.92, "brace": 3.0, "min_force": 1.087},
}


@chex.dataclass(frozen=True)
class DiskSumoV2State:
    """Fixed-shape state of one `DiskSumoV2` bout.

    As `DiskSumoState`, with `archetype` replacing `force_multiplier` and
    `prev_vel` holding the velocities at the start of the last resolved control
    step (for the optional velocity-change observation).
    """

    pos: chex.Array  # (2, 2) float32
    vel: chex.Array  # (2, 2) float32
    prev_vel: chex.Array  # (2, 2) float32
    pending: chex.Array  # (2,) float32, player 0's world-frame force this step
    pending_effort: chex.Array  # () float32, hidden player-0 effort in [0, 1]
    archetype: chex.Array  # (2,) int32, index into the enabled archetypes
    stamina: chex.Array  # (2,) float32 in [0, 1]
    result: chex.Array  # () float32 in {-1, 0, +1}
    done: chex.Array  # () bool
    turn: chex.Array  # () int32


def resolve_archetypes(
    archetypes: Sequence[str] | None,
    archetype_traits: Mapping[str, Mapping[str, float]] | None,
    defaults: Mapping[str, Mapping[str, float]] = DEFAULT_ARCHETYPES,
    trait_names: Sequence[str] = TRAIT_NAMES,
    neutral: Mapping[str, float] = _NEUTRAL,
) -> tuple[tuple[str, ...], dict[str, dict[str, float]]]:
    """Enabled names and their full trait rows, defaults merged with overrides.

    An override may name an archetype that is not in `defaults`; its
    unspecified traits are neutral. The last three arguments let a subclass
    (`games.disk_sumo_v3`) bring its own trait set.
    """
    table = {name: dict(traits) for name, traits in defaults.items()}
    for name, traits in (archetype_traits or {}).items():
        unknown = set(traits) - set(trait_names)
        if unknown:
            raise ValueError(f"unknown trait(s) {sorted(unknown)} for archetype {name!r}, "
                             f"choices: {list(trait_names)}")
        table.setdefault(name, {}).update({k: float(v) for k, v in traits.items()})
    names = tuple(archetypes) if archetypes is not None else tuple(defaults)
    if not names:
        raise ValueError("at least one archetype must be enabled")
    if len(set(names)) != len(names):
        raise ValueError(f"archetypes must be distinct, got {list(names)}")
    for name in names:
        if name not in table:
            raise ValueError(f"unknown archetype {name!r}, choices: {sorted(table)}")
    return names, {name: {**neutral, **table[name]} for name in names}


class DiskSumoV2(DiskSumo):
    """Disk sumo with randomly dealt, optionally hidden archetypes. See the module docstring."""

    TRAIT_NAMES = TRAIT_NAMES
    DEFAULT_ARCHETYPES = DEFAULT_ARCHETYPES
    NEUTRAL_TRAITS = _NEUTRAL

    def __init__(
        self,
        horizon: int = 100,
        ring_radius: float = 1.0,
        disk_radius: float = 0.15,
        start_distance: float = 0.45,
        start_jitter: float = 0.04,
        random_orientation: bool = True,
        random_start: bool = False,
        egocentric: bool = True,
        dt: float = 0.1,
        substeps: int = 10,
        mass: float = 1.0,
        max_force: float = 1.0,
        stamina_drain_rate: float = 0.5,
        stamina_recovery_rate: float = 0.25,
        stamina_min_force: float = 0.25,
        drag: float = 0.5,
        stiffness: float = 400.0,
        contact_damping: float = 10.0,
        margin_weight: float = 0.0,
        shaping_weight: float = 0.0,
        archetypes: Sequence[str] | None = None,
        archetype_traits: Mapping[str, Mapping[str, float]] | None = None,
        observe_opponent_archetype: bool = True,
        observe_opponent_stamina: bool | None = None,
        observe_velocity_change: bool = False,
    ):
        super().__init__(
            horizon=horizon, ring_radius=ring_radius, disk_radius=disk_radius,
            start_distance=start_distance, start_jitter=start_jitter,
            random_orientation=random_orientation, random_start=random_start,
            egocentric=egocentric, dt=dt, substeps=substeps, mass=mass, max_force=max_force,
            force_asymmetry=0.0, stamina_enabled=True, stamina_drain_rate=stamina_drain_rate,
            stamina_recovery_rate=stamina_recovery_rate, stamina_min_force=stamina_min_force,
            drag=drag, stiffness=stiffness, contact_damping=contact_damping,
            margin_weight=margin_weight, shaping_weight=shaping_weight,
        )
        names, table = resolve_archetypes(
            archetypes, archetype_traits, self.DEFAULT_ARCHETYPES, self.TRAIT_NAMES, self.NEUTRAL_TRAITS
        )
        for name, traits in table.items():
            for trait in ("force", "mass"):
                if traits[trait] <= 0.0:
                    raise ValueError(f"archetype {name!r}: {trait} must be positive, got {traits[trait]}")
            for trait in ("drag", "drain", "recovery", "brace"):
                if traits[trait] < 0.0:
                    raise ValueError(f"archetype {name!r}: {trait} must be non-negative, got {traits[trait]}")
            if not 0.0 <= traits["min_force"] * stamina_min_force <= 1.0:
                raise ValueError(
                    f"archetype {name!r}: min_force * stamina_min_force must be in [0, 1], "
                    f"got {traits['min_force'] * stamina_min_force}"
                )
        self.archetype_names = names
        self.archetype_table = table
        self.num_archetypes = len(names)
        self.observe_opponent_archetype = bool(observe_opponent_archetype)
        self.observe_opponent_stamina = (
            self.observe_opponent_archetype if observe_opponent_stamina is None
            else bool(observe_opponent_stamina)
        )
        self.observe_velocity_change = bool(observe_velocity_change)
        # (K, len(TRAIT_NAMES)) with the shared physics folded in, so the step
        # only ever gathers rows.
        base = self._base_traits()
        self._traits = jnp.asarray(
            [[table[name][t] * base[t] for t in self.TRAIT_NAMES] for name in names], dtype=jnp.float32
        )

    def _base_traits(self) -> dict[str, float]:
        """The shared physics each trait multiplier scales."""
        return {"force": self.max_force, "mass": self.mass, "drag": self.drag,
                "drain": self.stamina_drain_rate, "recovery": self.stamina_recovery_rate,
                "min_force": self.stamina_min_force, "brace": 1.0}

    # ---- shape/static information ------------------------------------------

    def obs_dim(self, player: int) -> int:
        return 11 + 2 * self.num_archetypes + 2 + 4 * self.observe_velocity_change

    def traits(self, archetype: chex.Array) -> dict[str, chex.Array]:
        """Absolute trait values (shared physics folded in) for archetype indices."""
        rows = self._traits[archetype]
        return {name: rows[..., i] for i, name in enumerate(self.TRAIT_NAMES)}

    def ring_radius_at(self, state: DiskSumoV2State) -> chex.Array:
        """Ring radius during the control step `state` is in; constant here."""
        del state
        return self.ring_radius

    # ---- the game tree ------------------------------------------------------

    def initial_state(self, key: chex.PRNGKey) -> DiskSumoV2State:
        """Both disks at rest; each seat's archetype drawn independently and uniformly."""
        pos = self._random_start(key) if self.random_start else self._facing_start(key)
        archetype = jax.random.randint(
            jax.random.fold_in(key, 1), (2,), 0, self.num_archetypes
        ).astype(jnp.int32)
        zeros = jnp.zeros((2, 2), dtype=jnp.float32)
        return DiskSumoV2State(
            pos=pos.astype(jnp.float32),
            vel=zeros,
            prev_vel=zeros,
            pending=jnp.zeros((2,), dtype=jnp.float32),
            pending_effort=jnp.zeros((), dtype=jnp.float32),
            archetype=archetype,
            stamina=jnp.ones((2,), dtype=jnp.float32),
            result=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=bool),
            turn=jnp.zeros((), dtype=jnp.int32),
        )

    def observation(self, player: int, state: DiskSumoV2State) -> chex.Array:
        """See the module docstring for the layout; `state.pending*` is never read."""
        own, opp = player, 1 - player
        rotation = self._frame(state.pos[own], state.pos[opp])
        radii = jnp.linalg.norm(state.pos, axis=-1) / self.ring_radius_at(state)
        time_left = 1.0 - (state.turn // 2).astype(jnp.float32) / self.horizon
        one_hot = jax.nn.one_hot(state.archetype, self.num_archetypes, dtype=jnp.float32)
        parts = [
            rotation @ state.pos[own] / self.ring_radius,
            rotation @ state.vel[own] / self._speed_scale,
            rotation @ state.pos[opp] / self.ring_radius,
            rotation @ state.vel[opp] / self._speed_scale,
            jnp.stack([1.0 - radii[own], 1.0 - radii[opp], time_left]),
            one_hot[own],
            one_hot[opp] * self.observe_opponent_archetype,
            jnp.stack([state.stamina[own], state.stamina[opp] * self.observe_opponent_stamina]),
        ]
        if self.observe_velocity_change:
            change = (state.vel - state.prev_vel) / self._speed_scale
            parts += [rotation @ change[own], rotation @ change[opp]]
        return jnp.concatenate(parts)

    def _step(self, state: DiskSumoV2State, action: HybridAction, key: chex.PRNGKey) -> DiskSumoV2State:
        del key
        player = state.turn % 2
        traits = self.traits(state.archetype)
        force, effort = self._world_force(state, player, action.value)
        efforts = jnp.stack([state.pending_effort, effort])
        pos, vel, result, done = self._integrate(
            state, state.pos, state.vel, jnp.stack([state.pending, force]), efforts
        )
        resolves = player == 1
        stamina_delta = self.dt * (traits["recovery"] * (1.0 - efforts) - traits["drain"] * efforts)
        next_stamina = jnp.clip(state.stamina + stamina_delta, 0.0, 1.0)
        return DiskSumoV2State(
            pos=jnp.where(resolves, pos, state.pos),
            vel=jnp.where(resolves, vel, state.vel),
            prev_vel=jnp.where(resolves, state.vel, state.prev_vel),
            pending=jnp.where(resolves, jnp.zeros_like(force), force),
            pending_effort=jnp.where(resolves, jnp.zeros_like(effort), effort),
            archetype=state.archetype,
            stamina=jnp.where(resolves, next_stamina, state.stamina),
            result=jnp.where(resolves, result, state.result),
            done=jnp.where(resolves, done, state.done),
            turn=state.turn + 1,
        )

    # ---- physics ------------------------------------------------------------

    def _world_force(self, state: DiskSumoV2State, player, value: chex.Array):
        """`DiskSumo._world_force` with the archetype's thrust and exhausted floor."""
        own, opp = state.pos[player], state.pos[1 - player]
        local = self._space.box.clip(value).astype(state.pos.dtype)
        local = local / jnp.maximum(jnp.linalg.norm(local), 1.0)
        traits = self.traits(state.archetype)
        min_force = traits["min_force"][player]
        stamina_scale = min_force + (1.0 - min_force) * state.stamina[player]
        force = traits["force"][player] * stamina_scale * (self._frame(own, opp).T @ local)
        return force, jnp.linalg.norm(local)

    def _integrate(self, state: DiskSumoV2State, pos, vel, forces, efforts):
        """Per-disk mass, and drag raised by `brace` for a disk that is barely thrusting."""
        traits = self.traits(state.archetype)
        drag = traits["drag"] * (1.0 + traits["brace"] * (1.0 - efforts))
        return self._simulate_bodies(pos, vel, forces, traits["mass"], drag)

    def _simulate_bodies(self, pos, vel, forces, mass, drag):
        """`DiskSumo._simulate` with per-disk `(2,)` mass and drag coefficients."""
        h = self.dt / self.substeps
        never = self.substeps
        inv_mass = (1.0 / mass)[:, None]
        drag = drag[:, None]

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
            vel = vel + h * (forces - drag * vel + contact) * inv_mass
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
