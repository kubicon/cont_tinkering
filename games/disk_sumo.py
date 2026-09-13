"""Two-disk sumo: push the opponent out of a circular ring, continuous 2-D forces.

A physics-flavoured stand-in for RoboSumo / sumo-ants that needs no physics
engine. Two rigid disks of equal mass start facing each other in a ring of
radius `ring_radius`; each control step both players choose a force, the disks
are integrated for `substeps` semi-implicit Euler steps under that force, linear
drag and a penalty-spring contact between them, and the first disk whose
*centre* leaves the ring loses. Everything is `jnp`, so the rollout `jit`s and
`vmap`s exactly like the card games do.

**Where a bout starts.** By default the disks face each other across the
centre, `start_distance` apart, on an axis that is uniform on the circle -- a
symmetric position, which is what makes a symmetric pair of policies stalemate
there. With `random_start` both centres are instead drawn uniformly in the ring
every bout, so a policy has to play from a disk hard against the rim, from the
middle, and from everything between, and no two bouts open alike.

**Optional force asymmetry.** `force_asymmetry=a` gives one disk a force
multiplier `1 + a` and the other `1 - a`. The strong disk is chosen uniformly
at the start of every bout, so neither seat has a systematic advantage. When
enabled, both multipliers are appended to each player's observation (own
first), allowing one policy to learn both attacking and defending roles.

**Why every control step is two decisions.** Sumo is a simultaneous-move game,
and `games.sequential` is turn-taking. The standard extensive-form encoding is
used, as in `games.sequential_blotto`: player 0 chooses a force, it is parked in
`DiskSumoState.pending` without touching the physics, player 1 chooses a force
*without seeing it* (their observation never reads `pending`), and only then is
the physics advanced. The horizon in decisions is therefore `2 * horizon`.

**Why the action lives in an egocentric frame.** With `egocentric=True` (the
default) a player's observation *and* action are expressed in a frame whose
x-axis points from their disk to the opponent's. "Charge" is then the constant
action `(1, 0)` wherever on the ring the bout happens, instead of a function
of the two positions the network would first have to learn to compute. This is
the same move as Blotto's bid-as-a-fraction: a reparameterization of the policy
the game performs for it, not a change of the game. The frame is right-handed
for both players, so the two are mirror images of each other and a symmetric
state gives both of them the identical observation.

The action box is `[-1, 1]^2`, projected onto the unit disk before scaling by
`max_force` so a diagonal push is not `sqrt(2)` times stronger than a straight
one.

**Payoff.** `+1` if player 1 is pushed out, `-1` if player 0 is. If both leave
the ring during the same control step the one who left at the earlier substep
loses; leaving at the very same substep is a draw. A bout that reaches the time
limit (or ends in that draw) pays `margin_weight * (d_1 - d_0) / ring_radius`,
where `d_p` is disk `p`'s distance from the centre -- zero-sum, in
`[-margin_weight, margin_weight]`, and a tie-breaker that rewards holding the
middle. `margin_weight=0` is pure sumo, where a stalemate is worth exactly
nothing; a positive value gives a policy that never gets anyone out something
to climb, much as `sharpness` does in Blotto. Keep it below 1 so that a win
always outranks any timeout.

**Dense reward.** `shaping_weight` adds a per-transition reward
`shaping_weight * (phi(s') - phi(s))` with the potential
`phi(s) = clip((d_1 - d_0) / ring_radius, -1, 1)`: player 0 is paid, as it
happens, for gaining ground on the opponent relative to the centre. Undiscounted,
the shaping telescopes to `shaping_weight * (phi(s_T) - phi(s_0))`, so its only
effect on the game is a terminal bonus of the same form as the timeout margin
(plus a start-state constant that no strategy affects) -- what it changes is
*when* that signal arrives, which is what a bootstrapping advantage estimator
(`training.vtrace`) can exploit and a Monte Carlo one cannot. Unlike the margin
it also applies to decided bouts, so a win is worth
`1 + shaping_weight * (phi(s_T) - phi(s_0))`.

**Information.** Each player sees the full physical state (both positions and
velocities) and the clock -- this is a stochastic game with simultaneous moves,
not a game of hidden cards, and the only thing withheld is the opponent's
current force. Past actions influence the future only through the state they
produced, so observing the state is also what perfect recall asks for here.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .sequential import TERMINAL, SequentialZeroSumGame
from .spaces import HybridAction, HybridSpace, hybrid

# A force is a pure continuous action: no parameterless choice alongside it.
NUM_ATOMS = 0

# Placement attempts `random_start` draws before falling back to the facing
# start. The gap it asks for is small next to the ring, so one attempt almost
# always suffices; 16 makes the fallback unreachable in practice.
_START_ATTEMPTS = 16

# Guards the contact normal and the egocentric frame against coincident centres
# (unreachable in play -- the contact spring keeps the disks apart -- but the
# computation must not produce a NaN on the way to being discarded).
_EPS = 1e-6


@chex.dataclass(frozen=True)
class DiskSumoState:
    """Fixed-shape state of one `DiskSumo` bout.

    `turn` is the decision clock: control step `turn // 2` is being played and
    player `turn % 2` is to choose. Player 0's force for the live control step
    waits in `pending` (in world coordinates, already scaled) until player 1 has
    chosen theirs; it is never part of either observation.

    `result` is `+1`/`-1` once somebody has been pushed out and `0` otherwise;
    `done` marks that the bout ended early, including by the simultaneous-exit
    draw, which leaves `result` at `0`.
    """

    pos: chex.Array  # (2, 2) float32, disk centres, ring centre at the origin
    vel: chex.Array  # (2, 2) float32
    pending: chex.Array  # (2,) float32, player 0's world-frame force this step
    force_multiplier: chex.Array  # (2,) float32, per-bout actuator strengths
    result: chex.Array  # () float32 in {-1, 0, +1}, from player 0's side
    done: chex.Array  # () bool, the bout ended before the time limit
    turn: chex.Array  # () int32 in [0, 2 * horizon]


class DiskSumo(SequentialZeroSumGame):
    """Two disks in a ring, each trying to push the other out. See the module docstring."""

    def __init__(
        self,
        horizon: int = 50,
        ring_radius: float = 1.0,
        disk_radius: float = 0.15,
        start_distance: float = 0.6,
        start_jitter: float = 0.05,
        random_orientation: bool = True,
        random_start: bool = False,
        egocentric: bool = True,
        dt: float = 0.1,
        substeps: int = 10,
        mass: float = 1.0,
        max_force: float = 1.0,
        force_asymmetry: float = 0.0,
        drag: float = 1.0,
        stiffness: float = 100.0,
        contact_damping: float = 5.0,
        margin_weight: float = 0.0,
        shaping_weight: float = 0.0,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}")
        for name, value in (("ring_radius", ring_radius), ("disk_radius", disk_radius),
                            ("dt", dt), ("mass", mass), ("max_force", max_force),
                            ("stiffness", stiffness)):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in (("start_jitter", start_jitter), ("drag", drag),
                            ("contact_damping", contact_damping)):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if start_distance < 2.0 * disk_radius:
            raise ValueError(
                f"start_distance ({start_distance}) must be at least 2 * disk_radius "
                f"({2.0 * disk_radius}), or the disks start overlapping"
            )
        # The farthest a jittered start can put a centre from the origin.
        if start_distance / 2.0 + start_jitter * 2.0 ** 0.5 >= ring_radius:
            raise ValueError(
                f"start_distance / 2 + jitter must stay inside the ring (radius {ring_radius}), "
                f"got start_distance={start_distance}, start_jitter={start_jitter}"
            )
        if shaping_weight < 0.0:
            raise ValueError(f"shaping_weight must be non-negative, got {shaping_weight}")
        if not 0.0 <= margin_weight <= 1.0:
            raise ValueError(
                f"margin_weight must be in [0, 1] so a win outranks any timeout, got {margin_weight}"
            )
        if not 0.0 <= force_asymmetry < 1.0:
            raise ValueError(f"force_asymmetry must be in [0, 1), got {force_asymmetry}")

        self.horizon = int(horizon)
        self.ring_radius = float(ring_radius)
        self.disk_radius = float(disk_radius)
        self.start_distance = float(start_distance)
        self.start_jitter = float(start_jitter)
        self.random_orientation = bool(random_orientation)
        self.random_start = bool(random_start)
        self.egocentric = bool(egocentric)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.mass = float(mass)
        self.max_force = float(max_force)
        self.force_asymmetry = float(force_asymmetry)
        self.drag = float(drag)
        self.stiffness = float(stiffness)
        self.contact_damping = float(contact_damping)
        self.margin_weight = float(margin_weight)
        self.shaping_weight = float(shaping_weight)
        # Speed a lone disk settles at under full force: the natural velocity
        # scale for the observation. Drag-free physics falls back to what full
        # force reaches in one control step.
        self._speed_scale = max_force / drag if drag > 0.0 else max_force * dt / mass
        self._space = hybrid(NUM_ATOMS, [-1.0, -1.0], [1.0, 1.0])

    # ---- shape/static information ------------------------------------------

    @property
    def max_steps(self) -> int:
        """One decision per player per control step."""
        return 2 * self.horizon

    def action_space(self, player: int) -> HybridSpace:
        return self._space

    def obs_dim(self, player: int) -> int:
        # Keep symmetric-game checkpoints compatible. The asymmetric role must
        # be visible, or strength becomes hidden chance instead of something a
        # policy can condition its attacking/defending behavior on.
        return 13 if self.force_asymmetry > 0.0 else 11

    # ---- the game tree ------------------------------------------------------

    def initial_state(self, key: chex.PRNGKey) -> DiskSumoState:
        """Both disks at rest, placed by whichever start `random_start` selects."""
        pos = self._random_start(key) if self.random_start else self._facing_start(key)
        strong_player = jax.random.bernoulli(jax.random.fold_in(key, 1)).astype(jnp.int32)
        weak, strong = 1.0 - self.force_asymmetry, 1.0 + self.force_asymmetry
        force_multiplier = jnp.where(
            jnp.arange(2) == strong_player, strong, weak
        ).astype(jnp.float32)
        return DiskSumoState(
            pos=pos.astype(jnp.float32),
            vel=jnp.zeros((2, 2), dtype=jnp.float32),
            pending=jnp.zeros((2,), dtype=jnp.float32),
            force_multiplier=force_multiplier,
            result=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=bool),
            turn=jnp.zeros((), dtype=jnp.int32),
        )

    def _facing_start(self, key: chex.PRNGKey) -> chex.Array:
        """`(2, 2)`: the disks face each other across the centre, `start_distance` apart.

        The axis they face along is uniform on the circle when
        `random_orientation`, and each centre is jittered by up to
        `start_jitter` per coordinate.
        """
        angle_key, jitter_key = jax.random.split(key)
        angle = (
            jax.random.uniform(angle_key, (), maxval=2.0 * jnp.pi)
            if self.random_orientation else jnp.zeros(())
        )
        axis = jnp.stack([jnp.cos(angle), jnp.sin(angle)])
        half = 0.5 * self.start_distance
        jitter = jax.random.uniform(
            jitter_key, (2, 2), minval=-self.start_jitter, maxval=self.start_jitter
        )
        return jnp.stack([-half * axis, half * axis]) + jitter

    def _random_start(self, key: chex.PRNGKey) -> chex.Array:
        """`(2, 2)`: both centres uniform in the ring, far enough apart not to start in contact.

        Uniform on the disk of radius `ring_radius - disk_radius` -- `sqrt` of a
        uniform radius, or the draw crowds the centre -- so a bout can open
        anywhere, including hard against the rim with the opponent between the
        disk and the middle. That is the point of it: the facing start is
        symmetric, and a policy trained only on symmetric starts has never seen
        the position it most needs to play well from. It also breaks the
        stalemate the symmetric start invites, where two equal disks push
        head-on and neither can gain ground.

        `_START_ATTEMPTS` independent pairs are drawn and the first with a legal
        gap is kept, which is uniform conditioned on that gap and -- unlike a
        rejection loop -- one fixed shape, so this still `jit`s and `vmap`s.
        The facing start is the fallback for the vanishingly unlikely draw where
        every pair overlaps.
        """
        radius = self.ring_radius - self.disk_radius
        radius_key, angle_key, fallback_key = jax.random.split(key, 3)
        # sqrt of a uniform is what makes the *area* uniform rather than the radius.
        radii = radius * jnp.sqrt(jax.random.uniform(radius_key, (_START_ATTEMPTS, 2)))
        angles = jax.random.uniform(angle_key, (_START_ATTEMPTS, 2), maxval=2.0 * jnp.pi)
        candidates = radii[..., None] * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        legal = jnp.linalg.norm(candidates[:, 1] - candidates[:, 0], axis=-1) > 2.0 * self.disk_radius
        first = jnp.argmax(legal)  # the first legal pair, or pair 0 when there is none
        return jnp.where(jnp.any(legal), candidates[first], self._facing_start(fallback_key))

    def current_player(self, state: DiskSumoState) -> chex.Array:
        over = state.done | (state.turn >= self.max_steps)
        return jnp.where(over, TERMINAL, state.turn % 2).astype(jnp.int32)

    def observation(self, player: int, state: DiskSumoState) -> chex.Array:
        """`(11,)`: own and opponent position and velocity, both edge margins, time left.

        Positions are relative to the ring centre in units of `ring_radius`,
        velocities in units of the full-force speed, and all four vectors are
        rotated into `player`'s egocentric frame (see the module docstring). The
        edge margins `1 - |pos| / ring_radius` are derivable from the positions
        but are the single most decision-relevant quantity, so they are handed
        over directly.

        `state.pending` is deliberately not read: it is player 0's force for the
        live control step, and showing it to player 1 would turn the
        simultaneous move into a sequential one.
        """
        own, opp = player, 1 - player
        rotation = self._frame(state.pos[own], state.pos[opp])
        radii = jnp.linalg.norm(state.pos, axis=-1) / self.ring_radius
        time_left = 1.0 - (state.turn // 2).astype(jnp.float32) / self.horizon
        observation = jnp.concatenate([
            rotation @ state.pos[own] / self.ring_radius,
            rotation @ state.vel[own] / self._speed_scale,
            rotation @ state.pos[opp] / self.ring_radius,
            rotation @ state.vel[opp] / self._speed_scale,
            jnp.stack([1.0 - radii[own], 1.0 - radii[opp], time_left]),
        ])
        if self.force_asymmetry > 0.0:
            observation = jnp.concatenate([
                observation,
                jnp.stack([state.force_multiplier[own], state.force_multiplier[opp]]),
            ])
        return observation

    def action_mask(self, player: int, state: DiskSumoState) -> chex.Array:
        """`(1,)` all-`True`: pushing is the only kind, and every force is legal."""
        del player, state
        return jnp.ones((self.num_kinds(0),), dtype=bool)

    def payoff(self, state: DiskSumoState) -> chex.Array:
        return jnp.where(state.result != 0.0, state.result, self.margin_weight * self.potential(state))

    def reward(self, state: DiskSumoState, action: HybridAction, next_state: DiskSumoState) -> chex.Array:
        """Potential-based shaping, `shaping_weight * (phi(s') - phi(s))`; see the module docstring.

        Zero on player 0's half of a control step, which moves nothing.
        """
        del action
        return self.shaping_weight * (self.potential(next_state) - self.potential(state))

    def potential(self, state: DiskSumoState) -> chex.Array:
        """`phi(s)`: how much farther from the centre player 1 is than player 0, in ring radii."""
        radii = jnp.linalg.norm(state.pos, axis=-1)
        return jnp.clip((radii[1] - radii[0]) / self.ring_radius, -1.0, 1.0)

    def _step(self, state: DiskSumoState, action: HybridAction, key: chex.PRNGKey) -> DiskSumoState:
        del key  # the physics is deterministic; the only chance is the start
        player = state.turn % 2
        own = state.pos[player]
        opp = state.pos[1 - player]

        # Keep the state's dtype: a float64 action (x64 on) would otherwise
        # promote `pending`, `pos` and `vel`, and break the scans that carry them.
        local = self._space.box.clip(action.value).astype(state.pos.dtype)
        local = local / jnp.maximum(jnp.linalg.norm(local), 1.0)
        force = self.max_force * state.force_multiplier[player] * (
            self._frame(own, opp).T @ local
        )

        # Player 0 only parks their force; player 1's choice closes the control
        # step, and both branches are computed because `player` is traced.
        pos, vel, result, done = self._simulate(
            state.pos, state.vel, jnp.stack([state.pending, force])
        )
        resolves = player == 1
        return DiskSumoState(
            pos=jnp.where(resolves, pos, state.pos),
            vel=jnp.where(resolves, vel, state.vel),
            pending=jnp.where(resolves, jnp.zeros_like(force), force),
            force_multiplier=state.force_multiplier,
            result=jnp.where(resolves, result, state.result),
            done=jnp.where(resolves, done, state.done),
            turn=state.turn + 1,
        )

    # ---- physics ------------------------------------------------------------

    def _frame(self, own: chex.Array, opp: chex.Array) -> chex.Array:
        """`(2, 2)` rotation taking world vectors into the egocentric frame.

        Rows are the frame's axes: x towards the opponent, y its left-hand
        perpendicular. The identity when `egocentric` is off.
        """
        if not self.egocentric:
            return jnp.eye(2, dtype=jnp.float32)
        delta = opp - own
        distance = jnp.linalg.norm(delta)
        forward = jnp.where(
            distance > _EPS,
            delta / jnp.maximum(distance, _EPS),
            jnp.array([1.0, 0.0], dtype=delta.dtype),
        )
        left = jnp.stack([-forward[1], forward[0]])
        return jnp.stack([forward, left])

    def _simulate(self, pos: chex.Array, vel: chex.Array, forces: chex.Array):
        """Advance one control step; return `(pos, vel, result, done)`.

        Semi-implicit Euler over `substeps` substeps of `dt / substeps`. The
        contact is a spring-damper along the line of centres, active only while
        the disks overlap and never pulling them together. A disk is out once
        its centre is beyond the ring; the substep at which each disk first went
        out is tracked, so a simultaneous exit is settled by who left first.
        """
        h = self.dt / self.substeps
        never = self.substeps

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
            vel = vel + h * (forces - self.drag * vel + contact) / self.mass
            pos = pos + h * vel
            out = jnp.linalg.norm(pos, axis=-1) > self.ring_radius
            exit_at = jnp.where(out & (exit_at == never), i, exit_at)
            return pos, vel, exit_at

        exit_at = jnp.full((2,), never, dtype=jnp.int32)
        pos, vel, exit_at = jax.lax.fori_loop(0, self.substeps, body, (pos, vel, exit_at))

        out_0, out_1 = exit_at[0] < never, exit_at[1] < never
        # `exit_at` is `never` for a disk still inside, so one comparison covers
        # both "only one left" and "both left, one first".
        result = jnp.where(
            exit_at[0] < exit_at[1], -1.0, jnp.where(exit_at[1] < exit_at[0], 1.0, 0.0)
        ).astype(jnp.float32)
        return pos, vel, result, out_0 | out_1
