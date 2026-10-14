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

**Optional stamina.** With `stamina_enabled`, full thrust cannot be sustained
indefinitely. Each disk starts with unit stamina; effort drains it and easing
off restores it. Available thrust falls linearly with stamina, down to
`stamina_min_force` at exhaustion. Both stamina levels are observable (own
first), but player 0's effort for the current simultaneous move remains hidden.
This makes a committed charge something an opponent can evade and punish rather
than a cost-free action that can be repeated for the whole bout.

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

**Optional observation delay.** With `observation_delay=d > 0` a player sees
the *opponent's* disk as it was `d` control steps ago -- position, velocity,
edge margin and stamina -- while their own disk and the clock stay current.
For the first `d` control steps that is the opponent's start. The opponent's
last `d` moves are then hidden for as long as they take to arrive, instead of
for a single control step, so a feint or a sidestep can no longer be answered
the moment it shows: whoever commits has to guess. Three things follow.

  * The egocentric frame is built from what the player sees, so its x-axis
    points at the opponent's *delayed* position, and the action is read in that
    same frame. "Charge" `(1, 0)` aims at where the opponent was; a frame on the
    true position would aim for the player and undo the delay. The frame swings
    round quickly when a player crosses the spot the opponent has just left;
    `egocentric=False` has no such artefact.
  * Contact is still felt at once, through the player's own velocity.
  * The observation is no longer a sufficient statistic: the player's own
    recent forces and the opponent's earlier sightings carry information that a
    memoryless policy does not get. `observation_frames` hands over the latter.

**Optional sighting memory.** `observation_frames=k > 1` shows the opponent's
last `k` sightings instead of one: the opponent `d`, `d + 1`, ..., `d + k - 1`
control steps ago. Nothing newer is revealed -- the delay is still `d` -- but a
run of sightings shows how the opponent has been turning, which one position
and velocity cannot. Each older sighting appends the opponent's position and
velocity (and stamina, when that is on) to the end of the observation, newest
first and all in the player's *current* frame, so the first 11 entries keep
their meaning. It works without a delay too, as plain frame stacking.

**Optional limited sensing.** `opponent_sensing` sets how much of the opponent
a player perceives at all. `"full"` (the default) is everything above.
`"bearing"` and `"none"` withdraw the opponent's position, velocity, edge
margin and stamina; what is left is the player's own disk, the clock and a
touch sensor -- the mean contact force on their own disk over the last control
step -- and, under `"bearing"`, a rough direction to the opponent. Without the
opponent's position a player can no longer be predicted from the state, only
from their policy, so a deterministic policy is one a best response can walk up
to and ambush; whoever wants not to be read has to randomize.
  * The egocentric frame can no longer point at the opponent, so it is the
    *radial* frame instead: x-axis from the ring centre out through the
    player's own disk, y its left-hand perpendicular. "Push towards the
    centre" is then the constant action `(-1, 0)`. `egocentric=False` keeps the
    world frame, as before.
  * The bearing is the angle to the opponent in that frame, plus a Gaussian
    error of `bearing_noise` radians, cut into `bearing_bins` equal sectors and
    shown one-hot. Sector 0 is centred on +x (opponent farther out than me),
    and they count anticlockwise, so with 4 bins: outward, left, inward,
    right. Without the error a change of sector would mark the moment the
    opponent crosses a sector boundary, an exact bearing; the error, drawn
    afresh every control step and the same for both decisions in it, blurs
    those crossings.
  * `observation_delay` delays the bearing, and `observation_frames > 1`
    appends the older sectors, each with the error drawn at its time and read
    in the current frame. Under `"none"` there is nothing to delay or
    remember, and both are rejected.
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

# How much of the opponent a player perceives; see "Optional limited sensing".
_SENSING = ("full", "bearing", "none")


@chex.dataclass(frozen=True)
class DiskSumoState:
    """Fixed-shape state of one `DiskSumo` bout.

    `turn` is the decision clock: control step `turn // 2` is being played and
    player `turn % 2` is to choose. Player 0's force and normalized effort for
    the live control step wait in `pending` / `pending_effort` until player 1
    has chosen theirs; neither is part of either observation.

    `past_pos`, `past_vel` and `past_stamina` hold the last
    `observation_delay + observation_frames - 1` control steps, newest first:
    index `k` is the state `k + 1` control steps ago, so the opponent is shown
    entry `observation_delay - 1` and, with more frames, the ones after it.
    They have length zero when there is neither a delay nor a memory.
    `past_bearing_error` is the same buffer for `bearing_error`.

    `touch` is the mean contact force each disk felt over the last control
    step, and `bearing_error` the angular error each player's bearing carries
    this control step. Both are only observed under limited `opponent_sensing`.

    `result` is `+1`/`-1` once somebody has been pushed out and `0` otherwise;
    `done` marks that the bout ended early, including by the simultaneous-exit
    draw, which leaves `result` at `0`.
    """

    pos: chex.Array  # (2, 2) float32, disk centres, ring centre at the origin
    vel: chex.Array  # (2, 2) float32
    pending: chex.Array  # (2,) float32, player 0's world-frame force this step
    pending_effort: chex.Array  # () float32, hidden player-0 effort in [0, 1]
    force_multiplier: chex.Array  # (2,) float32, per-bout actuator strengths
    stamina: chex.Array  # (2,) float32 in [0, 1]
    past_pos: chex.Array  # (history, 2, 2) float32
    past_vel: chex.Array  # (history, 2, 2) float32
    past_stamina: chex.Array  # (history, 2) float32
    touch: chex.Array  # (2, 2) float32, world-frame mean contact force on each disk
    bearing_error: chex.Array  # (2,) float32, radians, one per observing player
    past_bearing_error: chex.Array  # (history, 2) float32
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
        stamina_enabled: bool = False,
        stamina_drain_rate: float = 0.5,
        stamina_recovery_rate: float = 0.25,
        stamina_min_force: float = 0.25,
        drag: float = 1.0,
        stiffness: float = 100.0,
        contact_damping: float = 5.0,
        margin_weight: float = 0.0,
        shaping_weight: float = 0.0,
        observation_delay: int = 0,
        observation_frames: int = 1,
        opponent_sensing: str = "full",
        bearing_bins: int = 4,
        bearing_noise: float = 0.0,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if observation_delay < 0 or int(observation_delay) != observation_delay:
            raise ValueError(
                f"observation_delay must be a non-negative whole number of control steps, "
                f"got {observation_delay}"
            )
        if observation_frames < 1 or int(observation_frames) != observation_frames:
            raise ValueError(
                f"observation_frames must be a whole number of sightings, at least 1, "
                f"got {observation_frames}"
            )
        if opponent_sensing not in _SENSING:
            raise ValueError(
                f"opponent_sensing must be one of {_SENSING}, got {opponent_sensing!r}"
            )
        if bearing_bins < 2 or int(bearing_bins) != bearing_bins:
            raise ValueError(f"bearing_bins must be a whole number, at least 2, got {bearing_bins}")
        if bearing_noise < 0.0:
            raise ValueError(f"bearing_noise must be non-negative, got {bearing_noise}")
        if opponent_sensing == "none" and (observation_delay > 0 or observation_frames > 1):
            raise ValueError(
                "opponent_sensing='none' shows nothing of the opponent to delay or remember; "
                f"got observation_delay={observation_delay}, observation_frames={observation_frames}"
            )
        if substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}")
        for name, value in (("ring_radius", ring_radius), ("disk_radius", disk_radius),
                            ("dt", dt), ("mass", mass), ("max_force", max_force),
                            ("stiffness", stiffness)):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in (("start_jitter", start_jitter), ("drag", drag),
                            ("contact_damping", contact_damping),
                            ("stamina_drain_rate", stamina_drain_rate),
                            ("stamina_recovery_rate", stamina_recovery_rate)):
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
        if not 0.0 <= stamina_min_force <= 1.0:
            raise ValueError(
                f"stamina_min_force must be in [0, 1], got {stamina_min_force}"
            )

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
        self.stamina_enabled = bool(stamina_enabled)
        self.stamina_drain_rate = float(stamina_drain_rate)
        self.stamina_recovery_rate = float(stamina_recovery_rate)
        self.stamina_min_force = float(stamina_min_force)
        self.drag = float(drag)
        self.stiffness = float(stiffness)
        self.contact_damping = float(contact_damping)
        self.margin_weight = float(margin_weight)
        self.shaping_weight = float(shaping_weight)
        self.observation_delay = int(observation_delay)
        self.observation_frames = int(observation_frames)
        self.opponent_sensing = str(opponent_sensing)
        self.bearing_bins = int(bearing_bins)
        self.bearing_noise = float(bearing_noise)
        # Control steps of the past a state carries: up to the oldest sighting.
        self._history = self.observation_delay + self.observation_frames - 1
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
        # Disabled optional mechanics preserve old checkpoint input widths.
        # Strength and stamina must be visible so the policy can condition its
        # attacking/defending behavior on the current role and resources.
        if self.opponent_sensing != "full":
            # Own pos, vel and touch, own edge margin, time left; then the
            # sector one-hots, the newest and each older one; own stamina only.
            sectors = self.bearing_bins * (self.opponent_sensing == "bearing")
            return (8 + 2 * (self.force_asymmetry > 0.0) + self.stamina_enabled
                    + self.observation_frames * sectors)
        # Each older sighting is a position and a velocity, plus stamina when on.
        memory = (self.observation_frames - 1) * (4 + self.stamina_enabled)
        return 11 + 2 * (self.force_asymmetry > 0.0) + 2 * self.stamina_enabled + memory

    # ---- the game tree ------------------------------------------------------

    def initial_state(self, key: chex.PRNGKey) -> DiskSumoState:
        """Both disks at rest, placed by whichever start `random_start` selects."""
        pos = self._random_start(key) if self.random_start else self._facing_start(key)
        strong_player = jax.random.bernoulli(jax.random.fold_in(key, 1)).astype(jnp.int32)
        weak, strong = 1.0 - self.force_asymmetry, 1.0 + self.force_asymmetry
        force_multiplier = jnp.where(
            jnp.arange(2) == strong_player, strong, weak
        ).astype(jnp.float32)
        pos = pos.astype(jnp.float32)
        history = self._history
        return DiskSumoState(
            pos=pos,
            vel=jnp.zeros((2, 2), dtype=jnp.float32),
            pending=jnp.zeros((2,), dtype=jnp.float32),
            pending_effort=jnp.zeros((), dtype=jnp.float32),
            force_multiplier=force_multiplier,
            stamina=jnp.ones((2,), dtype=jnp.float32),
            # Before the bout there is only the start to have seen.
            past_pos=jnp.broadcast_to(pos, (history, 2, 2)),
            past_vel=jnp.zeros((history, 2, 2), dtype=jnp.float32),
            past_stamina=jnp.ones((history, 2), dtype=jnp.float32),
            touch=jnp.zeros((2, 2), dtype=jnp.float32),
            bearing_error=self._draw_bearing_error(jax.random.fold_in(key, 2)),
            past_bearing_error=jnp.zeros((history, 2), dtype=jnp.float32),
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
        """Base `(11,)`: positions, velocities, both edge margins, and time left.

        Observable force multipliers and stamina levels are appended, own
        first, when their corresponding optional mechanic is enabled.

        Positions are relative to the ring centre in units of `ring_radius`,
        velocities in units of the full-force speed, and all four vectors are
        rotated into `player`'s egocentric frame (see the module docstring). The
        edge margins `1 - |pos| / ring_radius` are derivable from the positions
        but are the single most decision-relevant quantity, so they are handed
        over directly.

        `state.pending` is deliberately not read: it is player 0's force for the
        live control step, and showing it to player 1 would turn the
        simultaneous move into a sequential one.

        With `observation_delay` everything about the opponent -- position,
        velocity, edge margin, stamina, and so the frame -- is
        `observation_delay` control steps old.

        With `observation_frames > 1` the older sightings follow at the very
        end, newest first: the opponent's position and velocity (then stamina,
        if enabled) one control step further back each time, scaled as above
        and rotated into the same, current frame.

        Under limited `opponent_sensing` the layout is `_sensed_observation`'s.
        """
        if self.opponent_sensing != "full":
            return self._sensed_observation(player, state)
        own, opp = player, 1 - player
        own_pos = state.pos[own]
        opp_pos, opp_vel, opp_stamina = self._seen_opponent(state, player)
        rotation = self._frame(own_pos, opp_pos)
        own_radius = jnp.linalg.norm(own_pos) / self.ring_radius
        opp_radius = jnp.linalg.norm(opp_pos) / self.ring_radius
        time_left = 1.0 - (state.turn // 2).astype(jnp.float32) / self.horizon
        observation = jnp.concatenate([
            rotation @ own_pos / self.ring_radius,
            rotation @ state.vel[own] / self._speed_scale,
            rotation @ opp_pos / self.ring_radius,
            rotation @ opp_vel / self._speed_scale,
            jnp.stack([1.0 - own_radius, 1.0 - opp_radius, time_left]),
        ])
        if self.force_asymmetry > 0.0:
            observation = jnp.concatenate([
                observation,
                jnp.stack([state.force_multiplier[own], state.force_multiplier[opp]]),
            ])
        if self.stamina_enabled:
            observation = jnp.concatenate([
                observation,
                jnp.stack([state.stamina[own], opp_stamina]),
            ])
        for frame in range(1, self.observation_frames):
            past_pos, past_vel, past_stamina = self._seen_opponent(state, player, frame)
            sighting = [
                rotation @ past_pos / self.ring_radius,
                rotation @ past_vel / self._speed_scale,
            ]
            if self.stamina_enabled:
                sighting.append(past_stamina[None])
            observation = jnp.concatenate([observation, *sighting])
        return observation

    def _sensed_observation(self, player: int, state: DiskSumoState) -> chex.Array:
        """`(8,)` and extras: what a player perceives without seeing the opponent.

        Own position, velocity and touch (the mean contact force on the own
        disk, in units of `max_force`), all in the radial frame; own edge
        margin; time left. Then both force multipliers and own stamina when
        those mechanics are on, and under `"bearing"` the opponent's sector,
        one-hot, newest first for each of the `observation_frames` sightings.
        """
        own = player
        own_pos = state.pos[own]
        rotation = self._radial_frame(own_pos)
        time_left = 1.0 - (state.turn // 2).astype(jnp.float32) / self.horizon
        observation = jnp.concatenate([
            rotation @ own_pos / self.ring_radius,
            rotation @ state.vel[own] / self._speed_scale,
            rotation @ state.touch[own] / self.max_force,
            jnp.stack([1.0 - jnp.linalg.norm(own_pos) / self.ring_radius, time_left]),
        ])
        if self.force_asymmetry > 0.0:
            observation = jnp.concatenate([
                observation,
                jnp.stack([state.force_multiplier[own], state.force_multiplier[1 - own]]),
            ])
        if self.stamina_enabled:
            observation = jnp.concatenate([observation, state.stamina[own][None]])
        if self.opponent_sensing == "bearing":
            observation = jnp.concatenate([
                observation,
                *(self._sector(state, player, frame) for frame in range(self.observation_frames)),
            ])
        return observation

    def _sector(self, state: DiskSumoState, player, frame: int = 0) -> chex.Array:
        """`(bearing_bins,)` one-hot: the sector `player` senses the opponent in.

        The opponent is the one `_seen_opponent` returns for `frame`, the error
        the one drawn at that time, and the frame the current radial one.
        """
        own_pos = state.pos[player]
        opp_pos = self._seen_opponent(state, player, frame)[0]
        age = self.observation_delay + frame
        error = (
            state.bearing_error[player] if age == 0
            else state.past_bearing_error[age - 1, player]
        )
        towards = self._radial_frame(own_pos) @ (opp_pos - own_pos)
        width = 2.0 * jnp.pi / self.bearing_bins
        angle = jnp.arctan2(towards[1], towards[0]) + error
        # Sector 0 is centred on +x; `%` wraps negative angles round.
        index = jnp.floor(angle / width + 0.5).astype(jnp.int32) % self.bearing_bins
        return jax.nn.one_hot(index, self.bearing_bins, dtype=jnp.float32)

    def _draw_bearing_error(self, key: chex.PRNGKey) -> chex.Array:
        """`(2,)`: this control step's bearing error for each observing player."""
        if self.opponent_sensing != "bearing" or self.bearing_noise == 0.0:
            return jnp.zeros((2,), dtype=jnp.float32)
        return self.bearing_noise * jax.random.normal(key, (2,), dtype=jnp.float32)

    def _seen_opponent(self, state: DiskSumoState, player, frame: int = 0):
        """`(pos, vel, stamina)` of `player`'s opponent as `player` sees it.

        `frame` 0 is the newest sighting, `observation_delay` control steps
        old -- the live state when there is no delay -- and each further frame
        is one control step older.
        """
        opp = 1 - player
        age = self.observation_delay + frame
        if age == 0:
            return state.pos[opp], state.vel[opp], state.stamina[opp]
        return (
            state.past_pos[age - 1, opp], state.past_vel[age - 1, opp],
            state.past_stamina[age - 1, opp],
        )

    def _remember(self, past: chex.Array, current: chex.Array) -> chex.Array:
        """Shift `current` into the front of a history buffer, dropping its oldest entry."""
        return jnp.concatenate([current[None], past])[: self._history]

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
        # The physics is deterministic; besides the start, the only chance is
        # the bearing error, drawn afresh for the next control step.
        player = state.turn % 2
        force, effort = self._world_force(state, player, action.value)
        efforts = jnp.stack([state.pending_effort, effort])

        # Player 0 only parks their force; player 1's choice closes the control
        # step, and both branches are computed because `player` is traced.
        pos, vel, result, done, touch = self._integrate(
            state, state.pos, state.vel, jnp.stack([state.pending, force]), efforts
        )
        resolves = player == 1
        stamina_delta = self.dt * (
            self.stamina_recovery_rate * (1.0 - efforts)
            - self.stamina_drain_rate * efforts
        )
        next_stamina = jnp.where(
            self.stamina_enabled,
            jnp.clip(state.stamina + stamina_delta, 0.0, 1.0),
            state.stamina,
        )
        return DiskSumoState(
            pos=jnp.where(resolves, pos, state.pos),
            vel=jnp.where(resolves, vel, state.vel),
            pending=jnp.where(resolves, jnp.zeros_like(force), force),
            pending_effort=jnp.where(resolves, jnp.zeros_like(effort), effort),
            force_multiplier=state.force_multiplier,
            stamina=jnp.where(resolves, next_stamina, state.stamina),
            past_pos=jnp.where(
                resolves, self._remember(state.past_pos, state.pos), state.past_pos
            ),
            past_vel=jnp.where(
                resolves, self._remember(state.past_vel, state.vel), state.past_vel
            ),
            past_stamina=jnp.where(
                resolves, self._remember(state.past_stamina, state.stamina), state.past_stamina
            ),
            touch=jnp.where(resolves, touch, state.touch),
            bearing_error=jnp.where(
                resolves, self._draw_bearing_error(key), state.bearing_error
            ),
            past_bearing_error=jnp.where(
                resolves,
                self._remember(state.past_bearing_error, state.bearing_error),
                state.past_bearing_error,
            ),
            result=jnp.where(resolves, result, state.result),
            done=jnp.where(resolves, done, state.done),
            turn=state.turn + 1,
        )

    # ---- physics ------------------------------------------------------------

    def _world_force(self, state: DiskSumoState, player, value: chex.Array):
        """`(force, effort)`: the world-frame force `player`'s raw action means in `state`.

        The action is clipped to the box and projected onto the unit disk (its
        norm is the `effort`), rotated out of the egocentric frame, and scaled by
        the actuator multiplier and stamina. Shared with `scripts/render_disk_sumo.py`,
        which has to recompute player 1's force: it is never stored in a state.

        The frame is the one `observation` uses, so under `observation_delay`
        its x-axis points at the opponent's delayed position, not the true one,
        and under limited `opponent_sensing` it is the radial frame.
        """
        # Keep the state's dtype: a float64 action (x64 on) would otherwise
        # promote `pending`, `pos` and `vel`, and break the scans that carry them.
        local = self._space.box.clip(value).astype(state.pos.dtype)
        local = local / jnp.maximum(jnp.linalg.norm(local), 1.0)
        stamina_scale = jnp.where(
            self.stamina_enabled,
            self.stamina_min_force + (1.0 - self.stamina_min_force) * state.stamina[player],
            1.0,
        )
        force = self.max_force * state.force_multiplier[player] * stamina_scale * (
            self._player_frame(state, player).T @ local
        )
        return force, jnp.linalg.norm(local)

    def _player_frame(self, state: DiskSumoState, player) -> chex.Array:
        """`(2, 2)`: the frame `player` observes and acts in, whichever sensing is on."""
        own = state.pos[player]
        if self.opponent_sensing != "full":
            return self._radial_frame(own)
        return self._frame(own, self._seen_opponent(state, player)[0])

    def _integrate(self, state: DiskSumoState, pos, vel, forces, efforts):
        """Advance one control step from `pos, vel`; return `(pos, vel, result, done, touch)`.

        `state` supplies the bodies and `efforts` is `(2,)`. Both disks are
        identical here, so this is `_simulate`; subclasses with per-disk bodies
        override it.
        """
        del state, efforts
        return self._simulate(pos, vel, forces)

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

    def _radial_frame(self, own: chex.Array) -> chex.Array:
        """`(2, 2)`: x outward from the ring centre through `own`; the identity unless `egocentric`.

        At the very centre the direction is undefined and the world x-axis
        stands in.
        """
        if not self.egocentric:
            return jnp.eye(2, dtype=jnp.float32)
        radius = jnp.linalg.norm(own)
        outward = jnp.where(
            radius > _EPS,
            own / jnp.maximum(radius, _EPS),
            jnp.array([1.0, 0.0], dtype=own.dtype),
        )
        return jnp.stack([outward, jnp.stack([-outward[1], outward[0]])])

    def _simulate(self, pos: chex.Array, vel: chex.Array, forces: chex.Array):
        """Advance one control step; return `(pos, vel, result, done, touch)`.

        Semi-implicit Euler over `substeps` substeps of `dt / substeps`. The
        contact is a spring-damper along the line of centres, active only while
        the disks overlap and never pulling them together. A disk is out once
        its centre is beyond the ring; the substep at which each disk first went
        out is tracked, so a simultaneous exit is settled by who left first.
        `touch` is the `(2, 2)` contact force on each disk, averaged over the
        substeps.
        """
        h = self.dt / self.substeps
        never = self.substeps

        def body(i, carry):
            pos, vel, exit_at, touch = carry
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
            return pos, vel, exit_at, touch + contact

        exit_at = jnp.full((2,), never, dtype=jnp.int32)
        pos, vel, exit_at, touch = jax.lax.fori_loop(
            0, self.substeps, body, (pos, vel, exit_at, jnp.zeros_like(vel))
        )

        out_0, out_1 = exit_at[0] < never, exit_at[1] < never
        # `exit_at` is `never` for a disk still inside, so one comparison covers
        # both "only one left" and "both left, one first".
        result = jnp.where(
            exit_at[0] < exit_at[1], -1.0, jnp.where(exit_at[1] < exit_at[0], 1.0, 0.0)
        ).astype(jnp.float32)
        return pos, vel, result, out_0 | out_1, touch / self.substeps
