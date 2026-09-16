"""Sumo on MuJoCo/MJX physics: pucks (`MjxSumo`) and ants (`MjxAntSumo`).

`games.disk_sumo` integrates its own semi-implicit Euler and models contact as a
penalty spring. This module plays sumo on MuJoCo's constraint solver via MJX;
both its portable JAX implementation and its NVIDIA-optimized Warp
implementation participate in the same `jit`/`vmap` training pipeline.

Two games live here, sharing `MjxSumoBase` -- every rule that is not about the
shape of a body:

  * `MjxSumo`: two pucks, no gravity, no floor, two slide joints each, one
    contact. It plays the *same game* as `games.disk_sumo` -- same
    11-dimensional observation, same egocentric force in `[-1, 1]^2`, same
    payoff -- so `DiskSumo` is the reference implementation it is tested
    against, and a bug shows up as a rule the two disagree about rather than as
    a training run that quietly fails to learn.

  * `MjxAntSumo`: two of the classic MuJoCo ants on a floor under gravity,
    pushing each other out of the ring -- the setup of Bansal et al. (2018) and
    Al-Shedivat et al. (2018), whose agents are torque-driven and can also lose
    by being knocked onto their backs. The action is 8 joint torques, the
    observation is 65-dimensional, and neither is a force in a ring frame.

**What is shared, and why the split falls where it does.** The sumo *rules* --
who acts when, that a control step is two decisions, that leaving the ring
loses, the timeout margin, the potential shaping, the egocentric frame -- do not
depend on whether an agent is a puck or an ant. The rules live in
`MjxSumoBase`; a subclass supplies the model, the start configuration, where the
torsos are, what counts as eliminated, and how an action becomes `ctrl`.

**The ring is a rule, not a geom.** An agent is out once its *centre* (the puck,
or the ant's torso) passes `ring_radius`, checked once per physics substep, so
the substep at which each agent first went settles a simultaneous exit. Nothing
in either MuJoCo model knows about the ring, and neither has an edge to fall
off.

**Why every control step is two decisions.** Sumo is a simultaneous-move game
and `games.sequential` is turn-taking, so the standard extensive-form encoding
is used, as in `games.disk_sumo` and `games.sequential_blotto`: player 0 chooses
a control, it is parked in `MjxSumoState.pending` without touching the physics,
player 1 chooses theirs *without seeing it* (their observation never reads
`pending`), and only then is the physics advanced. The horizon in decisions is
therefore `2 * horizon`.

**Information.** Each player sees the full physical state of both agents and the
clock. This is a stochastic game with simultaneous moves, not a game of hidden
cards; the only thing withheld is the opponent's control for the live step.
"""

from __future__ import annotations

import abc
import math
import pathlib
from typing import Any

import chex
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from .sequential import TERMINAL, SequentialZeroSumGame
from .spaces import HybridAction, HybridSpace, hybrid

# A control is a pure continuous action: no parameterless choice alongside it.
NUM_ATOMS = 0

# Guards the contact normal and the egocentric frame against coincident centres.
_EPS = 1e-6

_ASSETS = pathlib.Path(__file__).parent / "assets"

# MJX's `Data` is a pytree, but not one of ours; typing it loosely keeps this
# module's annotations honest without pretending to MJX's internal structure.
MjxData = Any

# Collision bitmasks. Two geoms collide when one's `contype` shares a bit with
# the other's `conaffinity`, so giving each agent its own bit -- and affinity
# for the floor and the *other* agent, never for itself -- is what makes two
# ants push each other while neither trips over its own legs.
_FLOOR_BIT, _AGENT_BITS = 1, (2, 4)


@chex.dataclass(frozen=True)
class MjxSumoState:
    """Fixed-shape state of one bout, for either game in this module.

    `data` is the MJX simulator state -- a pytree of fixed-shape arrays, which
    is the only thing `games.sequential` asks of it. Everything else is the
    game's own bookkeeping and matches `games.disk_sumo.DiskSumoState` field for
    field: `turn` is the decision clock (control step `turn // 2`, player
    `turn % 2` to choose), `pending` parks player 0's control until player 1 has
    chosen, `result` is `+1`/`-1`/`0` from player 0's side, and `done` marks an
    early finish.
    """

    data: MjxData
    pending: chex.Array  # (ctrl width of one agent,) float32, player 0's live control
    result: chex.Array  # () float32 in {-1, 0, +1}, from player 0's side
    done: chex.Array  # () bool, the bout ended before the time limit
    turn: chex.Array  # () int32 in [0, 2 * horizon]


class MjxSumoBase(SequentialZeroSumGame):
    """The sumo rules, for any pair of identical MuJoCo bodies.

    Subclasses build the model and answer five body-shaped questions:
    `_build_model`, `_initial_data`, `_positions`, `_eliminated` and `_control`
    (plus the usual `observation` / `obs_dim` / `action_space`). Everything
    else here -- the decision clock, the pending-control encoding of a
    simultaneous move, the per-substep elimination bookkeeping, the payoff, the
    timeout margin and the shaping -- is the same game whatever is doing the
    pushing.
    """

    def _init_rules(
        self,
        horizon: int,
        ring_radius: float,
        start_distance: float,
        start_jitter: float,
        random_orientation: bool,
        dt: float,
        substeps: int,
        solver_iterations: int,
        solver_ls_iterations: int,
        margin_weight: float,
        shaping_weight: float,
    ) -> None:
        """Validate and store the body-independent parameters. Call first from `__init__`."""
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}")
        for name, value in (("ring_radius", ring_radius), ("dt", dt)):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in (("solver_iterations", solver_iterations),
                            ("solver_ls_iterations", solver_ls_iterations)):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if start_jitter < 0.0:
            raise ValueError(f"start_jitter must be non-negative, got {start_jitter}")
        if start_distance <= 0.0:
            raise ValueError(f"start_distance must be positive, got {start_distance}")
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

        self.horizon = int(horizon)
        self.ring_radius = float(ring_radius)
        self.start_distance = float(start_distance)
        self.start_jitter = float(start_jitter)
        self.random_orientation = bool(random_orientation)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.solver_iterations = int(solver_iterations)
        self.solver_ls_iterations = int(solver_ls_iterations)
        self.margin_weight = float(margin_weight)
        self.shaping_weight = float(shaping_weight)

    def _init_model(
        self,
        physics_backend: str,
        warp_naconmax: int | None,
        warp_njmax: int | None,
        warp_graph_mode: str,
        warp_warn_solver_iterations: bool = False,
    ) -> None:
        """Compile the subclass's model onto the selected MJX backend."""
        if physics_backend not in ("auto", "jax", "warp"):
            raise ValueError(
                "physics_backend must be 'auto', 'jax' or 'warp', got "
                f"{physics_backend!r}"
            )
        if warp_graph_mode not in ("auto", "warp", "warp_staged", "warp_staged_ex"):
            raise ValueError(
                "warp_graph_mode must be 'auto', 'warp', 'warp_staged' or "
                f"'warp_staged_ex', got {warp_graph_mode!r}"
            )

        backend = physics_backend
        if backend == "auto":
            # Warp is NVIDIA-only for performance. Keep local CPU smoke tests
            # and non-NVIDIA accelerators on the portable JAX implementation.
            has_nvidia_gpu = any(
                device.platform == "gpu" and "nvidia" in device.device_kind.lower()
                for device in jax.devices()
            )
            backend = "warp" if has_nvidia_gpu else "jax"

        if backend == "warp":
            from mujoco.mjx import warp as mjx_warp

            if not mjx_warp.WARP_INSTALLED:
                raise RuntimeError(
                    "MJX-Warp was selected but warp-lang is unavailable; install the "
                    "project with its mujoco-mjx[warp] dependency"
                )
            for name, value in (
                ("warp_naconmax", warp_naconmax),
                ("warp_njmax", warp_njmax),
            ):
                if value is None or value < 1:
                    raise ValueError(
                        f"physics_backend {physics_backend!r} resolved to Warp, so {name} "
                        f"must be a positive integer, got {value}"
                    )
            graph_mode = None
            if warp_graph_mode != "auto":
                graph_mode = getattr(
                    mjx_warp.types.GraphMode, warp_graph_mode.upper()
                )
        else:
            graph_mode = None

        self.physics_backend = backend
        self.warp_naconmax = warp_naconmax
        self.warp_njmax = warp_njmax
        self.warp_graph_mode = warp_graph_mode
        self.warp_warn_solver_iterations = bool(warp_warn_solver_iterations)
        self._mj_model = self._build_model()
        self._model = mjx.put_model(
            self._mj_model, impl=backend, graph_mode=graph_mode
        )
        if backend == "warp" and not warp_warn_solver_iterations:
            # Warp prints a warning from inside the kernel for every world whose
            # (line search) solver hits its iteration cap -- with a few thousand
            # worlds that is millions of lines per run and a multi-GB log. The
            # capped iterations are a deliberate speed trade-off here, so drop
            # just those two bits; buffer overflows (dropped contacts) still warn.
            from mujoco.mjx.third_party.mujoco_warp import OverflowType

            opt = self._model.opt
            warn = opt._impl.warn_overflow & ~(
                OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS
            )
            self._model = self._model.replace(
                opt=opt.replace(_impl=opt._impl.replace(warn_overflow=int(warn)))
            )
        if backend == "warp":
            # Warp allocates contact storage across every later-vmapped world;
            # unlike the JAX backend it requires these capacities up front and
            # make_data must receive the original MjModel.
            self._init_data = mjx.make_data(
                self._mj_model,
                impl="warp",
                naconmax=warp_naconmax,
                njmax=warp_njmax,
            )
        else:
            self._init_data = mjx.make_data(self._model, impl="jax")

    @staticmethod
    def _select_state(
        use_old: chex.Array, new: MjxSumoState, old: MjxSumoState
    ) -> MjxSumoState:
        """Select a state while preserving MJX-Warp's non-vmapped metadata."""
        return new.replace(
            # Data.where is backend-aware; a raw tree_map/jnp.where loses the
            # shared fields in Warp's Data implementation.
            data=new.data.where(use_old, old.data),
            pending=jnp.where(use_old, old.pending, new.pending),
            result=jnp.where(use_old, old.result, new.result),
            done=jnp.where(use_old, old.done, new.done),
            turn=jnp.where(use_old, old.turn, new.turn),
        )

    # ---- what a body has to answer ------------------------------------------

    @abc.abstractmethod
    def _build_model(self) -> mujoco.MjModel:
        """The compiled MuJoCo model, with both agents in it and player 0's actuators first."""

    @abc.abstractmethod
    def _initial_data(self, positions: chex.Array, angle: chex.Array) -> MjxData:
        """A fresh `mjx.Data` with the agents placed at `positions` (2, 2), facing each other.

        `angle` is the facing axis: the direction from player 0 to player 1.
        """

    @abc.abstractmethod
    def _positions(self, state: MjxSumoState) -> chex.Array:
        """`(2, 2)`: each agent's centre in the ring plane."""

    @abc.abstractmethod
    def _eliminated(self, data: MjxData) -> chex.Array:
        """`(2,)` bool: which agents have lost as of this substep.

        Always includes leaving the ring; a body that can also be knocked over
        adds that here.
        """

    @abc.abstractmethod
    def _control(self, player: chex.Array, value: chex.Array, positions: chex.Array) -> chex.Array:
        """One agent's `ctrl` block from their action. `player` is *traced*, so no branching."""

    # ---- shape/static information ------------------------------------------

    @property
    def max_steps(self) -> int:
        """One decision per player per control step."""
        return 2 * self.horizon

    def action_space(self, player: int) -> HybridSpace:
        return self._space

    # ---- the game tree ------------------------------------------------------

    def initial_state(self, key: chex.PRNGKey) -> MjxSumoState:
        """The agents face each other across the centre, at rest.

        The axis they face along is uniform on the circle when
        `random_orientation`, and each centre is jittered by up to
        `start_jitter` per coordinate -- the only chance moves in the game.
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
        positions = jnp.stack([-half * axis, half * axis]) + jitter
        return MjxSumoState(
            data=self._initial_data(positions, angle),
            pending=jnp.zeros(self._space.box.shape, dtype=jnp.float32),
            result=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=bool),
            turn=jnp.zeros((), dtype=jnp.int32),
        )

    def current_player(self, state: MjxSumoState) -> chex.Array:
        over = state.done | (state.turn >= self.max_steps)
        return jnp.where(over, TERMINAL, state.turn % 2).astype(jnp.int32)

    def action_mask(self, player: int, state: MjxSumoState) -> chex.Array:
        """All-`True`: pushing is the only kind, and every control is legal."""
        del player, state
        return jnp.ones((self.num_kinds(0),), dtype=bool)

    def payoff(self, state: MjxSumoState) -> chex.Array:
        return jnp.where(state.result != 0.0, state.result, self.margin_weight * self.potential(state))

    def reward(self, state: MjxSumoState, action: HybridAction, next_state: MjxSumoState) -> chex.Array:
        """Potential-based shaping, `shaping_weight * (phi(s') - phi(s))`.

        Zero on player 0's half of a control step, which moves nothing.
        """
        del action
        return self.shaping_weight * (self.potential(next_state) - self.potential(state))

    def potential(self, state: MjxSumoState) -> chex.Array:
        """`phi(s)`: how much farther from the centre player 1 is than player 0, in ring radii."""
        radii = jnp.linalg.norm(self._positions(state), axis=-1)
        return jnp.clip((radii[1] - radii[0]) / self.ring_radius, -1.0, 1.0)

    def _step(self, state: MjxSumoState, action: HybridAction, key: chex.PRNGKey) -> MjxSumoState:
        del key  # the physics is deterministic; the only chance is the start
        player = state.turn % 2
        dtype = state.data.qpos.dtype
        control = self._control(player, action.value.astype(dtype), self._positions(state))

        # Player 0 only parks their control; player 1's choice closes the
        # control step. Both branches are computed because `player` is traced.
        # Player 0's actuators come first in the model, so the two blocks
        # concatenate in that order.
        data, result, done = self._simulate(
            state.data, jnp.concatenate([state.pending.astype(dtype), control])
        )
        resolves = player == 1
        return MjxSumoState(
            data=data.where(~resolves, state.data),
            pending=jnp.where(resolves, jnp.zeros_like(control), control).astype(jnp.float32),
            result=jnp.where(resolves, result, state.result),
            done=jnp.where(resolves, done, state.done),
            turn=state.turn + 1,
        )

    def step(
        self, state: MjxSumoState, action: HybridAction, key: chex.PRNGKey
    ) -> MjxSumoState:
        """The generic terminal guard, using backend-aware MJX Data selection."""
        stepped = self._step(state, action, key)
        return self._select_state(self.is_terminal(state), stepped, state)

    def park_action(self, state: MjxSumoState, action: HybridAction) -> MjxSumoState:
        """Apply player 0's half-turn without running the physics solver.

        The generic sequential-game ``_step`` above has to select on a traced
        player number, so its call to ``_simulate`` is evaluated even on player
        0's half-turn and then discarded.  MJX rollouts know statically that
        adjacent decisions are player 0 then player 1; they use this method to
        avoid that otherwise guaranteed extra simulation.

        Terminal states remain absorbing, matching ``SequentialZeroSumGame.step``.
        Callers use this only at the start of a control step (player 0 to act).
        """
        control = self._control(0, action.value.astype(state.data.qpos.dtype), self._positions(state))
        parked = state.replace(pending=control.astype(jnp.float32), turn=state.turn + 1)
        terminal = self.is_terminal(state)
        return self._select_state(terminal, parked, state)

    def resolve_action(self, state: MjxSumoState, action: HybridAction) -> MjxSumoState:
        """Apply player 1's half-turn and run exactly one control-step simulation.

        Paired with :meth:`park_action`, this is equivalent to two calls to
        ``step`` but invokes ``_simulate`` only once.  Terminal padding still
        has to be selected leafwise because terminal episodes can differ within
        a vmapped batch.
        """
        dtype = state.data.qpos.dtype
        control = self._control(1, action.value.astype(dtype), self._positions(state))
        data, result, done = self._simulate(
            state.data, jnp.concatenate([state.pending.astype(dtype), control])
        )
        resolved = MjxSumoState(
            data=data,
            pending=jnp.zeros_like(control, dtype=jnp.float32),
            result=result,
            done=done,
            turn=state.turn + 1,
        )
        terminal = self.is_terminal(state)
        return self._select_state(terminal, resolved, state)

    # ---- physics ------------------------------------------------------------

    def _frame(self, own: chex.Array, opp: chex.Array) -> chex.Array:
        """`(2, 2)` rotation taking world vectors into the egocentric frame.

        Rows are the frame's axes: x towards the opponent, y its left-hand
        perpendicular. Right-handed for both players, so the two are mirror
        images and a symmetric state gives both the identical observation.
        """
        delta = opp - own
        distance = jnp.linalg.norm(delta)
        forward = jnp.where(
            distance > _EPS,
            delta / jnp.maximum(distance, _EPS),
            jnp.array([1.0, 0.0], dtype=delta.dtype),
        )
        left = jnp.stack([-forward[1], forward[0]])
        return jnp.stack([forward, left])

    def _simulate(self, data: MjxData, control: chex.Array):
        """Advance one control step under a fixed `control`; return `(data, result, done)`.

        `substeps` calls to `mjx.step`, each of `dt / substeps` seconds (the
        model's `timestep`). `_eliminated` is checked after every one of them,
        and the substep at which each agent first went is kept, so a
        simultaneous exit is settled by who went first -- the same tie-break, at
        the same granularity, as `DiskSumo._simulate`.
        """
        never = self.substeps

        def body(i, carry):
            data, out_at = carry
            data = mjx.step(self._model, data)
            out = self._eliminated(data)
            return data, jnp.where(out & (out_at == never), i, out_at)

        data, out_at = jax.lax.fori_loop(
            0, self.substeps, body,
            (data.replace(ctrl=control), jnp.full((2,), never, dtype=jnp.int32)),
        )

        # `out_at` is `never` for an agent still in, so one comparison covers
        # both "only one went" and "both went, one first".
        result = jnp.where(
            out_at[0] < out_at[1], -1.0, jnp.where(out_at[1] < out_at[0], 1.0, 0.0)
        ).astype(jnp.float32)
        return data, result, jnp.any(out_at < never)


class MjxSumo(MjxSumoBase):
    """Two pucks in a ring: `games.disk_sumo` on MuJoCo's solver.

    The model (`assets/mjx_sumo.xml`) is the smallest one that still exercises
    the whole MJX path: no gravity, no floor, two slide joints per puck, one
    puck-puck contact. What it buys over `DiskSumo` is a real solver, real
    actuators and a real integrator -- and the plumbing a serious model needs:
    an `mjx.Data` riding in the `lax.scan` carry, the terminal-state guard
    applying leafwise across its 80-odd arrays, and the memory profile of a
    batched rollout.

    The one parameter that changes name from `DiskSumo` is contact softness:
    `stiffness` and `contact_damping` become MuJoCo's `solref` pair
    (`solref_timeconst`, `solref_dampratio`), the same idea in the units the
    solver takes.
    """

    # Own pos/vel, opponent pos/vel (2 each), both edge margins, time left.
    OBS_DIM = 11

    def __init__(
        self,
        horizon: int = 50,
        ring_radius: float = 1.0,
        disk_radius: float = 0.15,
        start_distance: float = 0.6,
        start_jitter: float = 0.05,
        random_orientation: bool = True,
        egocentric: bool = True,
        dt: float = 0.1,
        substeps: int = 10,
        mass: float = 1.0,
        max_force: float = 1.0,
        drag: float = 1.0,
        solref_timeconst: float = 0.02,
        solref_dampratio: float = 1.0,
        solver_iterations: int = 4,
        solver_ls_iterations: int = 8,
        margin_weight: float = 0.0,
        shaping_weight: float = 0.0,
        physics_backend: str = "jax",
        warp_naconmax: int | None = None,
        warp_njmax: int | None = None,
        warp_graph_mode: str = "auto",
        warp_warn_solver_iterations: bool = False,
    ):
        self._init_rules(
            horizon=horizon, ring_radius=ring_radius, start_distance=start_distance,
            start_jitter=start_jitter, random_orientation=random_orientation, dt=dt,
            substeps=substeps, solver_iterations=solver_iterations,
            solver_ls_iterations=solver_ls_iterations, margin_weight=margin_weight,
            shaping_weight=shaping_weight,
        )
        for name, value in (("disk_radius", disk_radius), ("mass", mass),
                            ("max_force", max_force), ("solref_timeconst", solref_timeconst),
                            ("solref_dampratio", solref_dampratio)):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if drag < 0.0:
            raise ValueError(f"drag must be non-negative, got {drag}")
        if start_distance < 2.0 * disk_radius:
            raise ValueError(
                f"start_distance ({start_distance}) must be at least 2 * disk_radius "
                f"({2.0 * disk_radius}), or the pucks start overlapping"
            )

        self.disk_radius = float(disk_radius)
        self.egocentric = bool(egocentric)
        self.mass = float(mass)
        self.max_force = float(max_force)
        self.drag = float(drag)
        self.solref_timeconst = float(solref_timeconst)
        self.solref_dampratio = float(solref_dampratio)
        # Speed a lone puck settles at under full thrust: the natural velocity
        # scale for the observation. Drag-free physics falls back to what full
        # force reaches in one control step.
        self._speed_scale = max_force / drag if drag > 0.0 else max_force * dt / mass
        self._space = hybrid(NUM_ATOMS, [-1.0, -1.0], [1.0, 1.0])
        self._init_model(
            physics_backend, warp_naconmax, warp_njmax, warp_graph_mode,
            warp_warn_solver_iterations,
        )

    def _build_model(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(
            (_ASSETS / "mjx_sumo.xml").read_text().format(
                timestep=self.dt / self.substeps,
                disk_radius=self.disk_radius,
                mass=self.mass,
                drag=self.drag,
                max_force=self.max_force,
                solref_timeconst=self.solref_timeconst,
                solref_dampratio=self.solref_dampratio,
                iterations=self.solver_iterations,
                ls_iterations=self.solver_ls_iterations,
            )
        )

    def obs_dim(self, player: int) -> int:
        return self.OBS_DIM

    def _initial_data(self, positions: chex.Array, angle: chex.Array) -> MjxData:
        """The bodies sit at the origin, so `qpos` *is* the pair of centres."""
        del angle  # a puck has no orientation to set
        data = self._init_data.replace(
            qpos=positions.reshape(-1).astype(self._init_data.qpos.dtype),
            qvel=jnp.zeros_like(self._init_data.qvel),
            ctrl=jnp.zeros_like(self._init_data.ctrl),
        )
        return mjx.forward(self._model, data)

    def _positions(self, state: MjxSumoState) -> chex.Array:
        """`(2, 2)`: the two puck centres. `qpos` is `[p0_x, p0_y, p1_x, p1_y]`."""
        return state.data.qpos.reshape(2, 2)

    def _velocities(self, state: MjxSumoState) -> chex.Array:
        """`(2, 2)`: the two puck velocities, laid out like `qpos`."""
        return state.data.qvel.reshape(2, 2)

    def _eliminated(self, data: MjxData) -> chex.Array:
        """A puck is out once its centre passes the ring. It cannot be knocked over."""
        return jnp.linalg.norm(data.qpos.reshape(2, 2), axis=-1) > self.ring_radius

    def _control(self, player: chex.Array, value: chex.Array, positions: chex.Array) -> chex.Array:
        """The egocentric force, projected onto the unit disk and rotated into the world.

        The unit-disk projection is what stops a diagonal push being `sqrt(2)`
        times stronger than a straight one; `max_force` is the actuator's gear,
        so the control leaving here is already the normalized force.
        """
        local = self._space.box.clip(value)
        local = local / jnp.maximum(jnp.linalg.norm(local), 1.0)
        if not self.egocentric:
            return local
        own, opp = positions[player], positions[1 - player]
        return self._frame(own, opp).T @ local

    def observation(self, player: int, state: MjxSumoState) -> chex.Array:
        """`(11,)`, exactly `DiskSumo`'s: own and opponent pos/vel, both edge margins, time left.

        Positions are in units of `ring_radius`, velocities in units of the
        full-thrust speed, and all four vectors are rotated into `player`'s
        egocentric frame. `state.pending` is deliberately not read: it is player
        0's control for the live step, and showing it to player 1 would turn the
        simultaneous move into a sequential one.
        """
        pos, vel = self._positions(state), self._velocities(state)
        own, opp = player, 1 - player
        rotation = (
            self._frame(pos[own], pos[opp]) if self.egocentric
            else jnp.eye(2, dtype=pos.dtype)
        )
        radii = jnp.linalg.norm(pos, axis=-1) / self.ring_radius
        time_left = 1.0 - (state.turn // 2).astype(jnp.float32) / self.horizon
        return jnp.concatenate([
            rotation @ pos[own] / self.ring_radius,
            rotation @ vel[own] / self._speed_scale,
            rotation @ pos[opp] / self.ring_radius,
            rotation @ vel[opp] / self._speed_scale,
            jnp.stack([1.0 - radii[own], 1.0 - radii[opp], time_left]),
        ])


def _quat_to_mat(quat: chex.Array) -> chex.Array:
    """`(3, 3)` rotation matrix from a MuJoCo `(w, x, y, z)` quaternion.

    Columns are the body axes in world coordinates, which is what the
    observation wants: column 0 is where the body is facing, column 2 which way
    is up for it.
    """
    w, x, y, z = quat
    return jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        jnp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        jnp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])


class MjxLeggedSumo(MjxSumoBase):
    """Two radially symmetric walkers pushing each other out of a ring -- RoboSumo, in JAX.

    A subclass is one line: `LEG_ANGLES`, where its legs point. `MjxAntSumo`
    (four legs on the diagonals) is the classic MuJoCo ant; `MjxBugSumo` (six)
    and `MjxSpiderSumo` (three) are the same animal with a different leg count,
    in the spirit of the body family of Al-Shedivat et al. (2018) rather than a
    port of their assets. Everything below is shared: the leg count reaches the
    model through `assets/mjx_legged_leg.xml` and the observation width through
    `__init_subclass__`, so no subclass repeats any of it.

    The bodies are two copies attached into `assets/mjx_legged_arena.xml` with
    `MjSpec.attach(prefix=...)`, which namespaces every joint, geom and
    actuator so nothing has to be renamed by hand. Gravity and a floor are back:
    an articulated agent's first problem is staying on its feet.

    **What is different from `MjxSumo`, beyond the body.**

      * *The action is torque, not force.* Two joint torques per leg in
        `[-1, 1]`, ordered `(hip, ankle)` per leg in leg order, scaled by the
        actuators' `gear`. There is no egocentric rotation of the action and there could
        not be one: a joint torque is already expressed in the body's own frame.
        The egocentric frame survives in the *observation* only.
      * *An agent can also be knocked over.* Losing the ring is no longer the
        only way out: a torso below `knockdown_height` has been put on its back,
        which is the second losing condition in Bansal et al. (2018). Both are
        checked per substep, so who went first still settles a tie.
      * *The observation is `OBS_DIM`-dimensional* (65 for the ant), and its
        scales are set by hand
        (`velocity_scale` and friends) rather than derived from a drag law --
        there is no closed-form top speed for a walking ant.

    **Cost.** Two walkers run a couple of hundred contact slots, and the whole
    `mjx.Data` rides in the rollout's `lax.scan` carry, so the batch size is
    what will exhaust GPU memory first. On CPU this is a smoke test, not a
    training run.
    """

    # Where this body's legs point, in degrees. A subclass sets this and
    # nothing else; everything derived from it is computed below.
    LEG_ANGLES: tuple[float, ...] = ()
    # Length of one leg segment. The stock ant's legs run to (0.2, 0.2), so a
    # diagonal segment of this length reproduces it exactly.
    LEG_SEGMENT = 0.2 * 2.0 ** 0.5
    # How far an ankle is bent at the start, in radians. Every ankle swings
    # about the axis left of its own leg (see `assets/mjx_legged_leg.xml`), so
    # one positive bend stands every body up, whatever its leg count.
    ANKLE_STANCE = 1.0
    # Per agent, before the joints: torso xy, height, facing axis, up axis,
    # linear and angular velocity.
    TORSO_OBS_DIM = 2 + 1 + 3 + 3 + 3 + 3

    # Filled in per subclass: the hinges of one body in the order the actuators
    # drive them, and the observation widths that follow from how many there are.
    HINGES: tuple[str, ...] = ()
    AGENT_OBS_DIM = 0
    OBS_DIM = 0

    def __init_subclass__(cls, **kwargs) -> None:
        """Derive the joint names and observation width from `LEG_ANGLES`.

        Done here rather than in `__init__` so that `OBS_DIM` is available on
        the class, the way a config or a test wants to read it.
        """
        super().__init_subclass__(**kwargs)
        if not cls.LEG_ANGLES:
            return
        cls.HINGES = tuple(
            f"{kind}_{leg}"
            for leg in range(1, len(cls.LEG_ANGLES) + 1)
            for kind in ("hip", "ankle")
        )
        # Each joint contributes an angle and a velocity.
        cls.AGENT_OBS_DIM = cls.TORSO_OBS_DIM + 2 * len(cls.HINGES)
        # Both agents, then both edge margins and the clock.
        cls.OBS_DIM = 2 * cls.AGENT_OBS_DIM + 3

    def __init__(
        self,
        horizon: int = 80,
        ring_radius: float = 3.0,
        start_distance: float = 2.4,
        start_jitter: float = 0.1,
        random_orientation: bool = True,
        dt: float = 0.05,
        substeps: int = 5,
        gear: float = 150.0,
        start_height: float = 0.55,
        knockdown_height: float = 0.3,
        velocity_scale: float = 5.0,
        angular_scale: float = 10.0,
        joint_velocity_scale: float = 10.0,
        solver_iterations: int = 4,
        solver_ls_iterations: int = 8,
        margin_weight: float = 0.0,
        shaping_weight: float = 0.0,
        physics_backend: str = "jax",
        warp_naconmax: int | None = None,
        warp_njmax: int | None = None,
        warp_graph_mode: str = "auto",
        warp_warn_solver_iterations: bool = False,
    ):
        self._init_rules(
            horizon=horizon, ring_radius=ring_radius, start_distance=start_distance,
            start_jitter=start_jitter, random_orientation=random_orientation, dt=dt,
            substeps=substeps, solver_iterations=solver_iterations,
            solver_ls_iterations=solver_ls_iterations, margin_weight=margin_weight,
            shaping_weight=shaping_weight,
        )
        for name, value in (("gear", gear), ("start_height", start_height),
                            ("knockdown_height", knockdown_height),
                            ("velocity_scale", velocity_scale), ("angular_scale", angular_scale),
                            ("joint_velocity_scale", joint_velocity_scale)):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if knockdown_height >= start_height:
            raise ValueError(
                f"knockdown_height ({knockdown_height}) must be below start_height "
                f"({start_height}), or a bout is over before it begins"
            )

        if not self.LEG_ANGLES:
            raise ValueError(
                f"{type(self).__name__} declares no LEG_ANGLES; instantiate a body "
                f"subclass such as MjxAntSumo, MjxBugSumo or MjxSpiderSumo"
            )

        self.gear = float(gear)
        self.start_height = float(start_height)
        self.knockdown_height = float(knockdown_height)
        self.velocity_scale = float(velocity_scale)
        self.angular_scale = float(angular_scale)
        self.joint_velocity_scale = float(joint_velocity_scale)
        self._space = hybrid(NUM_ATOMS, [-1.0] * len(self.HINGES), [1.0] * len(self.HINGES))
        self._init_model(
            physics_backend, warp_naconmax, warp_njmax, warp_graph_mode,
            warp_warn_solver_iterations,
        )
        self._index_model()

    # ---- the model ----------------------------------------------------------

    def _leg_blocks(self) -> tuple[str, str]:
        """The `<body>` and `<motor>` XML for this body's legs, one pair per leg.

        A leg's ankle swings about the axis a quarter turn left of the leg
        itself, which is what lets one positive joint range bend every ankle
        downwards regardless of how many legs there are.
        """
        fragment = (_ASSETS / "mjx_legged_leg.xml").read_text()
        legs, actuators = [], []
        for index, degrees in enumerate(self.LEG_ANGLES, start=1):
            radians = math.radians(degrees)
            ux, uy = math.cos(radians), math.sin(radians)
            legs.append(fragment.format(
                index=index,
                ax=self.LEG_SEGMENT * ux, ay=self.LEG_SEGMENT * uy,
                ex=2.0 * self.LEG_SEGMENT * ux, ey=2.0 * self.LEG_SEGMENT * uy,
                px=-uy, py=ux,
            ))
            actuators.append(f'    <motor joint="hip_{index}"/>\n'
                             f'    <motor joint="ankle_{index}"/>')
        return "\n".join(legs), "\n".join(actuators)

    def _build_model(self) -> mujoco.MjModel:
        """Two prefixed copies of this body, attached into the arena.

        `MjSpec.attach` needs somewhere to put the child, hence the empty frame
        added to the world per agent; it does the renaming, which is the part
        that is easy to get wrong by hand and impossible to notice afterwards.
        """
        arena = mujoco.MjSpec.from_string(
            (_ASSETS / "mjx_legged_arena.xml").read_text().format(
                timestep=self.dt / self.substeps,
                iterations=self.solver_iterations,
                ls_iterations=self.solver_ls_iterations,
            )
        )
        legs, actuators = self._leg_blocks()
        shell = (_ASSETS / "mjx_legged.xml").read_text()
        for player, bit in enumerate(_AGENT_BITS):
            # Affinity for the floor and the *other* agent, never for itself.
            other = _AGENT_BITS[1 - player]
            body = mujoco.MjSpec.from_string(shell.format(
                gear=self.gear, start_height=self.start_height,
                contype=bit, conaffinity=_FLOOR_BIT | other,
                legs=legs, actuators=actuators,
            ))
            arena.attach(body, prefix=f"p{player}_", frame=arena.worldbody.add_frame())
        return arena.compile()

    def _index_model(self) -> None:
        """Cache the `qpos`/`qvel` slices of each agent, by joint name.

        The layout is read off the compiled model rather than assumed: the
        attach order fixes it in practice, but a silently wrong index here would
        show up as an observation that is subtly about the wrong agent, which no
        rules test would catch.
        """
        model = self._mj_model

        def joint(name: str) -> int:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if index < 0:
                raise ValueError(f"the compiled model has no joint {name!r}")
            return index

        self._root_qpos, self._root_dof = [], []
        self._hinge_qpos, self._hinge_dof, self._hinge_mid, self._hinge_half = [], [], [], []
        for player in (0, 1):
            root = joint(f"p{player}_root")
            self._root_qpos.append(int(model.jnt_qposadr[root]))
            self._root_dof.append(int(model.jnt_dofadr[root]))
            hinges = [joint(f"p{player}_{name}") for name in self.HINGES]
            self._hinge_qpos.append(jnp.asarray([model.jnt_qposadr[j] for j in hinges]))
            self._hinge_dof.append(jnp.asarray([model.jnt_dofadr[j] for j in hinges]))
            low, high = model.jnt_range[hinges, 0], model.jnt_range[hinges, 1]
            self._hinge_mid.append(jnp.asarray((low + high) / 2.0, dtype=jnp.float32))
            self._hinge_half.append(jnp.asarray((high - low) / 2.0, dtype=jnp.float32))

        # `_step` concatenates player 0's control block with player 1's, so the
        # model's actuators had better be in that order.
        driven = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
            for i in range(model.nu)
        ]
        expected = [f"p{p}_{name}" for p in (0, 1) for name in self.HINGES]
        if driven != expected:
            raise ValueError(f"unexpected actuator order: {driven} != {expected}")

        # The stance the bout opens in: hips centred, ankles bent by
        # `ANKLE_STANCE` in whichever direction their own range allows.
        stance = []
        for player in (0, 1):
            mid, half = self._hinge_mid[player], self._hinge_half[player]
            bent = jnp.clip(jnp.sign(mid) * self.ANKLE_STANCE, mid - half, mid + half)
            stance.append(jnp.where(half > 0.0, jnp.where(mid == 0.0, 0.0, bent), 0.0))
        self._stance = stance

    # ---- shape/static information ------------------------------------------

    def obs_dim(self, player: int) -> int:
        return self.OBS_DIM

    # ---- the game tree ------------------------------------------------------

    def _initial_data(self, positions: chex.Array, angle: chex.Array) -> MjxData:
        """Both ants standing at `positions`, each turned to face the other."""
        dtype = self._init_data.qpos.dtype
        qpos = self._init_data.qpos
        for player in (0, 1):
            root = self._root_qpos[player]
            # Yaw about z, as a MuJoCo (w, x, y, z) quaternion. Player 1 faces
            # back down the same axis, which is a half turn further round.
            yaw = 0.5 * (angle + jnp.pi * player)
            qpos = qpos.at[root:root + 3].set(
                jnp.concatenate([positions[player], jnp.full((1,), self.start_height)]).astype(dtype)
            )
            qpos = qpos.at[root + 3:root + 7].set(
                jnp.stack([jnp.cos(yaw), 0.0 * yaw, 0.0 * yaw, jnp.sin(yaw)]).astype(dtype)
            )
            qpos = qpos.at[self._hinge_qpos[player]].set(self._stance[player].astype(dtype))
        data = self._init_data.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(self._init_data.qvel),
            ctrl=jnp.zeros_like(self._init_data.ctrl),
        )
        return mjx.forward(self._model, data)

    def _positions(self, state: MjxSumoState) -> chex.Array:
        """`(2, 2)`: the two torsos' positions in the ring plane."""
        return self._torso_positions(state.data)

    def _torso_positions(self, data: MjxData) -> chex.Array:
        return jnp.stack([data.qpos[self._root_qpos[p]:self._root_qpos[p] + 2] for p in (0, 1)])

    def _heights(self, data: MjxData) -> chex.Array:
        """`(2,)`: how high each torso is off the floor."""
        return jnp.stack([data.qpos[self._root_qpos[p] + 2] for p in (0, 1)])

    def _eliminated(self, data: MjxData) -> chex.Array:
        """Out of the ring, or flat on its back."""
        out = jnp.linalg.norm(self._torso_positions(data), axis=-1) > self.ring_radius
        return out | (self._heights(data) < self.knockdown_height)

    def _control(self, player: chex.Array, value: chex.Array, positions: chex.Array) -> chex.Array:
        """The eight joint torques, clipped to the box.

        No egocentric rotation, unlike `MjxSumo._control`: a joint torque is
        already in the body's own frame, so there is nothing to rotate it out of.
        """
        del player, positions
        return self._space.box.clip(value)

    def observation(self, player: int, state: MjxSumoState) -> chex.Array:
        """`(65,)`: both agents' full physical state, in `player`'s egocentric frame.

        Per agent: torso position and height, the body's facing and up axes, its
        linear and angular velocity, and the eight joint angles and velocities.
        Then both edge margins and the time left. Everything with a direction in
        the ring plane is rotated into the frame whose x-axis points from
        `player` at their opponent, so a symmetric bout looks identical from both
        seats and "charge" is one action rather than a function of where on the
        ring the bout happens.

        `state.pending` is deliberately not read: it is player 0's control for
        the live step, and showing it to player 1 would turn the simultaneous
        move into a sequential one.
        """
        own, opp = player, 1 - player
        positions = self._positions(state)
        rotation = self._frame(positions[own], positions[opp])
        radii = jnp.linalg.norm(positions, axis=-1) / self.ring_radius
        time_left = 1.0 - (state.turn // 2).astype(jnp.float32) / self.horizon
        return jnp.concatenate([
            self._agent_features(state.data, own, rotation),
            self._agent_features(state.data, opp, rotation),
            jnp.stack([1.0 - radii[own], 1.0 - radii[opp], time_left]),
        ])

    def _agent_features(self, data: MjxData, agent: int, rotation: chex.Array) -> chex.Array:
        """One agent's `AGENT_OBS_DIM` numbers, seen through `rotation`.

        `agent` is a Python int -- the observation is built for each seat and
        selected afterwards -- so these are static slices, not traced gathers.
        """
        root, dof = self._root_qpos[agent], self._root_dof[agent]
        # A free joint's linear velocity is in world coordinates and its angular
        # velocity in the body's own frame; only the former needs rotating.
        orientation = _quat_to_mat(data.qpos[root + 3:root + 7])
        planar = lambda v: jnp.concatenate([rotation @ v[:2], v[2:3]])
        return jnp.concatenate([
            rotation @ data.qpos[root:root + 2] / self.ring_radius,
            jnp.stack([data.qpos[root + 2] / self.start_height]),
            planar(orientation[:, 0]),
            planar(orientation[:, 2]),
            planar(data.qvel[dof:dof + 3]) / self.velocity_scale,
            data.qvel[dof + 3:dof + 6] / self.angular_scale,
            (data.qpos[self._hinge_qpos[agent]] - self._hinge_mid[agent]) / self._hinge_half[agent],
            data.qvel[self._hinge_dof[agent]] / self.joint_velocity_scale,
        ])


class MjxAntSumo(MjxLeggedSumo):
    """Four legs on the diagonals: the classic MuJoCo/Gym ant. 8 torques, 65 observations."""

    LEG_ANGLES = (45.0, 135.0, 225.0, 315.0)


class MjxBugSumo(MjxLeggedSumo):
    """Six legs evenly spaced. 12 torques, 81 observations.

    More feet on the ground than the ant, so it is harder to tip and slower to
    turn -- the heavy end of this family, and the most expensive to simulate.
    """

    LEG_ANGLES = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)


class MjxSpiderSumo(MjxLeggedSumo):
    """Three legs evenly spaced, one pointing forward. 6 torques, 57 observations.

    The smallest body here and the least stable: a tripod has no margin, so
    being shoved off one foot is much closer to being shoved over. The cheapest
    of the three to simulate, and the one where `knockdown_height` does the most
    work.
    """

    LEG_ANGLES = (0.0, 120.0, 240.0)
