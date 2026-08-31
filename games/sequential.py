"""Base class for two-player, zero-sum, *sequential* (extensive-form) games.

`games.base.ZeroSumGame` is one-shot: both players move once, simultaneously,
and `payoff(a1, a2)` closes the game out. This module is the extensive-form
counterpart -- players alternate over a game tree, each decision made from an
information set rather than from the full state, so a strategy is a
*behavioral* strategy (a distribution over actions at every infoset) instead of
a single distribution over actions.

Nothing about the policy network has to change for this: `MixtureActorCritic`
is already a map `obs -> (logits, means, scale_trils, value)`, so feeding it the
observation of an infoset (rather than the constant zeros a stateless
`ZeroSumGame` hands it) is exactly "produce the local behavioral strategy at
this decision point".

**Actions are hybrid.** A node in a poker-like tree offers a few parameterless
choices (fold, check, call) alongside a parameterized one (bet *how much*), so
an action here is a `HybridAction`: a discrete `kind`, plus a continuous
`value` read only on the continuous kind. See `games.spaces.HybridSpace` for
why the discrete part is a genuine atom rather than a region of a continuous
action.

**Not every kind is legal everywhere.** Calling is meaningless with no bet
outstanding, and folding is meaningless without one. `action_mask` reports
per-node legality, and a policy applies it to its categorical logits before
sampling -- which is also why the mask has to be recorded in the rollout and
re-applied in the loss, or the PPO importance ratios are taken against a
different distribution from the one that was sampled.

Design constraints, all of them driven by the fact that rollouts are `jit`ed
and `vmap`ed over a batch of environments (see `training/mixture.py`):

  * **The state is a fixed-shape pytree.** No Python-side branching on it, no
    growing history list. A game with a variable-length history encodes it into
    fixed-width fields (a node index, an outstanding bet, ...) instead.
  * **The horizon is statically bounded** by `max_steps`, so a trajectory is a
    `lax.scan` of that fixed length rather than a `while` loop. Episodes that
    finish early sit in a terminal state for the remaining steps, which is why
    `step` is required to be a no-op on terminal states -- `step` (the concrete
    wrapper here) guarantees that for every subclass, so `_step` never has to.
  * **`obs_dim`, `action_space` and the mask width do not depend on the
    state.** At a given step of a batched rollout, different environments are
    at different nodes with different players to act; a rollout deals with that
    by evaluating *both* players' networks and selecting with `jnp.where`,
    which needs one common observation shape and one common action shape. Only
    the mask's *contents* vary by node, never its width.
"""

from __future__ import annotations

import abc
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from .spaces import HybridAction, HybridSpace

# `current_player` returns this instead of 0/1 once the state is terminal.
TERMINAL = -1

# A game's state: any fixed-shape pytree (typically a frozen `chex.dataclass`).
State = chex.ArrayTree

# `(obs, action_mask, key) -> HybridAction`: what `play_episode` asks of each player.
ActionFn = Callable[[chex.Array, chex.Array, chex.PRNGKey], HybridAction]


def select_by_player(is_first: chex.Array, first: chex.ArrayTree, second: chex.ArrayTree) -> chex.ArrayTree:
    """`first` where player 0 acts, `second` where player 1 does -- leafwise, as a select.

    Whose turn it is is a *traced* value inside a `jit`ed rollout, so it cannot
    branch: a step evaluates both players' observations, masks and networks and
    then keeps the acting one. Both arguments must be pytrees of the same
    structure with broadcast-compatible leaves.
    """
    return jax.tree_util.tree_map(lambda a, b: jnp.where(is_first, a, b), first, second)


def _validate_player(player: int) -> None:
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")


class SequentialZeroSumGame(abc.ABC):
    """A turn-taking, two-player, zero-sum, imperfect-information game.

    Player 0's payoff is `payoff(terminal_state)`; player 1's is its negation,
    exactly as in `ZeroSumGame`. Subclasses implement the eight abstract members
    below; `is_terminal`, `num_kinds`, `step`, `play_episode` and
    `random_action_fn` are generic machinery built on top of them.

    The rollout contract, in the order a trainer uses it:

        state = initial_state(key)               # chance node(s): the deal
        while not is_terminal(state):
            player = current_player(state)
            obs    = observation(player, state)  # the acting player's infoset
            mask   = action_mask(player, state)  # which kinds are legal here
            action = policy(obs, mask)           # a `HybridAction`
            state  = step(state, action, key)
        reward = payoff(state)                   # player 0's payoff
    """

    # ---- shape/static information ------------------------------------------

    @property
    @abc.abstractmethod
    def max_steps(self) -> int:
        """Upper bound on the number of *decisions* in any single episode.

        Static (a Python int, not a traced value): it is the length of the
        `lax.scan` a batched rollout unrolls into.
        """

    @abc.abstractmethod
    def action_space(self, player: int) -> HybridSpace:
        """`player`'s action space, the same at every one of their decision nodes.

        State-independent by design -- see this module's docstring. Node-specific
        legality is `action_mask`'s job, not a varying space.
        """

    @abc.abstractmethod
    def obs_dim(self, player: int) -> int:
        """Width of `observation(player, ...)`, the same at every node."""

    def num_kinds(self, player: int) -> int:
        """`player`'s number of action kinds: their atoms plus the continuous branch."""
        return self.action_space(player).num_kinds

    # ---- the game tree ------------------------------------------------------

    @abc.abstractmethod
    def initial_state(self, key: chex.PRNGKey) -> State:
        """A fresh state, with any initial chance move (the deal) already resolved."""

    @abc.abstractmethod
    def current_player(self, state: State) -> chex.Array:
        """Whose turn it is: `0`, `1`, or `TERMINAL`. A traced scalar, not a Python int."""

    @abc.abstractmethod
    def observation(self, player: int, state: State) -> chex.Array:
        """`player`'s information set at `state`, as a `(obs_dim(player),)` vector.

        Must expose *only* what `player` can see -- in a poker-like game, their
        own card and the public betting history, never the opponent's card.
        Getting this wrong silently turns the game into one of perfect
        information, which is the single easiest way to make a sequential
        experiment meaningless.

        Called on terminal states too (a batched rollout evaluates every step of
        the scan regardless), so it must return a correctly shaped vector there;
        the value is masked out and never used.
        """

    @abc.abstractmethod
    def action_mask(self, player: int, state: State) -> chex.Array:
        """Which of `player`'s action kinds are legal at `state`: `(num_kinds(player),)` bool.

        Indexed the same way as `HybridAction.kind`: entry `i < num_atoms` is
        atom `i`, and the last entry is the continuous branch.

        Must have **at least one legal kind in every state**, terminal ones
        included -- an all-`False` mask sends every logit to `MASKED_LOGIT` and
        the resulting softmax is uniform garbage rather than an error. The
        conventional choice at a terminal state is all-`True`: nothing is
        actually sampled there, since the rollout masks those steps out of the
        loss entirely.
        """

    @abc.abstractmethod
    def payoff(self, state: State) -> chex.Array:
        """Player 0's scalar payoff. Only meaningful once `is_terminal(state)`."""

    @abc.abstractmethod
    def _step(self, state: State, action: HybridAction, key: chex.PRNGKey) -> State:
        """Apply `action` at a *non-terminal* `state`, resolving any chance move with `key`.

        Subclasses implement this; callers use `step`, which adds the
        terminal-state guard. `action` is assumed already projected onto
        `action_space(current_player(state))`, and its `kind` assumed legal
        under `action_mask` -- a policy samples from masked logits, so an
        illegal kind has probability zero. `_step` must still be *total* (some
        branch has to be taken for every `kind`), it simply need not make that
        branch meaningful.
        """

    # ---- generic machinery --------------------------------------------------

    def is_terminal(self, state: State) -> chex.Array:
        return self.current_player(state) == TERMINAL

    def step(self, state: State, action: HybridAction, key: chex.PRNGKey) -> State:
        """`_step`, made a no-op on terminal states.

        A fixed-length `lax.scan` keeps calling `step` on episodes that have
        already ended, so "stepping" a terminal state has to leave it (and in
        particular its payoff) exactly as it was. Doing it here once means no
        subclass has to remember to.
        """
        stepped = self._step(state, action, key)
        terminal = self.is_terminal(state)
        return jax.tree_util.tree_map(
            lambda new, old: jnp.where(terminal, old, new), stepped, state
        )

    def play_episode(
        self, action_fns: tuple[ActionFn, ActionFn], key: chex.PRNGKey
    ) -> tuple[State, chex.Array]:
        """Play one full episode with `action_fns[p]` acting for player `p`.

        A minimal, dependency-free reference implementation of the rollout
        contract -- enough to sanity-check a new game (are episodes really
        terminal within `max_steps`? is the payoff really zero-sum?) without
        pulling in a network or a trainer. The training rollout mirrors this
        structure but additionally records per-step observations, masks,
        log-probs and a done-mask; see `training/mixture.py`.

        Both players' action functions are evaluated at every step and the
        acting one is selected with `jnp.where` -- the same trick a batched
        rollout needs, since in a `vmap`ed batch different environments have
        different players to act.
        """
        for player, action_fn in enumerate(action_fns):
            if not callable(action_fn):
                raise TypeError(f"action_fns[{player}] must be callable")

        def body(state: State, step_key: chex.PRNGKey) -> tuple[State, None]:
            key_0, key_1, transition_key = jax.random.split(step_key, 3)
            player = self.current_player(state)
            action_0 = action_fns[0](
                self.observation(0, state), self.action_mask(0, state), key_0
            )
            action_1 = action_fns[1](
                self.observation(1, state), self.action_mask(1, state), key_1
            )
            action = select_by_player(player == 0, action_0, action_1)
            return self.step(state, action, transition_key), None

        init_key, scan_key = jax.random.split(key)
        keys = jax.random.split(scan_key, self.max_steps)
        final_state, _ = jax.lax.scan(body, self.initial_state(init_key), keys)
        return final_state, self.payoff(final_state)

    def random_action_fn(self, player: int) -> ActionFn:
        """Uniform-random *legal* play for `player`; a baseline opponent and a test fixture."""
        _validate_player(player)
        space = self.action_space(player)

        def action_fn(obs: chex.Array, mask: chex.Array, key: chex.PRNGKey) -> HybridAction:
            del obs
            return space.sample(key, mask)

        return action_fn
