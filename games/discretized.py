"""Playing a continuous game tree on a fixed grid of actions.

The discretization baseline -- "give the policy `bins` bet sizes and let it pick
one" -- can be built two ways round, and which one is chosen decides what the
comparison measures.

The way *not* taken here is a separate discrete policy: a categorical head over
the grid, its own rollout, its own loss. That is a second implementation of the
method, and any difference it then shows against the mixture policy is a
difference between two code paths as much as between two representations.

What this module does instead is discretize the **game**. `DiscretizedSequentialGame`
wraps a `SequentialZeroSumGame` and turns its one continuous branch into `bins`
extra *atoms*, laid out linearly across the action box, with the continuous
branch masked illegal at every node. Nothing downstream changes or even
notices: `MixtureActorCritic` already represents atoms alongside its Gaussians
(that is what `games.spaces.HybridSpace` is for), the batched rollout already
masks illegal kinds, and the PPO loss already reduces to its categorical factor
when every sampled action is an atom -- the Gaussian log-prob, entropy and KL
terms are all multiplied by `is_gaussian`, which is identically zero here. So a
run on a wrapped game is the *same update, the same rollout and the same
regularizers* as a run on the game itself, with only the head that is actually
used differing. That is the baseline worth having; see
`baselines/neural/mmd_discrete.py` for the one-shot counterpart of the same
argument.

Two consequences to keep in mind when reading a run:

  * **The policy still has a Gaussian head**, it is simply never sampled from
    (its logits are masked, so they receive no gradient either). Its *mean box
    penalty* would still reach the shared torso, which is why the runner zeroes
    `mean_box_penalty_coef` for a discretized run -- a dead head must not push
    the live one around.
  * **Only what a player can play is discretized, never what it observes.** A
    wrapped Kuhn player facing a bet still sees the real size, so exploitability
    measured with a fine-grid responder (`training.kuhn_evaluation`) reports the
    full cost of the grid, including deviations that lie between its points.
"""

from __future__ import annotations

import itertools

import chex
import jax.numpy as jnp
import numpy as np

from .sequential import SequentialZeroSumGame, State
from .spaces import HybridAction, HybridSpace, hybrid

# A categorical head this wide is already the point being demonstrated rather
# than a configuration; past it, the run is a memory error waiting to happen.
DEFAULT_MAX_ACTIONS = 1024


def linear_action_grid(space: HybridSpace, bins: int) -> np.ndarray:
    """`bins` points per axis spanning `space.box`, as a `(bins**d, d)` action set.

    Endpoints included: the extremes of a betting range are exactly where a
    poker-like equilibrium tends to put mass, and an open grid would miss them by
    half a cell for no reason. A degenerate box (`min_bet == max_bet`, i.e.
    classic Kuhn) collapses to `bins` copies of the single legal size, which is
    correct and harmless -- the categorical head simply splits one action's
    probability across identical actions.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    low = np.asarray(space.box.low, dtype=np.float64).reshape(-1)
    high = np.asarray(space.box.high, dtype=np.float64).reshape(-1)
    axes = [np.linspace(low[axis], high[axis], bins) for axis in range(low.shape[0])]
    return np.array(list(itertools.product(*axes)), dtype=np.float32).reshape(-1, low.shape[0])


class DiscretizedSequentialGame(SequentialZeroSumGame):
    """`game` with its continuous action replaced by `bins` linearly spaced atoms.

    The tree, the chance moves, the observations and the payoffs are the wrapped
    game's, untouched. What changes is one thing: `action_space(player)` has
    `num_atoms + bins**d` atoms instead of `num_atoms` atoms plus a continuum,
    and `action_mask` marks the continuous branch illegal everywhere, so a policy
    can only ever play a grid point.

    Both players must share an action shape and an atom count -- they already do
    in every sequential game here, because a batched rollout evaluates both
    players' heads and selects between them (`training.sequential_rollout`), and
    a wrapper is the wrong place to discover otherwise.
    """

    def __init__(self, game: SequentialZeroSumGame, bins: int,
                 max_actions: int = DEFAULT_MAX_ACTIONS):
        spaces = [game.action_space(player) for player in (0, 1)]
        for player, space in enumerate(spaces):
            if not isinstance(space, HybridSpace):
                raise ValueError(
                    f"player {player} acts in a {type(space).__name__}; discretization here "
                    "splits the continuous branch of a HybridSpace"
                )
        if spaces[0].num_atoms != spaces[1].num_atoms:
            raise ValueError(
                f"both players need the same atom count to share a rollout, got "
                f"{spaces[0].num_atoms} and {spaces[1].num_atoms}"
            )
        if spaces[0].shape != spaces[1].shape:
            raise ValueError(
                f"both players need the same action shape, got {spaces[0].shape} "
                f"and {spaces[1].shape}"
            )

        grids = [linear_action_grid(space, bins) for space in spaces]
        if grids[0].shape[0] > max_actions:
            raise ValueError(
                f"a {bins}-bin grid over a {spaces[0].shape[0]}-D action is "
                f"{grids[0].shape[0]} actions; that blow-up is the cost this baseline "
                f"exists to demonstrate, but it will not fit in a categorical head. "
                f"Lower bins, or raise max_actions past {max_actions} deliberately."
            )

        self.game = game
        self.bins = bins
        self.base_atoms = spaces[0].num_atoms
        self.num_grid = grids[0].shape[0]
        self._grids = tuple(jnp.asarray(grid) for grid in grids)
        self._spaces = tuple(
            hybrid(space.num_atoms + self.num_grid, space.box.low, space.box.high)
            for space in spaces
        )

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.game!r}, bins={self.bins}, "
                f"actions={self.base_atoms + self.num_grid})")

    def grid(self, player: int) -> chex.Array:
        """`player`'s action set: `(bins**d, d)`, row `k` played by kind `base_atoms + k`."""
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player}")
        return self._grids[player]

    # ---- shape/static information ------------------------------------------

    @property
    def max_steps(self) -> int:
        return self.game.max_steps

    def action_space(self, player: int) -> HybridSpace:
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player}")
        return self._spaces[player]

    def obs_dim(self, player: int) -> int:
        return self.game.obs_dim(player)

    # ---- the game tree ------------------------------------------------------

    def initial_state(self, key: chex.PRNGKey) -> State:
        return self.game.initial_state(key)

    def current_player(self, state: State) -> chex.Array:
        return self.game.current_player(state)

    def observation(self, player: int, state: State) -> chex.Array:
        """The wrapped game's, unchanged -- a player still *sees* real bet sizes.

        Discretizing the observation as well would be a different (and much
        weaker) game: an opponent's off-grid bet would become invisible rather
        than merely unanswerable in kind.
        """
        return self.game.observation(player, state)

    def action_mask(self, player: int, state: State) -> chex.Array:
        return self.discretize_kind_mask(self.game.action_mask(player, state))

    def payoff(self, state: State) -> chex.Array:
        return self.game.payoff(state)

    def _step(self, state: State, action: HybridAction, key: chex.PRNGKey) -> State:
        """Translate a grid kind back into the wrapped game's continuous action."""
        return self.game.step(state, self.base_action(state, action), key)

    # ---- the translation ----------------------------------------------------

    def discretize_kind_mask(self, mask: chex.Array) -> chex.Array:
        """A wrapped `(base_atoms + 1,)` kind mask, as this game's `(num_kinds,)` one.

        The original atoms keep their legality and their indices; the continuous
        entry's legality is copied onto every grid point (a node where betting was
        legal is a node where every bet size is); and the new continuous entry is
        `False` everywhere, which is what makes "the policy can only play a grid
        point" a property of the game rather than a convention the trainer has to
        respect.
        """
        atoms = mask[..., :self.base_atoms]
        continuous = mask[..., self.base_atoms:self.base_atoms + 1]
        return jnp.concatenate(
            [atoms, jnp.repeat(continuous, self.num_grid, axis=-1), jnp.zeros_like(continuous)],
            axis=-1,
        )

    def base_action(self, state: State, action: HybridAction) -> HybridAction:
        """This game's action as the wrapped game's: a kind, or a grid point.

        A kind at or past `base_atoms` is a grid index, and is played as the
        wrapped game's continuous branch at that grid point -- including the
        (masked, hence unreachable) continuous kind, which clamps onto the last
        grid point rather than passing an off-grid value through. `_step` must be
        total, and "total" here must not mean "leaks a continuous action".
        """
        index = jnp.clip(action.kind - self.base_atoms, 0, self.num_grid - 1)
        is_first = self.game.current_player(state) == 0
        # Cast to the proposed action's dtype rather than letting `where` promote:
        # the wrapped game must receive exactly the dtype it would have received
        # without the wrapper, or its state's dtypes stop matching across a
        # `lax.scan` carry.
        grid_value = jnp.where(
            is_first, self._grids[0][index], self._grids[1][index]
        ).astype(action.value.dtype)
        is_atom = action.kind < self.base_atoms
        return HybridAction(
            kind=jnp.where(is_atom, action.kind, self.base_atoms).astype(jnp.int32),
            value=jnp.where(is_atom, action.value, grid_value),
        )


def discretize(game: SequentialZeroSumGame, bins: int | None,
               max_actions: int = DEFAULT_MAX_ACTIONS) -> SequentialZeroSumGame:
    """`DiscretizedSequentialGame(game, bins)`, or `game` itself when `bins` is falsy."""
    if not bins:
        return game
    return DiscretizedSequentialGame(game, bins, max_actions=max_actions)


def base_game(game: SequentialZeroSumGame) -> SequentialZeroSumGame:
    """The game underneath a discretization, or `game` itself.

    Anything that reasons about the *tree* rather than about the action set --
    Kuhn's exact best response, its bet grid, its card count -- wants this one.
    """
    return game.game if isinstance(game, DiscretizedSequentialGame) else game
