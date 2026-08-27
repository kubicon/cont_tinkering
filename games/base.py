"""Base class for one-shot, two-player, zero-sum, continuous-action games."""

from __future__ import annotations

import abc

import chex
import jax
import jax.numpy as jnp
import optax

from .spaces import ActionSpace


def _validate_player(player: int) -> None:
    if player not in (0, 1):
        raise ValueError(f"player must be 0 or 1, got {player}")


class ZeroSumGame(abc.ABC):
    """A simultaneous-move, one-shot, two-player zero-sum game.

    Player 0 picks `action_1` from `action_space(0)`, player 1 independently
    picks `action_2` from `action_space(1)`. Player 0 receives `payoff(a1,
    a2)`, player 1 receives its negation. Subclasses implement `payoff` and
    `action_space`; everything else here is generic machinery (batched
    evaluation, sampling, projected-gradient best responses) that works for
    any pair of `ActionSpace`s.
    """

    # Every sample of a rollout sees the same observation -- true of the default
    # `observation` below (a constant), and hence of every game here. Override to
    # `False` alongside an `observation` that varies per episode (sampled context,
    # a repeated game's history): the trainers pass this to
    # `build_mixture_ppo_loss_fn(shared_obs=...)`, which lifts the policy's forward
    # pass out of the per-sample `vmap` when it holds.
    constant_observation: bool = True

    @abc.abstractmethod
    def action_space(self, player: int) -> ActionSpace:
        """The action space for `player` (0 or 1)."""

    def obs_dim(self, player: int) -> int:
        """Dimension of the observation passed to `player`'s policy.

        Games here are one-shot and stateless, so the default observation
        (see `observation`) is just a constant — this only exists so a
        policy network has a well-defined input shape. Override alongside
        `observation` for a game where players actually observe context
        (e.g. sampled per-episode parameters, or the other player's past
        actions in a repeated game).
        """
        _validate_player(player)
        return 1

    def observation(
        self, player: int, key: chex.PRNGKey, batch_shape: tuple[int, ...] = ()
    ) -> chex.Array:
        """`player`'s observation for one (or `batch_shape` many) episode(s). Default: constant zeros."""
        return jnp.zeros(batch_shape + (self.obs_dim(player),))

    @abc.abstractmethod
    def payoff(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        """Player 1's scalar payoff for one action profile. Player 2's payoff is -payoff."""

    def payoff_batch(self, action_1: chex.Array, action_2: chex.Array) -> chex.Array:
        """`payoff` vmapped over a leading batch axis of `action_1`/`action_2`."""
        return jax.vmap(self.payoff)(action_1, action_2)

    def sample_actions(
        self, key: chex.PRNGKey, batch_shape: tuple[int, ...] = ()
    ) -> tuple[chex.Array, chex.Array]:
        key_1, key_2 = jax.random.split(key)
        a1 = self.action_space(0).sample(key_1, batch_shape)
        a2 = self.action_space(1).sample(key_2, batch_shape)
        return a1, a2

    def best_response(
        self,
        player: int,
        opponent_action: chex.Array,
        key: chex.PRNGKey,
        num_steps: int = 200,
        learning_rate: float = 1e-1,
        init_action: chex.Array | None = None,
        num_restarts: int = 8,
    ) -> chex.Array:
        """`player`'s action against the other player's fixed `opponent_action`, via projected gradient ascent.

        Player 0 maximizes `payoff(action, opponent_action)`; player 1
        maximizes `-payoff(opponent_action, action)`.

        Runs `num_restarts` independent ascents from random starting points
        and keeps the best -- the best-response objective is not concave in
        general (e.g. a multi-modal game like `MultiPointGame` has a
        double-welled response landscape), so a single local ascent can stall
        in the wrong basin. Ignored when `init_action` is given (that pins a
        single deterministic start).
        """
        _validate_player(player)
        space = self.action_space(player)

        if player == 0:
            def loss(action):
                return -self.payoff(action, opponent_action)
        else:
            def loss(action):
                return self.payoff(opponent_action, action)

        if init_action is not None:
            return _projected_gradient_descent(loss, init_action, space, num_steps, learning_rate)

        inits = space.sample(key, batch_shape=(num_restarts,))
        candidates = jax.vmap(
            lambda init: _projected_gradient_descent(loss, init, space, num_steps, learning_rate)
        )(inits)
        losses = jax.vmap(loss)(candidates)
        return candidates[jnp.argmin(losses)]

    def exploitability(self, action_1: chex.Array, action_2: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """How much each player could gain by deviating (0 at a Nash equilibrium)."""
        key_1, key_2 = jax.random.split(key)
        br_0 = self.best_response(0, action_2, key_1)
        br_1 = self.best_response(1, action_1, key_2)
        gain_1 = self.payoff(br_0, action_2) - self.payoff(action_1, action_2)
        gain_2 = -self.payoff(action_1, br_1) - (-self.payoff(action_1, action_2))
        return gain_1 + gain_2

    def best_response_value(
        self,
        player: int,
        opponent_actions: chex.Array,
        key: chex.PRNGKey,
        num_steps: int = 200,
        learning_rate: float = 1e-1,
        num_restarts: int = 8,
    ) -> chex.Array:
        """`player`'s best achievable value against the empirical opponent distribution.

        `opponent_actions` is a *batch* `(num_samples, ...)` of the other
        player's sampled actions (a Monte-Carlo stand-in for their mixed
        strategy). Player 0 maximizes `E_{a2}[payoff(a1, a2)]`, player 1
        maximizes `E_{a1}[-payoff(a1, a2)]`; the returned scalar is that
        maximized expected payoff, found by projected gradient ascent on the
        single deterministic action being optimized (a best response to a
        fixed distribution is always pure).

        Uses `num_restarts` random starts and keeps the best: the expected
        payoff need not be concave in the responder's action (a multi-modal
        game like `MultiPointGame` has several local optima), so a
        single ascent can under-report the true best-response value -- which
        would make the exploitability read *too low* (even negative).
        """
        _validate_player(player)
        space = self.action_space(player)

        if player == 0:
            def loss(action):
                return -jnp.mean(jax.vmap(lambda a2: self.payoff(action, a2))(opponent_actions))
        else:
            def loss(action):
                return jnp.mean(jax.vmap(lambda a1: self.payoff(a1, action))(opponent_actions))

        inits = space.sample(key, batch_shape=(num_restarts,))
        candidates = jax.vmap(
            lambda init: _projected_gradient_descent(loss, init, space, num_steps, learning_rate)
        )(inits)
        return -jnp.min(jax.vmap(loss)(candidates))

    def mixture_exploitability(
        self, actions_1: chex.Array, actions_2: chex.Array, key: chex.PRNGKey
    ) -> chex.Array:
        """Exploitability of two mixed strategies given by their sampled actions.

        `actions_1`/`actions_2` are batches of actions drawn from each
        player's (possibly multi-modal) strategy. Unlike `exploitability`,
        which best-responds to a *single* deterministic action and so reports
        a large value for any mixed equilibrium, this best-responds to the
        opponent's whole action distribution:
        `max_{a1} E_{a2}[payoff] - min_{a2} E_{a1}[payoff]`, which is 0 at a
        Nash equilibrium regardless of how multi-modal the strategies are.
        """
        key_0, key_1 = jax.random.split(key)
        value_0 = self.best_response_value(0, actions_2, key_0)
        value_1 = self.best_response_value(1, actions_1, key_1)
        return value_0 + value_1


def _projected_gradient_descent(
    loss_fn,
    init: chex.Array,
    space: ActionSpace,
    num_steps: int,
    learning_rate: float,
) -> chex.Array:
    """Minimize `loss_fn` over `space` via Adam + projection back onto `space` each step."""
    optimizer = optax.adam(learning_rate)

    def step(carry, _):
        action, opt_state = carry
        grad = jax.grad(loss_fn)(action)
        updates, opt_state = optimizer.update(grad, opt_state, action)
        action = space.clip(optax.apply_updates(action, updates))
        return (action, opt_state), None

    (final_action, _), _ = jax.lax.scan(
        step, (init, optimizer.init(init)), xs=None, length=num_steps
    )
    return final_action
