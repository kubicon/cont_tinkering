"""Pins the three structural claims `CoupledRotationGame`'s docstring makes.

The game exists to isolate one thing -- whether the magnet's proximal term is
what makes the run converge -- so the properties it relies on are worth
pinning: the origin really is the Nash, the payoff really does couple a
player's own coordinates (so it is not `dim` independent 1-D games), and the
coupling really is not bilinear.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from games.examples import CoupledRotationGame


@pytest.mark.parametrize("dim", [2, 3, 5])
def test_origin_is_the_nash(dim: int) -> None:
    """`U(x, 0) <= 0 = U(0, 0) <= U(0, y)`: neither player gains by deviating."""
    game = CoupledRotationGame(dim=dim)
    lo, hi = game.action_space(0).low, game.action_space(0).high
    zero = jnp.zeros(dim)
    assert float(game.payoff(zero, zero)) == 0.0

    key = jax.random.PRNGKey(0)
    actions = jax.random.uniform(key, (256, dim), minval=lo, maxval=hi)
    against_zero = jax.vmap(lambda a: game.payoff(a, zero))(actions)   # player 0 deviates
    zero_against = jax.vmap(lambda a: game.payoff(zero, a))(actions)   # player 1 deviates
    assert float(jnp.max(against_zero)) <= 0.0
    assert float(jnp.min(zero_against)) >= 0.0
    # ... and strictly so away from the origin, which is what makes the Nash unique.
    assert float(jnp.max(against_zero)) < 0.0
    assert float(jnp.min(zero_against)) > 0.0


def test_payoff_does_not_separate_over_coordinates() -> None:
    """A separable payoff would satisfy `U(x, y) = sum_d U_d(x_d, y_d)`.

    Checked as the mixed second derivative `d^2 U / dx_i dx_j`, which vanishes
    identically for any sum of per-coordinate terms; here `||x||^4` makes it
    nonzero. `d^2 U / dx_i dy_j` off the diagonal is the other half: the skew
    coupling pays coordinate `i` against the opponent's `i +- 1`.
    """
    game = CoupledRotationGame(dim=3)
    x = jnp.array([0.4, -0.3, 0.2])
    y = jnp.array([-0.1, 0.5, 0.3])

    own = jax.hessian(lambda a: game.payoff(a, y))(x)
    assert abs(float(own[0, 1])) > 1e-6

    cross = jax.jacobian(jax.grad(lambda a, b: game.payoff(a, b)), argnums=1)(x, y)
    assert abs(float(cross[0, 1])) > 1e-6


def test_coupling_is_not_bilinear() -> None:
    """Bilinear would mean `U(s*x, y)` is linear in `s` once the wells are removed."""
    game = CoupledRotationGame(dim=2, damping=0.0)
    x = jnp.array([0.7, -0.4])
    y = jnp.array([0.2, 0.6])
    single = float(game.payoff(x, y))
    doubled = float(game.payoff(2 * x, y))
    assert not np.isclose(doubled, 2 * single, rtol=1e-3)

    # `warp=0` is the degenerate case the docstring calls out: then it *is* bilinear.
    linear = CoupledRotationGame(dim=2, warp=0.0, damping=0.0)
    assert np.isclose(float(linear.payoff(2 * x, y)), 2 * float(linear.payoff(x, y)), rtol=1e-5)
