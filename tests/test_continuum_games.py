"""The three games whose Nash is known analytically and (mostly) diffuse.

`MultiPointGame` and friends all have a Nash on finitely many points, so a
`K`-component mixture can represent them exactly. These three cannot be
represented exactly by any finite mixture (the auction and Glicksberg-Gross) or
only by a large enough one (the circle), which is what they are in the
comparison set for -- and which is only worth anything if the analytic strategy
written into each class really is a Nash.

So what is pinned here is ground truth, in the same metric every baseline in the
repo reports: `baselines.common.GridOracle`'s exploitability at the strategy
`nash_strategy` returns, the game's value there, and -- for Glicksberg-Gross --
the far stronger indifference condition its equilibrium satisfies (the reply
value is constant over the *whole* action space, not just a support).
"""

from __future__ import annotations

import numpy as np
import jax
import pytest

# As in `test_baselines.py`: importing a baseline turns x64 on process-wide.
_X64_BEFORE = jax.config.jax_enable_x64

from baselines.common import GridOracle  # noqa: E402
from games.examples import (  # noqa: E402
    AllPayAuctionGame,
    CircleGame,
    GlicksbergGrossGame,
    SilentDuelGame,
)

jax.config.update("jax_enable_x64", _X64_BEFORE)

GRID = 801
ATOMS = 512


@pytest.fixture(scope="module", autouse=True)
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", _X64_BEFORE)


def _numpy_strategy(game, num_atoms=ATOMS):
    support, weights = game.nash_strategy(num_atoms)
    return np.asarray(support, dtype=np.float64), np.asarray(weights, dtype=np.float64)


# --------------------------------------------------------------- all-pay auction


def test_all_pay_uniform_is_nash_and_worth_zero():
    game = AllPayAuctionGame()
    oracle = GridOracle(game, points=GRID)
    support, weights = _numpy_strategy(game)

    # The support is exactly [0, 2 * value] = [0, 1].
    assert support.min() > 0.0 and support.max() < 1.0
    assert np.isclose(support.mean(), 0.5, atol=1e-3)

    assert abs(oracle.value(support, weights, support, weights)) < 1e-6
    # Residual is the discretization (atoms of width 1/ATOMS), not the strategy.
    assert oracle.exploitability(support, weights, support, weights) < 5.0 / ATOMS


def test_all_pay_deviations_are_punished():
    """A degenerate bid -- what a collapsed mixture would produce -- is exploitable."""
    game = AllPayAuctionGame()
    oracle = GridOracle(game, points=GRID)
    nash_support, nash_weights = _numpy_strategy(game)

    for bid in (0.0, 0.25, 0.5, 1.0):
        pure = np.array([[bid]])
        one = np.array([1.0])
        assert oracle.exploitability(pure, one, nash_support, nash_weights) > 0.1


def test_all_pay_rejects_a_box_that_truncates_the_equilibrium():
    with pytest.raises(ValueError, match="cuts off the equilibrium support"):
        AllPayAuctionGame(value=1.0, high=1.0)


# ------------------------------------------------------------------- circle game


@pytest.mark.parametrize("harmonics", [1, 3])
def test_circle_uniform_support_is_exactly_nash(harmonics):
    """`M > harmonics` evenly spaced atoms annihilate the kernel exactly."""
    game = CircleGame(harmonics=harmonics)
    oracle = GridOracle(game, points=GRID)
    support, weights = _numpy_strategy(game, num_atoms=harmonics + 1)

    # "Exact" here means exact up to float32: the games, like every other one in
    # `games/examples.py`, carry float32 constants even when the oracle runs in x64.
    assert abs(oracle.value(support, weights, support, weights)) < 1e-6
    assert oracle.exploitability(support, weights, support, weights) < 1e-6


def test_circle_too_small_a_support_is_exploitable():
    """The claim the docstring makes in the other direction: `M <= harmonics` fails.

    Two atoms half a turn apart kill the first harmonic and leave the third, so
    this separates `harmonics=1` (where they are a Nash) from `harmonics=3`.
    """
    two_atoms = np.array([[0.0], [0.5]])
    weights = np.array([0.5, 0.5])

    single = GridOracle(CircleGame(harmonics=1), points=GRID)
    assert single.exploitability(two_atoms, weights, two_atoms, weights) < 1e-6

    triple = GridOracle(CircleGame(harmonics=3), points=GRID)
    assert triple.exploitability(two_atoms, weights, two_atoms, weights) > 0.1

    with pytest.raises(ValueError, match="does not cancel"):
        CircleGame(harmonics=3).nash_strategy(3)


# ------------------------------------------------------------- Glicksberg-Gross


def test_glicksberg_gross_value_and_exploitability():
    game = GlicksbergGrossGame()
    oracle = GridOracle(game, points=GRID)
    support, weights = _numpy_strategy(game)

    value = oracle.value(support, weights, support, weights)
    assert np.isclose(value, 4.0 / np.pi, atol=1e-3)
    assert oracle.exploitability(support, weights, support, weights) < 1e-3


def test_glicksberg_gross_equilibrium_is_indifferent_everywhere():
    """`E_{t~F}[u(x, t)] == 4 / pi` for *every* `x in [0, 1]`, not just on a support.

    The kernel is symmetric, so player 0 needs this to be `<= 4/pi` and player 1
    needs it `>= 4/pi`; both hold only if it is flat, which is a much sharper
    check on `nash_strategy` than exploitability alone.
    """
    game = GlicksbergGrossGame()
    oracle = GridOracle(game, points=GRID)
    support, weights = _numpy_strategy(game, num_atoms=4096)

    replies = oracle.payoff(oracle.grid0, support) @ weights   # (GRID,)
    assert np.max(np.abs(replies - 4.0 / np.pi)) < 1e-3


def test_glicksberg_gross_density_is_concentrated_near_zero():
    """`f(t) ~ 1/sqrt(t)`: half the mass sits below `tan(pi/8)^2 ~ 0.17`."""
    game = GlicksbergGrossGame()
    support, _ = _numpy_strategy(game)
    median = float(np.median(support))
    assert np.isclose(median, np.tan(np.pi / 8) ** 2, atol=1e-3)


# ------------------------------------------------------------------ silent duel


@pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0])
def test_silent_duel_equilibrium_is_nash_and_worth_zero(exponent):
    """Value 0 (the kernel is antisymmetric) and no profitable reply to `F`."""
    game = SilentDuelGame(exponent=exponent)
    oracle = GridOracle(game, points=GRID)
    support, weights = _numpy_strategy(game)

    # Nobody fires before their accuracy reaches 1/3, and everybody has fired
    # by point blank.
    assert support.min() >= game.support_low
    assert support.max() <= 1.0

    assert abs(oracle.value(support, weights, support, weights)) < 1e-6
    # Residual is the discretization (ATOMS quantile cells), not the strategy.
    assert oracle.exploitability(support, weights, support, weights) < 1e-2


def test_silent_duel_indifferent_on_support_and_losing_below_it():
    """`E[u(x, T)] == 0` on `[1/3, 1]`, and strictly negative for an early shot.

    The second half is what distinguishes this equilibrium from a merely flat
    one: firing before the opponent's support starts is not an indifferent
    deviation but a losing one, worth `(3 * accuracy(x) - 1) / 2`.
    """
    game = SilentDuelGame()
    oracle = GridOracle(game, points=GRID)
    support, weights = _numpy_strategy(game, num_atoms=4096)

    times = np.asarray(oracle.grid0, dtype=np.float64)[:, 0]
    replies = oracle.payoff(oracle.grid0, support) @ weights   # (GRID,)

    on_support = times >= game.support_low
    assert np.max(np.abs(replies[on_support])) < 1e-3
    below = times[~on_support]
    assert np.allclose(replies[~on_support], (3.0 * below - 1.0) / 2.0, atol=1e-3)


def test_silent_duel_payoff_is_antisymmetric_and_jumps_across_the_diagonal():
    """The discontinuity this game exists for: a jump of `2 * accuracy(t)^2`."""
    game = SilentDuelGame()
    times = np.linspace(0.05, 1.0, 20)[:, None]
    matrix = np.array([[float(game.payoff(x, y)) for y in times] for x in times])

    assert np.allclose(matrix, -matrix.T, atol=1e-12)
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-12)

    epsilon = 1e-9
    for t in (0.4, 0.8):
        just_before = float(game.payoff(np.array([t - epsilon]), np.array([t])))
        just_after = float(game.payoff(np.array([t + epsilon]), np.array([t])))
        assert np.isclose(just_before - just_after, 2.0 * t**2, atol=1e-6)


def test_silent_duel_sharpness_recovers_the_hard_rule_in_the_limit():
    hard = SilentDuelGame()
    smooth = SilentDuelGame(sharpness=1e4)
    times = np.linspace(0.05, 1.0, 20)[:, None]

    for x in times:
        for y in times:
            assert np.isclose(float(hard.payoff(x, y)), float(smooth.payoff(x, y)), atol=1e-6)


def test_silent_duel_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SilentDuelGame(exponent=0.0)
    with pytest.raises(ValueError):
        SilentDuelGame(sharpness=0.0)
