"""V-trace targets (Espeholt et al., 2018) for one player of an interleaved trajectory.

A sequential batch holds *both* players' decisions interleaved on one time axis,
plus the padding a finished episode is carried through (see
`training.mixture.Episode`). From player `p`'s side, the opponent's decisions
are part of the environment: V-trace runs over `p`'s own decisions only, and
every other row simply passes the recursion through. That is also how the
opponent's exploration is treated -- as environment, uncorrected -- which is
the same choice the `explore_eps` loss makes (see
`training.mixture.mixture_ppo_loss_from_outputs`).

For `p`'s own decisions `s_1 < s_2 < ...` (write `n(s)` for the next one after
`s`, and `V(n(s)) = 0` when there is none):

    R_s     = sum of p's rewards on rows s, s+1, ..., n(s) - 1
    delta_s = rho_s * (R_s + gamma * V(n(s)) - V(s))
    v_s     = V(s) + delta_s + gamma * c_s * (v_{n(s)} - V(n(s)))
    A_s     = R_s + gamma * v_{n(s)} - V(s)          (the policy-gradient advantage)

with `rho_s = min(rho_bar, pi(a_s) / mu(a_s))` and
`c_s = lambda * min(c_bar, pi(a_s) / mu(a_s))`. `gamma` discounts per *own
decision*, not per row. `A_s` carries no `rho_s` factor: the PPO surrogate it
feeds already multiplies by `pi / pi_old`, and the `explore_eps` loss by
`pi_old / mu` on top of that.

**Correcting the opponent's future.** With `opponent_log_rhos`, the opponent's
rows between `s` and `n(s)` are importance-weighted too: their ratios
`q_s = prod_k pi_j(a_k) / mu_j(a_k)` over the rows `s < k < n(s)` (to the end of
the episode for the last decision) fold into the segment's action, so
`pi(a_s) / mu(a_s)` above becomes `pi(a_s) q_s / mu(a_s)`, and the advantage
becomes `min(rho_bar, q_s) * (R_s + gamma * v_{n(s)} - V(s))`. The target is
then the value against the opponent's *policy* rather than its exploring
behavior from `s` onwards, while the states `s` themselves are still the
behavior's -- see `opponent_past_weight` for reweighting those.

All three recursions -- the reward accumulated up to the next own decision,
the value found there, and the V-trace correction itself -- are the same
backward linear recurrence `y_t = b_t + a_t * y_{t+1}` with `y_T = 0`, which
`jax.lax.associative_scan` evaluates in `O(log T)` depth rather than a
`T`-step sequential `lax.scan`.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp


def reverse_linear_recurrence(a: chex.Array, b: chex.Array, axis: int = -1) -> chex.Array:
    """`y` with `y_t = b_t + a_t * y_{t+1}` along `axis`, and `y` zero past the end.

    Each step is the affine map `f_t(y) = b_t + a_t * y`, and `y_t` is
    `f_t(f_{t+1}(... f_{T-1}(0)))`. Composition of affine maps is associative,
    so the whole suffix of compositions is one `associative_scan`; the constant
    term of each composed map is `y_t`.
    """
    chex.assert_equal_shape([a, b])

    def compose(later, earlier):
        # With `reverse=True` the scan accumulates from the end, so the first
        # argument covers the later rows and the second the earlier ones:
        # `f_earlier(f_later(y)) = b_e + a_e * (b_l + a_l * y)`.
        a_later, b_later = later
        a_earlier, b_earlier = earlier
        return a_earlier * a_later, b_earlier + a_earlier * b_later

    # `associative_scan` wants a nonnegative axis.
    _, y = jax.lax.associative_scan(compose, (a, b), reverse=True, axis=axis % a.ndim)
    return y


def _shift_left(x: chex.Array, fill) -> chex.Array:
    """`x[..., t + 1]` at position `t`, and `fill` at the last position (time is the last axis)."""
    pad = jnp.full(x.shape[:-1] + (1,), fill, dtype=x.dtype)
    return jnp.concatenate([x[..., 1:], pad], axis=-1)


@chex.dataclass(frozen=True)
class VTraceOutput:
    """Per-row V-trace quantities, `(..., T)`; meaningful only on the player's own rows."""

    vs: chex.Array  # value targets `v_s`
    pg_advantage: chex.Array  # `R_s + gamma * v_{n(s)} - V(s)`
    rho: chex.Array  # truncated importance weights `min(rho_bar, pi / mu)`, opponent's future included
    opponent_rho: chex.Array  # untruncated `q_s`; one without `opponent_log_rhos`


def vtrace(
    own: chex.Array,
    rewards: chex.Array,
    values: chex.Array,
    log_rhos: chex.Array,
    gamma: float = 1.0,
    lambda_: float = 1.0,
    rho_bar: float = 2.0,
    c_bar: float = 1.0,
    opponent_log_rhos: chex.Array | None = None,
) -> VTraceOutput:
    """V-trace targets for the rows `own` marks, time on the last axis.

    Args:
      own: `(..., T)` bool, the rows that are this player's decisions.
      rewards: `(..., T)` this player's reward on each row's transition, the
        terminal payoff included on the last decision row. Rows that are not
        `own` still contribute: an opponent's move can pay out.
      values: `(..., T)` the value estimate `V(s)` on `own` rows (anything elsewhere).
      log_rhos: `(..., T)` `log pi(a) - log mu(a)` on `own` rows (anything elsewhere).
      opponent_log_rhos: `(..., T)` the opponent's `log pi_j(a) - log mu_j(a)` on
        rows that are not `own` (ignored on `own` rows; zero on padding). `None`
        leaves the opponent's moves uncorrected, as environment.

    Nothing here is differentiated: the caller stops gradients on the inputs.
    """
    chex.assert_equal_shape([own, rewards, values, log_rhos])
    own = own.astype(values.dtype)
    rewards = rewards.astype(values.dtype)
    values = values * own
    log_rhos = jnp.where(own > 0, log_rhos, 0.0)

    own_next = _shift_left(own, 0.0)
    through = 1.0 - own_next  # the recursion continues past t unless t + 1 is an own row

    # `R_t`: rewards from row t up to (not including) the next own decision.
    segment_reward = reverse_linear_recurrence(through, rewards)
    # `V(n(t))`: the value at the next own decision, 0 past the last one.
    next_value = reverse_linear_recurrence(through, own_next * _shift_left(values, 0.0))

    if opponent_log_rhos is None:
        segment_log_rho = jnp.zeros_like(values)
    else:
        chex.assert_equal_shape([own, opponent_log_rhos])
        # `log q_t`: the opponent's rows from t up to (not including) the next own decision.
        opponent_log_rhos = jnp.where(own > 0, 0.0, opponent_log_rhos.astype(values.dtype))
        segment_log_rho = reverse_linear_recurrence(through, opponent_log_rhos)
        log_rhos = log_rhos + own * segment_log_rho

    rho = jnp.exp(jnp.minimum(log_rhos, jnp.log(rho_bar)))
    c = lambda_ * jnp.exp(jnp.minimum(log_rhos, jnp.log(c_bar)))
    delta = own * rho * (segment_reward + gamma * next_value - values)

    # `v - V` at the first own row at or after t: own rows apply the V-trace
    # step, every other row passes the next own row's correction through.
    correction = reverse_linear_recurrence(
        jnp.where(own > 0, gamma * c, 1.0), jnp.where(own > 0, delta, 0.0)
    )
    vs = values + correction
    next_vs = next_value + _shift_left(correction, 0.0)
    # `min(rho_bar, q_s)`: the PPO surrogate weights the own action, not the opponent's.
    opponent_rho_bar = jnp.exp(jnp.minimum(segment_log_rho, jnp.log(rho_bar)))
    pg_advantage = own * opponent_rho_bar * (segment_reward + gamma * next_vs - values)
    return VTraceOutput(
        vs=vs * own, pg_advantage=pg_advantage, rho=rho * own,
        opponent_rho=own * jnp.exp(jnp.minimum(segment_log_rho, 20.0)),
    )


def opponent_past_weight(
    own: chex.Array, opponent_log_rhos: chex.Array, floor: float
) -> chex.Array:
    """`max(floor, prod_{k < s} pi_j(a_k) / mu_j(a_k))` on each `own` row `s`, zero elsewhere.

    The opponent's rows before `s` decide which states the player is trained on.
    Weighting a decision by this product moves the state distribution from the
    opponent's exploring behavior back to its policy; the floor keeps the states
    only exploration reaches in training (at `floor` times their behavior
    weight), so the player still learns how to answer them. The floor is on the
    *cumulative* product -- a per-row floor would compound to `floor ** k`.
    """
    chex.assert_equal_shape([own, opponent_log_rhos])
    own = own.astype(opponent_log_rhos.dtype)
    opponent_log_rhos = jnp.where(own > 0, 0.0, opponent_log_rhos)
    # Rows strictly before t; an own row's own entry is zero, so through t is the same.
    past = jnp.cumsum(opponent_log_rhos, axis=-1)
    return own * jnp.exp(jnp.clip(past, jnp.log(floor), 20.0))
