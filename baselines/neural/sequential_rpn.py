"""Randomized policy networks on a game tree, trained by zeroth-order pseudo-gradient.

Martin & Sandholm's method (IJCAI 2023, arXiv:2211.15936; JPSPG, IJCAI 2025,
arXiv:2408.09306) as the fourth solver of `train_sequential.py`.
`baselines/neural/randomized_policy.py` is the one-shot implementation and its
docstring is the one to read for *why* the method looks like this -- an implicit
policy `a = f(o, z)` with no tractable density, and hence an update that never
differentiates the policy at all. Everything below is that method with the game
underneath it replaced by a tree, and the parts that do not care what the game is
are imported rather than rewritten: the perturbation draws, the common-random-number
keys, the noise contraction, both estimators (`separate` = IJCAI'23's per-player
SPG, `joint` = IJCAI'25's JPSPG) and the dynamics wrapper.

**What had to be decided here.** The paper's policy emits a continuous action; a
tree's action is hybrid (a discrete kind, plus a size on the continuous branch).
The extension used here keeps the method's defining property -- all randomness
enters through `z`, and nothing anywhere has a density:

    scores, size = f(obs, z);   kind = argmax(scores masked to the legal ones)

so as `z` varies the *kind* varies too, and any behavioral distribution over
kinds is representable. The size is squashed into the action box with a scaled
`tanh`, exactly as in the one-shot policy and for the same reason (a clip gives
the whole outside of the box zero gradient, and a zeroth-order estimator cannot
route around that).

**Budget it differently from the one-shot version.** There, one utility
evaluation is a vectorized payoff over `utility_samples` action pairs and costs
almost nothing, so the papers' `perturbation_batch = 256` is affordable. Here a
utility evaluation *plays hands*: one iteration costs
`perturbation_batch * utility_episodes` episodes with the joint estimator and
twice that with the separate one, so the defaults below are far smaller than the
papers'. That is a real cost of the method on trees rather than a shortcut, and
it is the reason `train_sequential.py` records `episodes` and `env_steps`:
compare this solver against the others on those, never on iteration counts.

Common random numbers do more work here than in the one-shot case, and are worth
understanding before touching the estimator. The key handed to a rollout draws
the deal *and* every `z` in it, and an antithetic pair shares that key, so
`u(theta + sigma z)` and `u(theta - sigma z)` are played over the same hands with
the same noise. Their difference is the perturbation's effect alone -- which is
the only reason a difference of two 64-hand estimates carries any signal.

**Measurement.** The policy has no density, so its behavioral strategy cannot be
read off the network the way `training.kuhn_evaluation` reads a mixture policy's:
it is *sampled*, by drawing `strategy_samples` noise vectors at every infoset.
Kuhn's `expl` is therefore exact-given-an-estimated-strategy rather than exact,
and its error shrinks as `1/sqrt(strategy_samples)`. Where there is no exact
metric at all, the bound is a best response trained by the same zeroth-order
ascent -- the method's own oracle, as PPO is PSRO's and NFSP's.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from games.discretized import base_game
from games.kuhn_best_response import (
    KuhnStrategy,
    best_response_value_first,
    best_response_value_second,
    bet_grid,
    game_value,
)
from games.sequential import TERMINAL, SequentialZeroSumGame, select_by_player
from games.sequential_examples import KIND_CALL, KIND_PASSIVE
from games.spaces import MASKED_LOGIT, HybridAction
from nets.activations import Activation
from training.optimizers import build_optimizer

from .common import print_row
from .jpspg import jpspg_pseudo_gradients
from .randomized_policy import (
    _apply_dynamics as apply_dynamics,  # the dynamics switch is game-agnostic
    batch_utility_keys,
    contract_noise,
    noise_batch,
    perturb,
    spg_pseudo_gradients,
    tree_add_scaled,
    tree_scale,
)
from .sequential_run import Budget, Stopwatch
from .sequential_scoring import has_exact_exploitability

ESTIMATORS = {"joint": jpspg_pseudo_gradients, "separate": spg_pseudo_gradients}


@dataclasses.dataclass(frozen=True)
class SequentialRPNHyperparams:
    """Everything needed to rebuild the policy and reproduce the run."""

    obs_dim: int
    num_kinds: int
    action_dim: int
    hidden_dims: tuple[int, ...]
    noise_dim: int = 8
    activation: str = "tanh"
    low: tuple[float, ...] = (0.0,)
    high: tuple[float, ...] = (1.0,)

    learning_rate: float = 1e-4
    optimizer: str = "adabelief"
    max_grad_norm: float = 0.0        # 0 disables clipping; the pseudo-gradient is noisy, not large
    sigma: float = 0.1                # smoothing radius
    # Hands per utility evaluation, and perturbations per iteration. Their
    # *product* is what an iteration costs; see the module docstring.
    utility_episodes: int = 64
    perturbation_batch: int = 64
    antithetic: bool = True
    dynamics: str = "simultaneous"    # simultaneous | extragradient | optimistic

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SequentialRPNHyperparams":
        data = dict(data)
        data["hidden_dims"] = tuple(data["hidden_dims"])
        data["low"], data["high"] = tuple(data["low"]), tuple(data["high"])
        return cls(**data)


class RandomizedHybridPolicy(nn.Module):
    """`(kind, size) = f(obs, z)`: an implicit distribution over a tree's hybrid actions.

    Two heads on one torso, and neither is a distribution: the kind head emits
    *scores* whose masked argmax is the action, and the size head emits one point
    of the box. The policy is deterministic given `z`, which is what makes it
    mixed without ever defining a density -- and what rules out every part of this
    repo's usual machinery (no `log pi`, so no PPO ratio, no trust region, no magnet).
    """

    num_kinds: int
    action_dim: int
    hidden_dims: tuple[int, ...]
    low: chex.Array
    high: chex.Array
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs: chex.Array, noise: chex.Array) -> tuple[chex.Array, chex.Array]:
        # He initialization, as both papers specify: Flax's default would start the
        # network a factor sqrt(2) per layer quieter, and a quiet network barely
        # propagates `noise` -- the one input that makes this policy mixed.
        dense = lambda dim, **kw: nn.Dense(dim, kernel_init=nn.initializers.he_normal(), **kw)
        x = jnp.concatenate([obs, noise], axis=-1)
        for dim in self.hidden_dims:
            x = Activation(kind=self.activation)(dense(dim)(x))
        scores = dense(self.num_kinds, name="kind_head")(x)
        raw = dense(self.action_dim, name="size_head")(x)
        size = self.low + (self.high - self.low) * 0.5 * (jnp.tanh(raw) + 1.0)
        return scores, size


def build_policy(hyperparams: SequentialRPNHyperparams) -> RandomizedHybridPolicy:
    return RandomizedHybridPolicy(
        num_kinds=hyperparams.num_kinds,
        action_dim=hyperparams.action_dim,
        hidden_dims=tuple(hyperparams.hidden_dims),
        low=jnp.asarray(hyperparams.low),
        high=jnp.asarray(hyperparams.high),
        activation=hyperparams.activation,
    )


def hyperparams_from_config(game: SequentialZeroSumGame, config) -> SequentialRPNHyperparams:
    """The policy and the estimator, from the shared `network:` section plus `rpn:`.

    The torso comes from the same `network:` block every other solver reads, so a
    comparison is not confounded by width -- but `num_components`, `clip_means`
    and the rest of the mixture head's fields mean nothing here and are ignored.
    """
    space = game.action_space(0)
    rpn = config.rpn
    return SequentialRPNHyperparams(
        obs_dim=game.obs_dim(0),
        num_kinds=game.num_kinds(0),
        action_dim=space.shape[0],
        hidden_dims=tuple(config.network.hidden_dims),
        noise_dim=rpn.noise_dim,
        activation=config.network.activation,
        low=tuple(float(x) for x in np.asarray(space.low).reshape(-1)),
        high=tuple(float(x) for x in np.asarray(space.high).reshape(-1)),
        learning_rate=rpn.learning_rate,
        optimizer=rpn.optimizer,
        max_grad_norm=rpn.max_grad_norm,
        sigma=rpn.sigma,
        utility_episodes=rpn.utility_episodes,
        perturbation_batch=rpn.perturbation_batch,
        antithetic=rpn.antithetic,
        dynamics=rpn.dynamics,
    )


def initial_params(game: SequentialZeroSumGame, policy: RandomizedHybridPolicy,
                   hyperparams: SequentialRPNHyperparams, seed: int = 0) -> tuple:
    """One parameter set per player, initialized on that player's observation shape."""
    key = jax.random.PRNGKey(seed)
    init_0, init_1, state_key = jax.random.split(key, 3)
    state = game.initial_state(state_key)
    noise = jnp.zeros(hyperparams.noise_dim)
    return tuple(
        policy.init(init_key, game.observation(player, state), noise)
        for player, init_key in ((0, init_0), (1, init_1))
    )


# --------------------------------------------------------------------------- playing


def policy_action_fn(policy: RandomizedHybridPolicy, params, noise_dim: int) -> Callable:
    """A `SequentialZeroSumGame.play_episode` action function backed by the policy.

    The key it is handed is the *only* source of randomness: it draws `z`, and the
    action follows deterministically. That is what lets an antithetic pair be
    replayed over identical hands (see the module docstring on common random
    numbers) -- and it is why nothing here needs, or could produce, a log-probability.
    """
    def action_fn(obs: chex.Array, mask: chex.Array, key: chex.PRNGKey) -> HybridAction:
        noise = jax.random.normal(key, (noise_dim,), dtype=obs.dtype)
        scores, size = policy.apply(params, obs, noise)
        kind = jnp.argmax(jnp.where(mask, scores, MASKED_LOGIT)).astype(jnp.int32)
        return HybridAction(kind=kind, value=size)

    return action_fn


def build_utility_fn(game: SequentialZeroSumGame, policy: RandomizedHybridPolicy,
                     hyperparams: SequentialRPNHyperparams):
    """`(params_pair, key) -> E[payoff to player 0]`, estimated by playing hands.

    The one-shot counterpart averages a payoff formula over sampled action pairs;
    this plays `utility_episodes` full episodes. Same contract, so the estimators
    in `randomized_policy`/`jpspg` take it unchanged -- which is the point of
    keeping this function the only game-aware part of the update.
    """
    episodes, noise_dim = hyperparams.utility_episodes, hyperparams.noise_dim

    def utility(params, key: chex.PRNGKey) -> chex.Array:
        action_fns = tuple(policy_action_fn(policy, params[player], noise_dim)
                           for player in (0, 1))
        payoff = jax.vmap(lambda k: game.play_episode(action_fns, k)[1])(
            jax.random.split(key, episodes))
        return jnp.mean(payoff)

    return utility


class RPNEvaluator:
    """Payoffs between two randomized policies, compiled once and reused.

    The same reason `training.best_response.PairEvaluator` exists: the parameters
    are *arguments*, not constants, so measuring after every log point does not
    recompile the rollout each time. Building the jitted function inside the
    measurement instead would spend more time in XLA than in the game.
    """

    def __init__(self, game: SequentialZeroSumGame, policy: RandomizedHybridPolicy,
                 noise_dim: int):
        self.game = game
        self.policy = policy
        self.noise_dim = noise_dim

        @jax.jit
        def batch_sums(params_0, params_1, keys):
            action_fns = (policy_action_fn(policy, params_0, noise_dim),
                          policy_action_fn(policy, params_1, noise_dim))
            payoff = jax.vmap(lambda k: game.play_episode(action_fns, k)[1])(keys)
            return jnp.sum(payoff), jnp.sum(payoff ** 2)

        self._batch_sums = batch_sums

    def evaluate(self, params, key: chex.PRNGKey, episodes: int = 20_000,
                 batch_size: int = 20_000) -> tuple[float, float]:
        """`(mean payoff to player 0, standard error)` over `episodes` hands."""
        if episodes < 2:
            raise ValueError(f"episodes must be at least 2, got {episodes}")
        total, total_squares, remaining = 0.0, 0.0, episodes
        while remaining > 0:
            key, batch_key = jax.random.split(key)
            size = min(batch_size, remaining)
            sums = self._batch_sums(params[0], params[1], jax.random.split(batch_key, size))
            total += float(sums[0])
            total_squares += float(sums[1])
            remaining -= size
        mean = total / episodes
        variance = max(total_squares / episodes - mean ** 2, 0.0)
        return mean, (variance / episodes) ** 0.5


def build_decision_counter(game: SequentialZeroSumGame, policy: RandomizedHybridPolicy,
                           noise_dim: int, episodes: int = 2_048):
    """`(params, key) -> mean decisions per hand`, compiled once -- the env-step price.

    `play_episode` returns a payoff and not a length, and the zeroth-order update
    never builds an `Episode` whose rows could be counted, so the count is taken
    here: the same scan, keeping the "somebody was to act" flag instead of the
    payoff. It is a measurement, so it is charged to neither step counter (see
    `baselines/neural/sequential_run.py`), and it is re-measured every log point
    because a policy that learns to fold shortens its own hands.
    """
    def one(params, episode_key: chex.PRNGKey) -> chex.Array:
        action_fns = tuple(policy_action_fn(policy, params[player], noise_dim)
                           for player in (0, 1))

        def body(state, step_key):
            key_0, key_1, transition_key = jax.random.split(step_key, 3)
            player = game.current_player(state)
            action = select_by_player(
                player == 0,
                action_fns[0](game.observation(0, state), game.action_mask(0, state), key_0),
                action_fns[1](game.observation(1, state), game.action_mask(1, state), key_1),
            )
            return game.step(state, action, transition_key), player != TERMINAL

        init_key, scan_key = jax.random.split(episode_key)
        _, acted = jax.lax.scan(body, game.initial_state(init_key),
                                jax.random.split(scan_key, game.max_steps))
        return jnp.sum(acted.astype(jnp.float32))

    @jax.jit
    def count(params, key: chex.PRNGKey) -> chex.Array:
        return jnp.mean(jax.vmap(lambda k: one(params, k))(jax.random.split(key, episodes)))

    return lambda params, key: float(count(params, key))


# --------------------------------------------------------------------------- the update


def player_pseudo_gradient(utility, params, player: int, key: chex.PRNGKey, sigma: float,
                           antithetic: bool = True, batch: int = 2) -> tuple:
    """One player's pseudo-gradient with the other frozen -- `spg` restricted to a side.

    What a best response is under this method: the same central-difference
    estimate, applied to one player's parameters only. Used by
    `exploitability_bound`; the self-play update uses the two-sided estimators
    imported above, unchanged.
    """
    noise_key, utility_key = jax.random.split(key)
    noise = noise_batch(noise_key, params[player], batch, antithetic)
    perturbed = perturb(params[player], tree_scale(noise, sigma))
    other = params[1 - player]

    def evaluate(own, sub_key):
        profile = (own, other) if player == 0 else (other, own)
        return utility(profile, sub_key)

    values = jax.vmap(evaluate)(perturbed, batch_utility_keys(utility_key, batch, antithetic))
    sign = 1.0 if player == 0 else -1.0            # player 1's utility is the negated payoff
    return contract_noise(noise, values, sign / (batch * sigma)), batch


def build_step(hyperparams: SequentialRPNHyperparams, utility, estimator, optimizer):
    """One iteration of self-play ascent: estimate, apply the dynamics, step both players.

    Lifted almost verbatim from `randomized_policy.run_pseudo_gradient` -- the
    update is not what changes when the game becomes a tree, and writing a second
    version of it would only create a second thing to keep correct.
    """
    def estimate(current, estimate_key):
        return estimator(utility, current, estimate_key, hyperparams.sigma,
                         hyperparams.antithetic, hyperparams.perturbation_batch)

    def step(carry, step_key):
        params, opt_states, previous = carry
        if hyperparams.dynamics == "extragradient":
            look_key, apply_key = jax.random.split(step_key)
            gradients, evaluations = estimate(params, look_key)
            lookahead = tuple(
                tree_add_scaled(params[i], gradients[i], hyperparams.learning_rate)
                for i in (0, 1))
            direction, extra = estimate(lookahead, apply_key)
            evaluations += extra
            raw = direction
        else:
            direction, raw, evaluations = apply_dynamics(
                hyperparams.dynamics, estimate, params, step_key, previous)

        new_params, new_states = [], []
        for player in (0, 1):
            # optax minimizes; these players ascend their own utility.
            updates, state = optimizer.update(
                tree_scale(direction[player], -1.0), opt_states[player], params[player])
            new_params.append(optax.apply_updates(params[player], updates))
            new_states.append(state)
        metrics = {
            "evaluations": jnp.asarray(evaluations, dtype=jnp.float32),
            "grad_norm_0": optax.tree.norm(direction[0]),
            "grad_norm_1": optax.tree.norm(direction[1]),
        }
        return (tuple(new_params), tuple(new_states), raw), metrics

    return step


def build_optimizer_chain(hyperparams: SequentialRPNHyperparams):
    transformations = [build_optimizer(hyperparams.optimizer, hyperparams.learning_rate)]
    if hyperparams.max_grad_norm > 0:
        transformations.insert(0, optax.clip_by_global_norm(hyperparams.max_grad_norm))
    return optax.chain(*transformations)


# --------------------------------------------------------------------------- measurement


def kuhn_strategy_of(game, policy: RandomizedHybridPolicy, params, player: int,
                     grid: chex.Array, key: chex.PRNGKey, samples: int,
                     noise_dim: int) -> KuhnStrategy:
    """The policy's behavioral strategy at Kuhn's infosets, *sampled*.

    There is no density to read, so every entry here is a frequency over `samples`
    noise draws: the probability of checking, the distribution of bet sizes (each
    draw's size landed on the nearest evaluation-grid point), and the probability
    of calling each size that could be faced. Consistent, with `1/sqrt(samples)`
    error -- which is the price of the representation, not of this function.

    One noise batch is drawn and reused across cards and sizes. That correlates
    the estimates of neighbouring infosets, which is harmless for the *value* of a
    best response to them (it is linear in each entry) and much cheaper than an
    independent batch per infoset.
    """
    base = base_game(game)
    open_node, faced_node = base.decision_nodes(player)
    open_mask = base.infoset_action_mask(open_node)
    faced_mask = base.infoset_action_mask(faced_node)
    cards = jnp.arange(base.num_cards)

    def act(obs, mask, noise):
        scores, size = policy.apply(params, obs, noise)
        return jnp.argmax(jnp.where(mask, scores, MASKED_LOGIT)), size[0]

    def at_open(card, noise):
        obs = base.infoset_observation(card, open_node, 0.0)
        kinds, sizes = jax.vmap(lambda z: act(obs, open_mask, z))(noise)
        column = jnp.argmin(jnp.abs(grid[None, :] - sizes[:, None]), axis=-1)
        # Joint, like every other reader here: the mass on a size already carries
        # "and it bet at all", so `open_check + open_bet.sum()` is 1.
        bet = (kinds != KIND_PASSIVE).astype(jnp.float32) / noise.shape[0]
        return (jnp.mean((kinds == KIND_PASSIVE).astype(jnp.float32)),
                jnp.zeros(grid.shape[0]).at[column].add(bet))

    def at_faced(card, bet, noise):
        obs = base.infoset_observation(card, faced_node, bet)
        kinds, _ = jax.vmap(lambda z: act(obs, faced_mask, z))(noise)
        return jnp.mean((kinds == KIND_CALL).astype(jnp.float32))

    open_key, faced_key = jax.random.split(key)
    open_noise = jax.random.normal(open_key, (samples, noise_dim))
    faced_noise = jax.random.normal(faced_key, (samples, noise_dim))
    open_check, open_bet = jax.vmap(lambda c: at_open(c, open_noise))(cards)
    call = jax.vmap(
        jax.vmap(lambda c, b: at_faced(c, b, faced_noise), in_axes=(None, 0)),
        in_axes=(0, None),
    )(cards, grid)
    return KuhnStrategy(open_check=open_check, open_bet=open_bet, call=call)


def exact_kuhn_exploitability(game, policy, hyperparams: SequentialRPNHyperparams, params,
                              key: chex.PRNGKey, samples: int,
                              grid_points: int | None = None) -> dict[str, float]:
    """Kuhn's exact best response against the *sampled* strategies of both players."""
    base = base_game(game)
    grid = bet_grid(base) if grid_points is None else bet_grid(base, grid_points)
    keys = jax.random.split(key, 2)
    strategies = [
        kuhn_strategy_of(game, policy, params[player], player, grid, keys[player], samples,
                         hyperparams.noise_dim)
        for player in (0, 1)
    ]
    br_0 = float(best_response_value_first(base, grid, strategies[1]))
    br_1 = float(best_response_value_second(base, grid, strategies[0]))
    return {"expl": br_0 + br_1, "br_0": br_0, "br_1": br_1,
            "value": float(game_value(base, grid, strategies[0], strategies[1]))}


def exploitability_bound(game, policy, hyperparams: SequentialRPNHyperparams, params,
                         key: chex.PRNGKey, iterations: int, episodes: int,
                         evaluator: "RPNEvaluator | None" = None) -> dict[str, float]:
    """Ascend each player against the frozen other, and add up what they win.

    A lower bound of the same shape as PSRO's and NFSP's `expl_lb`, produced by the
    method's own oracle: the responder is another randomized policy trained by the
    same zeroth-order ascent. Two caveats to read it with. It is *weaker* than the
    PPO responder the other solvers are scored with -- a first-order oracle sees
    gradients this one only estimates -- and it is warm-started from the run's own
    policy rather than cold (the perturbation, not the policy, is what explores
    here, so a warm start costs nothing structurally), which means it can only
    ever report *more* exploitability than the run already exhibits, never less.
    """
    utility = build_utility_fn(game, policy, hyperparams)
    optimizer = build_optimizer_chain(hyperparams)
    evaluator = RPNEvaluator(game, policy, hyperparams.noise_dim) if evaluator is None else evaluator
    values = []
    for player in (0, 1):
        key, train_key, eval_key = jax.random.split(key, 3)
        responder = jax.tree_util.tree_map(jnp.copy, params[player])
        opt_state = optimizer.init(responder)

        def step(carry, step_key, player=player):
            responder, opt_state = carry
            profile = (responder, params[1]) if player == 0 else (params[0], responder)
            gradient, _ = player_pseudo_gradient(
                utility, profile, player, step_key, hyperparams.sigma,
                hyperparams.antithetic, hyperparams.perturbation_batch)
            updates, opt_state = optimizer.update(
                tree_scale(gradient, -1.0), opt_state, responder)
            return (optax.apply_updates(responder, updates), opt_state), None

        (responder, _), _ = jax.lax.scan(
            step, (responder, opt_state), jax.random.split(train_key, iterations))
        profile = (responder, params[1]) if player == 0 else (params[0], responder)
        mean, _ = evaluator.evaluate(profile, eval_key, episodes=episodes)
        values.append(mean if player == 0 else -mean)
    return {"expl_lb": values[0] + values[1], "br_lb_0": values[0], "br_lb_1": values[1]}


def build_scorer(game: SequentialZeroSumGame, policy, hyperparams: SequentialRPNHyperparams,
                 config):
    """`(params_pair, t, full=True, force=False) -> metrics`, matching the other solvers'.

    Same columns and the same rules as `baselines.neural.sequential_scoring`:
    `h2h` always, `expl` where the tree admits an exact best response (against
    sampled strategies here), `expl_lb` every `scoring.score_every` log points.
    """
    exact = has_exact_exploitability(game)
    scoring, rpn = config.scoring, config.rpn
    evaluator = RPNEvaluator(game, policy, hyperparams.noise_dim)
    key = jax.random.PRNGKey(config.train.seed + 7919)

    def score(params, t: int, full: bool = True, force: bool = False) -> dict[str, float]:
        nonlocal key
        row: dict[str, float] = {}
        if full:
            key, h2h_key = jax.random.split(key)
            mean, stderr = evaluator.evaluate(params, h2h_key,
                                              episodes=min(scoring.episodes, 20_000))
            row.update({"h2h": mean, "h2h_stderr": stderr})
        if exact:
            key, exact_key = jax.random.split(key)
            row.update(exact_kuhn_exploitability(game, policy, hyperparams, params, exact_key,
                                                 rpn.strategy_samples, scoring.exact_grid))
        if full and scoring.score_every and (force or t % scoring.score_every == 0):
            key, bound_key = jax.random.split(key)
            row.update(exploitability_bound(game, policy, hyperparams, params, bound_key,
                                            rpn.br_iterations, scoring.episodes,
                                            evaluator=evaluator))
        return row

    return score


# --------------------------------------------------------------------------- the run


def run_sequential_rpn(
    game: SequentialZeroSumGame,
    config,
    iterations: int = 2_000,
    log_every: int = 50,
    estimator: str = "separate",
    seed: int = 0,
    scorer=None,
    writer=None,
    checkpoint_fn=None,
) -> dict:
    """Zeroth-order self-play with a randomized policy per player, on a game tree.

    One log point per `log_every` iterations, carrying the same cost columns every
    other solver reports. Two of the usual columns are missing and cannot exist:
    there is no `loss` (the method never forms an objective to differentiate --
    what it ascends is the utility, logged as `h2h`) and no per-player entropy or
    KL (there is no density). `grad_norm_0`/`grad_norm_1` are the norms of the
    *pseudo*-gradients, which is the diagnostic that matters most here: a norm that
    does not shrink while `expl` stalls means the estimator is returning noise, and
    the fix is `perturbation_batch`, not the learning rate.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r} (choices: {sorted(ESTIMATORS)})")
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    log_every = max(1, min(log_every, iterations))

    hyperparams = hyperparams_from_config(game, config)
    policy = build_policy(hyperparams)
    params = initial_params(game, policy, hyperparams, seed)
    utility = build_utility_fn(game, policy, hyperparams)
    optimizer = build_optimizer_chain(hyperparams)
    step = build_step(hyperparams, utility, ESTIMATORS[estimator], optimizer)
    run_chunk = jax.jit(lambda carry, keys: jax.lax.scan(step, carry, keys))
    count_decisions = build_decision_counter(game, policy, hyperparams.noise_dim)

    opt_states = tuple(optimizer.init(params[player]) for player in (0, 1))
    zero_gradients = tuple(
        jax.tree_util.tree_map(jnp.zeros_like, params[player]) for player in (0, 1))
    carry = (params, opt_states, zero_gradients)

    budget = Budget(default_episode_length=float(game.max_steps))
    clock = Stopwatch(running=True)
    history = writer.history if writer is not None else []
    key = jax.random.PRNGKey(seed + 1)
    done, t, total_evaluations = 0, 0, 0.0

    while done < iterations:
        length = min(log_every, iterations - done)
        key, chunk_key = jax.random.split(key)
        carry, metrics_stack = run_chunk(carry, jax.random.split(chunk_key, length))
        done += length
        t += 1
        clock.stop()

        params = carry[0]
        metrics = jax.device_get(metrics_stack)
        evaluations = float(np.sum(np.asarray(metrics["evaluations"])))
        total_evaluations += evaluations
        # Every utility evaluation played `utility_episodes` hands; how long a hand
        # is depends on the policy, so it is re-measured here rather than assumed.
        key, measure_key = jax.random.split(key)
        episode_length = count_decisions(params, measure_key)
        budget.add_iterations(length)
        budget.add_episodes(int(evaluations * hyperparams.utility_episodes),
                            episode_length=episode_length)

        entry = {
            "t": t,
            "wall_time": clock.seconds,
            **budget.row(),
            "utility_evaluations": total_evaluations,
            "episode_length": episode_length,
            "grad_norm_0": float(np.mean(np.asarray(metrics["grad_norm_0"]))),
            "grad_norm_1": float(np.mean(np.asarray(metrics["grad_norm_1"]))),
        }
        if scorer is not None:
            entry.update(scorer(params, t, force=(done >= iterations)))
        (writer.record(entry) if writer is not None else history.append(entry))
        print_row(entry, ("h2h", "expl_lb", "grad_norm_0", "utility_evaluations"))
        if checkpoint_fn is not None:
            checkpoint_fn(t, {f"player_{player}": (hyperparams, params[player])
                              for player in (0, 1)})
        clock.start()

    clock.stop()
    if writer is not None:
        for player in (0, 1):
            writer.save_params(f"player_{player}", hyperparams, params[player])
    return {"history": history, "params": params, "policy": policy,
            "hyperparams": hyperparams, "estimator": estimator,
            "utility_evaluations": total_evaluations,
            "train_seconds": clock.seconds, "budget": budget}
