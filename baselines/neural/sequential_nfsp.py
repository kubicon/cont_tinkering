"""Neural fictitious self-play on a game tree: NFSP for Kuhn, Leduc and Blotto.

The algorithm is `baselines/neural/nfsp.py`'s: each player keeps an RL best
response `beta` to what the opponents have been doing and a supervised `pi` that
imitates its own past best responses, and plays the anticipatory mixture
`eta * beta + (1 - eta) * pi`. `pi` approximates the fictitious-play average,
which in a zero-sum game converges to a Nash equilibrium; `beta` does not and is
not supposed to. NFSP was *written* for this setting -- Heinrich & Silver's
experiments are Leduc -- so the tree is less of a translation here than it is for
PSRO. What changes against the one-shot version:

  * **The reservoir stores decisions, not actions.** In a one-shot game a best
    response is one distribution over actions, so a reservoir of sampled actions
    describes it completely. In a tree it is a policy, and imitating it means
    storing `(infoset observation, legality mask, action)` triples from play --
    `sequential_oracle.sample_decision_rows`. Which infosets those rows come from
    is not a detail: the fictitious-play average is realization-weighted, and
    sampling the trajectories from the current anticipatory profile is what
    supplies that weighting for free.
  * **Only `beta`'s hands are stored.** Both players play their anticipatory
    mixtures, so the *trajectories* are drawn from the profile actually being
    played, but a hand is recorded only when the learner drew `beta` for it --
    which is NFSP's rule, and the reason `--reservoir-episodes` buys about `eta`
    times its number in rows.
  * **The average net shares the best response's architecture.** The one-shot
    version gives it its own capacity (`--average-components 8`) because it is a
    density estimator rather than the object under test; here it cannot, since a
    batched tree rollout gathers one member's parameters per episode out of one
    stacked pytree and selects between the two players' heads with `jnp.where`.
    `network.num_components` in the config therefore sets both. See
    `baselines/neural/sequential_oracle.py`.
  * **The supervised loss is the hybrid action's joint log-likelihood**: the
    masked categorical over kinds, plus -- on the continuous kind only -- the
    Gaussian mixture's marginal density at the bet size. Cloning only the bet
    size (the one-shot loss) would fit *how much* this policy bets while ignoring
    *whether* it bets, which in poker is most of the strategy.

`--average-head reservoir` has no counterpart here. A one-shot reservoir is
itself a strategy (an empirical distribution over actions, playable as is); a
reservoir of infoset rows is not, so on a tree the supervised fit is the only
average policy there is.

**The columns.** `expl` is the **average** policies' exploitability -- the object
NFSP's guarantee is about -- and exists only on Kuhn, where an exact tree best
response does; `--score-every N` adds `expl_lb`, an RL best-response lower bound,
on Leduc and Blotto. `br_expl`/`br_expl_lb` is the best-response pair's, which is
not expected to converge: a falling `expl` beside a non-falling `br_expl` is what
a working run looks like.

Usage:
    python -m baselines.neural.sequential_nfsp configs/kuhn.yaml --rounds 30
    python -m baselines.neural.sequential_nfsp configs/leduc.yaml --rounds 30 --score-every 5
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np

from games.sequential import SequentialZeroSumGame
from training.actor_critic import masked_log_softmax
from training.checkpoint import save_checkpoint_step_multi
from training.config import MixturePPOHyperparams
from training.mixture import build_mixture_network, mixture_marginal_log_prob
from training.ppo import create_train_state

from . import sequential_oracle as so
from .common import RunWriter, print_row
from .sequential_common import load_sequential_run, sequential_parser
from .sequential_run import Budget, mean_columns
from .sequential_scoring import build_scorer

# The entry names `best_response.py` reads a self-play checkpoint under. NFSP's
# average policies are written the same way on purpose: a finished run can then
# be scored offline by exactly the tool that scores a self-play run, with no
# NFSP-aware loader in the picture.
CHECKPOINT_ENTRY = "player_{}"


class DecisionReservoir:
    """Reservoir sample of a best response's *decisions* -- the supervised set.

    Reservoir rather than a sliding window, for the one-shot version's reason:
    the supervised target is the average of every best response the player has
    ever played, so early rounds must keep their weight instead of ageing out.
    Fixed-capacity device-shaped arrays with a live count keep one compiled shape
    across rounds.
    """

    def __init__(self, capacity: int, obs_dim: int, mask_width: int, action_dim: int,
                 seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_mask = np.zeros((capacity, mask_width), dtype=bool)
        self.action_kind = np.zeros(capacity, dtype=np.int32)
        self.raw_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, rows: dict[str, np.ndarray]) -> None:
        """Algorithm R over a batch of rows, so the buffer stays a uniform sample."""
        count = rows["obs"].shape[0]
        for index in range(count):
            self.seen += 1
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                slot = int(self.rng.integers(0, self.seen))
                if slot >= self.capacity:
                    continue
            self.obs[slot] = rows["obs"][index]
            self.action_mask[slot] = rows["action_mask"][index]
            self.action_kind[slot] = rows["action_kind"][index]
            self.raw_action[slot] = rows["raw_action"][index]


def fit_average_policy(
    game: SequentialZeroSumGame,
    player: int,
    hyperparams: MixturePPOHyperparams,
    reservoir: DecisionReservoir,
    steps: int,
    batch_size: int,
    key: chex.PRNGKey,
    init_params=None,
):
    """Maximum-likelihood fit of a mixture policy to the reservoir -- NFSP's SL step.

    The likelihood is the hybrid action's joint one, under the same mask the row
    was sampled behind:

        log P(kind) + [kind is continuous] * log p_mixture(bet size | continuous)

    with the second term the mixture's *marginal* density
    (`training.mixture.mixture_marginal_log_prob`), i.e. behaviour cloning with
    the component identity marginalized out -- the reservoir records what was
    played, not which component played it.

    Warm-started from the previous round's fit (`init_params`), unlike the best
    response: this net is a density estimator being handed a slowly growing
    dataset, not a policy that has to explore, so restarting it every round would
    only throw away the fit and add noise to the average NFSP's guarantee is about.
    """
    network = build_mixture_network(hyperparams)
    num_atoms = hyperparams.num_atoms
    init_key, train_key, state_key = jax.random.split(key, 3)
    if init_params is None:
        dummy_state = game.initial_state(state_key)
        init_params = network.init(init_key, game.observation(player, dummy_state))
    if reservoir.size == 0:
        # Nothing to imitate yet. A small `eta` and a small `reservoir_episodes`
        # can leave a round with no best-response hands at all, and fitting to an
        # empty buffer would sample indices out of an empty range and train on
        # whatever the zeroed rows happen to mean. The average policy simply
        # stays where it was.
        return network, init_params, float("nan")
    state = create_train_state(network, init_params, hyperparams)

    obs = jnp.asarray(reservoir.obs)
    action_mask = jnp.asarray(reservoir.action_mask)
    action_kind = jnp.asarray(reservoir.action_kind)
    raw_action = jnp.asarray(reservoir.raw_action)
    size = jnp.asarray(reservoir.size)

    def row_log_prob(params, obs_row, mask_row, kind, action):
        logits, means, scale_trils, _ = network.apply(params, obs_row)
        log_probs = masked_log_softmax(logits, mask_row)
        # The categorical head spreads the continuous branch over `num_components`
        # logits, so "the policy played a continuous action" is their logsumexp,
        # and the density given that choice is the marginal mixture.
        continuous = jax.nn.logsumexp(log_probs[num_atoms:]) + mixture_marginal_log_prob(
            logits, means, scale_trils, mask_row, action, num_atoms)
        atom = log_probs[jnp.minimum(kind, jnp.maximum(num_atoms - 1, 0))]
        return jnp.where(kind < num_atoms, atom, continuous)

    def loss_fn(params, index):
        log_probs = jax.vmap(row_log_prob, in_axes=(None, 0, 0, 0, 0))(
            params, obs[index], action_mask[index], action_kind[index], raw_action[index])
        return -jnp.mean(log_probs)

    grad_fn = jax.value_and_grad(loss_fn)

    def step(state, step_key):
        index = jax.random.randint(step_key, (batch_size,), 0, size)
        loss, grads = grad_fn(state.params, index)
        return state.apply_gradients(grads=grads), loss

    state, losses = jax.jit(
        lambda s, keys: jax.lax.scan(step, s, keys)
    )(state, jax.random.split(train_key, steps))
    return network, state.params, float(jnp.mean(losses))


def run_sequential_nfsp(
    game: SequentialZeroSumGame,
    config,
    rounds: int = 30,
    br_steps: int = 50,
    br_epochs: int = 20,
    eta: float = 0.1,
    reservoir_capacity: int = 200_000,
    reservoir_episodes: int = 4096,
    sl_steps: int = 400,
    sl_batch: int = 256,
    seed: int = 0,
    scorer=None,
    writer: RunWriter | None = None,
    checkpoint_fn=None,
) -> dict:
    """`rounds` of NFSP on a tree. Returns the average/best-response policies and history.

    `checkpoint_fn(step, entries, arrays)` -- `SequentialRunLog.checkpoint`'s
    signature -- is called once per round. The average policies go in under
    `player_0`/`player_1` and the best responses under `br_0`/`br_1`: the average
    is the object NFSP's guarantee is about, so it is the one a tool reading the
    checkpoint blind (`best_response.py`) should find.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")

    key = jax.random.PRNGKey(seed)
    hyperparams = [so.br_hyperparams(game, player, config) for player in (0, 1)]
    mask_width = hyperparams[0].num_atoms + hyperparams[0].num_components
    reservoirs = [
        DecisionReservoir(reservoir_capacity, game.obs_dim(player), mask_width,
                          hyperparams[player].action_dim, seed + player)
        for player in (0, 1)
    ]

    # Round 0: no best response and no reservoir yet, so both roles start as the
    # untrained policy -- fictitious play has to respond to *something* first.
    br_policies = [so.initial_policy(game, player, hyperparams[player], seed + player)
                   for player in (0, 1)]
    average_policies = [so.initial_policy(game, player, hyperparams[player], seed + 10 + player)
                        for player in (0, 1)]
    average_params = [None, None]

    history = writer.history if writer is not None else []
    train_seconds = 0.0
    # One PPO iteration of a best response plays `num_envs` hands and reports the
    # mean length of exactly those hands, so the two best responses are charged
    # exactly; the reservoir rollouts report no length and are charged at the
    # running mean. The supervised fit touches no environment at all -- it reads
    # the reservoir -- so it costs wall time and nothing else.
    budget = Budget(default_episode_length=float(game.max_steps))

    for round_index in range(1, rounds + 1):
        round_started = time.monotonic()
        # Both sides use the strategies as they stood at the start of the round
        # (simultaneous, not alternating), which is what fictitious play prescribes.
        anticipatory = [so.mix([br_policies[player], average_policies[player]], [eta, 1.0 - eta])
                        for player in (0, 1)]

        responses = [
            so.train_sequential_best_response(
                game, player, anticipatory[1 - player], hyperparams[player],
                steps=br_steps, epochs=br_epochs, seed=seed + round_index)
            for player in (0, 1)
        ]
        br_policies = [responses[player].as_mixture(f"br(player {player})") for player in (0, 1)]

        # The supervised set: what this round's best response plays, at the
        # infosets the *current profile* reaches. `own_member=0` keeps only the
        # hands in which the learner drew `beta` -- see the module docstring.
        rows = []
        for player in (0, 1):
            key, data_key = jax.random.split(key)
            own = so.mix([br_policies[player], average_policies[player]], [eta, 1.0 - eta])
            rows.append(so.sample_decision_rows(
                game, player, own, anticipatory[1 - player], data_key, reservoir_episodes,
                own_member=0))
            reservoirs[player].add(rows[player])

        sl_losses = []
        for player in (0, 1):
            key, fit_key = jax.random.split(key)
            network, params, loss = fit_average_policy(
                game, player, hyperparams[player], reservoirs[player], sl_steps, sl_batch,
                fit_key, init_params=average_params[player])
            average_params[player] = params
            average_policies[player] = so.single(network, hyperparams[player], params,
                                                 f"average(player {player})")
            sl_losses.append(loss)
        train_seconds += time.monotonic() - round_started

        for player in (0, 1):
            budget.add_training(responses[player].history, hyperparams[player].num_envs)
        budget.add_episodes(2 * reservoir_episodes)

        entry = {
            "t": round_index,
            "wall_time": train_seconds,
            **budget.row(),
            "reservoir": reservoirs[0].size,
            "sl_loss_0": sl_losses[0], "sl_loss_1": sl_losses[1],
            "br_value_0": responses[0].br_value, "br_value_1": responses[1].br_value,
            # The optimizer's own numbers, averaged over each best response's
            # whole training: this round's oracle, not the average policy.
            **mean_columns(responses[0].history, prefix="br0_"),
            **mean_columns(responses[1].history, prefix="br1_"),
        }
        if scorer is not None:
            entry.update(scorer((average_policies[0], average_policies[1]), round_index,
                                force=(round_index == rounds)))
            # The best-response pair is logged because a *rising* `br_expl`
            # beside a falling `expl` is what NFSP working looks like.
            br_row = scorer((br_policies[0], br_policies[1]), round_index, full=False)
            # `expl` only: the scorer's `value` column would land beside
            # `br_value_0`/`br_value_1`, which are a different quantity (each
            # response's payoff against the mixture it was trained on).
            entry.update({f"br_{k}": v for k, v in br_row.items() if k == "expl"})
        (writer.record(entry) if writer is not None else history.append(entry))
        print_row(entry, ("br_expl", "expl_lb", "h2h", "reservoir", "sl_loss_0"))
        if checkpoint_fn is not None:
            checkpoint_fn(round_index, {
                **{CHECKPOINT_ENTRY.format(player): (hyperparams[player], average_params[player])
                   for player in (0, 1)},
                **{f"br_{player}": (hyperparams[player], br_policies[player].params[0])
                   for player in (0, 1)},
            })

    if writer is not None and writer.directory is not None:
        for player in (0, 1):
            writer.save_params(f"average_{player}", hyperparams[player], average_params[player])
            writer.save_params(f"br_{player}", hyperparams[player], br_policies[player].params[0])
        # ... and the averages again in the layout `best_response.py` reads, so a
        # finished run can be scored offline by the repo's own tool.
        save_checkpoint_step_multi(
            Path(writer.directory) / "average_checkpoint", rounds,
            {CHECKPOINT_ENTRY.format(player): (hyperparams[player], average_params[player])
             for player in (0, 1)})

    return {"history": history, "average": average_policies, "br": br_policies,
            "reservoirs": reservoirs, "train_seconds": train_seconds, "budget": budget}


def main() -> None:
    ap = sequential_parser(__doc__)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--br-steps", type=int, default=50, help="training chunks per best response")
    ap.add_argument("--br-epochs", type=int, default=20, help="iterations per chunk")
    ap.add_argument("--eta", type=float, default=0.1, help="anticipatory parameter")
    ap.add_argument("--reservoir-capacity", type=int, default=200_000)
    ap.add_argument("--reservoir-episodes", type=int, default=4096,
                    help="hands played per player per round for the supervised set; about "
                         "`eta` of them are best-response hands and contribute rows")
    ap.add_argument("--sl-steps", type=int, default=400, help="supervised steps per round")
    ap.add_argument("--sl-batch", type=int, default=256)
    args = ap.parse_args()

    game, game_config, config = load_sequential_run(args.config)
    hyperparams = tuple(so.br_hyperparams(game, player, config) for player in (0, 1))
    scorer = build_scorer(game, hyperparams, exact_grid=args.exact_grid,
                          score_every=args.score_every, br_steps=args.score_br_steps,
                          br_epochs=args.score_br_epochs, episodes=args.score_episodes,
                          seed=args.seed)
    print(f"game    : {type(game).__name__}  {dataclasses.asdict(game_config)}")
    print(f"solver  : sequential NFSP  {args.rounds} rounds  eta={args.eta}  "
          f"BR {args.br_steps}x{args.br_epochs} PPO iterations/round  "
          f"SL {args.sl_steps}x{args.sl_batch} on {args.reservoir_episodes} hands/round")
    print(f"metric  : `expl` is the AVERAGE policies' (exact, Kuhn only); `br_expl` the "
          f"best responses'; `expl_lb` an RL best-response bound\n")

    meta = {"algorithm": "sequential_nfsp", "config": args.config,
            **{k: v for k, v in vars(args).items() if k != "config"}}
    writer = RunWriter(args.checkpoint_dir, meta) if args.checkpoint_dir else None
    result = run_sequential_nfsp(
        game, config, rounds=args.rounds, br_steps=args.br_steps, br_epochs=args.br_epochs,
        eta=args.eta, reservoir_capacity=args.reservoir_capacity,
        reservoir_episodes=args.reservoir_episodes, sl_steps=args.sl_steps,
        sl_batch=args.sl_batch, seed=args.seed, scorer=scorer, writer=writer)

    last = result["history"][-1]
    expl = last.get("expl", last.get("expl_lb"))
    if expl is None:
        print("\nfinal average policies: not scored (no exact best response for this game; "
              "pass --score-every N for an RL bound)")
    else:
        label = "exploitability" if "expl" in last else "exploitability lower bound"
        print(f"\nfinal average-policy {label} {expl:+.5f}  |  {last['wall_time']:.1f}s training")
    if writer is not None:
        writer.finish({"final": last})
        print(f"run -> {writer.directory}")
        print(f"score it offline with: python best_response.py {args.config} "
              f"--checkpoint-dir {writer.directory}/average_checkpoint --responder both")
    elif args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": meta, "history": result["history"]}, indent=2))
        print(f"saved history -> {args.out}")


if __name__ == "__main__":
    main()
