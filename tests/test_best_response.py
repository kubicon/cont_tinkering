"""Checks for `training.best_response` and the `best_response.py` entry point.

The load-bearing test is `test_neural_best_response_matches_kuhns_exact_one`:
Kuhn is the only game here whose best response can also be computed exactly, so
it is the only place the *learned* one can be checked against ground truth
rather than against itself. Everything else pins properties that must hold for
that comparison to mean anything -- that the opponent really is frozen, that the
greedy readout is never applied to it, and that a best response to a strategy
with an obvious hole finds the hole.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import best_response as script
from games.kuhn_best_response import (
    bet_grid,
    best_response_value_first,
    best_response_value_second,
)
from games.leduc import ANTE, ContinuousLeducHoldem
from games.sequential_examples import ContinuousKuhnPoker
from training.best_response import (
    FrozenPolicy,
    SequentialBestResponseTrainer,
    evaluate_pair,
    latest_checkpoint_step,
    load_frozen_policy,
    policy_action_fn,
    validate_responder,
    warn_if_regularized,
)
from training.hyperparams import build_hyperparams
from training.kuhn_evaluation import strategy_from_network
from training.mixture import build_mixture_network
from training.checkpoint import save_checkpoint_step_multi
from training.run_config import run_config_from_dict
from training.sequential_trainer import SequentialSelfPlayPPOTrainer

# Every regularizer that pulls a responder away from the pure maximization it
# should be doing; see `warn_if_regularized`.
UNREGULARIZED = dict(
    trpo_category_kl_coef=0.0,
    trpo_gaussian_kl_coef=0.0,
    magnet_category_kl_coef=0.0,
    magnet_gaussian_kl_coef=0.0,
    category_entropy_coef=0.005,
    gaussian_entropy_coef=0.0,
)


def _config(path: str = "configs/kuhn_classic.yaml", **train_overrides):
    raw = yaml.safe_load(open(path))
    raw["train"].update(train_overrides)
    return run_config_from_dict(raw)


def _hyperparams(game, config, player: int, **overrides):
    return dataclasses.replace(
        build_hyperparams(game, player, config), **{**UNREGULARIZED, **overrides}
    )


def _always_passive_policy(game, hyperparams) -> FrozenPolicy:
    """A strategy that always plays the passive atom: it checks, and folds to any bet.

    Built by hand rather than trained, which is the point of `FrozenPolicy`
    taking `(network, params)` instead of a checkpoint path. Zeroing the logits
    head's kernel makes the categorical ignore the observation entirely, and the
    bias puts all the mass on `KIND_PASSIVE`.
    """
    network = build_mixture_network(hyperparams)
    key = jax.random.PRNGKey(0)
    params = network.init(key, game.observation(0, game.initial_state(key)))
    head = params["params"]["logits_head"]
    bias = jnp.zeros_like(head["bias"]).at[0].set(50.0)
    params = {
        "params": {
            **params["params"],
            "logits_head": {"kernel": jnp.zeros_like(head["kernel"]), "bias": bias},
        }
    }
    return FrozenPolicy(network=network, params=params, hyperparams=hyperparams)


# ---- the guards ----------------------------------------------------------


def test_greedy_is_per_player_and_a_bare_bool_is_refused():
    """Playing the *opponent* greedily measures a different strategy; see `evaluate_pair`."""
    game = ContinuousKuhnPoker()
    config = _config()
    policy = _always_passive_policy(game, _hyperparams(game, config, 0))

    with pytest.raises(TypeError, match="per player"):
        evaluate_pair(game, (policy, policy), 0, jax.random.PRNGKey(0), num_episodes=64, greedy=True)


def test_validate_responder_rejects_a_component_mismatch():
    game = ContinuousKuhnPoker()
    config = _config()
    opponent = _always_passive_policy(game, _hyperparams(game, config, 1, num_components=1))

    validate_responder(_hyperparams(game, config, 0, num_components=1), opponent)
    with pytest.raises(ValueError, match="num_components"):
        validate_responder(_hyperparams(game, config, 0, num_components=3), opponent)


def test_warn_if_regularized_names_the_coefficients_that_weaken_a_responder():
    game = ContinuousKuhnPoker()
    config = _config()
    assert warn_if_regularized(_hyperparams(game, config, 0)) == []

    regularized = _hyperparams(game, config, 0, magnet_category_kl_coef=0.2, trpo_gaussian_kl_coef=0.1)
    assert set(warn_if_regularized(regularized)) == {
        "magnet_category_kl_coef",
        "trpo_gaussian_kl_coef",
    }


def test_responder_players_parses_every_spelling():
    assert script.responder_players("both") == (0, 1)
    assert script.responder_players(0) == (0,)
    assert script.responder_players("1") == (1,)
    with pytest.raises(ValueError, match="0, 1 or 'both'"):
        script.responder_players(2)


# ---- the trainer ---------------------------------------------------------


def test_the_frozen_opponent_never_moves():
    """The whole method rests on this: a best response to a *moving* target measures nothing."""
    game = ContinuousKuhnPoker()
    config = _config()
    opponent = _always_passive_policy(game, _hyperparams(game, config, 1))
    before = jax.tree_util.tree_map(jnp.copy, opponent.params)

    trainer = SequentialBestResponseTrainer(
        game, opponent, _hyperparams(game, config, 0, num_envs=64), responder=0, seed=0
    )
    trainer.train(2, epochs=3)

    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_array_equal(a, b), trainer.opponent.params, before
    )
    # And the responder did move.
    assert not jnp.allclose(
        trainer.params["params"]["logits_head"]["bias"],
        trainer.state.target_params["params"]["logits_head"]["bias"],
    )


@pytest.mark.parametrize("responder", [0, 1])
@pytest.mark.parametrize(
    "game_kwargs, game_cls",
    [
        (dict(num_cards=3, min_bet=1.0, max_bet=1.0), ContinuousKuhnPoker),
        (dict(min_bet=1.0, max_bet=1.0, max_raises=1), ContinuousLeducHoldem),
    ],
)
def test_best_response_to_an_always_passive_opponent_wins_exactly_one_ante(
    responder, game_kwargs, game_cls
):
    """An opponent who never bets and always folds is worth exactly `ANTE` a hand.

    Bet, and they fold their ante; that is also the most they can ever lose,
    since they put nothing else in. So the best-response value is analytically
    `+1` in ante units for either player -- an exact target with no equilibrium
    theory behind it, and the cheapest end-to-end check that the responder
    really is maximizing.
    """
    game = game_cls(**game_kwargs)
    config = _config(steps=6, epochs=150)
    opponent = _always_passive_policy(game, _hyperparams(game, config, 1 - responder, num_components=1))
    hyperparams = _hyperparams(game, config, responder, num_components=1, num_envs=256)

    trainer = SequentialBestResponseTrainer(
        game, opponent, hyperparams, responder=responder, seed=0
    )
    trainer.train(config.train.steps, epochs=config.train.epochs)
    value = trainer.evaluate(jax.random.PRNGKey(3), num_episodes=20_000, greedy=True).value

    assert value == pytest.approx(ANTE, abs=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("responder", [0, 1])
def test_neural_best_response_matches_kuhns_exact_one(responder):
    """Ground truth: the same quantity, computed exactly by tabular traversal.

    `games.kuhn_best_response` best-responds to a Kuhn strategy read off the
    network, with no learning involved. The trained responder must reach that
    value -- and, being a lower bound, must not exceed it by more than sampling
    error. This is the only place in the repo where the learned best response
    can be checked against something that is not itself learned, which is why
    the Leduc path is worth trusting only as far as this test holds.
    """
    game = ContinuousKuhnPoker(num_cards=3, min_bet=1.0, max_bet=1.0)
    config = _config(steps=8, epochs=200)

    # The strategy under test: a freshly initialized self-play pair. Untrained is
    # ideal here -- it is far from equilibrium, so the best-response value is
    # large and a responder that quietly does nothing cannot pass.
    self_play = SequentialSelfPlayPPOTrainer(
        game, _hyperparams(game, config, 0), _hyperparams(game, config, 1), seed=0
    )
    opponent_player = 1 - responder
    opponent = FrozenPolicy(
        network=self_play.networks[opponent_player],
        params=self_play.params[opponent_player],
        hyperparams=_hyperparams(game, config, opponent_player),
    )

    grid = bet_grid(game)
    strategy = strategy_from_network(
        game, opponent.network, opponent.params, opponent_player, grid
    )
    exact = float(
        best_response_value_first(game, grid, strategy)
        if responder == 0
        else best_response_value_second(game, grid, strategy)
    )

    trainer = SequentialBestResponseTrainer(
        game,
        opponent,
        _hyperparams(game, config, responder, num_envs=512),
        responder=responder,
        seed=1,
    )
    trainer.train(config.train.steps, epochs=config.train.epochs)
    learned = trainer.evaluate(jax.random.PRNGKey(5), num_episodes=100_000, greedy=True)

    assert learned.value == pytest.approx(exact, abs=0.03)
    # A lower bound may fall short; it may not run away upwards.
    assert learned.value < exact + 4 * learned.stderr


# ---- loading a strategy off disk -----------------------------------------


def test_load_frozen_policy_round_trips_a_self_play_checkpoint(tmp_path):
    game = ContinuousKuhnPoker()
    config = _config()
    self_play = SequentialSelfPlayPPOTrainer(
        game, _hyperparams(game, config, 0), _hyperparams(game, config, 1), seed=0
    )
    self_play.save(tmp_path, 0)
    self_play.save(tmp_path, 7)

    assert latest_checkpoint_step(tmp_path) == 7
    for player in (0, 1):
        loaded = load_frozen_policy(tmp_path, 7, player)
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_array_equal(a, b), loaded.params, self_play.params[player]
        )
        # The architecture is rebuilt from the checkpoint, not from any config.
        assert loaded.hyperparams.num_components == self_play.hyperparams[player].num_components
        obs = game.observation(player, game.initial_state(jax.random.PRNGKey(0)))
        assert loaded.network.apply(loaded.params, obs)[0].shape == (
            loaded.hyperparams.num_atoms + loaded.hyperparams.num_components,
        )


def test_load_frozen_policy_reads_the_target_iterate_too(tmp_path):
    """The averaged iterate is a different strategy, and a checkpoint now carries both."""
    game = ContinuousKuhnPoker()
    config = _config()
    self_play = SequentialSelfPlayPPOTrainer(
        game, _hyperparams(game, config, 0), _hyperparams(game, config, 1), seed=0
    )
    # One chunk of training, so the Polyak average has actually fallen behind
    # the live params and the two are distinguishable.
    self_play.train(1, epochs=4)
    self_play.save(tmp_path, 1)

    live = load_frozen_policy(tmp_path, 1, 0)
    target = load_frozen_policy(tmp_path, 1, 0, target=True)
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_array_equal(a, b), target.params, self_play.target_params[0]
    )
    assert not jnp.allclose(
        live.params["params"]["logits_head"]["bias"],
        target.params["params"]["logits_head"]["bias"],
    )


def test_asking_for_a_target_iterate_an_old_checkpoint_lacks_raises(tmp_path):
    """Substituting the live params for the averaged ones would mislabel the measurement."""
    game = ContinuousKuhnPoker()
    config = _config()
    self_play = SequentialSelfPlayPPOTrainer(
        game, _hyperparams(game, config, 0), _hyperparams(game, config, 1), seed=0
    )
    # A checkpoint in the old format: live params only.
    save_checkpoint_step_multi(
        tmp_path,
        2,
        {f"player_{p}": (self_play.hyperparams[p], self_play.params[p]) for p in (0, 1)},
    )

    load_frozen_policy(tmp_path, 2, 0)  # the live iterate is still readable
    with pytest.raises(KeyError, match="predates target params"):
        load_frozen_policy(tmp_path, 2, 0, target=True)


def test_load_frozen_policy_rejects_a_missing_entry(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_checkpoint_step(tmp_path)
    with pytest.raises(ValueError, match="player must be"):
        load_frozen_policy(tmp_path, 0, 2)


def test_opponent_iterates_parses_every_spelling():
    assert script.opponent_iterates("both") == ("live", "target")
    assert script.opponent_iterates("live") == ("live",)
    assert script.opponent_iterates("target") == ("target",)
    with pytest.raises(ValueError, match="'live', 'target' or 'both'"):
        script.opponent_iterates("averaged")


# ---- the entry point -----------------------------------------------------


def test_the_shipped_best_response_configs_load():
    for path in ("configs/kuhn_br.yaml", "configs/leduc_br.yaml"):
        config = run_config_from_dict(yaml.safe_load(open(path)))
        # The point of these files: no regularizer may weaken the responder.
        assert warn_if_regularized(build_hyperparams(config.game.build(), 0, config)) == []


def test_a_short_run_completes_through_the_entry_point(tmp_path, capsys):
    """`configs/leduc_br.yaml` end to end, against a checkpoint written here."""
    config = _config("configs/leduc.yaml")
    game = config.game.build()
    self_play = SequentialSelfPlayPPOTrainer(
        game, build_hyperparams(game, 0, config), build_hyperparams(game, 1, config), seed=0
    )
    self_play.save(tmp_path, 1)

    raw = yaml.safe_load(open("configs/leduc_br.yaml"))
    raw["train"].update(steps=1, epochs=2)
    raw["best_response"].update(
        checkpoint_dir=str(tmp_path),
        eval_episodes=2_000,
        progress_episodes=1_000,
        opponent_iterate="both",
    )
    run_config = run_config_from_dict(raw)
    settings = run_config.best_response

    results = {
        (iterate, responder): script.run_one_direction(game, run_config, responder, 1, iterate)
        for iterate in script.opponent_iterates(settings.opponent_iterate)
        for responder in script.responder_players(settings.responder)
    }
    script.report(results)

    printed = capsys.readouterr().out
    assert "exploitability(live) >=" in printed
    assert "exploitability(target) >=" in printed
    assert set(results) == {("live", 0), ("live", 1), ("target", 0), ("target", 1)}
    assert all(set(v) == {"sampled", "greedy"} for v in results.values())


def test_the_entry_point_refuses_a_one_shot_game(monkeypatch):
    monkeypatch.setattr("sys.argv", ["best_response.py", "configs/quadratic.yaml"])
    with pytest.raises(ValueError, match="sequential game"):
        script.main()


def test_a_policy_action_fn_only_ever_plays_legal_kinds():
    """Evaluation goes through `play_episode`, so its action fn must respect the mask."""
    game = ContinuousLeducHoldem()
    config = _config("configs/leduc.yaml")
    policy = _always_passive_policy(game, _hyperparams(game, config, 0))
    space = game.action_space(0)

    for greedy in (False, True):
        action_fn = policy_action_fn(policy.network, policy.params, space, greedy=greedy)
        state = game.initial_state(jax.random.PRNGKey(0))
        mask, obs = game.action_mask(0, state), game.observation(0, state)
        actions = jax.vmap(lambda k: action_fn(obs, mask, k))(jax.random.split(jax.random.PRNGKey(1), 256))
        assert bool(jnp.all(mask[actions.kind]))
        assert bool(jnp.all(space.contains(actions)))
