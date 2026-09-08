"""Shared pieces of the neural baselines: config loading, the discretized policy
head, strategy snapshots, and the per-run output directory.

Two conventions here matter more than the code:

**One metric.** A neural policy is scored exactly like a tabular one -- by handing
`baselines.common.GridOracle` a finitely supported strategy. A discretized policy
gives one exactly (its grid and its probabilities); a continuous policy gives one by
sampling (`empirical_strategy`), which is an unbiased sample of the same object. So
`expl` means the same thing in every baseline in this repo, and the sampling error is
the only difference between a neural number and a tabular one. Note this is a
*stronger* metric than `ZeroSumGame.mixture_exploitability`, which best-responds by
gradient ascent from random restarts: the grid maximum is global, so it never
under-reports the deviation and never flatters the baseline.

**One checkpoint format.** Every run writes `baselines.common.StrategyPair` npz files,
plus the network weights next to them (`training.checkpoint`). The strategy files are
what later analysis compares; the weights are what a later run resumes or replays.

Note that importing `baselines.common` turns JAX's x64 mode on process-wide (the
tabular solvers need it). Network parameters are `float32` regardless -- flax's
`param_dtype` -- so what doubles is the activations and the payoff arithmetic, which at
these network sizes costs little and keeps the metric identical to the tabular one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from games.base import ZeroSumGame
from games.configs import GAME_CONFIGS
from nets.activations import Activation
from nets.normalization import Normalization
from training.checkpoint import save_checkpoint
from training.run_config import RunConfig, run_config_from_dict

from ..common import CheckpointWriter, GridOracle, StrategyPair, action_bounds, base_parser, \
    effective_support, tensor_grid, top_atoms

# Sections of a config file that belong to a solver other than these baselines
# (`run_idealized.py`'s legacy schema, a sweep driver); dropped rather than rejected so
# that one game config can be handed to every solver in the repo unchanged.
_FOREIGN_SECTIONS = ("mmd", "init", "log", "idealized", "sweep", "best_response")


def load_run(path: str | Path) -> tuple[ZeroSumGame, Any, RunConfig]:
    """`(game, game_config, run_config)` from any config in `configs/`.

    The `network:`/`optimizer:`/`ppo:` sections are read the same way `train.py` reads
    them, so a baseline's policy network is configured by the file that configures the
    method it is being compared against -- the comparison is otherwise not a
    comparison. Missing sections fall back to `RunConfig`'s defaults.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    shared = {k: v for k, v in raw.items() if k not in _FOREIGN_SECTIONS}
    if "game" not in shared:
        raise ValueError(f"{path}: config.game is required")
    if shared["game"].get("name") not in GAME_CONFIGS:
        raise ValueError(f"{path}: unknown game {shared['game'].get('name')!r}")
    run_config = run_config_from_dict(shared)
    game = run_config.game.build()
    return game, run_config.game, run_config


# --------------------------------------------------------------------------- policy


class DiscreteActorCritic(nn.Module):
    """Shared torso, two heads: logits over a fixed action set, and a value.

    The action set is not the module's business -- it holds `num_actions` logits and
    the caller says which action each index means (`action_grid`). That is what lets
    the same module serve as a discretized policy over a grid and as a supervised
    average policy over the same grid (`nfsp`).
    """

    num_actions: int
    hidden_dims: tuple[int, ...]
    activation: str = "tanh"
    normalization: str = "none"

    @nn.compact
    def __call__(self, obs: chex.Array, train: bool = False) -> tuple[chex.Array, chex.Array]:
        torso = obs
        for dim in self.hidden_dims:
            torso = nn.Dense(dim)(torso)
            torso = Normalization(kind=self.normalization)(torso, use_running_average=not train)
            torso = Activation(kind=self.activation)(torso)
        logits = nn.Dense(self.num_actions, name="logits_head")(torso)
        value = nn.Dense(1, name="value_head")(torso)
        return logits, jnp.squeeze(value, axis=-1)


def action_grid(game: ZeroSumGame, player: int, bins: int) -> np.ndarray:
    """`(bins**d, d)` discretization of `player`'s action box -- the categorical head's
    action set. Reuses the tabular baselines' grid so a discretized *policy* and the
    tabular solver's *strategy* live on the same points."""
    lo, hi = action_bounds(game, player)
    size = bins ** lo.shape[0]
    if size > 100_000:
        raise ValueError(
            f"a {bins}-bin discretization of a {lo.shape[0]}-D action space is {size} "
            "actions; that is the cost this baseline exists to demonstrate, but it will "
            "not fit in a categorical head. Lower --bins."
        )
    return tensor_grid(lo, hi, bins)


def discrete_probs(network: DiscreteActorCritic, params, obs: chex.Array) -> chex.Array:
    logits, _ = network.apply(params, obs)
    return jax.nn.softmax(logits)


def sample_discrete_actions(network: DiscreteActorCritic, params, obs: chex.Array,
                            grid: chex.Array, key: chex.PRNGKey, num_samples: int) -> chex.Array:
    """`num_samples` actions drawn from the categorical policy at a single `obs`."""
    logits, _ = network.apply(params, obs)
    idx = jax.random.categorical(key, logits, shape=(num_samples,))
    return jnp.asarray(grid)[idx]


# --------------------------------------------------------------------------- strategies


def empirical_strategy(actions) -> tuple[np.ndarray, np.ndarray]:
    """`(support, weights)` for a batch of sampled actions: the actions themselves,
    weighted uniformly.

    This is how a *continuous* policy is scored and checkpointed. It is an unbiased
    sample of the policy's own distribution, so the exploitability computed from it is
    the policy's up to Monte-Carlo error -- which shrinks like `1/sqrt(n)` and is worth
    reporting honestly rather than hiding behind a mode.
    """
    support = np.asarray(actions, dtype=np.float64)
    return support, np.full(support.shape[0], 1.0 / support.shape[0])


def grid_strategy(grid, probs) -> tuple[np.ndarray, np.ndarray]:
    """`(support, weights)` for a discretized policy: exact, no sampling error."""
    return np.asarray(grid, dtype=np.float64), np.asarray(probs, dtype=np.float64)


def strategy_row(oracle: GridOracle, s0, w0, s1, w1, radius: float | None = None) -> dict:
    """The metrics every neural baseline logs for a strategy pair."""
    radius = oracle.cluster_radius() if radius is None else radius
    return {
        "expl": float(oracle.exploitability(s0, w0, s1, w1)),
        "value": float(oracle.value(s0, w0, s1, w1)),
        "support_0": effective_support(w0, support=np.asarray(s0), radius=radius),
        "support_1": effective_support(w1, support=np.asarray(s1), radius=radius),
    }


# --------------------------------------------------------------------------- output


class RunWriter:
    """One run's output directory: history, checkpoints, network weights, metadata.

    Deliberately the same layout `experiments/one_shot_tabular/run_baselines.py` writes
    for the tabular baselines, plus a `params/` subtree the tabular runs have no
    counterpart for.
    """

    def __init__(self, directory: str | Path | None, meta: dict | None = None):
        self.directory = Path(directory) if directory else None
        self.meta = dict(meta or {})
        self.history: list[dict] = []
        self.checkpoints = CheckpointWriter(self.directory / "checkpoints") if self.directory else None

    def record(self, entry: dict, strategy: StrategyPair | None = None) -> dict:
        self.history.append(entry)
        if strategy is not None and self.checkpoints is not None:
            self.checkpoints(strategy)
        return entry

    def save_params(self, name: str, hyperparams, params) -> None:
        """One policy's weights, in `training.checkpoint`'s two-file format, under
        `params/<name>/` -- so a population member or an average net can be reloaded
        with the repo's own loader rather than a pickle of this script's objects."""
        if self.directory is None:
            return
        save_checkpoint(self.directory / "params" / name, hyperparams, params)

    def save_arrays(self, name: str, **arrays) -> None:
        """A bundle of plain arrays as `<name>.npz` -- what a strategy that is *not* a
        single network needs saved beside `params/`: a PSRO population's meta-weights and
        its empirical payoff matrix, say, which the parameters alone do not determine."""
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez(self.directory / f"{name}.npz", **{k: np.asarray(v) for k, v in arrays.items()})

    def finish(self, extra: dict | None = None) -> dict:
        meta = {**self.meta, **(extra or {})}
        if self.directory is None:
            return meta
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "history.json").write_text(json.dumps(self.history, indent=2))
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.checkpoints is not None:
            self.checkpoints.write_index(meta)
        return meta


def neural_parser(description: str):
    """`baselines.common.base_parser` plus the flags every neural baseline shares."""
    ap = base_parser(description)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4096,
                    help="actions sampled from a continuous policy when scoring it")
    return ap


def print_row(entry: dict, columns: tuple[str, ...] = ("value", "support_0", "support_1")) -> None:
    """One history row. `expl` is absent on a run that defers scoring (the fast path --
    see `experiments/one_shot_neural/`), so the cost columns lead instead."""
    parts = [f"t={entry['t']:7d}"]
    if "expl" in entry:
        parts.append(f"expl={entry['expl']:+9.5f}")
    if "wall_time" in entry:
        parts.append(f"{entry['wall_time']:7.1f}s")
    if "payoff_evals" in entry:
        parts.append(f"{entry['payoff_evals']:9.3g} evals")
    parts += [
        f"{c}={entry[c]:+9.5f}" if isinstance(entry.get(c), float) else f"{c}={entry[c]}"
        for c in columns if c in entry
    ]
    print("  " + "  ".join(parts))


def report_final(oracle: GridOracle, s0, w0, s1, w1) -> None:
    radius = oracle.cluster_radius()
    print(f"  P0 {top_atoms(s0, w0, radius=radius)}")
    print(f"  P1 {top_atoms(s1, w1, radius=radius)}")
