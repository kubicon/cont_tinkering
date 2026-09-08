"""What a sequential run cost, in one schema for every solver.

Self-play PPO, NFSP and PSRO spend their budget in shapes that have nothing to do
with each other -- one long stream of gradient steps, versus rounds of
best-response training with supervised fits or payoff-matrix play in between --
so "how far along is it?" has no common answer in the units any one of them
counts natively (chunks, rounds, PPO iterations). Two things are common, and are
what a comparison across the three has to be plotted against:

  * **wall time**, split into the training clock and the total. The split is not
    bookkeeping fussiness: measuring a strategy in a tree can cost as much as
    producing it (an RL best-response bound *is* two PPO runs), so a run that
    scores itself every round would otherwise look several times slower than the
    same algorithm scored offline.
  * **environment steps**, meaning decision nodes visited. `Budget` counts these
    exactly wherever the number is already known -- every PPO iteration reports
    the mean episode length of the batch it just played, so `num_envs *
    episode_length` is the real count, not an estimate -- and falls back to the
    running mean episode length for the rollouts that report no length of their
    own (PSRO's payoff matrix, NFSP's reservoir play). Those are a minority of
    any run's episodes and the mean is stable to a few percent, so the total is
    well inside the 10% an honest comparison needs. `episodes` (hands played) is
    exact throughout, and is the number to prefer where a game's episode length
    is roughly constant.

**Measurement is not charged to the algorithm.** Scoring episodes count in
neither `episodes` nor `env_steps`, and the scoring clock is excluded from
`wall_time` (it is in `total_wall_time`). That is the same convention
`baselines/neural/psro.py` uses for its one-shot `payoff_evals`, and it is what
makes a curve comparable between a run scored every round and one scored twice.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from training.checkpoint import save_checkpoint_step_multi

from .common import RunWriter

# Per-iteration keys that are bookkeeping rather than measurements, and so are
# not averaged into a log row.
_NOT_A_METRIC = ("iteration",)


def mean_columns(rows: Sequence[dict], prefix: str = "") -> dict[str, float]:
    """The mean of every numeric column of `rows`, optionally renamed with `prefix`.

    `rows` is a slice of a trainer's `history` -- the iterations since the last
    log point -- so what comes back is that interval's average loss, gradient
    norm, entropy and so on, not an instantaneous reading. An interval mean is
    the honest summary here: one PPO iteration's loss on a batch of 512 hands of
    poker is mostly noise, and the log point exists to be plotted.
    """
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in _NOT_A_METRIC and isinstance(rows[0][k], (int, float))]
    return {f"{prefix}{k}": float(np.mean([float(r[k]) for r in rows])) for k in keys}


@dataclasses.dataclass
class Budget:
    """Cumulative environment cost: iterations, episodes, decision nodes.

    `default_episode_length` seeds the running mean for episodes played before
    any training has reported one (PSRO prices its seed pair's payoff entry
    before its first best response); pass the game's `max_steps`, which is an
    upper bound and so cannot make the early estimate optimistic.
    """

    default_episode_length: float
    iterations: int = 0
    episodes: int = 0
    env_steps: float = 0.0
    _length_weight: float = 0.0
    _length_total: float = 0.0

    @property
    def mean_episode_length(self) -> float:
        """Decisions per episode, averaged over every batch actually played."""
        if self._length_weight <= 0:
            return float(self.default_episode_length)
        return self._length_total / self._length_weight

    def add_training(self, rows: Sequence[dict], num_envs: int,
                     length_key: str = "episode_length") -> "Budget":
        """Charge `rows` PPO iterations of `num_envs` episodes each -- exactly.

        Every row carries the mean episode length of its own batch, so this needs
        no estimate; it also feeds the running mean the estimated paths use.
        """
        lengths = [float(row[length_key]) for row in rows]
        self.iterations += len(rows)
        self.episodes += num_envs * len(rows)
        self.env_steps += num_envs * float(sum(lengths))
        self._length_weight += num_envs * len(rows)
        self._length_total += num_envs * float(sum(lengths))
        return self

    def add_iterations(self, count: int) -> "Budget":
        """Count `count` optimizer steps whose episodes were charged separately.

        For a solver whose iteration does not *own* a batch of episodes -- a
        zeroth-order step plays as many hands as its perturbations and utility
        evaluations dictate, not `num_envs` of them -- so the two counters cannot
        be advanced together the way `add_training` does it.
        """
        self.iterations += int(count)
        return self

    def add_episodes(self, episodes: int, episode_length: float | None = None) -> "Budget":
        """Charge `episodes` hands played outside training (no per-batch length reported).

        Priced at the running mean episode length unless one is given. This is
        the only estimated part of the count; see the module docstring.
        """
        length = self.mean_episode_length if episode_length is None else episode_length
        self.episodes += int(episodes)
        self.env_steps += float(episodes) * float(length)
        return self

    def row(self) -> dict[str, float]:
        """The three cost columns, for a history row."""
        return {"iterations": self.iterations, "episodes": self.episodes,
                "env_steps": float(self.env_steps)}


class Stopwatch:
    """Elapsed time while running, with the parts you stop it for left out.

    Used to keep measurement out of `wall_time`: a driver stops it around
    scoring and starts it again afterwards, so what it holds is training time.
    `SequentialRunLog` keeps a second one that is never stopped, and every row
    carries both.
    """

    def __init__(self, running: bool = False):
        self.elapsed = 0.0
        self._started: float | None = time.monotonic() if running else None

    def start(self) -> "Stopwatch":
        if self._started is None:
            self._started = time.monotonic()
        return self

    def stop(self) -> "Stopwatch":
        if self._started is not None:
            self.elapsed += time.monotonic() - self._started
            self._started = None
        return self

    @property
    def seconds(self) -> float:
        running = 0.0 if self._started is None else time.monotonic() - self._started
        return self.elapsed + running


class SequentialRunLog(RunWriter):
    """A run directory for `train_sequential.py`: rows, weights, and what it cost.

    A `RunWriter` (so `save_params`, `save_arrays` and `finish` mean exactly what
    they do for every other baseline) plus the two things a long comparison run
    needs:

      * **rows are streamed** to `metrics.jsonl` as they happen, so a run killed
        by a scheduler still has every log point it reached. `history.json` is
        written at the end (and holds the same rows).
      * **weights are checkpointed per log point** into `checkpoints/{t}.pkl`,
        the layout `training.checkpoint.load_checkpoint_step_multi` and
        `best_response.py` read -- so a finished run of *any* of the three
        solvers can be measured offline by the repo's own tool rather than by
        something that knows which algorithm produced it.
    """

    def __init__(self, directory: str | Path | None, meta: dict | None = None,
                 checkpoint_name: str = "checkpoints"):
        super().__init__(directory, meta)
        self.checkpoint_name = checkpoint_name
        # A sequential strategy is a policy, never a `StrategyPair`, so the
        # inherited npz checkpoint writer has nothing to write; dropping it keeps
        # it from putting an empty index beside the `{step}.pkl` files below.
        self.checkpoints = None
        self.clock = Stopwatch(running=True)
        self._stream: Path | None = None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._stream = self.directory / "metrics.jsonl"
            self._stream.write_text("")
            # Written up front, not just at `finish`: a run that dies still says
            # what it was.
            (self.directory / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    def record(self, entry: dict, strategy=None) -> dict:
        entry = {**entry, "total_wall_time": self.clock.seconds}
        super().record(entry, strategy)
        if self._stream is not None:
            with self._stream.open("a") as f:
                f.write(json.dumps(entry, default=float) + "\n")
        return entry

    def checkpoint(self, step: int, entries: dict, arrays: dict | None = None) -> None:
        """One log point's weights (`checkpoints/{step}.pkl`) and any loose arrays.

        `entries` is `{name: (hyperparams, params)}`. Naming both players
        `player_0`/`player_1` is what makes a checkpoint readable by
        `best_response.py`; PSRO cannot (its strategy is a population, so its
        entries are `player{p}_policy{k}` and the meta-weights ride along in
        `arrays`), and that asymmetry is a fact about the algorithm rather than
        about the format.
        """
        if self.directory is None:
            return
        save_checkpoint_step_multi(self.directory / self.checkpoint_name, step, entries)
        if arrays:
            np.savez(self.directory / self.checkpoint_name / f"{step}.npz",
                     **{k: np.asarray(v) for k, v in arrays.items()})

    def finish(self, extra: dict | None = None) -> dict:
        """`history.json` and the final `meta.json`.

        Written here rather than by `RunWriter.finish` for one reason: `default=`.
        A row picks up numpy scalars from wherever it was assembled, and a run
        that trained for six hours must not fail to write its history over a
        `float32` the plain encoder does not recognize.
        """
        meta = {**self.meta, **(extra or {})}
        if self.directory is None:
            return meta
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "history.json").write_text(
            json.dumps(self.history, indent=2, default=float))
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta
