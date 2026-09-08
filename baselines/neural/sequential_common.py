"""Config loading and CLI flags shared by the sequential baselines.

The one-shot counterpart is `baselines.neural.common`, and the split is not
cosmetic: everything in that module is organized around a strategy being a
*finitely supported distribution over actions* (`(support, weights)`, scored by
`baselines.common.GridOracle` on a tensor grid). In a game tree a strategy is a
policy over infosets, there is no action grid to lay down, and exploitability is
a tree traversal rather than a grid maximization -- so the two share the run
directory (`RunWriter`) and the log formatting (`print_row`) and nothing below
that.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from games.configs import GAME_CONFIGS
from games.sequential import SequentialZeroSumGame
from training.run_config import RunConfig, run_config_from_dict

from .common import _FOREIGN_SECTIONS


def load_sequential_run(path: str | Path) -> tuple[SequentialZeroSumGame, Any, RunConfig]:
    """`(game, game_config, run_config)` from any sequential config in `configs/`.

    Reads the same `network:`/`optimizer:`/`ppo:` sections `train.py` reads, so a
    baseline's policy network is configured by the very file that configures the
    method it is being compared against. A one-shot config is rejected here with
    a pointer to the one-shot baselines, which is the mirror image of the
    rejection `baselines.common.load_game` gives a sequential one.
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
    if not isinstance(game, SequentialZeroSumGame):
        raise ValueError(
            f"{type(game).__name__} is a one-shot game; these baselines play a game tree. "
            "Use `python -m baselines.neural.psro` / `.nfsp` instead."
        )
    if run_config.network.policy != "gaussian_mixture":
        raise ValueError(
            f"network.policy {run_config.network.policy!r} is one-shot only; a game tree needs "
            "the mixture policy's atoms and legality masks (see training/expfam.py's scope note)"
        )
    return game, run_config.game, run_config


def sequential_parser(description: str) -> argparse.ArgumentParser:
    """The CLI flags both sequential baselines share.

    Deliberately *not* `baselines.neural.common.neural_parser`: its `--grid` and
    `--samples` are the one-shot metric's knobs (how finely to grid the action
    box, how many actions to draw from a continuous policy), and neither has a
    meaning here. The scoring flags below replace them.
    """
    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", help="path to a sequential config in configs/ (kuhn, leduc, sequential_blotto)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the full history to this JSON path")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="write history.json, meta.json and params/<name>/ into this directory")

    scoring = ap.add_argument_group(
        "scoring",
        "How exploitability is measured. Kuhn has an exact tree best response and uses it "
        "every round for free; Leduc and sequential Blotto have none, so the only available "
        "measurement is an RL best response, which costs as much as a round of the algorithm "
        "itself and is therefore off by default and reported as the lower bound it is.",
    )
    scoring.add_argument("--exact-grid", type=int, default=None,
                         help="bet-grid points for the exact Kuhn best response (default: the config's)")
    scoring.add_argument("--score-every", type=int, default=0,
                         help="train an RL best response to the current strategy every N rounds "
                              "(0 = never); the only exploitability bound available off Kuhn")
    scoring.add_argument("--score-br-steps", type=int, default=50, help="chunks per scoring best response")
    scoring.add_argument("--score-br-epochs", type=int, default=20, help="iterations per scoring chunk")
    scoring.add_argument("--score-episodes", type=int, default=20_000,
                         help="episodes used to read each scoring best response's value")
    return ap
