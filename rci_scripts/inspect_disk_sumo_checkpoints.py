"""Print what a disk-sumo self-play checkpoint plays: how its bouts end and whether it mixes.

For each requested step, both players' policies are rolled out against each
other and two things are read off the sampled decisions:

  * outcomes -- how many bouts somebody wins, against how many run out the clock;
  * the mixture -- per component, how often it is picked, where its mean sits
    and how wide it is, and then how much of the policy's spread at a state
    comes from the components *disagreeing* rather than from their own noise.

    python rci_scripts/inspect_disk_sumo_checkpoints.py data/disk_sumo_delayed --steps 0 10

The last number is the one that answers "is it mixing": a categorical head at
full entropy over components that all sit on the same mean is one Gaussian
written four times, not a mixed strategy.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from games.sequential import TERMINAL  # noqa: E402

from training.checkpoint import load_checkpoint_step_multi, target_entry  # noqa: E402
from training.config import MixturePPOHyperparams  # noqa: E402
from training.mixture import build_mixture_network  # noqa: E402
from training.run_config import CONFIG_FILENAME, load_run_config  # noqa: E402
from training.sequential_rollout import build_episode_sampler, collect_sequential_batch  # noqa: E402


def weighted(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Mean of `values` over the decisions `weight` selects; trailing axes are kept."""
    weight = weight.reshape(weight.shape + (1,) * (values.ndim - weight.ndim))
    return (values * weight).sum(axis=(0, 1)) / weight.sum()


def describe_player(batch, player: int) -> list[str]:
    own = np.asarray(batch.actor == player, dtype=np.float64)  # (N, T)
    logits = np.asarray(batch.logits, dtype=np.float64)
    means = np.asarray(batch.means, dtype=np.float64)  # (N, T, K, d)
    scale_trils = np.asarray(batch.scale_trils, dtype=np.float64)  # (N, T, K, d, d)
    num_components = means.shape[-2]

    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(-1)
    sigmas = np.sqrt((scale_trils ** 2).sum(-1))  # (N, T, K, d) per-axis std of each component

    mean_probs = weighted(probs, own)
    mean_means = weighted(means, own)
    # How far a component's mean moves from state to state: zero is a constant.
    mean_spread = np.sqrt(weighted((means - mean_means) ** 2, own))
    mean_sigmas = weighted(sigmas, own)

    lines = [f"  player {player}"]
    lines.append(
        f"    category entropy {float(weighted(entropy, own)):.4f} per state "
        f"(max {np.log(num_components):.4f}), "
        f"{float(-(mean_probs * np.log(np.clip(mean_probs, 1e-12, None))).sum()):.4f} of the average"
    )
    lines.append("    k | picked | mean (avg over states)  | mean moves by (std)     | sigma")
    for k in range(num_components):
        lines.append(
            f"    {k} | {mean_probs[k]:6.3f} | "
            + " ".join(f"{v:+.3f}" for v in mean_means[k]).ljust(23) + " | "
            + " ".join(f"{v:.3f}" for v in mean_spread[k]).ljust(23) + " | "
            + " ".join(f"{v:.3f}" for v in mean_sigmas[k])
        )

    # The policy's variance at one state splits into the components' own noise
    # and the spread of their means around the mixture mean.
    mixture_mean = (probs[..., None] * means).sum(-2, keepdims=True)
    between = (probs * ((means - mixture_mean) ** 2).sum(-1)).sum(-1)
    within = (probs * (sigmas ** 2).sum(-1)).sum(-1)
    share = weighted(between / (between + within), own)
    lines.append(
        f"    variance at a state: {float(weighted(within, own)):.4f} from component noise, "
        f"{float(weighted(between, own)):.4f} from the means disagreeing "
        f"({100 * float(share):.1f}% of it)"
    )

    # Distance between two components' means, in units of their widths: well
    # under 1 and the two are the same action; several and they are distinct.
    typical_sigma = np.sqrt((sigmas ** 2).mean(-1))  # (N, T, K)
    separations = []
    for i, j in combinations(range(num_components), 2):
        distance = np.linalg.norm(means[..., i, :] - means[..., j, :], axis=-1)
        scale = np.sqrt((typical_sigma[..., i] ** 2 + typical_sigma[..., j] ** 2) / 2)
        separations.append(f"{i}{j}:{float(weighted(distance / scale, own)):.2f}")
    lines.append("    mean separation in sigmas, per pair: " + "  ".join(separations))

    clipped = np.any(np.asarray(batch.raw_action) != np.asarray(batch.action_value), axis=-1)
    played = np.asarray(batch.action_value, dtype=np.float64)
    lines.append(
        f"    {100 * float(weighted(clipped.astype(np.float64), own)):.1f}% of sampled forces are "
        f"clipped to the box; played force averages "
        + " ".join(f"{v:+.3f}" for v in weighted(played, own))
        + f", magnitude {float(weighted(np.linalg.norm(played, axis=-1), own)):.3f}"
    )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint_dir", help="directory `train.py` wrote, holding `{step}.pkl` and `config.yaml`")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--config", default=None, help="run config; defaults to the checkpoint directory's copy")
    ap.add_argument("--target", action="store_true", help="inspect the Polyak (target) iterate")
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    config = load_run_config(args.config or checkpoint_dir / CONFIG_FILENAME)
    game = config.game.build()

    for step in args.steps:
        entries = load_checkpoint_step_multi(checkpoint_dir, step, hyperparams_cls=MixturePPOHyperparams)
        networks, params = [], []
        for player in (0, 1):
            name = f"player_{player}"
            hyperparams, p = entries[target_entry(name) if args.target else name]
            networks.append(build_mixture_network(hyperparams))
            params.append(p)

        sample_episode = build_episode_sampler(game, networks[0], networks[1])
        batch, payoff = jax.jit(
            lambda key: collect_sequential_batch(
                sample_episode, params[0], params[0], params[1], params[1], key, args.episodes
            )
        )(jax.random.PRNGKey(args.seed))
        payoff = np.asarray(payoff)
        lengths = np.asarray(batch.actor != TERMINAL).sum(-1)

        print(f"=== step {step} ({'target' if args.target else 'live'}), {args.episodes} bouts ===")
        print(
            f"  player 0 wins {100 * np.mean(payoff > 0):.2f}%, player 1 wins {100 * np.mean(payoff < 0):.2f}%, "
            f"nobody {100 * np.mean(payoff == 0):.2f}% | mean payoff {payoff.mean():+.4f} "
            f"| mean length {lengths.mean():.2f} of {game.max_steps}"
        )
        for player in (0, 1):
            print("\n".join(describe_player(batch, player)))
        print()


if __name__ == "__main__":
    main()
