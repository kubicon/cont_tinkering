"""Recompute exploitability from the stored checkpoints of the magnet grid, and plot it.

    python experiments/failing_gaussian_wo_magnet/rerun/analyze.py [--no-std] [options]

The 18 cells of `run_all.sh` write their iterates to disk but each reports
exploitability in its own units: the exact solvers integrate on their own quadrature
grid, the PPO cells Monte-Carlo sample it. This reads the *strategies* back out of
every checkpoint and re-scores them all with one measure, so a magnet and a no-magnet
curve -- and an `idealized`, a `sampled` and a `ppo` curve -- are the same quantity.

`--no-std` answers a different question: with the Gaussian spread set to zero, each
player is the point mass at its component means, so the number is the exploitability
of *where the policy is pointing* rather than of the distribution it actually plays.
For a game whose Nash is a point mass the two agree at convergence; away from it they
come apart, and a run whose means sit on the Nash while its spread is pinned at the
entropy floor reads as converged here and not in the default measure.

The per-cell result is written next to that cell's checkpoints as `exploitability.pkl`
(or `exploitability_no_std.pkl`), so the statistics can be redone later without
touching the checkpoints again. A cell that already has its file is loaded rather than
rescored -- rerunning to redraw a plot costs seconds, not the full pass -- so
`--force` is what recomputes one. Plots go to `--out-dir`.

The measure, for both modes, is the one the exact solvers already use: best responses
are maximized over the in-box nodes of a tensor-product grid, so it inherits that
grid's resolution (`--grid-points`, defaulting to each cell's own
`idealized.grid_points`). The two modes differ in what they need from it. With the
spread, the payoff matrix over all node pairs must be formed, which is `n^(2d)`
entries and is what caps the rotation games at 81 (d=2) and 21 (d=3) points per axis.
With the spread zeroed only `n^d` payoff evaluations are needed, so `--grid-points`
can be raised a long way for a sharper best response -- at the cost of no longer being
comparable to a default-mode run, which is why it is not raised by default.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

# Imported for its side effect as much as its contents: `run_idealized` enables
# jax x64 at import, which the quadrature payoff matrix is built in.
from run_idealized import (                                   # noqa: E402
    Params,
    QuadratureBackend,
    _bounds,
    _log_prob_at,
    _density_grid,
    _std_max,
    load_config,
)
from training.checkpoint import load_checkpoint_step_multi, target_entry  # noqa: E402
from training.config import MixturePPOHyperparams              # noqa: E402
from training.mixture import build_mixture_network             # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data" / "failing_gaussian_rerun"

DOMAINS = ("mp", "rot2", "rot3")
ENGINES = ("idealized", "sampled", "ppo")
MAGNETS = ("magnet", "nomagnet")

DOMAIN_LABEL = {
    "mp": "ContinuousMatchingPennies (d=1)",
    "rot2": "CoupledRotationGame (d=2)",
    "rot3": "CoupledRotationGame (d=3)",
}
ENGINE_LABEL = {
    "idealized": "idealized (exact, quadrature payoffs)",
    "sampled": "idealized (sampled payoffs)",
    "ppo": "PPO (neural network)",
}
MAGNET_LABEL = {"magnet": "with magnet", "nomagnet": "without magnet"}
MAGNET_COLOR = {"magnet": "#1f77b4", "nomagnet": "#d62728"}


def cell_name(domain: str, magnet: str, engine: str) -> str:
    return f"{domain}_{magnet}_{engine}"


# --------------------------------------------------------------- strategies on disk


class Strategy:
    """One player's mixture at one checkpoint: weights, means, and a Cholesky factor.

    `scale_tril` is `None` when the checkpoint did not store enough to rebuild the
    covariance -- see `_strategy_from_record`. It is never read in `--no-std` mode.
    """

    __slots__ = ("w", "means", "scale_tril")

    def __init__(self, w, means, scale_tril):
        self.w = np.asarray(w, dtype=np.float64)                    # (K,)
        self.means = np.asarray(means, dtype=np.float64)            # (K, d)
        self.scale_tril = None if scale_tril is None else np.asarray(scale_tril, dtype=np.float64)

    def params(self) -> Params:
        if self.scale_tril is None:
            raise ValueError("this checkpoint has no covariance; only --no-std can score it")
        return Params(
            logits=jnp.log(jnp.asarray(self.w) + 1e-300),
            means=jnp.asarray(self.means),
            scale_tril=jnp.asarray(self.scale_tril),
        )


def _strategy_from_record(record: dict, player: int) -> Strategy:
    """One player's strategy out of an `idealized_history.json` entry.

    The history stores the *marginal* standard deviations and, separately, the largest
    absolute off-diagonal correlation per component -- not the factor itself. A
    diagonal factor is therefore the most that can be rebuilt, which is exact for
    `mp` (`d == 1`, where a 1x1 factor is the standard deviation) and drops the
    correlations for the `full_covariance` rotation cells. `--report-drift` quantifies
    what that costs by comparing against the exploitability the run itself recorded;
    `--no-std` is unaffected, since it discards the covariance anyway.
    """
    std = np.asarray(record[f"std{player}"], dtype=np.float64)      # (K, d)
    scale_tril = np.stack([np.diag(s) for s in std])                # (K, d, d)
    return Strategy(record[f"w{player}"], record[f"means{player}"], scale_tril)


def _load_exact_cell(cell_dir: Path) -> dict:
    """Every logged iterate of an `idealized`/`sampled` cell.

    These solvers keep no target-iterate *parameters* -- `run()` records the averaged
    iterate's exploitability but not the iterate itself -- so only the live strategy
    can be rescored. The run's own numbers ride along for comparison.
    """
    history = json.loads((cell_dir / "idealized_history.json").read_text())
    return {
        "steps": [i for i, _ in enumerate(history)],
        "t": [int(r["t"]) for r in history],
        "live": [(_strategy_from_record(r, 0), _strategy_from_record(r, 1)) for r in history],
        "target": None,
        "expl_run": [float(r["expl"]) for r in history],
        "target_expl_run": [float(r["target_expl"]) for r in history],
    }


def _load_ppo_cell(cell_dir: Path, game, epochs: int) -> dict:
    """Every `{step}.pkl` of a PPO cell, live and Polyak-averaged.

    A checkpoint holds network parameters, not a strategy, so each one is pushed
    through the policy head at the run's (constant, all-zero) observation. Unlike the
    exact cells these do carry the full Cholesky factor, correlations included.
    """
    steps = sorted(int(p.stem) for p in cell_dir.glob("*.pkl") if p.stem.isdigit())
    if not steps:
        raise FileNotFoundError(f"no {{step}}.pkl checkpoints in {cell_dir}")

    entries = load_checkpoint_step_multi(cell_dir, steps[0], hyperparams_cls=MixturePPOHyperparams)
    heads = {}
    for player, name in ((0, "player_1"), (1, "player_2")):
        hyperparams, _ = entries[name]
        network = build_mixture_network(hyperparams)
        # The observation these one-shot games hand a policy is a constant (all
        # zeros, `ZeroSumGame.observation`), so the whole strategy lives at this
        # single input -- the same one the trainer logged and sampled at.
        obs = game.observation(player, jax.random.PRNGKey(0))
        heads[player] = jax.jit(lambda p, net=network, o=obs: net.apply(p, o)[:3])

    def strategy(player: int, params) -> Strategy:
        logits, means, scale_trils = jax.device_get(heads[player](params))
        weights = np.exp(logits - logits.max())
        return Strategy(weights / weights.sum(), means, scale_trils)

    live, target = [], []
    for step in steps:
        entries = load_checkpoint_step_multi(cell_dir, step, hyperparams_cls=MixturePPOHyperparams)
        live.append(tuple(strategy(player, entries[name][1])
                          for player, name in ((0, "player_1"), (1, "player_2"))))
        target.append(tuple(
            strategy(player, entries[target_entry(name)][1] if target_entry(name) in entries
                     else entries[name][1])
            for player, name in ((0, "player_1"), (1, "player_2"))
        ))

    return {
        "steps": steps,
        "t": [s * epochs for s in steps],
        "live": live,
        "target": target,
        "expl_run": None,
        "target_expl_run": None,
    }


# ------------------------------------------------------------------------ measures


class SpreadMeasure:
    """Exploitability of the two mixtures as they are, on a quadrature grid.

    Built on the solver's own `QuadratureBackend` -- its grid, its payoff matrix, its
    best-response rule -- so the number means what the `idealized` cells' logged number
    means. The one departure is how a component's density is normalized.

    `QuadratureBackend.handle` exponentiates the log-density and then divides by its
    sum. That is fine while a component is wide enough for the grid to see it, and
    silently catastrophic when it is not: a component whose standard deviation is well
    under the node spacing has a log-density around `-(delta/sigma)^2 / 2` at every
    node, which underflows to zero everywhere, so the sum is zero, the guarded divide
    returns the all-zero "density", and the exploitability of a maximally exploitable
    point mass comes back as exactly 0.0. The collapsed no-magnet PPO cells land
    exactly there -- they end at `sd 0.001` against a node spacing of 0.31, and 116 of
    `rot3_nomagnet_ppo`'s 201 checkpoints scored 0.0 before this.

    Subtracting each component's own maximum before exponentiating removes the
    underflow. It is algebraically the same normalization wherever the old one worked
    (the factor cancels), so nothing that was already correct moves; where it did not,
    a component far narrower than the grid now becomes what it should be -- unit mass
    on the nearest node, the point-mass limit.
    """

    needs_matrix = True

    def __init__(self, game, grid_points: int, std_max, max_gib: float):
        self.backend = QuadratureBackend(game, grid_points, std_max, normalize=True, max_gib=max_gib)
        nodes, dv = self.backend.grid, self.backend.dv

        @jax.jit
        def handle(params: Params):
            log_p = jax.vmap(_log_prob_at, in_axes=(None, 0, 0))(nodes, params.means,
                                                                 params.scale_tril)  # (K, M)
            log_p = log_p - jnp.max(log_p, axis=-1, keepdims=True)
            comp = jnp.exp(log_p)
            comp = comp / (jnp.sum(comp, axis=-1, keepdims=True) * dv)
            return jnp.sum(jax.nn.softmax(params.logits)[:, None] * comp, axis=0)      # (M,)

        self._handle = handle

    def __call__(self, s0: Strategy, s1: Strategy) -> float:
        h0 = self._handle(s0.params())
        h1 = self._handle(s1.params())
        return float(self.backend.exploitability(h0, h1))


class PointMassMeasure:
    """Exploitability with every component's spread set to zero.

    Each player becomes the discrete distribution putting `w_k` on its `k`-th mean, so
    the expected payoff is a finite sum and no density has to be integrated. The best
    responses are still maximized over the same in-box grid nodes the spread measure
    uses, which is what keeps the two comparable at equal `--grid-points`:

        expl = max_{a in box} sum_l w1_l payoff(a, mu1_l)
             - min_{a in box} sum_k w0_k payoff(mu0_k, a)

    The two `U` terms of the usual definition cancel, so the value never depends on
    the profile's own payoff. Note the means are used at full precision -- they are
    *not* snapped to a grid node, which at 21 nodes per axis would be an error of
    0.31 in each coordinate.
    """

    needs_matrix = False

    def __init__(self, game, grid_points: int, std_max, max_gib: float):
        lo, hi = _bounds(game)
        grid, _, _ = _density_grid(lo, hi, grid_points, std_max)
        in_box = jnp.all((grid >= jnp.asarray(lo)) & (grid <= jnp.asarray(hi)), axis=-1)
        self.nodes = grid[in_box]                                   # (M_in, d)

        payoff = game.payoff
        # `value_0[i]` is what player 0 gets from pure action `nodes[i]` against the
        # opponent's atoms; `value_1[j]` is what player 0 concedes at `nodes[j]`.
        self._value_0 = jax.jit(lambda mu, w: jnp.einsum(
            "l,il->i", w, jax.vmap(lambda a: jax.vmap(lambda b: payoff(a, b))(mu))(self.nodes)))
        self._value_1 = jax.jit(lambda mu, w: jnp.einsum(
            "k,kj->j", w, jax.vmap(lambda a: jax.vmap(lambda b: payoff(a, b))(self.nodes))(mu)))

    def __call__(self, s0: Strategy, s1: Strategy) -> float:
        best_0 = jnp.max(self._value_0(jnp.asarray(s1.means), jnp.asarray(s1.w)))
        best_1 = jnp.min(self._value_1(jnp.asarray(s0.means), jnp.asarray(s0.w)))
        return float(best_0 - best_1)


# ----------------------------------------------------------------------- the driver


def _config_path(cell: str) -> Path:
    return CONFIG_DIR / f"{cell}.yaml"


def build_measure(cell: str, no_std: bool, grid_points: int | None, max_gib: float):
    """The scoring function for `cell`'s game, plus the grid resolution it ended up at."""
    game_config, solver_config, _ = load_config(_config_path(cell))
    game = game_config.build()
    n = grid_points if grid_points is not None else solver_config.idealized.grid_points
    std_max = _std_max(game, solver_config)
    measure = (PointMassMeasure if no_std else SpreadMeasure)(game, n, std_max, max_gib)
    # `chunks` is the per-logged-step iteration count; every cell here logs uniformly,
    # so the first entry is the `train.epochs` that turns a step index into an iteration.
    epochs = solver_config.chunks[0] if solver_config.chunks else 1
    return measure, game, n, epochs


def score_cell(cell: str, measure, cell_data: dict) -> dict:
    """Every checkpoint of one cell, scored."""
    live = [measure(s0, s1) for s0, s1 in cell_data["live"]]
    target = (None if cell_data["target"] is None
              else [measure(s0, s1) for s0, s1 in cell_data["target"]])
    return {"expl_live": live, "expl_target": target}


def result_filename(no_std: bool) -> str:
    return "exploitability_no_std.pkl" if no_std else "exploitability.pkl"


def _load_saved(out_path: Path, grid_points: int | None) -> dict | None:
    """A previously saved result for this cell, if it is there and still applicable.

    Rescoring 201 checkpoints against a quadrature grid is the expensive half of this
    script, and it is pure -- the same checkpoints and the same grid give the same
    numbers -- so a saved result is reused rather than recomputed. Two things can make
    a saved file inapplicable: an explicit `--grid-points` that disagrees with the one
    it was computed at, and a file that cannot be read at all (a run killed mid-write).
    Both fall back to recomputing rather than failing.
    """
    if not out_path.exists():
        return None
    try:
        with out_path.open("rb") as f:
            saved = pickle.load(f)
    except Exception as error:
        print(f"    ({out_path.name} unreadable: {error}; recomputing)")
        return None
    if grid_points is not None and saved.get("grid_points") != grid_points:
        print(f"    ({out_path.name} was computed at grid {saved.get('grid_points')}/axis, "
              f"--grid-points asks for {grid_points}; recomputing)")
        return None
    return saved


def compute(cell: str, no_std: bool, grid_points: int | None, max_gib: float,
            measure_cache: dict, force: bool) -> dict:
    """Score one cell, reusing a cached measure and an on-disk result where possible."""
    cell_dir = DATA_ROOT / cell
    out_path = cell_dir / result_filename(no_std)

    if not force:
        saved = _load_saved(out_path, grid_points)
        if saved is not None:
            print(f"  {cell}: loaded {out_path.name} "
                  f"({len(saved['expl_live'])} checkpoints, grid {saved['grid_points']}/axis)")
            return saved

    domain, _, engine = cell.split("_")
    key = (domain, no_std, grid_points)
    if key not in measure_cache:
        measure_cache[key] = build_measure(cell, no_std, grid_points, max_gib)
    measure, game, n, epochs = measure_cache[key]

    data = (_load_ppo_cell(cell_dir, game, epochs) if engine == "ppo"
            else _load_exact_cell(cell_dir))
    scored = score_cell(cell, measure, data)

    result = {
        "cell": cell,
        "domain": domain,
        "engine": engine,
        "magnet": cell.split("_")[1],
        "no_std": no_std,
        "grid_points": n,
        "steps": data["steps"],
        "t": data["t"],
        "expl_live": scored["expl_live"],
        "expl_target": scored["expl_target"],
        "expl_run": data["expl_run"],
        "target_expl_run": data["target_expl_run"],
        "means_0": [np.asarray(s0.means) for s0, _ in data["live"]],
        "means_1": [np.asarray(s1.means) for _, s1 in data["live"]],
        "weights_0": [np.asarray(s0.w) for s0, _ in data["live"]],
        "weights_1": [np.asarray(s1.w) for _, s1 in data["live"]],
    }
    with out_path.open("wb") as f:
        pickle.dump(result, f)
    print(f"  {cell}: {len(result['expl_live'])} checkpoints, grid {n}/axis "
          f"-> {out_path.relative_to(REPO_ROOT)}")
    return result


# --------------------------------------------------------------------------- plots


def _symlog_axis(ax, values):
    """Log-ish y-axis that still shows an exactly-zero exploitability.

    `mp_magnet_idealized` converges to 0.0000, which a plain log axis drops silently.
    """
    positive = [v for v in values if v > 0]
    linthresh = max(min(positive) if positive else 1e-6, 1e-12)
    ax.set_yscale("symlog", linthresh=linthresh)


def plot_exploitability(results: dict, domain: str, engine: str, no_std: bool, out_dir: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    everything = []
    for magnet in MAGNETS:
        result = results.get(cell_name(domain, magnet, engine))
        if result is None:
            continue
        color = MAGNET_COLOR[magnet]
        ax.plot(result["t"], result["expl_live"], color=color, lw=1.6,
                label=f"{MAGNET_LABEL[magnet]} (live)")
        everything += list(result["expl_live"])
        if result["expl_target"] is not None:
            ax.plot(result["t"], result["expl_target"], color=color, lw=1.2, ls="--", alpha=0.75,
                    label=f"{MAGNET_LABEL[magnet]} (target)")
            everything += list(result["expl_target"])

    if not everything:
        plt.close(fig)
        return None

    _symlog_axis(ax, everything)
    ax.set_xlabel("iteration")
    ax.set_ylabel("exploitability" + (" (spread set to 0)" if no_std else ""))
    ax.set_title(f"{DOMAIN_LABEL[domain]}\n{ENGINE_LABEL[engine]}"
                 + ("  --  means only, spread zeroed" if no_std else ""))
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()

    suffix = "_no_std" if no_std else ""
    path = out_dir / f"expl_{domain}_{engine}{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mp_means(results: dict, engine: str, no_std: bool, out_dir: Path):
    """The `mp` trajectory in the plane of the two players' mean actions.

    Only meaningful for `mp`, whose action is a scalar per player, so player 0's mean
    is the x-coordinate and player 1's the y-coordinate and the Nash of the bilinear
    payoff `a0 * a1` on `[-1, 1]^2` is the origin. The single component of these runs
    (`num_components: 1`) makes "the mean" unambiguous.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    drawn = False
    for magnet in MAGNETS:
        result = results.get(cell_name("mp", magnet, engine))
        if result is None:
            continue
        x = np.array([m[0, 0] for m in result["means_0"]])
        y = np.array([m[0, 0] for m in result["means_1"]])
        color = MAGNET_COLOR[magnet]
        ax.plot(x, y, color=color, lw=1.0, alpha=0.85, label=MAGNET_LABEL[magnet])
        ax.scatter([x[0]], [y[0]], color=color, marker="o", s=45, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.scatter([x[-1]], [y[-1]], color=color, marker="X", s=80, zorder=3,
                   edgecolor="white", linewidth=0.8)
        drawn = True

    if not drawn:
        plt.close(fig)
        return None

    ax.scatter([0], [0], color="black", marker="*", s=160, zorder=4, label="Nash (0, 0)")
    ax.axhline(0, color="0.75", lw=0.7, zorder=0)
    ax.axvline(0, color="0.75", lw=0.7, zorder=0)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel("player 0 mean action")
    ax.set_ylabel("player 1 mean action")
    ax.set_title(f"ContinuousMatchingPennies mean trajectory\n{ENGINE_LABEL[engine]}"
                 "\n(circle = start, X = end)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    suffix = "_no_std" if no_std else ""
    path = out_dir / f"mp_means_{engine}{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- cli


def report_drift(results: dict) -> None:
    """How far the rescored curve sits from the one the run itself logged.

    Only the exact cells can be checked -- they are the ones that logged a comparable
    number -- and a nonzero gap there is the diagonal-covariance approximation of
    `_strategy_from_record` plus any `--grid-points` change.
    """
    rows = []
    for cell, result in sorted(results.items()):
        # `--no-std` deliberately scores a different strategy, so the run's own number
        # is not the thing it should agree with -- there is nothing to check there.
        if result["expl_run"] is None or result["no_std"]:
            continue
        recomputed = np.asarray(result["expl_live"], dtype=np.float64)
        original = np.asarray(result["expl_run"], dtype=np.float64)
        scale = max(np.max(np.abs(original)), 1e-12)
        rows.append((cell, np.max(np.abs(recomputed - original)),
                     np.max(np.abs(recomputed - original)) / scale))
    if not rows:
        return
    print("\nrescored vs the run's own exploitability (exact cells only)")
    print(f"  {'cell':28} {'max abs diff':>13} {'relative':>10}")
    for cell, absolute, relative in rows:
        print(f"  {cell:28} {absolute:13.6f} {relative:10.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-std", action="store_true",
        help="set every component's standard deviation to 0, scoring the means alone; "
             "writes exploitability_no_std.pkl instead of exploitability.pkl",
    )
    parser.add_argument(
        "--cells", nargs="+", default=None,
        help="cell names to process (default: all 18 that exist on disk)",
    )
    parser.add_argument(
        "--grid-points", type=int, default=None,
        help="best-response grid resolution per axis (default: each cell's own "
             "idealized.grid_points). Only comparable between runs that share it",
    )
    parser.add_argument(
        "--max-quadrature-gib", type=float, default=2.0,
        help="cap on the n^(2d) payoff matrix; ignored with --no-std, which never builds it",
    )
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "plots")
    parser.add_argument("--force", action="store_true", help="rescore even if a result file exists")
    parser.add_argument("--no-plots", action="store_true", help="compute and save only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cells = args.cells or [
        cell_name(domain, magnet, engine)
        for domain in DOMAINS for engine in ENGINES for magnet in MAGNETS
    ]
    cells = [c for c in cells if (DATA_ROOT / c).is_dir()]
    if not cells:
        raise SystemExit(f"no cell directories under {DATA_ROOT}")

    mode = "means only (spread = 0)" if args.no_std else "full mixtures"
    print(f"scoring {len(cells)} cells, {mode}")

    results, measure_cache = {}, {}
    for cell in cells:
        results[cell] = compute(
            cell, args.no_std, args.grid_points, args.max_quadrature_gib, measure_cache, args.force
        )

    report_drift(results)

    if args.no_plots:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for domain in DOMAINS:
        for engine in ENGINES:
            path = plot_exploitability(results, domain, engine, args.no_std, args.out_dir)
            if path is not None:
                written.append(path)
    for engine in ENGINES:
        path = plot_mp_means(results, engine, args.no_std, args.out_dir)
        if path is not None:
            written.append(path)

    print(f"\n{len(written)} plots -> {args.out_dir.relative_to(REPO_ROOT)}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
