"""Run the bad-/good-init configs side by side and plot the learning dynamics.

Adapted from `experiments/failing_gaussian_wo_magnet/plot_dynamics.py` -- see that
file's docstring for the general engine/plotting design. The differences here are
driven by the game: `multi_point` (peaks -1.0/+1.0, weights 0.5/0.5, `network.
num_components: 2`), not bilinear `matching_pennies` with one component, so:

  * The Nash is a *distribution*, `sum_k weights_k * delta(peaks_k)` -- here half the
    mass on -1 and half on +1 -- not a point in action space. A mean action therefore
    says almost nothing: `E[a] = 0` holds at the Nash, but equally at a single blob
    sitting on 0, or at any symmetric pair of components anywhere. So neither the
    phase plane nor the convergence metric here is built on `E[a]` (see the panel
    list below); both are read off the mixture itself, against the game's own
    `.peaks`/`.weights`.
  * `multi_point` exposes `.peaks`/`._target_moments`, so `idealized.backend: auto`
    picks the exact `closed_form` backend for the *training* dynamics. Scoring
    (exploitability, and the clipped-tail correction for the PPO engine) needs a
    grid, so `_summarize` always builds its own `QuadratureBackend` for that,
    independent of whichever backend actually drove the run.
  * With two meaningful components, the categorical head is no longer trivial, so
    `_seed_heads` also zeroes `logits_head`'s kernel and seeds its bias, giving the
    PPO run the *exact* `idealized.init_weights` (not just an untrained-network
    approximation of it) -- the "no head is zero-init except mean/log_std" caveat in
    the original script's `_seed_heads` no longer applies once weights matter.

Two engines can drive the same YAML pair (`--engine`):

  idealized -- `run_idealized.py`: exact gradients, no sampling, the policy *is*
               the mixture parameters.
  ppo       -- `train.py`: `MixtureSelfPlayPPOTrainer`, i.e. sampled rollouts, a
               learned critic, clipped ratios, Adam on network weights.
  both      -- both, overlaid; the PPO curves are dashed.

Each of the two settings (bad/good init) has its own pair of config files --
`*_idealized.yaml` and `*_ppo.yaml` -- identical except `optimizer.learning_rate`
(0.05 for the idealized solver, 0.001 for the PPO network -- the same field feeds
both `cfg.lr` and the network's Adam step size). There is deliberately no `--init`
override here (unlike the magnet script): the initialization *is* the independent
variable this script studies, so bad/good must keep their own.

The figure has three panels:

  * left   -- the mixture components, each on its own player's axis: player 0's K
              component means live on the x-axis (at y=0), player 1's on the y-axis
              (at x=0). The two players hold *independent* mixtures, so their means
              are never crossed into a joint point -- a mark at x=-1 says player 0
              has a component at -1 and says nothing about player 1. With K=2 that
              is 4 marks per run: two on each axis. Each is drawn as the trajectory
              of that component mean over training, fading light (t=0) to dark
              (t=T), so a component sliding from +2 down to +1 reads as a shortening
              segment; the end star is sized by the component's final categorical
              weight, so an abandoned component shrinks to a dot. The black crosses
              sit at the Nash support (`game.peaks`) on both axes -- every player
              must hold a component at every peak. Runs are nudged a hair off their
              axis only so they do not draw on top of each other; the off-axis
              coordinate carries no meaning. Landing on the crosses is necessary but
              not sufficient (the weights must be `game.weights` and the stds small),
              which is what the W1 panel scores.
  * top r. -- exploitability (NashConv) of the current and of the average
              (target-EMA) strategies, against iteration.
  * bot r. -- W1 (1-Wasserstein) distance from each player's action distribution to
              the Nash distribution, averaged over the two players, and the policy
              std, against iteration. W1 is zero *only* at the Nash: unlike a mean
              action it sees the component locations, their widths, and the split of
              the categorical weight all at once.

Both engines are traced **every iteration** and scored with the same quadrature
NashConv, so the curves are directly comparable.

Every run is pickled to `--cache` (default `dynamics_runs[_<engine>].pkl`) as

    {"<engine>:<config>.yaml": {"hyperparameters": {...},   # the whole resolved config
                                "stats":  {"t": ndarray, "x": ndarray, ...},
                                "params": {"player0": {"means": ndarray, ...}, ...}}}

-- plain dicts and numpy arrays, so a cached run can be re-analyzed (or replotted with
`--reuse`) without re-running anything.

usage:
    python experiments/bad_initialization/plot_dynamics.py
    python experiments/bad_initialization/plot_dynamics.py --engine ppo
    python experiments/bad_initialization/plot_dynamics.py --engine both
    python experiments/bad_initialization/plot_dynamics.py --reuse   # replot cached runs
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # repo root, so `import run_idealized` works

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import erf
from matplotlib.collections import LineCollection

import run_idealized as ri
import train as train_cli
from training import mixture_trainer as mt
from training.run_config import load_run_config

RUNS = [
    ("bad_init", "bad init", "Reds"),
    ("good_init", "good init", "Blues"),
]
ENGINES = ("idealized", "ppo")
STYLE = {"idealized": "-", "ppo": "--"}


def config_name(base: str, engine: str) -> str:
    """Each (base, engine) pair has its own config file -- see module docstring."""
    return f"{base}_{engine}.yaml"


def _asdict(obj):
    """`dataclasses.asdict`, passing anything that isn't a dataclass straight through."""
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def save_cache(path: Path, cache: dict) -> None:
    """Pickle `{f"{engine}:{config}": record}` -- plain dicts of numpy arrays."""
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- engines
#
# Each engine returns four stacked `ri.Params` (leading axis = iteration): the two
# players' current strategies and the two players' average (target-EMA) strategies.


def _run_idealized(game, cfg) -> tuple[ri.Params, ...]:
    p0, p1 = ri.build_init(game, cfg)
    backend = ri.build_backend(game, cfg)
    iteration = ri.build_iteration(game, cfg, backend)

    def trace(carry, _):
        carry, _ = iteration(carry, None)
        cur0, cur1, _, _, avg0, avg1, _, _ = carry
        return carry, (cur0, cur1, avg0, avg1)

    # `fixed_handle` is unused in self_play, but the carry shape must still match.
    fixed_handle = jnp.zeros_like(backend.handle(p0)) if backend.name == "quadrature" else None
    carry = (p0, p1, p0, p1, p0, p1, fixed_handle, jnp.zeros((), dtype=jnp.int64))
    _, ys = jax.jit(lambda c: jax.lax.scan(trace, c, None, length=cfg.total_iters))(carry)
    return ys


def _seed_heads(params, p: ri.Params):
    """Copy of `params` whose logits/mean/log-std head biases encode the mixture `p`.

    `means_head` and `log_std_head` already have a zero-initialized kernel (see
    `training.mixture.MixtureActorCritic`), so writing their bias is enough to make
    their output exactly `p.means` / `p.log_std`. `logits_head` does *not* have a
    zero kernel by default, so its bias alone would not reproduce `p.logits` --
    with `num_components: 2` the categorical head is no longer trivial (unlike the
    one-component matching-pennies script this one is adapted from), so its kernel
    is zeroed here too, then its bias set to `p.logits`.
    """
    seeded = jax.tree_util.tree_map(lambda x: x, params)  # rebuild the containers
    heads = seeded["params"]
    for name, value in (("means_head", p.means), ("log_std_head", p.log_std),
                        ("logits_head", p.logits)):
        bias = heads[name]["bias"]
        heads[name]["bias"] = jnp.asarray(value, dtype=bias.dtype).reshape(bias.shape)
    heads["logits_head"]["kernel"] = jnp.zeros_like(heads["logits_head"]["kernel"])
    return seeded


def _run_ppo(game, cfg, run_config, seed_init: bool) -> tuple[ri.Params, ...]:
    hp0 = train_cli.build_hyperparams(game, 0, run_config)
    hp1 = train_cli.build_hyperparams(game, 1, run_config)
    trainer = mt.MixtureSelfPlayPPOTrainer(game, hp0, hp1, seed=run_config.train.seed)

    if seed_init:
        p0, p1 = ri.build_init(game, cfg)
        for state_name, p in (("state_1", p0), ("state_2", p1)):
            state = getattr(trainer, state_name)
            seeded = _seed_heads(state.params, p)
            setattr(trainer, state_name,
                    state.replace(params=seeded, target_params=seeded, magnet_params=seeded))

    step = mt._build_self_play_train_step(game, (trainer.network_1, trainer.network_2), (hp0, hp1))
    obs = (game.observation(0, jax.random.PRNGKey(0)), game.observation(1, jax.random.PRNGKey(0)))

    def head(network, params, player: int) -> ri.Params:
        """The policy head at the game's constant observation, as a 1-D mixture."""
        logits, means, log_std, _ = network.apply(params, obs[player])
        return ri.Params(logits=logits, means=means[:, 0], log_std=log_std[:, 0])

    def trace(states, key):
        s0, s1, _ = step(states[0], states[1], key)
        n0, n1 = trainer.network_1, trainer.network_2
        return (s0, s1), (head(n0, s0.params, 0), head(n1, s1.params, 1),
                          head(n0, s0.target_params, 0), head(n1, s1.target_params, 1))

    keys = jax.random.split(trainer.key, cfg.total_iters)
    _, ys = jax.jit(lambda st, ks: jax.lax.scan(trace, st, ks))((trainer.state_1, trainer.state_2), keys)
    return ys


# --------------------------------------------------------------------------- scoring


def _scoring_backend(game, cfg) -> ri.QuadratureBackend:
    """A dedicated grid backend for scoring, independent of `idealized.backend`.

    `multi_point` exposes `.peaks`/`._target_moments`, so `backend: auto` picks the
    exact `ClosedFormBackend` for the training dynamics themselves -- correct and
    fast, but its `handle` is the identity (just the mixture `Params`), with no grid,
    so it cannot support `_clipped_handle`'s box-edge tail correction below. Scoring
    always uses quadrature instead, exactly as the matching-pennies script does
    (there `auto` already resolves to quadrature, since that game has no `.peaks`).
    """
    return ri.QuadratureBackend(
        game, cfg.idealized.grid_points, ri._std_max(game, cfg), cfg.idealized.normalize_density)


def _clipped_handle(backend, p: ri.Params):
    """Grid density of `clip(a, lo, hi)` for `a ~ mixture(p)`.

    `train.py` clips every sampled action into the box, so a policy whose mean has run
    far outside it still plays a point mass on the boundary -- while `backend.handle`,
    which integrates the raw Gaussian over a grid padded by only `4 * std_max`, would
    see no mass at all and report zero exploitability. This puts each component's
    out-of-box tail mass back as a spike on the boundary grid point, which is exactly
    the law of the clipped action.

    The in-box part comes from `backend.component_densities`, NOT from a fresh
    `exp(...)` on the grid: that method honours `normalize_density`, and PPO drives
    stds down to ~1e-3 while the scoring grid's `dx` is ~1.5e-2, so an unnormalized
    component an order of magnitude narrower than the grid spacing lands an
    essentially arbitrary amount of mass on whichever point it happens to straddle.
    Unnormalized, this returned densities integrating to ~2 instead of 1, which made
    both the PPO NashConv and the PPO W1 curves meaningless.
    """
    w = jax.nn.softmax(p.logits)
    mu, s = p.means[:, None], jnp.exp(p.log_std)[:, None]
    inside = jnp.where(backend.in_box[None, :], backend.component_densities(p), 0.0)

    cdf = lambda x: 0.5 * (1 + erf((x - mu[:, 0]) / (s[:, 0] * jnp.sqrt(2.0))))  # noqa: E731
    edge = jnp.zeros_like(inside)
    # numpy, not jnp: `.at[:, i]` needs a python int, and under `jit` a closed-over
    # jnp array becomes a tracer, so `int(jnp.argmin(...))` would fail here. The grid
    # is fixed at backend construction, so resolve the two boundary indices on the host.
    edges = np.asarray(backend.grid)
    lo_i = int(np.argmin(np.abs(edges - backend.lo)))
    hi_i = int(np.argmin(np.abs(edges - backend.hi)))
    edge = edge.at[:, lo_i].add(cdf(backend.lo) / backend.dx)
    edge = edge.at[:, hi_i].add((1.0 - cdf(backend.hi)) / backend.dx)
    return jnp.sum(w[:, None] * (inside + edge), axis=0)


def _nash_cdf(backend, game):
    """CDF of the Nash action law `sum_k weights_k * delta(peaks_k)`, on the grid."""
    peaks = jnp.asarray(game.peaks, dtype=jnp.float64)[:, None]
    weights = jnp.asarray(game.weights, dtype=jnp.float64)[:, None]
    return jnp.sum(weights * (backend.grid[None, :] >= peaks), axis=0)


def _w1_series(backend, handle, nash_cdf, p: ri.Params) -> np.ndarray:
    """`W1(policy_t, nash)` for every traced iteration of one player.

    In 1-D the 1-Wasserstein distance is `int |F_policy - F_nash| dx`, so one cumsum
    on the scoring grid gives it. This is the convergence metric this game needs: the
    Nash is a two-point law, and `E[a]` cannot tell it apart from a blob on 0 (the
    `good_init` run starts at `E[a] = 0` with exploitability 2.0), whereas W1 is zero
    only when the component means sit on the peaks, the categorical weights match
    `game.weights`, and the stds have collapsed.

    `lax.map` walks the iteration axis one at a time -- a `vmap` over all 30k traced
    iterations would materialize a `(T, K, grid_points)` float64 array.
    """
    def one(q: ri.Params) -> jnp.ndarray:
        cdf = jnp.cumsum(handle(q)) * backend.dx
        return jnp.sum(jnp.abs(cdf - nash_cdf)) * backend.dx

    stacked = ri.Params(*(jnp.asarray(v, dtype=jnp.float64)
                          for v in (p.logits, p.means, p.log_std)))
    return np.asarray(jax.jit(lambda t: jax.lax.map(one, t))(stacked))


def _summarize(game, cfg, run_config, traced, engine: str, expl_points: int, meta: dict) -> dict:
    """Turn four stacked `ri.Params` into one run record.

    The record is three plain dicts, all leaf values numpy arrays or python scalars:

      `hyperparameters` -- the whole resolved config (solver + run config), the game's
                           bounds and Nash point, the engine, and the realized init.
      `stats`           -- per-iteration arrays: `t`, the two players' `E[a]` (current
                           and average, kept for reference only -- `E[a]` is degenerate
                           as a convergence metric here, see `_w1_series`), the mean
                           policy std, `W1` to the Nash law per player and averaged,
                           and exploitability on its `expl_t` subsample.
      `params`          -- the raw traced mixture, `(iterations, num_components)` arrays
                           of `weights` / `means` / `std` per player, for both the
                           current and the average strategy. Everything in `stats` is
                           derived from these, so a later question about e.g. component
                           weights can be answered from the cache alone.
    """
    cur0, cur1, avg0, avg1 = traced
    total = int(cur0.means.shape[0])
    backend = _scoring_backend(game, cfg)
    # the idealized solver integrates the raw Gaussian, `train.py` clips its actions
    handle = backend.handle if engine == "idealized" else lambda p: _clipped_handle(backend, p)

    def arrays(p) -> dict[str, np.ndarray]:
        """One traced player's mixture as `(iterations, num_components)` arrays."""
        return {
            "weights": np.asarray(jax.nn.softmax(p.logits, axis=-1)),
            "means": np.asarray(p.means),
            "std": np.asarray(jnp.exp(p.log_std)),
            "logits": np.asarray(p.logits),
        }

    traced_arrays = {name: arrays(p) for name, p in
                     (("player0", cur0), ("player1", cur1),
                      ("player0_avg", avg0), ("player1_avg", avg1))}

    def mean_action(name: str) -> np.ndarray:
        """E[a] = sum_k w_k mu_k, per traced iteration."""
        a = traced_arrays[name]
        return np.sum(a["weights"] * a["means"], axis=-1)

    x, y = mean_action("player0"), mean_action("player1")
    x_avg, y_avg = mean_action("player0_avg"), mean_action("player1_avg")
    # kept in `stats` because they are cheap and occasionally useful, but NOT used as
    # a convergence metric -- see `_w1_series` for why `E[a]` is degenerate here.
    nash_cdf = _nash_cdf(backend, game)
    w1 = lambda p: _w1_series(backend, handle, nash_cdf, p)  # noqa: E731
    w1_0, w1_1 = w1(cur0), w1(cur1)
    w1_0_avg, w1_1_avg = w1(avg0), w1(avg1)
    std = (traced_arrays["player0"]["std"].mean(axis=-1)
           + traced_arrays["player1"]["std"].mean(axis=-1)) / 2
    t = np.arange(1, total + 1, dtype=float)

    # exploitability: eager (it builds a best-response grid), so subsample.
    idx = np.unique(np.linspace(0, total - 1, expl_points).round().astype(int))
    take = lambda p, i: ri.Params(*(jnp.asarray(v[i], dtype=jnp.float64)  # noqa: E731
                                    for v in (p.logits, p.means, p.log_std)))
    expl = lambda a, b: float(backend.exploitability(handle(a), handle(b)))  # noqa: E731

    lo, hi = ri._bounds(game)
    # The Nash: each player's marginal over the peaks equals `game.weights`, on
    # support `game.peaks` (see `games.examples.MultiPointGame`'s docstring). It is a
    # distribution, so it is recorded as one -- there is no scalar "the Nash action".
    nash_peaks = [float(v) for v in game.peaks]
    nash_weights = [float(v) for v in game.weights]
    return {
        "hyperparameters": {
            **meta,
            "engine": engine,
            "game": type(game).__name__,
            "action_low": lo,
            "action_high": hi,
            "nash_peaks": nash_peaks,
            "nash_weights": nash_weights,
            "iterations": total,
            "num_components": cfg.num_components,
            "lr": cfg.lr,
            "magnet_gaussian_kl_coef": cfg.magnet_gaussian_kl_coef,
            "init_mean_player0": float(x[0]),
            "init_mean_player1": float(y[0]),
            "solver": _asdict(cfg),               # SolverConfig, `idealized:` section included
            "run_config": _asdict(run_config),    # the train.py schema, or None for a legacy config
        },
        "stats": {
            "t": t,
            "x": x, "y": y,
            "x_avg": x_avg, "y_avg": y_avg,
            "std": std,
            "w1_0": w1_0, "w1_1": w1_1,
            "w1": (w1_0 + w1_1) / 2,
            "w1_avg": (w1_0_avg + w1_1_avg) / 2,
            "expl_t": t[idx],
            "expl": np.array([expl(take(cur0, i), take(cur1, i)) for i in idx]),
            "expl_avg": np.array([expl(take(avg0, i), take(avg1, i)) for i in idx]),
        },
        "params": traced_arrays,
    }


def simulate(config_path: Path, engine: str, expl_points: int, seed_init: bool = True) -> dict:
    """Run one config under one engine and return its run record."""
    game_config, cfg, run_config = ri.load_config(config_path)
    game = game_config.build()
    if cfg.mode != "self_play":
        raise SystemExit(f"{config_path.name}: this script only handles train.mode: self_play")

    if engine == "idealized":
        traced = _run_idealized(game, cfg)
    elif engine == "ppo":
        if run_config is None:  # a legacy `mmd:`-schema config has no train.py counterpart
            raise SystemExit(f"{config_path.name}: --engine ppo needs a train.py run config")
        traced = _run_ppo(game, cfg, load_run_config(config_path), seed_init)
    else:
        raise SystemExit(f"unknown engine {engine!r}")
    meta = {"config": config_path.name, "seeded_init": seed_init if engine == "ppo" else None}
    return _summarize(game, cfg, run_config, traced, engine, expl_points, meta)


# --------------------------------------------------------------------------- plotting


def _fade(ax, x, y, cmap, label, style, width=1.1, alpha=1.0):
    """Draw `(x, y)` as a polyline whose color darkens along the trajectory."""
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap), linewidth=width, linestyle=style,
                        alpha=alpha, zorder=3)
    lc.set_array(np.linspace(0.25, 1.0, len(segs)))  # start light, end dark
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)
    solid = plt.get_cmap(cmap)(0.85)
    if label is not None:
        ax.plot([], [], color=solid, linewidth=1.6, linestyle=style, label=label)
    return solid


def _endpoint(ax, pos: float, off: float, on_x: bool, kind: str, color, size: float,
              limits: tuple[float, float]) -> None:
    """One trajectory endpoint -- start circle or end star -- pinned if it is off-scale.

    The axes are the action box, but `train.py` runs with `clip_means: false`, so a
    PPO component mean is free to leave the box entirely (see the note in
    `training.mixture.MixtureActorCritic.__call__`; here player 1's abandoned
    component ends at +2.34 with the box at +-2). Matplotlib silently clips such a
    marker away, which reads as "that component is gone" when in fact it ran off the
    edge -- so instead it is drawn ON the boundary as an outward-pointing triangle
    with its true value printed beside it. The idealized solver never needs this: it
    projects every mean back into the box each step (`run_idealized.player_step`).
    """
    lo_ax, hi_ax = limits
    if lo_ax <= pos <= hi_ax:
        x, y = (pos, off) if on_x else (off, pos)
        if kind == "start":
            ax.plot(x, y, "o", color=color, markersize=7, markerfacecolor="white",
                    markeredgewidth=1.6, zorder=5)
        else:
            ax.plot(x, y, "*", color=color, markersize=size, zorder=6)
        return

    over = pos > hi_ax
    edge = hi_ax if over else lo_ax
    tri = (">" if over else "<") if on_x else ("^" if over else "v")
    x, y = (edge, off) if on_x else (off, edge)
    ax.plot(x, y, tri, color=color, markersize=0.6 * size + 3, markeredgecolor="black",
            markeredgewidth=0.7, zorder=8)
    ax.annotate(f"{pos:+.2f}", (x, y), textcoords="offset points",
                xytext=(-16 if over and on_x else 6, 6 if on_x else (-12 if over else 8)),
                fontsize=7, color=color, zorder=8,
                ha="right" if (over and on_x) else "left")


def _component_tracks(ax, r: dict, cmap: str, style: str, label: str, offsets, limits):
    """One run's mixture components, each on its OWN player's axis.

    Player 0's components live on the x-axis and player 1's on the y-axis, so a
    K-component mixture per player gives `2 * K` tracks, never a product of the two
    players' means: a point at `x = -1` says *player 0* holds a component at -1 and
    says nothing whatever about player 1. (The previous version crossed the two
    players into one 2-D point, which invented structure that isn't there -- the
    strategies are independent mixtures, not a joint distribution over pairs.)

    Each track is the trajectory of one component mean over training, drawn along its
    axis and fading light (t=0) to dark (t=T), so a component that moves from +2 to
    +1 reads as a segment shortening toward the peak.

    `offsets[k]` nudges component `k` of this run off the axis. Every component gets
    its own row, for two reasons: different runs would otherwise draw on top of each
    other, and -- the case that actually matters here -- a *collapsed* mixture puts
    both of its components at the identical mean, so on a single row the second star
    would hide behind the first and a 2-component mixture would be indistinguishable
    from a 1-component one. With one row each, "both stars at the same on-axis
    position" is exactly what a collapse looks like. The off-axis coordinate carries
    no meaning whatever; only the on-axis one does.

    The end star is sized by that component's final categorical weight -- a component
    the mixture has abandoned (weight -> 0) shrinks to a dot, which the mean alone
    would not show. An endpoint that has left the axes is pinned to the edge rather
    than dropped; see `_endpoint`.
    """
    solid = plt.get_cmap(cmap)(0.85)
    ax.plot([], [], color=solid, linewidth=1.6, linestyle=style, label=label)
    for who, on_x in (("player0", True), ("player1", False)):
        a = r["params"][who]
        mu, w = a["means"], a["weights"]
        for k in range(mu.shape[1]):
            pos, off = mu[:, k], np.full(mu.shape[0], offsets[k])
            x, y = (pos, off) if on_x else (off, pos)
            _fade(ax, x, y, cmap, None, style)
            size = 7.0 + 17.0 * float(w[-1, k])
            _endpoint(ax, float(pos[0]), offsets[k], on_x, "start", solid, size, limits)
            _endpoint(ax, float(pos[-1]), offsets[k], on_x, "end", solid, size, limits)
    return solid


def plot(results: list[tuple[str, dict, str, str]], out: Path) -> None:
    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.33, wspace=0.26)
    ax_phase = fig.add_subplot(gs[:, 0])
    ax_expl = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_std = ax_dist.twinx()

    hp0 = results[0][1]["hyperparameters"]
    lo, hi = hp0["action_low"], hp0["action_high"]
    peaks = np.sort(np.asarray(hp0["nash_peaks"], dtype=float))
    pad = 0.05 * (hi - lo)  # the action box plus a 5% margin

    # One off-axis row per (run, component), so nothing hides behind anything else --
    # in particular a collapsed mixture's two coincident components. Purely cosmetic:
    # only the on-axis coordinate means anything (see `_component_tracks`).
    span = (hi + pad) - (lo - pad)
    k = max(len(r["params"]["player0"]["means"][0]) for _, r, _, _ in results)
    n_rows = len(results) * k
    # one row every 2.8% of the box, but the whole stack capped at +-9% of it, so
    # `--engine both` (twice as many rows) does not drift far off the axes
    spacing = min(0.028 * span, 0.18 * span / max(n_rows - 1, 1))
    rows = (np.arange(n_rows) - (n_rows - 1) / 2) * spacing

    for i, (label, r, cmap, style) in enumerate(results):
        offsets = rows[i * k:(i + 1) * k]
        hp, st = r["hyperparameters"], r["stats"]
        t = st["t"]
        tag = f"{label} (init {hp['init_mean_player0']:+.2f})"
        solid = _component_tracks(ax_phase, r, cmap, style, tag, offsets,
                                  (lo - pad, hi + pad))
        ax_expl.plot(st["expl_t"], st["expl"], color=solid, linewidth=1.4, linestyle=style,
                     label=f"{label}: current")
        ax_expl.plot(st["expl_t"], st["expl_avg"], color=solid, linewidth=1.1, linestyle=":",
                     alpha=0.75, label=f"{label}: average")
        ax_dist.plot(t, st["w1"], color=solid, linewidth=1.4, linestyle=style, label=label)
        ax_std.plot(t, st["std"], color=solid, linewidth=1.0, linestyle=":", alpha=0.85)

    # the two axes each carry ONE player's mixture, so the Nash support shows up once
    # per axis: player 0 must hold components at every peak, and so must player 1.
    ax_phase.axhline(0.0, color="0.75", linewidth=1.0, zorder=1)
    ax_phase.axvline(0.0, color="0.75", linewidth=1.0, zorder=1)
    for j, peak in enumerate(peaks):
        kw = {"label": "Nash support (per player)"} if j == 0 else {}
        ax_phase.plot([peak, 0.0], [0.0, peak], "kx", markersize=11, markeredgewidth=2.2,
                      linestyle="none", zorder=7, **kw)
        ax_phase.axvline(peak, color="0.92", linewidth=0.8, zorder=0)
        ax_phase.axhline(peak, color="0.92", linewidth=0.8, zorder=0)
    engines = " + ".join(dict.fromkeys(r["hyperparameters"]["engine"] for _, r, _, _ in results))
    ax_phase.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
                 xlabel=r"player 0's component means (on $y=0$)",
                 ylabel=r"player 1's component means (on $x=0$)",
                 title=f"mixture components -- {hp0['game']} ({engines})\n"
                       "circle = start, star = end (sized by final weight), "
                       "color darkens with iteration\n"
                       "triangle = endpoint outside the box (value annotated); "
                       "one off-axis row per component, cosmetic only")
    ax_phase.set_aspect("equal")
    ax_phase.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8,
                    frameon=False)

    ax_expl.set(yscale="log", ylabel="exploitability", title="NashConv")
    ax_expl.legend(fontsize=7)
    ax_dist.set(yscale="log", xlabel="iteration",
                ylabel=r"$W_1(\pi_i, \pi^*)$, mean over $i$")
    ax_dist.set_title("Wasserstein distance to the Nash law (solid) and policy std (dotted)")
    ax_std.set_ylabel("policy std (dotted)")
    for ax in (ax_expl, ax_dist):
        ax.grid(alpha=0.25)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=(*ENGINES, "both"), default="both",
                    help="idealized = run_idealized.py's exact solver, ppo = train.py's "
                         "neural-network self-play, both = overlay (default: both)")
    ap.add_argument("--network-init", action="store_true",
                    help="--engine ppo: keep the trainer's own `_spread_bias_init` instead of "
                         "starting the network from idealized.init_means/init_weights")
    ap.add_argument("--expl-points", type=int, default=200,
                    help="number of iterations at which exploitability is evaluated")
    ap.add_argument("--out", type=Path, default=None, help="default: dynamics[_<engine>].png")
    ap.add_argument("--cache", type=Path, default=None,
                    help="pickled run records -- `{'<engine>:<config>': {hyperparameters, "
                         "stats, params}}`, all numpy; default: dynamics_runs[_<engine>].pkl")
    ap.add_argument("--reuse", action="store_true",
                    help="replot from the cached runs instead of re-running")
    args = ap.parse_args()

    suffix = "" if args.engine == "idealized" else f"_{args.engine}"
    out = args.out or HERE / f"dynamics{suffix}.png"
    cache_path = args.cache or HERE / f"dynamics_runs{suffix}.pkl"
    engines = ENGINES if args.engine == "both" else (args.engine,)

    def tag(engine: str, label: str) -> str:
        return f"{label} [{engine}]" if len(engines) > 1 else label

    if args.reuse:
        cached = load_cache(cache_path)
        results = [(tag(e, label), cached[f"{e}:{config_name(base, e)}"], cmap, STYLE[e])
                   for e in engines for base, label, cmap in RUNS]
    else:
        results, cache = [], {}
        for engine in engines:
            for base, label, cmap in RUNS:
                name = config_name(base, engine)
                print(f"running {name} [{engine}] ...", flush=True)
                r = simulate(HERE / name, engine, args.expl_points,
                             seed_init=not args.network_init)
                print(f"  final exploitability: current {r['stats']['expl'][-1]:+.6f} | "
                      f"average {r['stats']['expl_avg'][-1]:+.6f}", flush=True)
                results.append((tag(engine, label), r, cmap, STYLE[engine]))
                cache[f"{engine}:{name}"] = r
        save_cache(cache_path, cache)
        print(f"saved runs -> {cache_path}")

    plot(results, out)


if __name__ == "__main__":
    main()
