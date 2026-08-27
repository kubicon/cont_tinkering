"""Run the with-/without-magnet configs side by side and plot the learning dynamics.

Two engines can drive the same YAML (`--engine`):

  idealized -- `run_idealized.py`: exact gradients, no sampling, the policy *is*
               the mixture parameters.
  ppo       -- `train.py`: `MixtureSelfPlayPPOTrainer`, i.e. sampled rollouts, a
               learned critic, clipped ratios, Adam on network weights. The plotted
               strategy is the policy head read out at the game's (constant)
               observation.
  both      -- both, overlaid; the PPO curves are dashed.

The game is bilinear matching pennies, `payoff(a0, a1) = a0 * a1` on `[-1, 1]`, so
only the mean of each policy matters and the Nash is at the midpoint of the box. Each
of the two settings (with/without magnet) has its own pair of config files --
`*_idealized.yaml` and `*_ppo.yaml` -- identical except `optimizer.learning_rate`
(0.05 for the idealized solver, 0.001 for the PPO network -- the same field feeds
both `cfg.lr` and the network's Adam step size, so each engine gets its own value).
The only *intended* difference between with-magnet and without-magnet is
`ppo.magnet_gaussian_kl_coef`; pass `--init` to also equalize the starting means,
which the two settings as written do not share.

The figure has three panels:

  * left   -- the phase plane: E[a] of player 0 against E[a] of player 1, one curve
              per run, colored light (t=0) to dark (t=T). This is where the failure
              is visible: without the magnet the pair spirals away from the Nash
              instead of into it.
  * top r. -- exploitability (NashConv) of the current and of the average
              (target-EMA) strategies, against iteration.
  * bot r. -- distance of (E[a_0], E[a_1]) to the Nash point, and the policy std,
              against iteration.

Both engines are traced **every iteration** (the orbit period is only a few hundred
iterations, so logging once per outer step aliases the circle into a polygon), and
both are scored with the same quadrature NashConv from `run_idealized.py`, so the
curves are directly comparable. Exploitability needs an eager best-response sweep, so
it is evaluated on a subsample (`--expl-points`).

Every run is pickled to `--cache` (default `dynamics_runs[_<engine>].pkl`) as

    {"<engine>:<config>.yaml": {"hyperparameters": {...},   # the whole resolved config
                                "stats":  {"t": ndarray, "x": ndarray, ...},
                                "params": {"player0": {"means": ndarray, ...}, ...}}}

-- plain dicts and numpy arrays, so a cached run can be re-analyzed (or replotted with
`--reuse`) without re-running anything.

usage:
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --engine ppo
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --engine both --init 0.2
    python experiments/failing_gaussian_wo_magnet/plot_dynamics.py --reuse   # replot cached runs
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
    ("wo_magnet", "no magnet", "Reds"),
    ("with_magnet", "magnet", "Blues"),
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
    """Copy of `params` whose mean/log-std head biases encode the mixture `p`.

    Both heads have a zero-initialized kernel, so at initialization their output *is*
    the bias -- writing the bias makes the network start from exactly the policy
    `idealized.init_means` / `init_log_std` describe, which is what makes a PPO run
    comparable to an idealized one. Without this the network would start from
    `_spread_bias_init`, i.e. (for `num_components: 1`) at the box midpoint, which on
    this game is the Nash itself.

    `logits_head` is left alone: its kernel is *not* zero, so its output is not the
    bias, and with one component the categorical head is trivial anyway.
    """
    seeded = jax.tree_util.tree_map(lambda x: x, params)  # rebuild the containers
    heads = seeded["params"]
    for name, value in (("means_head", p.means), ("log_std_head", p.log_std)):
        bias = heads[name]["bias"]
        heads[name]["bias"] = jnp.asarray(value, dtype=bias.dtype).reshape(bias.shape)
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


def _clipped_handle(backend, p: ri.Params):
    """Grid density of `clip(a, lo, hi)` for `a ~ mixture(p)`.

    `train.py` clips every sampled action into the box, so a policy whose mean has run
    far outside it still plays a point mass on the boundary -- while `backend.handle`,
    which integrates the raw Gaussian over a grid padded by only `4 * std_max`, would
    see no mass at all and report zero exploitability. This puts each component's
    out-of-box tail mass back as a spike on the boundary grid point, which is exactly
    the law of the clipped action.
    """
    w = jax.nn.softmax(p.logits)
    mu, s = p.means[:, None], jnp.exp(p.log_std)[:, None]
    grid = backend.grid[None, :]
    pdf = jnp.exp(-((grid - mu) ** 2) / (2 * s**2)) / (jnp.sqrt(2 * jnp.pi) * s)
    inside = jnp.where(backend.in_box[None, :], pdf, 0.0)

    cdf = lambda x: 0.5 * (1 + erf((x - mu[:, 0]) / (s[:, 0] * jnp.sqrt(2.0))))  # noqa: E731
    edge = jnp.zeros_like(inside)
    lo_i = int(jnp.argmin(jnp.abs(backend.grid - backend.lo)))
    hi_i = int(jnp.argmin(jnp.abs(backend.grid - backend.hi)))
    edge = edge.at[:, lo_i].add(cdf(backend.lo) / backend.dx)
    edge = edge.at[:, hi_i].add((1.0 - cdf(backend.hi)) / backend.dx)
    return jnp.sum(w[:, None] * (inside + edge), axis=0)


def _summarize(game, cfg, run_config, traced, engine: str, expl_points: int, meta: dict) -> dict:
    """Turn four stacked `ri.Params` into one run record.

    The record is three plain dicts, all leaf values numpy arrays or python scalars:

      `hyperparameters` -- the whole resolved config (solver + run config), the game's
                           bounds and Nash point, the engine, and the realized init.
      `stats`           -- per-iteration arrays: `t`, the two players' `E[a]` (current
                           and average), the mean policy std, the distance to the Nash,
                           and exploitability on its `expl_t` subsample.
      `params`          -- the raw traced mixture, `(iterations, num_components)` arrays
                           of `weights` / `means` / `std` per player, for both the
                           current and the average strategy. Everything in `stats` is
                           derived from these, so a later question about e.g. component
                           weights can be answered from the cache alone.
    """
    cur0, cur1, avg0, avg1 = traced
    total = int(cur0.means.shape[0])
    backend = ri.build_backend(game, cfg)
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
    std = (traced_arrays["player0"]["std"].mean(axis=-1)
           + traced_arrays["player1"]["std"].mean(axis=-1)) / 2
    t = np.arange(1, total + 1, dtype=float)

    # exploitability: eager (it builds a best-response grid), so subsample.
    idx = np.unique(np.linspace(0, total - 1, expl_points).round().astype(int))
    take = lambda p, i: ri.Params(*(jnp.asarray(v[i], dtype=jnp.float64)  # noqa: E731
                                    for v in (p.logits, p.means, p.log_std)))
    expl = lambda a, b: float(backend.exploitability(handle(a), handle(b)))  # noqa: E731

    lo, hi = ri._bounds(game)
    nash = 0.5 * (lo + hi)  # the payoff is bilinear, so the Nash mean is the box midpoint
    return {
        "hyperparameters": {
            **meta,
            "engine": engine,
            "game": type(game).__name__,
            "action_low": lo,
            "action_high": hi,
            "nash": nash,
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
            "dist_nash": np.hypot(x - nash, y - nash),
            "dist_nash_avg": np.hypot(x_avg - nash, y_avg - nash),
            "expl_t": t[idx],
            "expl": np.array([expl(take(cur0, i), take(cur1, i)) for i in idx]),
            "expl_avg": np.array([expl(take(avg0, i), take(avg1, i)) for i in idx]),
        },
        "params": traced_arrays,
    }


def simulate(config_path: Path, engine: str, init: float | None,
             expl_points: int, seed_init: bool = True) -> dict:
    """Run one config under one engine and return its run record."""
    game_config, cfg, run_config = ri.load_config(config_path)
    game = game_config.build()
    if cfg.mode != "self_play":
        raise SystemExit(f"{config_path.name}: this script only handles train.mode: self_play")
    if init is not None:
        cfg = dataclasses.replace(
            cfg, idealized=dataclasses.replace(
                cfg.idealized, init_means=[init] * cfg.num_components))

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


def _fade(ax, x, y, cmap, label, style):
    """Draw `(x, y)` as a polyline whose color darkens along the trajectory."""
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap), linewidth=1.1, linestyle=style, zorder=3)
    lc.set_array(np.linspace(0.25, 1.0, len(segs)))  # start light, end dark
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)
    solid = plt.get_cmap(cmap)(0.85)
    ax.plot([], [], color=solid, linewidth=1.6, linestyle=style, label=label)
    ax.plot(x[0], y[0], "o", color=solid, markersize=8, markerfacecolor="white",
            markeredgewidth=1.8, zorder=5)
    ax.plot(x[-1], y[-1], "*", color=solid, markersize=15, zorder=5)
    return solid


def plot(results: list[tuple[str, dict, str, str]], out: Path) -> None:
    fig = plt.figure(figsize=(13.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], hspace=0.33, wspace=0.26)
    ax_phase = fig.add_subplot(gs[:, 0])
    ax_expl = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_std = ax_dist.twinx()

    hp0 = results[0][1]["hyperparameters"]
    lo, hi, nash = hp0["action_low"], hp0["action_high"], hp0["nash"]
    pad = 0.05 * (hi - lo)  # the action box plus a 5% margin, i.e. [-1.1, 1.1] here

    for label, r, cmap, style in results:
        hp, st = r["hyperparameters"], r["stats"]
        x, y, t = st["x"], st["y"], st["t"]
        tag = (f"{label} (init {hp['init_mean_player0']:+.2f}, "
               f"magnet {hp['magnet_gaussian_kl_coef']:g})")
        solid = _fade(ax_phase, x, y, cmap, tag, style)
        ax_expl.plot(st["expl_t"], st["expl"], color=solid, linewidth=1.4, linestyle=style,
                     label=f"{label}: current")
        ax_expl.plot(st["expl_t"], st["expl_avg"], color=solid, linewidth=1.1, linestyle=":",
                     alpha=0.75, label=f"{label}: average")
        ax_dist.plot(t, st["dist_nash"], color=solid, linewidth=1.4, linestyle=style, label=label)
        ax_std.plot(t, st["std"], color=solid, linewidth=1.0, linestyle=":", alpha=0.85)

    ax_phase.plot(nash, nash, "kx", markersize=12, markeredgewidth=2.5, zorder=6, label="Nash")
    ax_phase.axhline(nash, color="0.85", linewidth=0.8, zorder=1)
    ax_phase.axvline(nash, color="0.85", linewidth=0.8, zorder=1)
    engines = " + ".join(dict.fromkeys(r["hyperparameters"]["engine"] for _, r, _, _ in results))
    # Fixed to the action box: with `network.clip_means` on, that is where the strategies
    # live. A run whose means do leave it (clipping off) simply runs out of frame -- the
    # distance-to-Nash panel is what shows how far it went.
    ax_phase.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
                 xlabel=r"$E[a]$ of player 0", ylabel=r"$E[a]$ of player 1",
                 title=f"phase plane -- {hp0['game']} ({engines})\n"
                       "circle = start, star = end, color darkens with iteration")
    ax_phase.set_aspect("equal")
    ax_phase.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8,
                    frameon=False)  # the orbit fills the box, so keep the legend off it

    ax_expl.set(yscale="log", ylabel="exploitability", title="NashConv")
    ax_expl.legend(fontsize=7)
    ax_dist.set(yscale="log", xlabel="iteration",
                ylabel=r"$\|(E[a_0], E[a_1]) - \mathrm{Nash}\|$")
    ax_dist.set_title("distance to Nash (solid) and policy std (dotted)")
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
                         "neural-network self-play, both = overlay (default: idealized)")
    ap.add_argument("--init", type=float, default=0.2,
                    help="override both configs' idealized.init_means, so the runs differ "
                         "only in the magnet coefficient (default: use each config's own)")
    ap.add_argument("--network-init", action="store_true",
                    help="--engine ppo: keep the trainer's own `_spread_bias_init` instead of "
                         "starting the network from idealized.init_means (for one component "
                         "that init sits at the box midpoint, which here is the Nash)")
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
                r = simulate(HERE / name, engine, args.init, args.expl_points,
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
