"""
A self-contained implementation of Randomized Policy Networks on the Glicksberg-Gross game.

Finding mixed-strategy equilibria of continuous-action games without gradients using randomized policy networks
https://arxiv.org/abs/2211.15936
https://www.ijcai.org/proceedings/2023/317
https://dl.acm.org/doi/10.24963/ijcai.2023/317
"""

import argparse
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import optax
from flax import linen as nn
from jax import Array, lax, random
from jax import numpy as jnp
from jax.nn import mish
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


class Net(nn.Module):
    noise_dim: int
    hidden_dims: Sequence[int]
    output_shape: Sequence[int]
    normalization: Literal["rms", "layer", "none"]

    @nn.compact
    def __call__(self, key: Array, observation: Array):
        noise = random.normal(key, (self.noise_dim,))
        x = jnp.concatenate([noise, observation.ravel()])

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            match self.normalization:
                case "rms":
                    x = nn.RMSNorm()(x)
                case "layer":
                    x = nn.LayerNorm()(x)
                case "none":
                    pass
                case _:
                    raise ValueError(f"Invalid {self.normalization=}")
            x = mish(x)

        output_dim = math.prod(self.output_shape)
        x = nn.Dense(output_dim)(x)
        x = nn.sigmoid(x)
        x = x.reshape(self.output_shape)
        return x


class Game[State](ABC):
    @abstractmethod
    def get_num_players(self) -> int: ...

    @abstractmethod
    def get_observation_shape(self, player: int) -> Sequence[int]: ...

    @abstractmethod
    def get_action_shape(self, player: int) -> Sequence[int]: ...

    @abstractmethod
    def sample_state(self, key: Array) -> State: ...

    @abstractmethod
    def sample_observations(self, key: Array, state: State) -> Sequence[Array]: ...

    @abstractmethod
    def sample_utilities(
        self, key: Array, state: State, action_profile: Sequence[Array]
    ) -> Array: ...

    def get_solution_cdf(self, x: Array, player: int) -> Array:
        raise NotImplementedError()


class GlicksbergGross(Game):
    def get_num_players(self) -> int:
        return 2

    def get_observation_shape(self, player: int) -> Sequence[int]:
        return (0,)

    def get_action_shape(self, player: int) -> Sequence[int]:
        return ()

    def sample_state(self, key: Array) -> None:
        return None

    def sample_observations(self, key: Array, state: None) -> Sequence[Array]:
        return [jnp.zeros(0)] * 2

    def sample_utilities(
        self, key: Array, state: None, action_profile: Sequence[Array]
    ) -> Array:
        x, y = action_profile
        z = (1 + x) * (1 + y) * (1 - x * y) / jnp.square(1 + x * y)
        return jnp.stack([z, -z])

    def get_solution_cdf(self, x: Array, player: int) -> Array:
        return jnp.arctan(jnp.sqrt(x)) * 4 / jnp.pi


def sample_initial_params(key: Array, nets: Sequence[Net], game: Game) -> list:
    keys = random.split(key, game.get_num_players())
    return [
        net.init(subkey, subkey, jnp.empty(game.get_observation_shape(player)))
        for player, (net, subkey) in enumerate(zip(nets, keys, strict=True))
    ]


def sample_actions(
    key: Array, nets: Sequence[Net], params: Sequence, observations: Sequence
) -> list:
    keys = random.split(key, len(nets))
    return [
        player_net.apply(player_params, player_key, player_observation)
        for player_net, player_params, player_key, player_observation in zip(
            nets, params, keys, observations, strict=True
        )
    ]


def make_differentiable_proxy[Input, Output](
    fun: Callable[[Input, Array], Output],
    scale: float,
) -> Callable[[Input, Array], Output]:
    def new_fun(input: Input, key: Array) -> Output:
        key, subkey = random.split(key)
        proxy = optax.perturbations.make_perturbed_fun(
            fun=lambda input: fun(input, subkey),
            num_samples=1,
            sigma=scale,
            noise=optax.perturbations.Normal(),
        )
        return proxy(key, input)

    return new_fun


@dataclass
class Config:
    # game
    game: str = "glicksberg_gross"

    # net
    hidden_dims: Sequence[int] = (64, 64)
    noise_dim: int = 16
    normalization: Literal["rms", "layer", "none"] = "rms"

    # perturbation-based randomized smoothing
    smooth: int = 0
    smooth_scale: float = 0.1

    # optimizer
    optimizer: str = "optimistic"
    lr: float = 1e-2
    optimism: float = 1.0

    # training
    seed: int = 0
    batch_size: int = 64
    iterations: int = 10**5

    # exploitability reporting
    exploitability_every: int = 1000
    exploitability_samples: int = 512
    exploitability_grid: int = 201

    plot_samples: int = 1000


def parse_args() -> Config:
    """Build a `Config` from the command line.

    Mirrors the `Config` dataclass field by field; the dataclass defaults are the
    parser defaults, so running with no arguments reproduces `Config()`.
    """
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # game
    parser.add_argument(
        "--game", choices=["glicksberg_gross"], default=defaults.game
    )

    # net
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=list(defaults.hidden_dims),
        help="widths of the hidden layers",
    )
    parser.add_argument("--noise-dim", type=int, default=defaults.noise_dim)
    parser.add_argument(
        "--normalization",
        choices=["rms", "layer", "none"],
        default=defaults.normalization,
    )

    # perturbation-based randomized smoothing
    parser.add_argument(
        "--smooth",
        type=int,
        default=defaults.smooth,
        help="nonzero enables the perturbed differentiable proxy",
    )
    parser.add_argument("--smooth-scale", type=float, default=defaults.smooth_scale)

    # optimizer
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "optimistic"],
        default=defaults.optimizer,
    )
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--optimism", type=float, default=defaults.optimism)

    # training
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)

    # exploitability reporting
    parser.add_argument(
        "--exploitability-every",
        type=int,
        default=defaults.exploitability_every,
        help="report exploitability every this many iterations (0 disables it)",
    )
    parser.add_argument(
        "--exploitability-samples",
        type=int,
        default=defaults.exploitability_samples,
        help="action profiles drawn per exploitability estimate",
    )
    parser.add_argument(
        "--exploitability-grid",
        type=int,
        default=defaults.exploitability_grid,
        help="grid points searched over [0, 1] for each best response",
    )

    parser.add_argument("--plot-samples", type=int, default=defaults.plot_samples)

    args = parser.parse_args()
    return Config(
        game=args.game,
        hidden_dims=tuple(args.hidden_dims),
        noise_dim=args.noise_dim,
        normalization=args.normalization,
        smooth=args.smooth,
        smooth_scale=args.smooth_scale,
        optimizer=args.optimizer,
        lr=args.lr,
        optimism=args.optimism,
        seed=args.seed,
        batch_size=args.batch_size,
        iterations=args.iterations,
        exploitability_every=args.exploitability_every,
        exploitability_samples=args.exploitability_samples,
        exploitability_grid=args.exploitability_grid,
        plot_samples=args.plot_samples,
    )


def get_game(config: Config) -> Game:
    match config.game:
        case "glicksberg_gross":
            return GlicksbergGross()
        case _:
            raise ValueError(f"Invalid {config.game=}")


def get_nets(config: Config, game: Game) -> list[Net]:
    return [
        Net(
            noise_dim=config.noise_dim,
            hidden_dims=config.hidden_dims,
            output_shape=game.get_action_shape(player),
            normalization=config.normalization,
        )
        for player in range(game.get_num_players())
    ]


def get_optimizer(config: Config) -> optax.GradientTransformation:
    match config.optimizer:
        case "sgd":
            return optax.sgd(config.lr)
        case "optimistic":
            return optax.optimistic_gradient_descent(
                learning_rate=1.0,
                alpha=config.lr,
                beta=config.optimism,
            )
        case _:
            raise ValueError(f"Invalid {config.optimizer=}")


def replace[T](lst: list[T], index: int, item: T) -> list[T]:
    lst = list(lst)
    lst[index] = item
    return lst


def sample_exploitability(
    key: Array, config: Config, game: Game, nets: Sequence[Net], params: Sequence
) -> Array:
    """NashConv: the total gain from every player best-responding to the others.

    A network parametrizes a *distribution* over actions, so each best response
    is taken against the opponent's whole sampled action distribution --
    best-responding to a single sampled action would report a large value even
    at an exact mixed equilibrium. Each best response is found by a grid search
    over `[0, 1]`, which is where the final sigmoid puts the actions, so this
    estimate is restricted to games with scalar actions. Zero at a Nash
    equilibrium, up to grid resolution and Monte-Carlo error.
    """
    num_players = game.get_num_players()
    for player in range(num_players):
        if tuple(game.get_action_shape(player)) != ():
            raise ValueError(
                "exploitability needs scalar actions, got "
                f"{game.get_action_shape(player)=} for {player=}"
            )

    grid = jnp.linspace(0, 1, config.exploitability_grid)
    keys = random.split(key, config.exploitability_samples)

    def sample(key: Array) -> tuple[Array, Array]:
        key, subkey = random.split(key)
        state = game.sample_state(subkey)

        key, subkey = random.split(key)
        observations = game.sample_observations(subkey, state)

        key, subkey = random.split(key)
        actions = sample_actions(subkey, nets, params, observations)

        # the on-policy and the deviation payoffs share `key`, so both are
        # evaluated against the same utility noise (common random numbers)
        utilities = game.sample_utilities(key, state, actions)

        def deviation_utilities(player: int) -> Array:
            def fun(action: Array) -> Array:
                deviated = replace(actions, player, action)
                return game.sample_utilities(key, state, deviated)[player]

            return jax.vmap(fun)(grid)

        deviations = jnp.stack(
            [deviation_utilities(player) for player in range(num_players)]
        )
        return utilities, deviations

    # utilities: (samples, players), deviations: (samples, players, grid)
    utilities, deviations = jax.vmap(sample)(keys)
    values = utilities.mean(0)
    best_response_values = deviations.mean(0).max(-1)
    return jnp.sum(best_response_values - values)


def get_discrete_slider(valmax: int, valmin: int = 0, label: str = "") -> Slider:
    _fig, ax = plt.subplots(figsize=(10, 1))
    return Slider(
        ax=ax,
        label=label,
        valmin=valmin,
        valmax=valmax,
        valstep=1,
        valinit=valmax,
    )


def main(config: Config) -> None:
    game = get_game(config)
    nets = get_nets(config, game)
    optimizer = get_optimizer(config)

    def sample_utilities[T](params: list[T], key: Array, /) -> Array:
        key, subkey = random.split(key)
        state = game.sample_state(subkey)

        key, subkey = random.split(key)
        observations = game.sample_observations(subkey, state)

        key, subkey = random.split(key)
        actions = sample_actions(subkey, nets, params, observations)

        utilities = game.sample_utilities(key, state, actions)
        return utilities

    if config.smooth:
        sample_utilities = make_differentiable_proxy(
            fun=sample_utilities,
            scale=config.smooth_scale,
        )

    def get_player_grad[T](params: list[T], key: Array, player: int) -> T:
        keys = random.split(key, config.batch_size)

        def fun(player_params: T, key: Array) -> Array:
            utilities = sample_utilities(
                replace(params, player, player_params),
                key,
            )
            return utilities[player]

        def batched_fun(player_params: T) -> Array:
            outputs = jax.vmap(fun, [None, 0])(player_params, keys)
            return outputs.mean(0)

        return jax.grad(batched_fun)(params[player])

    def get_grad[T](params: list[T], key: Array) -> list[T]:
        return [
            get_player_grad(params, key, player)
            for player in range(game.get_num_players())
        ]

    def report_exploitability[T](iteration: Array, params: list[T], key: Array) -> None:
        """Print the exploitability of `params`, on the reporting iterations only.

        The whole run is one `lax.scan`, so this goes through `jax.debug.print`
        to get a line out mid-run rather than only once the scan has finished.
        The estimate itself is inside a `lax.cond`, so the untaken branch costs
        nothing on the iterations in between.
        """

        def report(params: list[T]) -> None:
            exploitability = sample_exploitability(key, config, game, nets, params)
            jax.debug.print(
                "iteration {iteration} exploitability {exploitability:.6f}",
                iteration=iteration,
                exploitability=exploitability,
            )

        lax.cond(
            iteration % config.exploitability_every == 0,
            report,
            lambda params: None,
            params,
        )

    def update_state[T](state: tuple[Array, list[T], optax.OptState], key: Array):
        iteration, params, optimizer_state = state

        key, subkey = random.split(key)
        grad = get_grad(params, subkey)
        updates, optimizer_state = optimizer.update(
            jax.tree.map(jnp.negative, grad), optimizer_state, params
        )
        params = optax.apply_updates(params, updates)

        iteration = iteration + 1
        if config.exploitability_every:
            report_exploitability(iteration, params, key)

        return (iteration, params, optimizer_state), params

    def run_trial(key: Array):
        key, subkey = random.split(key)
        params = sample_initial_params(subkey, nets, game)
        optimizer_state = optimizer.init(params)

        if config.exploitability_every:
            key, subkey = random.split(key)
            report_exploitability(jnp.asarray(0), params, subkey)

        state = jnp.asarray(0), params, optimizer_state
        keys = random.split(key, config.iterations)
        state, history = lax.scan(update_state, state, keys)
        _iteration, params, optimizer_state = state
        return {
            "final_params": params,
            "final_optimizer_state": optimizer_state,
            "params_history": history,
        }

    output = run_trial(random.key(config.seed))

    plt.style.use(
        {
            "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
        }
    )

    slider = get_discrete_slider(valmax=config.iterations, label="iteration")

    _fig, axs = plt.subplots(nrows=game.get_num_players())

    state = game.sample_state(random.key(0))
    observations = game.sample_observations(random.key(0), state)
    keys = random.split(random.key(0), config.plot_samples)

    @jax.jit
    def sample_action_profile_batches(params):
        def f(key: Array) -> Sequence[Array]:
            return sample_actions(key, nets, params, observations)

        action_profiles = jax.vmap(f)(keys)
        return action_profiles

    def update(_: float | None) -> None:
        iteration = int(slider.val)

        params = jax.tree.map(lambda leaf: leaf[iteration], output["params_history"])

        action_profiles = sample_action_profile_batches(params)

        for player, ax in enumerate(axs):
            ax.clear()
            ax.set(
                title=f"player {player}",
                xlabel="action",
                ylabel="cdf",
            )

            actions = action_profiles[player]
            ax.ecdf(actions, label="learned")

            try:
                x = jnp.linspace(0, 1, 200)
                y = jax.vmap(game.get_solution_cdf, [0, None])(x, player)
                ax.plot(x, y, label="exact")
            except NotImplementedError:
                pass

            ax.legend()
            ax.figure.canvas.draw_idle()

    slider.on_changed(update)
    update(None)
    plt.show()


if __name__ == "__main__":
    main(parse_args())