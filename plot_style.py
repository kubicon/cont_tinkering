"""One visual language for the method-comparison plots.

Both comparison experiments draw the same methods against the same two axes --
``experiments/one_shot_neural/score.py`` for the one-shot games,
``rci_scripts/plot_sequential_sweep.py`` for the sequential ones -- and a reader
putting their figures side by side should not have to re-learn the colours. This
module is the single place that says what a method looks like; both scripts
import it and neither keeps a palette of its own.

**Edit ``METHOD_STYLES``; that is the interface.** One entry per method:

    "mixture": {
        "label": "MGPO (ours)",      # legend name
        "short_label": "MGPO",       # used where the name already takes a qualifier
        "color": "#eb6834",          # the line, the band and the marker face
        "linestyle": "-",            # second encoding, so colour is never alone
        "marker": "o",               # third; drawn sparsely on training curves
        "ramp": MIXTURE_ORANGES,     # optional: light->dark by capacity
        "capacity_unit": "components",
        "aliases": ("self_play",),   # other names the same method runs under
    }

The two experiments name their runs differently -- the sequential sweep's
directories say ``self_play`` and ``discrete_mmd``, the one-shot tree's say
``mixture`` and ``mmd_discrete`` -- so every entry carries the aliases it answers
to and lookups go through ``style_of``, which resolves them. An unknown name
falls back to ``FALLBACK_STYLE`` rather than raising: a new ablation plots in
grey until it is given an entry here.

``ramp`` is what a capacity sweep needs: its per-K curves are shades of the
method's own hue, so ten lines read as two families rather than ten unrelated
colours. A method without one draws every K in its base colour.
"""

from __future__ import annotations

CI_LEVEL = 0.95

# Ink. The plots are drawn for a white page, in near-black rather than black.
TEXT = "#3d3d3a"
MUTED = "#73726c"
GRID = "#e4e3dd"

# Line weights and the confidence band's opacity, shared so two figures from
# different scripts sit on a page as one set.
LINEWIDTH = 1.6
SEED_LINEWIDTH = 0.7
BAND_ALPHA = 0.15
# A second, secondary series in the same colour (the capacity plots' best-checkpoint
# line) takes a fainter band, so two overlapping intervals stay readable.
SECONDARY_BAND_ALPHA = 0.08
MARKERSIZE = 5
# Markers are a secondary cue, not a data point count: a curve gets about this
# many of them however long it is.
MARKERS_PER_CURVE = 8

# ---------------------------------------------------------------- type sizes
#
# **Turn FONT_SCALE up to make every label in both experiments bigger at once**;
# the individual sizes below are points at scale 1.0, and each script reads them
# rather than hard-coding a number. A figure that will be shrunk into a
# two-column paper wants something like 1.3.
FONT_SCALE = 1.4

AXIS_LABEL_SIZE = 12 * FONT_SCALE
TICK_LABEL_SIZE = 11 * FONT_SCALE
LEGEND_SIZE = 10 * FONT_SCALE
# A dense grid of panels (the sequential sweep's `overview.png`) gets its own,
# smaller pair, since every panel there carries a label and a legend.
SMALL_AXIS_LABEL_SIZE = 10 * FONT_SCALE
SMALL_LEGEND_SIZE = 9 * FONT_SCALE
TITLE_SIZE = 13 * FONT_SCALE

# Wall-time is stored in seconds by both experiments, but they run on different
# clocks -- minutes for a one-shot game, hours for a sequential one -- so each
# script picks its unit from here by name.
WALL_TIME_UNITS = {
    "seconds": ("Training wall-time [s]", 1.0),
    "minutes": ("Training wall-time [min]", 1.0 / 60.0),
    "hours": ("Training wall-time [h]", 1.0 / 3600.0),
}
# The cost axis both experiments share, whatever each one counts underneath.
SAMPLES_LABEL = "Environment interactions"


def wall_time_axis(unit: str) -> tuple[str, float]:
    """`(axis label, factor to multiply the stored seconds by)` for one unit."""
    if unit not in WALL_TIME_UNITS:
        raise SystemExit(f"unknown wall-time unit {unit!r}; "
                         f"choices: {sorted(WALL_TIME_UNITS)}")
    return WALL_TIME_UNITS[unit]

# Ordered shades for a capacity sweep, light (small K) to dark (large K).
MIXTURE_ORANGES = {1: "#fbd5c2", 2: "#f6ab86", 3: "#f2954f", 4: "#ef7f4c", 6: "#ed7340",
                   8: "#eb6834", 12: "#d05726", 16: "#b8471c", 24: "#993a11",
                   32: "#7a2c05", 64: "#511d03"}
DISCRETE_MMD_BLUES = {1: "#c3dbf7", 2: "#a5c9f3", 3: "#95c0f1", 4: "#86b6ef", 6: "#5f9eea",
                      8: "#3987e5", 12: "#2a72c8", 16: "#1c5cab", 24: "#154989",
                      32: "#0d366b", 64: "#06203f"}

METHOD_STYLES: dict[str, dict] = {
    "mixture": {
        "label": "MGPO (ours)", "short_label": "MGPO",
        "color": "#eb6834", "linestyle": "-", "marker": "o",
        "ramp": MIXTURE_ORANGES, "capacity_unit": "components",
        "aliases": ("self_play",),
    },
    "ppo": {
        "label": "PPO", "color": "#1baf7a", "linestyle": "--", "marker": "s",
        "aliases": ("sac",),
    },
    "nfsp": {
        "label": "NFSP", "color": "#e87ba4", "linestyle": "-.", "marker": "^",
    },
    "psro": {
        "label": "PSRO", "color": "#4a3aa7", "linestyle": ":", "marker": "D",
    },
    "discrete_mmd": {
        "label": "Discrete MMD", "color": "#1c5cab", "linestyle": "-", "marker": "v",
        "ramp": DISCRETE_MMD_BLUES, "capacity_unit": "bins",
        "aliases": ("mmd_discrete", "mmd"),
    },
    "rpn": {
        "label": "RPN", "color": "#9c6644",
        "linestyle": (0, (5, 1, 1, 1)), "marker": "P",
        "aliases": ("rpn_pathwise",),
    },
    "spg": {
        "label": "Randomized policy (black-box)", "color": "#c9a227",
        "linestyle": (0, (3, 1, 1, 1)), "marker": "X",
    },
    "jpspg": {
        "label": "JPSPG", "color": "#00868b", "linestyle": (0, (4, 2)), "marker": "<",
    },
    "sisa": {
        "label": "SISA", "color": "#8a5d9e", "linestyle": (0, (1, 1)), "marker": ">",
    },
}
FALLBACK_STYLE = {"label": None, "color": "#8a8984", "linestyle": "-", "marker": "x",
                  "capacity_unit": "components"}

# Legend / draw order. Anything not listed sorts after, alphabetically.
LEGEND_ORDER = tuple(METHOD_STYLES)

# Methods left out of the *baseline* comparison, by canonical name. A plain run of one
# is dropped from the training-curve plots; its capacity cells are kept, because the
# capacity experiment is the one it is run for and has plots of its own.
EXCLUDED_BASELINES = ("discrete_mmd",)

# alias -> canonical key, built once from the entries above.
_ALIASES = {alias: key
            for key, style in METHOD_STYLES.items()
            for alias in (key, *style.get("aliases", ()))}


def canonical(method: str) -> str:
    """The key ``method`` is known by here (``self_play`` -> ``mixture``)."""
    return _ALIASES.get(method, method)


def style_of(method: str) -> dict:
    """One method's style entry, by any of its names; grey fallback if unknown."""
    return METHOD_STYLES.get(canonical(method), FALLBACK_STYLE)


def display_name(method: str) -> str:
    return style_of(method)["label"] or method.replace("_", " ")


def short_name(method: str) -> str:
    """The name to use where a qualifier already follows it.

    A capacity cell reads "MGPO (8 components)", not "MGPO (ours) (8 components)": the
    "(ours)" belongs on the baseline legend, where the reader is comparing methods, and
    only clutters a legend that is already one method's ramp.
    """
    style = style_of(method)
    return style.get("short_label") or display_name(method)


def excluded_baseline(method: str, capacity: int | None = None) -> bool:
    """True when this run should not appear as a baseline (see EXCLUDED_BASELINES).

    A run with a `capacity` is a cell of a capacity sweep, never a baseline, so it is
    kept whatever its method.
    """
    return capacity is None and canonical(method) in EXCLUDED_BASELINES


def capacity_unit(method: str) -> str:
    """What a capacity sweep of this method counts: components, bins, ..."""
    return style_of(method).get("capacity_unit", "components")


def capacity_color(method: str, k: int):
    """The shade for one cell of a capacity sweep, dark for large ``k``."""
    ramp = style_of(method).get("ramp")
    if not ramp:
        return style_of(method)["color"]
    if k in ramp:
        return ramp[k]
    # A K the ramp has no shade for takes the nearest one below it, so an
    # unexpected capacity still lands in the right family and the right order.
    below = [key for key in ramp if key <= k]
    return ramp[max(below)] if below else ramp[min(ramp)]


def capacity_label(method: str, k: int) -> str:
    return f"{short_name(method)} ({k} {capacity_unit(method)})"


def sort_key(method: str, capacity: int | None = None) -> tuple:
    """Draw / legend order: capacity cells first by method and K, then baselines."""
    key = canonical(method)
    rank = LEGEND_ORDER.index(key) if key in LEGEND_ORDER else len(LEGEND_ORDER)
    if capacity is not None:
        return (0, rank, capacity, key)
    return (1, rank, 0, key)


def markevery(n_points: int) -> int:
    """Marker spacing that keeps a curve to roughly ``MARKERS_PER_CURVE`` markers."""
    return max(1, n_points // MARKERS_PER_CURVE)


def style_axes(ax) -> None:
    """The frame both experiments use: no top/right spine, muted ticks, soft grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=TICK_LABEL_SIZE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_axes(ax, xlabel: str | None = None, ylabel: str | None = None,
               fontsize: float | None = None) -> None:
    """Set either axis label in the shared colour and type size."""
    size = AXIS_LABEL_SIZE if fontsize is None else fontsize
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=size)


def legend(ax, **kwargs):
    """The legend both experiments use: inside the panel, no frame."""
    options = dict(fontsize=LEGEND_SIZE, frameon=False, labelcolor=TEXT)
    options.update(kwargs)
    return ax.legend(**options)
