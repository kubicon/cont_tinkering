from pathlib import Path

import jax.numpy as jnp
import numpy as np

from games.mjx_sumo import MjxAntSumo, MjxSumo
from render_mjx import (
    _cap_scale_tril,
    game_type_name,
    match_output_path,
    numeric_checkpoint_steps,
)


def test_numeric_checkpoint_steps_ignores_unrelated_pickles(tmp_path: Path):
    for name in ("10.pkl", "2.pkl", "exploitability.pkl", "notes.txt"):
        (tmp_path / name).touch()

    assert numeric_checkpoint_steps(tmp_path) == [2, 10]


def test_game_type_name_uses_snake_case_class_name():
    game = object.__new__(MjxAntSumo)

    assert game_type_name(game) == "mjx_ant_sumo"


def test_match_output_path_groups_game_step_and_seed():
    game = object.__new__(MjxSumo)

    assert match_output_path(Path("renders"), game, 14, 23) == Path(
        "renders/mjx_sumo/step_14/seed_23.mp4"
    )


def test_match_output_path_separates_a_best_response_run():
    game = object.__new__(MjxSumo)

    assert match_output_path(
        Path("renders"), game, 2, 23, run_name="step14_responder0_vs_live"
    ) == Path(
        "renders/mjx_sumo/step14_responder0_vs_live/step_2/seed_23.mp4"
    )


def test_match_output_path_does_not_repeat_the_standard_run_name():
    game = object.__new__(MjxSumo)

    assert match_output_path(Path("renders"), game, 14, 23, run_name="mjx_sumo") == Path(
        "renders/mjx_sumo/step_14/seed_23.mp4"
    )


def test_cap_scale_tril_only_decreases_diagonal():
    scale_tril = jnp.array([[[0.2, 0.0], [0.3, 0.8]]])

    capped = _cap_scale_tril(scale_tril, 0.5)

    np.testing.assert_allclose(capped, [[[0.2, 0.0], [0.3, 0.5]]])
