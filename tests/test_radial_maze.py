from __future__ import annotations

import numpy as np

from pynacollada import (
    RadialMaze,
    RadialMazeTurns,
    radial_maze,
    radial_maze_turns,
)


def test_radial_maze_counts_and_indices() -> None:
    # data columns: rat, day, trial, configuration, group, arm
    data = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 4],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 5],
            [1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 3],
        ],
        dtype=float,
    )
    baited = {(1, 1): [1, 2, 3]}

    out = radial_maze(data, baited)
    counts_alias, indices_alias = RadialMaze(data, baited)
    counts = out["counts"]
    indices = out["indices"]

    assert counts.shape[0] == 1
    row = counts[0]
    # REF, WORK, REWARDS, VISITS, PERF, PENALTY, ADJUSTED
    np.testing.assert_allclose(row[5:12], np.array([2, 2, 3, 7, 3 / 7, 2, 2], dtype=float))
    np.testing.assert_allclose(counts, counts_alias)
    np.testing.assert_allclose(indices, indices_alias)

    # Count transforms should be sqrt(x+1), perf transform arcsin(sqrt(p))/(pi/2)
    np.testing.assert_allclose(indices[0, 5], np.sqrt(3.0))
    np.testing.assert_allclose(indices[0, 9], np.arcsin(np.sqrt(3.0 / 7.0)) / (np.pi / 2.0))


def test_radial_maze_turns_modes() -> None:
    # Two trials, one starts with preferred arm 1, one starts with arm 2.
    data = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1, 3],
            [1, 1, 1, 1, 1, 4],
            [1, 1, 2, 1, 1, 2],
            [1, 1, 2, 1, 1, 3],
            [1, 1, 2, 1, 1, 4],
            [1, 1, 2, 1, 1, 5],
        ],
        dtype=float,
    )
    out_pref = radial_maze_turns(data, mode="preferred")
    out_non = radial_maze_turns(data, mode="non-preferred")
    pooled_alias = RadialMazeTurns(data, mode="preferred")

    assert out_pref["pooled_distribution"].shape[0] == 8
    assert out_non["pooled_distribution"].shape[0] == 8
    assert np.sum(out_pref["pooled_distribution"]) > 0
    assert np.sum(out_non["pooled_distribution"]) > 0
    np.testing.assert_allclose(out_pref["pooled_distribution"], pooled_alias)

