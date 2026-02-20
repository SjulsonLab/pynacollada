from __future__ import annotations

import numpy as np

from pynacollada import (
    AdaptiveSmooth,
    CircularShift,
    DistanceTransform,
    adaptive_smooth,
    circular_shift,
    distance_transform,
)


def test_circular_shift_rows_and_columns() -> None:
    m = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=int)

    row_shift = np.array([1, -1], dtype=int)
    out_rows = circular_shift(m, row_shift)
    expected_rows = np.array([[4, 1, 2, 3], [6, 7, 8, 5]], dtype=int)
    np.testing.assert_array_equal(out_rows, expected_rows)
    np.testing.assert_array_equal(out_rows, CircularShift(m, row_shift))

    col_shift = np.array([0, 1, 0, -1], dtype=int)
    out_cols = circular_shift(m, col_shift)
    expected_cols = np.column_stack(
        (
            m[:, 0],
            np.roll(m[:, 1], 1),
            m[:, 2],
            np.roll(m[:, 3], -1),
        )
    )
    np.testing.assert_array_equal(out_cols, expected_cols)


def test_adaptive_smooth_vector_and_matrix() -> None:
    vec = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    smoothed_vec = adaptive_smooth(vec, 1.0)
    assert smoothed_vec.shape == vec.shape
    assert smoothed_vec[2] < vec[2]
    np.testing.assert_allclose(smoothed_vec, AdaptiveSmooth(vec, 1.0))

    mat = np.zeros((5, 5), dtype=float)
    mat[2, 2] = 1.0
    smoothed_mat = adaptive_smooth(mat, (1.0, 1.0))
    assert smoothed_mat.shape == mat.shape
    assert smoothed_mat[2, 2] < 1.0
    assert smoothed_mat[2, 2] > 0.0


def test_distance_transform_matches_expected_geometry() -> None:
    b = np.zeros((3, 3), dtype=int)
    b[1, 1] = 1
    dt = distance_transform(b)

    assert dt[1, 1] == 0.0
    np.testing.assert_allclose(dt[0, 0], np.sqrt(2.0))
    np.testing.assert_allclose(dt[0, 1], 1.0)
    np.testing.assert_allclose(dt, DistanceTransform(b))

