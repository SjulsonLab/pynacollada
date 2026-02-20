from __future__ import annotations

import numpy as np
import pynapple as nap

from pynacollada import (
    GetAngles,
    get_angles,
)


def test_get_angles_from_array_clean_mode() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],   # 0 rad
            [0.1, 0.0, 0.0, 0.0, 1.0],   # pi/2
            [0.2, 0.0, 0.0, -1.0, -1.0], # undetected
            [0.3, 0.0, 0.0, -2.0, 0.0],  # pi
        ],
        dtype=float,
    )
    out = get_angles(positions, mode="clean", smooth_window=1)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[1, 1], np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(np.abs(out[2, 1]), np.pi, atol=1e-12)


def test_get_angles_from_tsdframe_and_alias() -> None:
    t = np.array([0.0, 0.1, 0.2], dtype=float)
    coords = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],   # 0
            [0.0, 0.0, 0.0, 1.0],   # pi/2
            [0.0, 0.0, -2.0, 0.0],  # pi
        ],
        dtype=float,
    )
    frame = nap.TsdFrame(t=t, d=coords, time_units="s")
    out = get_angles(frame, mode="all", smooth_window=1)
    assert isinstance(out, nap.Tsd)
    vals = out.values
    np.testing.assert_allclose(vals[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(vals[1], np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(np.abs(vals[2]), np.pi, atol=1e-12)

    alias = GetAngles(frame, "mode", "all", "smoothWindow", 1)
    np.testing.assert_allclose(alias.values, out.values, atol=1e-12)
