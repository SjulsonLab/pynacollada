"""Angle computation helper inspired by FMAT GetAngles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pynapple as nap


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data
    kernel = np.ones(window, dtype=float) / float(window)
    out = np.empty_like(data, dtype=float)
    for col in range(data.shape[1]):
        out[:, col] = np.convolve(data[:, col], kernel, mode="same")
    return out


def get_angles(
    positions: np.ndarray | nap.TsdFrame,
    *,
    mode: str = "clean",
    smooth_window: int = 5,
    undetected_value: float = -1.0,
) -> np.ndarray | nap.Tsd:
    """
    Compute head angle from two tracked LEDs.

    Expected coordinate order is `(x1, y1, x2, y2)`.
    For numpy input with shape `(N, >=5)`, the first column is treated as time.
    """
    m = str(mode).lower()
    if m not in {"clean", "all"}:
        raise ValueError("mode must be 'clean' or 'all'.")
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1.")

    if isinstance(positions, nap.TsdFrame):
        arr = np.asarray(positions.values, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("TsdFrame positions must have at least 4 columns: x1,y1,x2,y2.")
        t = positions.as_units("s").index.values.astype(float, copy=False)
    else:
        arr_in = np.asarray(positions, dtype=float)
        if arr_in.ndim != 2 or arr_in.shape[1] < 5:
            raise ValueError("Array positions must be Nx5+: [t, x1, y1, x2, y2, ...].")
        t = arr_in[:, 0]
        arr = arr_in[:, 1:5]

    invalid = np.any(arr == float(undetected_value), axis=1)
    if m == "clean":
        keep = ~invalid
        arr = arr[keep]
        t = t[keep]
    else:
        arr = arr.copy()
        arr[invalid] = np.nan

    if arr.shape[0] == 0:
        if isinstance(positions, nap.TsdFrame):
            return nap.Tsd(t=np.array([], dtype=float), d=np.array([], dtype=float), time_units="s")
        return np.empty((0, 2), dtype=float)

    smoothed = _moving_average(arr, int(smooth_window))
    dx = smoothed[:, 2] - smoothed[:, 0]
    dy = smoothed[:, 3] - smoothed[:, 1]
    angle = np.angle(dx + 1j * dy)

    if isinstance(positions, nap.TsdFrame):
        return nap.Tsd(t=t, d=angle, time_units="s")
    return np.column_stack((t, angle))


def GetAngles(positions: np.ndarray | nap.TsdFrame, *args: Any, **kwargs: Any) -> np.ndarray | nap.Tsd:
    """MATLAB-style alias for :func:`get_angles`."""
    if len(args) % 2 != 0:
        raise ValueError("Positional options must be key/value pairs.")
    options = {str(k).lower(): v for k, v in kwargs.items()}
    for key, value in zip(args[0::2], args[1::2]):
        if not isinstance(key, str):
            raise TypeError("Option keys must be strings.")
        options[key.lower()] = value
    mapped = {
        "mode": options.pop("mode", "clean"),
        "smooth_window": int(options.pop("smooth_window", options.pop("smoothwindow", 5))),
        "undetected_value": float(options.pop("undetected_value", options.pop("undetectedvalue", -1.0))),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_angles(positions, **mapped)


__all__ = [
    "get_angles",
    "GetAngles",
]
