"""General FMAT-style transform helpers implemented in a pythonic API."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d


def circular_shift(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Circularly shift each row or column by a potentially different amount.

    If `len(s) == n_rows`, each row is shifted horizontally (right for positive shifts).
    If `len(s) == n_cols`, each column is shifted vertically (down for positive shifts).
    """
    arr = np.asarray(m)
    if arr.ndim != 2:
        raise ValueError("m must be a 2D matrix.")

    shifts = np.asarray(s)
    if shifts.ndim != 1:
        raise ValueError("s must be a 1D vector of integer shifts.")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("s contains non-finite values.")
    if not np.all(np.equal(shifts, np.round(shifts))):
        raise ValueError("s must contain integer shifts.")
    shifts = shifts.astype(int)

    n_rows, n_cols = arr.shape
    out = np.empty_like(arr)
    if shifts.shape[0] == n_rows:
        for i in range(n_rows):
            out[i, :] = np.roll(arr[i, :], int(shifts[i]))
        return out
    if shifts.shape[0] == n_cols:
        for j in range(n_cols):
            out[:, j] = np.roll(arr[:, j], int(shifts[j]))
        return out
    raise ValueError("Incompatible parameter sizes: len(s) must match number of rows or columns.")


def adaptive_smooth(data: np.ndarray, smooth: float | tuple[float, float] | list[float] | np.ndarray) -> np.ndarray:
    """
    Smooth vector/matrix with Gaussian kernel and mirrored boundaries.

    Parameters
    ----------
    data
        1D vector or 2D matrix.
    smooth
        For vectors: scalar standard deviation in samples.
        For matrices: scalar or pair `(Sv, Sh)` in samples.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("adaptive_smooth only supports vectors or 2D matrices.")

    smooth_arr = np.asarray(smooth, dtype=float).reshape(-1)
    if smooth_arr.size == 0:
        raise ValueError("smooth must contain at least one value.")
    if np.any(smooth_arr < 0):
        raise ValueError("smooth values must be non-negative.")

    is_vector = arr.ndim == 1 or min(arr.shape) == 1
    if is_vector:
        if smooth_arr.size != 1:
            raise ValueError("Vector smoothing requires exactly one standard deviation value.")
        sigma = float(smooth_arr[0])
        if sigma == 0:
            return arr.copy()
        flat = arr.reshape(-1)
        filtered = gaussian_filter1d(flat, sigma=sigma, mode="reflect")
        return filtered.reshape(arr.shape)

    if smooth_arr.size == 1:
        sigma_v = float(smooth_arr[0])
        sigma_h = float(smooth_arr[0])
    elif smooth_arr.size == 2:
        sigma_v = float(smooth_arr[0])
        sigma_h = float(smooth_arr[1])
    else:
        raise ValueError("Matrix smoothing accepts either one value or a pair (Sv, Sh).")

    if sigma_v == 0 and sigma_h == 0:
        return arr.copy()
    return gaussian_filter(arr, sigma=(sigma_v, sigma_h), mode="reflect")


def distance_transform(b: np.ndarray) -> np.ndarray:
    """
    Distance to the nearest `1` value in a binary matrix, using Euclidean metric.
    """
    arr = np.asarray(b)
    if arr.ndim != 2:
        raise ValueError("b must be a 2D matrix.")
    if arr.size == 0:
        return np.asarray(arr, dtype=float)
    binary = arr.astype(bool)
    return distance_transform_edt(~binary)


def CircularShift(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """MATLAB-compatibility alias for `circular_shift`."""
    return circular_shift(m, s)


def AdaptiveSmooth(data: np.ndarray, smooth: float | tuple[float, float] | list[float] | np.ndarray) -> np.ndarray:
    """MATLAB-compatibility alias for `adaptive_smooth`."""
    return adaptive_smooth(data, smooth)


def DistanceTransform(b: np.ndarray) -> np.ndarray:
    """MATLAB-compatibility alias for `distance_transform`."""
    return distance_transform(b)

