"""FMAT-style place-field shift/remapping/skewness analyses."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp


def _validate_field_matrices(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mats = tuple(np.asarray(a, dtype=float) for a in arrays)
    if any(m.ndim != 2 for m in mats):
        raise ValueError("All inputs must be 2D matrices (M fields x N bins).")
    shape0 = mats[0].shape
    if any(m.shape != shape0 for m in mats[1:]):
        raise ValueError("All field matrices must have the same shape.")
    return mats


def _crosscorr_mode(test_row: np.ndarray, control_row: np.ndarray, field_type: str) -> tuple[np.ndarray, float, np.ndarray]:
    n = control_row.shape[0]
    if field_type == "linear":
        xc = np.correlate(test_row, control_row, mode="full")
        # Positive lag means rightward shift of test relative to control (FMAT convention).
        lags = -np.arange(-(n - 1), n, dtype=float) / float(n)
    elif field_type == "circular":
        half = int(np.floor(n / 2))
        shifts = np.arange(-half, half + 1, dtype=int)
        xc = np.array([np.dot(np.roll(test_row, -s), control_row) for s in shifts], dtype=float)
        lags = shifts.astype(float) / float(n)
    else:
        raise ValueError("field_type must be 'linear' or 'circular'.")
    mode = float(lags[int(np.argmax(xc))])
    return xc, mode, lags


def field_shift(control: np.ndarray, test: np.ndarray, field_type: str = "linear") -> dict[str, Any]:
    """
    Estimate place-field shift between control and test conditions.

    Returns absolute and size-normalized (relative) shifts per cell.
    """
    control_m, test_m = _validate_field_matrices(control, test)
    m, n = control_m.shape
    field_type_use = str(field_type).lower()

    absolute = np.zeros(m, dtype=float)
    xc_rows: list[np.ndarray] = []
    lags: np.ndarray | None = None
    for i in range(m):
        row_xc, mode, row_lags = _crosscorr_mode(test_m[i], control_m[i], field_type_use)
        if lags is None:
            lags = row_lags
        xc_rows.append(row_xc)
        absolute[i] = mode
    xc = np.vstack(xc_rows)
    if lags is None:
        lags = np.array([], dtype=float)

    field_size = np.mean(
        np.column_stack((np.sum(control_m > 0, axis=1), np.sum(test_m > 0, axis=1))),
        axis=1,
    ) / float(n)
    relative = absolute / np.maximum(field_size, 1e-12)
    return {"relative_shift": relative, "absolute_shift": absolute, "xc": xc, "lags": lags}


def _bootstrap_mean_ci(values: np.ndarray, alpha: float, iterations: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    n = x.shape[0]
    samples = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        samples[i] = np.mean(x[idx])
    return np.array(
        [np.percentile(samples, 100.0 * alpha / 2.0), np.percentile(samples, 100.0 * (1.0 - alpha / 2.0))],
        dtype=float,
    )


def test_remapping(
    control: np.ndarray,
    repeat: np.ndarray,
    test: np.ndarray,
    field_type: str = "linear",
    alpha: float = 0.05,
    iterations: int = 150,
    *,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Bootstrap test of remapping against random field assignment null.
    """
    control_m, repeat_m, test_m = _validate_field_matrices(control, repeat, test)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    if iterations <= 0:
        raise ValueError("iterations must be > 0.")

    fs_control = field_shift(control_m, repeat_m, field_type=field_type)
    fs_test = field_shift(control_m, test_m, field_type=field_type)
    relative_control = fs_control["relative_shift"]
    relative_test = fs_test["relative_shift"]
    absolute_test = fs_test["absolute_shift"]

    observed = float(np.mean(np.abs(relative_test)))
    rng = np.random.default_rng(random_seed)
    m = control_m.shape[0]
    null_shift = np.empty(iterations, dtype=float)
    for i in range(iterations):
        perm = rng.permutation(m)
        rel_perm = field_shift(control_m, test_m[perm, :], field_type=field_type)["relative_shift"]
        null_shift[i] = np.mean(np.abs(rel_perm))

    p = float((1.0 + np.sum(null_shift <= observed)) / (iterations + 1.0))
    h = bool(p < alpha)
    ca = _bootstrap_mean_ci(absolute_test, alpha=alpha, iterations=iterations, rng=rng)
    cr = _bootstrap_mean_ci(relative_test, alpha=alpha, iterations=iterations, rng=rng)
    return {
        "h": h,
        "p": p,
        "cr": cr,
        "ca": ca,
        "mean_unsigned_relative_shift": observed,
        "relative_shift_control": relative_control,
        "relative_shift_test": relative_test,
        "absolute_shift_test": absolute_test,
        "null_distribution": null_shift,
        "field_shift_control": fs_control,
        "field_shift_test": fs_test,
    }


def _unbiased_skewness(distributions: np.ndarray) -> np.ndarray:
    d = np.asarray(distributions, dtype=float)
    m, n = d.shape
    x = np.tile(np.arange(1, n + 1, dtype=float), (m, 1))
    mu = np.sum(x * d, axis=1)
    x0 = x - mu[:, None]
    m3 = np.sum((x0**3) * d, axis=1)
    s2 = np.sum((x0**2) * d, axis=1)
    s = m3 / np.maximum(s2**1.5, 1e-12)
    if n > 2:
        s = np.sqrt(n * (n - 1)) / (n - 2) * s
    return s


def _semedian(values: np.ndarray, iterations: int, rng: np.random.Generator) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    n = x.shape[0]
    if n <= 1:
        return float(np.nan)
    med = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        med[i] = np.median(x[idx])
    return float(np.std(med, ddof=1))


def test_skewness(
    control: np.ndarray,
    repeat: np.ndarray,
    test: np.ndarray,
    alpha: float = 0.05,
    iterations: int = 150,
    *,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Test whether field-skewness distributions change between conditions.
    """
    control_m, repeat_m, test_m = _validate_field_matrices(control, repeat, test)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    if iterations <= 0:
        raise ValueError("iterations must be > 0.")

    # Discard rows with NaNs and normalize each row to a probability distribution.
    def _prepare(x: np.ndarray) -> np.ndarray:
        keep = ~np.any(np.isnan(x), axis=1)
        y = x[keep].copy()
        denom = np.sum(y, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return y / denom

    c = _prepare(control_m)
    r = _prepare(repeat_m)
    t = _prepare(test_m)
    skew_c = _unbiased_skewness(c)
    skew_r = _unbiased_skewness(r)
    skew_t = _unbiased_skewness(t)

    _, p_control_repeat = ks_2samp(skew_c, skew_r, alternative="two-sided", mode="auto")
    h, p = ks_2samp(skew_c, skew_t, alternative="two-sided", mode="auto")
    rng = np.random.default_rng(random_seed)
    stats = {
        "control": {"m": float(np.median(skew_c)), "s": _semedian(skew_c, iterations=iterations, rng=rng)},
        "repeat": {"m": float(np.median(skew_r)), "s": _semedian(skew_r, iterations=iterations, rng=rng)},
        "test": {"m": float(np.median(skew_t)), "s": _semedian(skew_t, iterations=iterations, rng=rng)},
    }
    return {
        "h": bool(h and p < alpha),
        "p": float(p),
        "p_control_repeat": float(p_control_repeat),
        "stats": stats,
        "skewness_control": skew_c,
        "skewness_repeat": skew_r,
        "skewness_test": skew_t,
    }


def FieldShift(control: np.ndarray, test: np.ndarray, field_type: str = "linear") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB-compatibility alias for `field_shift`."""
    out = field_shift(control, test, field_type=field_type)
    return out["relative_shift"], out["absolute_shift"], out["xc"]


def TestRemapping(
    control: np.ndarray,
    repeat: np.ndarray,
    test: np.ndarray,
    field_type: str = "linear",
    alpha: float = 0.05,
    iterations: int = 150,
    *,
    random_seed: int | None = None,
) -> tuple[bool, float, np.ndarray, np.ndarray]:
    """MATLAB-compatibility alias for `test_remapping`."""
    out = test_remapping(
        control,
        repeat,
        test,
        field_type=field_type,
        alpha=alpha,
        iterations=iterations,
        random_seed=random_seed,
    )
    return out["h"], out["p"], out["cr"], out["ca"]


def TestSkewness(
    control: np.ndarray,
    repeat: np.ndarray,
    test: np.ndarray,
    alpha: float = 0.05,
    iterations: int = 150,
    *,
    random_seed: int | None = None,
) -> tuple[bool, float, dict[str, Any]]:
    """MATLAB-compatibility alias for `test_skewness`."""
    out = test_skewness(
        control,
        repeat,
        test,
        alpha=alpha,
        iterations=iterations,
        random_seed=random_seed,
    )
    return out["h"], out["p"], out["stats"]


# Prevent pytest from auto-collecting API functions imported in test modules.
test_remapping.__test__ = False  # type: ignore[attr-defined]
test_skewness.__test__ = False  # type: ignore[attr-defined]
