"""FMAT-style damped-sine fitting for cross-correlograms."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks


def _damped_sine_model(t: np.ndarray, beta: np.ndarray) -> np.ndarray:
    a = abs(float(beta[0]))
    tau = abs(float(beta[1]))
    mu = float(beta[2])
    omega = float(beta[3])
    phi = float(beta[4])
    b = float(beta[5])
    return (a * (np.sin(2.0 * np.pi * omega * t + phi) + 1.0) + b) * np.exp(-tau * np.abs(t - mu))


def _closest_peak_to_zero(t: np.ndarray, y: np.ndarray) -> float:
    peaks, _ = find_peaks(y)
    if peaks.size == 0:
        return float(t[int(np.argmax(y))])
    peak_t = t[peaks]
    pos = peak_t[peak_t > 0]
    neg = peak_t[peak_t < 0]
    if pos.size == 0 and neg.size == 0:
        return float(peak_t[np.argmin(np.abs(peak_t))])
    if pos.size == 0:
        return float(neg[-1])
    if neg.size == 0:
        return float(pos[0])
    return float(pos[0] if pos[0] < abs(neg[-1]) else neg[-1])


def fit_ccg(t: np.ndarray, ccg: np.ndarray) -> dict[str, Any]:
    """
    Fit damped sine-wave model to a cross-correlogram (FMAT `FitCCG` style).

    Returns
    -------
    dict
        Keys include:
        - `dt1`: first-peak estimate from smoothed empirical CCG
        - `dt2`: first-peak estimate from fitted model
        - `travel`: travel-time estimate (maximum of lightly smoothed fit)
        - `index`: theta modulation index (`a / b`)
        - `beta`: fitted parameters `[a, tau, mu, omega, phi, b]`
        - `fit`: fitted model values at `t`
    """
    tt = np.asarray(t, dtype=float).reshape(-1)
    yy = np.asarray(ccg, dtype=float).reshape(-1)
    if tt.shape[0] != yy.shape[0]:
        raise ValueError("t and ccg must have the same length.")
    if tt.shape[0] < 5:
        raise ValueError("fit_ccg requires at least 5 samples.")

    smoothed = uniform_filter1d(yy, size=3, mode="nearest")
    dt1 = _closest_peak_to_zero(tt, smoothed)

    b0 = float(np.mean(yy))
    beta0 = np.array([b0, 0.5, 0.0, 7.0, 2.0 * np.pi * 7.0 * dt1, b0], dtype=float)

    def objective(beta: np.ndarray) -> float:
        model = _damped_sine_model(tt, beta)
        return float(np.sum((yy - model) ** 2))

    res = minimize(objective, beta0, method="Nelder-Mead")
    beta = np.asarray(res.x, dtype=float)
    fit = _damped_sine_model(tt, beta)
    dt2 = _closest_peak_to_zero(tt, fit)
    travel = float(tt[int(np.argmax(uniform_filter1d(fit, size=5, mode="nearest")))])

    a = abs(float(beta[0]))
    b = float(beta[5])
    index = float(a / max(abs(b), 1e-12))

    beta_out = np.array([a, -abs(float(beta[1])), float(beta[2]), float(beta[3]), float(beta[4]), b], dtype=float)
    return {"dt1": float(dt1), "dt2": float(dt2), "travel": travel, "index": index, "beta": beta_out, "fit": fit}


def FitCCG(t: np.ndarray, ccg: np.ndarray) -> tuple[float, float, float, float]:
    """MATLAB-compatibility alias for `fit_ccg`."""
    out = fit_ccg(t, ccg)
    return out["dt1"], out["dt2"], out["travel"], out["index"]

