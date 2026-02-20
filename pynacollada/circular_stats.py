"""Circular statistics utilities inspired by FMAToolbox General functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import f
from scipy.stats import norm


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def circular_mean(angles: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Estimate circular mean for angles in radians."""
    a = np.asarray(angles, dtype=float)
    return np.angle(np.mean(np.exp(1j * a), axis=axis))


def circular_variance(angles: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate circular variance and standard deviation.

    Returns `(variance, std)` to match FMAT `CircularVariance`.
    """
    a = np.asarray(angles, dtype=float)
    r_bar = np.abs(np.mean(np.exp(1j * a), axis=axis))
    variance = 1.0 - r_bar
    std = np.sqrt(np.maximum(0.0, -2.0 * np.log(np.maximum(r_bar, 1e-15))))
    return variance, std


def concentration(angles: np.ndarray) -> float:
    """
    Estimate von-Mises concentration parameter kappa.

    Uses the approximation in Fisher, matching FMAT `Concentration`.
    """
    a = np.asarray(angles, dtype=float).reshape(-1)
    n = int(a.shape[0])
    if n == 0:
        return np.nan

    r_bar = float(np.abs(np.mean(np.exp(1j * a))))
    if r_bar < 0.53:
        kappa = 2.0 * r_bar + r_bar**3 + 5.0 * r_bar**5 / 6.0
    elif r_bar < 0.85:
        kappa = -0.4 + 1.39 * r_bar + 0.43 / max(1e-12, 1.0 - r_bar)
    else:
        denom = r_bar**3 - 4.0 * r_bar**2 + 3.0 * r_bar
        kappa = 1.0 / np.sign(denom) / max(np.abs(denom), 1e-12)

    if n <= 15:
        if kappa < 2:
            kappa = max(kappa - 2.0 / max(n * max(kappa, 1e-12), 1e-12), 0.0)
        else:
            kappa = (n - 1) ** 3 * kappa / (n**3 + n)
    return float(kappa)


def circular_confidence_intervals(
    angles: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int | None = None,
    random_seed: int | None = None,
) -> tuple[float, np.ndarray]:
    """
    Circular mean with confidence interval bounds.

    Returns `(mean, np.array([lower, upper]))`.
    """
    a = np.asarray(angles, dtype=float).reshape(-1)
    if a.size == 0:
        return np.nan, np.array([np.nan, np.nan], dtype=float)
    if a.size == 1:
        m = float(a[0])
        return m, np.array([m, m], dtype=float)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    m = float(circular_mean(a))
    n = int(a.size)
    if n_bootstrap is None:
        n_bootstrap = 200 if n < 25 else 0

    if n_bootstrap == 0:
        r1 = float(np.abs(np.mean(np.exp(1j * a))))
        r2 = float(np.abs(np.mean(np.exp(1j * (2.0 * a)))))
        denom = max(2.0 * (r1**2), 1e-12)
        delta = (1.0 - r2) / denom
        sigma = np.sqrt(max(delta / n, 0.0))
        sinarg = norm.ppf(1.0 - alpha / 2.0) * sigma
        err = np.arcsin(np.clip(sinarg, -1.0, 1.0))
        bounds = np.array([m - err, m + err], dtype=float)
    else:
        rng = np.random.default_rng(random_seed)
        boot = np.empty(int(n_bootstrap), dtype=float)
        for i in range(int(n_bootstrap)):
            sample = a[rng.integers(0, n, size=n)]
            boot[i] = float(circular_mean(sample))
        unwrapped = (boot - m + np.pi) % (2.0 * np.pi) + m - np.pi
        bounds = np.array(
            [
                np.percentile(unwrapped, 100.0 * (alpha / 2.0)),
                np.percentile(unwrapped, 100.0 - 100.0 * (alpha / 2.0)),
            ],
            dtype=float,
        )
    return m, _wrap_to_pi(bounds)


def _circular_regression_rse(beta: np.ndarray, x: np.ndarray, phi: np.ndarray) -> float:
    a = float(beta[0])
    b = float(beta[1])
    y1 = a * x + b
    y2 = y1 + 2.0 * np.pi
    y3 = y1 - 2.0 * np.pi
    d = np.minimum(np.minimum((phi - y1) ** 2, (phi - y2) ** 2), (phi - y3) ** 2)
    return float(np.sum(d))


def circular_regression(
    x: np.ndarray,
    angles: np.ndarray,
    slope: float = 0.0,
    *,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Non-parametric linear-circular regression with Theil-Sen refinement.

    Returns dictionary with keys:
    - `beta`: least-squares `[slope, intercept]`
    - `R2`: least-squares coefficient of determination
    - `beta_ts`: Theil-Sen `[slope, intercept]`
    - `R2_ts`: Theil-Sen coefficient of determination
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    phi = np.asarray(angles, dtype=float).reshape(-1)
    if x_arr.shape[0] != phi.shape[0]:
        raise ValueError("x and angles must have the same length.")
    if x_arr.shape[0] == 0:
        return {
            "beta": np.array([np.nan, np.nan], dtype=float),
            "R2": np.nan,
            "beta_ts": np.array([np.nan, np.nan], dtype=float),
            "R2_ts": np.nan,
        }

    res = minimize(
        lambda b: _circular_regression_rse(np.asarray(b, dtype=float), x_arr, phi),
        x0=np.array([float(slope), 0.0], dtype=float),
        method="Nelder-Mead",
    )
    beta = np.asarray(res.x, dtype=float)
    rse = _circular_regression_rse(beta, x_arr, phi)
    tse = float(np.linalg.norm(phi - circular_mean(phi)) ** 2)
    r2 = 1.0 - rse / max(tse, 1e-12)

    beta_ts = np.array([np.nan, np.nan], dtype=float)
    r2_ts = np.nan
    if x_arr.shape[0] >= 2:
        d = phi - beta[0] * x_arr + beta[1]
        rd = np.sign(d) * np.floor(np.abs(d / (2.0 * np.pi)))
        phi_adj = phi + rd * 2.0 * np.pi

        n = min(int(x_arr.shape[0]), 500)
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(x_arr.shape[0])[:n]
        x0 = x_arr[perm]
        phi0 = phi_adj[perm]

        xx_i = x0[:, None]
        xx_j = x0[None, :]
        pp_i = phi0[:, None]
        pp_j = phi0[None, :]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        denom = (xx_i - xx_j)[mask]
        numer = (pp_i - pp_j)[mask]
        valid = np.abs(denom) > 1e-12
        if np.any(valid):
            slopes = numer[valid] / denom[valid]
            beta_ts[0] = np.median(slopes)
            beta_ts[1] = np.median(phi_adj - beta_ts[0] * x_arr)
            rse_ts = float(np.sum((phi_adj - (beta_ts[0] * x_arr + beta_ts[1])) ** 2))
            r2_ts = 1.0 - rse_ts / max(tse, 1e-12)

    return {"beta": beta, "R2": float(r2), "beta_ts": beta_ts, "R2_ts": float(r2_ts)}


def CircularVariance(angles: np.ndarray, dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-compatibility alias for `circular_variance`."""
    axis = None if dim is None else int(dim) - 1
    return circular_variance(angles, axis=axis)


def Concentration(angles: np.ndarray) -> float:
    """MATLAB-compatibility alias for `concentration`."""
    return concentration(angles)


def CircularConfidenceIntervals(
    angles: np.ndarray,
    alpha: float = 0.05,
    nBootstrap: int | None = None,
    randomSeed: int | None = None,
) -> tuple[float, np.ndarray]:
    """MATLAB-compatibility alias for `circular_confidence_intervals`."""
    return circular_confidence_intervals(angles, alpha=alpha, n_bootstrap=nBootstrap, random_seed=randomSeed)


def CircularRegression(
    x: np.ndarray,
    angles: np.ndarray,
    slope: float = 0.0,
    *,
    randomSeed: int | None = None,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """MATLAB-compatibility alias for `circular_regression`."""
    out = circular_regression(x, angles, slope=slope, random_seed=randomSeed)
    return out["beta"], out["R2"], out["beta_ts"], out["R2_ts"]


def concentration_test(
    angles: np.ndarray,
    group: np.ndarray,
    alpha: float = 0.05,
    n_randomizations: int = 1000,
    *,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Test homogeneity of concentration parameters across groups.

    Implements FMAT `ConcentrationTest` behavior:
    - asymptotic F-test when median kappa >= 1
    - randomization test otherwise
    """
    a = np.asarray(angles, dtype=float).reshape(-1)
    g = np.asarray(group).reshape(-1)
    if a.shape[0] != g.shape[0]:
        raise ValueError("angles and group must have the same length.")
    if a.shape[0] == 0:
        return {"h": False, "p": np.nan, "fr": np.nan, "kappa": np.array([], dtype=float), "kappa_median": np.nan}
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    unique = np.unique(g)
    r = int(unique.shape[0])
    if r < 2:
        raise ValueError("concentration_test requires at least two groups.")

    counts = np.array([np.sum(g == u) for u in unique], dtype=int)
    if np.min(counts) < 10:
        raise ValueError("concentration_test requires at least 10 samples per group.")

    kappas = np.array([concentration(a[g == u]) for u in unique], dtype=float)
    kappa_med = float(np.median(kappas))
    mus = np.array([circular_mean(a[g == u]) for u in unique], dtype=float)

    def _compute_fr(group_assign: np.ndarray, mu_values: np.ndarray) -> float:
        d = np.zeros(r, dtype=float)
        s = np.zeros(r, dtype=float)
        for i, u in enumerate(unique):
            vals = a[group_assign == u]
            dif = np.abs(np.sin(vals - mu_values[i]))
            d[i] = np.sum(dif) / vals.shape[0]
            s[i] = np.sum((dif - d[i]) ** 2)
        d_bar = float(np.sum(counts * d) / np.sum(counts))
        denom = max((r - 1) * np.sum(s), 1e-12)
        return float((np.sum(counts) - r) * np.sum(counts * (d - d_bar) ** 2) / denom)

    fr = _compute_fr(g, mus)
    if kappa_med >= 1.0:
        p = 1.0 - f.cdf(fr, r - 1, int(np.sum(counts) - r))
        p = 2.0 * min(p, 1.0 - p)
    else:
        rng = np.random.default_rng(random_seed)
        centered = a.copy()
        for i, u in enumerate(unique):
            centered[g == u] = centered[g == u] - mus[i]
        fr_rand = np.zeros(int(n_randomizations), dtype=float)
        for i in range(int(n_randomizations)):
            perm = rng.permutation(centered.shape[0])
            g_perm = g[perm]
            mu_perm = np.array([circular_mean(centered[g_perm == u]) for u in unique], dtype=float)
            fr_rand[i] = _compute_fr(g_perm, mu_perm)
        fr_rand = np.sort(fr_rand)
        ge = np.flatnonzero(fr_rand >= fr)
        if ge.size == 0:
            p = 0.0
        else:
            first = int(ge[0])
            n_ties = int(np.sum(ge == first))
            if n_ties == 1:
                p = (n_randomizations - first) / n_randomizations
            else:
                p = (n_randomizations - first - 1) / n_randomizations + n_ties / (2.0 * n_randomizations)

    return {
        "h": bool(p < alpha),
        "p": float(p),
        "fr": float(fr),
        "kappa": kappas,
        "kappa_median": kappa_med,
        "group_ids": unique,
        "group_counts": counts,
    }


def ConcentrationTest(
    angles: np.ndarray,
    group: np.ndarray,
    alpha: float = 0.05,
    nRandomizations: int = 1000,
    *,
    randomSeed: int | None = None,
) -> tuple[bool, float]:
    """MATLAB-compatibility alias for `concentration_test`."""
    out = concentration_test(
        angles,
        group,
        alpha=alpha,
        n_randomizations=nRandomizations,
        random_seed=randomSeed,
    )
    return out["h"], out["p"]
