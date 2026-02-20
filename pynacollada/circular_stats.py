"""Circular statistics utilities inspired by FMAToolbox General functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import bartlett as _bartlett
from scipy.stats import f as f_dist
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
        p = 1.0 - f_dist.cdf(fr, r - 1, int(np.sum(counts) - r))
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


_WATSON_M1 = np.array(
    [5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, np.inf],
    dtype=float,
)
_WATSON_M2 = np.array(
    [5, 6, 7, 8, 9, 10, 11, 12, 6, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12, 8, 9, 10, 11, 12, 9, 10, 11, 12, 10, 11, 12, np.inf],
    dtype=float,
)
_WATSON_A01 = np.array(
    [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.28,
        0.289,
        0.297,
        0.261,
        np.nan,
        0.282,
        0.298,
        0.262,
        0.248,
        0.262,
        0.259,
        0.304,
        0.272,
        0.255,
        0.262,
        0.253,
        0.252,
        0.25,
        0.258,
        0.249,
        0.252,
        0.252,
        0.266,
        0.254,
        0.255,
        0.254,
        0.255,
        0.255,
        0.255,
        0.268,
    ],
    dtype=float,
)
_WATSON_A05 = np.array(
    [
        0.225,
        0.242,
        0.2,
        0.215,
        0.191,
        0.196,
        0.19,
        0.186,
        0.206,
        0.194,
        0.196,
        0.193,
        0.19,
        0.187,
        0.183,
        0.199,
        0.182,
        0.182,
        0.187,
        0.184,
        0.186,
        0.184,
        0.186,
        0.185,
        0.184,
        0.185,
        0.187,
        0.186,
        0.185,
        0.185,
        0.185,
        0.186,
        0.185,
        0.187,
    ],
    dtype=float,
)


def _watson_u2_statistic(group1: np.ndarray, group2: np.ndarray) -> float:
    g1 = np.asarray(group1, dtype=float).reshape(-1)
    g2 = np.asarray(group2, dtype=float).reshape(-1)
    n1 = int(g1.shape[0])
    n2 = int(g2.shape[0])
    n = n1 + n2
    data = np.column_stack((np.concatenate((g1, g2)), np.concatenate((np.zeros(n1, dtype=int), np.ones(n2, dtype=int)))))
    data = data[np.argsort(data[:, 0])]
    in_g1 = data[:, 1] == 0
    i = np.cumsum(in_g1.astype(float)) / n1
    j = np.cumsum((~in_g1).astype(float)) / n2
    dk = i - j
    return float(n1 * n2 / (n**2) * (np.sum(dk**2) - (np.sum(dk) ** 2) / n))


def _watson_critical_value(n1: int, n2: int, alpha: float) -> float:
    k1 = float(min(n1, n2))
    k2 = float(max(n1, n2))
    if k1 > 10:
        k1 = np.inf
    if k2 > 12:
        k2 = np.inf
    if alpha == 0.01:
        table = _WATSON_A01
    elif alpha == 0.05:
        table = _WATSON_A05
    else:
        raise ValueError("Watson critical-table mode supports alpha=0.05 or alpha=0.01.")
    idx = np.flatnonzero((_WATSON_M1 == k1) & (_WATSON_M2 == k2))
    if idx.size == 0 or np.isnan(table[idx[0]]):
        raise ValueError(f"Watson U2 critical value unavailable for n1={n1}, n2={n2}, alpha={alpha}.")
    return float(table[idx[0]])


def watson_u2_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    n_randomizations: int = 2000,
    *,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Watson U2 two-sample test for circular data.

    For alpha in {0.05, 0.01}, uses FMAT critical-value table.
    For other alpha values, falls back to randomization p-value estimation.
    """
    g1 = np.asarray(group1, dtype=float).reshape(-1)
    g2 = np.asarray(group2, dtype=float).reshape(-1)
    n1 = int(g1.shape[0])
    n2 = int(g2.shape[0])
    if n1 < 5 or n2 < 5:
        raise ValueError("watson_u2_test requires at least 5 angles per group.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    u2 = _watson_u2_statistic(g1, g2)
    if alpha in (0.05, 0.01):
        critical = _watson_critical_value(n1, n2, alpha)
        return {"h": bool(u2 >= critical), "U2": float(u2), "critical": float(critical), "p": np.nan, "method": "table"}

    rng = np.random.default_rng(random_seed)
    all_data = np.concatenate((g1, g2))
    labels = np.concatenate((np.zeros(n1, dtype=int), np.ones(n2, dtype=int)))
    surrogates = np.empty(int(n_randomizations), dtype=float)
    for i in range(int(n_randomizations)):
        perm = rng.permutation(labels.shape[0])
        lp = labels[perm]
        surrogates[i] = _watson_u2_statistic(all_data[lp == 0], all_data[lp == 1])
    p = float((1.0 + np.sum(surrogates >= u2)) / (surrogates.shape[0] + 1.0))
    critical = float(np.percentile(surrogates, 100.0 * (1.0 - alpha)))
    return {"h": bool(p < alpha), "U2": float(u2), "critical": critical, "p": p, "method": "randomization"}


def fisher_test(samples1: np.ndarray, samples2: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Two-sample F-test for equality of variances."""
    x1 = np.asarray(samples1, dtype=float).reshape(-1)
    x2 = np.asarray(samples2, dtype=float).reshape(-1)
    if x1.size == 0 or x2.size == 0:
        return {"h": False, "p": 0.0, "f": 0.0, "df1": 0, "df2": 0}
    if x1.size < 2 or x2.size < 2:
        raise ValueError("fisher_test requires at least 2 samples per group.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    var1 = float(np.var(x1, ddof=1))
    var2 = float(np.var(x2, ddof=1))
    df1 = int(x1.size - 1)
    df2 = int(x2.size - 1)
    if np.isclose(var1, 0.0) and np.isclose(var2, 0.0):
        return {"h": False, "p": 1.0, "f": 1.0, "df1": df1, "df2": df2}

    ratio = var1 / max(var2, 1e-15)
    if ratio >= 1.0:
        f_stat = float(ratio)
        p = float(2.0 * f_dist.sf(f_stat, df1, df2))
    else:
        f_stat = float(1.0 / max(ratio, 1e-15))
        p = float(2.0 * f_dist.sf(f_stat, df2, df1))
    p = float(np.clip(p, 0.0, 1.0))
    return {"h": bool(p < alpha), "p": p, "f": f_stat, "df1": df1, "df2": df2}


def bartlett_test(values: np.ndarray, group: np.ndarray | None = None, alpha: float = 0.05) -> dict[str, Any]:
    """
    Bartlett test for homogeneity of variances across groups.

    Parameters
    ----------
    values
        Values vector or Nx2 matrix `[value, group]` when `group is None`.
    group
        Group labels when `values` is a vector.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    if group is None:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("When group is None, values must be an Nx2 matrix [value, group].")
        x = arr[:, 0]
        g = arr[:, 1]
    else:
        x = np.asarray(values, dtype=float).reshape(-1)
        g = np.asarray(group).reshape(-1)
        if x.shape[0] != g.shape[0]:
            raise ValueError("values and group must have the same length.")

    unique = np.unique(g)
    if unique.shape[0] < 2:
        raise ValueError("bartlett_test requires at least two groups.")
    groups = [x[g == u] for u in unique]
    if any(v.shape[0] < 2 for v in groups):
        raise ValueError("bartlett_test requires at least 2 observations per group.")

    t_stat, p = _bartlett(*groups)
    variances = np.array([np.var(v, ddof=1) for v in groups], dtype=float)
    counts = np.array([v.shape[0] for v in groups], dtype=int)
    return {
        "h": bool(p < alpha),
        "p": float(p),
        "T": float(t_stat),
        "variances": variances,
        "group_ids": unique,
        "group_counts": counts,
    }


def _oneway_location_statistic(angles: np.ndarray, group: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    a = np.asarray(angles, dtype=float).reshape(-1)
    g = np.asarray(group).reshape(-1)
    unique = np.unique(g)
    q = int(unique.shape[0])
    n = int(a.shape[0])
    if q < 2:
        raise ValueError("At least two groups are required.")
    mu = float(circular_mean(a))
    ssb = 0.0
    ssw = 0.0
    counts = np.zeros(q, dtype=int)
    for i, gid in enumerate(unique):
        vals = a[g == gid]
        counts[i] = int(vals.shape[0])
        mu_i = float(circular_mean(vals))
        ssb += vals.shape[0] * (1.0 - np.cos(mu_i - mu))
        ssw += float(np.sum(1.0 - np.cos(vals - mu_i)))
    if np.any(counts < 2):
        raise ValueError("Each group must contain at least two observations.")
    f_stat = float((n - q) / max(q - 1, 1) * ssb / max(ssw, 1e-12))
    return f_stat, unique, counts


def _twoway_lr_stats(angles: np.ndarray, factor1: np.ndarray, factor2: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    a = np.asarray(angles, dtype=float).reshape(-1)
    f1 = np.asarray(factor1).reshape(-1)
    f2 = np.asarray(factor2).reshape(-1)
    lv1 = np.unique(f1)
    lv2 = np.unique(f2)
    if lv1.shape[0] != 2 or lv2.shape[0] != 2:
        raise ValueError("Two-way lr mode currently requires a 2x2 design.")
    counts = np.zeros((2, 2), dtype=int)
    cell_means = np.zeros((2, 2), dtype=complex)
    for i, a1 in enumerate(lv1):
        for j, a2 in enumerate(lv2):
            mask = (f1 == a1) & (f2 == a2)
            vals = a[mask]
            counts[i, j] = int(vals.shape[0])
            if vals.shape[0] == 0:
                raise ValueError("Two-way lr mode requires non-empty cells.")
            cell_means[i, j] = np.mean(np.exp(1j * vals))
    if not np.all(counts == counts[0, 0]):
        raise ValueError("Two-way lr mode requires a balanced design.")
    m = float(counts[0, 0])

    A = np.exp(1j * a)
    theta = float(np.angle(np.mean(A)))
    theta_i = np.array([float(np.angle(np.mean(A[f1 == lv1[k]]))) for k in range(2)], dtype=float)
    theta_j = np.array([float(np.angle(np.mean(A[f2 == lv2[k]]))) for k in range(2)], dtype=float)
    theta_ij = np.array([[float(np.angle(cell_means[i, j])) for j in range(2)] for i in range(2)], dtype=float)

    phi_l = np.array(
        [
            float(np.angle(np.mean(np.array([cell_means[0, 0], cell_means[1, 1]], dtype=complex)))),
            float(np.angle(np.mean(np.array([cell_means[0, 1], cell_means[1, 0]], dtype=complex)))),
        ],
        dtype=float,
    )

    ss_a = float(4.0 * m * np.sum(1.0 - np.cos(theta_i - theta)))
    ss_b = float(4.0 * m * np.sum(1.0 - np.cos(theta_j - theta)))
    ss_ab = float(4.0 * m * np.sum(1.0 - np.cos(phi_l - theta)))

    ss_r = 0.0
    for i, a1 in enumerate(lv1):
        for j, a2 in enumerate(lv2):
            vals = a[(f1 == a1) & (f2 == a2)]
            ss_r += float(np.sum(2.0 * (1.0 - np.cos(vals - theta_ij[i, j]))))

    ss_r = max(ss_r, 1e-12)
    stats = np.array([ss_a / ss_r, ss_b / ss_r, ss_ab / ss_r], dtype=float)
    meta = {"levels1": lv1, "levels2": lv2, "cell_counts": counts}
    return stats, meta


def circular_anova(
    angles: np.ndarray,
    factors: np.ndarray,
    method: str = "ww",
    n_randomizations: int = 500,
    *,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Circular ANOVA on one-way or two-way factors.

    Notes
    -----
    Supported modes:
    - one-way: `ww`, `l2`, `lr`
    - two-way (Nx2 factors): `lr` (2x2 balanced design)
    """
    a = np.asarray(angles, dtype=float).reshape(-1)
    m = str(method).lower()
    if a.shape[0] == 0:
        return {"p": np.nan, "F": np.nan, "method": m}

    f_raw = np.asarray(factors)
    if f_raw.ndim == 1:
        f = f_raw.reshape(-1)
        if a.shape[0] != f.shape[0]:
            raise ValueError("angles and factors must have the same length.")
        group_ids = np.unique(f)
        q = int(group_ids.shape[0])
        n = int(a.shape[0])
        if q < 2:
            raise ValueError("circular_anova requires at least two groups.")
        if n <= q:
            raise ValueError("Not enough samples for ANOVA degrees of freedom.")

        if m == "ww":
            A = np.exp(1j * a)
            R = float(np.abs(np.mean(A)))
            Ri = np.zeros(q, dtype=float)
            Ni = np.zeros(q, dtype=float)
            ki = np.zeros(q, dtype=float)
            for i, gid in enumerate(group_ids):
                mask = f == gid
                group_angles = a[mask]
                Ni[i] = float(group_angles.shape[0])
                Ri[i] = float(np.abs(np.mean(np.exp(1j * group_angles))))
                ki[i] = float(concentration(group_angles))
            if np.any(Ni < 2):
                raise ValueError("Each group must contain at least two observations.")

            ssw = float(n - np.sum(Ni * Ri))
            ssb = float(np.sum(Ni * Ri) - n * R)
            f_stat = float((n - q) / (q - 1) * ssb / max(ssw, 1e-12))
            kappa = float(np.sum(ki * Ni) / n)
            if 2.0 < kappa < 10.0:
                f_stat *= 1.0 + 3.0 / (8.0 * kappa)
            p = float(1.0 - f_dist.cdf(f_stat, q - 1, n - q))
            return {
                "p": p,
                "F": f_stat,
                "method": "ww",
                "df_between": int(q - 1),
                "df_within": int(n - q),
                "group_ids": group_ids,
                "group_counts": Ni.astype(int),
            }

        if m not in ("l2", "lr"):
            raise ValueError("One-way circular_anova method must be one of {'ww','l2','lr'}.")
        f_stat, _, counts = _oneway_location_statistic(a, f)
        rng = np.random.default_rng(random_seed)
        surrogates = np.empty(int(n_randomizations), dtype=float)
        for i in range(int(n_randomizations)):
            perm = rng.permutation(f.shape[0])
            surrogates[i], _, _ = _oneway_location_statistic(a, f[perm])
        p = float((1.0 + np.sum(surrogates >= f_stat)) / (int(n_randomizations) + 1.0))
        return {
            "p": p,
            "F": float(f_stat),
            "method": m,
            "group_ids": group_ids,
            "group_counts": counts.astype(int),
            "n_randomizations": int(n_randomizations),
        }

    if f_raw.ndim == 2 and f_raw.shape[1] == 2:
        if a.shape[0] != f_raw.shape[0]:
            raise ValueError("angles and factors must have matching first dimension.")
        if m != "lr":
            raise NotImplementedError("Two-way circular_anova currently supports method='lr' only.")

        f1 = f_raw[:, 0]
        f2 = f_raw[:, 1]
        f_obs, meta = _twoway_lr_stats(a, f1, f2)
        rng = np.random.default_rng(random_seed)
        f_perm = np.empty((int(n_randomizations), 3), dtype=float)
        for i in range(int(n_randomizations)):
            perm = rng.permutation(a.shape[0])
            f_perm[i, :], _ = _twoway_lr_stats(a[perm], f1, f2)
        p = (1.0 + np.sum(f_perm >= f_obs[None, :], axis=0)) / (int(n_randomizations) + 1.0)
        return {
            "p": p.astype(float),
            "F": f_obs.astype(float),
            "method": "lr",
            "terms": np.array(["factor1", "factor2", "interaction"], dtype=object),
            "n_randomizations": int(n_randomizations),
            "levels1": meta["levels1"],
            "levels2": meta["levels2"],
            "cell_counts": meta["cell_counts"],
        }

    raise ValueError("factors must be a 1D group vector or Nx2 matrix for two-way analysis.")


def multinomial_confidence_intervals(samples: np.ndarray, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """
    Simultaneous multinomial confidence intervals (Fitzpatrick & Scott style).

    Matches FMAT behavior for alpha levels 0.1, 0.05, and 0.01.
    """
    s = np.asarray(samples, dtype=float).reshape(-1)
    if s.size == 0:
        return {"p": np.array([], dtype=float), "boundaries": np.empty((2, 0), dtype=float)}
    if np.any(s < 0):
        raise ValueError("samples must be non-negative.")
    n = float(np.sum(s))
    if n <= 0:
        raise ValueError("sum(samples) must be positive.")

    if alpha == 0.1:
        k = 1.0
    elif alpha == 0.05:
        k = 1.13
    elif alpha == 0.01:
        k = 1.40
    else:
        raise ValueError("alpha must be one of {0.1, 0.05, 0.01}.")

    p = s / n
    b = float(k / np.sqrt(n))
    boundaries = np.vstack((np.maximum(p - b, 0.0), np.minimum(p + b, 1.0)))
    return {"p": p, "boundaries": boundaries}


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


def WatsonU2Test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """MATLAB-compatibility alias for `watson_u2_test`."""
    out = watson_u2_test(group1, group2, alpha=alpha)
    return out["h"], out["U2"]


def FisherTest(
    samples1: np.ndarray,
    samples2: np.ndarray,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """MATLAB-compatibility alias for `fisher_test`."""
    out = fisher_test(samples1, samples2, alpha=alpha)
    return out["h"], out["p"], out["f"]


def BartlettTest(
    data: np.ndarray,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """MATLAB-compatibility alias for `bartlett_test` with Nx2 input."""
    out = bartlett_test(data, alpha=alpha)
    return out["h"], out["p"], out["T"]


def CircularANOVA(
    angles: np.ndarray,
    factors: np.ndarray,
    method: str = "ww",
) -> tuple[float, float]:
    """MATLAB-compatibility alias for `circular_anova`."""
    out = circular_anova(angles, factors, method=method)
    return out["p"], out["F"]


def MultinomialConfidenceIntervals(
    samples: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-compatibility alias for `multinomial_confidence_intervals`."""
    out = multinomial_confidence_intervals(samples, alpha=alpha)
    return out["p"], out["boundaries"]
