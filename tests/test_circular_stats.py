from __future__ import annotations

import numpy as np

from pynacollada import (
    CircularConfidenceIntervals,
    CircularRegression,
    CircularVariance,
    Concentration,
    circular_confidence_intervals,
    circular_mean,
    circular_regression,
    circular_variance,
    concentration,
)


def test_circular_mean_and_variance_basic() -> None:
    angles = np.array([0.0, 0.1, -0.1, 0.05, -0.05], dtype=float)
    mean_angle = circular_mean(angles)
    var, std = circular_variance(angles)
    var_alias, std_alias = CircularVariance(angles)

    assert abs(mean_angle) < 0.1
    assert 0 <= var < 0.1
    assert std >= 0
    np.testing.assert_allclose(var, var_alias)
    np.testing.assert_allclose(std, std_alias)


def test_concentration_alias() -> None:
    rng = np.random.default_rng(0)
    tight = 0.1 * rng.standard_normal(500)
    wide = rng.uniform(-np.pi, np.pi, 500)

    k_tight = concentration(tight)
    k_wide = concentration(wide)
    k_tight_alias = Concentration(tight)

    assert np.isfinite(k_tight)
    assert np.isfinite(k_wide)
    assert k_tight > k_wide
    np.testing.assert_allclose(k_tight, k_tight_alias)


def test_circular_confidence_intervals_bootstrap_and_analytic() -> None:
    rng = np.random.default_rng(0)
    angles = 0.4 + 0.2 * rng.standard_normal(120)

    m_a, bounds_a = circular_confidence_intervals(angles, alpha=0.05, n_bootstrap=0)
    m_b, bounds_b = CircularConfidenceIntervals(angles, alpha=0.05, nBootstrap=100, randomSeed=0)

    assert np.isfinite(m_a)
    assert bounds_a.shape == (2,)
    assert np.isfinite(m_b)
    assert bounds_b.shape == (2,)
    assert np.isfinite(bounds_a).all()
    assert np.isfinite(bounds_b).all()


def test_circular_regression_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 250)
    slope_true = 0.75
    intercept_true = -0.5
    noise = 0.15 * rng.standard_normal(x.shape[0])
    angles = (slope_true * x + intercept_true + noise + np.pi) % (2 * np.pi) - np.pi

    out = circular_regression(x, angles, slope=0.5, random_seed=0)
    beta2, r2, beta_ts, r2_ts = CircularRegression(x, angles, slope=0.5, randomSeed=0)

    assert np.isfinite(out["beta"]).all()
    assert np.isfinite(out["R2"])
    assert abs(out["beta"][0] - slope_true) < 0.2
    assert out["R2"] > 0.3

    np.testing.assert_allclose(out["beta"], beta2)
    np.testing.assert_allclose(out["R2"], r2)
    np.testing.assert_allclose(out["beta_ts"], beta_ts, equal_nan=True)
    np.testing.assert_allclose(out["R2_ts"], r2_ts, equal_nan=True)
