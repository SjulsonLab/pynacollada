from __future__ import annotations

import numpy as np

from pynacollada import (
    BartlettTest,
    CircularANOVA,
    CircularConfidenceIntervals,
    ConcentrationTest,
    FisherTest,
    MultinomialConfidenceIntervals,
    CircularRegression,
    CircularVariance,
    Concentration,
    WatsonU2Test,
    bartlett_test,
    circular_anova,
    circular_confidence_intervals,
    circular_mean,
    concentration_test,
    fisher_test,
    multinomial_confidence_intervals,
    circular_regression,
    circular_variance,
    concentration,
    watson_u2_test,
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


def test_concentration_test_detects_dispersion_difference() -> None:
    rng = np.random.default_rng(0)
    group1 = 0.15 * rng.standard_normal(120)
    group2 = rng.uniform(-np.pi, np.pi, 120)
    angles = np.concatenate((group1, group2))
    groups = np.concatenate((np.ones(group1.size, dtype=int), 2 * np.ones(group2.size, dtype=int)))

    out = concentration_test(angles, groups, alpha=0.05, n_randomizations=500, random_seed=0)
    h_alias, p_alias = ConcentrationTest(angles, groups, alpha=0.05, nRandomizations=500, randomSeed=0)

    assert out["h"] is True
    assert out["p"] < 0.05
    assert h_alias is True
    np.testing.assert_allclose(out["p"], p_alias)


def test_concentration_test_no_difference_case() -> None:
    rng = np.random.default_rng(1)
    group1 = 0.2 * rng.standard_normal(150)
    group2 = np.pi / 2 + 0.2 * rng.standard_normal(150)
    angles = np.concatenate((group1, group2))
    groups = np.concatenate((np.ones(group1.size, dtype=int), 2 * np.ones(group2.size, dtype=int)))

    out = concentration_test(angles, groups, alpha=0.05, n_randomizations=500, random_seed=0)
    assert out["p"] > 0.01


def test_fisher_test_detects_variance_difference() -> None:
    rng = np.random.default_rng(0)
    x1 = rng.normal(0.0, 0.3, size=200)
    x2 = rng.normal(0.0, 1.2, size=200)
    out = fisher_test(x1, x2, alpha=0.05)
    h_alias, p_alias, f_alias = FisherTest(x1, x2, alpha=0.05)

    assert out["h"] is True
    assert out["p"] < 0.05
    assert h_alias is True
    np.testing.assert_allclose(out["p"], p_alias)
    np.testing.assert_allclose(out["f"], f_alias)


def test_bartlett_test_detects_group_variance_difference() -> None:
    rng = np.random.default_rng(1)
    g1 = rng.normal(0.0, 0.4, size=150)
    g2 = rng.normal(0.0, 1.0, size=150)
    g3 = rng.normal(0.0, 0.5, size=150)
    values = np.concatenate((g1, g2, g3))
    group = np.concatenate((np.ones(g1.size), 2 * np.ones(g2.size), 3 * np.ones(g3.size)))
    data = np.column_stack((values, group))

    out = bartlett_test(values, group, alpha=0.05)
    h_alias, p_alias, t_alias = BartlettTest(data, alpha=0.05)

    assert out["h"] is True
    assert out["p"] < 0.05
    assert h_alias is True
    np.testing.assert_allclose(out["p"], p_alias)
    np.testing.assert_allclose(out["T"], t_alias)


def test_watson_u2_test_detects_circular_difference() -> None:
    rng = np.random.default_rng(2)
    g1 = 0.2 * rng.standard_normal(120)
    g2 = np.pi + 0.2 * rng.standard_normal(120)

    out = watson_u2_test(g1, g2, alpha=0.05)
    h_alias, u2_alias = WatsonU2Test(g1, g2, alpha=0.05)

    assert out["h"] is True
    assert out["U2"] > 0.0
    assert h_alias is True
    np.testing.assert_allclose(out["U2"], u2_alias)


def test_circular_anova_oneway_detects_mean_difference() -> None:
    rng = np.random.default_rng(3)
    g1 = 0.2 * rng.standard_normal(140)
    g2 = 1.2 + 0.2 * rng.standard_normal(140)
    angles = np.concatenate((g1, g2))
    groups = np.concatenate((np.ones(g1.size, dtype=int), 2 * np.ones(g2.size, dtype=int)))

    out = circular_anova(angles, groups, method="ww")
    p_alias, f_alias = CircularANOVA(angles, groups, method="ww")

    assert out["p"] < 0.05
    assert out["F"] > 0.0
    np.testing.assert_allclose(out["p"], p_alias)
    np.testing.assert_allclose(out["F"], f_alias)


def test_multinomial_confidence_intervals_basic_properties() -> None:
    samples = np.array([20, 30, 50], dtype=float)
    out = multinomial_confidence_intervals(samples, alpha=0.05)
    p_alias, b_alias = MultinomialConfidenceIntervals(samples, alpha=0.05)

    assert out["p"].shape == (3,)
    assert out["boundaries"].shape == (2, 3)
    np.testing.assert_allclose(np.sum(out["p"]), 1.0)
    assert np.all(out["boundaries"][0] <= out["p"])
    assert np.all(out["p"] <= out["boundaries"][1])
    np.testing.assert_allclose(out["p"], p_alias)
    np.testing.assert_allclose(out["boundaries"], b_alias)
