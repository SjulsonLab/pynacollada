from __future__ import annotations

import numpy as np

from pynacollada import (
    FieldShift,
    TestRemapping,
    TestSkewness,
    field_shift,
    test_remapping,
    test_skewness,
)


def _gaussian_fields(centers: np.ndarray, n_bins: int, sigma: float = 3.0) -> np.ndarray:
    x = np.arange(n_bins, dtype=float)[None, :]
    c = centers[:, None]
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


def test_field_shift_recovers_known_circular_shifts() -> None:
    n_cells = 8
    n_bins = 60
    centers = np.linspace(8, 52, n_cells)
    control = _gaussian_fields(centers, n_bins=n_bins, sigma=2.5)
    shifts = np.array([2, -3, 0, 1, -1, 4, -2, 3], dtype=int)
    test = np.vstack([np.roll(control[i], shifts[i]) for i in range(n_cells)])

    out = field_shift(control, test, field_type="circular")
    rel_alias, abs_alias, xc_alias = FieldShift(control, test, field_type="circular")

    np.testing.assert_allclose(out["absolute_shift"], shifts / n_bins, atol=1.0 / n_bins)
    np.testing.assert_allclose(out["absolute_shift"], abs_alias)
    np.testing.assert_allclose(out["relative_shift"], rel_alias)
    np.testing.assert_allclose(out["xc"], xc_alias)


def test_test_remapping_detects_nonrandom_alignment() -> None:
    rng = np.random.default_rng(0)
    n_cells = 24
    n_bins = 80
    centers = np.linspace(10, 70, n_cells)
    control = _gaussian_fields(centers, n_bins=n_bins, sigma=3.0)
    repeat = control + 0.02 * rng.standard_normal(control.shape)
    test = control + 0.02 * rng.standard_normal(control.shape)

    out = test_remapping(control, repeat, test, field_type="linear", alpha=0.05, iterations=250, random_seed=0)
    h_alias, p_alias, cr_alias, ca_alias = TestRemapping(
        control,
        repeat,
        test,
        field_type="linear",
        alpha=0.05,
        iterations=250,
        random_seed=0,
    )

    assert out["h"] is True
    assert out["p"] < 0.05
    assert out["cr"].shape == (2,)
    assert out["ca"].shape == (2,)
    assert h_alias is True
    np.testing.assert_allclose(out["p"], p_alias)
    np.testing.assert_allclose(out["cr"], cr_alias)
    np.testing.assert_allclose(out["ca"], ca_alias)


def test_test_skewness_detects_distribution_change() -> None:
    rng = np.random.default_rng(1)
    n_cells = 20
    n_bins = 50
    x = np.linspace(0.0, 1.0, n_bins, dtype=float)
    base = np.exp(-4.0 * x)[None, :]
    control = base + 0.03 * rng.random((n_cells, n_bins))
    repeat = base + 0.03 * rng.random((n_cells, n_bins))
    test = control[:, ::-1] + 0.01 * rng.random((n_cells, n_bins))

    out = test_skewness(control, repeat, test, alpha=0.05, iterations=120, random_seed=0)
    h_alias, p_alias, stats_alias = TestSkewness(control, repeat, test, alpha=0.05, iterations=120, random_seed=0)

    assert out["h"] is True
    assert out["p"] < 0.05
    assert "control" in out["stats"] and "repeat" in out["stats"] and "test" in out["stats"]
    assert h_alias is True
    np.testing.assert_allclose(out["p"], p_alias)
    assert set(stats_alias.keys()) == {"control", "repeat", "test"}

