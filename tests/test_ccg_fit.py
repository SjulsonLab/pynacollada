from __future__ import annotations

import numpy as np

from pynacollada import FitCCG, fit_ccg


def test_fit_ccg_recovers_reasonable_peak_times() -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(-0.5, 0.5, 1001, dtype=float)

    a = 3.0
    tau = 6.0
    mu = 0.01
    omega = 8.0
    phi = 0.4
    b = 0.8
    model = (a * (np.sin(2.0 * np.pi * omega * t + phi) + 1.0) + b) * np.exp(-tau * np.abs(t - mu))
    ccg = model + 0.05 * rng.standard_normal(t.shape[0])

    out = fit_ccg(t, ccg)
    dt1_alias, dt2_alias, travel_alias, index_alias = FitCCG(t, ccg)

    assert np.isfinite(out["dt1"])
    assert np.isfinite(out["dt2"])
    assert np.isfinite(out["travel"])
    assert out["index"] > 0.0
    assert abs(out["dt2"] - out["dt1"]) < 0.15
    assert -0.2 <= out["travel"] <= 0.2

    np.testing.assert_allclose(out["dt1"], dt1_alias)
    np.testing.assert_allclose(out["dt2"], dt2_alias)
    np.testing.assert_allclose(out["travel"], travel_alias)
    np.testing.assert_allclose(out["index"], index_alias)


def test_fit_ccg_rejects_mismatched_input_lengths() -> None:
    t = np.linspace(-0.1, 0.1, 11)
    ccg = np.ones(10)
    try:
        fit_ccg(t, ccg)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched lengths.")

