from __future__ import annotations

import numpy as np

from pynacollada import (
    AngularVelocity,
    CCGParameters,
    CV,
    CoherenceBands,
    Distance,
    FilterLFP,
    Frequency,
    LinearVelocity,
    MovementPeriods,
    Phase,
    QuietPeriods,
    SpectrogramBands,
    angular_velocity,
    ccg_parameters,
    coherence_bands,
    cv,
    distance,
    filter_lfp,
    frequency,
    linear_velocity,
    movement_periods,
    phase,
    quiet_periods,
    spectrogram_bands,
)


def test_linear_and_angular_velocity_outputs() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 1000)
    x = np.cos(t)
    y = np.sin(t)
    pos = np.column_stack((t, x, y))

    lv = linear_velocity(pos)
    av = angular_velocity(pos)

    assert lv.shape == (pos.shape[0], 2)
    assert av.shape == (pos.shape[0], 2)
    assert np.nanmedian(lv[:, 1]) > 0.95
    assert np.nanmedian(lv[:, 1]) < 1.05
    assert np.nanmedian(av[:, 1]) > 0.95
    assert np.nanmedian(av[:, 1]) < 1.05

    np.testing.assert_allclose(lv, LinearVelocity(pos))
    np.testing.assert_allclose(av, AngularVelocity(pos))


def test_distance_linear_and_circular_modes() -> None:
    t = np.linspace(0.0, 1.0, 5)
    x = np.array([0.1, 0.2, 0.9, 0.95, 0.05])
    d_linear = distance(np.column_stack((t, x)), reference=0.0)
    d_circular = distance(np.column_stack((t, x)), reference=0.0, position_type="c")

    assert d_linear.shape == (5, 2)
    assert d_circular.shape == (5, 2)
    assert np.max(d_circular[:, 1]) < np.max(d_linear[:, 1])

    np.testing.assert_allclose(d_linear, Distance(np.column_stack((t, x)), 0.0, type="linear"))
    np.testing.assert_allclose(d_circular, Distance(np.column_stack((t, x)), 0.0, type="c"))


def test_quiet_and_movement_period_detection() -> None:
    t = np.arange(0.0, 10.0, 0.1)
    v = np.zeros_like(t)
    v[(t >= 2.0) & (t <= 4.5)] = 15.0
    v[(t >= 6.0) & (t <= 7.2)] = 12.0
    vv = np.column_stack((t, v))

    move_periods, move_state = movement_periods(vv, velocity=5.0, duration=0.5)
    quiet_periods_out, quiet_state = quiet_periods(vv, velocity=5.0, duration=0.5)

    assert move_periods.shape[0] == 2
    assert quiet_periods_out.shape[0] >= 2
    assert np.sum(move_state[:, 1]) > 0
    assert np.sum(quiet_state[:, 1]) > 0

    p2, s2 = MovementPeriods(vv, velocity=5.0, duration=0.5)
    p3, s3 = QuietPeriods(vv, velocity=5.0, duration=0.5)
    np.testing.assert_allclose(move_periods, p2)
    np.testing.assert_allclose(move_state, s2)
    np.testing.assert_allclose(quiet_periods_out, p3)
    np.testing.assert_allclose(quiet_state, s3)


def test_phase_and_interpolated_phase_shapes() -> None:
    t = np.arange(0.0, 2.0, 0.001)
    signal = np.sin(2.0 * np.pi * 8.0 * t)
    samples = np.column_stack((t, signal))

    p, a, u = phase(samples)
    tq = np.arange(0.05, 1.95, 0.002)
    p_i, a_i, u_i = phase(samples, times=tq)

    assert p.shape == samples.shape
    assert a.shape == samples.shape
    assert u.shape == samples.shape
    assert p_i.shape == (tq.shape[0], 2)
    assert a_i.shape == (tq.shape[0], 2)
    assert u_i.shape == (tq.shape[0], 2)
    assert np.nanmin(p[:, 1]) >= 0.0
    assert np.nanmax(p[:, 1]) <= 2.0 * np.pi

    p2, a2, u2 = Phase(samples, times=tq)
    np.testing.assert_allclose(p_i, p2)
    np.testing.assert_allclose(a_i, a2)
    np.testing.assert_allclose(u_i, u2)


def test_frequency_methods_and_alias() -> None:
    ts = np.arange(0.0, 5.0, 0.05)
    f_fixed = frequency(ts, method="fixed", bin_size=0.05, smooth=2)
    f_adapt = frequency(ts, method="adaptive", bin_size=0.05, smooth=2)
    f_inv = frequency(ts, method="inverse")

    assert f_fixed.shape[1] == 2
    assert f_adapt.shape == f_fixed.shape
    assert f_inv.shape == (ts.shape[0], 2)
    assert np.nanmean(f_fixed[:, 1]) > 15.0
    assert np.nanmean(f_fixed[:, 1]) < 25.0

    np.testing.assert_allclose(f_fixed, Frequency(ts, "method", "fixed", "binSize", 0.05, "smooth", 2))


def test_cv_variants_and_alias() -> None:
    ts = np.cumsum(np.r_[0.1, np.repeat(0.1, 400)])

    coeff_cv, local_cv = cv(ts, measure="cv")
    coeff_cv2, local_cv2 = cv(ts, measure="cv2")
    coeff_cvo, local_cvo = cv(ts, measure="cvo", method="fixed")

    assert coeff_cv < 1e-6
    assert coeff_cv2 < 1e-6
    assert np.isfinite(coeff_cvo)
    assert local_cv.size == 0
    assert local_cvo.size == 0
    assert local_cv2.size > 0

    coeff_cv_alias, _ = CV(ts, "measure", "cv")
    assert abs(coeff_cv_alias - coeff_cv) < 1e-10


def test_spectrogram_and_coherence_band_helpers() -> None:
    freqs = np.linspace(0.0, 250.0, 501)
    t_bins = 80
    spec = np.zeros((freqs.shape[0], t_bins), dtype=float)
    theta_idx = (freqs >= 7.0) & (freqs <= 10.0)
    delta_idx = (freqs >= 0.0) & (freqs <= 4.0)
    spec[theta_idx, :] = 5.0
    spec[delta_idx, :] = 1.0

    bands = spectrogram_bands(spec, freqs)
    assert "ratios" in bands
    assert bands["theta"].shape[0] == t_bins
    assert np.nanmean(bands["ratios"]["hippocampus"]) > 4.0

    bands_alias = SpectrogramBands(spec, freqs)
    np.testing.assert_allclose(bands["theta"], bands_alias["theta"])

    coh = np.tile(np.linspace(0.0, 1.0, freqs.shape[0]).reshape(-1, 1), (1, t_bins))
    cb = coherence_bands(coh, freqs)
    assert cb["theta"].shape[0] == t_bins
    cb_alias = CoherenceBands(coh, freqs)
    np.testing.assert_allclose(cb["theta"], cb_alias["theta"])


def test_ccg_parameter_reformatting() -> None:
    s1 = np.array([0.1, 0.2, 0.3])
    s2 = np.array([[0.15, 1], [0.4, 2]])

    t, cid, grp = ccg_parameters(s1, 1, s2, np.array([2, 2]))
    assert t.shape == cid.shape == grp.shape
    assert np.max(cid) >= 3
    assert np.all(grp[:3] == 1)

    t2, cid2, grp2 = CCGParameters(s1, 1, s2, np.array([2, 2]))
    np.testing.assert_allclose(t, t2)
    np.testing.assert_array_equal(cid, cid2)
    np.testing.assert_array_equal(grp, grp2)


def test_filter_lfp_band_alias_and_shape() -> None:
    fs = 1250.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    x = np.sin(2.0 * np.pi * 8.0 * t) + 0.2 * np.sin(2.0 * np.pi * 80.0 * t)
    lfp = np.column_stack((t, x))

    filt = filter_lfp(lfp, passband="theta")
    assert filt.shape == lfp.shape
    assert np.std(filt[:, 1]) > 0.1

    filt_alias = FilterLFP(lfp, "passband", "theta")
    np.testing.assert_allclose(filt, filt_alias)
