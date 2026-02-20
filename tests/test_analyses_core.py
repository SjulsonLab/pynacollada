from __future__ import annotations

import numpy as np
import pynapple as nap

from pynacollada import (
    AngularVelocity,
    CCGParameters,
    CV,
    CompareDistributions,
    CoherenceBands,
    DefineZone,
    Distance,
    FilterLFP,
    Frequency,
    IsInZone,
    LinearVelocity,
    MovementPeriods,
    Phase,
    QuietPeriods,
    SpectrogramBands,
    ThresholdSpikes,
    angular_velocity,
    ccg_parameters,
    compare_distributions,
    coherence_bands,
    cv,
    define_zone,
    distance,
    filter_lfp,
    frequency,
    is_in_zone,
    linear_velocity,
    movement_periods,
    phase,
    quiet_periods,
    spectrogram_bands,
    threshold_spikes,
)


def test_linear_and_angular_velocity_outputs() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 1000)
    x = np.cos(t)
    y = np.sin(t)
    pos = np.column_stack((t, x, y))

    lv = linear_velocity(pos)
    av = angular_velocity(pos)

    assert isinstance(lv, nap.Tsd)
    assert isinstance(av, nap.Tsd)
    assert lv.shape[0] == pos.shape[0]
    assert av.shape[0] == pos.shape[0]
    assert np.nanmedian(lv.values) > 0.95
    assert np.nanmedian(lv.values) < 1.05
    assert np.nanmedian(av.values) > 0.95
    assert np.nanmedian(av.values) < 1.05

    lv_legacy = LinearVelocity(pos)
    av_legacy = AngularVelocity(pos)
    np.testing.assert_allclose(np.column_stack((lv.index.values, lv.values)), lv_legacy)
    np.testing.assert_allclose(np.column_stack((av.index.values, av.values)), av_legacy)


def test_distance_linear_and_circular_modes() -> None:
    t = np.linspace(0.0, 1.0, 5)
    x = np.array([0.1, 0.2, 0.9, 0.95, 0.05])
    d_linear = distance(np.column_stack((t, x)), reference=0.0)
    d_circular = distance(np.column_stack((t, x)), reference=0.0, position_type="c")

    assert isinstance(d_linear, nap.Tsd)
    assert isinstance(d_circular, nap.Tsd)
    assert d_linear.shape[0] == 5
    assert d_circular.shape[0] == 5
    assert np.max(d_circular.values) < np.max(d_linear.values)

    np.testing.assert_allclose(
        np.column_stack((d_linear.index.values, d_linear.values)),
        Distance(np.column_stack((t, x)), 0.0, type="linear"),
    )
    np.testing.assert_allclose(
        np.column_stack((d_circular.index.values, d_circular.values)),
        Distance(np.column_stack((t, x)), 0.0, type="c"),
    )


def test_quiet_and_movement_period_detection() -> None:
    t = np.arange(0.0, 10.0, 0.1)
    v = np.zeros_like(t)
    v[(t >= 2.0) & (t <= 4.5)] = 15.0
    v[(t >= 6.0) & (t <= 7.2)] = 12.0
    vv = np.column_stack((t, v))

    move_periods, move_state = movement_periods(vv, velocity=5.0, duration=0.5)
    quiet_periods_out, quiet_state = quiet_periods(vv, velocity=5.0, duration=0.5)

    assert isinstance(move_periods, nap.IntervalSet)
    assert isinstance(quiet_periods_out, nap.IntervalSet)
    assert isinstance(move_state, nap.Tsd)
    assert isinstance(quiet_state, nap.Tsd)
    assert move_periods.shape[0] == 2
    assert quiet_periods_out.shape[0] >= 2
    assert np.sum(move_state.values) > 0
    assert np.sum(quiet_state.values) > 0

    p2, s2 = MovementPeriods(vv, velocity=5.0, duration=0.5)
    p3, s3 = QuietPeriods(vv, velocity=5.0, duration=0.5)
    np.testing.assert_allclose(np.asarray(move_periods.values, dtype=float), p2)
    np.testing.assert_allclose(np.column_stack((move_state.index.values, move_state.values)), s2)
    np.testing.assert_allclose(np.asarray(quiet_periods_out.values, dtype=float), p3)
    np.testing.assert_allclose(np.column_stack((quiet_state.index.values, quiet_state.values)), s3)


def test_phase_and_interpolated_phase_shapes() -> None:
    t = np.arange(0.0, 2.0, 0.001)
    signal = np.sin(2.0 * np.pi * 8.0 * t)
    samples = np.column_stack((t, signal))

    p, a, u = phase(samples)
    tq = np.arange(0.05, 1.95, 0.002)
    p_i, a_i, u_i = phase(samples, times=tq)

    assert isinstance(p, nap.Tsd)
    assert isinstance(a, nap.Tsd)
    assert isinstance(u, nap.Tsd)
    assert isinstance(p_i, nap.Tsd)
    assert isinstance(a_i, nap.Tsd)
    assert isinstance(u_i, nap.Tsd)
    assert p.shape[0] == samples.shape[0]
    assert p_i.shape[0] == tq.shape[0]
    assert np.nanmin(p.values) >= 0.0
    assert np.nanmax(p.values) <= 2.0 * np.pi

    p2, a2, u2 = Phase(samples, times=tq)
    np.testing.assert_allclose(np.column_stack((p_i.index.values, p_i.values)), p2)
    np.testing.assert_allclose(np.column_stack((a_i.index.values, a_i.values)), a2)
    np.testing.assert_allclose(np.column_stack((u_i.index.values, u_i.values)), u2)


def test_frequency_methods_and_alias() -> None:
    ts = np.arange(0.0, 5.0, 0.05)
    f_fixed = frequency(ts, method="fixed", bin_size=0.05, smooth=2)
    f_adapt = frequency(ts, method="adaptive", bin_size=0.05, smooth=2)
    f_inv = frequency(ts, method="inverse")

    assert isinstance(f_fixed, nap.Tsd)
    assert isinstance(f_adapt, nap.Tsd)
    assert isinstance(f_inv, nap.Tsd)
    assert f_adapt.shape == f_fixed.shape
    assert f_inv.shape[0] == ts.shape[0]
    assert np.nanmean(f_fixed.values) > 15.0
    assert np.nanmean(f_fixed.values) < 25.0

    np.testing.assert_allclose(
        np.column_stack((f_fixed.index.values, f_fixed.values)),
        Frequency(ts, "method", "fixed", "binSize", 0.05, "smooth", 2),
        atol=1e-12,
    )


def test_cv_variants_and_alias() -> None:
    ts = np.cumsum(np.r_[0.1, np.repeat(0.1, 400)])

    coeff_cv, local_cv = cv(ts, measure="cv")
    coeff_cv2, local_cv2 = cv(ts, measure="cv2")
    coeff_cvo, local_cvo = cv(ts, measure="cvo", method="fixed")

    assert coeff_cv < 1e-6
    assert coeff_cv2 < 1e-6
    assert np.isfinite(coeff_cvo)
    assert isinstance(local_cv, nap.Tsd)
    assert isinstance(local_cvo, nap.Tsd)
    assert isinstance(local_cv2, nap.Tsd)
    assert local_cv.shape[0] == 0
    assert local_cvo.shape[0] == 0
    assert local_cv2.shape[0] > 0

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

    t = np.linspace(0.0, 7.9, t_bins)
    bands = spectrogram_bands(spec, freqs, times=t)
    assert "ratios" in bands
    assert isinstance(bands["theta"], nap.Tsd)
    assert bands["theta"].shape[0] == t_bins
    assert np.nanmean(bands["ratios"]["hippocampus"].values) > 4.0

    bands_alias = SpectrogramBands(spec, freqs, "times", t)
    np.testing.assert_allclose(
        np.column_stack((bands["theta"].index.values, bands["theta"].values)),
        np.column_stack((t, bands_alias["theta"])),
    )

    coh = np.tile(np.linspace(0.0, 1.0, freqs.shape[0]).reshape(-1, 1), (1, t_bins))
    cb = coherence_bands(coh, freqs, times=t)
    assert isinstance(cb["theta"], nap.Tsd)
    assert cb["theta"].shape[0] == t_bins
    cb_alias = CoherenceBands(coh, freqs, "times", t)
    np.testing.assert_allclose(np.column_stack((t, cb["theta"].values)), np.column_stack((t, cb_alias["theta"])))


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
    assert isinstance(filt, nap.Tsd)
    assert filt.shape[0] == lfp.shape[0]
    assert np.std(filt.values) > 0.1

    filt_alias = FilterLFP(lfp, "passband", "theta")
    np.testing.assert_allclose(np.column_stack((filt.index.values, filt.values)), filt_alias)


def test_define_zone_and_is_in_zone_intervalset_output() -> None:
    zone = define_zone((100, 100), "rectangle", (20, 30, 20, 20))
    assert zone.dtype == bool
    assert zone.shape == (100, 100)

    t = np.linspace(0.0, 1.0, 1000)
    x = np.linspace(0.0, 1.0, 1000)
    y = np.full_like(x, 0.4)
    pos = np.column_stack((t, x, y))

    intervals = is_in_zone(pos, zone)
    assert isinstance(intervals, nap.IntervalSet)
    assert intervals.shape[0] >= 1

    intervals2, mask_tsd = IsInZone(pos, zone, return_mask=True)
    assert isinstance(intervals2, nap.IntervalSet)
    assert isinstance(mask_tsd, nap.Tsd)
    assert np.sum(mask_tsd.values) > 0

    zone_alias = DefineZone((100, 100), "rectangle", (20, 30, 20, 20))
    np.testing.assert_array_equal(zone, zone_alias)


def test_compare_distributions_and_threshold_spikes() -> None:
    rng = np.random.default_rng(0)
    g1 = rng.normal(loc=1.0, scale=0.2, size=(80, 20))
    g2 = rng.normal(loc=0.0, scale=0.2, size=(80, 20))

    h, stats = compare_distributions(g1, g2, n_shuffles=600, alpha=0.05, random_seed=1)
    assert h is True
    assert "observed" in stats
    assert stats["observed"].shape[0] == g1.shape[1]

    h2, _ = CompareDistributions(g1, g2, "nShuffles", 600, "alpha", 0.05, "randomSeed", 1)
    assert h2 is True

    amps = np.array(
        [
            [0.1, 1, 2, 50],
            [0.2, 1, 2, 20],
            [0.3, 1, 2, 10],
            [0.4, 1, 3, 80],
            [0.5, 1, 3, 20],
        ],
        dtype=float,
    )
    kept = threshold_spikes(amps, factor=1.5)
    assert kept.shape[1] == 3
    assert kept.shape[0] < amps.shape[0]

    kept2 = ThresholdSpikes(amps, 1.5)
    np.testing.assert_allclose(kept, kept2)
