from __future__ import annotations

import numpy as np
import pynapple as nap

from pynacollada import (
    FMATRippleDetectorParams,
    FindRipples,
    SWRDetectorParams,
    bandpass_filter,
    compute_ripple_feature_stats,
    compute_ripple_quality_metrics,
    detect_oscillatory_events,
    detect_ripples_fmat,
    detect_ripples_nss,
    compute_ripple_event_stats,
    compute_ripple_spike_coupling,
    detect_swr,
    detect_swr_jlong,
    find_ripples_fmat,
    ripple_event_schema,
    ripple_feature_stats,
    ripple_quality_metrics,
    ripple_spike_coupling,
    ripple_stats,
    standardize_ripple_events,
)


def _make_synthetic_lfp(
    *,
    fs: float = 1250.0,
    duration_s: float = 20.0,
    event_times: tuple[float, ...] = (5.0, 10.0, 15.0),
    n_channels: int = 4,
) -> tuple[nap.TsdFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(0.0, duration_s, 1.0 / fs, dtype=float)
    data = 0.18 * rng.standard_normal((t.size, n_channels))
    data += 0.10 * np.sin(2.0 * np.pi * 8.0 * t)[:, None]

    for t0 in event_times:
        dt = t - t0
        ripple_env = np.exp(-0.5 * (dt / 0.015) ** 2)
        ripple = np.sin(2.0 * np.pi * 150.0 * dt) * ripple_env
        slow_wave = np.exp(-0.5 * (dt / 0.030) ** 2)

        data[:, 0] += 2.2 * slow_wave + 2.8 * ripple
        data[:, -1] += -2.0 * slow_wave + 1.4 * ripple
        if n_channels > 2:
            data[:, 1:-1] += 1.8 * ripple[:, None]

    support = nap.IntervalSet(start=t[0], end=t[-1])
    lfp = nap.TsdFrame(t=t, d=data, columns=np.arange(n_channels), time_support=support)
    return lfp, np.asarray(event_times, dtype=float)


def _default_params() -> SWRDetectorParams:
    return SWRDetectorParams(
        per_thres_swd=20.0,
        per_thres_rip=30.0,
        win_size_ms=120.0,
        ns_chk_s=1.0,
        thres_sd_swd=(0.25, 1.5),
        thres_sd_rip=(0.25, 1.5),
        min_isi_s=0.05,
        min_dur_sw_s=0.01,
        max_dur_sw_s=0.25,
        min_dur_rp_s=0.008,
    )


def test_detect_swr_jlong_finds_synthetic_events() -> None:
    lfp, truth_peaks = _make_synthetic_lfp()
    out = detect_swr_jlong(lfp, params=_default_params(), random_seed=0)

    assert out["timestamps"].ndim == 2
    assert out["timestamps"].shape[1] == 2
    assert out["timestamps"].shape[0] >= 2
    assert isinstance(out["events"], nap.IntervalSet)
    assert out["detectorName"] == "detect_swr_jlong"

    match_count = 0
    for t0 in truth_peaks:
        if np.any((out["timestamps"][:, 0] <= t0) & (out["timestamps"][:, 1] >= t0)):
            match_count += 1
    assert match_count >= 2


def test_detect_swr_training_mode_outputs_surfaces() -> None:
    lfp, truth_peaks = _make_synthetic_lfp()
    labels = nap.IntervalSet(start=truth_peaks - 0.05, end=truth_peaks + 0.05)

    out = detect_swr(
        lfp,
        params=_default_params(),
        training_labels=labels,
        random_seed=0,
    )
    training = out["training"]
    assert training is not None
    assert training["precision_surface"].shape == (21, 21)
    assert training["recall_surface"].shape == (21, 21)
    assert training["f1_surface"].shape == (21, 21)
    assert "best_threshold" in training
    assert "kmeans_accuracy" in training


def test_compute_ripple_event_stats_with_spikes() -> None:
    lfp, _ = _make_synthetic_lfp()
    out = detect_swr_jlong(lfp, params=_default_params(), random_seed=0)

    spike_data = {}
    for unit in range(3):
        unit_times = []
        for start_t, end_t in out["timestamps"]:
            unit_times.extend(np.linspace(start_t, end_t, num=3 + unit))
        spike_data[unit] = nap.Ts(t=np.asarray(unit_times, dtype=float))
    spikes = nap.TsGroup(spike_data, time_support=lfp.time_support)

    stats = compute_ripple_event_stats(lfp, out, spikes=spikes, channel=0)
    assert stats.shape[0] == out["timestamps"].shape[0]
    assert {"duration_s", "peak_amplitude", "rms", "dominant_freq_hz", "total_spike_count"}.issubset(stats.columns)
    assert np.all(stats["duration_s"].values >= 0)
    assert np.all(stats["total_spike_count"].values >= 0)


def test_empty_epoch_and_empty_stats() -> None:
    lfp, _ = _make_synthetic_lfp()
    outside = nap.IntervalSet(start=100.0, end=101.0)
    out = detect_swr_jlong(lfp, epochs=outside, params=_default_params(), random_seed=0)
    assert out["timestamps"].shape[0] == 0

    stats = ripple_stats(lfp, out)
    assert stats.empty


def test_archive_backed_nss_detection_api() -> None:
    lfp, _ = _make_synthetic_lfp(duration_s=10.0)
    lfp_single = lfp[:, 0]
    epoch = lfp.time_support

    ep1, peaks1 = detect_oscillatory_events(
        lfp=lfp_single,
        epoch=epoch,
        freq_band=(100.0, 300.0),
        thres_band=(0.5, 15.0),
        duration_band=(0.01, 0.25),
        min_inter_duration=0.01,
    )
    ep2, peaks2 = detect_ripples_nss(
        lfp=lfp_single,
        epoch=epoch,
        freq_band=(100.0, 300.0),
        thres_band=(0.5, 15.0),
        duration_band=(0.01, 0.25),
        min_inter_duration=0.01,
    )

    np.testing.assert_allclose(ep1.as_units("s").values, ep2.as_units("s").values)
    np.testing.assert_allclose(peaks1.as_units("s").index.values, peaks2.as_units("s").index.values)


def test_find_ripples_fmat_detects_synthetic_events_and_aliases() -> None:
    lfp, truth_peaks = _make_synthetic_lfp(duration_s=15.0, n_channels=3)
    lfp_single = lfp[:, 0]
    params = FMATRippleDetectorParams(
        thresholds=(1.0, 2.2),
        durations_ms=(20.0, 150.0),
        min_duration_ms=8.0,
        passband=(100.0, 250.0),
    )

    out = find_ripples_fmat(lfp_single, epoch=lfp.time_support, params=params)
    out_alias1 = detect_ripples_fmat(lfp_single, epoch=lfp.time_support, params=params)
    out_alias2 = FindRipples(lfp_single, epoch=lfp.time_support, params=params)

    assert out["timestamps"].ndim == 2
    assert out["timestamps"].shape[1] == 2
    assert out["timestamps"].shape[0] >= 2
    assert out["detectorName"] == "find_ripples_fmat"
    assert out["noise"]["times"].shape[1] == 2

    match_count = 0
    for t0 in truth_peaks:
        if np.any((out["timestamps"][:, 0] <= t0) & (out["timestamps"][:, 1] >= t0)):
            match_count += 1
    assert match_count >= 2

    np.testing.assert_allclose(out["timestamps"], out_alias1["timestamps"])
    np.testing.assert_allclose(out["timestamps"], out_alias2["timestamps"])


def test_find_ripples_fmat_noise_channel_rejection() -> None:
    fs = 1250.0
    t = np.arange(0.0, 16.0, 1.0 / fs, dtype=float)
    rng = np.random.default_rng(0)

    ripple_times = (5.0, 12.0)
    lfp_signal = 0.12 * rng.standard_normal(t.size)
    noise_signal = 0.12 * rng.standard_normal(t.size)

    for t0 in ripple_times:
        dt = t - t0
        ripple_env = np.exp(-0.5 * (dt / 0.015) ** 2)
        ripple = np.sin(2.0 * np.pi * 150.0 * dt) * ripple_env
        lfp_signal += 3.0 * ripple

    # Inject strong ripple-like artifact only on the noise channel at 12 s.
    dt_bad = t - 12.0
    artifact = np.sin(2.0 * np.pi * 150.0 * dt_bad) * np.exp(-0.5 * (dt_bad / 0.012) ** 2)
    noise_signal += 6.0 * artifact

    support = nap.IntervalSet(start=t[0], end=t[-1])
    lfp = nap.Tsd(t=t, d=lfp_signal, time_support=support)
    noise_lfp = nap.Tsd(t=t, d=noise_signal, time_support=support)

    params = FMATRippleDetectorParams(
        thresholds=(0.8, 2.0),
        durations_ms=(20.0, 180.0),
        min_duration_ms=8.0,
        passband=(100.0, 250.0),
    )
    out_no_noise = find_ripples_fmat(lfp, epoch=support, params=params)
    out_with_noise = find_ripples_fmat(lfp, epoch=support, params=params, noise_lfp=noise_lfp)

    assert out_with_noise["timestamps"].shape[0] <= out_no_noise["timestamps"].shape[0]
    assert out_with_noise["noise"]["times"].shape[0] >= 1

    has_12s_without_noise = np.any((out_no_noise["timestamps"][:, 0] <= 12.0) & (out_no_noise["timestamps"][:, 1] >= 12.0))
    has_12s_with_noise = np.any((out_with_noise["timestamps"][:, 0] <= 12.0) & (out_with_noise["timestamps"][:, 1] >= 12.0))
    assert has_12s_without_noise
    assert not has_12s_with_noise


def test_compute_ripple_feature_stats_maps_data_stats() -> None:
    lfp, _ = _make_synthetic_lfp()
    out = detect_swr_jlong(lfp, params=_default_params(), random_seed=0)
    filtered = bandpass_filter(lfp[:, 0], 100.0, 250.0, lfp.rate)

    maps, data, stats = compute_ripple_feature_stats(filtered, out, durations=(-0.05, 0.05))
    maps_alias, data_alias, stats_alias = ripple_feature_stats(filtered, out, durations=(-0.05, 0.05))

    assert maps["ripples"].shape[0] == out["timestamps"].shape[0]
    assert maps["frequency"].shape == maps["ripples"].shape
    assert maps["phase"].shape == maps["ripples"].shape
    assert maps["amplitude"].shape == maps["ripples"].shape
    assert maps["t"].shape[0] == maps["ripples"].shape[1]
    assert data["peakFrequency"].shape[0] == out["timestamps"].shape[0]
    assert data["peakAmplitude"].shape[0] == out["timestamps"].shape[0]
    assert data["duration"].shape[0] == out["timestamps"].shape[0]
    assert "acg" in stats and "data" in stats["acg"] and "t" in stats["acg"]
    assert "amplitudeFrequency" in stats
    assert "durationFrequency" in stats
    assert "durationAmplitude" in stats

    np.testing.assert_allclose(maps["ripples"], maps_alias["ripples"])
    np.testing.assert_allclose(data["duration"], data_alias["duration"])
    np.testing.assert_allclose(stats["acg"]["data"], stats_alias["acg"]["data"])


def test_compute_ripple_quality_metrics_outputs_expected_columns() -> None:
    lfp, _ = _make_synthetic_lfp()
    out = detect_swr_jlong(lfp, params=_default_params(), random_seed=0)

    quality = compute_ripple_quality_metrics(lfp, out, channel=0)
    quality_alias = ripple_quality_metrics(lfp, out, channel=0)

    assert quality.shape[0] == out["timestamps"].shape[0]
    expected_columns = {
        "ripple_to_sharpwave_ratio",
        "ripple_to_broadband_ratio",
        "waveform_asymmetry",
        "cycle_count",
        "cycle_frequency_hz",
        "spectral_entropy",
        "broadband_peak_z",
        "phase_at_peak",
    }
    assert expected_columns.issubset(quality.columns)
    assert np.isfinite(quality["ripple_to_broadband_ratio"]).any()
    assert np.isfinite(quality["spectral_entropy"]).any()
    np.testing.assert_allclose(quality["duration_s"].values, quality_alias["duration_s"].values)


def test_standardize_ripple_events_schema() -> None:
    lfp, _ = _make_synthetic_lfp()
    out = detect_swr_jlong(lfp, params=_default_params(), random_seed=0)

    schema = standardize_ripple_events(out)
    schema_alias = ripple_event_schema(out)
    schema_arr = standardize_ripple_events(out["timestamps"])

    assert schema["table"].shape[0] == out["timestamps"].shape[0]
    assert isinstance(schema["events"], nap.IntervalSet)
    assert {"start", "end", "peak_time", "duration_s", "peak_normed_power"}.issubset(schema["table"].columns)
    assert schema["nwb"]["start_time"].shape[0] == out["timestamps"].shape[0]
    assert schema_arr["table"].shape[0] == out["timestamps"].shape[0]
    np.testing.assert_allclose(schema["table"]["duration_s"].values, schema_alias["table"]["duration_s"].values)


def test_compute_ripple_spike_coupling() -> None:
    lfp, truth_peaks = _make_synthetic_lfp()
    out = detect_swr_jlong(lfp, params=_default_params(), random_seed=0)
    support = lfp.time_support
    support_start = float(np.asarray(support.start).reshape(-1)[0])
    support_end = float(np.asarray(support.end).reshape(-1)[0])
    rng = np.random.default_rng(0)

    # Unit 0: concentrated near ripple peaks
    u0 = []
    for t0 in truth_peaks:
        u0.extend((t0 + 0.01 * rng.standard_normal(25)).tolist())
    u0 = np.asarray(u0, dtype=float)
    u0 = u0[(u0 >= support_start) & (u0 <= support_end)]

    # Unit 1: near-uniform baseline spikes
    u1 = np.sort(rng.uniform(support_start, support_end, size=120))
    spikes = nap.TsGroup(
        {0: nap.Ts(t=np.sort(u0)), 1: nap.Ts(t=u1)},
        time_support=support,
    )

    coupling = compute_ripple_spike_coupling(spikes, out, window=(-0.08, 0.08), bin_size=0.01)
    coupling_alias = ripple_spike_coupling(spikes, out, window=(-0.08, 0.08), bin_size=0.01)

    summary = coupling["summary"]
    assert summary.shape[0] == 2
    assert "participation_probability" in summary.columns
    assert "rate_modulation_index" in summary.columns
    assert coupling["peri_event_rate"].shape[1] == 2
    assert coupling["peri_event_rate"].shape[0] == coupling["bins"].shape[0]

    # Ripple-locked unit should be more coupled than baseline unit.
    assert summary.loc[0, "participation_probability"] > summary.loc[1, "participation_probability"]

    np.testing.assert_allclose(coupling["bins"], coupling_alias["bins"])
