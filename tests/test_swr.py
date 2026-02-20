from __future__ import annotations

import numpy as np
import pynapple as nap

from pynacollada import (
    SWRDetectorParams,
    bandpass_filter,
    compute_ripple_feature_stats,
    detect_oscillatory_events,
    detect_ripples_nss,
    compute_ripple_event_stats,
    detect_swr,
    detect_swr_jlong,
    ripple_feature_stats,
    ripple_stats,
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
