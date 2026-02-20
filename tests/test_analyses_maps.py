from __future__ import annotations

import numpy as np

from pynacollada import (
    FindFieldHelper,
    FiringCurve,
    FiringMap,
    Map,
    MapStats,
    NormalizeFields,
    bz_Map,
    compute_map,
    firing_curve,
    firing_map,
    find_field_helper,
    map_stats,
    normalize_fields,
)


def test_compute_map_point_process_1d_and_wrappers() -> None:
    t = np.linspace(0.0, 10.0, 2001)
    x = (np.sin(2.0 * np.pi * 0.1 * t) + 1.0) / 2.0
    samples = np.column_stack((t, x))

    spike_idx = np.where((x > 0.72) & (x < 0.78))[0][::5]
    spikes = t[spike_idx]

    m = compute_map(samples, spikes, n_bins=60, smooth=1.5, sample_type="ll")
    assert m["z"].shape == (60,)
    assert m["count"].shape == (60,)
    assert np.max(m["z"]) > np.mean(m["z"]) * 2.0

    m2 = Map(samples, spikes, "nBins", 60, "smooth", 1.5, "type", "ll")
    assert m2["z"].shape == (60,)

    m3 = bz_Map(samples, spikes, "nBins", 60, "smooth", 1.5, "type", "ll")
    assert m3["z"].shape == (60,)


def test_compute_map_and_stats_2d() -> None:
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 20.0, 4000)
    x = np.clip(np.cumsum(rng.normal(0.0, 0.01, size=t.shape[0])) + 0.5, 0.0, 1.0)
    y = np.clip(np.cumsum(rng.normal(0.0, 0.01, size=t.shape[0])) + 0.5, 0.0, 1.0)
    pos = np.column_stack((t, x, y))

    in_field = (x > 0.55) & (y > 0.45)
    spikes = t[in_field][::3]

    fmap, stats = firing_map(
        pos,
        spikes,
        n_bins=(40, 35),
        smooth=(1.2, 1.0),
        min_size=5,
        min_peak=0.01,
        return_stats=True,
    )
    assert fmap["rate"].shape == (35, 40)
    assert np.nanmax(fmap["rate"]) > 0
    assert "specificity" in stats
    assert stats["peak"].size >= 1

    fmap2, stats2 = FiringMap(
        pos,
        spikes,
        "nBins",
        [40, 35],
        "smooth",
        [1.2, 1.0],
        "minSize",
        5,
        "minPeak",
        0.01,
    )
    assert fmap2["rate"].shape == (35, 40)
    assert stats2["peak"].size >= 1


def test_firing_curve_and_map_stats() -> None:
    t = np.linspace(0.0, 8.0, 1600)
    angle = ((2.0 * np.pi * 0.2 * t) % (2.0 * np.pi)) / (2.0 * np.pi)
    samples = np.column_stack((t, angle))
    spikes = t[(angle > 0.22) & (angle < 0.30)][::3]

    curve, stats = firing_curve(
        samples,
        spikes,
        n_bins=50,
        smooth=2.0,
        curve_type="circular",
        min_size=2,
        min_peak=0.01,
        return_stats=True,
    )
    assert curve["rate"].shape == (50,)
    assert stats["peak"].size >= 1
    assert np.isfinite(stats["specificity"])

    curve2, stats2 = FiringCurve(
        samples,
        spikes,
        "nBins",
        50,
        "smooth",
        2.0,
        "type",
        "circular",
        "minSize",
        2,
        "minPeak",
        0.01,
    )
    assert curve2["rate"].shape == (50,)
    assert stats2["peak"].size >= 1

    ms = map_stats({"z": curve["rate"], "count": curve["count"], "time": curve["time"], "x": curve["x"]}, sample_type="c")
    ms2 = MapStats({"z": curve["rate"], "count": curve["count"], "time": curve["time"], "x": curve["x"]}, "type", "c")
    assert np.isfinite(ms["specificity"])
    assert np.isfinite(ms2["specificity"])


def test_normalize_fields_and_find_field_helper() -> None:
    fields = np.array(
        [
            [0, 0, 1, 3, 2, 0, 0],
            [0, 0, 0, 2, 4, 1, 0],
        ],
        dtype=float,
    )
    n = normalize_fields(fields)
    assert n.shape == fields.shape
    assert np.allclose(np.max(n, axis=1), 1.0)

    n2 = NormalizeFields(fields, "rate", "on")
    np.testing.assert_allclose(n, n2)

    z = np.zeros((8, 10), dtype=float)
    z[2:6, 4:8] = 2.0
    z[3:5, 5:7] = 5.0
    f = find_field_helper(z, x=5, y=3, threshold=1.0, circ_x=False, circ_y=False)
    assert f.shape == z.shape
    assert np.sum(f) > 0

    # MATLAB wrapper is 1-based x/y
    f2 = FindFieldHelper(z, 6, 4, 1.0, False, False)
    np.testing.assert_array_equal(f, f2)
