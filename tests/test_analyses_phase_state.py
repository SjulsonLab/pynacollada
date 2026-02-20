from __future__ import annotations

import numpy as np
import pynapple as nap

from pynacollada import (
    BrainStates,
    PhaseCurve,
    PhaseDistribution,
    PhaseMap,
    RippleStats,
    brain_states,
    phase_curve,
    phase_distribution,
    phase_map,
)


def test_phase_distribution_groups() -> None:
    rng = np.random.default_rng(2)
    p1 = rng.vonmises(mu=0.5, kappa=5.0, size=300)
    p2 = rng.vonmises(mu=2.5, kappa=4.0, size=300)
    phases = np.r_[p1, p2]
    groups = np.r_[np.ones(300, dtype=int), np.full(300, 2, dtype=int)]

    dist, angles, stats = phase_distribution(phases, n_bins=60, smooth=1.0, groups=groups)
    assert dist.shape == (60, 2)
    assert angles.shape == (60,)
    assert stats["m"].shape == (2,)

    dist2, angles2, _ = PhaseDistribution(phases, "nBins", 60, "smooth", 1.0, "groups", groups)
    np.testing.assert_allclose(dist, dist2)
    np.testing.assert_allclose(angles, angles2)


def test_phase_curve_and_phase_map() -> None:
    t = np.linspace(0.0, 10.0, 2000)
    x = (np.sin(2.0 * np.pi * 0.1 * t) + 1.0) / 2.0
    y = (np.cos(2.0 * np.pi * 0.08 * t) + 1.0) / 2.0
    phase = (2.0 * np.pi * 8.0 * t) % (2.0 * np.pi)

    curve = phase_curve(np.column_stack((t, x)), phase, n_bins=50, smooth=1.5, curve_type="linear")
    assert curve["phase"].shape == (50,)

    curve2 = PhaseCurve(np.column_stack((t, x)), phase, "nBins", 50, "smooth", 1.5, "type", "linear")
    assert curve2["phase"].shape == (50,)

    pmap = phase_map(np.column_stack((t, x, y)), phase, n_bins=(40, 30), smooth=(1.0, 1.0))
    assert pmap["phase"].shape == (30, 40)

    pmap2 = PhaseMap(np.column_stack((t, x, y)), phase, "nBins", [40, 30], "smooth", [1.0, 1.0])
    assert pmap2["phase"].shape == (30, 40)


def test_brain_states_api() -> None:
    t = np.linspace(0.0, 100.0, 600)
    f = np.linspace(0.5, 120.0, 200)

    spec = np.ones((f.shape[0], t.shape[0]), dtype=float)
    theta_idx = (f >= 7.0) & (f <= 10.0)
    delta_idx = (f >= 0.5) & (f <= 4.0)

    sleep_mask = ((t > 20.0) & (t < 40.0)) | ((t > 60.0) & (t < 80.0))
    spec[theta_idx][:, sleep_mask] *= 4.0
    spec[delta_idx][:, sleep_mask] *= 1.2

    q = np.column_stack((t, sleep_mask.astype(float)))
    emg = np.column_stack((t, 0.5 + 0.2 * np.sin(2.0 * np.pi * 0.02 * t)))

    out = brain_states(spec, t, f, q, emg=emg, method="hippocampus", n_clusters=2, random_state=0)
    assert isinstance(out["exploration"], nap.Tsd)
    assert isinstance(out["sws"], nap.Tsd)
    assert isinstance(out["rem"], nap.Tsd)

    exploration, sws, rem = BrainStates(spec, t, f, q, emg, "method", "hippocampus", "nClusters", 2)
    assert exploration.shape == (t.shape[0], 2)
    assert sws.shape == (t.shape[0], 2)
    assert rem.shape == (t.shape[0], 2)


def test_ripple_stats_alias() -> None:
    fs = 1250.0
    t = np.arange(0.0, 4.0, 1.0 / fs)
    sig = 0.2 * np.sin(2.0 * np.pi * 8.0 * t)
    sig += 0.8 * np.sin(2.0 * np.pi * 140.0 * t) * ((t > 1.0) & (t < 1.08))
    sig += 0.8 * np.sin(2.0 * np.pi * 150.0 * t) * ((t > 2.0) & (t < 2.09))

    filtered = nap.Tsd(t=t, d=sig)
    ripples = np.array([[1.0, 1.08], [2.0, 2.09]], dtype=float)

    maps, data, stats = RippleStats(filtered, ripples)
    assert "ripples" in maps
    assert maps["ripples"].shape[0] == 2
    assert data["duration"].shape[0] == 2
    assert "acg" in stats
