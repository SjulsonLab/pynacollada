from __future__ import annotations

import numpy as np
import pynapple as nap

from pynacollada import PhasePrecession, phase_precession


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def test_phase_precession_with_explicit_boundaries() -> None:
    dt = 0.01
    t = np.arange(0.0, 100.0, dt, dtype=float)
    lap_period = 10.0
    x = (t % lap_period) / lap_period
    theta_phase = _wrap_to_pi(2.0 * np.pi * 8.0 * t - 2.0 * np.pi * x)

    support = nap.IntervalSet(start=t[0], end=t[-1])
    pos_tsd = nap.Tsd(t=t, d=x, time_support=support)
    phase_tsd = nap.Tsd(t=t, d=theta_phase, time_support=support)

    spike_t = np.arange(0.05, 100.0, 0.2, dtype=float)
    spike_x = (spike_t % lap_period) / lap_period
    spike_t = spike_t[(spike_x >= 0.2) & (spike_x <= 0.8)]
    spikes = nap.Ts(t=spike_t, time_support=support)

    data, stats = phase_precession(
        pos_tsd,
        spikes,
        phase_tsd,
        max_gap=0.05,
        boundaries=(0.2, 0.8),
        slope=0.0,
    )
    data_alias, stats_alias = PhasePrecession(
        pos_tsd,
        spikes,
        phase_tsd,
        maxGap=0.05,
        boundaries=(0.2, 0.8),
        slope=0.0,
    )

    assert data["position"]["t"].shape[0] > 20
    assert np.isfinite(stats["slope"])
    assert abs(stats["slope"]) > 0.1
    assert len(stats["lap"]["slope"]) > 0
    np.testing.assert_allclose(stats["slope"], stats_alias["slope"])
    np.testing.assert_allclose(data["rate"]["r"], data_alias["rate"]["r"])


def test_phase_precession_without_positions() -> None:
    t = np.linspace(0.0, 20.0, 4000)
    phase = _wrap_to_pi(2.0 * np.pi * 7.0 * t)
    phase_tsd = nap.Tsd(t=t, d=phase, time_support=nap.IntervalSet(start=t[0], end=t[-1]))
    spikes = nap.Ts(t=np.arange(0.1, 20.0, 0.3), time_support=phase_tsd.time_support)

    data, stats = phase_precession(None, spikes, phase_tsd)

    assert data["position"]["t"].shape[0] == 0
    assert data["rate"]["t"].shape[0] > 0
    assert np.isnan(stats["slope"])
