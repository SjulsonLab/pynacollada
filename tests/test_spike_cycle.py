from __future__ import annotations

import numpy as np
import pynapple as nap

from pynacollada import (
    CountSpikesPerCycle,
    SelectSpikes,
    count_spikes_per_cycle,
    select_spikes,
)


def test_select_spikes_bursts_and_single_modes() -> None:
    spikes = np.array([0.000, 0.002, 0.004, 0.050, 0.090, 0.092], dtype=float)

    burst_mask = select_spikes(spikes, mode="bursts", isi=0.006)
    single_mask = select_spikes(spikes, mode="single", isi=0.020)
    burst_mask_alias = SelectSpikes(spikes, mode="bursts", isi=0.006)

    expected_burst = np.array([True, True, True, False, True, True], dtype=bool)
    expected_single = np.array([False, False, False, True, False, False], dtype=bool)
    np.testing.assert_array_equal(burst_mask, expected_burst)
    np.testing.assert_array_equal(single_mask, expected_single)
    np.testing.assert_array_equal(burst_mask, burst_mask_alias)


def test_count_spikes_per_cycle_with_phase_tsd() -> None:
    fs = 1000.0
    freq = 8.0
    t = np.arange(0.0, 2.0, 1.0 / fs, dtype=float)
    phase = (2.0 * np.pi * freq * t + np.pi) % (2.0 * np.pi) - np.pi
    phase_tsd = nap.Tsd(t=t, d=phase, time_support=nap.IntervalSet(start=t[0], end=t[-1]))

    period = 1.0 / freq
    spike_times = np.arange(period * 0.5, t[-1], period, dtype=float)
    spikes = nap.Ts(t=spike_times, time_support=phase_tsd.time_support)

    count, cycles = count_spikes_per_cycle(spikes, phase_tsd)
    count_alias, cycles_alias = CountSpikesPerCycle(spikes, phase_tsd)

    assert count.shape[0] == np.asarray(cycles.as_units("s").values).shape[0]
    assert count.shape[0] >= 10
    assert np.mean(count) > 0.8
    assert np.mean(count) < 1.2
    np.testing.assert_array_equal(count, count_alias)
    np.testing.assert_allclose(np.asarray(cycles.as_units("s").values), cycles_alias)

