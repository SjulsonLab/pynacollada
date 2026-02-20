from __future__ import annotations

import numpy as np
import pynapple as nap

from pynacollada import refineSleepFromAccel, refine_sleep_from_accel


def test_refine_sleep_from_accel_detects_quiet_intervals() -> None:
    fs = 10.0
    t = np.arange(0.0, 120.0, 1.0 / fs, dtype=float)
    acc = 0.001 * np.sin(2.0 * np.pi * 0.2 * t)

    # Movement bursts separated by quiet periods > 15 s.
    for t0 in (20.0, 40.0, 80.0, 100.0):
        idx = int(np.argmin(np.abs(t - t0)))
        acc[idx : idx + 3] += np.array([0.0, 0.4, 0.0], dtype=float)

    support = nap.IntervalSet(start=t[0], end=t[-1])
    accel = nap.TsdFrame(t=t, d=acc[:, None], columns=np.array([0]), time_support=support)
    sleep_ep = nap.IntervalSet(start=0.0, end=120.0)

    refined = refine_sleep_from_accel(accel, sleep_ep, peak_threshold=0.05, min_quiet_duration_s=15.0, merge_gap_s=0.05)
    refined_default = refine_sleep_from_accel(accel, sleep_ep)
    refined_alias = refineSleepFromAccel(accel, sleep_ep)

    values = np.asarray(refined.as_units("s").values, dtype=float)
    assert values.shape[0] >= 1
    assert np.all(values[:, 1] - values[:, 0] > 15.0)
    assert np.sum(values[:, 1] - values[:, 0]) > 50.0
    np.testing.assert_allclose(np.asarray(refined_default.as_units("s").values), np.asarray(refined_alias.as_units("s").values))
