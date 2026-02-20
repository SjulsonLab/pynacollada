"""Brain-state helper utilities based on archive workflows."""

from __future__ import annotations

import numpy as np
import pynapple as nap
from scipy.signal import find_peaks


def refine_sleep_from_accel(
    acceleration: nap.Tsd | nap.TsdFrame,
    sleep_ep: nap.IntervalSet,
    *,
    peak_threshold: float = 0.025,
    min_quiet_duration_s: float = 15.0,
    merge_gap_s: float = 0.1,
) -> nap.IntervalSet:
    """
    Refine sleep intervals from accelerometer-derived movement bursts.

    This follows the archive algorithm:
    1. Differentiate absolute acceleration in candidate sleep.
    2. Detect movement peaks above threshold.
    3. Keep inter-peak intervals longer than `min_quiet_duration_s`.
    4. Merge close intervals and intersect with original sleep epoch.
    """
    if not isinstance(sleep_ep, nap.IntervalSet):
        raise TypeError("sleep_ep must be a pynapple.IntervalSet.")
    if peak_threshold <= 0:
        raise ValueError("peak_threshold must be positive.")
    if min_quiet_duration_s <= 0:
        raise ValueError("min_quiet_duration_s must be positive.")
    if merge_gap_s < 0:
        raise ValueError("merge_gap_s must be non-negative.")

    if isinstance(acceleration, nap.TsdFrame):
        acc = acceleration[:, 0]
    elif isinstance(acceleration, nap.Tsd):
        acc = acceleration
    else:
        raise TypeError("acceleration must be a pynapple.Tsd or TsdFrame.")

    acc_r = acc.restrict(sleep_ep).as_units("s")
    if acc_r.shape[0] < 3:
        return nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))

    t = np.asarray(acc_r.index.values, dtype=float)
    v = np.asarray(acc_r.values, dtype=float)
    dv = np.abs(np.diff(v))
    dt = t[1:]
    peak_idx, _ = find_peaks(dv, height=peak_threshold)
    if peak_idx.size < 2:
        return nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))

    peak_t = dt[peak_idx]
    durations = np.diff(peak_t)
    starts = peak_t[:-1]
    ends = peak_t[1:]
    keep = durations > min_quiet_duration_s
    if not np.any(keep):
        return nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))

    s_keep = starts[keep]
    e_keep = np.maximum(ends[keep] - 1e-5, s_keep + 1e-5)
    quiet = nap.IntervalSet(start=s_keep, end=e_keep)
    if merge_gap_s > 0:
        quiet = quiet.merge_close_intervals(merge_gap_s, time_units="s")
    return sleep_ep.intersect(quiet)


def refineSleepFromAccel(
    acceleration: nap.Tsd | nap.TsdFrame,
    sleep_ep: nap.IntervalSet,
) -> nap.IntervalSet:
    """MATLAB/archive-compatibility alias for `refine_sleep_from_accel`."""
    return refine_sleep_from_accel(acceleration, sleep_ep)
