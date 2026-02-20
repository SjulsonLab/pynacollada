"""FMAT-style spike-process helpers built around pynapple objects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pynapple as nap


def _coerce_spike_times(spikes: np.ndarray | nap.Ts) -> tuple[np.ndarray, nap.IntervalSet | None]:
    if isinstance(spikes, nap.Ts):
        times = np.asarray(spikes.as_units("s").index.values, dtype=float)
        return times.reshape(-1), spikes.time_support
    arr = np.asarray(spikes, dtype=float).reshape(-1)
    return arr, None


def select_spikes(
    spikes: np.ndarray | nap.Ts,
    mode: str = "bursts",
    isi: float | None = None,
) -> np.ndarray:
    """
    Discriminate burst spikes vs isolated spikes using neighboring ISIs.

    Parameters
    ----------
    spikes
        Spike timestamps (`nap.Ts` or 1D array, in seconds).
    mode
        Either `"bursts"` or `"single"`.
    isi
        Threshold in seconds. Defaults:
        - bursts: 0.006 s
        - single: 0.020 s
    """
    t, _ = _coerce_spike_times(spikes)
    n = t.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    mode_use = str(mode).lower()
    if mode_use not in ("bursts", "single"):
        raise ValueError("mode must be 'bursts' or 'single'.")
    if isi is None:
        isi = 0.006 if mode_use == "bursts" else 0.020
    if isi <= 0:
        raise ValueError("isi must be positive.")

    dt = np.diff(t)
    if mode_use == "bursts":
        near = dt < float(isi)
        selected = np.concatenate(([False], near)) | np.concatenate((near, [False]))
    else:
        far = dt > float(isi)
        selected = np.concatenate(([False], far)) & np.concatenate((far, [False]))
    return selected.astype(bool)


def _coerce_phase_series(phases: np.ndarray | nap.Tsd) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(phases, nap.Tsd):
        t = np.asarray(phases.as_units("s").index.values, dtype=float).reshape(-1)
        p = np.asarray(phases.values, dtype=float).reshape(-1)
        return t, p
    arr = np.asarray(phases, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("phases must be a pynapple.Tsd or an Nx2 array [time, phase].")
    return arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)


def _positive_zero_crossings(times: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Positive-going zero crossings for wrapped phase in [-pi, pi]."""
    p = (phase + np.pi) % (2.0 * np.pi) - np.pi
    up = (p[:-1] < 0.0) & (p[1:] >= 0.0)
    idx = np.flatnonzero(up)
    if idx.size == 0:
        return np.array([], dtype=float)
    t0 = times[idx]
    t1 = times[idx + 1]
    p0 = p[idx]
    p1 = p[idx + 1]
    denom = p1 - p0
    denom[np.abs(denom) < 1e-15] = 1e-15
    frac = -p0 / denom
    return t0 + frac * (t1 - t0)


def count_spikes_per_cycle(
    spikes: np.ndarray | nap.Ts,
    phases: np.ndarray | nap.Tsd,
) -> tuple[np.ndarray, nap.IntervalSet]:
    """
    Count spikes per oscillatory cycle from instantaneous phase samples.

    Cycles are delimited by successive positive-going zero crossings of the
    wrapped phase.
    """
    spike_t, support = _coerce_spike_times(spikes)
    phase_t, phase_v = _coerce_phase_series(phases)
    if phase_t.shape[0] < 3:
        empty = nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))
        return np.array([], dtype=int), empty

    crossings = _positive_zero_crossings(phase_t, phase_v)
    if crossings.shape[0] < 2:
        empty = nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))
        return np.array([], dtype=int), empty

    eps = 1e-5
    starts = crossings[:-1]
    ends = crossings[1:]
    valid = ends > starts + eps
    starts = starts[valid]
    ends = ends[valid]
    if starts.shape[0] == 0:
        empty = nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))
        return np.array([], dtype=int), empty
    ends = np.maximum(ends - eps, starts + eps)
    i0 = np.searchsorted(spike_t, starts, side="left")
    i1 = np.searchsorted(spike_t, ends, side="right")
    count = (i1 - i0).astype(int)

    if support is None:
        cycles = nap.IntervalSet(start=starts, end=ends)
    else:
        support_values = np.asarray(support.as_units("s").values, dtype=float)
        keep = np.zeros(starts.shape[0], dtype=bool)
        for s0, s1 in support_values:
            keep |= (starts >= s0) & (ends <= s1)
        starts = starts[keep]
        ends = ends[keep]
        valid = ends > starts + eps
        starts = starts[valid]
        ends = ends[valid]
        if starts.shape[0] == 0:
            empty = nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))
            return np.array([], dtype=int), empty
        cycles = nap.IntervalSet(start=starts, end=ends)
        i0 = np.searchsorted(spike_t, starts, side="left")
        i1 = np.searchsorted(spike_t, ends, side="right")
        count = (i1 - i0).astype(int)
    return count, cycles


def SelectSpikes(
    spikes: np.ndarray | nap.Ts,
    mode: str = "bursts",
    isi: float | None = None,
) -> np.ndarray:
    """MATLAB-compatibility alias for `select_spikes`."""
    return select_spikes(spikes, mode=mode, isi=isi)


def CountSpikesPerCycle(
    spikes: np.ndarray | nap.Ts,
    phases: np.ndarray | nap.Tsd,
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-compatibility alias for `count_spikes_per_cycle`."""
    count, cycles = count_spikes_per_cycle(spikes, phases)
    return count, np.asarray(cycles.as_units("s").values, dtype=float)
