"""FMAT-style phase precession analysis with pynapple-native inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pynapple as nap

from .circular_stats import circular_confidence_intervals, circular_regression, circular_variance


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _coerce_ts(ts: np.ndarray | nap.Ts) -> np.ndarray:
    if isinstance(ts, nap.Ts):
        return np.asarray(ts.as_units("s").index.values, dtype=float).reshape(-1)
    return np.asarray(ts, dtype=float).reshape(-1)


def _coerce_tsd(tsd: np.ndarray | nap.Tsd) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(tsd, nap.Tsd):
        t = np.asarray(tsd.as_units("s").index.values, dtype=float).reshape(-1)
        v = np.asarray(tsd.values, dtype=float).reshape(-1)
        return t, v
    arr = np.asarray(tsd, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Input must be a pynapple.Tsd or Nx2 array [time, value].")
    return arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)


def _interp_with_gap(sample_t: np.ndarray, sample_v: np.ndarray, query_t: np.ndarray, max_gap: float) -> tuple[np.ndarray, np.ndarray]:
    if sample_t.shape[0] < 2:
        return np.full(query_t.shape[0], np.nan, dtype=float), np.zeros(query_t.shape[0], dtype=bool)
    out = np.interp(query_t, sample_t, sample_v, left=np.nan, right=np.nan)
    idx = np.searchsorted(sample_t, query_t, side="left")
    left_idx = np.clip(idx - 1, 0, sample_t.shape[0] - 1)
    right_idx = np.clip(idx, 0, sample_t.shape[0] - 1)
    left_gap = np.abs(query_t - sample_t[left_idx])
    right_gap = np.abs(sample_t[right_idx] - query_t)
    ok = np.isfinite(out) & (np.minimum(left_gap, right_gap) <= max_gap)
    out[~ok] = np.nan
    return out, ok


def _interval_labels(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    labels = np.zeros(times.shape[0], dtype=int)
    for i, (s, e) in enumerate(intervals, start=1):
        mask = (times >= s) & (times <= e)
        labels[mask] = i
    return labels


def _lap_intervals_from_boundaries(pos_t: np.ndarray, pos_x: np.ndarray, boundaries: tuple[float, float]) -> np.ndarray:
    start_b, stop_b = map(float, boundaries)
    if start_b < stop_b:
        inside = (pos_x >= start_b) & (pos_x <= stop_b)
    else:
        inside = (pos_x >= start_b) | (pos_x <= stop_b)
    enters = np.flatnonzero(~inside[:-1] & inside[1:]) + 1
    leaves = np.flatnonzero(inside[:-1] & ~inside[1:])
    if inside[0]:
        enters = np.insert(enters, 0, 0)
    if inside[-1]:
        leaves = np.append(leaves, pos_t.shape[0] - 1)
    if enters.size == 0 or leaves.size == 0:
        return np.empty((0, 2), dtype=float)
    if enters[0] > leaves[0]:
        leaves = leaves[1:]
    if enters.size and leaves.size and enters[-1] > leaves[-1]:
        enters = enters[:-1]
    n = min(enters.shape[0], leaves.shape[0])
    enters = enters[:n]
    leaves = leaves[:n]
    valid = leaves > enters
    return np.column_stack((pos_t[enters[valid]], pos_t[leaves[valid]]))


def phase_precession(
    positions: np.ndarray | nap.Tsd | None,
    spikes: np.ndarray | nap.Ts,
    phases: np.ndarray | nap.Tsd,
    *,
    max_gap: float = 0.1,
    boundaries: str | tuple[float, float] = "count",
    slope: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Compute FMAT-like phase-precession outputs.

    Parameters
    ----------
    positions
        Linearized positions in [0,1] as `nap.Tsd` (or Nx2 array). Can be `None`.
    spikes
        Spike timestamps (`nap.Ts` or 1D array).
    phases
        Phase samples (`nap.Tsd` or Nx2 array) in radians.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative.")
    spike_t = _coerce_ts(spikes)
    phase_t, phase_v = _coerce_tsd(phases)
    if spike_t.shape[0] == 0:
        return _empty_phase_precession_output(positions)

    phase_unwrapped = np.unwrap(phase_v)
    spike_phase_unwrapped = np.interp(spike_t, phase_t, phase_unwrapped, left=np.nan, right=np.nan)
    spike_phase = _wrap_to_pi(spike_phase_unwrapped)
    valid_phase = np.isfinite(spike_phase)
    spike_t = spike_t[valid_phase]
    spike_phase = spike_phase[valid_phase]
    spike_phase_unwrapped = spike_phase_unwrapped[valid_phase]
    if spike_t.shape[0] == 0:
        return _empty_phase_precession_output(positions)

    # Output skeleton
    data, stats = _empty_phase_precession_output(positions)
    data["rate"]["t"] = spike_t
    data["rate"]["phase"] = spike_phase

    # Spike rate per +/- one cycle in unwrapped-phase domain
    cycle = 2.0 * np.pi
    sorted_phase = np.sort(spike_phase_unwrapped)
    lo = np.searchsorted(sorted_phase, spike_phase_unwrapped - cycle, side="left")
    hi = np.searchsorted(sorted_phase, spike_phase_unwrapped + cycle, side="right")
    data["rate"]["r"] = (hi - lo).astype(int)

    if positions is None:
        data["rate"]["lap"] = np.zeros(spike_t.shape[0], dtype=int)
        return data, stats

    pos_t, pos_x = _coerce_tsd(positions)
    pos_x = np.asarray(pos_x, dtype=float)
    if np.nanmax(pos_x) > 1.0 or np.nanmin(pos_x) < 0.0:
        xmin = np.nanmin(pos_x)
        xmax = np.nanmax(pos_x)
        if xmax > xmin:
            pos_x = (pos_x - xmin) / (xmax - xmin)
    data["x"] = np.column_stack((pos_t, pos_x))

    spike_x, ok_pos = _interp_with_gap(pos_t, pos_x, spike_t, max_gap=max_gap)
    data["position"]["ok"] = ok_pos
    data["position"]["t"] = spike_t[ok_pos]
    data["position"]["x"] = spike_x[ok_pos]
    data["position"]["phase"] = spike_phase[ok_pos]
    data["position"]["lap"] = np.zeros(np.sum(ok_pos), dtype=int)

    # Lap labels
    if isinstance(boundaries, (tuple, list, np.ndarray)):
        b = np.asarray(boundaries, dtype=float).reshape(-1)
        if b.shape[0] != 2:
            raise ValueError("boundaries must be 'count' or a (start, stop) pair.")
        stats["boundaries"] = np.array([b[0], b[1]], dtype=float)
        intervals = _lap_intervals_from_boundaries(pos_t, pos_x, (float(b[0]), float(b[1])))
    elif str(boundaries).lower() == "count":
        stats["boundaries"] = np.array([], dtype=float)
        # Heuristic fallback: one lap between successive low-rate / high-rate transitions
        rb = np.searchsorted(sorted_phase, spike_phase_unwrapped - 8.0 * cycle, side="left")
        ra = np.searchsorted(sorted_phase, spike_phase_unwrapped + 8.0 * cycle, side="right")
        rate_before = (np.searchsorted(sorted_phase, spike_phase_unwrapped, side="right") - rb).astype(int)
        rate_after = (ra - np.searchsorted(sorted_phase, spike_phase_unwrapped, side="left")).astype(int)
        start = (rate_before <= 4) & (rate_after >= 16)
        stop = (rate_before >= 16) & (rate_after <= 4)
        si = np.flatnonzero(start)
        ei = np.flatnonzero(stop)
        if si.size and ei.size:
            if si[0] >= ei[0]:
                ei = ei[1:]
            if si.size and ei.size and si[-1] >= ei[-1]:
                si = si[:-1]
            n = min(si.size, ei.size)
            intervals = np.column_stack((spike_t[si[:n]], spike_t[ei[:n]])) if n else np.empty((0, 2), dtype=float)
        else:
            intervals = np.empty((0, 2), dtype=float)
    else:
        raise ValueError("boundaries must be 'count' or an explicit (start, stop) pair.")

    if intervals.shape[0]:
        data["position"]["lap"] = _interval_labels(data["position"]["t"], intervals)
        data["rate"]["lap"] = _interval_labels(data["rate"]["t"], intervals)
    else:
        data["rate"]["lap"] = np.zeros(data["rate"]["t"].shape[0], dtype=int)

    # Regression and subfield stats
    x = data["position"]["x"]
    ph = data["position"]["phase"]
    ok = np.isfinite(x) & np.isfinite(ph)
    if np.sum(ok) >= 3:
        reg = circular_regression(x[ok], ph[ok], slope=slope)
        stats["slope"] = float(reg["beta"][0])
        stats["intercept"] = float(reg["beta"][1])
        stats["r2"] = float(reg["R2"])
        stats["p"] = np.nan

        for lap in np.unique(data["position"]["lap"]):
            if lap == 0:
                continue
            lap_mask = data["position"]["lap"] == lap
            if np.sum(lap_mask) < 3:
                continue
            reg_lap = circular_regression(data["position"]["x"][lap_mask], data["position"]["phase"][lap_mask], slope=stats["slope"])
            stats["lap"]["slope"].append(float(reg_lap["beta"][0]))
            stats["lap"]["intercept"].append(float(reg_lap["beta"][1]))
            stats["lap"]["r2"].append(float(reg_lap["R2"]))
            stats["lap"]["p"].append(np.nan)

        x0 = float(np.nanmin(x[ok]))
        x1 = float(np.nanmax(x[ok]))
        dx = x1 - x0
        if dx > 0:
            for i in range(3):
                lo = x0 + dx / 3.0 * i
                hi = x0 + dx / 3.0 * (i + 1)
                mask = ok & (x > lo) & (x < hi)
                if np.sum(mask) == 0:
                    continue
                stats["all"][i] = data["position"]["phase"][mask]
                stats["x"][i] = x0 + dx / 6.0 * (2 * i + 1)
                var_i, std_i = circular_variance(data["position"]["phase"][mask])
                mean_i, conf_i = circular_confidence_intervals(data["position"]["phase"][mask], alpha=0.05)
                stats["var"][i] = float(var_i)
                stats["std"][i] = float(std_i)
                stats["mean"][i] = float(mean_i)
                stats["conf"][:, i] = conf_i

    # Phase-vs-rate stats
    max_rate = int(np.nanmax(data["rate"]["r"])) if data["rate"]["r"].size else 0
    if max_rate > 0:
        stats["rate"]["mean"] = np.full(max_rate, np.nan, dtype=float)
        stats["rate"]["conf"] = np.full((2, max_rate), np.nan, dtype=float)
        for rate in range(1, max_rate + 1):
            mask = data["rate"]["r"] == rate
            if np.any(mask):
                mean_r, conf_r = circular_confidence_intervals(data["rate"]["phase"][mask], alpha=0.05)
                stats["rate"]["mean"][rate - 1] = mean_r
                stats["rate"]["conf"][:, rate - 1] = conf_r

    return data, stats


def _empty_phase_precession_output(positions: np.ndarray | nap.Tsd | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if positions is None:
        x_data = np.empty((0, 2), dtype=float)
    elif isinstance(positions, nap.Tsd):
        t = np.asarray(positions.as_units("s").index.values, dtype=float)
        d = np.asarray(positions.values, dtype=float)
        x_data = np.column_stack((t, d))
    else:
        arr = np.asarray(positions, dtype=float)
        x_data = arr if arr.ndim == 2 and arr.shape[1] >= 2 else np.empty((0, 2), dtype=float)

    data = {
        "x": x_data,
        "position": {
            "ok": np.array([], dtype=bool),
            "t": np.array([], dtype=float),
            "x": np.array([], dtype=float),
            "phase": np.array([], dtype=float),
            "lap": np.array([], dtype=int),
        },
        "rate": {
            "t": np.array([], dtype=float),
            "r": np.array([], dtype=int),
            "phase": np.array([], dtype=float),
            "lap": np.array([], dtype=int),
        },
    }
    stats = {
        "slope": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "p": np.nan,
        "x": np.full(3, np.nan, dtype=float),
        "mean": np.full(3, np.nan, dtype=float),
        "var": np.full(3, np.nan, dtype=float),
        "std": np.full(3, np.nan, dtype=float),
        "conf": np.full((2, 3), np.nan, dtype=float),
        "all": [np.array([np.nan], dtype=float), np.array([np.nan], dtype=float), np.array([np.nan], dtype=float)],
        "rate": {"mean": np.full(16, np.nan, dtype=float), "conf": np.full((2, 16), np.nan, dtype=float)},
        "boundaries": np.array([], dtype=float),
        "lap": {"slope": [], "intercept": [], "r2": [], "p": []},
    }
    return data, stats


def PhasePrecession(
    positions: np.ndarray | nap.Tsd | None,
    spikes: np.ndarray | nap.Ts,
    phases: np.ndarray | nap.Tsd,
    *,
    maxGap: float = 0.1,
    boundaries: str | tuple[float, float] = "count",
    slope: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """MATLAB-compatibility alias for `phase_precession`."""
    return phase_precession(positions, spikes, phases, max_gap=maxGap, boundaries=boundaries, slope=slope)

