"""Event-synchronization and short-time correlogram utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


_EPS = 1e-12


def _collect_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2 != 0:
        raise ValueError("Positional options must be key/value pairs.")
    options: dict[str, Any] = {}
    for k, v in kwargs.items():
        if not isinstance(k, str):
            raise TypeError("Option keys must be strings.")
        options[k.lower()] = v
    for k, v in zip(args[0::2], args[1::2]):
        if not isinstance(k, str):
            raise TypeError("Option keys must be strings.")
        options[k.lower()] = v
    return options


def _as_sample_array(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError("samples must be 1D timestamps or 2D array.")


def sync(
    samples: np.ndarray,
    sync_times: np.ndarray,
    *,
    durations: tuple[float, float] = (-0.5, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    """Align sample timestamps relative to synchronizing events."""
    s = _as_sample_array(samples)
    sync_t = np.asarray(sync_times, dtype=float).reshape(-1)
    if s.size == 0 or sync_t.size == 0:
        return np.empty((0, s.shape[1]), dtype=float), np.array([], dtype=int)

    if s.shape[1] >= 1:
        t = s[:, 0]
    else:
        raise ValueError("samples must include timestamps in first column.")

    d0, d1 = float(durations[0]), float(durations[1])
    if d0 >= d1:
        raise ValueError("durations must be increasing (start, end).")

    out_rows: list[np.ndarray] = []
    out_idx: list[np.ndarray] = []
    prev = 0
    for i, st in enumerate(sync_t):
        lo = st + d0
        hi = st + d1
        j0 = np.searchsorted(t, lo, side="left", sorter=None)
        if j0 < prev:
            j0 = prev
        j1 = np.searchsorted(t, hi, side="right", sorter=None)
        if j1 <= j0:
            continue
        block = s[j0:j1].copy()
        block[:, 0] = block[:, 0] - st
        out_rows.append(block)
        out_idx.append(np.full(block.shape[0], i + 1, dtype=int))  # MATLAB-style 1-based trial IDs
        prev = j0

    if not out_rows:
        return np.empty((0, s.shape[1]), dtype=float), np.array([], dtype=int)
    return np.vstack(out_rows), np.concatenate(out_idx)


def _bin_time(values: np.ndarray, durations: tuple[float, float], n_bins: int, trim: bool = True) -> np.ndarray:
    lo, hi = float(durations[0]), float(durations[1])
    x = np.asarray(values, dtype=float)
    idx = np.floor((x - lo) / max(hi - lo, _EPS) * n_bins).astype(int)
    if trim:
        keep = (idx >= 0) & (idx < n_bins)
        out = np.full(idx.shape, -1, dtype=int)
        out[keep] = idx[keep]
        return out
    return np.clip(idx, 0, n_bins - 1)


def _maybe_smooth1d(values: np.ndarray, smooth: float | tuple[float, float] | None) -> np.ndarray:
    if smooth is None:
        return values
    sm = np.asarray(smooth, dtype=float).reshape(-1)
    sigma = float(sm[0]) if sm.size > 0 else 0.0
    if sigma <= 0:
        return values
    return gaussian_filter1d(values, sigma=sigma, mode="reflect")


def sync_map(
    synchronized: np.ndarray,
    indices: np.ndarray,
    *,
    durations: tuple[float, float] = (-0.5, 0.5),
    n_bins: int = 100,
    smooth: float | tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a trial-by-time map from synchronized samples."""
    s = _as_sample_array(synchronized)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if s.shape[0] != idx.shape[0]:
        raise ValueError("synchronized and indices must have same length.")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0.")
    if s.shape[0] == 0:
        return np.empty((0, n_bins), dtype=float), np.linspace(durations[0], durations[1], n_bins)

    if smooth is None:
        smooth_use: float | tuple[float, float] = float(np.ceil(0.01 * n_bins))
    else:
        smooth_use = smooth

    n_trials = int(np.max(idx))
    dt = (durations[1] - durations[0]) / float(n_bins)
    time_bins = np.linspace(durations[0], durations[1] - dt, n_bins) + 0.5 * dt

    tb = _bin_time(s[:, 0], durations, n_bins, trim=True)
    keep = tb >= 0
    trials = idx[keep] - 1
    bins = tb[keep]

    if s.shape[1] == 1:
        m = np.zeros((n_trials, n_bins), dtype=float)
        np.add.at(m, (trials, bins), 1.0)
        if np.isscalar(smooth_use):
            sm = gaussian_filter(m, sigma=(0.0, float(smooth_use)), mode="reflect")
        else:
            sv, sh = np.asarray(smooth_use, dtype=float).reshape(-1)[:2]
            sm = gaussian_filter(m, sigma=(float(sv), float(sh)), mode="reflect")
        return sm, time_bins

    values = s[keep, 1]
    sums = np.zeros((n_trials, n_bins), dtype=float)
    counts = np.zeros((n_trials, n_bins), dtype=float)
    np.add.at(sums, (trials, bins), values)
    np.add.at(counts, (trials, bins), 1.0)

    if np.isscalar(smooth_use):
        sv, sh = float(smooth_use), 0.0
    else:
        a = np.asarray(smooth_use, dtype=float).reshape(-1)
        sv = float(a[0]) if a.size > 0 else 0.0
        sh = float(a[1]) if a.size > 1 else 0.0
    sums_sm = gaussian_filter(sums, sigma=(sv, sh), mode="reflect")
    counts_sm = gaussian_filter(counts, sigma=(sv, sh), mode="reflect")
    return sums_sm / np.maximum(counts_sm, _EPS), time_bins


def sync_hist(
    synchronized: np.ndarray,
    indices: np.ndarray,
    *,
    mode: str = "sum",
    durations: tuple[float, float] = (-0.5, 0.5),
    n_bins: int = 100,
    data_type: str = "linear",
    smooth: float | tuple[float, float] = 0.0,
    bins: tuple[float, float, int] | None = None,
    error: str = "std",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute sum/mean/distribution histograms from synchronized data."""
    s = _as_sample_array(synchronized)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if s.shape[0] != idx.shape[0]:
        raise ValueError("synchronized and indices must have same length.")
    if s.shape[0] == 0:
        return np.array([]), np.array([]), None

    mode_use = str(mode).lower()
    if mode_use not in {"sum", "mean", "dist"}:
        raise ValueError("mode must be one of {'sum', 'mean', 'dist'}.")

    dt = (durations[1] - durations[0]) / float(n_bins)
    time_bins = np.linspace(durations[0], durations[1] - dt, n_bins) + 0.5 * dt

    tb = _bin_time(s[:, 0], durations, n_bins, trim=True)
    keep = tb >= 0
    trial = idx[keep] - 1
    tb = tb[keep]
    n_trials = int(np.max(idx))

    if s.shape[1] == 1:
        values = np.ones(tb.shape[0], dtype=float)
        point_process = True
    else:
        values = s[keep, 1]
        point_process = False

    if mode_use == "sum":
        out = np.zeros((n_bins,), dtype=float)
        np.add.at(out, tb, values)
        out = _maybe_smooth1d(out, smooth)
        return out, time_bins, None

    if mode_use == "mean":
        if point_process:
            cnt = np.zeros((n_bins,), dtype=float)
            np.add.at(cnt, tb, 1.0)
            cnt = _maybe_smooth1d(cnt, smooth)
            mean = cnt / max(n_trials * dt, _EPS)
            return mean, time_bins, None

        sums = np.zeros((n_bins,), dtype=float)
        n = np.zeros((n_bins,), dtype=float)
        sq = np.zeros((n_bins,), dtype=float)
        np.add.at(sums, tb, values)
        np.add.at(n, tb, 1.0)
        np.add.at(sq, tb, values**2)

        sums = _maybe_smooth1d(sums, smooth)
        n = _maybe_smooth1d(n, smooth)
        sq = _maybe_smooth1d(sq, smooth)
        mean = sums / np.maximum(n, _EPS)

        err_mode = str(error).lower()
        if str(data_type).lower().startswith("c"):
            return mean, time_bins, None
        if err_mode == "std":
            var = np.maximum(sq / np.maximum(n, _EPS) - mean**2, 0.0)
            e = np.sqrt(var)
            return mean, time_bins, e
        if err_mode == "sem":
            var = np.maximum(sq / np.maximum(n, _EPS) - mean**2, 0.0)
            e = np.sqrt(var) / np.sqrt(np.maximum(n, 1.0))
            return mean, time_bins, e
        if err_mode == "95%":
            lo = np.full(n_bins, np.nan, dtype=float)
            hi = np.full(n_bins, np.nan, dtype=float)
            for i in range(n_bins):
                vals = values[tb == i]
                if vals.size > 0:
                    lo[i], hi[i] = np.percentile(vals, [5, 95])
            ci = np.column_stack((lo, hi))
            return mean, time_bins, ci
        raise ValueError("error must be one of {'std','sem','95%'}.")

    # mode == 'dist'
    if point_process:
        trial_time = np.zeros((n_trials, n_bins), dtype=float)
        np.add.at(trial_time, (trial, tb), 1.0)
        max_n = int(np.max(trial_time))
        dist = np.zeros((max_n + 1, n_bins), dtype=float)
        for b in range(n_bins):
            col = trial_time[:, b].astype(int)
            h = np.bincount(col, minlength=max_n + 1)
            dist[:, b] = h / max(n_trials, 1)
        val_bins = np.arange(max_n + 1, dtype=float)
        return dist, time_bins, val_bins

    if bins is None:
        lo = float(np.min(values))
        hi = float(np.max(values))
        n_v = 100
    else:
        lo, hi, n_v = float(bins[0]), float(bins[1]), int(bins[2])
    if n_v <= 0:
        raise ValueError("bins third value (nBins) must be > 0.")
    edges = np.linspace(lo, hi, n_v + 1)
    centers = edges[:-1] + 0.5 * (edges[1] - edges[0])

    dist = np.zeros((n_v, n_bins), dtype=float)
    for b in range(n_bins):
        vals = values[tb == b]
        if vals.size == 0:
            continue
        h, _ = np.histogram(vals, bins=edges)
        dist[:, b] = h / max(vals.size, 1)

    if np.size(smooth):
        sm = np.asarray(smooth, dtype=float).reshape(-1)
        if sm.size == 1:
            dist = gaussian_filter(dist, sigma=(float(sm[0]), 0.0), mode="reflect")
        else:
            dist = gaussian_filter(dist, sigma=(float(sm[0]), float(sm[1])), mode="reflect")

    return dist, time_bins, centers


def _ccg_counts(ref: np.ndarray, target: np.ndarray, bin_size: float, duration: float, auto: bool) -> tuple[np.ndarray, np.ndarray]:
    half = float(duration) / 2.0
    edges = np.arange(-half, half + bin_size, bin_size)
    centers = edges[:-1] + 0.5 * bin_size
    counts = np.zeros(centers.shape[0], dtype=float)

    tgt = np.asarray(target, dtype=float)
    for t in np.asarray(ref, dtype=float):
        lo = t - half
        hi = t + half
        j0 = np.searchsorted(tgt, lo, side="left")
        j1 = np.searchsorted(tgt, hi, side="right")
        d = tgt[j0:j1] - t
        if auto:
            d = d[np.abs(d) > 1e-12]
        if d.size > 0:
            h, _ = np.histogram(d, bins=edges)
            counts += h
    return counts, centers


def short_time_ccg(
    times1: np.ndarray,
    times2: np.ndarray | None = None,
    *,
    bin_size: float = 0.01,
    duration: float = 2.0,
    window: float = 5 * 60,
    overlap: float | None = None,
    smooth: float | None = None,
    mode: str = "count",
    min_events: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute time-varying cross/auto correlograms over sliding windows."""
    t1 = np.sort(np.asarray(times1, dtype=float).reshape(-1))
    if t1.size == 0:
        return np.empty((0, 0), dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    if times2 is None:
        auto = True
        t2 = t1
    else:
        auto = False
        t2 = np.sort(np.asarray(times2, dtype=float).reshape(-1))
        if t2.size == 0:
            return np.empty((0, 0), dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    if overlap is None:
        overlap = 0.8 * float(window)
    step = float(window) - float(overlap)
    if step <= 0:
        raise ValueError("window-overlap must be > 0.")

    start = min(t1[0], t2[0]) + 0.5 * float(window)
    stop = max(t1[-1], t2[-1]) - 0.5 * float(window)
    if stop <= start:
        return np.empty((0, 0), dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    centers = []
    ccg_cols = []
    t_center = float(start)
    lag_bins = None

    while t_center + 0.5 * float(window) <= stop + 1e-12:
        w0 = t_center - 0.5 * float(window)
        w1 = t_center + 0.5 * float(window)

        a0 = np.searchsorted(t1, w0, side="left")
        a1 = np.searchsorted(t1, w1, side="right")
        b0 = np.searchsorted(t2, w0, side="left")
        b1 = np.searchsorted(t2, w1, side="right")
        r = t1[a0:a1]
        g = t2[b0:b1]

        if r.size + g.size < int(min_events):
            if lag_bins is None:
                _, lag_bins = _ccg_counts(np.array([0.0]), np.array([0.0]), bin_size=float(bin_size), duration=float(duration), auto=True)
            ccg_cols.append(np.full(lag_bins.shape[0], np.nan, dtype=float))
        else:
            counts, lags = _ccg_counts(r, g, bin_size=float(bin_size), duration=float(duration), auto=auto)
            lag_bins = lags
            if auto and counts.size > 0:
                counts[counts.size // 2] = 0.0
            if str(mode).lower().startswith("norm"):
                s = np.sum(counts)
                if s > 0:
                    counts = counts / s
            ccg_cols.append(counts)

        centers.append(t_center)
        t_center += step

    if lag_bins is None:
        return np.empty((0, 0), dtype=float), np.asarray(centers, dtype=float), np.array([], dtype=float)

    ccg = np.column_stack(ccg_cols) if ccg_cols else np.empty((lag_bins.shape[0], 0), dtype=float)
    if smooth is not None and float(smooth) > 0 and ccg.size > 0:
        ccg = gaussian_filter1d(ccg, sigma=float(smooth), axis=0, mode="reflect")
    return ccg, np.asarray(centers, dtype=float), lag_bins


def Sync(samples: np.ndarray, sync_times: np.ndarray, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-compatible alias for :func:`sync`."""
    options = _collect_options(args, kwargs)
    durations = tuple(options.pop("durations", (-0.5, 0.5)))
    _ = options.pop("verbose", None)
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return sync(samples, sync_times, durations=durations)


def SyncMap(synchronized: np.ndarray, indices: np.ndarray, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-compatible alias for :func:`sync_map`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "durations": tuple(options.pop("durations", (-0.5, 0.5))),
        "n_bins": int(options.pop("nbins", options.pop("n_bins", 100))),
        "smooth": options.pop("smooth", None),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return sync_map(synchronized, indices, **mapped)


def SyncHist(synchronized: np.ndarray, indices: np.ndarray, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """MATLAB-compatible alias for :func:`sync_hist`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "mode": options.pop("mode", "sum"),
        "durations": tuple(options.pop("durations", (-0.5, 0.5))),
        "n_bins": int(options.pop("nbins", options.pop("n_bins", 100))),
        "data_type": options.pop("type", options.pop("data_type", "linear")),
        "smooth": options.pop("smooth", 0.0),
        "error": options.pop("error", "std"),
    }
    bins_opt = options.pop("bins", options.pop("hist", None))
    if bins_opt is not None:
        bins_arr = np.asarray(bins_opt, dtype=float).reshape(-1)
        if bins_arr.size != 3:
            raise ValueError("bins must be [min, max, nBins].")
        mapped["bins"] = (float(bins_arr[0]), float(bins_arr[1]), int(bins_arr[2]))
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return sync_hist(synchronized, indices, **mapped)


def ShortTimeCCG(times1: np.ndarray, times2: np.ndarray | None = None, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB-compatible alias for :func:`short_time_ccg`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "bin_size": float(options.pop("binsize", options.pop("bin_size", 0.01))),
        "duration": float(options.pop("duration", 2.0)),
        "window": float(options.pop("window", 5 * 60)),
        "overlap": options.pop("overlap", None),
        "smooth": options.pop("smooth", None),
        "mode": options.pop("mode", "count"),
        "min_events": int(options.pop("min", options.pop("min_events", 1))),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return short_time_ccg(times1, times2, **mapped)


__all__ = [
    "sync",
    "sync_map",
    "sync_hist",
    "short_time_ccg",
    "Sync",
    "SyncMap",
    "SyncHist",
    "ShortTimeCCG",
]
