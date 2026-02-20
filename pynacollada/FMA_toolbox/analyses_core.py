"""Core FMAT Analyses utilities implemented with NumPy/SciPy and pynapple."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from scipy.signal import cheby2, filtfilt, firwin, hilbert, lfilter


_EPS = 1e-12
_BAND_ALIASES = {
    "delta": (0.0, 4.0),
    "theta": (4.0, 10.0),
    "spindles": (10.0, 20.0),
    "gamma": (30.0, 80.0),
    "ripples": (100.0, 250.0),
}


@dataclass(frozen=True)
class _BandConfig:
    smooth: float = 2.0
    custom: tuple[float, float] = (0.0, 250.0)
    delta: tuple[float, float] = (0.0, 4.0)
    theta: tuple[float, float] = (7.0, 10.0)
    spindles: tuple[float, float] = (10.0, 20.0)
    low_gamma: tuple[float, float] = (30.0, 80.0)
    high_gamma: tuple[float, float] = (80.0, 120.0)
    ripples: tuple[float, float] = (100.0, 250.0)
    broad_low: tuple[float, float] = (1.0, 12.0)
    amy_gamma: tuple[float, float] = (45.0, 65.0)


def _collect_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2 != 0:
        raise ValueError("Positional options must be provided as key/value pairs.")
    options: dict[str, Any] = {}
    for key, value in kwargs.items():
        if not isinstance(key, str):
            raise TypeError("Option keys must be strings.")
        options[key.lower()] = value
    for key, value in zip(args[0::2], args[1::2]):
        if not isinstance(key, str):
            raise TypeError("Option keys must be strings.")
        options[key.lower()] = value
    return options


def _as_timestamps(timestamps: np.ndarray | nap.Ts) -> np.ndarray:
    if isinstance(timestamps, nap.Ts):
        return np.asarray(timestamps.as_units("s").index.values, dtype=float).reshape(-1)
    return np.asarray(timestamps, dtype=float).reshape(-1)


def _as_samples(samples: np.ndarray | nap.Tsd | nap.TsdFrame) -> np.ndarray:
    if isinstance(samples, nap.Tsd):
        t = np.asarray(samples.as_units("s").index.values, dtype=float).reshape(-1)
        d = np.asarray(samples.values, dtype=float).reshape(-1, 1)
        return np.column_stack((t, d))
    if isinstance(samples, nap.TsdFrame):
        t = np.asarray(samples.as_units("s").index.values, dtype=float).reshape(-1)
        d = np.asarray(samples.values, dtype=float)
        return np.column_stack((t, d))
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("samples must be an NxM array with M >= 2 (time in first column).")
    return arr


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.copy()
    return gaussian_filter1d(x, sigma=float(sigma), mode="reflect")


def _differentiate(samples: np.ndarray, smooth: float = 0.0) -> np.ndarray:
    arr = _as_samples(samples)
    t = arr[:, 0]
    if arr.shape[0] < 2:
        out = np.zeros_like(arr)
        out[:, 0] = t
        return out
    if np.any(np.diff(t) <= 0):
        raise ValueError("time column must be strictly increasing.")

    vals = arr[:, 1:]
    deriv = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        y = vals[:, j]
        if smooth > 0:
            y = _gaussian_smooth(y, smooth)
        deriv[:, j] = np.gradient(y, t)
    return np.column_stack((t, deriv))


def _coerce_band(passband: str | tuple[float, float] | list[float] | np.ndarray) -> tuple[float, float]:
    if isinstance(passband, str):
        key = passband.strip().lower()
        if key not in _BAND_ALIASES:
            raise ValueError(f"Unknown passband alias '{passband}'.")
        return _BAND_ALIASES[key]
    pb = np.asarray(passband, dtype=float).reshape(-1)
    if pb.size != 2:
        raise ValueError("passband must be a name or a pair [low, high].")
    low, high = float(pb[0]), float(pb[1])
    if low < 0 or high <= low:
        raise ValueError("passband must satisfy 0 <= low < high.")
    return low, high


def _interpolate_columns(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    out = np.empty((query.shape[0], values.shape[1]), dtype=float)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(query, times, values[:, i], left=np.nan, right=np.nan)
    return out


def _periods_from_mask(
    times: np.ndarray,
    mask: np.ndarray,
    *,
    min_duration: float,
    brief_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if t.shape[0] != m.shape[0]:
        raise ValueError("times and mask must have same length.")
    if t.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)

    pad = np.r_[False, m, False].astype(int)
    delta = np.diff(pad)
    starts = np.flatnonzero(delta == 1)
    stops = np.flatnonzero(delta == -1)

    if starts.size == 0:
        state = np.column_stack((t, np.zeros_like(t, dtype=float)))
        return np.empty((0, 2), dtype=float), state

    if brief_gap > 0 and starts.size > 1:
        merged_starts = [int(starts[0])]
        merged_stops: list[int] = []
        current_stop = int(stops[0])
        for i in range(1, starts.size):
            gap_start = int(stops[i - 1])
            gap_end = int(starts[i])
            gap_duration = float(t[min(gap_end, t.size - 1)] - t[max(gap_start - 1, 0)])
            if gap_duration <= brief_gap:
                current_stop = int(stops[i])
            else:
                merged_stops.append(current_stop)
                merged_starts.append(int(starts[i]))
                current_stop = int(stops[i])
        merged_stops.append(current_stop)
        starts = np.asarray(merged_starts, dtype=int)
        stops = np.asarray(merged_stops, dtype=int)

    keep = []
    for s, e in zip(starts, stops):
        s_idx = int(np.clip(s, 0, t.size - 1))
        e_idx = int(np.clip(e - 1, 0, t.size - 1))
        if t[e_idx] - t[s_idx] >= min_duration:
            keep.append((s_idx, e_idx))

    state = np.column_stack((t, np.zeros_like(t, dtype=float)))
    if not keep:
        return np.empty((0, 2), dtype=float), state

    for s_idx, e_idx in keep:
        state[s_idx : e_idx + 1, 1] = 1.0

    periods = np.array([[t[s], t[e]] for s, e in keep], dtype=float)
    return periods, state


def linear_velocity(positions: np.ndarray | nap.TsdFrame, smooth: float = 0.0) -> np.ndarray:
    """Compute instantaneous linear velocity from position samples [t x y]."""
    arr = _as_samples(positions)
    if arr.shape[1] < 3:
        raise ValueError("positions must contain at least [t, x, y].")
    d = _differentiate(arr[:, :3], smooth=smooth)
    speed = np.linalg.norm(d[:, 1:3], axis=1)
    return np.column_stack((arr[:, 0], speed))


def angular_velocity(positions: np.ndarray | nap.TsdFrame, smooth: float = 0.0) -> np.ndarray:
    """Compute instantaneous angular velocity for vector samples [t x y]."""
    arr = _as_samples(positions)
    if arr.shape[1] < 3:
        raise ValueError("positions must contain at least [t, x, y].")
    xy = arr[:, 1:3]
    n = np.linalg.norm(xy, axis=1, keepdims=True)
    n[n < _EPS] = 1.0
    unit = xy / n
    unit_samples = np.column_stack((arr[:, 0], unit))
    d = _differentiate(unit_samples, smooth=smooth)
    omega = unit[:, 0] * d[:, 2] - unit[:, 1] * d[:, 1]
    return np.column_stack((arr[:, 0], omega))


def distance(
    positions: np.ndarray | nap.TsdFrame,
    reference: float | tuple[float, float] | list[float] | np.ndarray,
    *,
    position_type: str = "linear",
) -> np.ndarray:
    """Compute instantaneous distance from positions to a fixed reference point."""
    arr = _as_samples(positions)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    dims = arr.shape[1] - 1
    if ref.size != dims:
        raise ValueError("reference dimensionality must match positions dimensionality.")

    out = np.empty((arr.shape[0], 2), dtype=float)
    out[:, 0] = arr[:, 0]

    if dims == 1:
        x = arr[:, 1]
        if str(position_type).lower().startswith("c"):
            x0 = x
            xmin = np.nanmin(x0)
            xmax = np.nanmax(x0)
            if xmax - xmin > 0 and (np.nanmin(x0) < 0.0 or np.nanmax(x0) > 1.0):
                x = (x0 - xmin) / (xmax - xmin)
            d1 = np.abs(x - ref[0])
            d2 = 1.0 - d1
            out[:, 1] = np.minimum(d1, d2)
        else:
            out[:, 1] = np.abs(x - ref[0])
        return out

    delta = arr[:, 1:] - ref.reshape(1, -1)
    out[:, 1] = np.linalg.norm(delta, axis=1)
    return out


def movement_periods(v: np.ndarray, velocity: float, duration: float, brief: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Find periods where linear velocity remains above threshold."""
    arr = _as_samples(v)
    if arr.shape[1] < 2:
        raise ValueError("v must be [t, velocity].")
    return _periods_from_mask(arr[:, 0], arr[:, 1] > float(velocity), min_duration=float(duration), brief_gap=float(brief))


def quiet_periods(v: np.ndarray, velocity: float, duration: float, brief: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Find periods where linear velocity remains below threshold."""
    arr = _as_samples(v)
    if arr.shape[1] < 2:
        raise ValueError("v must be [t, velocity].")
    return _periods_from_mask(arr[:, 0], arr[:, 1] < float(velocity), min_duration=float(duration), brief_gap=float(brief))


def phase(samples: np.ndarray | nap.Tsd | nap.TsdFrame, times: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute wrapped phase, amplitude, and unwrapped phase via Hilbert transform."""
    arr = _as_samples(samples)
    t = arr[:, 0]
    x = arr[:, 1:]

    analytic = hilbert(x, axis=0)
    wrapped = np.mod(np.angle(analytic), 2.0 * np.pi)
    amp = np.abs(analytic)
    unwrapped = np.unwrap(wrapped, axis=0)

    if times is None:
        return np.column_stack((t, wrapped)), np.column_stack((t, amp)), np.column_stack((t, unwrapped))

    tq = np.asarray(times, dtype=float).reshape(-1)
    if tq.size == 0:
        empty = np.empty((0, x.shape[1] + 1), dtype=float)
        return empty, empty, empty

    # Circular interpolation for wrapped phase through the complex plane.
    z = np.exp(1j * wrapped)
    zr = _interpolate_columns(t, np.real(z), tq)
    zi = _interpolate_columns(t, np.imag(z), tq)
    wrapped_q = np.mod(np.arctan2(zi, zr), 2.0 * np.pi)
    amp_q = _interpolate_columns(t, amp, tq)
    unwrapped_q = _interpolate_columns(t, unwrapped, tq)
    return np.column_stack((tq, wrapped_q)), np.column_stack((tq, amp_q)), np.column_stack((tq, unwrapped_q))


def frequency(
    timestamps: np.ndarray | nap.Ts,
    *,
    method: str = "fixed",
    limits: tuple[float, float] | list[float] | np.ndarray | None = None,
    bin_size: float = 0.05,
    smooth: float = 2.0,
) -> np.ndarray:
    """Compute instantaneous point-process frequency by kernel or ISI methods."""
    ts = np.asarray(_as_timestamps(timestamps), dtype=float).reshape(-1)
    if ts.size == 0:
        return np.empty((0, 2), dtype=float)
    if np.any(np.diff(ts) < 0):
        ts = np.sort(ts)

    method_use = str(method).lower()
    if method_use in ("inverse", "iisi"):
        if ts.size <= 2:
            return np.column_stack((ts, np.zeros_like(ts)))
        ds = np.diff(ts)
        ds[ds <= 0] = np.nan
        mid = ts[:-1] + 0.5 * np.diff(ts)
        iisi = 1.0 / ds
        interp_t = ts[1:-1]
        interp_f = np.interp(interp_t, mid, iisi)
        out_t = np.r_[ts[0], interp_t, ts[-1]]
        out_f = np.r_[0.0, interp_f, 0.0]
        return np.column_stack((out_t, out_f))

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    if limits is None:
        lo = float(ts[0] - 10.0 * bin_size)
        hi = float(ts[-1] + 10.0 * bin_size)
    else:
        lim = np.asarray(limits, dtype=float).reshape(-1)
        if lim.size != 2 or lim[1] <= lim[0]:
            raise ValueError("limits must be [start, stop] with stop > start.")
        lo, hi = float(lim[0]), float(lim[1])

    t = np.arange(lo, hi + 0.5 * bin_size, bin_size, dtype=float)
    if t.size < 2:
        return np.empty((0, 2), dtype=float)

    counts = np.histogram(ts, bins=np.r_[t - 0.5 * bin_size, t[-1] + 0.5 * bin_size])[0].astype(float)
    pilot = _gaussian_smooth(counts / bin_size, smooth)

    if method_use == "fixed":
        return np.column_stack((t, pilot))
    if method_use != "adaptive":
        raise ValueError("method must be one of {'fixed', 'adaptive', 'inverse', 'iisi'}.")

    nz = pilot[pilot > 0]
    if nz.size == 0:
        return np.column_stack((t, pilot))
    mu = float(np.exp(np.mean(np.log(nz))))
    lam = np.sqrt(np.maximum(pilot / max(mu, _EPS), _EPS))
    sigma_sec = (smooth * bin_size) / np.maximum(lam, _EPS)
    sigma_sec = np.clip(sigma_sec, 0.0, t.size * bin_size / 3.0)

    padded = np.r_[counts[::-1], counts, counts[::-1]]
    adaptive = np.zeros_like(pilot)
    for i in range(t.size):
        sigma = float(sigma_sec[i])
        if sigma <= _EPS:
            adaptive[i] = counts[i] / bin_size
            continue
        x = np.arange(0.0, 3.0 * sigma + bin_size, bin_size)
        kernel_x = np.r_[-x[:0:-1], x]
        kernel = np.exp(-(kernel_x**2) / max(sigma**2, _EPS))
        kernel_sum = np.sum(kernel)
        if kernel_sum <= 0:
            adaptive[i] = counts[i] / bin_size
            continue
        kernel /= kernel_sum
        half = kernel.shape[0] // 2
        center = t.size + i
        seg = padded[center - half : center + half + 1]
        adaptive[i] = float(np.sum(seg * kernel) / bin_size)
    return np.column_stack((t, adaptive))


def cv(
    timestamps: np.ndarray | nap.Ts,
    *,
    measure: str = "cv",
    order: int = 1,
    method: str = "fixed",
    bin_size: float = 0.001,
    smooth: float = 25.0,
) -> tuple[float, np.ndarray]:
    """Compute CV, CV2, or operational-time CV for a point process."""
    ts = _as_timestamps(timestamps)
    if ts.size < 2:
        return float(np.nan), np.array([], dtype=float)
    if np.any(np.diff(ts) < 0):
        ts = np.sort(ts)

    order_use = int(order)
    if order_use <= 0:
        raise ValueError("order must be > 0.")

    def _ndiff(x: np.ndarray, n: int) -> np.ndarray:
        if x.shape[0] <= n:
            return np.array([], dtype=float)
        if n == 1:
            return np.diff(x)
        return x[n:] - x[:-n]

    measure_use = str(measure).lower()
    if measure_use == "cv":
        dt = _ndiff(ts, order_use)
        if dt.size == 0 or np.mean(dt) <= 0:
            return float(np.nan), np.array([], dtype=float)
        return float(np.std(dt, ddof=1) / np.mean(dt)), np.array([], dtype=float)

    if measure_use == "cvo":
        f = frequency(ts, method=method, bin_size=bin_size, smooth=smooth)
        if f.size == 0:
            return float(np.nan), np.array([], dtype=float)
        operational = np.cumsum(f[:, 1]) * bin_size
        operational_interp = np.interp(ts, f[:, 0], operational)
        operational_interp = operational_interp + ts[0] - operational_interp[0]
        dto = _ndiff(operational_interp, order_use)
        if dto.size == 0 or np.mean(dto) <= 0:
            return float(np.nan), np.array([], dtype=float)
        return float(np.std(dto, ddof=1) / np.mean(dto)), np.array([], dtype=float)

    if measure_use == "cv2":
        dt = _ndiff(ts, order_use)
        if dt.size < 2:
            return float(np.nan), np.array([], dtype=float)
        dt1 = dt[:-1]
        dt2 = dt[1:]
        x = dt1 / np.maximum(dt2, _EPS)
        local = 2.0 * np.abs(x - 1.0) / np.maximum(x + 1.0, _EPS)
        return float(np.mean(local)), local

    raise ValueError("measure must be one of {'cv', 'cvo', 'cv2'}.")


def filter_lfp(
    lfp: np.ndarray | nap.Tsd | nap.TsdFrame,
    *,
    passband: str | tuple[float, float] | list[float] | np.ndarray = (4.0, 10.0),
    order: int = 4,
    ripple: float = 20.0,
    nyquist: float | None = None,
    filter_type: str = "cheby2",
) -> np.ndarray:
    """Filter LFP samples [t, v1, v2, ...] with Cheby2 or FIR filtering."""
    arr = _as_samples(lfp)
    t = arr[:, 0]
    x = arr[:, 1:]

    if x.size == 0:
        return arr.copy()

    if nyquist is None:
        if t.shape[0] < 2:
            raise ValueError("Need at least 2 samples to infer nyquist.")
        dt = np.median(np.diff(t))
        if dt <= 0:
            raise ValueError("time column must be strictly increasing.")
        nyq = 0.5 / dt
    else:
        nyq = float(nyquist)
    if nyq <= 0:
        raise ValueError("nyquist must be positive.")

    low, high = _coerce_band(passband)
    low_n = max(low / nyq, 0.0)
    high_n = min(high / nyq, 0.999999)
    if high_n <= 0:
        raise ValueError("passband must overlap frequencies above 0 Hz.")

    filt = str(filter_type).lower()
    if filt not in ("cheby2", "fir1"):
        raise ValueError("filter_type must be 'cheby2' or 'fir1'.")

    if filt == "cheby2":
        if low_n <= 0:
            b, a = cheby2(order, ripple, high_n, btype="low")
        else:
            b, a = cheby2(order, ripple, [low_n, high_n], btype="band")
        y = filtfilt(b, a, x, axis=0)
    else:
        taps = max(3, int(order) * 8 + 1)
        if low_n <= 0:
            b = firwin(taps, high_n, pass_zero="lowpass")
        else:
            b = firwin(taps, [low_n, high_n], pass_zero="bandpass")
        y = lfilter(b, [1.0], x, axis=0)

    return np.column_stack((t, y))


def spectrogram_bands(
    spectrogram: np.ndarray,
    frequencies: np.ndarray,
    *,
    smooth: float = 2.0,
    delta: tuple[float, float] = (0.0, 4.0),
    theta: tuple[float, float] = (7.0, 10.0),
    spindles: tuple[float, float] = (10.0, 20.0),
    low_gamma: tuple[float, float] = (30.0, 80.0),
    high_gamma: tuple[float, float] = (80.0, 120.0),
    ripples: tuple[float, float] = (100.0, 250.0),
    broad_low: tuple[float, float] = (1.0, 12.0),
    amy_gamma: tuple[float, float] = (45.0, 65.0),
) -> dict[str, Any]:
    """Compute power trajectories in physiological bands from a spectrogram."""
    s = np.asarray(spectrogram, dtype=float)
    f = np.asarray(frequencies, dtype=float).reshape(-1)
    if s.ndim != 2:
        raise ValueError("spectrogram must be 2D [frequency x time].")
    if f.shape[0] != s.shape[0]:
        raise ValueError("frequencies length must match spectrogram frequency axis.")

    cfg = _BandConfig(
        smooth=float(smooth),
        delta=tuple(map(float, delta)),
        theta=tuple(map(float, theta)),
        spindles=tuple(map(float, spindles)),
        low_gamma=tuple(map(float, low_gamma)),
        high_gamma=tuple(map(float, high_gamma)),
        ripples=tuple(map(float, ripples)),
        broad_low=tuple(map(float, broad_low)),
        amy_gamma=tuple(map(float, amy_gamma)),
    )

    def _band(lo_hi: tuple[float, float]) -> np.ndarray:
        lo, hi = lo_hi
        idx = (f >= lo) & (f <= hi)
        if not np.any(idx):
            return np.full(s.shape[1], np.nan, dtype=float)
        return np.nanmean(s[idx, :], axis=0)

    out: dict[str, Any] = {
        "theta": _band(cfg.theta),
        "delta": _band(cfg.delta),
        "spindles": _band(cfg.spindles),
        "lowGamma": _band(cfg.low_gamma),
        "highGamma": _band(cfg.high_gamma),
        "ripples": _band(cfg.ripples),
        "broadLow": _band(cfg.broad_low),
        "amyGamma": _band(cfg.amy_gamma),
    }

    theta_delta = out["theta"] / np.maximum(out["delta"], _EPS)
    ratio_hip = _gaussian_smooth(theta_delta, cfg.smooth)

    c1 = _band((0.5, 4.5)) / np.maximum(_band((0.5, 9.0)), _EPS)
    c2 = _band((0.5, 20.0)) / np.maximum(_band((0.5, 55.0)), _EPS)
    ratio_cortex = np.column_stack((_gaussian_smooth(c1, cfg.smooth), _gaussian_smooth(c2, cfg.smooth)))

    ratio_amy = _gaussian_smooth(out["amyGamma"] / np.maximum(out["broadLow"], _EPS), cfg.smooth)

    out["ratios"] = {
        "hippocampus": ratio_hip,
        "cortex": ratio_cortex,
        "amygdala": ratio_amy,
    }
    out["ratio"] = ratio_hip
    out["ratio1"] = ratio_cortex
    out["ratio2"] = ratio_cortex[:, 1]
    out["ratio3"] = ratio_amy
    return out


def coherence_bands(
    coherogram: np.ndarray,
    frequencies: np.ndarray,
    *,
    custom: tuple[float, float] = (0.0, 250.0),
    delta: tuple[float, float] = (0.0, 4.0),
    theta: tuple[float, float] = (7.0, 10.0),
    spindles: tuple[float, float] = (10.0, 20.0),
    low_gamma: tuple[float, float] = (30.0, 80.0),
    high_gamma: tuple[float, float] = (80.0, 120.0),
    ripples: tuple[float, float] = (100.0, 250.0),
) -> dict[str, np.ndarray]:
    """Compute coherence trajectories in physiological bands from a coherogram."""
    c = np.asarray(coherogram, dtype=float)
    f = np.asarray(frequencies, dtype=float).reshape(-1)
    if c.ndim != 2:
        raise ValueError("coherogram must be 2D [frequency x time].")
    if f.shape[0] != c.shape[0]:
        raise ValueError("frequencies length must match coherogram frequency axis.")

    def _band(lo_hi: tuple[float, float]) -> np.ndarray:
        lo, hi = map(float, lo_hi)
        idx = (f >= lo) & (f <= hi)
        if not np.any(idx):
            return np.full(c.shape[1], np.nan, dtype=float)
        return np.nanmean(c[idx, :], axis=0)

    return {
        "custom": _band(custom),
        "theta": _band(theta),
        "delta": _band(delta),
        "spindles": _band(spindles),
        "lowGamma": _band(low_gamma),
        "highGamma": _band(high_gamma),
        "ripples": _band(ripples),
    }


def ccg_parameters(*series_and_groups: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reformat timestamp series for CCG computation."""
    if len(series_and_groups) < 1:
        raise ValueError("Provide at least one series.")

    data_times: list[np.ndarray] = []
    data_ids: list[np.ndarray] = []
    data_groups: list[np.ndarray] = []
    id_offset = 0

    i = 0
    while i < len(series_and_groups):
        s_raw = np.asarray(series_and_groups[i], dtype=float)
        i += 1
        if s_raw.ndim == 1:
            times = s_raw.reshape(-1)
            ids = np.ones(times.shape[0], dtype=int)
        elif s_raw.ndim == 2 and s_raw.shape[1] == 2:
            times = s_raw[:, 0].reshape(-1)
            ids = np.asarray(np.round(s_raw[:, 1]), dtype=int).reshape(-1)
            if np.any(ids <= 0):
                raise ValueError("IDs in two-column series must be strictly positive integers.")
        else:
            raise ValueError("Each series must be 1D timestamps or Nx2 [time, id].")

        group = np.ones(times.shape[0], dtype=int)
        if i < len(series_and_groups):
            g_raw = np.asarray(series_and_groups[i])
            g_is_group = False
            if g_raw.ndim == 0:
                g_is_group = np.isfinite(float(g_raw))
            elif g_raw.ndim == 1 and g_raw.size in (1, times.size):
                g_is_group = np.issubdtype(g_raw.dtype, np.number)
            if g_is_group:
                g_vals = np.asarray(np.round(g_raw), dtype=int).reshape(-1)
                if g_vals.size == 1:
                    group = np.full(times.shape[0], int(g_vals[0]), dtype=int)
                elif g_vals.size == times.size:
                    group = g_vals
                else:
                    raise ValueError("group must be scalar or same length as series.")
                i += 1

        ids = ids + id_offset
        if ids.size:
            id_offset = int(np.max(ids))
        data_times.append(times)
        data_ids.append(ids)
        data_groups.append(group)

    out_times = np.concatenate(data_times) if data_times else np.array([], dtype=float)
    out_ids = np.concatenate(data_ids) if data_ids else np.array([], dtype=int)
    out_groups = np.concatenate(data_groups) if data_groups else np.array([], dtype=int)
    return out_times, out_ids, out_groups


def define_zone(
    size_xy: tuple[int, int] | list[int] | np.ndarray,
    shape: str,
    points: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Define a rectangular or circular boolean zone mask."""
    s = np.asarray(size_xy, dtype=int).reshape(-1)
    if s.size != 2 or np.any(s <= 0):
        raise ValueError("size_xy must be [width, height] with strictly positive values.")
    width, height = int(s[0]), int(s[1])
    zone = np.zeros((height, width), dtype=bool)

    shape_use = str(shape).strip().lower()
    p = np.asarray(points, dtype=float).reshape(-1)
    if shape_use == "rectangle":
        if p.size != 4:
            raise ValueError("rectangle points must be [x, y, width, height].")
        x, y, w, h = np.round(p).astype(int)
        if w <= 0 or h <= 0:
            return zone
        x0 = np.clip(x - 1, 0, width)
        y0 = np.clip(y - 1, 0, height)
        x1 = np.clip(x0 + w, 0, width)
        y1 = np.clip(y0 + h, 0, height)
        zone[y0:y1, x0:x1] = True
        return zone
    if shape_use == "circle":
        if p.size != 3:
            raise ValueError("circle points must be [x, y, radius].")
        x, y, r = p
        if r <= 0:
            return zone
        xs = np.arange(1, width + 1, dtype=float)
        ys = np.arange(1, height + 1, dtype=float)
        xx, yy = np.meshgrid(xs, ys)
        zone = (xx - x) ** 2 + (yy - y) ** 2 <= float(r) ** 2
        return zone
    raise ValueError("shape must be 'rectangle' or 'circle'.")


def _zone_occupancy_mask(positions: np.ndarray | nap.TsdFrame, zone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return timestamps and boolean occupancy mask for each position sample."""
    arr = _as_samples(positions)
    z = np.asarray(zone, dtype=bool)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    if z.ndim != 2:
        raise ValueError("zone must be a boolean vector or matrix.")

    if arr.size == 0:
        return np.array([], dtype=float), np.zeros(0, dtype=bool)
    if np.any(arr[:, 1] < 0) or np.any(arr[:, 1] > 1):
        raise ValueError("X coordinates should contain values in [0, 1].")

    x_bins = z.shape[1]
    x_idx = np.clip(np.floor(arr[:, 1] * x_bins).astype(int), 0, x_bins - 1)

    if z.shape[0] == 1:
        return arr[:, 0], z[0, x_idx]

    if arr.shape[1] < 3:
        raise ValueError("positions must contain y coordinates when zone is 2D.")
    if np.any(arr[:, 2] < 0) or np.any(arr[:, 2] > 1):
        raise ValueError("Y coordinates should contain values in [0, 1].")
    y_bins = z.shape[0]
    y_idx = np.clip(np.floor(arr[:, 2] * y_bins).astype(int), 0, y_bins - 1)
    return arr[:, 0], z[y_idx, x_idx]


def _mask_to_intervalset(times: np.ndarray, mask: np.ndarray) -> nap.IntervalSet:
    t = np.asarray(times, dtype=float).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if t.size == 0 or m.size == 0 or not np.any(m):
        return nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))
    if t.shape[0] != m.shape[0]:
        raise ValueError("times and mask must have same length.")
    edges = np.diff(np.r_[False, m, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return nap.IntervalSet(start=t[starts], end=t[stops])


def is_in_zone(
    positions: np.ndarray | nap.TsdFrame,
    zone: np.ndarray,
    *,
    return_mask: bool = False,
) -> nap.IntervalSet | tuple[nap.IntervalSet, nap.Tsd]:
    """Return in-zone occupancy as an IntervalSet, with optional boolean mask."""
    times, mask = _zone_occupancy_mask(positions, zone)
    intervals = _mask_to_intervalset(times, mask)
    if not return_mask:
        return intervals
    mask_tsd = nap.Tsd(t=times, d=mask.astype(np.int8))
    return intervals, mask_tsd


def compare_distributions(
    group1: np.ndarray,
    group2: np.ndarray,
    *,
    n_shuffles: int = 5000,
    alpha: float = 0.05,
    max_iterations: int = 6,
    tolerance: float = 0.8,
    tail: str = "two",
    random_seed: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Bootstrap comparison of two multivariate distributions."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if g1.ndim == 1:
        g1 = g1.reshape(-1, 1)
    if g2.ndim == 1:
        g2 = g2.reshape(-1, 1)
    if g1.ndim != 2 or g2.ndim != 2:
        raise ValueError("group1 and group2 must be 1D or 2D arrays.")
    if g1.shape[1] != g2.shape[1]:
        raise ValueError("group1 and group2 must have the same number of columns.")
    if g1.shape[0] == 0 or g2.shape[0] == 0:
        raise ValueError("group1 and group2 must be non-empty.")
    if n_shuffles <= 0:
        raise ValueError("n_shuffles must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if max_iterations < 2:
        raise ValueError("max_iterations must be >= 2.")
    if tolerance <= 0:
        raise ValueError("tolerance must be > 0.")

    n1 = g1.shape[0]
    n2 = g2.shape[0]
    n_bins = g1.shape[1]
    is_multidim = n_bins > 1

    combined = np.vstack((g1, g2))
    rng = np.random.default_rng(random_seed)
    diffs = np.empty((n_shuffles, n_bins), dtype=float)
    for i in range(n_shuffles):
        perm = rng.permutation(n1 + n2)
        s1 = combined[perm[:n1], :]
        s2 = combined[perm[n1:], :]
        diffs[i, :] = np.nanmean(s1, axis=0) - np.nanmean(s2, axis=0)

    tail_use = str(tail).lower()
    if tail_use not in ("one", "two"):
        raise ValueError("tail must be 'one' or 'two'.")

    iteration = 0
    deviation = np.inf
    alpha_work = float(alpha)
    alpha_path: list[float] = []
    p_path: list[float] = []
    pointwise: np.ndarray | None = None
    global_ci: np.ndarray | None = None

    while deviation * 100.0 > float(tolerance):
        iteration += 1
        if iteration > int(max_iterations):
            break

        if tail_use == "one":
            quantiles = np.array([1.0 - alpha_work], dtype=float)
            ci = np.quantile(diffs, quantiles, axis=0, method="linear").reshape(1, -1)
            ci = np.vstack((ci, np.full_like(ci, -np.inf)))
        else:
            quantiles = np.array([alpha_work / 2.0, 1.0 - alpha_work / 2.0], dtype=float)
            q = np.quantile(diffs, quantiles, axis=0, method="linear")
            ci = np.vstack((q[1, :], q[0, :]))
        pointwise = ci

        if not is_multidim:
            break

        significant = (diffs > ci[0, :]) | (diffs < ci[1, :])
        p_global = float(np.mean(np.any(significant, axis=1)))
        alpha_path.append(alpha_work)
        p_path.append(p_global)
        global_ci = ci
        deviation = abs(p_global - alpha)
        if iteration == 1:
            alpha_work = 0.003
        else:
            sgn = np.sign(p_global - alpha)
            if deviation > 0.05:
                alpha_work = alpha_work - sgn * deviation * 0.00075
            else:
                alpha_work = alpha_work - sgn * deviation * 0.033
            alpha_work = max(alpha_work, 1e-3)

    observed = np.nanmean(g1, axis=0) - np.nanmean(g2, axis=0)
    stats: dict[str, Any] = {
        "observed": observed,
        "null": np.nanmean(diffs, axis=0),
        "pointwise": pointwise if pointwise is not None else np.full((2, n_bins), np.nan, dtype=float),
        "global": np.array([]),
        "above": np.array([], dtype=bool),
        "below": np.array([], dtype=bool),
        "alpha": np.asarray(alpha_path, dtype=float),
        "p": np.asarray(p_path, dtype=float),
    }

    if not is_multidim:
        h = bool((observed > stats["pointwise"][0, :]) | (observed < stats["pointwise"][1, :]))
        return h, stats

    if global_ci is None:
        global_ci = stats["pointwise"]
    stats["global"] = global_ci
    above_point = observed > stats["pointwise"][0, :]
    below_point = observed < stats["pointwise"][1, :]
    above = (observed > global_ci[0, :]) & above_point
    below = (observed < global_ci[1, :]) & below_point
    stats["above"] = above
    stats["below"] = below
    h = bool(np.any(above) or np.any(below))
    return h, stats


def threshold_spikes(
    amplitudes: np.ndarray,
    factor: float,
    *,
    units: np.ndarray | None = None,
) -> np.ndarray:
    """Apply post-hoc threshold scaling to spike amplitude tuples."""
    amps = np.asarray(amplitudes, dtype=float)
    if amps.ndim != 2 or amps.shape[1] < 4:
        raise ValueError("amplitudes must be an NxM matrix with M >= 4.")
    if not np.isfinite(factor):
        raise ValueError("factor must be finite.")

    all_units = None if units is None else np.asarray(units, dtype=float)
    if all_units is not None and (all_units.ndim != 2 or all_units.shape[1] != 2):
        raise ValueError("units must be an Nx2 matrix [group, cluster].")

    unit_pairs = np.unique(amps[:, 1:3], axis=0)
    out_rows: list[np.ndarray] = []
    for group, cluster in unit_pairs:
        this = amps[(amps[:, 1] == group) & (amps[:, 2] == cluster), :]
        if this.shape[0] == 0:
            continue
        peak = np.nanmax(np.abs(this[:, 3:]), axis=1)
        threshold = float(factor) * float(np.nanmin(peak))
        keep = this[peak >= threshold, :3].copy()
        if keep.shape[0] == 0:
            continue
        if all_units is not None:
            group_rows = all_units[all_units[:, 0] == group, 1]
            new_cluster = int(np.nanmax(group_rows) + 1) if group_rows.size > 0 else 1
            all_units = np.vstack((all_units, np.array([[group, new_cluster]], dtype=float)))
            keep[:, 2] = float(new_cluster)
        out_rows.append(keep)

    if not out_rows:
        return np.empty((0, 3), dtype=float)
    return np.vstack(out_rows)


def AngularVelocity(X: np.ndarray | nap.TsdFrame, smooth: float = 0.0) -> np.ndarray:
    """MATLAB-style alias for :func:`angular_velocity`."""
    return angular_velocity(X, smooth=smooth)


def LinearVelocity(X: np.ndarray | nap.TsdFrame, smooth: float = 0.0) -> np.ndarray:
    """MATLAB-style alias for :func:`linear_velocity`."""
    return linear_velocity(X, smooth=smooth)


def Distance(positions: np.ndarray | nap.TsdFrame, reference: Any, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`distance`."""
    options = _collect_options(args, kwargs)
    position_type = options.pop("type", options.pop("position_type", "linear"))
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return distance(positions, reference, position_type=position_type)


def MovementPeriods(v: np.ndarray, velocity: float, duration: float, brief: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-style alias for :func:`movement_periods`."""
    return movement_periods(v, velocity=velocity, duration=duration, brief=brief)


def QuietPeriods(v: np.ndarray, velocity: float, duration: float, brief: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB-style alias for :func:`quiet_periods`."""
    return quiet_periods(v, velocity=velocity, duration=duration, brief=brief)


def Phase(samples: np.ndarray | nap.Tsd | nap.TsdFrame, times: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB-style alias for :func:`phase`."""
    return phase(samples, times=times)


def Frequency(timestamps: np.ndarray | nap.Ts, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`frequency`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "method": options.pop("method", "fixed"),
        "limits": options.pop("limits", None),
        "bin_size": float(options.pop("binsize", options.pop("bin_size", 0.05))),
        "smooth": float(options.pop("smooth", 2.0)),
    }
    _ = options.pop("show", None)
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return frequency(timestamps, **mapped)


def CV(timestamps: np.ndarray | nap.Ts, *args: Any, **kwargs: Any) -> tuple[float, np.ndarray]:
    """MATLAB-style alias for :func:`cv`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "measure": options.pop("measure", "cv"),
        "order": int(options.pop("order", 1)),
        "method": options.pop("method", "fixed"),
        "bin_size": float(options.pop("binsize", options.pop("bin_size", 0.001))),
        "smooth": float(options.pop("smooth", 25.0)),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return cv(timestamps, **mapped)


def FilterLFP(lfp: np.ndarray | nap.Tsd | nap.TsdFrame, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`filter_lfp`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "passband": options.pop("passband", (4.0, 10.0)),
        "order": int(options.pop("order", 4)),
        "ripple": float(options.pop("ripple", 20.0)),
        "nyquist": options.pop("nyquist", None),
        "filter_type": options.pop("filter", options.pop("filter_type", "cheby2")),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return filter_lfp(lfp, **mapped)


def SpectrogramBands(spectrogram: np.ndarray, frequencies: np.ndarray, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """MATLAB-style alias for :func:`spectrogram_bands`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "smooth": float(options.pop("smooth", 2.0)),
        "delta": tuple(options.pop("delta", (0.0, 4.0))),
        "theta": tuple(options.pop("theta", (7.0, 10.0))),
        "spindles": tuple(options.pop("spindles", (10.0, 20.0))),
        "low_gamma": tuple(options.pop("lowgamma", options.pop("low_gamma", (30.0, 80.0)))),
        "high_gamma": tuple(options.pop("highgamma", options.pop("high_gamma", (80.0, 120.0)))),
        "ripples": tuple(options.pop("ripples", (100.0, 250.0))),
        "broad_low": tuple(options.pop("broadlow", options.pop("broad_low", (1.0, 12.0)))),
        "amy_gamma": tuple(options.pop("amygamma", options.pop("amy_gamma", (45.0, 65.0)))),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return spectrogram_bands(spectrogram, frequencies, **mapped)


def CoherenceBands(coherogram: np.ndarray, frequencies: np.ndarray, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    """MATLAB-style alias for :func:`coherence_bands`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "custom": tuple(options.pop("custom", (0.0, 250.0))),
        "delta": tuple(options.pop("delta", (0.0, 4.0))),
        "theta": tuple(options.pop("theta", (7.0, 10.0))),
        "spindles": tuple(options.pop("spindles", (10.0, 20.0))),
        "low_gamma": tuple(options.pop("lowgamma", options.pop("low_gamma", (30.0, 80.0)))),
        "high_gamma": tuple(options.pop("highgamma", options.pop("high_gamma", (80.0, 120.0)))),
        "ripples": tuple(options.pop("ripples", (100.0, 250.0))),
    }
    _ = options.pop("smooth", None)
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return coherence_bands(coherogram, frequencies, **mapped)


def CCGParameters(*series_and_groups: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB-style alias for :func:`ccg_parameters`."""
    return ccg_parameters(*series_and_groups)


def DefineZone(s: tuple[int, int] | list[int] | np.ndarray, shape: str, points: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`define_zone`."""
    return define_zone(s, shape=shape, points=points)


def IsInZone(
    positions: np.ndarray | nap.TsdFrame,
    zone: np.ndarray,
    *args: Any,
    **kwargs: Any,
) -> nap.IntervalSet | tuple[nap.IntervalSet, nap.Tsd]:
    """MATLAB-style alias for :func:`is_in_zone`."""
    options = _collect_options(args, kwargs)
    return_mask = bool(options.pop("returnmask", options.pop("return_mask", False)))
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return is_in_zone(positions, zone, return_mask=return_mask)


def CompareDistributions(group1: np.ndarray, group2: np.ndarray, *args: Any, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
    """MATLAB-style alias for :func:`compare_distributions`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "n_shuffles": int(options.pop("nshuffles", options.pop("n_shuffles", 5000))),
        "alpha": float(options.pop("alpha", 0.05)),
        "max_iterations": int(options.pop("max", options.pop("max_iterations", 6))),
        "tolerance": float(options.pop("tolerance", 0.8)),
        "tail": options.pop("tail", "two"),
        "random_seed": options.pop("randomseed", options.pop("random_seed", None)),
    }
    _ = options.pop("show", None)
    _ = options.pop("verbose", None)
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return compare_distributions(group1, group2, **mapped)


def ThresholdSpikes(amplitudes: np.ndarray, factor: float, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`threshold_spikes`."""
    options = _collect_options(args, kwargs)
    units = options.pop("units", None)
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return threshold_spikes(amplitudes, factor=factor, units=units)


__all__ = [
    "angular_velocity",
    "linear_velocity",
    "distance",
    "movement_periods",
    "quiet_periods",
    "phase",
    "frequency",
    "cv",
    "filter_lfp",
    "spectrogram_bands",
    "coherence_bands",
    "ccg_parameters",
    "define_zone",
    "is_in_zone",
    "compare_distributions",
    "threshold_spikes",
    "AngularVelocity",
    "LinearVelocity",
    "Distance",
    "MovementPeriods",
    "QuietPeriods",
    "Phase",
    "Frequency",
    "CV",
    "FilterLFP",
    "SpectrogramBands",
    "CoherenceBands",
    "CCGParameters",
    "DefineZone",
    "IsInZone",
    "CompareDistributions",
    "ThresholdSpikes",
]
