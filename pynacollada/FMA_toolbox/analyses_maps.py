"""Spatial map/field analyses inspired by FMAToolbox Analyses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pynapple as nap
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, label


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


def _as_samples(v: np.ndarray | nap.Tsd | nap.TsdFrame) -> np.ndarray:
    if isinstance(v, nap.Tsd):
        t = np.asarray(v.as_units("s").index.values, dtype=float).reshape(-1)
        x = np.asarray(v.values, dtype=float).reshape(-1, 1)
        return np.column_stack((t, x))
    if isinstance(v, nap.TsdFrame):
        t = np.asarray(v.as_units("s").index.values, dtype=float).reshape(-1)
        x = np.asarray(v.values, dtype=float)
        return np.column_stack((t, x))
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("samples must be NxM with at least [time, x].")
    return arr


def _as_timestamps(z: np.ndarray | nap.Ts) -> np.ndarray:
    if isinstance(z, nap.Ts):
        return np.asarray(z.as_units("s").index.values, dtype=float).reshape(-1)
    arr = np.asarray(z, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].reshape(-1)
    raise ValueError("Point-process z must be timestamps (1D).")


def _as_continuous(z: np.ndarray | nap.Tsd | nap.TsdFrame) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(z, nap.Tsd):
        t = np.asarray(z.as_units("s").index.values, dtype=float).reshape(-1)
        val = np.asarray(z.values, dtype=float).reshape(-1)
        return t, val
    if isinstance(z, nap.TsdFrame):
        if z.shape[1] != 1:
            raise ValueError("Continuous z must be 1D (single signal column).")
        t = np.asarray(z.as_units("s").index.values, dtype=float).reshape(-1)
        val = np.asarray(z.values, dtype=float).reshape(-1)
        return t, val
    arr = np.asarray(z, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Continuous z must be Nx2 [time, value].")
    return arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi - lo <= _EPS:
        return np.zeros_like(x)
    if lo < 0.0 or hi > 1.0:
        return (x - lo) / (hi - lo)
    return x


def _bin_indices(x: np.ndarray, n_bins: int) -> np.ndarray:
    idx = np.floor(np.asarray(x, dtype=float) * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def _smooth_1d(values: np.ndarray, sigma: float, circular: bool) -> np.ndarray:
    if sigma <= 0:
        return values.copy()
    mode = "wrap" if circular else "reflect"
    return gaussian_filter1d(values, sigma=float(sigma), mode=mode)


def _smooth_2d(values: np.ndarray, sigma_y: float, sigma_x: float, circ_y: bool, circ_x: bool) -> np.ndarray:
    if sigma_x <= 0 and sigma_y <= 0:
        return values.copy()
    modes = ("wrap" if circ_y else "reflect", "wrap" if circ_x else "reflect")
    return gaussian_filter(values, sigma=(sigma_y, sigma_x), mode=modes)


def _interp_1d(values: np.ndarray, valid: np.ndarray, max_distance: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    good = np.asarray(valid, dtype=bool)
    if np.sum(good) < 2:
        return out
    x = np.arange(values.shape[0], dtype=float)
    filled = np.interp(x, x[good], out[good])

    left = np.full_like(x, np.nan, dtype=float)
    right = np.full_like(x, np.nan, dtype=float)
    good_idx = np.flatnonzero(good)
    for i in range(x.shape[0]):
        l = good_idx[good_idx <= i]
        r = good_idx[good_idx >= i]
        if l.size > 0:
            left[i] = i - l[-1]
        if r.size > 0:
            right[i] = r[0] - i
    dist = np.nanmin(np.column_stack((left, right)), axis=1)
    mask = (~good) & (dist <= float(max_distance))
    out[mask] = filled[mask]
    return out


def _interp_2d(values: np.ndarray, valid: np.ndarray, max_distance: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    good = np.asarray(valid, dtype=bool)
    if np.sum(good) == 0:
        return out
    dist, indices = distance_transform_edt(~good, return_indices=True)
    fill_mask = (~good) & (dist <= float(max_distance))
    nearest_vals = out[indices[0], indices[1]]
    out[fill_mask] = nearest_vals[fill_mask]
    return out


def _count_in_intervals(events: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    ev = np.asarray(events, dtype=float).reshape(-1)
    ev = np.sort(ev)
    i0 = np.searchsorted(ev, starts, side="left")
    i1 = np.searchsorted(ev, ends, side="right")
    return (i1 - i0).astype(float)


def _interpolate_continuous(tz: np.ndarray, zv: np.ndarray, t: np.ndarray, max_gap: float, circular: bool) -> np.ndarray:
    if tz.size == 0:
        return np.full(t.shape, np.nan, dtype=float)
    order = np.argsort(tz)
    tz = tz[order]
    zv = zv[order]

    if circular:
        zr = np.interp(t, tz, np.real(np.exp(1j * zv)), left=np.nan, right=np.nan)
        zi = np.interp(t, tz, np.imag(np.exp(1j * zv)), left=np.nan, right=np.nan)
        interp = np.angle(zr + 1j * zi)
        interp = np.exp(1j * interp)
    else:
        interp = np.interp(t, tz, zv, left=np.nan, right=np.nan)

    j = np.searchsorted(tz, t, side="left")
    left = np.where(j > 0, np.abs(t - tz[np.clip(j - 1, 0, tz.size - 1)]), np.inf)
    right = np.where(j < tz.size, np.abs(t - tz[np.clip(j, 0, tz.size - 1)]), np.inf)
    d = np.minimum(left, right)
    interp[d > float(max_gap)] = np.nan
    return interp


def _accumulate_1d(idx: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
    w = np.asarray(weights)
    if np.iscomplexobj(w):
        r = np.bincount(idx, weights=np.real(w), minlength=n_bins).astype(float)
        im = np.bincount(idx, weights=np.imag(w), minlength=n_bins).astype(float)
        return r + 1j * im
    return np.bincount(idx, weights=w.astype(float), minlength=n_bins).astype(float)


def _accumulate_2d(x_idx: np.ndarray, y_idx: np.ndarray, weights: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
    w = np.asarray(weights)
    out = np.zeros((n_y, n_x), dtype=np.complex128 if np.iscomplexobj(w) else float)
    np.add.at(out, (y_idx, x_idx), w)
    return out


def compute_map(
    variables: np.ndarray | nap.TsdFrame,
    z: np.ndarray | nap.Ts | nap.Tsd | nap.TsdFrame | None,
    *,
    smooth: float | tuple[float, float] = 2.0,
    n_bins: int | tuple[int, int] = 50,
    min_time: float = 0.0,
    mode: str = "discard",
    max_distance: float = 5.0,
    max_gap: float = 0.1,
    sample_type: str = "lll",
) -> dict[str, Any]:
    """Compute occupancy and value/rate maps for 1D or 2D variables in [0,1]."""
    v = _as_samples(variables)
    if v.shape[0] < 2:
        return {"x": np.array([], dtype=float), "y": np.array([], dtype=float), "count": np.array([]), "time": np.array([]), "z": np.array([])}

    t = v[:, 0]
    if np.any(np.diff(t) <= 0):
        raise ValueError("variables time column must be strictly increasing.")

    x = _normalize01(v[:, 1])
    y = _normalize01(v[:, 2]) if v.shape[1] >= 3 else None

    is_point = z is None
    if z is not None:
        if isinstance(z, nap.Ts):
            is_point = True
        else:
            z_arr = np.asarray(z)
            is_point = (z_arr.ndim == 1) or (z_arr.ndim == 2 and z_arr.shape[1] == 1)

    if isinstance(n_bins, int):
        n_x = int(n_bins)
        n_y = int(n_bins)
    else:
        nb = np.asarray(n_bins, dtype=int).reshape(-1)
        if nb.size == 1:
            n_x = int(nb[0])
            n_y = int(nb[0])
        else:
            n_x, n_y = int(nb[0]), int(nb[1])
    if n_x <= 0 or (y is not None and n_y <= 0):
        raise ValueError("n_bins must be positive.")

    x_idx = _bin_indices(x, n_x)
    y_idx = _bin_indices(y, n_y) if y is not None else None

    dt = np.diff(t)
    dt = np.r_[dt, dt[-1]]
    dt = np.minimum(dt, float(max_gap))

    if is_point:
        event_t = _as_timestamps(np.array([], dtype=float) if z is None else z)
        n_per_sample = _count_in_intervals(event_t, t, t + dt)
        z_per_sample: np.ndarray | None = None
    else:
        tz, zv = _as_continuous(z)  # type: ignore[arg-type]
        circ_z = str(sample_type).lower().endswith("c")
        z_interp = _interpolate_continuous(tz, zv, t, max_gap=float(max_gap), circular=circ_z)
        valid = np.isfinite(z_interp)
        n_per_sample = valid.astype(float)
        if circ_z:
            z_per_sample = np.where(valid, z_interp, 0.0 + 0.0j)
        else:
            z_per_sample = np.where(valid, z_interp, 0.0)

    if y is None:
        count = _accumulate_1d(x_idx, n_per_sample, n_x)
        time = _accumulate_1d(x_idx, dt, n_x)
        valid_bins = time > float(min_time)

        z_acc: np.ndarray
        if is_point:
            z_acc = count / np.maximum(time, _EPS)
        else:
            assert z_per_sample is not None
            z_sum = _accumulate_1d(x_idx, z_per_sample, n_x)
            z_acc = z_sum / np.maximum(count, _EPS)
            if np.iscomplexobj(z_acc):
                z_acc = np.angle(z_acc)

        mode_use = str(mode).lower()
        if mode_use == "interpolate":
            count = _interp_1d(count, valid_bins, max_distance)
            time = _interp_1d(time, valid_bins, max_distance)
            z_acc = _interp_1d(np.asarray(z_acc, dtype=float), valid_bins, max_distance)
        elif mode_use != "discard":
            raise ValueError("mode must be 'discard' or 'interpolate'.")

        smooth_arr = np.asarray(smooth, dtype=float).reshape(-1)
        sigma = float(smooth_arr[0]) if smooth_arr.size > 0 else 0.0
        circ_x = str(sample_type).lower().startswith("c")
        count = _smooth_1d(count, sigma=sigma, circular=circ_x)
        time = _smooth_1d(time, sigma=sigma, circular=circ_x)
        z_acc = _smooth_1d(np.asarray(z_acc, dtype=float), sigma=sigma, circular=circ_x)

        if mode_use == "discard":
            z_acc[time <= float(min_time)] = 0.0

        return {
            "x": np.linspace(0.0, 1.0, n_x),
            "y": np.array([], dtype=float),
            "count": count,
            "time": time,
            "z": z_acc,
        }

    # 2D case
    assert y_idx is not None
    count2 = _accumulate_2d(x_idx, y_idx, n_per_sample, n_x, n_y)
    time2 = _accumulate_2d(x_idx, y_idx, dt, n_x, n_y)
    valid2 = time2 > float(min_time)

    if is_point:
        z2 = count2 / np.maximum(time2, _EPS)
    else:
        assert z_per_sample is not None
        z_sum2 = _accumulate_2d(x_idx, y_idx, z_per_sample, n_x, n_y)
        z2 = z_sum2 / np.maximum(count2, _EPS)
        if np.iscomplexobj(z2):
            z2 = np.angle(z2)

    mode_use = str(mode).lower()
    if mode_use == "interpolate":
        count2 = _interp_2d(count2, valid2, max_distance)
        time2 = _interp_2d(time2, valid2, max_distance)
        z2 = _interp_2d(np.asarray(z2, dtype=float), valid2, max_distance)
    elif mode_use != "discard":
        raise ValueError("mode must be 'discard' or 'interpolate'.")

    smooth_arr = np.asarray(smooth, dtype=float).reshape(-1)
    if smooth_arr.size == 1:
        sigma_x = sigma_y = float(smooth_arr[0])
    elif smooth_arr.size >= 2:
        sigma_x = float(smooth_arr[0])
        sigma_y = float(smooth_arr[1])
    else:
        sigma_x = sigma_y = 0.0

    stype = str(sample_type).lower()
    circ_x = len(stype) >= 1 and stype[0] == "c"
    circ_y = len(stype) >= 2 and stype[1] == "c"

    count2 = _smooth_2d(count2, sigma_y=sigma_y, sigma_x=sigma_x, circ_y=circ_y, circ_x=circ_x)
    time2 = _smooth_2d(time2, sigma_y=sigma_y, sigma_x=sigma_x, circ_y=circ_y, circ_x=circ_x)
    z2 = _smooth_2d(np.asarray(z2, dtype=float), sigma_y=sigma_y, sigma_x=sigma_x, circ_y=circ_y, circ_x=circ_x)

    if mode_use == "discard":
        z2[time2 <= float(min_time)] = 0.0

    return {
        "x": np.linspace(0.0, 1.0, n_x),
        "y": np.linspace(0.0, 1.0, n_y),
        "count": count2,
        "time": time2,
        "z": z2,
    }


def _find_connected_field(mask: np.ndarray, y: int, x: int, circ_x: bool, circ_y: bool) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    ny, nx = m.shape
    if ny == 0 or nx == 0 or not m[y, x]:
        return np.zeros_like(m, dtype=bool)

    if not circ_x and not circ_y:
        labeled, _ = label(m)
        field_id = labeled[y, x]
        return labeled == field_id

    tile_y = 3 if circ_y else 1
    tile_x = 3 if circ_x else 1
    mt = np.tile(m, (tile_y, tile_x))
    y0 = y + (ny if circ_y else 0)
    x0 = x + (nx if circ_x else 0)
    labeled, _ = label(mt)
    field_id = labeled[y0, x0]
    comp = labeled == field_id

    out = np.zeros_like(m, dtype=bool)
    for iy in range(tile_y):
        for ix in range(tile_x):
            out |= comp[iy * ny : (iy + 1) * ny, ix * nx : (ix + 1) * nx]
    return out


def _field_boundaries(field: np.ndarray, circ_x: bool, circ_y: bool) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(field, dtype=bool)
    x_any = np.flatnonzero(np.any(f, axis=0))
    y_any = np.flatnonzero(np.any(f, axis=1))
    if x_any.size == 0:
        x_bounds = np.array([np.nan, np.nan], dtype=float)
    else:
        x_bounds = np.array([x_any[0], x_any[-1]], dtype=float)
    if y_any.size == 0:
        y_bounds = np.array([np.nan, np.nan], dtype=float)
    else:
        y_bounds = np.array([y_any[0], y_any[-1]], dtype=float)

    if circ_x and x_any.size > 0 and x_any[0] == 0 and x_any[-1] == f.shape[1] - 1:
        gaps = np.flatnonzero(~np.any(f, axis=0))
        if gaps.size > 0:
            x_bounds = np.array([gaps[-1], gaps[0]], dtype=float)
    if circ_y and y_any.size > 0 and y_any[0] == 0 and y_any[-1] == f.shape[0] - 1:
        gaps = np.flatnonzero(~np.any(f, axis=1))
        if gaps.size > 0:
            y_bounds = np.array([gaps[-1], gaps[0]], dtype=float)
    return x_bounds, y_bounds


def map_stats(
    map_data: dict[str, Any],
    *,
    threshold: float = 0.2,
    min_size: float | None = None,
    min_peak: float = 1.0,
    sample_type: str = "ll",
) -> dict[str, Any]:
    """Compute field and specificity statistics for map/curve outputs."""
    if "z" in map_data:
        z = np.asarray(map_data["z"], dtype=float)
    elif "rate" in map_data:
        z = np.asarray(map_data["rate"], dtype=float)
    else:
        raise ValueError("map_data must contain 'z' or 'rate'.")
    count = np.asarray(map_data.get("count", np.zeros_like(z)), dtype=float)
    time = np.asarray(map_data.get("time", np.zeros_like(z)), dtype=float)

    if z.ndim == 1:
        n_dims = 1
        z2 = z.reshape(1, -1)
        count2 = count.reshape(1, -1)
        time2 = time.reshape(1, -1)
    elif z.ndim == 2:
        n_dims = 2
        z2 = z
        count2 = count
        time2 = time
    else:
        raise ValueError("map_data['z'] must be 1D or 2D.")

    stype = str(sample_type).lower()
    circ_x = (stype[0] == "c") if len(stype) >= 1 else False
    circ_y = (stype[1] == "c") if len(stype) >= 2 else False

    if min_size is None:
        min_size = 100.0 if n_dims == 2 else 10.0

    stats: dict[str, Any] = {
        "x": np.array([], dtype=float),
        "y": np.array([], dtype=float),
        "field": np.zeros(z2.shape + (0,), dtype=bool),
        "size": np.array([], dtype=float),
        "peak": np.array([], dtype=float),
        "mean": np.array([], dtype=float),
        "fieldX": np.empty((0, 2), dtype=float),
        "fieldY": np.empty((0, 2), dtype=float),
        "specificity": 0.0,
        "m": np.nan,
        "r": np.nan,
        "mode": np.nan,
        "k": np.nan,
    }

    total_time = float(np.nansum(time2))
    if total_time > 0:
        occupancy = time2 / (total_time + _EPS)
        mean_rate = np.nansum(count2) / max(np.nansum(time2), _EPS)
        if mean_rate > 0:
            log_arg = count2 / mean_rate
            log_arg[log_arg <= 1.0] = 1.0
            stats["specificity"] = float(np.nansum(count2 * np.log2(log_arg) * occupancy) / mean_rate)

    if np.nanmax(z2) <= 0:
        stats["field"] = np.zeros(z2.shape + (0,), dtype=bool)
        return stats

    z_work = z2.copy()
    fields: list[np.ndarray] = []
    sizes: list[float] = []
    peaks: list[float] = []
    means: list[float] = []
    x_peaks: list[float] = []
    y_peaks: list[float] = []
    bounds_x: list[np.ndarray] = []
    bounds_y: list[np.ndarray] = []

    while True:
        if np.all(np.isnan(z_work)):
            break
        idx = int(np.nanargmax(z_work))
        peak = float(np.nanmax(z_work))
        if peak < float(min_peak):
            break
        y0, x0 = np.unravel_index(idx, z_work.shape)
        mask = np.asarray(z_work >= peak * float(threshold), dtype=bool)
        field = _find_connected_field(mask, y0, x0, circ_x=circ_x and z_work.shape[1] > 1, circ_y=circ_y and z_work.shape[0] > 1)
        field_size = float(np.sum(field))

        if field_size > float(min_size):
            fields.append(field)
            sizes.append(field_size)
            peaks.append(peak)
            means.append(float(np.nanmean(z_work[field])))
            y_peak, x_peak = np.unravel_index(np.nanargmax(np.where(field, z_work, np.nan)), z_work.shape)
            x_peaks.append(float(x_peak))
            y_peaks.append(float(y_peak))
            bx, by = _field_boundaries(field, circ_x=circ_x, circ_y=circ_y)
            bounds_x.append(bx)
            bounds_y.append(by)

        z_work[field] = np.nan

    if fields:
        field_stack = np.stack(fields, axis=-1)
        stats["field"] = field_stack if n_dims == 2 else field_stack.reshape(field_stack.shape[1], field_stack.shape[2])
        stats["size"] = np.asarray(sizes, dtype=float)
        stats["peak"] = np.asarray(peaks, dtype=float)
        stats["mean"] = np.asarray(means, dtype=float)
        stats["x"] = np.asarray(x_peaks, dtype=float)
        stats["y"] = np.asarray(y_peaks, dtype=float)
        stats["fieldX"] = np.vstack(bounds_x)
        stats["fieldY"] = np.vstack(bounds_y)
    else:
        stats["field"] = np.zeros(z2.shape + (0,), dtype=bool)

    if n_dims == 1 and circ_x:
        x = np.asarray(map_data.get("x", np.linspace(0.0, 1.0, z.shape[0])), dtype=float).reshape(-1)
        z1 = np.asarray(z, dtype=float).reshape(-1)
        denom = np.nansum(z1)
        if denom > 0:
            c = z1 * np.exp(1j * x * 2.0 * np.pi) / denom
            m = np.angle(np.nanmean(c))
            r = np.abs(np.nansum(c))
            mode_i = int(np.nanargmax(z1))
            mode = float(x[mode_i] * 2.0 * np.pi)
            if r < 0.53:
                k = 2 * r + r**3 + 5 * r**5 / 6
            elif r < 0.85:
                k = -0.4 + 1.39 * r + 0.43 / max(1 - r, _EPS)
            else:
                k = 1 / max(r**3 - 4 * r**2 + 3 * r, _EPS)
            stats["m"] = float(m)
            stats["r"] = float(r)
            stats["mode"] = mode
            stats["k"] = float(k)

    return stats


def firing_map(
    positions: np.ndarray | nap.TsdFrame,
    spikes: np.ndarray | nap.Ts,
    *,
    return_stats: bool = True,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Compute firing map as a convenience wrapper around :func:`compute_map`."""
    opts = dict(kwargs)
    stats_opts = {
        "threshold": float(opts.pop("threshold", 0.2)),
        "min_size": opts.pop("min_size", opts.pop("minsize", None)),
        "min_peak": float(opts.pop("min_peak", opts.pop("minpeak", 1.0))),
    }
    stype = str(opts.pop("sample_type", opts.pop("type", "ll")))
    out = compute_map(positions, spikes, sample_type=f"{stype[:2]}l", **opts)
    out["rate"] = out.pop("z")
    if not return_stats:
        return out
    stats = map_stats(out, sample_type=stype, **stats_opts)
    return out, stats


def firing_curve(
    samples: np.ndarray | nap.Tsd,
    spikes: np.ndarray | nap.Ts,
    *,
    curve_type: str = "linear",
    return_stats: bool = True,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Compute firing curve for 1D linear or circular variables."""
    opts = dict(kwargs)
    stats_opts = {
        "threshold": float(opts.pop("threshold", 0.2)),
        "min_size": opts.pop("min_size", opts.pop("minsize", None)),
        "min_peak": float(opts.pop("min_peak", opts.pop("minpeak", 1.0))),
    }
    st = "cl" if str(curve_type).lower().startswith("c") else "ll"
    out = compute_map(samples, spikes, sample_type=st, **opts)
    out["rate"] = out.pop("z")
    if not return_stats:
        return out
    stats = map_stats(out, sample_type=st[0], **stats_opts)
    return out, stats


def normalize_fields(fields: np.ndarray, *, normalize_rate: bool = True) -> np.ndarray:
    """Normalize 1D field profiles in space and optionally in rate."""
    arr = np.asarray(fields, dtype=float)
    if arr.ndim != 2:
        raise ValueError("fields must be an MxN matrix.")
    m, n = arr.shape
    if n == 0:
        return arr.copy()

    is_field = arr > 0
    trans_start = np.diff(np.column_stack((np.zeros(m, dtype=bool), is_field)), axis=1)
    start = np.ones(m, dtype=int)
    rows, cols = np.where(trans_start == 1)
    start[rows] = cols + 1

    aligned = np.empty_like(arr)
    for i in range(m):
        aligned[i] = np.roll(arr[i], -(start[i] - 1))

    is_field_aligned = aligned > 0
    trans_stop = np.diff(np.column_stack((np.zeros(m, dtype=bool), is_field_aligned, np.zeros(m, dtype=bool))), axis=1)
    stop = np.full(m, n, dtype=int)
    r2, c2 = np.where(trans_stop == -1)
    stop[r2] = c2

    normalized = np.zeros_like(arr)
    target = np.linspace(1, 1, n)
    for i in range(m):
        s = int(np.clip(stop[i], 1, n))
        if s == 1:
            normalized[i, :] = aligned[i, 0]
            continue
        src_x = np.linspace(1, s, s)
        dst_x = np.linspace(1, s, n)
        normalized[i, :] = np.interp(dst_x, src_x, aligned[i, :s])

    if normalize_rate:
        mx = np.nanmax(normalized, axis=1, keepdims=True)
        mx[mx <= 0] = 1.0
        normalized = normalized / mx
    return normalized


def find_field_helper(map_values: np.ndarray, x: int, y: int, threshold: float, circ_x: bool, circ_y: bool) -> np.ndarray:
    """Return connected field containing (x, y) above threshold."""
    z = np.asarray(map_values, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    yy = int(y)
    xx = int(x)
    if yy < 0 or yy >= z.shape[0] or xx < 0 or xx >= z.shape[1]:
        raise ValueError("x/y out of bounds.")
    mask = z >= float(threshold)
    return _find_connected_field(mask, yy, xx, circ_x=bool(circ_x), circ_y=bool(circ_y))


def Map(v: np.ndarray | nap.TsdFrame, z: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """MATLAB-compatible alias for :func:`compute_map`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "smooth": options.pop("smooth", 2.0),
        "n_bins": options.pop("nbins", options.pop("n_bins", 50)),
        "min_time": float(options.pop("mintime", options.pop("min_time", 0.0))),
        "mode": options.pop("mode", "discard"),
        "max_distance": float(options.pop("maxdistance", options.pop("max_distance", 5.0))),
        "max_gap": float(options.pop("maxgap", options.pop("max_gap", 0.1))),
        "sample_type": options.pop("type", options.pop("sample_type", "lll")),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return compute_map(v, z, **mapped)


def bz_Map(v: np.ndarray | nap.TsdFrame, z: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the buzcode variant of Map."""
    return Map(v, z, *args, **kwargs)


def MapStats(map_obj: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """MATLAB-compatible alias for :func:`map_stats`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "threshold": float(options.pop("threshold", 0.2)),
        "min_size": options.pop("minsize", options.pop("min_size", None)),
        "min_peak": float(options.pop("minpeak", options.pop("min_peak", 1.0))),
        "sample_type": options.pop("type", options.pop("sample_type", "ll")),
    }
    _ = options.pop("verbose", None)
    _ = options.pop("debug", None)
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return map_stats(map_obj, **mapped)


def FiringMap(positions: np.ndarray | nap.TsdFrame, spikes: np.ndarray | nap.Ts, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """MATLAB-compatible alias for :func:`firing_map`."""
    options = _collect_options(args, kwargs)

    map_options: dict[str, Any] = {}
    stats_options: dict[str, Any] = {}
    for k, v in options.items():
        if k in {"threshold", "minsize", "min_size", "minpeak", "min_peak", "verbose", "debug", "type"}:
            stats_options[k] = v
        else:
            map_options[k] = v

    map_options["sample_type"] = f"{str(stats_options.get('type', 'll'))[:2]}l"
    out = compute_map(positions, spikes, **{
        "smooth": map_options.get("smooth", 2.0),
        "n_bins": map_options.get("nbins", map_options.get("n_bins", (50, 50))),
        "min_time": float(map_options.get("mintime", map_options.get("min_time", 0.0))),
        "mode": map_options.get("mode", "discard"),
        "max_distance": float(map_options.get("maxdistance", map_options.get("max_distance", 5.0))),
        "max_gap": float(map_options.get("maxgap", map_options.get("max_gap", 0.1))),
        "sample_type": map_options["sample_type"],
    })
    out["rate"] = out.pop("z")
    stats = map_stats(
        out,
        threshold=float(stats_options.get("threshold", 0.2)),
        min_size=stats_options.get("minsize", stats_options.get("min_size", None)),
        min_peak=float(stats_options.get("minpeak", stats_options.get("min_peak", 1.0))),
        sample_type=str(stats_options.get("type", "ll")),
    )
    return out, stats


def FiringCurve(samples: np.ndarray | nap.Tsd, spikes: np.ndarray | nap.Ts, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """MATLAB-compatible alias for :func:`firing_curve`."""
    options = _collect_options(args, kwargs)
    curve_type = str(options.pop("type", "linear"))
    out = compute_map(
        samples,
        spikes,
        smooth=options.pop("smooth", 2.0),
        n_bins=options.pop("nbins", options.pop("n_bins", 50)),
        min_time=float(options.pop("mintime", options.pop("min_time", 0.0))),
        mode=options.pop("mode", "discard"),
        max_distance=float(options.pop("maxdistance", options.pop("max_distance", 5.0))),
        max_gap=float(options.pop("maxgap", options.pop("max_gap", 0.1))),
        sample_type=("cl" if curve_type.lower().startswith("c") else "ll"),
    )
    out["rate"] = out.pop("z")
    stats = map_stats(
        out,
        threshold=float(options.pop("threshold", 0.2)),
        min_size=options.pop("minsize", options.pop("min_size", None)),
        min_peak=float(options.pop("minpeak", options.pop("min_peak", 1.0))),
        sample_type=("c" if curve_type.lower().startswith("c") else "l"),
    )
    return out, stats


def NormalizeFields(fields: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-compatible alias for :func:`normalize_fields`."""
    options = _collect_options(args, kwargs)
    rate_opt = str(options.pop("rate", "on")).lower()
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return normalize_fields(fields, normalize_rate=(rate_opt != "off"))


def FindFieldHelper(map_values: np.ndarray, x: int, y: int, threshold: float, circX: bool, circY: bool) -> np.ndarray:
    """MATLAB-compatible helper wrapper using 1-based x/y indices."""
    return find_field_helper(map_values, x=int(x) - 1, y=int(y) - 1, threshold=threshold, circ_x=bool(circX), circ_y=bool(circY))


__all__ = [
    "compute_map",
    "map_stats",
    "firing_map",
    "firing_curve",
    "normalize_fields",
    "find_field_helper",
    "Map",
    "MapStats",
    "FiringMap",
    "FiringCurve",
    "NormalizeFields",
    "FindFieldHelper",
    "bz_Map",
]
