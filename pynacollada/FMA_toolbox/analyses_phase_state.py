"""Phase and brain-state analyses inspired by FMAToolbox Analyses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .analyses_core import spectrogram_bands
from .analyses_maps import compute_map


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


def _normalize_map_options(options: dict[str, Any]) -> dict[str, Any]:
    opts = dict(options)
    mapped = {
        "smooth": opts.pop("smooth", 2.0),
        "n_bins": opts.pop("nbins", opts.pop("n_bins", 50)),
        "min_time": float(opts.pop("mintime", opts.pop("min_time", 0.0))),
        "mode": opts.pop("mode", "discard"),
        "max_distance": float(opts.pop("maxdistance", opts.pop("max_distance", 5.0))),
        "max_gap": float(opts.pop("maxgap", opts.pop("max_gap", 0.1))),
    }
    # Keep any additional compute_map-compatible options.
    mapped.update(opts)
    return mapped


def _as_time_series(x: np.ndarray | nap.Tsd) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(x, nap.Tsd):
        t = np.asarray(x.as_units("s").index.values, dtype=float).reshape(-1)
        d = np.asarray(x.values, dtype=float).reshape(-1)
        return t, d
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return np.arange(arr.shape[0], dtype=float), arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)
    raise ValueError("Input must be Tsd, 1D array, or Nx2 [time, value].")


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
        raise ValueError("samples must be NxM with at least [time, x].")
    return arr


def _coerce_phase_signal(phases: np.ndarray | nap.Tsd, reference_times: np.ndarray | None = None) -> np.ndarray:
    if isinstance(phases, nap.Tsd):
        t = np.asarray(phases.as_units("s").index.values, dtype=float).reshape(-1)
        p = np.asarray(phases.values, dtype=float).reshape(-1)
        return np.column_stack((t, p))

    arr = np.asarray(phases, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 1:
        if reference_times is None or reference_times.shape[0] != arr.shape[0]:
            raise ValueError("1D phase arrays require reference_times with matching length.")
        return np.column_stack((reference_times, arr))
    raise ValueError("phases must be 1D, Nx2 [time, phase], or Tsd.")


def _interp_to(times: np.ndarray, signal_t: np.ndarray, signal_v: np.ndarray) -> np.ndarray:
    return np.interp(times, signal_t, signal_v, left=np.nan, right=np.nan)


def phase_distribution(
    phases: np.ndarray | nap.Tsd,
    *,
    n_bins: int = 100,
    smooth: float = 0.0,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compute one or more phase distributions and circular summary statistics."""
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0.")
    _, p = _as_time_series(phases)
    p = np.mod(p, 2.0 * np.pi).reshape(-1)

    if groups is None:
        g = np.ones((p.shape[0], 1), dtype=bool)
    else:
        gr = np.asarray(groups)
        if gr.ndim == 1:
            ids = np.asarray(gr, dtype=int).reshape(-1)
            if ids.shape[0] != p.shape[0]:
                raise ValueError("groups vector must match number of phases.")
            n_groups = int(np.max(ids))
            g = np.zeros((p.shape[0], n_groups), dtype=bool)
            for i in range(1, n_groups + 1):
                g[ids == i, i - 1] = True
        elif gr.ndim == 2:
            if gr.shape[0] != p.shape[0]:
                raise ValueError("groups matrix must have one row per phase.")
            g = np.asarray(gr, dtype=bool)
        else:
            raise ValueError("groups must be vector or boolean matrix.")

    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    angles = edges[:-1] + 0.5 * (edges[1] - edges[0])

    n_groups = g.shape[1]
    dist = np.zeros((n_bins, n_groups), dtype=float)
    mean_angle = np.full(n_groups, np.nan, dtype=float)
    mode = np.full(n_groups, np.nan, dtype=float)
    r = np.full(n_groups, np.nan, dtype=float)
    k = np.full(n_groups, np.nan, dtype=float)
    p_rayleigh = np.full(n_groups, np.nan, dtype=float)

    for i in range(n_groups):
        ph = p[g[:, i]]
        if ph.size == 0:
            continue
        h, _ = np.histogram(ph, bins=edges)
        h = h.astype(float)
        if smooth > 0:
            h = gaussian_filter1d(h, sigma=float(smooth), mode="wrap")
        if np.sum(h) > 0:
            h = h / np.sum(h)
        dist[:, i] = h

        c = np.mean(np.exp(1j * ph))
        mean_angle[i] = float(np.angle(c) % (2.0 * np.pi))
        r_i = float(np.abs(c))
        r[i] = r_i

        if r_i < 0.53:
            k_i = 2 * r_i + r_i**3 + 5 * r_i**5 / 6
        elif r_i < 0.85:
            k_i = -0.4 + 1.39 * r_i + 0.43 / max(1 - r_i, _EPS)
        else:
            k_i = 1 / max(r_i**3 - 4 * r_i**2 + 3 * r_i, _EPS)
        k[i] = float(k_i)

        n = float(ph.size)
        R = r_i * n
        p_rayleigh[i] = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n)))

        mode_idx = int(np.argmax(h))
        mode[i] = float(angles[mode_idx])

    stats = {"m": mean_angle, "mode": mode, "r": r, "k": k, "p": p_rayleigh}
    return dist, angles, stats


def phase_curve(samples: np.ndarray | nap.Tsd, phases: np.ndarray | nap.Tsd, **kwargs: Any) -> dict[str, Any]:
    """Compute 1D phase curve and occupancy/count curves."""
    s = _as_samples(samples)
    phase_signal = _coerce_phase_signal(phases, reference_times=s[:, 0])

    opts = dict(kwargs)
    curve_type = str(opts.pop("curve_type", opts.pop("type", "linear"))).lower()
    x_type = "c" if curve_type.startswith("c") else "l"
    sample_type = f"{x_type}c"

    out = compute_map(s[:, :2], phase_signal, sample_type=sample_type, **_normalize_map_options(opts))
    out["phase"] = out.pop("z")
    return out


def phase_map(positions: np.ndarray | nap.TsdFrame, phases: np.ndarray | nap.Tsd, **kwargs: Any) -> dict[str, Any]:
    """Compute 2D phase map and occupancy/count maps."""
    p = _as_samples(positions)
    phase_signal = _coerce_phase_signal(phases, reference_times=p[:, 0])

    opts = dict(kwargs)
    sample_type = str(opts.pop("sample_type", opts.pop("type", "llc")))
    if len(sample_type) < 3:
        sample_type = f"{sample_type}c"
    out = compute_map(p, phase_signal, sample_type=sample_type, **_normalize_map_options(opts))
    out["phase"] = out.pop("z")
    return out


def brain_states(
    spectrogram: np.ndarray,
    times: np.ndarray,
    frequencies: np.ndarray,
    quiescence: np.ndarray | nap.Tsd,
    emg: np.ndarray | nap.Tsd | None = None,
    *,
    n_clusters: int = 2,
    method: str = "hippocampus",
    n_components: int = 0,
    random_state: int = 0,
) -> dict[str, Any]:
    """Classify exploration/SWS/REM states from spectrogram, quiescence, and EMG."""
    s = np.asarray(spectrogram, dtype=float)
    t = np.asarray(times, dtype=float).reshape(-1)
    f = np.asarray(frequencies, dtype=float).reshape(-1)
    if s.ndim != 2:
        raise ValueError("spectrogram must be 2D [frequency x time].")
    if s.shape[0] != f.shape[0] or s.shape[1] != t.shape[0]:
        raise ValueError("spectrogram dimensions must match frequencies and times.")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0.")

    method_use = str(method).lower()
    if method_use in ("direct",):
        method_use = "hippocampus"
    if method_use in ("ratios",):
        method_use = "cortex"
    if method_use not in {"pca", "hippocampus", "cortex", "amygdala"}:
        raise ValueError("method must be one of {'pca','hippocampus','cortex','amygdala'}.")

    q_t, q_v = _as_time_series(quiescence)
    q0 = _interp_to(t, q_t, q_v)
    q0 = np.where(np.isfinite(q0), (q0 > 0.5).astype(float), np.nan)

    emg0 = None
    if emg is not None:
        e_t, e_v = _as_time_series(emg)
        e_i = _interp_to(t, e_t, np.abs(e_v))
        emg0 = gaussian_filter1d(np.nan_to_num(e_i, nan=np.nanmedian(e_i[np.isfinite(e_i)]) if np.any(np.isfinite(e_i)) else 0.0), sigma=5)

    usable = np.isfinite(q0)
    exploration = np.zeros_like(t, dtype=bool)
    exploration[usable] = q0[usable] < 0.5

    sleep = np.zeros_like(t, dtype=bool)
    sleep[usable] = q0[usable] >= 0.5

    bands = spectrogram_bands(s, f, times=t, as_array=True)
    if method_use == "pca":
        S = s[:, sleep].T
        S = S.copy()
        S[:, f > 30.0] = 0.0
        pca = PCA()
        proj = pca.fit_transform(S)
        if n_components <= 0:
            csum = np.cumsum(pca.explained_variance_ratio_)
            n_components_use = int(np.searchsorted(csum, 0.85) + 1)
        else:
            n_components_use = int(n_components)
        n_components_use = max(1, min(n_components_use, proj.shape[1]))
        features = proj[:, :n_components_use]
    elif method_use == "hippocampus":
        features = np.asarray(bands["ratios"]["hippocampus"], dtype=float)[sleep].reshape(-1, 1)
    elif method_use == "cortex":
        features = np.asarray(bands["ratios"]["cortex"], dtype=float)[sleep]
    else:
        features = np.asarray(bands["ratios"]["amygdala"], dtype=float)[sleep].reshape(-1, 1)

    if emg0 is not None:
        features = np.column_stack((features, emg0[sleep]))

    mu = np.nanmean(features, axis=0)
    sd = np.nanstd(features, axis=0)
    sd[sd <= 0] = 1.0
    features = (features - mu) / sd

    kmeans = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(random_state))
    cluster = kmeans.fit_predict(features)

    ratio_ref = (
        np.asarray(bands["ratios"]["amygdala"], dtype=float)
        if method_use == "amygdala"
        else np.asarray(bands["ratios"]["hippocampus"], dtype=float)
    )
    ratio_sleep = ratio_ref[sleep]

    mean_ratio = np.array(
        [np.nanmean(ratio_sleep[cluster == i]) if np.any(cluster == i) else np.nan for i in range(int(n_clusters))],
        dtype=float,
    )

    rem_idx = int(np.nanargmax(mean_ratio))
    sws_idx = int(np.nanargmin(mean_ratio))

    sws = np.zeros_like(t, dtype=bool)
    rem = np.zeros_like(t, dtype=bool)
    sleep_idx = np.flatnonzero(sleep)
    sws[sleep_idx] = cluster == sws_idx
    rem[sleep_idx] = cluster == rem_idx

    highest = float(mean_ratio[rem_idx]) if np.isfinite(mean_ratio[rem_idx]) else 0.0
    lowest = float(mean_ratio[sws_idx]) if np.isfinite(mean_ratio[sws_idx]) else 0.0
    if highest < 2.0 * max(lowest, _EPS):
        sws[sleep] = True
        rem[sleep] = False

    return {
        "exploration": nap.Tsd(t=t, d=exploration.astype(np.int8)),
        "sws": nap.Tsd(t=t, d=sws.astype(np.int8)),
        "rem": nap.Tsd(t=t, d=rem.astype(np.int8)),
        "cluster": cluster,
        "sleep_mask": sleep,
        "mean_ratio": mean_ratio,
    }


def PhaseDistribution(phases: np.ndarray | nap.Tsd, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """MATLAB-compatible alias for :func:`phase_distribution`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "n_bins": int(options.pop("nbins", options.pop("n_bins", 100))),
        "smooth": float(options.pop("smooth", 0.0)),
        "groups": options.pop("groups", None),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return phase_distribution(phases, **mapped)


def PhaseCurve(samples: np.ndarray | nap.Tsd, phases: np.ndarray | nap.Tsd, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """MATLAB-compatible alias for :func:`phase_curve`."""
    options = _collect_options(args, kwargs)
    if "type" in options and "curve_type" not in options:
        options["curve_type"] = options.pop("type")
    return phase_curve(samples, phases, **options)


def PhaseMap(positions: np.ndarray | nap.TsdFrame, phases: np.ndarray | nap.Tsd, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """MATLAB-compatible alias for :func:`phase_map`."""
    options = _collect_options(args, kwargs)
    return phase_map(positions, phases, **options)


def BrainStates(
    s: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    q: np.ndarray | nap.Tsd,
    emg: np.ndarray | nap.Tsd | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB-compatible alias for :func:`brain_states`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "n_clusters": int(options.pop("nclusters", options.pop("n_clusters", 2))),
        "method": options.pop("method", "hippocampus"),
        "n_components": int(options.pop("ncomponents", options.pop("n_components", 0))),
        "random_state": int(options.pop("randomstate", options.pop("random_state", 0))),
    }
    _ = options.pop("show", None)
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")

    out = brain_states(s, t, f, q, emg, **mapped)
    tt = np.asarray(t, dtype=float).reshape(-1)
    exploration = np.column_stack((tt, np.asarray(out["exploration"].values, dtype=float)))
    sws = np.column_stack((tt, np.asarray(out["sws"].values, dtype=float)))
    rem = np.column_stack((tt, np.asarray(out["rem"].values, dtype=float)))
    return exploration, sws, rem


__all__ = [
    "phase_distribution",
    "phase_curve",
    "phase_map",
    "brain_states",
    "PhaseDistribution",
    "PhaseCurve",
    "PhaseMap",
    "BrainStates",
]
