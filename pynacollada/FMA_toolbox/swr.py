"""Sharp-wave ripple detection and ripple event analytics for pynacollada."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.signal import filtfilt, hilbert, lfilter, savgol_filter, welch
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

from ..archive.eeg_processing.eeg_processing import (
    bandpass_filter as _archive_bandpass_filter,
)
from ..archive.eeg_processing.eeg_processing import (
    detect_oscillatory_events as _archive_detect_oscillatory_events,
)

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - runtime fallback when numba is unavailable
    _HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _decorator(func: Any) -> Any:
            return func

        return _decorator


_EPS = 1e-12


@dataclass(frozen=True)
class SWRDetectorParams:
    """Parameters for the John Long style SWR detector."""

    sw_bp: tuple[float, float] = (2.0, 50.0)
    rip_bp: tuple[float, float] = (80.0, 250.0)
    per_thres_swd: float = 10.0
    per_thres_rip: float = 50.0
    win_size_ms: float = 200.0
    ns_chk_s: float = 5.0
    thres_sd_swd: tuple[float, float] = (0.5, 2.5)
    thres_sd_rip: tuple[float, float] = (0.5, 2.5)
    min_isi_s: float = 0.10
    min_dur_sw_s: float = 0.02
    max_dur_sw_s: float = 0.50
    min_dur_rp_s: float = 0.025


@dataclass(frozen=True)
class FMATRippleDetectorParams:
    """Parameters for FMAT/buzcode-style NSS ripple detection."""

    thresholds: tuple[float, float] = (2.0, 5.0)
    durations_ms: tuple[float, float] = (30.0, 100.0)
    min_duration_ms: float = 20.0
    passband: tuple[float, float] = (130.0, 200.0)
    smooth_window_samples: int = 11
    filter_order: int = 3


def _gaussian_lowpass_kernel(fc: float, fs: float, support_sd: float = 6.0) -> np.ndarray:
    """Gaussian LP FIR kernel equivalent to `makegausslpfir` behavior."""
    fc = float(fc)
    fs = float(fs)
    if fc <= 0 or fs <= 0:
        raise ValueError("fc and fs must be positive.")
    support_sd = max(float(support_sd), 3.0)
    sd = fs / (2.0 * np.pi * fc)
    x = np.arange(-int(np.ceil(support_sd * sd)), int(np.ceil(support_sd * sd)) + 1, dtype=float)
    gwin = (1.0 / (2.0 * np.pi * sd)) * np.exp(-(x**2) / (2.0 * sd**2))
    return gwin / np.sum(gwin)


def _fir_mirror_filter(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FIR filter with mirrored boundaries and centered trimming (`firfilt` style)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
        flatten = True
    elif arr.ndim == 2:
        flatten = False
    else:
        raise ValueError("x must be 1D or 2D.")

    kernel = np.asarray(kernel, dtype=float).reshape(-1)
    if kernel.size == 0:
        raise ValueError("kernel cannot be empty.")
    if kernel.size > arr.shape[0]:
        out = np.full(arr.shape, np.nan, dtype=float)
        return out.reshape(-1) if flatten else out

    c = int(kernel.size)
    d = int(np.ceil(c / 2.0) - 1)
    ext = np.vstack((np.flipud(arr[:c, :]), arr, np.flipud(arr[-c:, :])))
    y = lfilter(kernel, [1.0], ext, axis=0)
    y = y[c + d : ext.shape[0] - c + d, :]
    return y.reshape(-1) if flatten else y


def _difference_of_gaussians_bandpass(data: np.ndarray, fs: float, passband: tuple[float, float]) -> np.ndarray:
    """Difference-of-Gaussians bandpass path used by John Long's detector."""
    low, high = map(float, passband)
    if not (0 < low < high):
        raise ValueError(f"Invalid passband {passband}.")
    h_low = _gaussian_lowpass_kernel(low, fs, support_sd=6.0)
    h_high = _gaussian_lowpass_kernel(high, fs, support_sd=6.0)
    lowpassed_high = _fir_mirror_filter(data, h_high)
    lowpassed_low = _fir_mirror_filter(lowpassed_high, h_low)
    return lowpassed_high - lowpassed_low


def _get_epoch_ranges(times: np.ndarray, epochs: nap.IntervalSet) -> tuple[np.ndarray, np.ndarray]:
    """Convert IntervalSet to sample-index ranges [start, end)."""
    starts: list[int] = []
    ends: list[int] = []
    for start_t, end_t in np.asarray(epochs.as_units("s").values, dtype=float):
        i0 = int(np.searchsorted(times, start_t, side="left"))
        i1 = int(np.searchsorted(times, end_t, side="right"))
        if i1 - i0 >= 2:
            starts.append(i0)
            ends.append(i1)
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


@njit(cache=True)
def _window_candidates(sw_diff: np.ndarray, epoch_starts: np.ndarray, epoch_ends: np.ndarray, win_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Find window maxima with boundary-aware 3-point peak checks."""
    n_total = 0
    for i in range(epoch_starts.shape[0]):
        span = epoch_ends[i] - epoch_starts[i]
        if span > win_size:
            n_total += span // win_size

    samples = np.empty(n_total, dtype=np.int64)
    values = np.empty(n_total, dtype=np.float64)
    n = 0
    n_samples = sw_diff.shape[0]

    for e in range(epoch_starts.shape[0]):
        block_start = epoch_starts[e]
        block_end_epoch = epoch_ends[e]
        while block_start + win_size <= block_end_epoch:
            block_end = block_start + win_size
            max_idx = 0
            max_val = sw_diff[block_start]
            for idx in range(block_start + 1, block_end):
                if sw_diff[idx] > max_val:
                    max_val = sw_diff[idx]
                    max_idx = idx - block_start
            sample = block_start + max_idx

            if max_idx == 0 or max_idx == win_size - 1:
                if sample <= 0 or sample >= n_samples - 1:
                    block_start += win_size
                    continue
                if not (sw_diff[sample] >= sw_diff[sample - 1] and sw_diff[sample] >= sw_diff[sample + 1]):
                    block_start += win_size
                    continue

            samples[n] = sample
            values[n] = max_val
            n += 1
            block_start += win_size

    return samples[:n], values[:n]


@njit(cache=True)
def _max_around_centers(values: np.ndarray, centers: np.ndarray, half_window: int) -> np.ndarray:
    out = np.empty(centers.shape[0], dtype=np.float64)
    n = values.shape[0]
    for i in range(centers.shape[0]):
        c = centers[i]
        lo = c - half_window
        if lo < 0:
            lo = 0
        hi = c + half_window + 1
        if hi > n:
            hi = n
        m = values[lo]
        for j in range(lo + 1, hi):
            if values[j] > m:
                m = values[j]
        out[i] = m
    return out


def _in_intervals(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Boolean membership test for times inside [start, end] intervals."""
    if intervals.size == 0:
        return np.zeros(times.shape[0], dtype=bool)
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    idx = np.searchsorted(starts, times, side="right") - 1
    valid = idx >= 0
    mask = np.zeros(times.shape[0], dtype=bool)
    if np.any(valid):
        mask[valid] = times[valid] <= ends[idx[valid]]
    return mask


def _safe_moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Moving-average smoothing with filtfilt fallback for short vectors."""
    w = int(max(3, window_samples))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / float(w)
    if values.shape[0] <= max(3 * (w - 1), w + 1):
        return np.convolve(values, kernel, mode="same")
    return filtfilt(kernel, [1.0], values)


def _threshold_segments(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert boolean mask into inclusive [start_idx, stop_idx] sample segments."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size == 0 or not np.any(m):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.flatnonzero(~m[:-1] & m[1:]) + 1
    stops = np.flatnonzero(m[:-1] & ~m[1:])
    if m[0]:
        starts = np.insert(starts, 0, 0)
    if m[-1]:
        stops = np.append(stops, m.size - 1)
    return starts.astype(np.int64), stops.astype(np.int64)


def _merge_segments(starts: np.ndarray, stops: np.ndarray, min_gap_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if starts.size == 0:
        return starts.astype(np.int64), stops.astype(np.int64)
    merged_starts = [int(starts[0])]
    merged_stops = [int(stops[0])]
    for start_i, stop_i in zip(starts[1:], stops[1:]):
        if int(start_i) - int(merged_stops[-1]) < int(min_gap_samples):
            merged_stops[-1] = int(stop_i)
        else:
            merged_starts.append(int(start_i))
            merged_stops.append(int(stop_i))
    return np.asarray(merged_starts, dtype=np.int64), np.asarray(merged_stops, dtype=np.int64)


def _precision_recall(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = np.nan
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _empty_detection_output(epochs: nap.IntervalSet, params: SWRDetectorParams) -> dict[str, Any]:
    empty_ep = nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))
    return {
        "events": empty_ep,
        "peaks_tsd": nap.Tsd(t=np.array([], dtype=float), d=np.array([], dtype=float), time_support=epochs),
        "timestamps": np.empty((0, 2), dtype=float),
        "peaks": np.array([], dtype=float),
        "peakNormedPower": np.array([], dtype=float),
        "SwMax": np.empty((0, 2), dtype=float),
        "RipMax": np.empty((0, 2), dtype=float),
        "detectorName": "detect_swr_jlong",
        "detectorinfo": {
            "detectorname": "detect_swr_jlong",
            "detectionparms": params.__dict__.copy(),
            "detectiondate": np.datetime64("today").astype(str),
            "detectionintervals": np.asarray(epochs.as_units("s").values, dtype=float),
            "numba_enabled": _HAS_NUMBA,
        },
    }


def _build_training_output(
    *,
    training_labels: nap.IntervalSet | None,
    feature_times: np.ndarray,
    labels: np.ndarray,
    swr_cluster: int,
    sw_diff_all: np.ndarray,
    rip_power_all: np.ndarray,
    idx_swr: np.ndarray,
    idx_non: np.ndarray,
    candidate_mask: np.ndarray,
    percentiles: np.ndarray | None,
    centers: np.ndarray,
) -> dict[str, Any] | None:
    if training_labels is None:
        return None

    if not isinstance(training_labels, nap.IntervalSet):
        raise TypeError("training_labels must be a pynapple.IntervalSet.")

    labels_truth = _in_intervals(feature_times, np.asarray(training_labels.as_units("s").values, dtype=float))
    swr_label_mask = labels == swr_cluster
    kmeans_acc = float(max(np.mean(swr_label_mask == labels_truth), np.mean(~swr_label_mask == labels_truth)))

    if percentiles is None:
        percentiles = np.arange(0, 101, 5, dtype=float)
    percentiles = np.asarray(percentiles, dtype=float)
    if percentiles.ndim != 1 or percentiles.size < 2:
        raise ValueError("training_percentiles must be a 1D array with at least two elements.")

    precision_surface = np.full((percentiles.size, percentiles.size), np.nan, dtype=float)
    recall_surface = np.full((percentiles.size, percentiles.size), np.nan, dtype=float)
    f1_surface = np.full((percentiles.size, percentiles.size), np.nan, dtype=float)

    for i, p_sw in enumerate(percentiles):
        th_sw = np.percentile(sw_diff_all[idx_swr], p_sw) if np.any(idx_swr) else np.percentile(sw_diff_all, p_sw)
        for j, p_rip in enumerate(percentiles):
            th_rip = np.percentile(rip_power_all[idx_non], p_rip) if np.any(idx_non) else np.percentile(rip_power_all, p_rip)
            pred = swr_label_mask & (sw_diff_all > th_sw) & (rip_power_all > th_rip)
            precision, recall, f1 = _precision_recall(labels_truth, pred)
            precision_surface[i, j] = precision
            recall_surface[i, j] = recall
            f1_surface[i, j] = f1

    if np.all(np.isnan(f1_surface)):
        best = {"sw_percentile": np.nan, "rip_percentile": np.nan, "f1": np.nan}
    else:
        best_idx = int(np.nanargmax(f1_surface))
        bi, bj = np.unravel_index(best_idx, f1_surface.shape)
        best = {"sw_percentile": float(percentiles[bi]), "rip_percentile": float(percentiles[bj]), "f1": float(f1_surface[bi, bj])}

    candidate_precision, candidate_recall, candidate_f1 = _precision_recall(labels_truth, candidate_mask)
    return {
        "kmeans_accuracy": kmeans_acc,
        "truth_positive_count": int(np.sum(labels_truth)),
        "truth_negative_count": int(np.sum(~labels_truth)),
        "cluster_centers": centers,
        "candidate_precision": candidate_precision,
        "candidate_recall": candidate_recall,
        "candidate_f1": candidate_f1,
        "percentiles": percentiles,
        "precision_surface": precision_surface,
        "recall_surface": recall_surface,
        "f1_surface": f1_surface,
        "best_threshold": best,
    }


def detect_swr_jlong(
    lfp: nap.TsdFrame,
    epochs: nap.IntervalSet | None = None,
    *,
    params: SWRDetectorParams = SWRDetectorParams(),
    training_labels: nap.IntervalSet | None = None,
    training_percentiles: np.ndarray | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """
    Detect sharp-wave ripple events from multichannel LFP using a John Long style pipeline.

    Parameters
    ----------
    lfp
        Multichannel LFP as `nap.TsdFrame`. First channel is treated as superficial,
        last channel as deep (matching detector feature construction).
    epochs
        Detection intervals. If `None`, use `lfp.time_support`.
    params
        Detector parameters.
    training_labels
        Optional labeled SWR intervals for training diagnostics.
    training_percentiles
        Optional percentile grid for training precision/recall surfaces.
    random_seed
        Random seed for k-means reproducibility.
    """
    if not isinstance(lfp, nap.TsdFrame):
        raise TypeError("lfp must be a pynapple.TsdFrame.")
    if lfp.shape[1] < 2:
        raise ValueError("lfp must include at least 2 channels.")

    epochs_use = lfp.time_support if epochs is None else epochs
    if not isinstance(epochs_use, nap.IntervalSet):
        raise TypeError("epochs must be a pynapple.IntervalSet.")

    lfp_s = lfp.as_units("s")
    times = np.asarray(lfp_s.index.values, dtype=float)
    data = np.asarray(lfp_s.values, dtype=float)
    if np.any(~np.isfinite(data)):
        raise ValueError("lfp contains non-finite values.")

    fs = float(lfp.rate)
    if fs <= 0:
        raise ValueError("lfp sampling rate must be positive.")

    epoch_starts, epoch_ends = _get_epoch_ranges(times, epochs_use)
    if epoch_starts.size == 0:
        return _empty_detection_output(epochs_use, params)

    # Feature extraction
    sw_band = _difference_of_gaussians_bandpass(data, fs, params.sw_bp)
    sw_diff = sw_band[:, 0] - sw_band[:, -1]

    centered = data - data.mean(axis=1, keepdims=True)
    rip_band = _difference_of_gaussians_bandpass(centered, fs, params.rip_bp)
    rip_abs = np.abs(rip_band)
    rip_window = np.pi / np.mean(np.asarray(params.rip_bp, dtype=float))
    power_kernel = _gaussian_lowpass_kernel(1.0 / rip_window, fs, support_sd=6.0)
    rip_power = np.nanmax(_fir_mirror_filter(rip_abs, power_kernel), axis=1)

    # Windowed candidate generation
    win_size = max(3, int(np.floor(params.win_size_ms * fs / 1000.0)))
    half_win = max(1, win_size // 2)
    feature_samples, sw_diff_all = _window_candidates(sw_diff.astype(np.float64), epoch_starts, epoch_ends, win_size)
    if feature_samples.size == 0:
        return _empty_detection_output(epochs_use, params)
    rip_power_all = _max_around_centers(rip_power.astype(np.float64), feature_samples.astype(np.int64), half_win)

    # K-means feature partition
    features = np.column_stack((sw_diff_all, rip_power_all))
    if feature_samples.size < 2:
        labels = np.zeros(feature_samples.shape[0], dtype=int)
        centers = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=float)
    else:
        km = KMeans(n_clusters=2, n_init=20, random_state=random_seed)
        labels = km.fit_predict(features)
        centers = km.cluster_centers_

    counts = np.bincount(labels, minlength=2)
    nonzero_clusters = np.flatnonzero(counts > 0)
    if nonzero_clusters.size == 0:
        swr_cluster = 0
    elif nonzero_clusters.size == 1:
        swr_cluster = int(nonzero_clusters[0])
    else:
        swr_cluster = int(nonzero_clusters[np.argmin(counts[nonzero_clusters])])
    idx_swr = labels == swr_cluster
    idx_non = ~idx_swr

    thres_l_swd = np.percentile(sw_diff_all[idx_swr], params.per_thres_swd) if np.any(idx_swr) else np.percentile(sw_diff_all, params.per_thres_swd)
    thres_l_rip = np.percentile(rip_power_all[idx_non], params.per_thres_rip) if np.any(idx_non) else np.percentile(rip_power_all, params.per_thres_rip)

    candidate_mask = idx_swr & (sw_diff_all > thres_l_swd) & (rip_power_all > thres_l_rip)
    candidate_feature_idx = np.flatnonzero(candidate_mask)
    candidate_samples = feature_samples[candidate_mask]
    if candidate_samples.size == 0:
        out = _empty_detection_output(epochs_use, params)
        out["training"] = _build_training_output(
            training_labels=training_labels,
            feature_times=times[feature_samples],
            labels=labels,
            swr_cluster=swr_cluster,
            sw_diff_all=sw_diff_all,
            rip_power_all=rip_power_all,
            idx_swr=idx_swr,
            idx_non=idx_non,
            candidate_mask=candidate_mask,
            percentiles=training_percentiles,
            centers=centers,
        )
        return out

    # Local screening around each candidate
    bound = max(1, int(np.round(params.ns_chk_s * fs)))
    valid_bounds = (candidate_samples - bound >= 0) & (candidate_samples + bound < sw_diff.shape[0])
    candidate_samples = candidate_samples[valid_bounds]
    candidate_feature_idx = candidate_feature_idx[valid_bounds]
    if candidate_samples.size == 0:
        out = _empty_detection_output(epochs_use, params)
        out["training"] = _build_training_output(
            training_labels=training_labels,
            feature_times=times[feature_samples],
            labels=labels,
            swr_cluster=swr_cluster,
            sw_diff_all=sw_diff_all,
            rip_power_all=rip_power_all,
            idx_swr=idx_swr,
            idx_non=idx_non,
            candidate_mask=candidate_mask,
            percentiles=training_percentiles,
            centers=centers,
        )
        return out

    min_isi_samples = max(1, int(np.round(params.min_isi_s * fs)))
    min_dur_sw_samp = max(1, int(np.round(params.min_dur_sw_s * fs)))
    max_dur_sw_samp = max(1, int(np.round(params.max_dur_sw_s * fs)))
    min_dur_rp_samp = max(1, int(np.round(params.min_dur_rp_s * fs)))

    starts: list[int] = []
    stops: list[int] = []
    peaks: list[int] = []
    sw_meta: list[tuple[float, float]] = []
    rip_meta: list[tuple[float, float]] = []
    peak_normed_power: list[float] = []

    last_accepted = -10**12
    center_idx = bound
    for sample, feat_idx in zip(candidate_samples, candidate_feature_idx):
        if sample - last_accepted < min_isi_samples:
            continue

        sw_chk = sw_diff[sample - bound : sample + bound + 1]
        rip_chk = rip_power[sample - bound : sample + bound + 1]
        sw_med = float(np.median(sw_chk))
        sw_std = max(float(np.std(sw_chk)), _EPS)
        rip_med = float(np.median(rip_chk))
        rip_std = max(float(np.std(rip_chk)), _EPS)

        if sw_diff[sample] < sw_med + params.thres_sd_swd[1] * sw_std:
            continue
        if rip_power[sample] < rip_med + params.thres_sd_rip[1] * rip_std:
            continue

        start_candidates = np.flatnonzero(sw_chk[: center_idx + 1] < sw_med + params.thres_sd_swd[0] * sw_std)
        stop_candidates = np.flatnonzero(sw_chk[center_idx:] < sw_med + params.thres_sd_swd[0] * sw_std)
        start_rel = int(start_candidates[-1]) if start_candidates.size else 0
        stop_rel = int(stop_candidates[0]) if stop_candidates.size else (sw_chk.shape[0] - center_idx - 1)
        dur_sw = (center_idx - start_rel) + stop_rel

        rp_lo = max(0, center_idx - half_win)
        rp_hi = min(rip_chk.shape[0], center_idx + half_win + 1)
        rp_peak_rel = rp_lo + int(np.argmax(rip_chk[rp_lo:rp_hi]))
        start_rp_candidates = np.flatnonzero(rip_chk[: rp_peak_rel + 1] < rip_med + params.thres_sd_rip[0] * rip_std)
        stop_rp_candidates = np.flatnonzero(rip_chk[rp_peak_rel:] < rip_med + params.thres_sd_rip[0] * rip_std)
        start_rp_rel = int(start_rp_candidates[-1]) if start_rp_candidates.size else 0
        stop_rp_rel = int(stop_rp_candidates[0]) if stop_rp_candidates.size else (rip_chk.shape[0] - rp_peak_rel - 1)
        dur_rp = (rp_peak_rel - start_rp_rel) + stop_rp_rel

        if dur_sw < min_dur_sw_samp or dur_sw > max_dur_sw_samp or dur_rp < min_dur_rp_samp:
            continue

        start_sample = sample - (center_idx - start_rel)
        stop_sample = sample + stop_rel
        if start_sample < 0 or stop_sample <= start_sample:
            continue

        sw_peak = float(sw_diff_all[feat_idx])
        rip_peak = float(rip_power_all[feat_idx])
        sw_z = (sw_peak - sw_med) / sw_std
        rip_z = (rip_peak - rip_med) / rip_std
        sw_pct = float(np.mean(sw_chk < sw_peak))
        rip_pct = float(np.mean(rip_chk < rip_peak))

        starts.append(int(start_sample))
        stops.append(int(stop_sample))
        peaks.append(int(sample))
        sw_meta.append((sw_z, sw_pct))
        rip_meta.append((rip_z, rip_pct))
        peak_normed_power.append(rip_peak)
        last_accepted = int(sample)

    if not starts:
        out = _empty_detection_output(epochs_use, params)
        out["training"] = _build_training_output(
            training_labels=training_labels,
            feature_times=times[feature_samples],
            labels=labels,
            swr_cluster=swr_cluster,
            sw_diff_all=sw_diff_all,
            rip_power_all=rip_power_all,
            idx_swr=idx_swr,
            idx_non=idx_non,
            candidate_mask=candidate_mask,
            percentiles=training_percentiles,
            centers=centers,
        )
        return out

    starts_arr = np.asarray(starts, dtype=np.int64)
    stops_arr = np.asarray(stops, dtype=np.int64)
    peaks_arr = np.asarray(peaks, dtype=np.int64)
    sw_meta_arr = np.asarray(sw_meta, dtype=float)
    rip_meta_arr = np.asarray(rip_meta, dtype=float)
    peak_normed_arr = np.asarray(peak_normed_power, dtype=float)

    # Peak timing via trough nearest ripple envelope maximum.
    rip_first = _difference_of_gaussians_bandpass(data[:, 0], fs, params.rip_bp)
    sign_env = 2.0 * (rip_first**2)
    if sign_env.size >= 11:
        win = min(101, sign_env.size if sign_env.size % 2 == 1 else sign_env.size - 1)
        win = max(win, 11)
        envelope = savgol_filter(sign_env, window_length=win, polyorder=4, mode="interp")
    else:
        envelope = sign_env
    envelope = np.sqrt(np.abs(envelope))

    trough_half = max(1, int(np.round(0.01 * fs)))
    raw_ch0 = data[:, 0]
    trough_peaks: list[int] = []
    keep_event_idx: list[int] = []
    for i, (start_i, stop_i) in enumerate(zip(starts_arr, stops_arr)):
        idxs = np.arange(start_i, stop_i + 1, dtype=np.int64)
        if idxs.size < 3:
            continue
        env_peak = int(idxs[np.argmax(envelope[idxs])])
        lo = max(0, env_peak - trough_half)
        hi = min(raw_ch0.size, env_peak + trough_half + 1)
        if hi - lo < 2:
            continue
        trough_idx = int(lo + np.argmin(raw_ch0[lo:hi]))
        if start_i < trough_idx < stop_i:
            trough_peaks.append(trough_idx)
            keep_event_idx.append(i)

    if not keep_event_idx:
        out = _empty_detection_output(epochs_use, params)
        out["training"] = _build_training_output(
            training_labels=training_labels,
            feature_times=times[feature_samples],
            labels=labels,
            swr_cluster=swr_cluster,
            sw_diff_all=sw_diff_all,
            rip_power_all=rip_power_all,
            idx_swr=idx_swr,
            idx_non=idx_non,
            candidate_mask=candidate_mask,
            percentiles=training_percentiles,
            centers=centers,
        )
        return out

    keep_idx = np.asarray(keep_event_idx, dtype=np.int64)
    starts_arr = starts_arr[keep_idx]
    stops_arr = stops_arr[keep_idx]
    peaks_arr = np.asarray(trough_peaks, dtype=np.int64)
    sw_meta_arr = sw_meta_arr[keep_idx]
    rip_meta_arr = rip_meta_arr[keep_idx]
    peak_normed_arr = peak_normed_arr[keep_idx]

    timestamps = np.column_stack((times[starts_arr], times[stops_arr]))
    peak_times = times[peaks_arr]

    metadata = pd.DataFrame(
        {
            "peak_time": peak_times,
            "peakNormedPower": peak_normed_arr,
            "sw_zscore": sw_meta_arr[:, 0],
            "sw_percentile": sw_meta_arr[:, 1],
            "rip_zscore": rip_meta_arr[:, 0],
            "rip_percentile": rip_meta_arr[:, 1],
        }
    )
    events = nap.IntervalSet(start=timestamps[:, 0], end=timestamps[:, 1], metadata=metadata)
    peaks_tsd = nap.Tsd(t=peak_times, d=peak_normed_arr, time_support=epochs_use)

    detectorinfo = {
        "detectorname": "detect_swr_jlong",
        "detectionparms": params.__dict__.copy(),
        "detectiondate": np.datetime64("today").astype(str),
        "detectionintervals": np.asarray(epochs_use.as_units("s").values, dtype=float),
        "clustering_centers": centers,
        "feature_thresholds": {"sw_diff": float(thres_l_swd), "rip_power": float(thres_l_rip)},
        "numba_enabled": _HAS_NUMBA,
    }

    out = {
        "events": events,
        "peaks_tsd": peaks_tsd,
        "timestamps": timestamps,
        "peaks": peak_times,
        "peakNormedPower": peak_normed_arr,
        "SwMax": sw_meta_arr,
        "RipMax": rip_meta_arr,
        "detectorName": "detect_swr_jlong",
        "detectorinfo": detectorinfo,
    }
    out["training"] = _build_training_output(
        training_labels=training_labels,
        feature_times=times[feature_samples],
        labels=labels,
        swr_cluster=swr_cluster,
        sw_diff_all=sw_diff_all,
        rip_power_all=rip_power_all,
        idx_swr=idx_swr,
        idx_non=idx_non,
        candidate_mask=candidate_mask,
        percentiles=training_percentiles,
        centers=centers,
    )
    return out


def _empty_find_ripples_output(
    epoch: nap.IntervalSet,
    params: FMATRippleDetectorParams,
    *,
    stdev: float | None = None,
) -> dict[str, Any]:
    empty_ep = nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))
    return {
        "events": empty_ep,
        "peaks_tsd": nap.Tsd(t=np.array([], dtype=float), d=np.array([], dtype=float), time_support=epoch),
        "timestamps": np.empty((0, 2), dtype=float),
        "peaks": np.array([], dtype=float),
        "peakNormedPower": np.array([], dtype=float),
        "stdev": float(np.nan if stdev is None else stdev),
        "noise": {
            "times": np.empty((0, 2), dtype=float),
            "peaks": np.array([], dtype=float),
            "peakNormedPower": np.array([], dtype=float),
        },
        "nss": nap.Tsd(t=np.array([], dtype=float), d=np.array([], dtype=float), time_support=epoch),
        "detectorName": "find_ripples_fmat",
        "detectorinfo": {
            "detectorname": "find_ripples_fmat",
            "detectiondate": np.datetime64("today").astype(str),
            "detectionintervals": np.asarray(epoch.as_units("s").values, dtype=float),
            "detectionparms": params.__dict__.copy(),
        },
    }


def _compute_nss(
    lfp: nap.Tsd,
    *,
    passband: tuple[float, float],
    fs: float,
    filter_order: int,
    smooth_window_samples: int,
    baseline_mask: np.ndarray | None = None,
    fixed_std: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lfp_s = lfp.as_units("s")
    times = np.asarray(lfp_s.index.values, dtype=float)
    filtered = _archive_bandpass_filter(lfp, passband[0], passband[1], fs, order=filter_order)
    filtered_values = np.asarray(filtered.values, dtype=float)
    squared = filtered_values**2
    smoothed = _safe_moving_average(squared, smooth_window_samples)

    if baseline_mask is not None and baseline_mask.shape[0] == smoothed.shape[0] and np.any(baseline_mask):
        baseline = smoothed[baseline_mask]
    else:
        baseline = smoothed

    mean_baseline = float(np.mean(baseline))
    std_baseline = float(np.std(baseline))
    if fixed_std is not None:
        std_baseline = float(fixed_std)
    std_baseline = max(std_baseline, _EPS)
    nss = (smoothed - mean_baseline) / std_baseline
    return times, filtered_values, nss, std_baseline


def find_ripples_fmat(
    lfp: nap.Tsd,
    epoch: nap.IntervalSet | None = None,
    *,
    params: FMATRippleDetectorParams = FMATRippleDetectorParams(),
    restrict: nap.IntervalSet | None = None,
    stdev: float | None = None,
    noise_lfp: nap.Tsd | None = None,
    emg: nap.Tsd | None = None,
    emg_threshold: float | None = None,
) -> dict[str, Any]:
    """
    FMAT/buzcode-style ripple detection using normalized squared signal thresholding.

    This mode is complementary to `detect_swr_jlong` and is useful when a classic
    NSS threshold workflow (FindRipples-like) is preferred.
    """
    if not isinstance(lfp, nap.Tsd):
        raise TypeError("lfp must be a pynapple.Tsd.")
    if epoch is None:
        epoch = lfp.time_support
    if not isinstance(epoch, nap.IntervalSet):
        raise TypeError("epoch must be a pynapple.IntervalSet.")
    if restrict is not None and not isinstance(restrict, nap.IntervalSet):
        raise TypeError("restrict must be a pynapple.IntervalSet when provided.")
    if noise_lfp is not None and not isinstance(noise_lfp, nap.Tsd):
        raise TypeError("noise_lfp must be a pynapple.Tsd when provided.")
    if emg is not None and not isinstance(emg, nap.Tsd):
        raise TypeError("emg must be a pynapple.Tsd when provided.")
    if stdev is not None and stdev <= 0:
        raise ValueError("stdev must be positive when provided.")

    low_th, high_th = map(float, params.thresholds)
    if not (0.0 <= low_th < high_th):
        raise ValueError("params.thresholds must satisfy 0 <= low < high.")
    min_inter_ms, max_duration_ms = map(float, params.durations_ms)
    if min_inter_ms < 0 or max_duration_ms <= 0:
        raise ValueError("params.durations_ms must be non-negative and positive.")
    min_duration_s = float(params.min_duration_ms) / 1000.0
    if min_duration_s < 0:
        raise ValueError("params.min_duration_ms must be non-negative.")
    max_duration_s = max_duration_ms / 1000.0
    if min_duration_s > max_duration_s:
        raise ValueError("params.min_duration_ms cannot exceed params.durations_ms[1].")

    lfp_epoch = lfp.restrict(epoch)
    if lfp_epoch.shape[0] < 3:
        return _empty_find_ripples_output(epoch, params, stdev=stdev)

    fs = float(lfp.rate)
    baseline_mask: np.ndarray | None = None
    if restrict is not None:
        lfp_epoch_times = np.asarray(lfp_epoch.as_units("s").index.values, dtype=float)
        baseline_mask = _in_intervals(lfp_epoch_times, np.asarray(restrict.as_units("s").values, dtype=float))

    times, filtered, nss, used_stdev = _compute_nss(
        lfp_epoch,
        passband=params.passband,
        fs=fs,
        filter_order=int(params.filter_order),
        smooth_window_samples=int(params.smooth_window_samples),
        baseline_mask=baseline_mask,
        fixed_std=stdev,
    )
    nss_tsd = nap.Tsd(t=times, d=nss, time_support=epoch)

    starts, stops = _threshold_segments(nss > low_th)
    if starts.size == 0:
        out = _empty_find_ripples_output(epoch, params, stdev=used_stdev)
        out["nss"] = nss_tsd
        return out

    min_inter_samples = int(np.round((min_inter_ms / 1000.0) * fs))
    starts, stops = _merge_segments(starts, stops, min_inter_samples)

    keep_starts: list[int] = []
    keep_stops: list[int] = []
    keep_peaks: list[int] = []
    keep_peak_power: list[float] = []
    for start_i, stop_i in zip(starts, stops):
        segment = slice(int(start_i), int(stop_i) + 1)
        seg_nss = nss[segment]
        if seg_nss.shape[0] == 0:
            continue
        peak_power = float(np.max(seg_nss))
        if peak_power <= high_th:
            continue
        seg_filtered = filtered[segment]
        peak_idx = int(start_i + np.argmin(seg_filtered))
        keep_starts.append(int(start_i))
        keep_stops.append(int(stop_i))
        keep_peaks.append(peak_idx)
        keep_peak_power.append(peak_power)

    if not keep_starts:
        out = _empty_find_ripples_output(epoch, params, stdev=used_stdev)
        out["nss"] = nss_tsd
        return out

    starts_arr = np.asarray(keep_starts, dtype=np.int64)
    stops_arr = np.asarray(keep_stops, dtype=np.int64)
    peaks_arr = np.asarray(keep_peaks, dtype=np.int64)
    peak_power_arr = np.asarray(keep_peak_power, dtype=float)

    start_t = times[starts_arr]
    stop_t = times[stops_arr]
    peak_t = times[peaks_arr]
    duration = stop_t - start_t
    keep_duration = (duration <= max_duration_s) & (duration >= min_duration_s)
    starts_arr = starts_arr[keep_duration]
    stops_arr = stops_arr[keep_duration]
    peaks_arr = peaks_arr[keep_duration]
    peak_power_arr = peak_power_arr[keep_duration]
    start_t = start_t[keep_duration]
    stop_t = stop_t[keep_duration]
    peak_t = peak_t[keep_duration]

    if starts_arr.size == 0:
        out = _empty_find_ripples_output(epoch, params, stdev=used_stdev)
        out["nss"] = nss_tsd
        return out

    excluded = np.zeros(starts_arr.shape[0], dtype=bool)
    if noise_lfp is not None:
        noise_epoch = noise_lfp.restrict(epoch)
        if noise_epoch.shape[0] >= 3:
            noise_times, _, noise_nss, _ = _compute_nss(
                noise_epoch,
                passband=params.passband,
                fs=fs,
                filter_order=int(params.filter_order),
                smooth_window_samples=int(params.smooth_window_samples),
                fixed_std=used_stdev,
            )
            for i, (event_start, event_stop) in enumerate(zip(start_t, stop_t)):
                i0 = int(np.searchsorted(noise_times, event_start, side="left"))
                i1 = int(np.searchsorted(noise_times, event_stop, side="right"))
                if i1 > i0 and np.any(noise_nss[i0:i1] > high_th):
                    excluded[i] = True

    if emg is not None:
        if emg_threshold is None:
            emg_threshold = 0.9
        emg_values = np.asarray(emg.values, dtype=float)
        emg_times = np.asarray(emg.as_units("s").index.values, dtype=float)
        if emg_values.size:
            if emg_values.size == 1:
                excluded |= emg_values[0] > float(emg_threshold)
            else:
                idx = np.searchsorted(emg_times, start_t, side="left")
                idx = np.clip(idx, 1, emg_times.shape[0] - 1)
                left = emg_times[idx - 1]
                right = emg_times[idx]
                use_left = np.abs(start_t - left) <= np.abs(start_t - right)
                idx[use_left] -= 1
                excluded |= emg_values[idx] > float(emg_threshold)

    bad_times = np.empty((0, 4), dtype=float)
    if np.any(excluded):
        bad_times = np.column_stack((start_t[excluded], peak_t[excluded], stop_t[excluded], peak_power_arr[excluded]))
        keep = ~excluded
        starts_arr = starts_arr[keep]
        stops_arr = stops_arr[keep]
        peaks_arr = peaks_arr[keep]
        peak_power_arr = peak_power_arr[keep]
        start_t = start_t[keep]
        stop_t = stop_t[keep]
        peak_t = peak_t[keep]

    if starts_arr.size == 0:
        out = _empty_find_ripples_output(epoch, params, stdev=used_stdev)
        out["noise"] = {
            "times": bad_times[:, [0, 2]] if bad_times.size else np.empty((0, 2), dtype=float),
            "peaks": bad_times[:, 1] if bad_times.size else np.array([], dtype=float),
            "peakNormedPower": bad_times[:, 3] if bad_times.size else np.array([], dtype=float),
        }
        out["nss"] = nss_tsd
        return out

    timestamps = np.column_stack((start_t, stop_t))
    metadata = pd.DataFrame({"peak_time": peak_t, "peakNormedPower": peak_power_arr})
    events = nap.IntervalSet(start=timestamps[:, 0], end=timestamps[:, 1], metadata=metadata)
    peaks_tsd = nap.Tsd(t=peak_t, d=peak_power_arr, time_support=epoch)

    noise = {
        "times": bad_times[:, [0, 2]] if bad_times.size else np.empty((0, 2), dtype=float),
        "peaks": bad_times[:, 1] if bad_times.size else np.array([], dtype=float),
        "peakNormedPower": bad_times[:, 3] if bad_times.size else np.array([], dtype=float),
    }
    detectorinfo = {
        "detectorname": "find_ripples_fmat",
        "detectiondate": np.datetime64("today").astype(str),
        "detectionintervals": np.asarray((restrict if restrict is not None else epoch).as_units("s").values, dtype=float),
        "detectionparms": params.__dict__.copy(),
        "noisechannel": int(noise_lfp is not None),
        "emg_threshold": float(np.nan if emg_threshold is None else emg_threshold),
    }
    return {
        "events": events,
        "peaks_tsd": peaks_tsd,
        "timestamps": timestamps,
        "peaks": peak_t,
        "peakNormedPower": peak_power_arr,
        "stdev": float(used_stdev),
        "noise": noise,
        "nss": nss_tsd,
        "detectorName": "find_ripples_fmat",
        "detectorinfo": detectorinfo,
    }


def detect_ripples_fmat(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for `find_ripples_fmat`."""
    return find_ripples_fmat(*args, **kwargs)


def FindRipples(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """MATLAB-compatibility alias for `find_ripples_fmat`."""
    return find_ripples_fmat(*args, **kwargs)


def bandpass_filter(data: nap.Tsd | nap.TsdFrame, lowcut: float, highcut: float, fs: float, order: int = 4) -> nap.Tsd | nap.TsdFrame:
    """
    Archive-backed bandpass filtering helper.

    This deliberately reuses the existing archive implementation instead of duplicating it.
    """
    return _archive_bandpass_filter(data, lowcut, highcut, fs, order=order)


def detect_oscillatory_events(
    lfp: nap.Tsd,
    epoch: nap.IntervalSet,
    freq_band: tuple[float, float],
    thres_band: tuple[float, float],
    duration_band: tuple[float, float],
    min_inter_duration: float,
    wsize: int = 51,
) -> tuple[nap.IntervalSet, nap.Tsd]:
    """
    Archive-backed oscillatory event detection (NSS threshold workflow).

    This reuses the validated archive implementation to avoid functionality duplication.
    """
    return _archive_detect_oscillatory_events(
        lfp=lfp,
        epoch=epoch,
        freq_band=freq_band,
        thres_band=thres_band,
        duration_band=duration_band,
        min_inter_duration=min_inter_duration,
        wsize=wsize,
    )


def detect_ripples_nss(
    lfp: nap.Tsd,
    epoch: nap.IntervalSet,
    *,
    freq_band: tuple[float, float] = (100.0, 300.0),
    thres_band: tuple[float, float] = (1.0, 10.0),
    duration_band: tuple[float, float] = (0.02, 0.2),
    min_inter_duration: float = 0.02,
    wsize: int = 51,
) -> tuple[nap.IntervalSet, nap.Tsd]:
    """Convenience alias for archive-backed NSS ripple detection."""
    return detect_oscillatory_events(
        lfp=lfp,
        epoch=epoch,
        freq_band=freq_band,
        thres_band=thres_band,
        duration_band=duration_band,
        min_inter_duration=min_inter_duration,
        wsize=wsize,
    )


def _coerce_events_array(ripples: Any) -> np.ndarray:
    if isinstance(ripples, nap.IntervalSet):
        return np.asarray(ripples.as_units("s").values, dtype=float)
    if isinstance(ripples, dict):
        if "events" in ripples and isinstance(ripples["events"], nap.IntervalSet):
            return np.asarray(ripples["events"].as_units("s").values, dtype=float)
        if "timestamps" in ripples:
            return np.asarray(ripples["timestamps"], dtype=float)
    arr = np.asarray(ripples, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("ripples must be IntervalSet, detector output dict, or Nx2 array.")
    return arr


def _coerce_peak_times(ripples: Any, events: np.ndarray, peaks: np.ndarray | None = None) -> np.ndarray:
    if peaks is not None:
        peaks_arr = np.asarray(peaks, dtype=float).reshape(-1)
    elif isinstance(ripples, dict) and "peaks" in ripples:
        peaks_arr = np.asarray(ripples["peaks"], dtype=float).reshape(-1)
    elif isinstance(ripples, dict) and "peaks_tsd" in ripples and isinstance(ripples["peaks_tsd"], nap.Tsd):
        peaks_arr = np.asarray(ripples["peaks_tsd"].as_units("s").index.values, dtype=float).reshape(-1)
    else:
        peaks_arr = np.mean(events, axis=1)

    if peaks_arr.shape[0] != events.shape[0]:
        return np.mean(events, axis=1)
    return peaks_arr


def _coerce_peak_power(ripples: Any, n_events: int, peak_power: np.ndarray | None = None) -> np.ndarray:
    if peak_power is not None:
        arr = np.asarray(peak_power, dtype=float).reshape(-1)
    elif isinstance(ripples, dict) and "peakNormedPower" in ripples:
        arr = np.asarray(ripples["peakNormedPower"], dtype=float).reshape(-1)
    elif isinstance(ripples, dict) and "peaks_tsd" in ripples and isinstance(ripples["peaks_tsd"], nap.Tsd):
        arr = np.asarray(ripples["peaks_tsd"].values, dtype=float).reshape(-1)
    else:
        arr = np.full(n_events, np.nan, dtype=float)
    if arr.shape[0] != n_events:
        return np.full(n_events, np.nan, dtype=float)
    return arr


def standardize_ripple_events(
    ripples: Any,
    *,
    peaks: np.ndarray | None = None,
    peak_power: np.ndarray | None = None,
    time_support: nap.IntervalSet | None = None,
) -> dict[str, Any]:
    """
    Standardize ripple detections into a consistent schema.

    Returns a dictionary with:
    - `events`: `nap.IntervalSet` with metadata
    - `table`: pandas DataFrame
    - `nwb`: dict of aligned arrays suitable for interval writing
    """
    events_arr = _coerce_events_array(ripples)
    if events_arr.shape[0] == 0:
        empty_ep = nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))
        empty_table = pd.DataFrame(columns=["start", "end", "peak_time", "duration_s", "peak_normed_power"])
        return {
            "events": empty_ep,
            "table": empty_table,
            "nwb": {
                "start_time": np.array([], dtype=float),
                "stop_time": np.array([], dtype=float),
                "peak_time": np.array([], dtype=float),
                "duration_s": np.array([], dtype=float),
                "peak_normed_power": np.array([], dtype=float),
            },
        }

    peak_times = _coerce_peak_times(ripples, events_arr, peaks=peaks)
    peak_power_arr = _coerce_peak_power(ripples, events_arr.shape[0], peak_power=peak_power)
    durations = events_arr[:, 1] - events_arr[:, 0]
    table = pd.DataFrame(
        {
            "start": events_arr[:, 0],
            "end": events_arr[:, 1],
            "peak_time": peak_times,
            "duration_s": durations,
            "peak_normed_power": peak_power_arr,
        }
    )
    metadata = table[["peak_time", "duration_s", "peak_normed_power"]].copy()
    if time_support is None:
        time_support = nap.IntervalSet(start=np.min(events_arr[:, 0]), end=np.max(events_arr[:, 1]))
    events = nap.IntervalSet(start=events_arr[:, 0], end=events_arr[:, 1], metadata=metadata)
    nwb = {
        "start_time": np.asarray(table["start"].values, dtype=float),
        "stop_time": np.asarray(table["end"].values, dtype=float),
        "peak_time": np.asarray(table["peak_time"].values, dtype=float),
        "duration_s": np.asarray(table["duration_s"].values, dtype=float),
        "peak_normed_power": np.asarray(table["peak_normed_power"].values, dtype=float),
    }
    return {"events": events, "table": table, "nwb": nwb, "time_support": time_support}


def _event_aligned_matrix(tsd: nap.Tsd, peak_times: np.ndarray, durations: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    if peak_times.size == 0:
        return np.empty((0, 0), dtype=float), np.array([], dtype=float)
    ref = nap.Ts(t=peak_times, time_support=tsd.time_support)
    aligned = nap.compute_perievent_continuous(tsd, ref, durations, time_unit="s")
    if not isinstance(aligned, nap.TsdFrame):
        return np.empty((0, 0), dtype=float), np.array([], dtype=float)
    values = np.asarray(aligned.values, dtype=float)
    return values.T, np.asarray(aligned.index.values, dtype=float)


def _nearest_indices(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, query, side="left")
    idx = np.clip(idx, 1, times.shape[0] - 1)
    left = times[idx - 1]
    right = times[idx]
    use_left = np.abs(query - left) <= np.abs(right - query)
    idx[use_left] = idx[use_left] - 1
    return idx


def _autocorrelogram(peak_times: np.ndarray, bin_size: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    if peak_times.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    peak_times = np.sort(np.asarray(peak_times, dtype=float))
    edges = np.arange(-duration, duration + bin_size, bin_size, dtype=float)
    if edges.size < 2:
        raise ValueError("corr_bin_size/corr_duration produce invalid histogram bins.")
    counts = np.zeros(edges.size - 1, dtype=float)
    for i in range(peak_times.size):
        delta = peak_times - peak_times[i]
        keep = (delta >= -duration) & (delta <= duration) & (delta != 0.0)
        if np.any(keep):
            hist, _ = np.histogram(delta[keep], bins=edges)
            counts += hist
    centers = (edges[:-1] + edges[1:]) / 2.0
    return counts, centers


def _corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return np.nan, np.nan
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)


def compute_ripple_feature_stats(
    filtered: nap.Tsd,
    ripples: Any,
    *,
    peaks: np.ndarray | None = None,
    durations: tuple[float, float] = (-0.075, 0.075),
    corr_bin_size: float = 0.01,
    corr_duration: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """
    Compute FMAT/buzcode-style ripple `maps`, `data`, and `stats`.

    Parameters
    ----------
    filtered
        Ripple-band filtered single-channel signal as `nap.Tsd`.
    ripples
        Ripple events as detector output dict, `IntervalSet`, or `Nx2` array.
    peaks
        Optional peak times overriding values derived from `ripples`.
    durations
        Window around each peak in seconds `(start, end)`.
    corr_bin_size
        Autocorrelogram bin size in seconds.
    corr_duration
        Autocorrelogram half-window in seconds.
    """
    if not isinstance(filtered, nap.Tsd):
        raise TypeError("filtered must be a pynapple.Tsd.")
    if durations[0] >= durations[1]:
        raise ValueError("durations must be an increasing (start, end) tuple.")

    tsd = filtered
    tsd_s = tsd.as_units("s")
    times = np.asarray(tsd_s.index.values, dtype=float)
    signal = np.asarray(tsd_s.values, dtype=float)
    events = _coerce_events_array(ripples)
    if events.shape[0] == 0 or signal.size < 3:
        empty_maps = {
            "ripples": np.empty((0, 0), dtype=float),
            "frequency": np.empty((0, 0), dtype=float),
            "phase": np.empty((0, 0), dtype=float),
            "amplitude": np.empty((0, 0), dtype=float),
            "t": np.array([], dtype=float),
        }
        empty_data = {
            "peakFrequency": np.array([], dtype=float),
            "peakAmplitude": np.array([], dtype=float),
            "duration": np.array([], dtype=float),
        }
        empty_stats: dict[str, Any] = {
            "acg": {"data": np.array([], dtype=float), "t": np.array([], dtype=float)},
            "amplitudeFrequency": {"rho": np.nan, "p": np.nan},
            "durationFrequency": {"rho": np.nan, "p": np.nan},
            "durationAmplitude": {"rho": np.nan, "p": np.nan},
        }
        return empty_maps, empty_data, empty_stats

    peak_times = _coerce_peak_times(ripples, events, peaks=peaks)

    h = hilbert(signal)
    phase = np.angle(h)
    amplitude = np.abs(h)
    unwrapped = np.unwrap(phase)
    dt = float(np.median(np.diff(times)))
    inst_frequency = np.gradient(unwrapped, dt) / (2.0 * np.pi)

    freq_tsd = nap.Tsd(t=times, d=inst_frequency, time_support=tsd.time_support)
    phase_tsd = nap.Tsd(t=times, d=phase, time_support=tsd.time_support)
    amp_tsd = nap.Tsd(t=times, d=amplitude, time_support=tsd.time_support)

    map_ripples, t_aligned = _event_aligned_matrix(tsd, peak_times, durations)
    map_frequency, _ = _event_aligned_matrix(freq_tsd, peak_times, durations)
    map_phase, _ = _event_aligned_matrix(phase_tsd, peak_times, durations)
    map_amplitude, _ = _event_aligned_matrix(amp_tsd, peak_times, durations)

    peak_idx = _nearest_indices(times, peak_times)
    peak_frequency = inst_frequency[peak_idx]
    peak_amplitude = amplitude[peak_idx]
    duration = events[:, 1] - events[:, 0]

    acg_data, acg_t = _autocorrelogram(peak_times, corr_bin_size, corr_duration)
    rho_af, p_af = _corr_pair(peak_amplitude, peak_frequency)
    rho_df, p_df = _corr_pair(duration, peak_frequency)
    rho_da, p_da = _corr_pair(duration, peak_amplitude)

    maps = {
        "ripples": map_ripples,
        "frequency": map_frequency,
        "phase": map_phase,
        "amplitude": map_amplitude,
        "t": t_aligned,
    }
    data = {
        "peakFrequency": peak_frequency,
        "peakAmplitude": peak_amplitude,
        "duration": duration,
    }
    stats: dict[str, Any] = {
        "acg": {"data": acg_data, "t": acg_t},
        "amplitudeFrequency": {"rho": rho_af, "p": p_af},
        "durationFrequency": {"rho": rho_df, "p": p_df},
        "durationAmplitude": {"rho": rho_da, "p": p_da},
    }
    return maps, data, stats


def compute_ripple_event_stats(
    lfp: nap.Tsd | nap.TsdFrame,
    ripples: Any,
    *,
    peaks: np.ndarray | None = None,
    spikes: nap.TsGroup | None = None,
    channel: int = 0,
) -> pd.DataFrame:
    """
    Compute event-level ripple metrics from existing ripple intervals.

    This function intentionally avoids re-running detection.
    """
    if not isinstance(lfp, (nap.Tsd, nap.TsdFrame)):
        raise TypeError("lfp must be a pynapple.Tsd or pynapple.TsdFrame.")

    lfp_s = lfp.as_units("s")
    times = np.asarray(lfp_s.index.values, dtype=float)
    fs = float(lfp.rate)

    if isinstance(lfp, nap.TsdFrame):
        if channel < 0 or channel >= lfp.shape[1]:
            raise ValueError(f"channel {channel} is out of range for lfp with {lfp.shape[1]} channels.")
        signal = np.asarray(lfp_s.values[:, channel], dtype=float)
    else:
        signal = np.asarray(lfp_s.values, dtype=float)

    events = _coerce_events_array(ripples)
    if events.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "duration_s",
                "peak_time",
                "peak_amplitude",
                "rms",
                "dominant_freq_hz",
                "n_samples",
                "total_spike_count",
            ]
        )

    if peaks is None:
        if isinstance(ripples, dict) and "peaks" in ripples:
            peaks_arr = np.asarray(ripples["peaks"], dtype=float)
        else:
            peaks_arr = np.full(events.shape[0], np.nan, dtype=float)
    else:
        peaks_arr = np.asarray(peaks, dtype=float)
    if peaks_arr.shape[0] != events.shape[0]:
        peaks_arr = np.full(events.shape[0], np.nan, dtype=float)

    if spikes is not None:
        if not isinstance(spikes, nap.TsGroup):
            raise TypeError("spikes must be a pynapple.TsGroup.")
        pooled = [np.asarray(spikes[unit].as_units("s").index.values, dtype=float) for unit in spikes.keys()]
        all_spikes = np.sort(np.concatenate(pooled)) if pooled else np.array([], dtype=float)
    else:
        all_spikes = np.array([], dtype=float)

    rows = []
    for i, (start_t, end_t) in enumerate(events):
        i0 = int(np.searchsorted(times, start_t, side="left"))
        i1 = int(np.searchsorted(times, end_t, side="right"))
        seg = signal[i0:i1]
        if seg.size < 2:
            rows.append(
                {
                    "start": float(start_t),
                    "end": float(end_t),
                    "duration_s": float(max(0.0, end_t - start_t)),
                    "peak_time": float(peaks_arr[i]),
                    "peak_amplitude": np.nan,
                    "rms": np.nan,
                    "dominant_freq_hz": np.nan,
                    "n_samples": int(seg.size),
                    "total_spike_count": 0,
                }
            )
            continue

        peak_amp = float(np.max(np.abs(seg)))
        rms = float(np.sqrt(np.mean(seg**2)))
        nperseg = min(256, seg.size)
        freqs, psd = welch(seg, fs=fs, nperseg=nperseg)
        dom_freq = float(freqs[np.argmax(psd)]) if psd.size else np.nan

        if all_spikes.size:
            n_spikes = int(np.searchsorted(all_spikes, end_t, side="right") - np.searchsorted(all_spikes, start_t, side="left"))
        else:
            n_spikes = 0

        rows.append(
            {
                "start": float(start_t),
                "end": float(end_t),
                "duration_s": float(end_t - start_t),
                "peak_time": float(peaks_arr[i]),
                "peak_amplitude": peak_amp,
                "rms": rms,
                "dominant_freq_hz": dom_freq,
                "n_samples": int(seg.size),
                "total_spike_count": n_spikes,
            }
        )

    return pd.DataFrame(rows)


def _coerce_band_for_fs(band: tuple[float, float], fs: float) -> tuple[float, float]:
    low, high = map(float, band)
    nyq = 0.5 * float(fs)
    high = min(high, 0.98 * nyq)
    low = max(low, 1e-3)
    if not (0 < low < high):
        raise ValueError(f"Invalid band {band} for sampling rate {fs} Hz.")
    return low, high


def compute_ripple_quality_metrics(
    lfp: nap.Tsd | nap.TsdFrame,
    ripples: Any,
    *,
    peaks: np.ndarray | None = None,
    channel: int = 0,
    ripple_band: tuple[float, float] = (100.0, 250.0),
    sharp_wave_band: tuple[float, float] = (2.0, 40.0),
    broadband_band: tuple[float, float] = (1.0, 400.0),
    filter_order: int = 4,
) -> pd.DataFrame:
    """
    Compute ripple quality-control metrics from raw LFP and ripple intervals.

    Metrics include band-energy ratios, waveform asymmetry, cycle counts,
    spectral entropy, and broadband artifact index.
    """
    if not isinstance(lfp, (nap.Tsd, nap.TsdFrame)):
        raise TypeError("lfp must be a pynapple.Tsd or pynapple.TsdFrame.")

    lfp_s = lfp.as_units("s")
    times = np.asarray(lfp_s.index.values, dtype=float)
    fs = float(lfp.rate)
    if isinstance(lfp, nap.TsdFrame):
        if channel < 0 or channel >= lfp.shape[1]:
            raise ValueError(f"channel {channel} is out of range for lfp with {lfp.shape[1]} channels.")
        signal = np.asarray(lfp_s.values[:, channel], dtype=float)
        signal_tsd = nap.Tsd(t=times, d=signal, time_support=lfp.time_support)
    else:
        signal = np.asarray(lfp_s.values, dtype=float)
        signal_tsd = nap.Tsd(t=times, d=signal, time_support=lfp.time_support)

    events = _coerce_events_array(ripples)
    if events.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "duration_s",
                "peak_time",
                "ripple_rms",
                "sharpwave_rms",
                "broadband_rms",
                "ripple_to_sharpwave_ratio",
                "ripple_to_broadband_ratio",
                "waveform_asymmetry",
                "cycle_count",
                "cycle_frequency_hz",
                "spectral_entropy",
                "broadband_peak_z",
                "phase_at_peak",
            ]
        )

    peak_times = _coerce_peak_times(ripples, events, peaks=peaks)
    rip_low, rip_high = _coerce_band_for_fs(ripple_band, fs)
    sw_low, sw_high = _coerce_band_for_fs(sharp_wave_band, fs)
    bb_low, bb_high = _coerce_band_for_fs(broadband_band, fs)

    ripple_sig = np.asarray(
        bandpass_filter(signal_tsd, rip_low, rip_high, fs, order=filter_order).as_units("s").values,
        dtype=float,
    )
    sw_sig = np.asarray(
        bandpass_filter(signal_tsd, sw_low, sw_high, fs, order=filter_order).as_units("s").values,
        dtype=float,
    )
    bb_sig = np.asarray(
        bandpass_filter(signal_tsd, bb_low, bb_high, fs, order=filter_order).as_units("s").values,
        dtype=float,
    )
    ripple_phase = np.angle(hilbert(ripple_sig))

    bb_abs = np.abs(bb_sig)
    bb_med = float(np.median(bb_abs))
    bb_mad = float(np.median(np.abs(bb_abs - bb_med))) * 1.4826
    bb_mad = max(bb_mad, _EPS)

    rows = []
    for i, (start_t, end_t) in enumerate(events):
        i0 = int(np.searchsorted(times, start_t, side="left"))
        i1 = int(np.searchsorted(times, end_t, side="right"))
        duration = float(max(0.0, end_t - start_t))
        if i1 - i0 < 3:
            rows.append(
                {
                    "start": float(start_t),
                    "end": float(end_t),
                    "duration_s": duration,
                    "peak_time": float(peak_times[i]),
                    "ripple_rms": np.nan,
                    "sharpwave_rms": np.nan,
                    "broadband_rms": np.nan,
                    "ripple_to_sharpwave_ratio": np.nan,
                    "ripple_to_broadband_ratio": np.nan,
                    "waveform_asymmetry": np.nan,
                    "cycle_count": 0,
                    "cycle_frequency_hz": np.nan,
                    "spectral_entropy": np.nan,
                    "broadband_peak_z": np.nan,
                    "phase_at_peak": np.nan,
                }
            )
            continue

        seg_rip = ripple_sig[i0:i1]
        seg_sw = sw_sig[i0:i1]
        seg_bb = bb_sig[i0:i1]
        seg_raw = signal[i0:i1]

        ripple_rms = float(np.sqrt(np.mean(seg_rip**2)))
        sharpwave_rms = float(np.sqrt(np.mean(seg_sw**2)))
        broadband_rms = float(np.sqrt(np.mean(seg_bb**2)))
        ratio_rs = ripple_rms / max(sharpwave_rms, _EPS)
        ratio_rb = ripple_rms / max(broadband_rms, _EPS)

        pos = float(np.max(seg_rip))
        neg = float(np.abs(np.min(seg_rip)))
        waveform_asymmetry = (pos - neg) / max(pos + neg, _EPS)

        zero_cross = np.flatnonzero(np.diff(np.signbit(seg_rip)))
        cycle_count = int(zero_cross.shape[0] // 2)
        cycle_frequency = cycle_count / duration if duration > 0 else np.nan

        nperseg = min(128, seg_raw.shape[0])
        freqs, psd = welch(seg_raw, fs=fs, nperseg=nperseg)
        if psd.size and np.sum(psd) > 0:
            p = psd / np.sum(psd)
            spectral_entropy = float(-np.sum(p * np.log(np.maximum(p, _EPS))) / np.log(p.shape[0]))
        else:
            spectral_entropy = np.nan

        bb_peak_z = float((np.max(np.abs(seg_bb)) - bb_med) / bb_mad)
        peak_idx = int(np.clip(_nearest_indices(times, np.array([peak_times[i]], dtype=float))[0], i0, i1 - 1))
        phase_at_peak = float(ripple_phase[peak_idx])

        rows.append(
            {
                "start": float(start_t),
                "end": float(end_t),
                "duration_s": duration,
                "peak_time": float(peak_times[i]),
                "ripple_rms": ripple_rms,
                "sharpwave_rms": sharpwave_rms,
                "broadband_rms": broadband_rms,
                "ripple_to_sharpwave_ratio": ratio_rs,
                "ripple_to_broadband_ratio": ratio_rb,
                "waveform_asymmetry": waveform_asymmetry,
                "cycle_count": cycle_count,
                "cycle_frequency_hz": cycle_frequency,
                "spectral_entropy": spectral_entropy,
                "broadband_peak_z": bb_peak_z,
                "phase_at_peak": phase_at_peak,
            }
        )

    return pd.DataFrame(rows)


def _count_spikes_in_intervals(times: np.ndarray, intervals: np.ndarray) -> tuple[int, np.ndarray]:
    counts = np.zeros(intervals.shape[0], dtype=int)
    total = 0
    for i, (start_t, end_t) in enumerate(intervals):
        i0 = int(np.searchsorted(times, start_t, side="left"))
        i1 = int(np.searchsorted(times, end_t, side="right"))
        c = max(0, i1 - i0)
        counts[i] = c
        total += c
    return total, counts


def _peri_event_rate(times: np.ndarray, events: np.ndarray, window: tuple[float, float], bin_size: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(window[0], window[1] + bin_size, bin_size, dtype=float)
    if edges.size < 2:
        raise ValueError("Invalid window/bin_size for peri-event histogram.")
    hist = np.zeros(edges.size - 1, dtype=float)
    for t0 in events:
        lo = t0 + window[0]
        hi = t0 + window[1]
        i0 = int(np.searchsorted(times, lo, side="left"))
        i1 = int(np.searchsorted(times, hi, side="right"))
        if i1 <= i0:
            continue
        rel = times[i0:i1] - t0
        h, _ = np.histogram(rel, bins=edges)
        hist += h
    n_events = max(1, events.shape[0])
    rate = hist / (n_events * bin_size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return rate, centers


def compute_ripple_spike_coupling(
    spikes: nap.TsGroup,
    ripples: Any,
    *,
    peaks: np.ndarray | None = None,
    window: tuple[float, float] = (-0.1, 0.1),
    bin_size: float = 0.005,
) -> dict[str, Any]:
    """
    Compute per-unit ripple coupling summaries and peri-event firing rates.

    Returns dictionary with:
    - `summary`: pandas DataFrame indexed by unit id
    - `bins`: peri-event bin centers
    - `peri_event_rate`: DataFrame (rows=bins, cols=unit ids, values in Hz)
    """
    if not isinstance(spikes, nap.TsGroup):
        raise TypeError("spikes must be a pynapple.TsGroup.")
    if window[0] >= window[1]:
        raise ValueError("window must be an increasing (start, end) tuple.")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    intervals = _coerce_events_array(ripples)
    event_peaks = _coerce_peak_times(ripples, intervals, peaks=peaks)

    if intervals.shape[0] == 0:
        return {
            "summary": pd.DataFrame(
                columns=[
                    "n_spikes_in_ripple",
                    "n_spikes_total",
                    "participation_count",
                    "participation_probability",
                    "rate_in_ripple_hz",
                    "rate_out_ripple_hz",
                    "rate_modulation_index",
                ]
            ),
            "bins": np.array([], dtype=float),
            "peri_event_rate": pd.DataFrame(),
        }

    ripple_duration = float(np.sum(intervals[:, 1] - intervals[:, 0]))
    total_duration = float(spikes.time_support.tot_length())
    out_duration = max(total_duration - ripple_duration, 1e-12)

    unit_ids = list(spikes.keys())
    rows = []
    peri_columns: dict[Any, np.ndarray] = {}
    bins = np.array([], dtype=float)

    for unit in unit_ids:
        unit_times = np.asarray(spikes[unit].as_units("s").index.values, dtype=float)
        n_total = int(unit_times.shape[0])
        n_in, interval_counts = _count_spikes_in_intervals(unit_times, intervals)
        participation_count = int(np.sum(interval_counts > 0))
        participation_probability = participation_count / max(1, intervals.shape[0])
        n_out = max(0, n_total - n_in)

        rate_in = n_in / max(ripple_duration, 1e-12)
        rate_out = n_out / out_duration
        denom = rate_in + rate_out
        modulation = (rate_in - rate_out) / denom if denom > 0 else np.nan

        rate_hist, centers = _peri_event_rate(unit_times, event_peaks, window, bin_size)
        if bins.size == 0:
            bins = centers
        peri_columns[unit] = rate_hist

        rows.append(
            {
                "unit": unit,
                "n_spikes_in_ripple": n_in,
                "n_spikes_total": n_total,
                "participation_count": participation_count,
                "participation_probability": participation_probability,
                "rate_in_ripple_hz": rate_in,
                "rate_out_ripple_hz": rate_out,
                "rate_modulation_index": modulation,
            }
        )

    summary = pd.DataFrame(rows).set_index("unit")
    peri_event_rate = pd.DataFrame(peri_columns, index=bins)
    peri_event_rate.index.name = "time_from_peak_s"
    return {"summary": summary, "bins": bins, "peri_event_rate": peri_event_rate}


def detect_swr(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for `detect_swr_jlong`."""
    return detect_swr_jlong(*args, **kwargs)


def ripple_stats(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Alias for `compute_ripple_event_stats`."""
    return compute_ripple_event_stats(*args, **kwargs)


def ripple_feature_stats(*args: Any, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Alias for `compute_ripple_feature_stats`."""
    return compute_ripple_feature_stats(*args, **kwargs)


def ripple_event_schema(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for `standardize_ripple_events`."""
    return standardize_ripple_events(*args, **kwargs)


def ripple_quality_metrics(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Alias for `compute_ripple_quality_metrics`."""
    return compute_ripple_quality_metrics(*args, **kwargs)


def ripple_spike_coupling(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for `compute_ripple_spike_coupling`."""
    return compute_ripple_spike_coupling(*args, **kwargs)
