"""Sharp-wave ripple detection and ripple event analytics for pynacollada."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.signal import lfilter, savgol_filter, welch
from sklearn.cluster import KMeans

from .archive.eeg_processing.eeg_processing import (
    bandpass_filter as _archive_bandpass_filter,
)
from .archive.eeg_processing.eeg_processing import (
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


def detect_swr(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for `detect_swr_jlong`."""
    return detect_swr_jlong(*args, **kwargs)


def ripple_stats(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Alias for `compute_ripple_event_stats`."""
    return compute_ripple_event_stats(*args, **kwargs)
