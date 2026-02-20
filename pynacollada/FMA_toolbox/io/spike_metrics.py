"""Spike waveform/amplitude convenience wrappers built on top of GetSpikes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .spikes import get_spikes


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


def get_spike_waveforms(
    unit: np.ndarray | list[int] | tuple[int, int],
    *,
    base_path: str | Path | None = None,
    basename: str | None = None,
    source: str = "auto",
    filtered: bool = False,
    waveform_window_s: tuple[float, float] = (0.001, 0.002),
    waveform_max_spikes: int = 1000,
    waveform_sample_mode: str = "deterministic",
    waveform_random_seed: int | None = None,
    waveform_highpass_hz: float = 500.0,
) -> np.ndarray:
    """
    Return a first-pass mean waveform for one `[group, cluster]` unit.
    """
    u = np.asarray(unit, dtype=int).reshape(-1)
    if u.size != 2:
        raise ValueError("unit must be a [group, cluster] pair.")
    spikes = get_spikes(
        units=u.reshape(1, 2),
        base_path=base_path,
        basename=basename,
        source=source,
        get_waveforms=True,
        waveform_window_s=waveform_window_s,
        waveform_max_spikes=waveform_max_spikes,
        waveform_sample_mode=waveform_sample_mode,
        waveform_random_seed=waveform_random_seed,
        waveform_highpass_hz=waveform_highpass_hz,
        as_tsgroup=False,
    )
    key = "filtWaveform" if filtered else "rawWaveform"
    waveforms = spikes.get(key, [])
    if not waveforms:
        return np.array([], dtype=float)
    return np.asarray(waveforms[0], dtype=float).reshape(-1)


def get_spike_amplitudes(
    units: np.ndarray | list[list[int]] | None = None,
    *,
    base_path: str | Path | None = None,
    basename: str | None = None,
    source: str = "auto",
    filtered: bool = False,
    waveform_window_s: tuple[float, float] = (0.001, 0.002),
    waveform_max_spikes: int = 1000,
    waveform_sample_mode: str = "deterministic",
    waveform_random_seed: int | None = None,
    waveform_highpass_hz: float = 500.0,
) -> np.ndarray:
    """
    Return per-unit peak-to-trough amplitudes.

    Output columns: `[shankID, cluID, UID, amplitude]`.
    """
    spikes = get_spikes(
        units=units,
        base_path=base_path,
        basename=basename,
        source=source,
        get_waveforms=True,
        waveform_window_s=waveform_window_s,
        waveform_max_spikes=waveform_max_spikes,
        waveform_sample_mode=waveform_sample_mode,
        waveform_random_seed=waveform_random_seed,
        waveform_highpass_hz=waveform_highpass_hz,
        as_tsgroup=False,
    )
    uid = np.asarray(spikes.get("UID", []), dtype=int).reshape(-1)
    shank = np.asarray(spikes.get("shankID", []), dtype=int).reshape(-1)
    clu = np.asarray(spikes.get("cluID", []), dtype=int).reshape(-1)
    key = "filtWaveform" if filtered else "rawWaveform"
    waveforms = spikes.get(key, [])
    if uid.size == 0 or len(waveforms) == 0:
        return np.empty((0, 4), dtype=float)
    amps = np.array(
        [
            (float(np.nanmax(np.asarray(wf, dtype=float)) - np.nanmin(np.asarray(wf, dtype=float)))
             if np.asarray(wf, dtype=float).size > 0
             else np.nan)
            for wf in waveforms
        ],
        dtype=float,
    )
    return np.column_stack((shank.astype(float), clu.astype(float), uid.astype(float), amps))


def GetSpikeWaveforms(unit: np.ndarray | list[int] | tuple[int, int], *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`get_spike_waveforms`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "basename": options.pop("basename", None),
        "source": options.pop("source", "auto"),
        "filtered": bool(options.pop("filtered", False)),
        "waveform_window_s": tuple(options.pop("waveform_window_s", options.pop("waveformwindows", (0.001, 0.002)))),
        "waveform_max_spikes": int(options.pop("waveform_max_spikes", options.pop("waveformmaxspikes", 1000))),
        "waveform_sample_mode": options.pop("waveform_sample_mode", options.pop("waveformsamplemode", "deterministic")),
        "waveform_random_seed": options.pop("waveform_random_seed", options.pop("waveformrandomseed", None)),
        "waveform_highpass_hz": float(options.pop("waveform_highpass_hz", options.pop("waveformhphz", 500.0))),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_spike_waveforms(unit, **mapped)


def GetSpikeAmplitudes(
    units: np.ndarray | list[list[int]] | None = None,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """MATLAB-style alias for :func:`get_spike_amplitudes`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "basename": options.pop("basename", None),
        "source": options.pop("source", "auto"),
        "filtered": bool(options.pop("filtered", False)),
        "waveform_window_s": tuple(options.pop("waveform_window_s", options.pop("waveformwindows", (0.001, 0.002)))),
        "waveform_max_spikes": int(options.pop("waveform_max_spikes", options.pop("waveformmaxspikes", 1000))),
        "waveform_sample_mode": options.pop("waveform_sample_mode", options.pop("waveformsamplemode", "deterministic")),
        "waveform_random_seed": options.pop("waveform_random_seed", options.pop("waveformrandomseed", None)),
        "waveform_highpass_hz": float(options.pop("waveform_highpass_hz", options.pop("waveformhphz", 500.0))),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_spike_amplitudes(units, **mapped)


__all__ = [
    "get_spike_waveforms",
    "get_spike_amplitudes",
    "GetSpikeWaveforms",
    "GetSpikeAmplitudes",
]
