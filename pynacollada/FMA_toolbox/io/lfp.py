"""FMAT/buzcode-style LFP loading helpers returning pynapple objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pynapple as nap

from .binary import load_binary
from .parameters import load_parameters


def _resolve_lfp_file(base_path: Path, basename: str | None, from_dat: bool) -> tuple[str, Path]:
    if basename is not None and str(basename).strip():
        base = str(basename).strip()
        candidates = [base_path / f"{base}.dat"] if from_dat else [base_path / f"{base}.lfp", base_path / f"{base}.eeg"]
        for candidate in candidates:
            if candidate.exists():
                return base, candidate
        raise FileNotFoundError(f"Could not find LFP file for basename '{base}' in {base_path}")

    default_base = base_path.resolve().name
    candidates = [base_path / f"{default_base}.dat"] if from_dat else [base_path / f"{default_base}.lfp", base_path / f"{default_base}.eeg"]
    for candidate in candidates:
        if candidate.exists():
            return default_base, candidate

    patterns = ["*.dat"] if from_dat else ["*.lfp", "*.eeg"]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(base_path.glob(pattern)))
    if len(matches) == 1:
        return matches[0].stem, matches[0]
    raise FileNotFoundError(f"Expected one LFP file in {base_path}, found {len(matches)} candidates.")


def _as_intervalset(intervals: Any) -> nap.IntervalSet | None:
    if intervals is None:
        return None
    if isinstance(intervals, nap.IntervalSet):
        return intervals
    arr = np.asarray(intervals, dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("intervals must be Nx2.")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("intervals must be Nx2.")
    return nap.IntervalSet(start=arr[:, 0], end=arr[:, 1], time_units="s")


def _collect_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2 != 0:
        raise ValueError("Positional options must be key/value pairs.")
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


def get_lfp(
    channels: list[int] | np.ndarray | str,
    *,
    base_path: str | Path | None = None,
    basename: str | None = None,
    intervals: Any = None,
    restrict: Any = None,
    downsample: int = 1,
    from_dat: bool = False,
    precision: str = "int16",
) -> nap.Tsd | nap.TsdFrame:
    """
    Load LFP/EEG data and return pynapple objects.

    `channels` are zero-indexed, matching buzcode conventions.
    """
    if downsample <= 0:
        raise ValueError("downsample must be > 0.")

    base = Path.cwd() if base_path is None else Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"base_path does not exist: {base}")
    if not base.is_dir():
        raise ValueError("base_path must be a session directory.")

    _, signal_file = _resolve_lfp_file(base, basename, from_dat)
    params = load_parameters(base)
    n_channels = int(params["nChannels"])
    if n_channels <= 0:
        raise ValueError("Could not infer a positive nChannels from session parameters.")

    if isinstance(channels, str):
        if channels.lower() != "all":
            raise ValueError("String channels value must be 'all'.")
        selected = np.asarray(params.get("channels", np.arange(n_channels, dtype=int)), dtype=int).reshape(-1)
    else:
        selected = np.asarray(channels, dtype=int).reshape(-1)
        if selected.size == 0:
            selected = np.asarray(params.get("channels", np.arange(n_channels, dtype=int)), dtype=int).reshape(-1)
    if np.any(selected < 0) or np.any(selected >= n_channels):
        raise ValueError("channels must be in [0, nChannels-1].")

    rates = params.get("rates", {})
    fs = float(rates.get("wideband") if from_dat else rates.get("lfp"))
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")
    fs_out = fs / float(downsample)
    if fs_out <= 0:
        raise ValueError("downsample leads to non-positive output sampling rate.")

    data = load_binary(
        signal_file,
        frequency=fs,
        n_channels=n_channels,
        channels=(selected + 1).tolist(),
        precision=precision,
        downsample=downsample,
    )
    t = np.arange(data.shape[0], dtype=float) / fs_out

    if data.shape[1] == 1:
        out: nap.Tsd | nap.TsdFrame = nap.Tsd(t=t, d=data[:, 0], time_units="s")
    else:
        out = nap.TsdFrame(t=t, d=data, time_units="s", columns=selected.tolist())

    interval_obj = _as_intervalset(intervals if intervals is not None else restrict)
    if interval_obj is not None:
        out = out.restrict(interval_obj)
    return out


def GetLFP(channels: list[int] | np.ndarray | str, *args: Any, **kwargs: Any) -> nap.Tsd | nap.TsdFrame:
    """MATLAB-style alias for :func:`get_lfp`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "basename": options.pop("basename", None),
        "intervals": options.pop("intervals", None),
        "restrict": options.pop("restrict", None),
        "downsample": options.pop("downsample", 1),
        "from_dat": options.pop("fromdat", options.pop("from_dat", False)),
        "precision": options.pop("precision", "int16"),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_lfp(channels, **mapped)


__all__ = [
    "get_lfp",
    "GetLFP",
]
