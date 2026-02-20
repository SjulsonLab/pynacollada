"""FMAT-style binary readers for multiplexed electrophysiology files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


_PRECISION_MAP: dict[str, np.dtype] = {
    "uchar": np.dtype("uint8"),
    "unsigned char": np.dtype("uint8"),
    "schar": np.dtype("int8"),
    "signed char": np.dtype("int8"),
    "int8": np.dtype("int8"),
    "uint8": np.dtype("uint8"),
    "int16": np.dtype("int16"),
    "uint16": np.dtype("uint16"),
    "int32": np.dtype("int32"),
    "uint32": np.dtype("uint32"),
    "single": np.dtype("float32"),
    "float32": np.dtype("float32"),
    "int64": np.dtype("int64"),
    "uint64": np.dtype("uint64"),
    "double": np.dtype("float64"),
    "float64": np.dtype("float64"),
}


def _resolve_dtype(precision: str | np.dtype[Any]) -> np.dtype[Any]:
    if isinstance(precision, np.dtype):
        return precision
    key = str(precision).strip().lower()
    if key not in _PRECISION_MAP:
        raise ValueError(f"Unsupported precision: {precision}")
    return _PRECISION_MAP[key]


def _resolve_channels(channels: np.ndarray | list[int] | None, n_channels: int) -> np.ndarray:
    if channels is None:
        arr = np.arange(1, n_channels + 1, dtype=int)
    else:
        arr = np.asarray(channels, dtype=int).reshape(-1)
        if arr.size == 0:
            arr = np.arange(1, n_channels + 1, dtype=int)
    if np.any(arr <= 0) or np.any(arr > n_channels):
        raise ValueError("channels must be one-indexed and within [1, n_channels].")
    return arr


def _read_frames(
    handle: BinaryIO,
    *,
    dtype: np.dtype[Any],
    n_channels: int,
    n_frames: int,
    skip: int,
) -> np.ndarray:
    sample_size = dtype.itemsize
    frame_bytes = n_channels * sample_size

    if n_frames <= 0:
        return np.empty((0, n_channels), dtype=dtype)

    if skip == 0:
        chunk = handle.read(frame_bytes * n_frames)
        n_complete = len(chunk) // frame_bytes
        if n_complete <= 0:
            return np.empty((0, n_channels), dtype=dtype)
        data_bytes = chunk[: n_complete * frame_bytes]
        data = np.frombuffer(data_bytes, dtype=dtype).reshape(n_complete, n_channels)
        return data.copy()

    stride = frame_bytes + skip
    chunk = handle.read(stride * n_frames)
    n_complete = len(chunk) // stride
    if n_complete <= 0:
        return np.empty((0, n_channels), dtype=dtype)
    raw = np.frombuffer(chunk[: n_complete * stride], dtype=np.uint8).reshape(n_complete, stride)
    kept = np.ascontiguousarray(raw[:, :frame_bytes])
    return kept.view(dtype).reshape(n_complete, n_channels)


def load_binary(
    filename: str | Path,
    *,
    frequency: float = 20000.0,
    start: float = 0.0,
    duration: float = math.inf,
    offset: int = 0,
    samples: int | float = math.inf,
    n_channels: int = 1,
    channels: np.ndarray | list[int] | None = None,
    precision: str | np.dtype[Any] = "int16",
    skip: int = 0,
    downsample: int = 1,
) -> np.ndarray:
    """
    Load multiplexed binary data with FMAT-compatible options.

    Channels are one-indexed for compatibility with MATLAB APIs.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if n_channels <= 0:
        raise ValueError("n_channels must be > 0.")
    if frequency <= 0:
        raise ValueError("frequency must be > 0.")
    if downsample <= 0:
        raise ValueError("downsample must be > 0.")
    if skip < 0:
        raise ValueError("skip must be >= 0.")

    dtype = _resolve_dtype(precision)
    channel_ids = _resolve_channels(channels, n_channels)

    time_mode = (start != 0.0) or (duration != math.inf)
    sample_mode = (offset != 0) or (samples != math.inf)
    if time_mode and sample_mode:
        raise ValueError("Specify subset with start/duration or offset/samples, not both.")
    if duration < 0:
        raise ValueError("duration must be >= 0.")
    if offset < 0:
        raise ValueError("offset must be >= 0.")
    if samples is not math.inf and float(samples) < 0:
        raise ValueError("samples must be >= 0.")

    sample_size = dtype.itemsize
    frame_bytes = n_channels * sample_size

    if time_mode:
        data_offset = int(math.floor(max(start, 0.0) * frequency)) * frame_bytes
        requested_frames = math.inf if duration == math.inf else int(math.floor(duration * frequency))
    else:
        data_offset = int(offset) * frame_bytes
        requested_frames = math.inf if samples == math.inf else int(samples)

    file_size = path.stat().st_size
    if data_offset > file_size:
        raise ValueError("Requested start position is past the end of file.")

    available_frames = (file_size - data_offset) // frame_bytes
    n_frames = int(available_frames if requested_frames == math.inf else min(requested_frames, available_frames))

    effective_skip = int(skip)
    if downsample > 1:
        # Match FMAT's downsample mode by stepping at frame granularity.
        effective_skip = n_channels * (downsample - 1) * sample_size
        n_frames = int(math.floor(n_frames / downsample))

    if n_frames <= 0:
        return np.empty((0, channel_ids.shape[0]), dtype=dtype)

    with path.open("rb") as handle:
        handle.seek(data_offset)
        data = _read_frames(
            handle,
            dtype=dtype,
            n_channels=n_channels,
            n_frames=n_frames,
            skip=effective_skip,
        )

    return data[:, channel_ids - 1]


def load_binary_chunk(
    handle: BinaryIO,
    *,
    duration: float = 1.0,
    frequency: float = 20000.0,
    start: float | None = None,
    n_channels: int = 1,
    channels: np.ndarray | list[int] | None = None,
    precision: str | np.dtype[Any] = "int16",
    skip: int = 0,
) -> np.ndarray:
    """
    Read a chunk from an already-open binary file handle.

    If `start` is `None`, reading begins at the current file pointer.
    Otherwise, the file pointer is moved to `start` seconds from file begin.
    """
    if n_channels <= 0:
        raise ValueError("n_channels must be > 0.")
    if frequency <= 0:
        raise ValueError("frequency must be > 0.")
    if duration < 0:
        raise ValueError("duration must be >= 0.")
    if skip < 0:
        raise ValueError("skip must be >= 0.")

    dtype = _resolve_dtype(precision)
    channel_ids = _resolve_channels(channels, n_channels)
    n_frames = int(math.floor(duration * frequency))
    if start is not None:
        start_frame = int(math.floor(max(start, 0.0) * frequency))
        handle.seek(start_frame * n_channels * dtype.itemsize)

    data = _read_frames(
        handle,
        dtype=dtype,
        n_channels=n_channels,
        n_frames=n_frames,
        skip=int(skip),
    )
    return data[:, channel_ids - 1]


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


def LoadBinary(filename: str | Path, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`load_binary`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "frequency": options.pop("frequency", 20000.0),
        "start": options.pop("start", 0.0),
        "duration": options.pop("duration", math.inf),
        "offset": options.pop("offset", 0),
        "samples": options.pop("samples", math.inf),
        "n_channels": options.pop("nchannels", 1),
        "channels": options.pop("channels", None),
        "precision": options.pop("precision", "int16"),
        "skip": options.pop("skip", 0),
        "downsample": options.pop("downsample", 1),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return load_binary(filename, **mapped)


def LoadBinaryChunk(handle: BinaryIO, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`load_binary_chunk`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "duration": options.pop("duration", 1.0),
        "frequency": options.pop("frequency", 20000.0),
        "start": options.pop("start", None),
        "n_channels": options.pop("nchannels", 1),
        "channels": options.pop("channels", None),
        "precision": options.pop("precision", "int16"),
        "skip": options.pop("skip", 0),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return load_binary_chunk(handle, **mapped)


__all__ = [
    "load_binary",
    "load_binary_chunk",
    "LoadBinary",
    "LoadBinaryChunk",
]
