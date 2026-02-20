from __future__ import annotations

import numpy as np
import pytest

from pynacollada import (
    LoadBinary,
    LoadBinaryChunk,
    load_binary,
    load_binary_chunk,
)


def _write_multiplexed(path: str, n_samples: int = 24, n_channels: int = 4) -> np.ndarray:
    data = (
        100 * np.arange(n_samples, dtype=np.int16)[:, None]
        + np.arange(1, n_channels + 1, dtype=np.int16)[None, :]
    )
    data.tofile(path)
    return data


def test_load_binary_full_read(tmp_path) -> None:
    path = tmp_path / "raw.dat"
    expected = _write_multiplexed(str(path))
    out = load_binary(path, n_channels=4, precision="int16")
    np.testing.assert_array_equal(out, expected)


def test_load_binary_subset_modes_and_channel_selection(tmp_path) -> None:
    path = tmp_path / "raw.dat"
    expected = _write_multiplexed(str(path))

    out_samples = load_binary(path, n_channels=4, channels=[1, 3], offset=5, samples=6)
    np.testing.assert_array_equal(out_samples, expected[5:11][:, [0, 2]])

    out_time = load_binary(path, n_channels=4, channels=[2, 4], frequency=10.0, start=0.4, duration=0.3)
    np.testing.assert_array_equal(out_time, expected[4:7][:, [1, 3]])


def test_load_binary_rejects_mixed_time_and_sample_modes(tmp_path) -> None:
    path = tmp_path / "raw.dat"
    _write_multiplexed(str(path))
    with pytest.raises(ValueError):
        load_binary(path, n_channels=4, frequency=10.0, start=0.2, samples=3)


def test_load_binary_downsample_matches_stride(tmp_path) -> None:
    path = tmp_path / "raw.dat"
    expected = _write_multiplexed(str(path), n_samples=25, n_channels=4)
    out = load_binary(path, n_channels=4, downsample=3)
    np.testing.assert_array_equal(out, expected[::3][: out.shape[0]])


def test_load_binary_chunk_sequential_and_seeked_reads(tmp_path) -> None:
    path = tmp_path / "raw.dat"
    expected = _write_multiplexed(str(path))

    with path.open("rb") as handle:
        chunk1 = load_binary_chunk(handle, n_channels=4, frequency=10.0, duration=0.2)
        chunk2 = load_binary_chunk(handle, n_channels=4, frequency=10.0, duration=0.3, channels=[2, 4])
    np.testing.assert_array_equal(chunk1, expected[:2])
    np.testing.assert_array_equal(chunk2, expected[2:5][:, [1, 3]])

    with path.open("rb") as handle:
        chunk3 = load_binary_chunk(handle, n_channels=4, frequency=10.0, start=0.6, duration=0.2, channels=[1])
    np.testing.assert_array_equal(chunk3, expected[6:8][:, [0]])


def test_matlab_style_aliases(tmp_path) -> None:
    path = tmp_path / "raw.dat"
    expected = _write_multiplexed(str(path))

    out = LoadBinary(path, "nChannels", 4, "channels", [2, 4], "offset", 1, "samples", 3)
    np.testing.assert_array_equal(out, expected[1:4][:, [1, 3]])

    with path.open("rb") as handle:
        chunk = LoadBinaryChunk(handle, "nChannels", 4, "frequency", 10.0, "duration", 0.2, "channels", [1, 3])
    np.testing.assert_array_equal(chunk, expected[:2][:, [0, 2]])
