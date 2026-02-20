from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from pynacollada import (
    IsEvents,
    LoadEvents,
    SaveEvents,
    is_events,
    load_events,
    save_events,
)


def _example_events() -> dict[str, object]:
    return {
        "timestamps": np.array([[0.10, 0.16], [0.52, 0.60]], dtype=float),
        "peaks": np.array([0.13, 0.56], dtype=float),
        "detectorinfo": {
            "detectorname": "unit_test_detector",
            "detectionparms": {"threshold": 3.0},
            "detectionintervals": np.array([[0.0, 1.0]], dtype=float),
            "detectiondate": "2026-02-20",
        },
    }


def test_is_events_validation() -> None:
    events = _example_events()
    assert is_events(events) is True
    assert IsEvents(events) is True
    assert is_events({"timestamps": np.array([1.0])}) is False


def test_save_and_load_events_roundtrip(tmp_path) -> None:
    session_dir = tmp_path / "sessionA"
    session_dir.mkdir()
    events = _example_events()

    file_path = save_events(session_dir, events, "ripples")
    loaded, loaded_path = load_events(session_dir, "ripples")

    assert loaded_path == file_path
    assert isinstance(loaded, Mapping)
    assert "timestamps" in loaded
    np.testing.assert_allclose(np.asarray(loaded["timestamps"], dtype=float), np.asarray(events["timestamps"], dtype=float))
    assert is_events(loaded) is True


def test_load_events_autodiscovery_and_missing(tmp_path) -> None:
    session_dir = tmp_path / "sessionB"
    session_dir.mkdir()
    save_events(session_dir, _example_events(), "alpha")
    save_events(session_dir, _example_events(), "beta")

    loaded, loaded_path = load_events(session_dir)
    assert isinstance(loaded, Mapping)
    assert loaded_path is not None
    assert loaded_path.endswith(".alpha.events.mat")

    missing, missing_path = load_events(session_dir, "gamma")
    assert missing is None
    assert missing_path is not None
    assert missing_path.endswith(".gamma.events.mat")


def test_matlab_aliases_and_overwrite_behavior(tmp_path) -> None:
    session_dir = tmp_path / "sessionC"
    session_dir.mkdir()
    events = _example_events()

    path = SaveEvents(session_dir, events, "ripples")
    loaded, loaded_path = LoadEvents(session_dir, "ripples")
    assert loaded_path == path
    assert isinstance(loaded, Mapping)

    with pytest.raises(FileExistsError):
        save_events(session_dir, events, "ripples")
    save_events(session_dir, events, "ripples", overwrite=True)
