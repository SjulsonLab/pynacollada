from __future__ import annotations

import numpy as np

from pynacollada import (
    GetEventTypes,
    GetEvents,
    clear_current_session,
    get_event_types,
    get_events,
    save_events,
    set_current_session,
)


def _write_events(session_dir, basename: str = "sessionEvt") -> None:
    events = {
        "time": np.array([[0.10, 0.12], [0.30, 0.32], [0.50, 0.52]], dtype=float),
        "description": np.array(["Ripple", "Sharp Wave", "Ripple beginning"], dtype=object),
        "detectorinfo": {
            "detectorname": "detectorX",
            "detectionparms": {"threshold": 3.0},
            "detectionintervals": np.array([[0.0, 1.0]], dtype=float),
            "detectiondate": "2026-02-20",
        },
    }
    save_events(session_dir, events, "evt", basename=basename)


def test_get_events_outputs_and_regex_selection(tmp_path) -> None:
    session = tmp_path / "sessionEvt"
    session.mkdir()
    _write_events(session)

    all_times = get_events(base_path=session, events_name="evt", output="times")
    assert all_times.shape == (3, 2)

    ripple_only = get_events("Ripple", base_path=session, events_name="evt", output="times")
    assert ripple_only.shape[0] == 1
    np.testing.assert_allclose(ripple_only[0], np.array([0.10, 0.12]))

    idx = get_events(["Ripple", "Sharp Wave"], base_path=session, events_name="evt", output="indices")
    np.testing.assert_array_equal(idx, np.array([0, 1], dtype=int))

    logical = get_events([".*beginning"], base_path=session, events_name="evt", output="logical")
    np.testing.assert_array_equal(logical, np.array([False, False, True]))

    desc = get_events(base_path=session, events_name="evt", output="descriptions")
    assert desc == ["Ripple", "Ripple beginning", "Sharp Wave"]


def test_get_event_types_and_aliases_with_current_session(tmp_path) -> None:
    session = tmp_path / "sessionEvt"
    session.mkdir()
    _write_events(session)
    clear_current_session()
    set_current_session(session)

    types = get_event_types(["Ripple", ".*beginning"], events_name="evt")
    assert types == ["Ripple", "Ripple beginning"]

    alias_idx = GetEvents(["Sharp Wave"], "eventsName", "evt", "output", "indices")
    np.testing.assert_array_equal(alias_idx, np.array([1], dtype=int))

    alias_types = GetEventTypes([".*Wave"], eventsName="evt")
    assert alias_types == ["Sharp Wave"]

    clear_current_session()
