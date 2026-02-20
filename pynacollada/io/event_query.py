"""Event querying helpers inspired by FMAT GetEvents/GetEventTypes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .events import load_events
from .session import get_current_session


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


def _select_mapping(events: Any) -> Mapping[str, Any]:
    if isinstance(events, Mapping):
        if "time" in events or "timestamps" in events:
            return events
        if len(events) == 1:
            only = next(iter(events.values()))
            if isinstance(only, Mapping):
                return only
    raise ValueError("Could not parse events mapping: expected 'time' or 'timestamps' fields.")


def _event_table(events: Mapping[str, Any], fallback_description: str = "event") -> tuple[np.ndarray, np.ndarray]:
    if "time" in events:
        times = np.asarray(events["time"], dtype=float)
    elif "timestamps" in events:
        times = np.asarray(events["timestamps"], dtype=float)
    else:
        raise ValueError("Events mapping has neither 'time' nor 'timestamps'.")

    if times.ndim == 0:
        times = times.reshape(1)
    if "description" in events:
        desc_list = [str(d) for d in np.asarray(events["description"], dtype=object).reshape(-1)]
    else:
        n = int(times.shape[0]) if times.ndim > 1 else int(times.size)
        desc = fallback_description
        detector = events.get("detectorinfo")
        if isinstance(detector, Mapping):
            desc = str(detector.get("detectorname", desc))
        desc_list = [desc] * n

    if times.ndim > 1 and times.shape[0] != len(desc_list):
        if times.shape[1] == len(desc_list):
            times = times.T
        else:
            raise ValueError("Event times and descriptions have incompatible lengths.")
    if times.ndim == 1 and times.size != len(desc_list):
        raise ValueError("Event times and descriptions have incompatible lengths.")
    return times, np.asarray(desc_list, dtype=object)


def _selection_mask(descriptions: np.ndarray, selection: str | list[str] | tuple[str, ...] | None) -> np.ndarray:
    if selection is None:
        patterns = [".*"]
    elif isinstance(selection, str):
        patterns = [selection]
    else:
        patterns = [str(p) for p in selection]
        if len(patterns) == 0:
            patterns = [".*"]

    mask = np.zeros(descriptions.shape[0], dtype=bool)
    for pattern in patterns:
        rx = re.compile(rf"^{pattern}$")
        mask |= np.array([bool(rx.search(str(desc))) for desc in descriptions], dtype=bool)
    return mask


def get_events(
    selection: str | list[str] | tuple[str, ...] | None = None,
    *,
    output: str = "times",
    events: Mapping[str, Any] | None = None,
    base_path: str | Path | None = None,
    events_name: str | None = None,
    basename: str | None = None,
) -> np.ndarray | list[str]:
    """
    Query events by description regex patterns.

    Output modes:
    - `times`
    - `indices` (0-based)
    - `logical`
    - `descriptions` (unique selected descriptions)
    """
    out_mode = str(output).lower()
    if out_mode not in {"times", "indices", "logical", "descriptions"}:
        raise ValueError("output must be one of {'times','indices','logical','descriptions'}.")

    ev: Mapping[str, Any] | None
    if events is not None:
        ev = events
    else:
        if base_path is None:
            current = get_current_session()
            base = current.base_path if current is not None else Path.cwd()
        else:
            base = Path(base_path)
        loaded, _ = load_events(base, events_name=events_name, basename=basename)
        if loaded is None:
            if out_mode == "times":
                return np.array([], dtype=float)
            if out_mode == "indices":
                return np.array([], dtype=int)
            if out_mode == "logical":
                return np.array([], dtype=bool)
            return []
        ev = _select_mapping(loaded)

    table = _select_mapping(ev)
    fallback = str(events_name) if events_name is not None else "event"
    times, descriptions = _event_table(table, fallback_description=fallback)
    mask = _selection_mask(descriptions, selection)

    if out_mode == "times":
        return times[mask] if times.ndim > 1 else times[mask]
    if out_mode == "indices":
        return np.flatnonzero(mask).astype(int)
    if out_mode == "logical":
        return mask
    return sorted(set(str(d) for d in descriptions[mask]))


def get_event_types(
    selection: str | list[str] | tuple[str, ...] | None = None,
    *,
    events: Mapping[str, Any] | None = None,
    base_path: str | Path | None = None,
    events_name: str | None = None,
    basename: str | None = None,
) -> list[str]:
    """Return unique event descriptions matching optional regex patterns."""
    out = get_events(
        selection=selection,
        output="descriptions",
        events=events,
        base_path=base_path,
        events_name=events_name,
        basename=basename,
    )
    return list(out) if isinstance(out, list) else [str(x) for x in np.asarray(out, dtype=object).reshape(-1)]


def GetEvents(
    selection: str | list[str] | tuple[str, ...] | None = None,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray | list[str]:
    """MATLAB-style alias for :func:`get_events`."""
    if isinstance(selection, str) and len(args) > 0 and len(args) % 2 == 1:
        args = (selection, *args)
        selection = None
    options = _collect_options(args, kwargs)
    mapped = {
        "output": options.pop("output", "times"),
        "events": options.pop("events", None),
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "events_name": options.pop("eventsname", options.pop("events_name", None)),
        "basename": options.pop("basename", None),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_events(selection=selection, **mapped)


def GetEventTypes(
    selection: str | list[str] | tuple[str, ...] | None = None,
    *args: Any,
    **kwargs: Any,
) -> list[str]:
    """MATLAB-style alias for :func:`get_event_types`."""
    if isinstance(selection, str) and len(args) > 0 and len(args) % 2 == 1:
        args = (selection, *args)
        selection = None
    options = _collect_options(args, kwargs)
    mapped = {
        "events": options.pop("events", None),
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "events_name": options.pop("eventsname", options.pop("events_name", None)),
        "basename": options.pop("basename", None),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_event_types(selection=selection, **mapped)


__all__ = [
    "get_events",
    "get_event_types",
    "GetEvents",
    "GetEventTypes",
]
