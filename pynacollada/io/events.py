"""FMAT/buzcode-style event structure IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat, savemat


def _session_basename(base_path: Path, basename: str | None = None) -> str:
    if basename is not None and str(basename).strip():
        return str(basename).strip()
    return base_path.resolve().name


def is_events(events: Mapping[str, Any]) -> bool:
    """
    Validate a buzcode-style events mapping.

    This checks for core `detectorinfo` and `timestamps` fields.
    """
    if not isinstance(events, Mapping):
        return False
    if "detectorinfo" not in events or "timestamps" not in events:
        return False

    detectorinfo = events["detectorinfo"]
    timestamps = np.asarray(events["timestamps"])
    if not isinstance(detectorinfo, Mapping):
        return False
    if not np.issubdtype(timestamps.dtype, np.number):
        return False

    required = {"detectorname", "detectionparms", "detectionintervals", "detectiondate"}
    return required.issubset(set(detectorinfo.keys()))


def load_events(
    base_path: str | Path,
    events_name: str | None = None,
    *,
    basename: str | None = None,
) -> tuple[Any, str | None]:
    """
    Load session events from `<basename>.<events_name>.events.mat`.

    If `events_name` is omitted, the first matching events file is loaded.
    """
    path = Path(base_path)
    if path.is_file():
        if events_name is not None:
            raise ValueError("events_name must be omitted when base_path points to a file.")
        file_path = path
    else:
        if not path.exists():
            raise FileNotFoundError(f"Session path does not exist: {path}")
        base = _session_basename(path, basename)
        if events_name is None:
            matches = sorted(path.glob(f"{base}.*.events.mat"))
            if not matches:
                return None, None
            file_path = matches[0]
        else:
            file_path = path / f"{base}.{events_name}.events.mat"

    if not file_path.exists():
        return None, str(file_path)

    loaded = loadmat(file_path, simplify_cells=True)
    vars_in_file = [key for key in loaded.keys() if not key.startswith("__")]
    if len(vars_in_file) == 1:
        events = loaded[vars_in_file[0]]
    else:
        events = {key: loaded[key] for key in vars_in_file}
    return events, str(file_path)


def save_events(
    base_path: str | Path,
    events: Mapping[str, Any],
    events_name: str,
    *,
    basename: str | None = None,
    variable_name: str | None = None,
    overwrite: bool = False,
) -> str:
    """
    Save a buzcode-style events mapping to `<basename>.<events_name>.events.mat`.
    """
    if not isinstance(events, Mapping):
        raise TypeError("events must be a mapping.")
    if not isinstance(events_name, str) or not events_name.strip():
        raise ValueError("events_name must be a non-empty string.")

    path = Path(base_path)
    if path.is_file():
        raise ValueError("base_path must point to a session directory, not a file.")
    if not path.exists():
        raise FileNotFoundError(f"Session path does not exist: {path}")

    base = _session_basename(path, basename)
    file_path = path / f"{base}.{events_name}.events.mat"
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {file_path}")

    key = variable_name if variable_name is not None else events_name
    if not isinstance(key, str) or not key.strip():
        raise ValueError("variable_name must be a non-empty string when provided.")

    savemat(file_path, {key: dict(events)}, do_compression=True)
    return str(file_path)


def LoadEvents(basePath: str | Path, eventsName: str | None = None, *args: Any, **kwargs: Any) -> tuple[Any, str | None]:
    """MATLAB-style alias for :func:`load_events`."""
    if args:
        raise TypeError("Unexpected positional arguments.")
    basename = kwargs.pop("basename", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return load_events(basePath, eventsName, basename=basename)


def SaveEvents(
    basePath: str | Path,
    events: Mapping[str, Any],
    eventsName: str,
    *args: Any,
    **kwargs: Any,
) -> str:
    """MATLAB-style alias for :func:`save_events`."""
    if args:
        raise TypeError("Unexpected positional arguments.")
    basename = kwargs.pop("basename", None)
    variable_name = kwargs.pop("variable_name", kwargs.pop("variableName", None))
    overwrite = bool(kwargs.pop("overwrite", False))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return save_events(
        basePath,
        events,
        eventsName,
        basename=basename,
        variable_name=variable_name,
        overwrite=overwrite,
    )


def IsEvents(events: Mapping[str, Any]) -> bool:
    """MATLAB-style alias for :func:`is_events`."""
    return is_events(events)


__all__ = [
    "is_events",
    "load_events",
    "save_events",
    "IsEvents",
    "LoadEvents",
    "SaveEvents",
]
