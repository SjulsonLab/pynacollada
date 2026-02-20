"""Unit/channel query helpers inspired by FMAT GetUnits/GetChannels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .parameters import load_parameters
from .spikes import get_spikes
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


def _resolve_base_path(base_path: str | Path | None) -> Path:
    if base_path is None:
        current = get_current_session()
        base = current.base_path if current is not None else Path.cwd()
    else:
        base = Path(base_path)
    if base.is_file():
        base = base.parent
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"base_path does not exist or is not a directory: {base}")
    return base


def get_channels(
    groups: np.ndarray | list[int] | tuple[int, ...] | None = None,
    *,
    base_path: str | Path | None = None,
) -> np.ndarray:
    """
    Return channel ids for selected spike groups.

    Spike-group ids are one-indexed, matching FMAT conventions.
    """
    base = _resolve_base_path(base_path)
    params = load_parameters(base)
    elec_groups = params.get("ElecGp", [])
    if not isinstance(elec_groups, list):
        return np.asarray(params.get("channels", []), dtype=int).reshape(-1)
    n_groups = len(elec_groups)

    if groups is None:
        selected = np.arange(1, n_groups + 1, dtype=int)
    else:
        selected = np.asarray(groups, dtype=int).reshape(-1)
        if selected.size == 0:
            return np.array([], dtype=int)
        if np.any(selected < 1) or np.any(selected > n_groups):
            raise ValueError("groups must be within [1, n_groups].")

    if selected.size == 0:
        return np.array([], dtype=int)
    channels = [np.asarray(elec_groups[g - 1], dtype=int).reshape(-1) for g in selected]
    if not channels:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(channels, axis=0)).astype(int)


def get_units(
    groups: np.ndarray | list[int] | tuple[int, ...] | None = None,
    *,
    base_path: str | Path | None = None,
    basename: str | None = None,
    source: str = "auto",
    include_noise: bool = False,
) -> np.ndarray:
    """
    Return unique `[shankID, cluID]` unit pairs.

    By default this excludes clusters 0 and 1, matching FMAT behavior.
    """
    spikes = get_spikes(
        base_path=base_path,
        basename=basename,
        source=source,
        as_tsgroup=False,
    )
    shank = np.asarray(spikes.get("shankID", []), dtype=int).reshape(-1)
    clu = np.asarray(spikes.get("cluID", []), dtype=int).reshape(-1)
    if shank.size == 0 or clu.size == 0:
        return np.empty((0, 2), dtype=int)
    units = np.unique(np.column_stack((shank, clu)), axis=0)
    if not include_noise:
        units = units[(units[:, 1] != 0) & (units[:, 1] != 1)]

    if groups is not None:
        selected = np.asarray(groups, dtype=int).reshape(-1)
        if selected.size == 0:
            return np.empty((0, 2), dtype=int)
        units = units[np.isin(units[:, 0], selected)]
    return units.astype(int)


def GetChannels(groups: np.ndarray | list[int] | tuple[int, ...] | None = None, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`get_channels`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "base_path": options.pop("basepath", options.pop("base_path", None)),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_channels(groups=groups, **mapped)


def GetUnits(groups: np.ndarray | list[int] | tuple[int, ...] | None = None, *args: Any, **kwargs: Any) -> np.ndarray:
    """MATLAB-style alias for :func:`get_units`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "basename": options.pop("basename", None),
        "source": options.pop("source", "auto"),
        "include_noise": bool(options.pop("include_noise", options.pop("includenoise", False))),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_units(groups=groups, **mapped)


__all__ = [
    "get_channels",
    "get_units",
    "GetChannels",
    "GetUnits",
]
