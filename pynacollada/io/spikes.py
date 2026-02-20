"""FMAT-style spike timestamp readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .parameters import load_parameters
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


def _parse_base_and_group(filename: str | Path) -> tuple[Path, str, int]:
    path = Path(filename)
    parts = path.name.split(".")
    if len(parts) < 3 or parts[-2].lower() not in {"res", "clu"}:
        raise ValueError("filename must point to a .res.N or .clu.N file.")
    try:
        group = int(parts[-1])
    except ValueError as exc:
        raise ValueError("Could not parse electrode-group index from filename.") from exc
    base_name = ".".join(parts[:-2])
    return path.parent, base_name, group


def _load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.asarray(np.loadtxt(path, dtype=float), dtype=float).reshape(-1)
    return arr


def load_spike_times(filename: str | Path, rate: float) -> np.ndarray:
    """
    Load spike timestamps from paired `.res.N` and `.clu.N` files.

    Returns columns `[timestamp_s, electrode_group, cluster]`.
    """
    if rate <= 0:
        raise ValueError("rate must be > 0.")
    folder, base_name, group = _parse_base_and_group(filename)
    res_file = folder / f"{base_name}.res.{group}"
    clu_file = folder / f"{base_name}.clu.{group}"

    res = _load_vector(res_file)
    clu_raw = _load_vector(clu_file)
    if clu_raw.size == res.size + 1:
        clu = clu_raw[1:]
    elif clu_raw.size == res.size:
        clu = clu_raw
    else:
        raise ValueError(
            f"Inconsistent .res/.clu lengths for group {group}: "
            f"{res.size} timestamps vs {clu_raw.size} cluster entries."
        )

    if res.size == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack((res / float(rate), group * np.ones(res.size, dtype=float), clu.astype(float)))


def _discover_groups(base_path: Path, basename: str | None) -> tuple[str | None, list[int]]:
    if basename is not None and str(basename).strip():
        base = str(basename).strip()
        matches = sorted(base_path.glob(f"{base}.res.*"))
    else:
        inferred = base_path.resolve().name
        matches = sorted(base_path.glob(f"{inferred}.res.*"))
        if not matches:
            matches = sorted(base_path.glob("*.res.*"))

    by_base: dict[str, set[int]] = {}
    for match in matches:
        parts = match.name.split(".")
        if len(parts) < 3 or parts[-2] != "res":
            continue
        try:
            group = int(parts[-1])
        except ValueError:
            continue
        base = ".".join(parts[:-2])
        clu = base_path / f"{base}.clu.{group}"
        if not clu.exists():
            continue
        by_base.setdefault(base, set()).add(group)

    if not by_base:
        return None, []
    if basename is not None:
        selected = str(basename).strip()
        if selected not in by_base:
            return selected, []
        return selected, sorted(by_base[selected])
    if len(by_base) > 1:
        raise ValueError("Multiple spike basenames found; specify basename explicitly.")
    selected = next(iter(by_base.keys()))
    return selected, sorted(by_base[selected])


def get_spike_times(
    units: np.ndarray | list[list[int]] | None = None,
    *,
    base_path: str | Path | None = None,
    basename: str | None = None,
    rate: float | None = None,
    output: str = "time",
) -> np.ndarray:
    """
    Load and optionally filter spikes from `.res/.clu` files.

    Output modes:
    - `time`: timestamps only
    - `full`: timestamps + electrode group + cluster
    - `numbered`: timestamps + unique unit id
    - `total`: timestamps + electrode group + cluster + unique unit id
    """
    out_mode = str(output).lower()
    if out_mode not in {"time", "full", "numbered", "total"}:
        raise ValueError("output must be one of {'time','full','numbered','total'}.")

    if base_path is None:
        current = get_current_session()
        base = current.base_path if current is not None else Path.cwd()
    else:
        base = Path(base_path)
    if base.is_file():
        base = base.parent
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"base_path does not exist or is not a directory: {base}")

    if rate is None:
        params = load_parameters(base)
        rate = float(params["rates"]["wideband"])
    if rate <= 0:
        raise ValueError("rate must be > 0.")

    selected_base, groups = _discover_groups(base, basename)
    if selected_base is None or not groups:
        if out_mode == "time":
            return np.array([], dtype=float)
        if out_mode == "numbered":
            return np.empty((0, 2), dtype=float)
        if out_mode == "full":
            return np.empty((0, 3), dtype=float)
        return np.empty((0, 4), dtype=float)

    chunks = [load_spike_times(base / f"{selected_base}.res.{group}", float(rate)) for group in groups]
    spikes = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=float)
    if spikes.size == 0:
        if out_mode == "time":
            return np.array([], dtype=float)
        if out_mode == "numbered":
            return np.empty((0, 2), dtype=float)
        if out_mode == "full":
            return np.empty((0, 3), dtype=float)
        return np.empty((0, 4), dtype=float)
    order = np.argsort(spikes[:, 0], kind="mergesort")
    spikes = spikes[order]

    if units is not None and np.asarray(units).size > 0:
        u = np.asarray(units, dtype=int)
        if u.ndim == 1:
            if u.size != 2:
                raise ValueError("units must be Nx2 [group, cluster] pairs.")
            u = u.reshape(1, 2)
        if u.ndim != 2 or u.shape[1] != 2:
            raise ValueError("units must be Nx2 [group, cluster] pairs.")

        selected = np.zeros(spikes.shape[0], dtype=bool)
        groups_col = spikes[:, 1].astype(int)
        clusters_col = spikes[:, 2].astype(int)
        for group, cluster in u:
            if cluster == -1:
                mask = (groups_col == group) & (clusters_col != 0) & (clusters_col != 1)
            elif cluster == -2:
                mask = (groups_col == group) & (clusters_col != 0)
            elif cluster == -3:
                mask = groups_col == group
            else:
                mask = (groups_col == group) & (clusters_col == cluster)
            selected |= mask
        spikes = spikes[selected]

    if out_mode == "time":
        return spikes[:, 0]
    if out_mode == "full":
        return spikes

    pairs = spikes[:, 1:3].astype(int, copy=False)
    _, inv = np.unique(pairs, axis=0, return_inverse=True)
    unit_id = (inv + 1).astype(float)
    if out_mode == "numbered":
        return np.column_stack((spikes[:, 0], unit_id))
    return np.column_stack((spikes, unit_id))


def LoadSpikeTimes(filename: str | Path, rate: float) -> np.ndarray:
    """MATLAB-style alias for :func:`load_spike_times`."""
    return load_spike_times(filename, rate)


def GetSpikeTimes(
    units: np.ndarray | list[list[int]] | str | None = None,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """MATLAB-style alias for :func:`get_spike_times`."""
    if isinstance(units, str):
        args = (units, *args)
        units = None
    options = _collect_options(args, kwargs)
    mapped = {
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "basename": options.pop("basename", None),
        "rate": options.pop("rate", None),
        "output": options.pop("output", "time"),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_spike_times(units=units, **mapped)


__all__ = [
    "load_spike_times",
    "get_spike_times",
    "LoadSpikeTimes",
    "GetSpikeTimes",
]
