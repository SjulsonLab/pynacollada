"""FMAT-style spike timestamp readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pynapple as nap
from scipy.io import loadmat

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


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [item for item in value.reshape(-1)]
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return [value]
        return [value[idx] for idx in range(value.shape[0])]
    return [value]


def _normalize_times(times_value: Any) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for item in _as_list(times_value):
        arr = np.asarray(item, dtype=float).reshape(-1)
        out.append(arr)
    return out


def _discover_cellinfo_file(base_path: Path, basename: str | None) -> Path | None:
    if basename is not None and str(basename).strip():
        candidate = base_path / f"{str(basename).strip()}.spikes.cellinfo.mat"
        return candidate if candidate.exists() else None

    default = base_path / f"{base_path.resolve().name}.spikes.cellinfo.mat"
    if default.exists():
        return default

    matches = sorted(base_path.glob("*.spikes.cellinfo.mat"))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    raise ValueError("Multiple .spikes.cellinfo.mat files found; specify basename explicitly.")


def _normalize_cellinfo_field(value: Any, n_units: int) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 1 and arr.shape[0] == n_units and arr.dtype != object:
        return arr
    if arr.ndim == 2 and arr.shape[0] == n_units and arr.dtype != object:
        return [np.asarray(arr[i]) for i in range(arr.shape[0])]

    items = _as_list(value)
    if len(items) != n_units:
        return value
    if all(np.isscalar(item) for item in items):
        return np.asarray(items)
    return [np.asarray(item) if isinstance(item, (list, tuple, np.ndarray)) else item for item in items]


def _load_spikes_from_cellinfo(path: Path, rate_override: float | None = None) -> dict[str, Any]:
    loaded = loadmat(path, simplify_cells=True)
    vars_in_file = [key for key in loaded.keys() if not key.startswith("__")]
    if "spikes" in loaded:
        raw = loaded["spikes"]
    elif len(vars_in_file) == 1:
        raw = loaded[vars_in_file[0]]
    else:
        raw = {key: loaded[key] for key in vars_in_file}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Unexpected cellinfo format in {path}")

    out = dict(raw)
    if "times" in out:
        times = _normalize_times(out["times"])
    elif "ts" in out and rate_override is not None and rate_override > 0:
        ts = _normalize_times(out["ts"])
        times = [np.asarray(cell, dtype=float) / float(rate_override) for cell in ts]
    elif "spindices" in out:
        sp = np.asarray(out["spindices"], dtype=float)
        if sp.ndim != 2 or sp.shape[1] < 2:
            raise ValueError(f"Invalid spindices in {path}")
        uid_from_file = np.asarray(out.get("UID", np.unique(sp[:, 1])), dtype=int).reshape(-1)
        times = [sp[sp[:, 1] == uid, 0].astype(float, copy=False) for uid in uid_from_file]
    else:
        times = []

    n_units = len(times)
    uid = np.asarray(out.get("UID", np.arange(1, n_units + 1)), dtype=int).reshape(-1)
    if uid.size != n_units:
        uid = np.arange(1, n_units + 1, dtype=int)

    spikes: dict[str, Any] = {"times": [np.asarray(t, dtype=float).reshape(-1) for t in times], "UID": uid}
    if "ts" in out:
        spikes["ts"] = _normalize_times(out["ts"])
    elif rate_override is not None and rate_override > 0:
        spikes["ts"] = [np.asarray(np.round(t * float(rate_override)), dtype=float) for t in spikes["times"]]

    for key, value in out.items():
        if key in {"times", "UID", "ts"}:
            continue
        norm = _normalize_cellinfo_field(value, n_units)
        spikes[key] = norm

    if "shankID" in spikes:
        spikes["shankID"] = np.asarray(spikes["shankID"], dtype=int).reshape(-1)
    if "cluID" in spikes:
        spikes["cluID"] = np.asarray(spikes["cluID"], dtype=int).reshape(-1)
    if "maxWaveformCh" in spikes:
        spikes["maxWaveformCh"] = np.asarray(spikes["maxWaveformCh"], dtype=int).reshape(-1)

    all_times = np.concatenate([np.asarray(t, dtype=float).reshape(-1) for t in spikes["times"]], axis=0) if spikes["times"] else np.array([], dtype=float)
    if all_times.size:
        all_groups = np.concatenate(
            [np.full(np.asarray(t, dtype=float).reshape(-1).shape, float(uid[i]), dtype=float) for i, t in enumerate(spikes["times"])],
            axis=0,
        )
        order = np.argsort(all_times, kind="mergesort")
        spikes["spindices"] = np.column_stack((all_times[order], all_groups[order]))
    else:
        spikes["spindices"] = np.empty((0, 2), dtype=float)
    spikes["numcells"] = int(uid.size)
    spikes["source"] = "cellinfo"
    if rate_override is not None:
        spikes["samplingRate"] = float(rate_override)
    return spikes


def _spikes_struct_from_full(full: np.ndarray, rate: float) -> dict[str, Any]:
    if full.size == 0:
        return {
            "times": [],
            "UID": np.array([], dtype=int),
            "shankID": np.array([], dtype=int),
            "cluID": np.array([], dtype=int),
            "numcells": 0,
            "spindices": np.empty((0, 2), dtype=float),
            "samplingRate": float(rate),
            "source": "clu_res",
        }
    pairs = full[:, 1:3].astype(int, copy=False)
    unique_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
    uid = np.arange(1, unique_pairs.shape[0] + 1, dtype=int)
    unit_times = [full[inv == i, 0].astype(float, copy=False) for i in range(unique_pairs.shape[0])]
    spindices = np.column_stack((full[:, 0], (inv + 1).astype(float)))
    return {
        "times": [np.asarray(t, dtype=float) for t in unit_times],
        "UID": uid,
        "shankID": unique_pairs[:, 0].astype(int),
        "cluID": unique_pairs[:, 1].astype(int),
        "numcells": int(unique_pairs.shape[0]),
        "spindices": spindices.astype(float),
        "samplingRate": float(rate),
        "source": "clu_res",
    }


def _subset_spike_units(spikes: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_units = int(np.asarray(spikes.get("UID", [])).reshape(-1).size)
    keep = np.asarray(mask, dtype=bool).reshape(-1)
    if keep.size != n_units:
        raise ValueError("Unit-selection mask size mismatch.")

    for key, value in spikes.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == n_units:
            out[key] = value[keep]
        elif isinstance(value, list) and len(value) == n_units:
            out[key] = [value[i] for i, ok in enumerate(keep) if ok]
        else:
            out[key] = value

    uid = np.asarray(out.get("UID", []), dtype=int).reshape(-1)
    times = [np.asarray(t, dtype=float).reshape(-1) for t in out.get("times", [])]
    out["times"] = times
    out["numcells"] = int(uid.size)
    if uid.size and times:
        all_times = np.concatenate(times, axis=0)
        all_groups = np.concatenate(
            [np.full(np.asarray(t, dtype=float).reshape(-1).shape, float(uid[i]), dtype=float) for i, t in enumerate(times)],
            axis=0,
        )
        order = np.argsort(all_times, kind="mergesort")
        out["spindices"] = np.column_stack((all_times[order], all_groups[order]))
    else:
        out["spindices"] = np.empty((0, 2), dtype=float)
    return out


def _apply_units_filter(spikes: dict[str, Any], units: np.ndarray | list[list[int]] | None) -> dict[str, Any]:
    if units is None or np.asarray(units).size == 0:
        return spikes
    shank = np.asarray(spikes.get("shankID", []), dtype=int).reshape(-1)
    clu = np.asarray(spikes.get("cluID", []), dtype=int).reshape(-1)
    if shank.size == 0 or clu.size == 0:
        raise ValueError("Unit filtering requires both shankID and cluID metadata.")

    u = np.asarray(units, dtype=int)
    if u.ndim == 1:
        if u.size != 2:
            raise ValueError("units must be Nx2 [group, cluster] pairs.")
        u = u.reshape(1, 2)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("units must be Nx2 [group, cluster] pairs.")

    selected = np.zeros(shank.shape[0], dtype=bool)
    for group, cluster in u:
        if cluster == -1:
            mask = (shank == group) & (clu != 0) & (clu != 1)
        elif cluster == -2:
            mask = (shank == group) & (clu != 0)
        elif cluster == -3:
            mask = shank == group
        else:
            mask = (shank == group) & (clu == cluster)
        selected |= mask
    return _subset_spike_units(spikes, selected)


def _spikes_to_tsgroup(spikes: dict[str, Any]) -> nap.TsGroup:
    uid = np.asarray(spikes.get("UID", []), dtype=int).reshape(-1)
    times = [np.asarray(t, dtype=float).reshape(-1) for t in spikes.get("times", [])]
    if uid.size == 0 or len(times) == 0:
        return nap.TsGroup({})

    all_times = np.concatenate(times, axis=0) if times else np.array([], dtype=float)
    if all_times.size == 0:
        return nap.TsGroup({})
    t_start = float(np.min(all_times))
    t_end = float(np.max(all_times))
    if t_end <= t_start:
        t_end = t_start + 1e-6
    support = nap.IntervalSet(start=np.array([t_start]), end=np.array([t_end]), time_units="s")

    data = {int(uid[i]): nap.Ts(t=np.asarray(times[i], dtype=float), time_support=support) for i in range(uid.size)}
    group = nap.TsGroup(data, time_support=support)

    info: dict[str, Any] = {}
    for key, value in spikes.items():
        if key in {"times", "ts", "spindices", "numcells", "samplingRate", "source"}:
            continue
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == uid.size:
            info[key] = value
        elif isinstance(value, list) and len(value) == uid.size and all(np.isscalar(v) or isinstance(v, str) for v in value):
            info[key] = np.asarray(value, dtype=object)
    if info:
        try:
            group.set_info(**info)
        except Exception:
            pass
    return group


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


def get_spikes(
    units: np.ndarray | list[list[int]] | None = None,
    *,
    base_path: str | Path | None = None,
    basename: str | None = None,
    rate: float | None = None,
    source: str = "auto",
    as_tsgroup: bool = True,
) -> nap.TsGroup | dict[str, Any]:
    """
    Load spikes as a `TsGroup` or dictionary.

    Source modes:
    - `auto`: use `.spikes.cellinfo.mat` when present, else `.res/.clu`
    - `cellinfo`: require `.spikes.cellinfo.mat`
    - `clu`: require `.res/.clu`
    """
    src = str(source).lower()
    if src not in {"auto", "cellinfo", "clu"}:
        raise ValueError("source must be one of {'auto','cellinfo','clu'}.")

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
        try:
            params = load_parameters(base)
            rate_value = float(params["rates"]["wideband"])
        except Exception:
            rate_value = np.nan
    else:
        rate_value = float(rate)

    spikes: dict[str, Any] | None = None
    if src in {"auto", "cellinfo"}:
        cellinfo_path = _discover_cellinfo_file(base, basename)
        if cellinfo_path is not None and cellinfo_path.exists():
            spikes = _load_spikes_from_cellinfo(cellinfo_path, rate_override=rate_value if np.isfinite(rate_value) else None)
        elif src == "cellinfo":
            target = f"{str(basename).strip()}.spikes.cellinfo.mat" if basename is not None else "*.spikes.cellinfo.mat"
            raise FileNotFoundError(f"No cellinfo spikes file found in {base} (expected {target}).")

    if spikes is None:
        full = get_spike_times(
            units=None,
            base_path=base,
            basename=basename,
            rate=rate_value if np.isfinite(rate_value) else None,
            output="full",
        )
        resolved_rate = float(rate_value) if np.isfinite(rate_value) else 0.0
        if not np.isfinite(rate_value):
            params = load_parameters(base)
            resolved_rate = float(params["rates"]["wideband"])
        spikes = _spikes_struct_from_full(full, resolved_rate)

    spikes = _apply_units_filter(spikes, units)
    if "samplingRate" not in spikes:
        spikes["samplingRate"] = float(rate_value) if np.isfinite(rate_value) else np.nan

    if as_tsgroup:
        return _spikes_to_tsgroup(spikes)
    return spikes


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


def GetSpikes(
    units: np.ndarray | list[list[int]] | str | None = None,
    *args: Any,
    **kwargs: Any,
) -> nap.TsGroup | dict[str, Any]:
    """MATLAB-style alias for :func:`get_spikes`."""
    if isinstance(units, str):
        args = (units, *args)
        units = None
    options = _collect_options(args, kwargs)
    mapped = {
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "basename": options.pop("basename", None),
        "rate": options.pop("rate", None),
        "source": options.pop("source", "auto"),
        "as_tsgroup": bool(options.pop("as_tsgroup", options.pop("astsgroup", False))),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_spikes(units=units, **mapped)


__all__ = [
    "load_spike_times",
    "get_spike_times",
    "get_spikes",
    "LoadSpikeTimes",
    "GetSpikeTimes",
    "GetSpikes",
]
