"""Position loading/cleaning helpers inspired by FMAT LoadPositions/GetPositions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pynapple as nap

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


def load_positions(filename: str | Path, rate: float | None = None) -> np.ndarray:
    """
    Load position samples from `.whl`, `.pos`, or `.mqa`.

    For `.whl`/`.pos`, timestamps are generated from `rate`.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Position file not found: {path}")

    ext = path.suffix.lower()
    if ext in {".whl", ".pos"}:
        if rate is None or float(rate) <= 0:
            raise ValueError("A positive sampling rate is required for .whl/.pos files.")
        data = np.asarray(np.loadtxt(path, dtype=float), dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        t = np.arange(data.shape[0], dtype=float) / float(rate)
        return np.column_stack((t, data))
    if ext == ".mqa":
        data = np.asarray(np.loadtxt(path, dtype=float), dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    raise ValueError(f"Unsupported position file extension: {ext}")


def _resolve_position_file(base_path: Path, basename: str | None, filename: str | Path | None) -> Path:
    if filename is not None:
        path = Path(filename)
        if not path.is_absolute():
            path = base_path / path
        return path

    if basename is not None and str(basename).strip():
        base = str(basename).strip()
        for ext in (".whl", ".pos", ".mqa"):
            candidate = base_path / f"{base}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No position file found for basename '{base}' in {base_path}")

    default = base_path.resolve().name
    for ext in (".whl", ".pos", ".mqa"):
        candidate = base_path / f"{default}{ext}"
        if candidate.exists():
            return candidate

    matches: list[Path] = []
    for ext in ("*.whl", "*.pos", "*.mqa"):
        matches.extend(sorted(base_path.glob(ext)))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No position file found in {base_path}")
    raise ValueError("Multiple position files found; specify basename or filename explicitly.")


def get_positions(
    *,
    positions: np.ndarray | None = None,
    base_path: str | Path | None = None,
    basename: str | None = None,
    filename: str | Path | None = None,
    rate: float | None = None,
    mode: str = "clean",
    coordinates: str = "normalized",
    pixel: float | None = None,
    discard: str = "partial",
    distances: tuple[float, float] = (0.0, np.inf),
    as_tsdframe: bool = False,
) -> np.ndarray | nap.TsdFrame:
    """
    Load and optionally clean/transform position samples.

    Matrix format is `[t, x1, y1, x2, y2, ...]`.
    """
    m = str(mode).lower()
    if m not in {"clean", "all"}:
        raise ValueError("mode must be 'clean' or 'all'.")
    coord = str(coordinates).lower()
    if coord not in {"video", "normalized", "real"}:
        raise ValueError("coordinates must be one of {'video','normalized','real'}.")
    discard_mode = str(discard).lower()
    if discard_mode not in {"partial", "none"}:
        raise ValueError("discard must be 'partial' or 'none'.")
    if coord == "real" and (pixel is None or float(pixel) <= 0):
        raise ValueError("A positive pixel size is required for coordinates='real'.")
    d = np.asarray(distances, dtype=float).reshape(-1)
    if d.size != 2 or np.any(d < 0):
        raise ValueError("distances must be a pair [min, max] with non-negative values.")
    min_dist, max_dist = float(d[0]), float(d[1])

    if positions is None:
        if base_path is None:
            current = get_current_session()
            base = current.base_path if current is not None else Path.cwd()
        else:
            base = Path(base_path)
        if base.is_file():
            base = base.parent
        file_path = _resolve_position_file(base, basename, filename)
        sample_rate = rate
        if file_path.suffix.lower() in {".whl", ".pos"} and (sample_rate is None or float(sample_rate) <= 0):
            params = load_parameters(base)
            sample_rate = float(params["rates"]["video"])
        arr = load_positions(file_path, sample_rate)
    else:
        arr = np.asarray(positions, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("positions must be a 2D array with at least columns [t, x, y].")

    pos = np.asarray(arr, dtype=float).copy()
    if m == "clean" and pos.shape[1] >= 5:
        distance = np.sqrt((pos[:, 3] - pos[:, 1]) ** 2 + (pos[:, 4] - pos[:, 2]) ** 2)
        selected = (distance >= min_dist) & (distance <= max_dist)
        pos = pos[selected]

    if pos.size == 0:
        if as_tsdframe:
            return nap.TsdFrame(t=np.array([], dtype=float), d=np.empty((0, 0), dtype=float), time_units="s")
        return pos

    coords = pos[:, 1:]
    if discard_mode == "none":
        undetected = np.all(coords == -1, axis=1)
        coords[coords == -1] = np.nan
    else:
        undetected = np.any(coords == -1, axis=1)
    if m == "clean":
        keep = ~undetected
        pos = pos[keep]
        coords = pos[:, 1:]

    if pos.shape[0] == 0:
        if as_tsdframe:
            return nap.TsdFrame(t=np.array([], dtype=float), d=np.empty((0, 0), dtype=float), time_units="s")
        return pos

    if coord == "normalized":
        finite = np.isfinite(coords)
        if np.any(finite):
            maxima = np.nanmax(np.where(finite, coords, np.nan), axis=0)
            maxima[maxima == 0] = 1.0
            coords = coords / maxima
            pos[:, 1:] = coords
    elif coord == "real":
        pos[:, 1:] = coords * float(pixel)

    if as_tsdframe:
        t = pos[:, 0]
        dvals = pos[:, 1:]
        columns = [f"coord_{i}" for i in range(dvals.shape[1])]
        return nap.TsdFrame(t=t, d=dvals, time_units="s", columns=columns)
    return pos


def LoadPositions(filename: str | Path, rate: float | None = None) -> np.ndarray:
    """MATLAB-style alias for :func:`load_positions`."""
    return load_positions(filename, rate=rate)


def GetPositions(*args: Any, **kwargs: Any) -> np.ndarray | nap.TsdFrame:
    """MATLAB-style alias for :func:`get_positions`."""
    options = _collect_options(args, kwargs)
    mapped = {
        "positions": options.pop("positions", None),
        "base_path": options.pop("basepath", options.pop("base_path", None)),
        "basename": options.pop("basename", None),
        "filename": options.pop("filename", None),
        "rate": options.pop("rate", None),
        "mode": options.pop("mode", "clean"),
        "coordinates": options.pop("coordinates", "normalized"),
        "pixel": options.pop("pixel", None),
        "discard": options.pop("discard", "partial"),
        "distances": tuple(options.pop("distances", (0.0, np.inf))),
        "as_tsdframe": bool(options.pop("as_tsdframe", options.pop("astsdframe", False))),
    }
    if options:
        unexpected = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_positions(**mapped)


__all__ = [
    "load_positions",
    "get_positions",
    "LoadPositions",
    "GetPositions",
]
