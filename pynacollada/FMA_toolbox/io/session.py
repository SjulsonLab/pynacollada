"""Session-context helpers inspired by FMAT session utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .parameters import load_parameters


@dataclass(frozen=True)
class SessionContext:
    """Small immutable session descriptor."""

    base_path: Path
    basename: str
    parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_path": str(self.base_path),
            "basename": self.basename,
            "parameters": self.parameters,
        }


_CURRENT_SESSION: SessionContext | None = None


def _normalize_base_path(base_path: str | Path | None) -> Path:
    base = Path.cwd() if base_path is None else Path(base_path)
    if base.is_file():
        base = base.parent
    if not base.exists():
        raise FileNotFoundError(f"Session path does not exist: {base}")
    if not base.is_dir():
        raise ValueError(f"Session path must be a directory: {base}")
    return base.resolve()


def open_session(base_path: str | Path | None = None, *, load_parameters_file: bool = True) -> SessionContext:
    """Open a session context without mutating global state."""
    base = _normalize_base_path(base_path)
    params: dict[str, Any] | None = None
    if load_parameters_file:
        try:
            params = load_parameters(base)
        except FileNotFoundError:
            params = None
    return SessionContext(base_path=base, basename=base.name, parameters=params)


def set_current_session(
    base_path: str | Path | None = None,
    *,
    load_parameters_file: bool = True,
    force_reload: bool = False,
) -> SessionContext:
    """Set the process-local current session context."""
    global _CURRENT_SESSION

    if isinstance(base_path, str) and base_path.lower() == "same":
        if _CURRENT_SESSION is None:
            raise ValueError("No current session is set; cannot reuse 'same'.")
        target = _CURRENT_SESSION.base_path
    else:
        target = base_path

    candidate = open_session(target, load_parameters_file=load_parameters_file)
    if not force_reload and _CURRENT_SESSION is not None and _CURRENT_SESSION.base_path == candidate.base_path:
        return _CURRENT_SESSION
    _CURRENT_SESSION = candidate
    return candidate


def get_current_session(*, default_to_cwd: bool = False, load_parameters_file: bool = True) -> SessionContext | None:
    """Return the current session context (or optionally initialize from cwd)."""
    if _CURRENT_SESSION is None and default_to_cwd:
        return set_current_session(None, load_parameters_file=load_parameters_file)
    return _CURRENT_SESSION


def clear_current_session() -> None:
    """Clear the process-local current session context."""
    global _CURRENT_SESSION
    _CURRENT_SESSION = None


def SetCurrentSession(basePath: str | Path | None = None, *args: Any, **kwargs: Any) -> SessionContext:
    """MATLAB-style alias for :func:`set_current_session`."""
    if args:
        raise TypeError("Unexpected positional arguments.")
    force_reload = bool(kwargs.pop("force_reload", kwargs.pop("forceReload", False)))
    load_params = bool(kwargs.pop("load_parameters_file", kwargs.pop("loadParameters", True)))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return set_current_session(basePath, load_parameters_file=load_params, force_reload=force_reload)


def GetCurrentSession(*args: Any, **kwargs: Any) -> SessionContext | None:
    """MATLAB-style alias for :func:`get_current_session`."""
    if args:
        raise TypeError("Unexpected positional arguments.")
    default_to_cwd = bool(kwargs.pop("default_to_cwd", kwargs.pop("defaultToCwd", False)))
    load_params = bool(kwargs.pop("load_parameters_file", kwargs.pop("loadParameters", True)))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(f"Unexpected options: {unexpected}")
    return get_current_session(default_to_cwd=default_to_cwd, load_parameters_file=load_params)


def ClearCurrentSession() -> None:
    """MATLAB-style alias for :func:`clear_current_session`."""
    clear_current_session()


__all__ = [
    "SessionContext",
    "open_session",
    "set_current_session",
    "get_current_session",
    "clear_current_session",
    "SetCurrentSession",
    "GetCurrentSession",
    "ClearCurrentSession",
]
