"""FMAT-style parameter XML readers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


def _resolve_xml_file(filename: str | Path | None) -> Path:
    if filename is None:
        base = Path.cwd()
    else:
        base = Path(filename)

    if base.is_file():
        if base.suffix.lower() != ".xml":
            raise ValueError(f"Expected an .xml file, got: {base}")
        return base

    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")
    if not base.is_dir():
        raise ValueError(f"Expected a directory or .xml file, got: {base}")

    matches = sorted(base.glob("*.xml"))
    if not matches:
        raise FileNotFoundError(f"No .xml file found in {base}")
    if len(matches) == 1:
        return matches[0]

    target = f"{base.resolve().name}.xml"
    for candidate in matches:
        if candidate.name == target:
            return candidate
    return matches[0]


def _node_text(node: ET.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value if value else None


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(text: str | None) -> int | None:
    value = _to_float(text)
    if value is None:
        return None
    return int(round(value))


def _parse_channel_groups(root: ET.Element) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for group in root.findall("./anatomicalDescription/channelGroups/group"):
        channels: list[int] = []
        direct = group.findall("./channel")
        nested = group.findall("./channels/channel")
        for channel_node in [*direct, *nested]:
            value = _to_int(_node_text(channel_node))
            if value is not None:
                channels.append(value)
        if not channels:
            raw = _node_text(group)
            if raw is not None:
                channels = [int(x) for x in re.findall(r"-?\d+", raw)]
        if channels:
            groups.append(np.asarray(channels, dtype=int))
    return groups


def load_parameters(filename: str | Path | None = None) -> dict[str, Any]:
    """
    Load key session parameters from a Neuroscope-style XML file.

    This is a pragmatic subset of FMAT `LoadParameters`, focused on fields
    used by downstream IO/analysis helpers.
    """
    xml_file = _resolve_xml_file(filename)
    root = ET.parse(xml_file).getroot()

    n_channels = _to_int(_node_text(root.find("./acquisitionSystem/nChannels"))) or 0
    n_bits = _to_int(_node_text(root.find("./acquisitionSystem/nBits"))) or 0
    wideband_rate = _to_float(_node_text(root.find("./acquisitionSystem/samplingRate"))) or 0.0

    lfp_rate = _to_float(_node_text(root.find("./fieldPotentials/lfpSamplingRate")))
    if lfp_rate is None:
        for file_node in root.findall("./files/file"):
            extension = (_node_text(file_node.find("./extension")) or "").lower()
            if extension == "lfp":
                candidate = _to_float(_node_text(file_node.find("./samplingRate")))
                if candidate is not None:
                    lfp_rate = candidate
                    break
    if lfp_rate is None:
        lfp_rate = wideband_rate

    video_rate = _to_float(_node_text(root.find("./video/samplingRate"))) or 0.0
    elec_groups = _parse_channel_groups(root)

    if n_channels > 0:
        channels = np.arange(n_channels, dtype=int)
    else:
        channels = np.array([], dtype=int)

    params: dict[str, Any] = {
        "session": {"path": str(xml_file.parent), "name": xml_file.stem},
        "nChannels": int(n_channels),
        "channels": channels,
        "nBits": int(n_bits),
        "rates": {
            "lfp": float(lfp_rate),
            "wideband": float(wideband_rate),
            "video": float(video_rate),
        },
        "lfpSampleRate": float(lfp_rate),
        "FileName": xml_file.stem,
        "SampleTime": float(1e6 / wideband_rate) if wideband_rate > 0 else np.nan,
        "nElecGps": len(elec_groups),
        "ElecGp": elec_groups,
        "HiPassFreq": 500.0,
    }
    return params


def load_par(filename: str | Path | None = None) -> dict[str, Any]:
    """Backwards-compatible alias for :func:`load_parameters`."""
    return load_parameters(filename)


def LoadParameters(filename: str | Path | None = None) -> dict[str, Any]:
    """MATLAB-style alias for :func:`load_parameters`."""
    return load_parameters(filename)


def LoadPar(filename: str | Path | None = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """MATLAB-style alias for :func:`load_par`."""
    if args or kwargs:
        raise TypeError("LoadPar currently supports only the filename argument.")
    return load_par(filename)


__all__ = [
    "load_parameters",
    "load_par",
    "LoadParameters",
    "LoadPar",
]
