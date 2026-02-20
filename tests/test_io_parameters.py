from __future__ import annotations

import textwrap

import numpy as np
import pytest

from pynacollada import (
    LoadPar,
    LoadParameters,
    load_par,
    load_parameters,
)


def _write_xml(path, n_channels: int, wideband: float, lfp: float) -> None:
    xml = textwrap.dedent(
        f"""\
        <parameters>
          <acquisitionSystem>
            <nChannels>{n_channels}</nChannels>
            <nBits>16</nBits>
            <samplingRate>{wideband}</samplingRate>
          </acquisitionSystem>
          <fieldPotentials>
            <lfpSamplingRate>{lfp}</lfpSamplingRate>
          </fieldPotentials>
          <video>
            <samplingRate>30</samplingRate>
          </video>
          <anatomicalDescription>
            <channelGroups>
              <group><channel>0</channel><channel>1</channel></group>
              <group><channel>2</channel><channel>3</channel></group>
            </channelGroups>
          </anatomicalDescription>
        </parameters>
        """
    )
    path.write_text(xml, encoding="utf-8")


def test_load_parameters_from_directory(tmp_path) -> None:
    session = tmp_path / "sessionX"
    session.mkdir()
    xml_path = session / "sessionX.xml"
    _write_xml(xml_path, n_channels=4, wideband=20000.0, lfp=1250.0)

    out = load_parameters(session)
    assert out["session"]["name"] == "sessionX"
    assert out["nChannels"] == 4
    np.testing.assert_array_equal(out["channels"], np.array([0, 1, 2, 3], dtype=int))
    assert out["nBits"] == 16
    assert out["rates"]["wideband"] == 20000.0
    assert out["rates"]["lfp"] == 1250.0
    assert out["rates"]["video"] == 30.0
    assert out["nElecGps"] == 2
    np.testing.assert_array_equal(out["ElecGp"][0], np.array([0, 1], dtype=int))
    np.testing.assert_array_equal(out["ElecGp"][1], np.array([2, 3], dtype=int))


def test_load_parameters_prefers_basename_xml_when_multiple(tmp_path) -> None:
    session = tmp_path / "sessionY"
    session.mkdir()
    _write_xml(session / "other.xml", n_channels=2, wideband=10000.0, lfp=500.0)
    _write_xml(session / "sessionY.xml", n_channels=8, wideband=30000.0, lfp=1500.0)

    out = load_parameters(session)
    assert out["nChannels"] == 8
    assert out["rates"]["wideband"] == 30000.0


def test_load_par_and_matlab_aliases(tmp_path) -> None:
    session = tmp_path / "sessionZ"
    session.mkdir()
    xml_path = session / "sessionZ.xml"
    _write_xml(xml_path, n_channels=4, wideband=20000.0, lfp=1250.0)

    out_a = load_par(session)
    out_b = LoadParameters(xml_path)
    out_c = LoadPar(session)

    assert out_a["nChannels"] == 4
    assert out_b["nChannels"] == 4
    assert out_c["nChannels"] == 4


def test_load_par_rejects_extra_args(tmp_path) -> None:
    session = tmp_path / "sessionW"
    session.mkdir()
    _write_xml(session / "sessionW.xml", n_channels=4, wideband=20000.0, lfp=1250.0)
    with pytest.raises(TypeError):
        LoadPar(session, "unexpected")
