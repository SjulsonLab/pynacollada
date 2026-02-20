from __future__ import annotations

import textwrap

import numpy as np
from scipy.io import savemat

from pynacollada import (
    GetChannels,
    GetUnits,
    get_channels,
    get_units,
)


def _write_session(session_dir, basename: str = "sessionUnits") -> None:
    xml = textwrap.dedent(
        f"""\
        <parameters>
          <acquisitionSystem>
            <nChannels>4</nChannels>
            <nBits>16</nBits>
            <samplingRate>20000</samplingRate>
          </acquisitionSystem>
          <fieldPotentials>
            <lfpSamplingRate>1250</lfpSamplingRate>
          </fieldPotentials>
          <anatomicalDescription>
            <channelGroups>
              <group><channel>0</channel><channel>1</channel></group>
              <group><channel>2</channel><channel>3</channel></group>
            </channelGroups>
          </anatomicalDescription>
        </parameters>
        """
    )
    (session_dir / f"{basename}.xml").write_text(xml, encoding="utf-8")

    np.savetxt(session_dir / f"{basename}.res.1", np.array([100, 200, 500, 600], dtype=int), fmt="%d")
    np.savetxt(session_dir / f"{basename}.clu.1", np.array([3, 2, 1, 2, 1], dtype=int), fmt="%d")

    np.savetxt(session_dir / f"{basename}.res.2", np.array([150, 250, 350, 450], dtype=int), fmt="%d")
    np.savetxt(session_dir / f"{basename}.clu.2", np.array([4, 3, 0, 3, 0], dtype=int), fmt="%d")


def _write_cellinfo(session_dir, basename: str = "sessionUnits") -> None:
    times = np.empty((3,), dtype=object)
    times[0] = np.array([0.01, 0.02], dtype=float)
    times[1] = np.array([0.03], dtype=float)
    times[2] = np.array([0.04], dtype=float)
    spikes = {
        "UID": np.array([10, 20, 30], dtype=int),
        "times": times,
        "shankID": np.array([1, 1, 2], dtype=int),
        "cluID": np.array([0, 2, 3], dtype=int),
    }
    savemat(session_dir / f"{basename}.spikes.cellinfo.mat", {"spikes": spikes})


def test_get_channels_from_group_mapping(tmp_path) -> None:
    session = tmp_path / "sessionUnits"
    session.mkdir()
    _write_session(session)

    all_ch = get_channels(base_path=session)
    np.testing.assert_array_equal(all_ch, np.array([0, 1, 2, 3], dtype=int))

    g1 = get_channels(groups=[1], base_path=session)
    np.testing.assert_array_equal(g1, np.array([0, 1], dtype=int))

    alias = GetChannels([2], basePath=session)
    np.testing.assert_array_equal(alias, np.array([2, 3], dtype=int))


def test_get_units_with_noise_filtering(tmp_path) -> None:
    session = tmp_path / "sessionUnits"
    session.mkdir()
    _write_session(session)

    units_default = get_units(base_path=session, source="clu")
    np.testing.assert_array_equal(units_default, np.array([[1, 2], [2, 3]], dtype=int))

    units_all = get_units(base_path=session, source="clu", include_noise=True)
    np.testing.assert_array_equal(units_all, np.array([[1, 1], [1, 2], [2, 0], [2, 3]], dtype=int))

    alias = GetUnits([1], basePath=session, source="clu")
    np.testing.assert_array_equal(alias, np.array([[1, 2]], dtype=int))


def test_get_units_auto_prefers_cellinfo(tmp_path) -> None:
    session = tmp_path / "sessionUnits"
    session.mkdir()
    _write_session(session)
    _write_cellinfo(session)

    units_auto = get_units(base_path=session, source="auto", include_noise=True)
    np.testing.assert_array_equal(units_auto, np.array([[1, 0], [1, 2], [2, 3]], dtype=int))
