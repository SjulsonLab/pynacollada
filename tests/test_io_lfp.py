from __future__ import annotations

import textwrap

import numpy as np
import pynapple as nap

from pynacollada import (
    GetLFP,
    get_lfp,
)


def _write_session_files(session_dir, basename: str = "sessionLFP") -> tuple[np.ndarray, np.ndarray]:
    xml = textwrap.dedent(
        f"""\
        <parameters>
          <acquisitionSystem>
            <nChannels>4</nChannels>
            <nBits>16</nBits>
            <samplingRate>20</samplingRate>
          </acquisitionSystem>
          <fieldPotentials>
            <lfpSamplingRate>10</lfpSamplingRate>
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

    lfp = (10 * np.arange(12, dtype=np.int16)[:, None] + np.arange(1, 5, dtype=np.int16)[None, :]).astype(np.int16)
    lfp.tofile(session_dir / f"{basename}.lfp")

    dat = (100 * np.arange(20, dtype=np.int16)[:, None] + np.arange(1, 5, dtype=np.int16)[None, :]).astype(np.int16)
    dat.tofile(session_dir / f"{basename}.dat")
    return lfp, dat


def test_get_lfp_multichannel_and_all_channels(tmp_path) -> None:
    session = tmp_path / "sessionLFP"
    session.mkdir()
    lfp, _ = _write_session_files(session)

    out = get_lfp([0, 2], base_path=session)
    assert isinstance(out, nap.TsdFrame)
    np.testing.assert_array_equal(out.values, lfp[:, [0, 2]])
    times = out.as_units("s").index.values
    np.testing.assert_allclose(np.diff(times[:3]), 0.1, atol=1e-12)

    all_out = get_lfp("all", base_path=session)
    assert isinstance(all_out, nap.TsdFrame)
    assert all_out.values.shape[1] == 4


def test_get_lfp_from_dat_downsample_and_single_channel(tmp_path) -> None:
    session = tmp_path / "sessionLFP"
    session.mkdir()
    _, dat = _write_session_files(session)

    out = get_lfp([1], base_path=session, from_dat=True, downsample=2)
    assert isinstance(out, nap.Tsd)
    np.testing.assert_array_equal(out.values, dat[:, 1][::2])
    times = out.as_units("s").index.values
    np.testing.assert_allclose(np.diff(times[:3]), 0.1, atol=1e-12)


def test_get_lfp_interval_restrict_and_alias(tmp_path) -> None:
    session = tmp_path / "sessionLFP"
    session.mkdir()
    _write_session_files(session)

    intervals = nap.IntervalSet(start=np.array([0.25]), end=np.array([0.55]), time_units="s")
    out = get_lfp([0], base_path=session, intervals=intervals)
    times = out.as_units("s").index.values
    assert times.size > 0
    assert times.min() >= 0.25 - 1e-12
    assert times.max() <= 0.55 + 1e-12

    alias = GetLFP([0], "basePath", session, "restrict", np.array([[0.25, 0.55]]))
    np.testing.assert_array_equal(alias.values, out.values)
    np.testing.assert_allclose(alias.as_units("s").index.values, out.as_units("s").index.values, atol=1e-12)
