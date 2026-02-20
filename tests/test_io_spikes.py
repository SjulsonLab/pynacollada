from __future__ import annotations

import textwrap

import numpy as np

from pynacollada import (
    GetSpikeTimes,
    LoadSpikeTimes,
    clear_current_session,
    get_spike_times,
    load_spike_times,
    set_current_session,
)


def _write_session(session_dir, basename: str = "sessionSpk") -> None:
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
        </parameters>
        """
    )
    (session_dir / f"{basename}.xml").write_text(xml, encoding="utf-8")

    np.savetxt(session_dir / f"{basename}.res.1", np.array([100, 200, 500], dtype=int), fmt="%d")
    np.savetxt(session_dir / f"{basename}.clu.1", np.array([3, 2, 1, 2], dtype=int), fmt="%d")

    np.savetxt(session_dir / f"{basename}.res.2", np.array([150, 250], dtype=int), fmt="%d")
    np.savetxt(session_dir / f"{basename}.clu.2", np.array([4, 3, 0], dtype=int), fmt="%d")


def test_load_spike_times_single_group(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)

    out = load_spike_times(session / "sessionSpk.res.1", rate=20000.0)
    np.testing.assert_allclose(out[:, 0], np.array([100, 200, 500], dtype=float) / 20000.0)
    np.testing.assert_array_equal(out[:, 1].astype(int), np.array([1, 1, 1], dtype=int))
    np.testing.assert_array_equal(out[:, 2].astype(int), np.array([2, 1, 2], dtype=int))


def test_get_spike_times_output_modes(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)

    full = get_spike_times(base_path=session, output="full")
    assert full.shape == (5, 3)
    assert np.all(np.diff(full[:, 0]) >= 0)

    t = get_spike_times(base_path=session, output="time")
    np.testing.assert_allclose(t, full[:, 0])

    numbered = get_spike_times(base_path=session, output="numbered")
    assert numbered.shape == (5, 2)
    assert np.unique(numbered[:, 1]).size == 4

    total = get_spike_times(base_path=session, output="total")
    assert total.shape == (5, 4)


def test_get_spike_times_unit_filter_conventions(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)

    one_unit = get_spike_times(units=[[1, 2]], base_path=session, output="full")
    assert one_unit.shape[0] == 2

    all_single_except_mua_artifact = get_spike_times(units=[[2, -1]], base_path=session, output="full")
    assert all_single_except_mua_artifact.shape[0] == 1
    assert int(all_single_except_mua_artifact[0, 2]) == 3

    all_except_artifact = get_spike_times(units=[[2, -2]], base_path=session, output="full")
    assert all_except_artifact.shape[0] == 1

    all_clusters = get_spike_times(units=[[2, -3]], base_path=session, output="full")
    assert all_clusters.shape[0] == 2


def test_matlab_aliases_and_session_default(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)
    clear_current_session()
    set_current_session(session)

    full = GetSpikeTimes("output", "full")
    assert full.shape == (5, 3)

    group1 = LoadSpikeTimes(session / "sessionSpk.clu.1", 20000.0)
    assert group1.shape == (3, 3)

    clear_current_session()
