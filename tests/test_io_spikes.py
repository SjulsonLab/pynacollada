from __future__ import annotations

import textwrap

import numpy as np
import pytest
from scipy.io import savemat

from pynacollada import (
    GetSpikes,
    GetSpikeTimes,
    LoadSpikeTimes,
    clear_current_session,
    get_spikes,
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

    np.savetxt(session_dir / f"{basename}.res.1", np.array([100, 200, 500, 600], dtype=int), fmt="%d")
    np.savetxt(session_dir / f"{basename}.clu.1", np.array([3, 2, 1, 2, 1], dtype=int), fmt="%d")

    np.savetxt(session_dir / f"{basename}.res.2", np.array([150, 250, 350, 450], dtype=int), fmt="%d")
    np.savetxt(session_dir / f"{basename}.clu.2", np.array([4, 3, 0, 3, 0], dtype=int), fmt="%d")


def _write_cellinfo(session_dir, basename: str = "sessionSpk") -> None:
    times = np.empty((3,), dtype=object)
    times[0] = np.array([0.010, 0.018], dtype=float)
    times[1] = np.array([0.025, 0.040, 0.055], dtype=float)
    times[2] = np.array([0.030], dtype=float)
    raw_waveforms = np.empty((3,), dtype=object)
    raw_waveforms[0] = np.array([-20.0, -50.0, -10.0], dtype=float)
    raw_waveforms[1] = np.array([-10.0, -30.0, -8.0], dtype=float)
    raw_waveforms[2] = np.array([-5.0, -15.0, -4.0], dtype=float)

    spikes = {
        "UID": np.array([10, 20, 30], dtype=int),
        "times": times,
        "shankID": np.array([1, 1, 2], dtype=int),
        "cluID": np.array([0, 2, 1], dtype=int),
        "region": np.array(["CA1", "CA1", "CA3"], dtype=object),
        "maxWaveformCh": np.array([4, 5, 6], dtype=int),
        "rawWaveform": raw_waveforms,
    }
    savemat(session_dir / f"{basename}.spikes.cellinfo.mat", {"spikes": spikes})


def test_load_spike_times_single_group(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)

    out = load_spike_times(session / "sessionSpk.res.1", rate=20000.0)
    np.testing.assert_allclose(out[:, 0], np.array([100, 200, 500, 600], dtype=float) / 20000.0)
    np.testing.assert_array_equal(out[:, 1].astype(int), np.array([1, 1, 1, 1], dtype=int))
    np.testing.assert_array_equal(out[:, 2].astype(int), np.array([2, 1, 2, 1], dtype=int))


def test_get_spike_times_output_modes(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)

    full = get_spike_times(base_path=session, output="full")
    assert full.shape == (8, 3)
    assert np.all(np.diff(full[:, 0]) >= 0)

    t = get_spike_times(base_path=session, output="time")
    np.testing.assert_allclose(t, full[:, 0])

    numbered = get_spike_times(base_path=session, output="numbered")
    assert numbered.shape == (8, 2)
    assert np.unique(numbered[:, 1]).size == 4

    total = get_spike_times(base_path=session, output="total")
    assert total.shape == (8, 4)


def test_get_spike_times_unit_filter_conventions(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)

    one_unit = get_spike_times(units=[[1, 2]], base_path=session, output="full")
    assert one_unit.shape[0] == 2

    all_single_except_mua_artifact = get_spike_times(units=[[2, -1]], base_path=session, output="full")
    assert all_single_except_mua_artifact.shape[0] == 2
    assert int(all_single_except_mua_artifact[0, 2]) == 3

    all_except_artifact = get_spike_times(units=[[2, -2]], base_path=session, output="full")
    assert all_except_artifact.shape[0] == 2

    all_clusters = get_spike_times(units=[[2, -3]], base_path=session, output="full")
    assert all_clusters.shape[0] == 4


def test_matlab_aliases_and_session_default(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)
    clear_current_session()
    set_current_session(session)

    full = GetSpikeTimes("output", "full")
    assert full.shape == (8, 3)

    group1 = LoadSpikeTimes(session / "sessionSpk.clu.1", 20000.0)
    assert group1.shape == (4, 3)

    struct = GetSpikes(basePath=session)
    assert isinstance(struct, dict)
    assert struct["numcells"] == 4
    assert len(struct["times"]) == 4
    assert struct["spindices"].shape[1] == 2

    tsg = get_spikes(base_path=session, as_tsgroup=True)
    assert len(tsg) == 4

    clear_current_session()


def test_get_spikes_cellinfo_source_and_auto_preference(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)
    _write_cellinfo(session)

    auto_struct = get_spikes(base_path=session, source="auto", as_tsgroup=False)
    assert auto_struct["source"] == "cellinfo"
    np.testing.assert_array_equal(auto_struct["UID"], np.array([10, 20, 30], dtype=int))
    assert auto_struct["numcells"] == 3
    assert "rawWaveform" in auto_struct
    assert len(auto_struct["times"]) == 3

    cellinfo_tsg = get_spikes(base_path=session, source="cellinfo", as_tsgroup=True)
    assert len(cellinfo_tsg) == 3


def test_get_spikes_cellinfo_unit_filtering(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)
    _write_cellinfo(session)

    # group 1 has clusters [0, 2]; -1 excludes 0 and 1, so only cluster 2.
    single_units = get_spikes(base_path=session, source="cellinfo", units=[[1, -1]], as_tsgroup=False)
    assert single_units["numcells"] == 1
    np.testing.assert_array_equal(single_units["cluID"], np.array([2], dtype=int))

    all_non_artifact = get_spikes(base_path=session, source="cellinfo", units=[[1, -2]], as_tsgroup=False)
    assert all_non_artifact["numcells"] == 1
    np.testing.assert_array_equal(all_non_artifact["cluID"], np.array([2], dtype=int))

    all_clusters = get_spikes(base_path=session, source="cellinfo", units=[[1, -3]], as_tsgroup=False)
    assert all_clusters["numcells"] == 2

    alias_struct = GetSpikes(basePath=session, source="cellinfo")
    assert isinstance(alias_struct, dict)
    assert alias_struct["source"] == "cellinfo"


def test_get_spikes_uid_and_region_filters(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)
    _write_cellinfo(session)

    by_uid = get_spikes(base_path=session, source="cellinfo", uid=[20], as_tsgroup=False)
    assert by_uid["numcells"] == 1
    np.testing.assert_array_equal(by_uid["UID"], np.array([20], dtype=int))

    by_region = get_spikes(base_path=session, source="cellinfo", region="CA3", as_tsgroup=False)
    assert by_region["numcells"] == 1
    np.testing.assert_array_equal(by_region["shankID"], np.array([2], dtype=int))

    with pytest.raises(ValueError):
        get_spikes(base_path=session, source="clu", region="CA1", as_tsgroup=False)


def test_get_spikes_waveform_extraction_from_dat(tmp_path) -> None:
    session = tmp_path / "sessionSpk"
    session.mkdir()
    _write_session(session)

    fs = 20000
    n_samples = 1200
    dat = np.zeros((n_samples, 4), dtype=np.int16)
    waveform = np.array([-5, -12, -25, -12, -5], dtype=np.int16)

    # Unit mapping from .res/.clu generated in _write_session:
    # (shank=1,clu=1) -> samples [200, 600]
    # (shank=1,clu=2) -> samples [100, 500]
    # (shank=2,clu=0) -> samples [250, 450]
    # (shank=2,clu=3) -> samples [150, 350]
    for s in [200, 600]:
        dat[s - 2 : s + 3, 0] += waveform
    for s in [100, 500]:
        dat[s - 2 : s + 3, 1] += waveform
    for s in [250, 450]:
        dat[s - 2 : s + 3, 2] += waveform
    for s in [150, 350]:
        dat[s - 2 : s + 3, 3] += waveform
    dat.tofile(session / "sessionSpk.dat")

    out = get_spikes(
        base_path=session,
        source="clu",
        get_waveforms=True,
        waveform_window_s=(2 / fs, 3 / fs),
        waveform_highpass_hz=0.0,
        as_tsgroup=False,
    )

    assert out["numcells"] == 4
    assert "rawWaveform" in out
    assert "filtWaveform" in out
    assert "maxWaveformCh" in out
    assert len(out["rawWaveform"]) == 4
    np.testing.assert_array_equal(out["maxWaveformCh"], np.array([0, 1, 2, 3], dtype=int))
    for wf in out["rawWaveform"]:
        assert np.asarray(wf).shape == (5,)
