from __future__ import annotations

import textwrap

import numpy as np
import pynapple as nap

from pynacollada import (
    GetPositions,
    LoadPositions,
    get_positions,
    load_positions,
)


def _write_session_with_whl(session_dir, basename: str = "sessionPos") -> None:
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
          <video>
            <samplingRate>10</samplingRate>
          </video>
        </parameters>
        """
    )
    (session_dir / f"{basename}.xml").write_text(xml, encoding="utf-8")

    # x1 y1 x2 y2
    whl = np.array(
        [
            [10.0, 10.0, 20.0, 10.0],
            [-1.0, -1.0, -1.0, -1.0],
            [10.0, 10.0, 10.0, 20.0],
            [12.0, 12.0, 24.0, 12.0],
        ],
        dtype=float,
    )
    np.savetxt(session_dir / f"{basename}.whl", whl, fmt="%.6f")


def test_load_positions_from_whl_and_alias(tmp_path) -> None:
    session = tmp_path / "sessionPos"
    session.mkdir()
    _write_session_with_whl(session)

    out = load_positions(session / "sessionPos.whl", rate=10.0)
    assert out.shape == (4, 5)
    np.testing.assert_allclose(out[:, 0], np.array([0.0, 0.1, 0.2, 0.3]))

    alias = LoadPositions(session / "sessionPos.whl", 10.0)
    np.testing.assert_allclose(alias, out)


def test_get_positions_clean_and_coordinate_modes(tmp_path) -> None:
    session = tmp_path / "sessionPos"
    session.mkdir()
    _write_session_with_whl(session)

    clean_norm = get_positions(base_path=session, mode="clean", coordinates="normalized")
    assert clean_norm.shape[0] == 3  # one undetected row removed
    assert clean_norm.shape[1] == 5
    assert np.isfinite(clean_norm).all()

    clean_video = get_positions(base_path=session, mode="clean", coordinates="video")
    clean_real = get_positions(base_path=session, mode="clean", coordinates="real", pixel=0.5)
    np.testing.assert_allclose(clean_real[:, 1:], clean_video[:, 1:] * 0.5, atol=1e-12)

    all_keep = get_positions(base_path=session, mode="all", discard="none", coordinates="video")
    assert all_keep.shape[0] == 4
    assert np.isnan(all_keep[1, 1:]).all()


def test_get_positions_tsdframe_and_alias(tmp_path) -> None:
    session = tmp_path / "sessionPos"
    session.mkdir()
    _write_session_with_whl(session)

    tsd = get_positions(base_path=session, as_tsdframe=True, mode="clean", coordinates="video")
    assert isinstance(tsd, nap.TsdFrame)
    assert tsd.values.shape[1] == 4

    alias = GetPositions("basePath", session, "mode", "clean", "coordinates", "video", "asTsdFrame", True)
    assert isinstance(alias, nap.TsdFrame)
    np.testing.assert_allclose(alias.values, tsd.values)
