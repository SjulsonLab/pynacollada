from __future__ import annotations

import textwrap

from pynacollada import (
    ClearCurrentSession,
    GetCurrentSession,
    SetCurrentSession,
    clear_current_session,
    get_current_session,
    set_current_session,
)


def _write_xml(session_dir, basename: str) -> None:
    xml = textwrap.dedent(
        """\
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


def test_set_and_get_current_session(tmp_path) -> None:
    session = tmp_path / "sessionCtx"
    session.mkdir()
    _write_xml(session, "sessionCtx")
    clear_current_session()

    ctx = set_current_session(session)
    cur = get_current_session()

    assert cur is not None
    assert cur.base_path == session.resolve()
    assert cur.basename == "sessionCtx"
    assert ctx.parameters is not None
    assert ctx.parameters["nChannels"] == 4


def test_matlab_aliases_and_same_keyword(tmp_path) -> None:
    session = tmp_path / "sessionAlias"
    session.mkdir()
    _write_xml(session, "sessionAlias")
    ClearCurrentSession()

    first = SetCurrentSession(session)
    second = SetCurrentSession("same")
    assert first.base_path == second.base_path

    current = GetCurrentSession()
    assert current is not None
    assert current.base_path == session.resolve()


def test_get_current_session_default_to_cwd(tmp_path, monkeypatch) -> None:
    clear_current_session()
    monkeypatch.chdir(tmp_path)
    ctx = get_current_session(default_to_cwd=True, load_parameters_file=False)
    assert ctx is not None
    assert ctx.base_path == tmp_path.resolve()


def test_clear_current_session(tmp_path) -> None:
    session = tmp_path / "sessionClear"
    session.mkdir()
    _write_xml(session, "sessionClear")
    set_current_session(session)
    clear_current_session()
    assert get_current_session() is None
