from __future__ import annotations

import numpy as np

from pynacollada import (
    ShortTimeCCG,
    Sync,
    SyncHist,
    SyncMap,
    short_time_ccg,
    sync,
    sync_hist,
    sync_map,
)


def test_sync_point_process_and_continuous() -> None:
    sync_t = np.array([1.0, 2.0, 3.0], dtype=float)
    spikes = np.array([0.7, 0.95, 1.05, 1.4, 2.1, 2.3, 3.2], dtype=float)
    s, idx = sync(spikes, sync_t, durations=(-0.2, 0.3))

    assert s.shape[1] == 1
    assert idx.shape[0] == s.shape[0]
    assert np.min(s[:, 0]) >= -0.2 - 1e-9
    assert np.max(s[:, 0]) <= 0.3 + 1e-9

    s2, idx2 = Sync(spikes, sync_t, "durations", [-0.2, 0.3])
    np.testing.assert_allclose(s, s2)
    np.testing.assert_array_equal(idx, idx2)

    t = np.linspace(0.0, 4.0, 2001)
    v = np.sin(2.0 * np.pi * 5.0 * t)
    cont = np.column_stack((t, v))
    sc, ic = sync(cont, sync_t, durations=(-0.1, 0.1))
    assert sc.shape[1] == 2
    assert ic.shape[0] == sc.shape[0]


def test_sync_map_and_hist() -> None:
    sync_t = np.array([1.0, 2.0, 3.0], dtype=float)
    spikes = np.array([0.95, 1.05, 1.10, 1.95, 2.01, 2.06, 2.95, 3.02], dtype=float)
    s, idx = sync(spikes, sync_t, durations=(-0.2, 0.2))

    m, tb = sync_map(s, idx, durations=(-0.2, 0.2), n_bins=40)
    assert m.shape == (3, 40)
    assert tb.shape == (40,)

    m2, tb2 = SyncMap(s, idx, "durations", [-0.2, 0.2], "nBins", 40)
    np.testing.assert_allclose(m, m2)
    np.testing.assert_allclose(tb, tb2)

    hsum, tbin, _ = sync_hist(s, idx, mode="sum", durations=(-0.2, 0.2), n_bins=40)
    assert hsum.shape == (40,)
    assert tbin.shape == (40,)

    hmean, tbin2, _ = sync_hist(s, idx, mode="mean", durations=(-0.2, 0.2), n_bins=40)
    assert hmean.shape == (40,)
    assert tbin2.shape == (40,)

    hdist, tbin3, vb = sync_hist(s, idx, mode="dist", durations=(-0.2, 0.2), n_bins=40)
    assert hdist.shape[1] == 40
    assert tbin3.shape == (40,)
    assert vb is not None

    hsum2, tbin4, _ = SyncHist(s, idx, "mode", "sum", "durations", [-0.2, 0.2], "nBins", 40)
    np.testing.assert_allclose(hsum, hsum2)
    np.testing.assert_allclose(tbin, tbin4)

    # Continuous-valued mean mode keeps return ordering as (mean, time, error).
    t = np.linspace(0.0, 4.0, 2001)
    v = np.sin(2.0 * np.pi * 3.0 * t)
    cont = np.column_stack((t, v))
    sc, ic = sync(cont, sync_t, durations=(-0.2, 0.2))
    mcont, tbcont, econt = sync_hist(sc, ic, mode="mean", durations=(-0.2, 0.2), n_bins=40, error="std")
    assert mcont.shape == (40,)
    assert tbcont.shape == (40,)
    assert econt is not None
    assert econt.shape == (40,)


def test_short_time_ccg() -> None:
    t1 = np.arange(0.0, 60.0, 0.5)
    t2 = t1 + 0.03

    ccg, x, y = short_time_ccg(t1, t2, bin_size=0.01, duration=0.2, window=20.0, overlap=10.0)
    assert ccg.shape[0] == y.shape[0]
    assert ccg.shape[1] == x.shape[0]
    assert np.nanmax(ccg) > 0

    ccg2, x2, y2 = ShortTimeCCG(t1, t2, "binSize", 0.01, "duration", 0.2, "window", 20.0, "overlap", 10.0)
    np.testing.assert_allclose(ccg, ccg2)
    np.testing.assert_allclose(x, x2)
    np.testing.assert_allclose(y, y2)
