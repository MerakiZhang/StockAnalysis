"""实盘跟踪段（selection/segments.py）：规则哈希、建段 / 开新段、段间持仓与账户衔接；跟踪报告的区间分布。"""
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stockanalysis.backtest.points_engine import account
from stockanalysis.selection import segments as S
from stockanalysis.selection.points import PointSignal

from test_positions import DATES, make_ps, spec


def fake_pool(raw, live_start="2021-01-04", points=None):
    return SimpleNamespace(name="p", raw=raw, points=points or SimpleNamespace(live_start=live_start),
                           config_hash="h", benchmark="X", account=1e6,
                           cost={"commission": 0.0, "stamp_tax": 0.0, "slippage": 0.0, "min_cost": 0})


def test_rules_hash_ignores_display_and_live_start():
    base = {"status": "observe", "enabled": True, "data": {"a": 1}, "points": {"live_start": "2026-01-01", "stop_loss": 0.08}}
    other = {"status": "production", "enabled": False, "data": {"a": 2}, "points": {"live_start": "2026-02-01", "stop_loss": 0.08}}
    assert S.rules_hash(base) == S.rules_hash(other)
    assert S.rules_hash(base) != S.rules_hash({**base, "points": {"stop_loss": 0.05}})


def test_ensure_and_new_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "pool_dir", lambda name: tmp_path / name)
    p1 = fake_pool({"points": {"stop_loss": 0.08}})
    segs = S.ensure_segments(p1)
    assert len(segs) == 1 and segs[0]["start"] == "2021-01-04"
    p2 = fake_pool({"points": {"stop_loss": 0.05}})
    with pytest.raises(RuntimeError):
        S.ensure_segments(p2)                       # 改了规则必须显式开新段
    with pytest.raises(ValueError):
        S.new_segment(p2, "2021-01-01")             # 新段须晚于当前段
    S.new_segment(p2, "2021-02-01")
    assert [s["start"] for s in S.ensure_segments(p2)] == ["2021-01-04", "2021-02-01"]
    with pytest.raises(ValueError):
        S.new_segment(p2, "2021-03-01")             # 规则相同无需开新段


def _window(ps: PointSignal, s, e) -> PointSignal:
    m = (ps.dates >= pd.Timestamp(s)) & (ps.dates <= pd.Timestamp(e))
    kw = {f.name: getattr(ps, f.name) for f in fields(PointSignal) if f.init}
    for k, v in kw.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (len(ps.dates),):
            kw[k] = v[m]
    kw["dates"] = ps.dates[m]
    return PointSignal(**kw)


def test_live_chain_carries_positions_and_account_state(monkeypatch):
    import stockanalysis.data.market as M
    import stockanalysis.selection.points as P

    buy = np.zeros((len(DATES), 3), bool)
    buy[0, 0] = True
    close = np.tile(np.linspace(10, 11.1, len(DATES))[:, None], (1, 3))
    full = make_ps(buy=buy, close=close)
    pools = [fake_pool({"v": 1}, points=spec()), fake_pool({"v": 2}, points=spec(buy={"y": {}}))]
    segs = [{"start": str(DATES[0].date()), "hash": "a", "config": {}},
            {"start": str(DATES[6].date()), "hash": "b", "config": {}}]
    monkeypatch.setattr(S, "segment_pools", lambda pool, segs: pools)
    monkeypatch.setattr(P, "build_point_signal", lambda p, s, e: _window(full, s, e))
    monkeypatch.setattr(M, "trade_calendar", lambda s=None, e=None: DATES)

    legs = S.live_chain(pools[1], DATES[-1], segs)
    assert [len(leg.ps.dates) for leg in legs] == [6, 6]
    carried = legs[1].sim.holdings[0]["SH600000"]                 # 换规则不清仓，买入信息保留
    assert carried.entry_date == DATES[1] and carried.days == 5
    r1 = account(legs[0].sim, legs[0].ps, pools[0])
    r2 = account(legs[1].sim, legs[1].ps, pools[1], state=r1.attrs["state"])
    nav = pd.concat([r1, r2])["nav"]
    assert nav.iloc[-1] == pytest.approx(1e6 / 2 * close[-1, 0] / close[1, 0] + 1e6 / 2)   # 半仓持有到期末


def test_window_distribution():
    from stockanalysis.research.track import window_distribution

    rep = pd.DataFrame({"return": [0.01, 0.01, -0.02], "cost": 0.0, "bench": [0.0, 0.01, 0.0]})
    r, ex = window_distribution(rep, 2)
    assert r == pytest.approx([0.0201, 1.01 * 0.98 - 1])
    assert ex == pytest.approx([0.0201 - 0.01, 1.01 * 0.98 - 1 - 0.01])
