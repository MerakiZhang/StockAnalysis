"""买卖点状态机（selection/positions.py）与记账（backtest/points_engine.py）：时序、成交约束、各类卖点、排队与上限。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stockanalysis.backtest.points_engine import account, reason_key
from stockanalysis.config.schema import PointsSpec
from stockanalysis.selection.points import PointSignal
from stockanalysis.selection.positions import round_trips, simulate

DATES = pd.bdate_range("2021-01-04", periods=12)


def make_ps(n_codes=3, close=None, buy=None, score=None, **kw) -> PointSignal:
    shape = (len(DATES), n_codes)
    codes = [f"SH60000{j}" for j in range(n_codes)]
    close = np.full(shape, 10.0) if close is None else np.asarray(close, dtype=float)
    args = dict(dates=DATES, codes=codes, close=close,
                buy=np.zeros(shape, bool) if buy is None else np.asarray(buy, bool),
                buy_hits={}, sell_hits={}, exit_ok=np.ones(shape, bool), eligible=np.ones(shape, bool),
                score=np.tile(np.arange(n_codes, 0, -1, dtype=float), (len(DATES), 1)) if score is None else score,
                score_pct=np.full(shape, 0.9), atr=np.full(shape, 0.5),
                can_buy=np.ones(shape, bool), can_sell=np.ones(shape, bool),
                regime_ok=np.ones(len(DATES), bool), signal_day=np.ones(len(DATES), bool),
                industry=[None] * n_codes)
    args.update(kw)
    return PointSignal(**args)


def spec(**kw) -> PointsSpec:
    return PointsSpec(**{"max_positions": 2, "buy": {"x": {}}, **kw})


def test_buy_executes_next_day_at_close():
    buy = np.zeros((len(DATES), 3), bool)
    buy[0, 0] = True
    close = np.full((len(DATES), 3), 10.0)
    close[1, 0] = 10.5
    sim = simulate(make_ps(buy=buy, close=close), spec())
    assert sim.holdings[0] == {}                               # 信号日当天不成交
    p = sim.holdings[1]["SH600000"]
    assert p.entry_date == DATES[1] and p.entry_price == 10.5 and p.days == 0
    assert sim.holdings[3]["SH600000"].days == 2
    assert sim.trades[0].signal_date == DATES[0]


def test_limit_up_on_execution_day_skips_buy():
    buy = np.zeros((len(DATES), 3), bool)
    buy[0, 0] = True
    can_buy = np.ones((len(DATES), 3), bool)
    can_buy[1, 0] = False
    sim = simulate(make_ps(buy=buy, can_buy=can_buy), spec())
    assert "SH600000" not in sim.holdings[1] and (DATES[1], "SH600000", "buy") in sim.unfilled


def test_stop_loss_and_limit_down_delay():
    buy = np.zeros((len(DATES), 3), bool)
    buy[0, 0] = True
    close = np.full((len(DATES), 3), 10.0)
    close[4:, 0] = 9.0                                         # 第 4 天收盘跌 10% → 触发止损 8%
    can_sell = np.ones((len(DATES), 3), bool)
    can_sell[5, 0] = False                                     # 第 5 天跌停卖不出
    sim = simulate(make_ps(buy=buy, close=close, can_sell=can_sell), spec(stop_loss=0.08))
    assert "SH600000" in sim.holdings[5]
    assert "SH600000" not in sim.holdings[6]
    sell = [t for t in sim.trades if t.side == "sell"][0]
    assert sell.date == DATES[6] and "止损" in sell.reason


def test_trailing_stop_uses_peak_close():
    buy = np.zeros((len(DATES), 3), bool)
    buy[0, 0] = True
    close = np.full((len(DATES), 3), 10.0)
    close[2:6, 0] = [12, 14, 13, 11.1]                          # 高点 14，回撤 20% 线 = 11.2
    sim = simulate(make_ps(buy=buy, close=close), spec(trailing_stop=0.2))
    assert "SH600000" in sim.holdings[5] and "SH600000" not in sim.holdings[6]


def test_priority_capacity_and_industry_cap():
    buy = np.ones((len(DATES), 3), bool)
    score = np.tile([1.0, 3.0, 2.0], (len(DATES), 1))
    sim = simulate(make_ps(buy=buy, score=score), spec())
    assert set(sim.holdings[1]) == {"SH600001", "SH600002"}    # 上限 2，按分数取前 2
    sim2 = simulate(make_ps(buy=buy, score=score, industry=["A", "B", "B"]), spec(industry_max=1))
    assert set(sim2.holdings[1]) == {"SH600001", "SH600000"}   # 行业 B 只能 1 只


def test_regime_and_weekly_block_new_buys_but_not_stops():
    buy = np.ones((len(DATES), 3), bool)
    sim = simulate(make_ps(buy=buy, regime_ok=np.zeros(len(DATES), bool)), spec())
    assert all(h == {} for h in sim.holdings)
    signal_day = np.zeros(len(DATES), bool)
    signal_day[0] = True
    close = np.full((len(DATES), 3), 10.0)
    close[3:, :] = 8.0
    sim = simulate(make_ps(buy=buy, close=close, signal_day=signal_day), spec(stop_loss=0.1))
    assert len(sim.holdings[1]) == 2 and sim.holdings[4] == {}  # 非信号日止损照常执行
    assert sim.holdings[-1] == {}                               # 非信号日不补仓


def test_fill_always_and_max_hold():
    sim = simulate(make_ps(), spec(fill="always", max_hold=3))
    assert len(sim.holdings[1]) == 2
    assert all(p.reason.startswith("补位") for p in sim.holdings[1].values())
    first = [t for t in sim.trades if t.side == "sell"][0]
    assert "持有满 3 天" in first.reason


def test_exit_rule_forces_sell():
    buy = np.zeros((len(DATES), 3), bool)
    buy[0, 0] = True
    exit_ok = np.ones((len(DATES), 3), bool)
    exit_ok[3, 0] = False
    sim = simulate(make_ps(buy=buy, exit_ok=exit_ok), spec())
    assert "SH600000" not in sim.holdings[4]
    assert reason_key([t for t in sim.trades if t.side == "sell"][0].reason) == "持有规则不通过"


def test_account_returns_and_costs():
    buy = np.zeros((len(DATES), 1), bool)
    buy[0, 0] = True
    close = np.full((len(DATES), 1), 10.0)
    close[2:, 0] = 11.0                                         # 买入次日涨 10%
    close[4:, 0] = 8.0
    ps = make_ps(n_codes=1, buy=buy, close=close)
    sp = spec(max_positions=1, stop_loss=0.1)
    sim = simulate(ps, sp)
    pool = SimpleNamespace(account=1e6, points=sp,
                           cost={"commission": 0.0, "stamp_tax": 0.001, "slippage": 0.0, "min_cost": 0.0})
    rep = account(sim, ps, pool)
    nav_after_buy = rep["nav"].iloc[1]
    assert rep["return"].iloc[2] == pytest.approx(0.1 * nav_after_buy / rep["nav"].iloc[1], rel=1e-6)
    sell_day = [t.date for t in sim.trades if t.side == "sell"][0]
    assert rep.loc[sell_day, "cost"] > 0 and rep.loc[sell_day, "n_pos"] == 0
    trips = round_trips(sim)
    assert trips["收益"].iloc[0] == pytest.approx(8.0 / 10.0 - 1)


def test_grid_expand_cartesian_and_pipe_separator():
    from stockanalysis.research.points_grid import expand

    out = expand(["points.stop_loss=0.05,null", "points.buy={a: {}, b: {}}|{a: {}}"])
    assert len(out) == 4
    assert out[0] == ["points.stop_loss=0.05", "points.buy={a: {}, b: {}}"]
    assert expand([]) == [[]]


def test_sell_lines_in_raw_price_and_near_buy():
    import pandas as pd

    from stockanalysis.selection.points_daily import _sell_lines, near_buy
    from stockanalysis.selection.positions import Position

    n = 3
    ps = make_ps(n_codes=n)
    shape = ps.close.shape
    ps.extra = {"factor": np.full(shape, 2.0),                          # 复权价 / 2 = 不复权价
                "ma20": np.full(shape, 18.0),
                "buy_parts": {"a": np.ones(shape, bool), "b": np.zeros(shape, bool)},
                "buy_any_parts": {}, "avoid_parts": {}}
    ps.extra["buy_parts"]["b"][:, 2] = True                             # 第 3 只两条都满足（已是买点）
    ps.buy[:, 2] = True
    sp = spec(stop_loss=0.1, trailing_stop=0.2, max_hold=10, sell={"below_ma": {"n": 20}}, buy={"a": {}, "b": {}})
    p = Position("SH600000", pd.Timestamp("2021-01-04"), 20.0, 1.0, 25.0, 4, "买点")
    lines = _sell_lines(p, 5, 0, ps, sp)
    assert lines["止损价"] == pytest.approx(9.0)          # 20 × 0.9 / 2
    assert lines["移动止盈价"] == pytest.approx(10.0)     # 25 × 0.8 / 2
    assert lines["MA20"] == pytest.approx(9.0)
    assert lines["剩余天数"] == 6
    near = near_buy(ps, sp, 5, held={"SH600000"})
    assert near["code"].tolist() == ["SH600001"] and near["缺少条件"].iloc[0] == "b"


def test_slice_from_and_ensemble():
    from stockanalysis.research.points_grid import ensemble
    from stockanalysis.selection.points import slice_from

    buy = np.ones((len(DATES), 3), bool)
    ps = make_ps(buy=buy, signal_day=np.zeros(len(DATES), bool))
    sub = slice_from(ps, 3)
    assert len(sub.dates) == len(DATES) - 3 and sub.signal_day[0] and not sub.signal_day[1:].any()
    sim = simulate(sub, spec())
    assert len(sim.holdings[1]) == 2                     # 起点当天建仓、次日成交
    r1 = pd.DataFrame({"return": [0.01, 0.02, 0.0], "cost": 0.0, "turnover": 0.0, "n_pos": 1, "cash": 0.0,
                       "bench": 0.0}, index=DATES[:3])
    r2 = pd.DataFrame({"return": [0.03, 0.0], "cost": 0.0, "turnover": 0.0, "n_pos": 1, "cash": 0.0,
                       "bench": 0.0}, index=DATES[1:3])
    ens = ensemble([r1, r2])
    assert ens["return"].tolist() == pytest.approx([0.01, 0.025, 0.0])   # 未开始的起点不参与平均
