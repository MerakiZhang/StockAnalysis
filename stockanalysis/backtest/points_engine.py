"""买卖点池回测：selection/positions.simulate() 的成交记录 → 逐日净值（含成本），输出与 qlib report 同列。

记账口径：
    - 复权收盘价逐日计收益（含分红送转）；停牌日市值不变，复牌后按复牌价接续
    - 成交价为 points.deal_price（收盘价或当日均价）；均价成交时买入当天计入"均价 → 收盘"的盈亏，卖出按均价折回
    - 每日先卖后买；买入金额 = 买入前净值 / max_positions（现金不足则用尽现金），不足 max_positions 时留现金
    - 成本：买 = 佣金 + 滑点；卖 = 佣金 + 印花税 + 滑点；佣金单笔不低于 min_cost
    - 不考虑整手（100 股）取整与成交量上限（单只仓位约为净值 / 10，对流动性门槛以上的股票影响很小）
report 列：return（未扣成本日收益）、cost（成本 / 前一日净值）、bench（基准日收益）、turnover（成交额 / 前一日净值）、
          n_pos（持仓只数）、cash（现金占比）。与 backtest/metrics.segment_metrics 兼容。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _bench_returns(symbol: str, dates: pd.DatetimeIndex) -> pd.Series:
    from ..data.market import load_wide

    s, e = str((dates[0] - pd.Timedelta(days=10)).date()), str(dates[-1].date())
    c = load_wide([symbol], s, e, ("close",))["close"][symbol].astype(float).reindex(dates)
    gaps = int(c.isna().sum())
    if gaps:
        print(f"[warn] 基准 {symbol} 在回测区间缺 {gaps} 个交易日（缺日收益记 0，次日计入完整涨跌）", flush=True)
    return c.ffill().pct_change(fill_method=None).fillna(0.0)


def account(sim, ps, pool, bench: pd.Series | None = None, state: dict | None = None) -> pd.DataFrame:
    """逐日记账。state：起始账户 {cash, value: {代码: 收盘市值}, last: {代码: 复权收盘}}（跟踪段衔接用），
    默认 pool.account 现金空仓；期末账户写入 report.attrs["state"]。"""
    cost = pool.cost
    comm, stamp, slip = float(cost["commission"]), float(cost["stamp_tax"]), float(cost.get("slippage", 0.0))
    min_cost = float(cost.get("min_cost", 5))
    n_max = pool.points.max_positions
    col = ps.colmap

    def fee(value: float, sell: bool) -> float:
        return max(value * comm, min_cost) + value * (slip + (stamp if sell else 0.0))

    by_day: dict[pd.Timestamp, list] = {}
    for t in sim.trades:
        by_day.setdefault(t.date, []).append(t)

    if state is None:
        cash, value, last = float(pool.account), {}, {}
    else:
        cash, value, last = float(state["cash"]), dict(state["value"]), dict(state["last"])
    max_w = pool.points.max_weight
    trim: set[str] = set()   # 上一检查日超过权重上限、今日减到 1 / n_max 的股票
    rows = []
    for i, d in enumerate(sim.dates):
        nav_prev = cash + sum(value.values())
        pnl = 0.0
        for code in list(value):
            c = ps.close[i, col[code]]
            if np.isfinite(c) and np.isfinite(last.get(code, np.nan)):
                new = value[code] * c / last[code]
                pnl += new - value[code]
                value[code] = new
            if np.isfinite(c):
                last[code] = c
        fees, traded = 0.0, 0.0
        todays = by_day.get(d, [])
        for t in (x for x in todays if x.side == "sell"):
            v = value.pop(t.code)
            c = ps.close[i, col[t.code]]
            if np.isfinite(c) and c > 0 and t.price != c:   # 均价成交：从收盘市值折回成交价
                pnl += v * (t.price / c - 1)
                v *= t.price / c
            last.pop(t.code, None)
            f = fee(v, sell=True)
            cash += v - f
            fees, traded = fees + f, traded + v
        nav_now = cash + sum(value.values())
        for code in sorted(trim & set(value)):
            j = col[code]
            excess = value[code] - nav_now / n_max
            if excess <= 0 or not ps.can_sell[i, j]:
                continue
            c, px = ps.close[i, j], ps.fill[i, j]
            v = excess * (px / c) if np.isfinite(c) and c > 0 and np.isfinite(px) else excess   # 收盘市值 → 成交额
            pnl += v - excess
            value[code] -= excess
            f = fee(v, sell=True)
            cash += v - f
            fees, traded = fees + f, traded + v
        trim = set()
        nav_now = cash + sum(value.values())
        for t in (x for x in todays if x.side == "buy"):
            target = nav_now / n_max
            rate = comm + slip
            amt = max(min(target, cash / (1 + rate)), 0.0)   # 现金用尽时记零仓位，保持与模拟器持仓一致
            f = fee(amt, sell=False) if amt > 0 else 0.0
            cash -= amt + f
            c = ps.close[i, col[t.code]]
            mark = amt * c / t.price if np.isfinite(c) and t.price > 0 else amt   # 成交价 → 当日收盘市值
            pnl += mark - amt
            value[t.code], last[t.code] = mark, (c if np.isfinite(c) else t.price)
            fees, traded = fees + f, traded + amt
        held = set(sim.holdings[i])
        if set(value) != held:   # 与模拟器持仓逐日一致（同一实现的自检）
            raise AssertionError(f"{d.date()} 记账持仓 {sorted(value)} 与模拟器 {sorted(held)} 不一致")
        nav = cash + sum(value.values())
        if max_w is not None and ps.signal_day[i] and nav > 0:
            trim = {c for c, v in value.items() if v > max_w * nav}
        rows.append({"return": pnl / nav_prev if nav_prev > 0 else 0.0, "cost": fees / nav_prev if nav_prev > 0 else 0.0,
                     "turnover": traded / nav_prev if nav_prev > 0 else 0.0,
                     "n_pos": len(value), "cash": cash / nav if nav > 0 else 1.0, "nav": nav,
                     "max_w": max(value.values()) / nav if value and nav > 0 else 0.0})
    report = pd.DataFrame(rows, index=sim.dates)
    report["bench"] = bench.reindex(sim.dates).fillna(0.0) if bench is not None else 0.0
    report.attrs["state"] = {"cash": cash, "value": dict(value), "last": dict(last)}
    return report


def trade_stats(trips: pd.DataFrame, pool) -> dict:
    """完整交易统计（扣往返成本）：笔数、胜率、平均收益、盈亏比、平均持有天数。"""
    done = trips.dropna(subset=["收益"])
    if done.empty:
        return {"trades": 0}
    c = pool.cost
    rt = 2 * float(c["commission"]) + float(c["stamp_tax"]) + 2 * float(c.get("slippage", 0.0))
    r = done["收益"] - rt
    win, loss = r[r > 0], r[r <= 0]
    return {"trades": int(len(r)), "win_rate": float((r > 0).mean()), "avg_ret": float(r.mean()),
            "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else np.nan,
            "avg_days": float(done["持有天数"].mean())}


def run_points_backtest(pool, start: str, end: str, ctx: dict | None = None, ps=None) -> dict:
    """买卖点池回测：信号 → 模拟 → 记账 → 指标。ps 可传入已构造的 PointSignal（参数网格复用）。"""
    from ..selection.points import build_point_signal
    from ..selection.positions import round_trips, simulate
    from .metrics import segment_metrics

    ps = ps or build_point_signal(pool, start, end, ctx)
    sim = simulate(ps, pool.points)
    report = account(sim, ps, pool, _bench_returns(pool.benchmark, ps.dates))
    trips = round_trips(sim)
    out = {"report": report, "sim": sim, "ps": ps, "trips": trips, **segment_metrics(report, start, end),
           **trade_stats(trips, pool), "avg_pos": float(report["n_pos"].mean()),
           "exposure": float(1 - report["cash"].mean())}
    reasons = trips["卖出原因"].str.split("；").explode().map(reason_key)
    out["sell_reasons"] = reasons.value_counts().to_dict() if len(trips) else {}
    return out


def reason_key(reason: str) -> str:
    """卖出原因归类（去掉参数）："止损 -8%" → "止损"、"移动止盈（高点回撤 20%）" → "移动止盈"。"""
    return re.split(r"[\s（]", str(reason), maxsplit=1)[0]
