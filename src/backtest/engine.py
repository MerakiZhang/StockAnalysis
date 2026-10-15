"""回测引擎（qlib 0.9.7），池无关：策略、成本、账户、撮合约束全部取自 PoolSpec。

信号对齐约定（重要）：
qlib TopkDropoutStrategy 系策略用 `get_step_time(trade_step, shift=1)` 取**前一交易日**
的分数在当日下单（deal_price=close → T 日打分、T+1 收盘成交）。
因此 Signal 的日期就是"信号日"；daily 池每天都是信号日，weekly/monthly 池由
scores_on_rebalance_dates() 折叠到调仓日的前一交易日。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..data.market import init_qlib
from ..selection.signal import Signal
from .strategies import STRATEGIES


def scores_on_rebalance_dates(score_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """把日频分数宽表折叠成只在信号日有值的表（其余日期 NaN）。

    freq:
        "daily"   —— 每天都是信号日（原样返回）
        "monthly" —— 每月首个交易日调仓，信号放在上月最后一个交易日
        "weekly"  —— 每周首个交易日调仓（周五收盘打分），信号放在上周五

    由于策略 shift=1 取数，信号日 = 调仓日的前一交易日；
    回测区间的第一个周期没有"前一交易日"可用时，信号放在首日
    （实际成交发生在第 2 个交易日，仅影响第一个周期）。
    """
    if freq == "daily":
        return score_daily
    dates = pd.DatetimeIndex(score_daily.index)
    if freq == "monthly":
        period = dates.to_period("M")
    elif freq == "weekly":
        period = dates.to_period("W")
    else:
        raise ValueError(f"unsupported freq: {freq}")

    # 每个周期首个交易日 = 调仓日（注意：PeriodIndex 不能直接用 Series.diff）
    p = np.asarray(period)
    is_period_first = np.empty(len(p), dtype=bool)
    is_period_first[0] = True
    is_period_first[1:] = p[1:] != p[:-1]
    first_idx = np.flatnonzero(is_period_first)
    signal_idx = np.array([i - 1 if i > 0 else 0 for i in first_idx])
    signal_dates = dates[signal_idx]

    out = pd.DataFrame(np.nan, index=dates, columns=score_daily.columns)
    out.loc[signal_dates] = score_daily.loc[signal_dates].to_numpy()
    return out


def segments(bt_start: str, in_sample_end: str, bt_end: str) -> list[tuple[str, str, str]]:
    """逐年 + 研发区 / 保留集 + 全区间。返回 (label, start, end) 列表。"""
    b0, b1, ins = pd.Timestamp(bt_start), pd.Timestamp(bt_end), pd.Timestamp(in_sample_end)
    segs: list[tuple[str, str, str]] = []
    for y in range(b0.year, b1.year + 1):
        s, e = max(b0, pd.Timestamp(f"{y}-01-01")), min(b1, pd.Timestamp(f"{y}-12-31"))
        if s <= e:
            segs.append((str(y), str(s.date()), str(e.date())))
    if b0 <= ins:
        segs.append(("研发区", str(b0.date()), str(min(ins, b1).date())))
    if ins < b1:
        segs.append(("保留集", str((ins + pd.Timedelta(days=1)).date()), str(b1.date())))
    segs.append(("全区间", str(b0.date()), str(b1.date())))
    return segs


def segment_metrics(report: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """从 qlib 日频 report 切片算区间指标（含成本，复利口径）。"""
    from qlib.contrib.evaluate import risk_analysis

    seg = report.loc[start:end]
    if len(seg) < 2:
        return {}
    net = seg["return"] - seg["cost"]
    excess = net - seg["bench"]
    # product 模式：复利口径（年化为几何年化、回撤不会超过 -100%）；qlib 默认 sum 为简单累加
    ana = risk_analysis(excess, mode="product")["risk"]
    ana_net = risk_analysis(net, mode="product")["risk"]
    return {
        "ann": float(ana_net["annualized_return"]),
        "excess_ann": float(ana["annualized_return"]),
        "mdd": float(ana["max_drawdown"]),
        "ir": float(ana["information_ratio"]),
        "win_rate": float((excess > 0).mean()),
        "turnover": float(seg["turnover"].mean()),
        "cost": float(seg["cost"].sum() / max(len(seg) / 238, 1e-9)),  # 年化成本占比
    }


def _exchange_kwargs(pool, codes: list[str]) -> dict:
    cost = pool.cost
    slippage = float(cost.get("slippage", 0.0))
    kw = {
        "codes": codes,
        "deal_price": pool.exchange.get("deal_price", "close"),
        "open_cost": float(cost["commission"]) + slippage,                               # 买：佣金 + 滑点
        "close_cost": float(cost["commission"]) + float(cost["stamp_tax"]) + slippage,   # 卖：+ 印花税
        "min_cost": float(cost.get("min_cost", 5)),
    }
    vl = pool.exchange.get("volume_limit")
    if vl:
        # qlib 下单数量单位 = 成交额 / 复权价（股）；本数据 $volume 单位为"手"，不能直接比，
        # 改用成交额折算：当日可成交股数 = $amount(千元) × 1000 / $close
        kw["volume_threshold"] = {"all": ("current", f"{float(vl)} * $amount * 1000 / $close")}
    boards = pool.filters.get("boards", {}).get("boards")
    if boards and set(boards) - {"main"}:
        # qlib cn 默认涨跌停阈值 9.5%（主板口径），对创业板/科创板/北交所偏严：会把 10%~20% 的
        # 涨幅误判为涨停而拒绝买入。目前三池均为主板，如开放其他板块需改为按板块的表达式。
        print(f"[warn] 池 {pool.name} 含非主板 {boards}：qlib 涨跌停阈值仍为 9.5%，回测偏保守")
    return kw


def run_backtest(signal: Signal, pool, start: str, end: str) -> dict:
    """跑一个池的回测。

    signal: build_signal() 的结果（日频）；按 pool.rebalance 折叠到信号日。
    返回指标 dict，含 report（qlib 日频表）与 positions（{日期: Position}）。
    """
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor

    init_qlib()
    folded = replace(signal, score=scores_on_rebalance_dates(signal.score, pool.rebalance))
    if folded.score.dropna(how="all").empty:
        return {"error": "signal is empty"}

    strategy = STRATEGIES.get(pool.strategy["name"])(folded, **pool.strategy_params)
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    portfolio_metric_dict, _ = backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor,
        benchmark=pool.benchmark,
        account=pool.account,
        exchange_kwargs=_exchange_kwargs(pool, list(signal.score.columns)),
    )
    report, positions = portfolio_metric_dict["1day"]
    result = {"report": report, "positions": positions, **segment_metrics(report, start, end)}
    return result


def holdings_by_date(positions: dict) -> dict[pd.Timestamp, list[str]]:
    """qlib positions -> {交易日: 收盘后持仓代码}。"""
    out = {}
    for d, pos in positions.items():
        out[pd.Timestamp(d)] = sorted(pos.get_stock_list())
    return out
