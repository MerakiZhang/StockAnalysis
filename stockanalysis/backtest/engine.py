"""回测引擎（qlib 0.9.7），池无关：策略、成本、账户、撮合约束全部取自 PoolSpec。

信号对齐约定（重要）：
qlib TopkDropoutStrategy 系策略用 `get_step_time(trade_step, shift=1)` 取**前一交易日**
的分数在当日下单（deal_price=close → T 日打分、T+1 收盘成交）。
因此 Signal 的日期就是"信号日"；daily 池每天都是信号日，weekly/monthly 池由
scores_on_rebalance_dates() 折叠到调仓日的前一交易日。
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from ..data.market import init_qlib
from ..selection.schedule import signal_mask
from ..selection.signal import Signal
from .metrics import segment_metrics
from .strategies import STRATEGIES


def scores_on_rebalance_dates(score_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """把日频分数宽表折叠成只在信号日有值的表（其余日期 NaN，策略当天不交易）。

    信号日见 selection/schedule.py（与选股同一日程）：daily 每天；weekly / monthly 为区间首日 +
    每个周期的最后一个交易日（策略 shift=1，下一周期首个交易日成交）。
    """
    if freq == "daily":
        return score_daily
    keep = signal_mask(score_daily.index, freq)
    return score_daily.where(pd.Series(keep, index=score_daily.index), axis=0)


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

    signal: build_signal() 的结果（日频）；按 pool.rebalance.freq 折叠到信号日。
    返回指标 dict，含 report（qlib 日频表）与 positions（{日期: Position}）。
    """
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor

    init_qlib()
    folded = replace(signal, score=scores_on_rebalance_dates(signal.score, pool.rebalance.freq))
    if folded.score.dropna(how="all").empty:
        return {"error": "signal is empty"}

    strategy = STRATEGIES.get(pool.strategy["name"])(folded, pool.rebalance, **pool.strategy_params)
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
