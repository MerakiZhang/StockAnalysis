"""回测引擎（qlib 0.9.7）。

- stable 池：TopkDropoutStrategy 月度调仓
- short 池：ShortPoolStrategy（见 strategies.py），在周度调仓之外
  叠加单票止损 / 最长持有天数强制卖出

信号对齐约定（重要）：
qlib 的 TopkDropoutStrategy.generate_trade_decision 用
`get_step_time(trade_step, shift=1)` 取**前一交易日**的分数在当日下单。
因此 scores 宽表中，分数必须放在"调仓日的前一交易日"，
由 scores_on_rebalance_dates() 完成折叠，调用方只需给日频分数表。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.contrib.strategy import TopkDropoutStrategy

from ..data.market import init_qlib
from .strategies import ShortPoolStrategy

ACCOUNT = 1e8  # 回测初始资金


def scores_on_rebalance_dates(score_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    """把日频分数宽表折叠成只在信号日有值的表（其余日期 NaN）。

    freq:
        "monthly" —— 每月首个交易日调仓，信号放在上月最后一个交易日
        "weekly"  —— 每周首个交易日调仓（周五收盘打分），信号放在上周五

    由于策略 shift=1 取数，信号日 = 调仓日的前一交易日；
    回测区间的第一个周期没有"前一交易日"可用时，信号放在首日
    （实际成交发生在第 2 个交易日，仅影响第一个周期）。
    """
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


def _to_signal(scores: pd.DataFrame) -> pd.Series:
    """宽表（index=交易日, columns=代码）-> MultiIndex(datetime, instrument) Series。

    全 NaN 的行（非信号日）被丢弃，策略在这些日期取到 None，持仓不动。
    """
    s = scores.stack()  # pandas>=2 默认 dropna
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


def wf_segments(bt_start: str, in_sample_end: str, bt_end: str) -> list[tuple[str, str, str]]:
    """walk-forward 三段 + 样本内/外合计。返回 (label, start, end) 列表。"""
    b0, b1 = pd.Timestamp(bt_start), pd.Timestamp(bt_end)
    ins = pd.Timestamp(in_sample_end)
    bounds = [
        (b0, pd.Timestamp("2020-12-31")),
        (pd.Timestamp("2021-01-01"), ins),
        (ins + pd.Timedelta(days=1), b1),
    ]
    segs: list[tuple[str, str, str]] = []
    for s, e in bounds:
        s, e = max(s, b0), min(e, b1)
        if s <= e:
            segs.append((f"{s.year}~{e.year}", str(s.date()), str(e.date())))
    segs.append(("样本内", str(b0.date()), str(ins.date())))
    segs.append(("样本外", str((ins + pd.Timedelta(days=1)).date()), str(b1.date())))
    return segs


def _empty_result(reason: str) -> dict:
    return {
        "annualized_return": np.nan,
        "excess_annualized_return": np.nan,
        "max_drawdown": np.nan,
        "information_ratio": np.nan,
        "win_rate": np.nan,
        "turnover_mean": np.nan,
        "report": None,
        "excess_analysis": None,
        "error": reason,
    }


def run_backtest(
    scores: pd.DataFrame,
    pool: str,
    start: str,
    end: str,
    cfg: dict,
) -> dict:
    """跑一个池的回测。

    scores: 宽表（index=交易日, columns=qlib 代码），只在信号日有值
            （由 scores_on_rebalance_dates 生成），值越大越好。
    pool:   "stable" | "short"，读取 cfg 对应段（topk/n_drop/benchmark/止损等）。
    返回:   指标 dict，含 report（qlib report_normal 日频表，可画净值）。
    """
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.evaluate import risk_analysis

    init_qlib()
    if pool not in ("stable", "short"):
        raise ValueError(f"unknown pool: {pool}")
    pcfg = cfg[pool]
    cost = cfg["cost"]
    slippage = cost["slippage_stable" if pool == "stable" else "slippage_short"]
    open_cost = cost["commission"] + slippage                      # 买入：佣金 + 滑点
    close_cost = cost["commission"] + cost["stamp_tax"] + slippage  # 卖出：佣金 + 印花税 + 滑点

    signal = _to_signal(scores)
    if signal.empty:
        return _empty_result("signal is empty")

    strategy_kwargs = dict(
        topk=int(pcfg["topk"]),
        n_drop=int(pcfg["n_drop"]),
        method_sell="bottom",
        method_buy="top",
        hold_thresh=int(pcfg.get("hold_thresh", 1)),
        only_tradable=True,
        signal=signal,
    )
    if pool == "short":
        sl = pcfg.get("stop_loss", -0.05)
        strategy = ShortPoolStrategy(
            stop_loss=None if sl is None else float(sl),
            max_hold_days=int(pcfg.get("max_hold_days", 10)),
            **strategy_kwargs,
        )
    else:
        strategy = TopkDropoutStrategy(**strategy_kwargs)

    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    exchange_kwargs = {
        "codes": list(scores.columns),
        "deal_price": "close",
        "open_cost": open_cost,
        "close_cost": close_cost,
        "min_cost": 5,
    }
    portfolio_metric_dict, _ = backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor,
        benchmark=pcfg["benchmark"],
        account=ACCOUNT,
        exchange_kwargs=exchange_kwargs,
    )

    report_normal, _ = portfolio_metric_dict["1day"]
    # qlib report 的 return/bench 列是小数日收益（非 1+r 比率），cost 是小数费率
    net = report_normal["return"] - report_normal["cost"]          # 含成本日收益
    excess = net - report_normal["bench"]                          # 含成本超额日收益
    ana = risk_analysis(excess)
    ana_net = risk_analysis(net)

    result = {
        "annualized_return": float(ana_net.loc["annualized_return", "risk"]),
        "excess_annualized_return": float(ana.loc["annualized_return", "risk"]),
        "max_drawdown": float(ana.loc["max_drawdown", "risk"]),
        "information_ratio": float(ana.loc["information_ratio", "risk"]),
        "win_rate": float((excess > 0).mean()),                    # 日度跑赢基准比例
        "turnover_mean": float(report_normal["turnover"].mean()),  # 日均单边换手
        "report": report_normal,
        "excess_analysis": ana,
    }
    if isinstance(strategy, ShortPoolStrategy):
        result["stop_loss_count"] = strategy.stop_loss_count
        result["max_hold_count"] = strategy.max_hold_count
    return result
