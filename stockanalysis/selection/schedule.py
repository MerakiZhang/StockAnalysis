"""调仓日程（选股与回测共用）：哪些交易日是"信号日"。

daily   每个交易日都是信号日
weekly  每周最后一个交易日（周五收盘打分，下周首个交易日成交）
monthly 每月最后一个交易日

回测（backtest/engine.py）在信号日以外不交易；选股（selection/daily.py）在非信号日沿用上期股池，
两者口径一致。区间首日总是信号日（建仓）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FREQ_PERIOD = {"weekly": "W", "monthly": "M"}


def signal_mask(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """dates 中每个交易日是否为信号日：首日 + 每个周期的最后一个交易日（= 下一周期首日的前一交易日）。"""
    dates = pd.DatetimeIndex(dates)
    if freq == "daily":
        return np.ones(len(dates), dtype=bool)
    if freq not in FREQ_PERIOD:
        raise ValueError(f"unsupported freq: {freq}")
    p = np.asarray(dates.to_period(FREQ_PERIOD[freq]))
    mask = np.zeros(len(dates), dtype=bool)
    if len(dates):
        mask[0] = True
        mask[:-1] |= p[1:] != p[:-1]
    return mask


def is_signal_day(date, freq: str) -> bool:
    """实盘：date 是否为信号日——下一个交易日（qlib 未来日历 day_future.txt）落在不同周期。

    没有未来日历时保守地视为信号日（照常检查买点与条件卖点）。
    """
    if freq == "daily":
        return True
    from qlib.data import D

    from ..data.market import init_qlib

    init_qlib()
    d = pd.Timestamp(date)
    try:
        nxt = [x for x in pd.DatetimeIndex(D.calendar(start_time=d, future=True)) if x > d]
    except Exception:  # noqa: BLE001 - 无 day_future.txt
        return True
    if not nxt:
        return True
    period = FREQ_PERIOD[freq]
    return d.to_period(period) != nxt[0].to_period(period)
