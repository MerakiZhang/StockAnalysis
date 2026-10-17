"""每日选股流程：确定信号日 → 买卖点模拟器从 points.live_start 推演到 T（selection/points_daily.py）→ 写股池与简报。"""
from __future__ import annotations

import pandas as pd

from ..config import load_pool


def signal_date(date: str | None = None) -> pd.Timestamp:
    """不晚于 date 的最近交易日（默认数据最新交易日）。"""
    from ..data.market import latest_trade_date, trade_calendar

    if date is None:
        return latest_trade_date()
    cal = trade_calendar(end=date)
    if not len(cal):
        raise ValueError(f"{date} 之前无交易日数据")
    return cal[-1]


def run_selection(name: str, date: str | None = None, names: dict[str, str] | None = None,
                  health: bool = True, sets: list[str] | None = None):
    from .points_daily import run_points_selection

    pool = load_pool(name, sets=sets)
    t = signal_date(date)
    print(f"\n===== {name} 买卖点池 @ {t.date()}（status={pool.status}）=====", flush=True)
    return run_points_selection(pool, t, names or {}, health=health)
