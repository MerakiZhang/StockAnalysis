"""滚动训练窗口（walk-forward）。

每个测试区间一个窗口（默认按年）：train（train_years 年）→ gap → valid（valid_months 个月）→ gap → test。
gap = label_horizon + 1 个交易日：标签用到未来 h+1 天的收盘价，不留空档会把下一段价格泄漏进训练。
qlib RollingGen 不支持空档，因此在此生成。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Window:
    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]

    @property
    def name(self) -> str:
        return f"test_{self.test[0]}"


def rolling_windows(calendar: pd.DatetimeIndex, first_test: str, end: str, train_years: int = 4,
                    valid_months: int = 6, gap: int = 6, retrain: str = "yearly") -> list[Window]:
    cal = pd.DatetimeIndex(calendar)
    end_ts = min(pd.Timestamp(end), cal[-1])
    freq = {"yearly": "YS", "quarterly": "QS"}[retrain]
    starts = pd.date_range(pd.Timestamp(first_test), end_ts, freq=freq)
    if len(starts) == 0 or starts[0] > pd.Timestamp(first_test):
        starts = pd.DatetimeIndex([pd.Timestamp(first_test)]).append(starts)
    offset = pd.DateOffset(years=1) if retrain == "yearly" else pd.DateOffset(months=3)

    def day(i: int) -> str:
        return str(cal[max(i, 0)].date())

    windows = []
    for ps in starts:
        i = cal.searchsorted(ps)
        if i >= len(cal) or cal[i] > end_ts:
            break
        period_end = (pd.Timestamp(ps.year, ps.month, 1) + offset) - pd.Timedelta(days=1)
        te = cal[cal.searchsorted(min(period_end, end_ts), side="right") - 1]
        ve_i = i - gap - 1
        vs_i = cal.searchsorted(cal[ve_i] - pd.DateOffset(months=valid_months) + pd.Timedelta(days=1))
        tre_i = vs_i - gap - 1
        trs_i = cal.searchsorted(cal[tre_i] - pd.DateOffset(years=train_years) + pd.Timedelta(days=1))
        windows.append(Window(train=(day(trs_i), day(tre_i)), valid=(day(vs_i), day(ve_i)),
                              test=(day(i), str(te.date()))))
    return windows


def pool_windows(pool, calendar: pd.DatetimeIndex, first_test: str, end: str) -> list[Window]:
    """按池 model 配置生成窗口。"""
    m = pool.model
    horizon = int(m.get("label_horizon", 5))
    return rolling_windows(calendar, first_test, end, int(m.get("train_years", 4)), int(m.get("valid_months", 6)),
                           horizon + 1, m.get("retrain", "yearly"))
