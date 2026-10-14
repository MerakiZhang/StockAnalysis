"""股票池过滤：ST、次新、停牌、涨跌停、流动性、板块资格。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..data.market import data_dir
from ..factors.base import daily_returns
from ..factors.technical import limit_up_threshold


@lru_cache(maxsize=1)
def listing_dates() -> dict[str, pd.Timestamp]:
    """从 qlib instruments/all.txt 读取每只股票的起始日期（近似上市日）。"""
    path = Path(data_dir()) / "instruments" / "all.txt"
    out: dict[str, pd.Timestamp] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = pd.Timestamp(parts[1])
    return out


def listed_days(code: str, date: pd.Timestamp) -> int:
    ld = listing_dates().get(code)
    return (date - ld).days if ld is not None else 9999


def board_of(code: str) -> str:
    c = code[-6:]
    if c.startswith("68"):
        return "kcb"
    if c.startswith("30"):
        return "cyb"
    if c.startswith(("4", "8", "92")):
        return "bse"
    return "main"


def in_boards(code: str, boards: list[str] | tuple[str, ...] | None) -> bool:
    """code 是否在允许的板块内。boards 为空/None 表示不限制。"""
    if not boards:
        return True
    return board_of(code) in boards


_STOCK_PREFIX = ("60", "68", "000", "001", "002", "003", "30", "43", "83", "87", "88", "92")


def is_stock(code: str) -> bool:
    """剔除指数（SH000xxx/SZ399xxx）、基金等，只留 A 股个股。"""
    c = code[-6:]
    if code.startswith("SH") and c.startswith("000"):
        return False  # 上证指数族
    if code.startswith("SZ") and c.startswith("399"):
        return False  # 深证指数族
    return c.startswith(_STOCK_PREFIX)


def build_filter_mask(
    date: pd.Timestamp,
    codes: list[str],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
    st_names: set[str] | None = None,
    cfg: dict | None = None,
) -> pd.Series:
    """返回 date 当日各 code 是否通过过滤的 Series（True=保留）。

    cfg 键：min_list_days / min_amount / exclude_st / exclude_suspend_days /
    exclude_limit_on_rebalance / boards（允许板块，如 ["main"]）/
    chase_limit(短线用, 近5日涨幅上限) / falling_knife(短线用, 近5日跌幅下限)
    st_names: 当前 ST 代码集合（历史 ST 变动无数据源，为已知近似）
    """
    cfg = cfg or {}
    date = pd.Timestamp(date)
    codes = [c for c in codes if c in close.columns]
    ok = pd.Series(True, index=codes)

    boards = cfg.get("boards")
    if boards:
        ok &= pd.Series([in_boards(c, boards) for c in codes], index=codes)

    if cfg.get("exclude_st", True) and st_names is not None:
        ok &= ~ok.index.isin(st_names)

    min_days = int(cfg.get("min_list_days", 0))
    if min_days > 0:
        ok &= pd.Series(
            [listed_days(c, date) >= min_days for c in codes], index=codes
        )

    # 停牌：近 N 日有成交量为 NaN 或 0
    susp_days = int(cfg.get("exclude_suspend_days", 0))
    if susp_days > 0:
        hist_vol = volume.loc[:date, codes].tail(susp_days)
        suspended = hist_vol.isna().any() | hist_vol.eq(0).any()
        ok &= ~suspended

    # 调仓日涨跌停不可成交
    if cfg.get("exclude_limit_on_rebalance", True) and date in close.index:
        ret = daily_returns(close.loc[:date, codes]).iloc[-1]
        thr = pd.Series({c: limit_up_threshold(c) for c in codes})
        at_limit = ret.abs() >= (thr - 0.005)
        ok &= ~at_limit.fillna(False)

    # 流动性：当日成交额下限（qlib amount 单位为千元，阈值单位为元）
    min_amount = float(cfg.get("min_amount", 0))
    if min_amount > 0 and date in amount.index:
        ok &= amount.loc[date, codes].fillna(0) * 1000 >= min_amount

    # 短线防追高：近5日累计涨幅上限
    chase = cfg.get("chase_limit")
    if chase is not None and date in close.index:
        mom5 = close.loc[:date, codes].iloc[-1] / close.loc[:date, codes].iloc[-6] - 1
        ok &= ~(mom5 > float(chase)).fillna(False)

    # 短线不接飞刀：近5日累计跌幅下限
    knife = cfg.get("falling_knife")
    if knife is not None and date in close.index:
        mom5 = close.loc[:date, codes].iloc[-1] / close.loc[:date, codes].iloc[-6] - 1
        ok &= ~(mom5 < float(knife)).fillna(False)

    return ok
