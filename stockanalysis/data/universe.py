"""指数成分与个股参考数据（来自 qlib instruments 文件）：板块判定、个股识别、上市日、成分时点面板。

过滤规则见 selection/filters.py。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .market import data_dir


@lru_cache(maxsize=1)
def listing_spans() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """qlib instruments/all.txt：每只证券的 (数据起始日, 数据截止日)；截止日早于最新交易日即已退市 / 转码。"""
    path = Path(data_dir()) / "instruments" / "all.txt"
    out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out[parts[0]] = (pd.Timestamp(parts[1]), pd.Timestamp(parts[2]))
    return out


def listing_dates() -> dict[str, pd.Timestamp]:
    """每只股票的起始日期（近似上市日）。"""
    return {c: s for c, (s, _) in listing_spans().items()}


def market_stocks(start: str, end: str, exchanges: tuple[str, ...] = ("SH", "SZ", "BJ")) -> list[str]:
    """[start, end] 内有交易数据的全部 A 股个股（含其后退市的，避免幸存者偏差）。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return sorted(c for c, (a, b) in listing_spans().items()
                  if is_stock(c) and c[:2] in exchanges and a <= e and b >= s)


def board_of(code: str) -> str:
    c = code[-6:]
    if c.startswith("68"):
        return "kcb"
    if c.startswith("30"):
        return "cyb"
    if c.startswith(("4", "8", "92")):
        return "bse"
    return "main"


_STOCK_PREFIX = ("60", "68", "000", "001", "002", "003", "30", "43", "83", "87", "88", "92")


def is_stock(code: str) -> bool:
    """剔除指数（SH000xxx/SZ399xxx）、基金等，只留 A 股个股。"""
    c = code[-6:]
    if code.startswith("SH") and c.startswith("000"):
        return False  # 上证指数族
    if code.startswith("SZ") and c.startswith("399"):
        return False  # 深证指数族
    return c.startswith(_STOCK_PREFIX)


# ------------------------------------------------------------------ 成分时点
def universe_spans(universe: str, start: str, end: str) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """指数成分区间 {code: [(纳入日, 调出日), ...]}，只含 [start, end] 内有交集的个股。

    qlib 成分文件含历史成分（含已退市股票），同一股票可能多次进出指数。
    """
    from qlib.data import D

    from .market import init_qlib

    init_qlib()
    spans = D.list_instruments(D.instruments(universe), start_time=start, end_time=end, as_list=False)
    return {c: [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in v] for c, v in spans.items() if is_stock(c)}


def membership_panel(
    spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    dates: pd.DatetimeIndex,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """date×code 布尔面板：当日是否为指数成分（防"用未来成分选股"的前视偏差）。"""
    codes = list(codes) if codes is not None else sorted(spans)
    dates = pd.DatetimeIndex(dates)
    out = pd.DataFrame(False, index=dates, columns=codes)
    for c in codes:
        col = out[c].to_numpy().copy()
        for s, e in spans.get(c, []):
            col |= (dates >= s) & (dates <= e)
        out[c] = col
    return out
