"""因子层公共工具：宽表行情加载与行情上下文。

约定：宽表 DataFrame（index=交易日，columns=qlib 代码），float32。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.market import load_ohlcv


def load_wide(
    codes: list[str],
    start: str,
    end: str,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount"),
) -> dict[str, pd.DataFrame]:
    """加载行情并透视成 {field: 宽表(date x code)}，float32 节省内存。"""
    df = load_ohlcv(codes, start, end, fields)
    wide: dict[str, pd.DataFrame] = {}
    for f in fields:
        if f in df.columns:
            wide[f] = (
                df[f]
                .unstack(level="instrument")
                .astype(np.float32)
                .sort_index()
            )
    return wide


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


CTX_FIELDS = ("open", "high", "low", "close", "volume", "amount")


def market_ctx(
    codes: list[str],
    start: str,
    end: str,
    fields: tuple[str, ...] = CTX_FIELDS,
) -> dict[str, pd.DataFrame]:
    """行情上下文：{field: 宽表} + "wide" 键（原始宽表字典），供过滤规则与打分器共用、只加载一次。"""
    wide = load_wide(codes, start, end, fields)
    ctx = dict(wide)
    ctx["wide"] = wide
    return ctx
