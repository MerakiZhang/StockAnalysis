"""因子层公共工具：宽表行情加载。

约定：因子宽表 DataFrame（index=交易日，columns=qlib 代码）。
technical.py 的函数输出原始方向（如 mom_n 为正向动量），
"越大越好"的方向统一在注册表（registry.py FACTORS）中完成。
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
