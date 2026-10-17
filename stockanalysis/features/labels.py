"""前向收益（唯一定义处）。

T 日收盘出信号、T+1 收盘成交 → h 日前向收益 = T+1 收盘到 T+1+h 收盘的收益：close.shift(-(h+1)) / close.shift(-1) - 1。
事件研究、烂股体检、RankIC 都用这一口径。前向收益仅限研究，禁止进入选股信号。
"""
from __future__ import annotations

import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close.shift(-(horizon + 1)) / close.shift(-1) - 1
