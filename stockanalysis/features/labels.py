"""标签与前向收益（唯一定义处）。

T 日收盘打分、T+1 收盘成交 → h 日标签 = T+1 收盘到 T+1+h 收盘的收益：
    qlib 表达式：Ref($close, -(h+1)) / Ref($close, -1) - 1
    宽表计算：  close.shift(-(h+1)) / close.shift(-1) - 1
训练标签、研发区评估（RankIC / TopK）、特征体检都用这一口径。前向收益仅限研究，禁止进入选股信号。
"""
from __future__ import annotations

import pandas as pd


def label_expr(horizon: int) -> str:
    return f"Ref($close, -{horizon + 1}) / Ref($close, -1) - 1"


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close.shift(-(horizon + 1)) / close.shift(-1) - 1
