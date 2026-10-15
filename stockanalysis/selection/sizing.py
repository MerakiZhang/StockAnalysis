"""池容规则：池容 = topk（若指定）否则 max(min_topk, ceil(top_pct × 当日准入通过只数))。

改池容只需改 config/pool_defaults.yaml 的 sizing 段（或单个池覆盖），不改代码。
"""
from __future__ import annotations

import math

import pandas as pd


def calc_topk(n_entry: int, top_pct: float, min_topk: int, fixed: int | None = None) -> int:
    if fixed is not None:
        return int(fixed)
    # 先四舍五入到 1e-9 再取整，避免 0.01 × 700 = 7.000000000000001 被 ceil 成 8
    return max(int(min_topk), math.ceil(round(top_pct * n_entry, 9)))


def topk_series(entry: pd.DataFrame, sizing) -> pd.Series:
    """逐日池容。entry：准入布尔面板；sizing：config.schema.Sizing。"""
    return entry.sum(axis=1).map(lambda n: calc_topk(int(n), sizing.top_pct, sizing.min_topk, sizing.topk))
