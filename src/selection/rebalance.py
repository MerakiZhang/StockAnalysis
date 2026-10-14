"""池容与换仓规则（纯函数，选股与回测策略共用，保证同口径）。"""
from __future__ import annotations

import math


def calc_topk(n_entry: int, top_pct: float, min_topk: int, fixed: int | None = None) -> int:
    """池容 = max(min_topk, ceil(top_pct × 准入通过只数))；fixed 非空时直接用 fixed。"""
    if fixed is not None:
        return int(fixed)
    # 先四舍五入到 1e-9 再取整，避免 0.01 × 700 = 7.000000000000001 被 ceil 成 8
    return max(int(min_topk), math.ceil(round(top_pct * n_entry, 9)))
