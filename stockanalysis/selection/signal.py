"""行情上下文 + 过滤面板（选股与回测共用）。

    build_ctx      加载 [start - warmup, end] 的行情宽表、在市面板
    combine_masks  按池 yaml 启用的过滤规则求 exit（持有）/ entry（准入）交集
"""
from __future__ import annotations

import pandas as pd

from ..data.market import CTX_FIELDS, market_ctx, trade_calendar
from ..data.universe import membership_panel, universe_spans
from .filters import FILTERS, scope_of

DEFAULT_WARMUP_DAYS = 60  # 自然日：覆盖 20 日滚动类规则的预热


def build_ctx(pool, start: str, end: str, warmup_days: int = DEFAULT_WARMUP_DAYS,
              fields: tuple[str, ...] = CTX_FIELDS) -> dict:
    """加载 [start - warmup, end] 的行情宽表与成分面板，行对齐 qlib 交易日历。

    codes = 区间内曾在市的个股（只作为加载范围；当日资格由 membership 决定）。
    """
    load_start = str((pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).date())
    spans = universe_spans(pool.universe, start, end)
    codes = sorted(spans)
    dates = trade_calendar(load_start, end)
    ctx = {k: v.reindex(index=dates, columns=codes) for k, v in market_ctx(codes, load_start, end, fields).items()}
    ctx.update(codes=codes, dates=dates, start=pd.Timestamp(start), end=pd.Timestamp(end),
               membership=membership_panel(spans, dates, codes))
    return ctx


def combine_masks(pool, ctx: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按池 yaml 启用的规则分别求 exit / entry 交集。"""
    base = ctx["close"]
    masks = {"exit": pd.DataFrame(True, index=base.index, columns=base.columns),
             "entry": pd.DataFrame(True, index=base.index, columns=base.columns)}
    for name, params in pool.filters.items():
        m = FILTERS.get(name)(ctx, **params)
        masks[scope_of(name)] &= m.reindex_like(base).fillna(False).astype(bool)
    return masks["exit"], masks["entry"]

