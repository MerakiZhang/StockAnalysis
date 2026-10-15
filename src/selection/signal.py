"""信号层：打分 + 过滤 → Signal（选股与回测共用）。

    Signal.score  分数宽表，已套 exit（持有）规则：不通过 → NaN → 已持有的被调出
    Signal.entry  entry（准入）规则交集：只用于挑选新买入候选
    Signal.topk   逐日池容 calc_topk(当日准入通过只数)

选股取最后一行；回测用整张面板。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.market import init_qlib, latest_trade_date
from ..factors.base import CTX_FIELDS, market_ctx
from ..models.scorers import make_scorer
from .filters import FILTERS, scope_of
from .rebalance import calc_topk
from .universe import current_st_codes, membership_panel, universe_spans

DEFAULT_WARMUP_DAYS = 60  # 自然日：覆盖 20 日滚动类规则的预热


@dataclass
class Signal:
    score: pd.DataFrame
    entry: pd.DataFrame
    topk: pd.Series

    def on(self, date) -> tuple[pd.Series, pd.Series, int]:
        """某交易日截面：(分数, 准入, 池容)。"""
        d = pd.Timestamp(date)
        return self.score.loc[d], self.entry.loc[d], int(self.topk.loc[d])


def build_ctx(pool, start: str, end: str, warmup_days: int = DEFAULT_WARMUP_DAYS,
              fields: tuple[str, ...] = CTX_FIELDS) -> dict:
    """加载 [start - warmup, end] 的行情宽表与成分面板，行对齐 qlib 交易日历。

    codes = 区间内曾为指数成分的个股（只作为加载范围；当日资格由 membership 决定）。
    """
    from qlib.data import D

    init_qlib()
    load_start = str((pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).date())
    spans = universe_spans(pool.universe, start, end)
    codes = sorted(spans)
    dates = pd.DatetimeIndex(D.calendar(start_time=load_start, end_time=end))
    ctx = {k: v.reindex(index=dates, columns=codes) for k, v in market_ctx(codes, load_start, end, fields).items()
           if k != "wide"}
    ctx.update(codes=codes, dates=dates, start=pd.Timestamp(start), end=pd.Timestamp(end),
               membership=membership_panel(spans, dates, codes), st_codes=set())
    # ST 只有当前快照：仅当区间末日就是数据最新交易日（实盘选股）时才使用
    if "exclude_st" in pool.filters and dates[-1] >= latest_trade_date():
        ctx["st_codes"] = current_st_codes()
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


def build_signal(pool, start: str, end: str, ctx: dict | None = None, scorer=None) -> Signal:
    scorer = scorer or make_scorer(pool)
    if ctx is None:
        warmup = max(DEFAULT_WARMUP_DAYS, int(getattr(scorer, "warmup_days", 0)))
        ctx = build_ctx(pool, start, end, warmup)
    raw = scorer.predict(pool, ctx).reindex(index=ctx["dates"], columns=ctx["codes"])
    if pool.score_ema:
        raw = raw.ewm(span=pool.score_ema, min_periods=1).mean()

    exit_ok, entry_ok = combine_masks(pool, ctx)
    score = raw.where(exit_ok)
    entry = entry_ok & score.notna()

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    score, entry = score.loc[s:e], entry.loc[s:e]
    n_entry = entry.sum(axis=1)
    topk = n_entry.map(lambda n: calc_topk(int(n), pool.top_pct, pool.min_topk, pool.topk))
    return Signal(score=score, entry=entry, topk=topk)
