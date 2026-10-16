"""买卖点信号：按池 yaml 的 points 段把条件库（features/conditions.py）组装成逐日面板，供 positions.simulate() 使用。

    buy        买点：buy 组全部满足 ∧ buy_any 组至少一个 ∧ 不命中 avoid 组 ∧ 通过全部过滤规则（exit ∧ entry）
    sell_hits  卖点（静态条件）：{条件名: 面板}，任一为 True 即卖
    exit_ok    持有规则（烂股剔除等 exit 过滤）：不通过的持仓强制卖出
    eligible   可新买入（exit ∧ entry 过滤都通过）
    score      优先分（池 yaml 的 scorer，通常为 multifactor）；score_pct 为在可买股票中的分位
    can_buy / can_sell   成交可行性：当日有成交且未涨停（买）/ 未跌停（卖）——用于 T+1 执行日
    regime_ok  择时：当日可开新仓
    signal_day 新买入与条件卖出的检查日（points.freq）

所有面板只用当日及之前的数据（条件库与过滤规则各自保证）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WARMUP_DAYS = 420  # 自然日：250 日均线、年度动量、一年内风险公告


@dataclass
class PointSignal:
    dates: pd.DatetimeIndex
    codes: list[str]
    close: np.ndarray            # 复权收盘价（交易日 × 代码），停牌为 NaN
    buy: np.ndarray              # bool
    buy_hits: dict[str, np.ndarray]    # buy_any 各条件（记录买入原因）
    sell_hits: dict[str, np.ndarray]
    exit_ok: np.ndarray
    eligible: np.ndarray
    score: np.ndarray
    score_pct: np.ndarray
    atr: np.ndarray
    can_buy: np.ndarray
    can_sell: np.ndarray
    regime_ok: np.ndarray        # (交易日,)
    signal_day: np.ndarray       # (交易日,)
    industry: list[str | None]
    extra: dict = field(default_factory=dict)   # 简报用：均线等卖出线、各买点条件面板
    fill: np.ndarray | None = None   # 成交价（复权）；None = 收盘价。均价成交时为 vwap，缺失处回退收盘价
    colmap: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.colmap = {c: j for j, c in enumerate(self.codes)}
        if self.fill is None:
            self.fill = self.close
        else:
            self.fill = np.where(np.isfinite(self.fill), self.fill, self.close)


def _cond(ctx: dict, name: str, params: dict | None) -> pd.DataFrame:
    from ..features.conditions import CONDITIONS

    out = CONDITIONS.get(name)(ctx, **(params or {}))
    return out.reindex(index=ctx["close"].index, columns=ctx["close"].columns).fillna(False).astype(bool)


def build_point_signal(pool, start: str, end: str, ctx: dict | None = None, with_extra: bool = True) -> PointSignal:
    """with_extra=False：不算简报用的均线 / 条件分解面板（网格研究省内存）。"""
    from ..features import regime
    from ..features import technical as tech
    from ..features.fundamental import industry_map
    from ..models.scorers import make_scorer
    from .filters import limit_hits
    from .schedule import signal_mask
    from .signal import build_ctx, combine_masks

    spec = pool.points
    if spec is None:
        raise ValueError(f"池 {pool.name} 不是买卖点池（strategy.name 须为 signal_points）")
    scorer = make_scorer(pool)
    if ctx is None:
        warmup = max(WARMUP_DAYS, int(getattr(scorer, "warmup_days", 0)))
        ctx = build_ctx(pool, start, end, warmup)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    idx = ctx["close"].loc[s:e].index

    def cut(df: pd.DataFrame) -> np.ndarray:
        return df.reindex(index=ctx["close"].index, columns=ctx["close"].columns).loc[s:e].to_numpy()

    exit_ok, entry_ok = combine_masks(pool, ctx)
    eligible = exit_ok & entry_ok & ctx["close"].notna()

    buy = eligible.copy()
    for name, params in spec.buy.items():
        buy &= _cond(ctx, name, params)
    buy_hits = {name: _cond(ctx, name, params) for name, params in spec.buy_any.items()}
    if buy_hits:
        any_hit = None
        for hit in buy_hits.values():
            any_hit = hit if any_hit is None else (any_hit | hit)
        buy &= any_hit
    for name, params in spec.avoid.items():
        buy &= ~_cond(ctx, name, params)
    sell_hits = {name: cut(_cond(ctx, name, params)) for name, params in spec.sell.items()}

    raw = scorer.predict(pool, ctx).reindex(index=ctx["close"].index, columns=ctx["close"].columns)
    score = raw.where(eligible | exit_ok)
    score_pct = raw.where(eligible).rank(axis=1, pct=True)

    up, down = limit_hits(ctx["close"])
    traded = ctx["close"].notna() & (ctx["volume"].fillna(0) > 0)
    reg = regime.state(ctx, pool.benchmark, **spec.regime).reindex(ctx["close"].index).fillna(True)

    ind = industry_map().reindex(ctx["close"].columns)
    extra = {} if not with_extra else {"ma10": cut(tech.get(ctx, "ma10")), "ma20": cut(tech.get(ctx, "ma20")),
             "ma60": cut(tech.get(ctx, "ma60")), "ma250": cut(tech.get(ctx, "ma250")),
             "factor": cut(ctx["factor"]).astype(np.float64),   # 复权价 / factor = 不复权价（简报显示用）
             "buy_parts": {n: cut(_cond(ctx, n, p)) for n, p in spec.buy.items()},
             "buy_any_parts": {n: cut(h) for n, h in buy_hits.items()},
             "avoid_parts": {n: cut(_cond(ctx, n, p)) for n, p in spec.avoid.items()}}
    return PointSignal(
        dates=pd.DatetimeIndex(idx), codes=list(ctx["close"].columns),
        close=cut(ctx["close"]).astype(np.float64), buy=cut(buy).astype(bool),
        buy_hits={n: cut(p).astype(bool) for n, p in buy_hits.items()}, sell_hits=sell_hits,
        exit_ok=cut(exit_ok).astype(bool), eligible=cut(eligible).astype(bool),
        score=cut(score).astype(np.float64), score_pct=cut(score_pct).astype(np.float64),
        atr=cut(tech.get(ctx, "atr14")).astype(np.float64),
        can_buy=cut(traded & ~up).astype(bool), can_sell=cut(traded & ~down).astype(bool),
        regime_ok=reg.loc[s:e].to_numpy(dtype=bool), signal_day=signal_mask(idx, spec.freq),
        industry=[None if pd.isna(x) else str(x) for x in ind], extra=extra,
        fill=cut(ctx["vwap"]).astype(np.float64) if spec.deal_price == "vwap" else None,
    )


def slice_from(ps: PointSignal, i0: int) -> PointSignal:
    """从第 i0 个交易日起的子信号（多起点稳健性检验：不同建仓日的结果差异）。"""
    def cut(x):
        if isinstance(x, np.ndarray):
            return x[i0:]
        if isinstance(x, dict):
            return {k: cut(v) for k, v in x.items()}
        return x

    sub = PointSignal(
        dates=ps.dates[i0:], codes=ps.codes, close=ps.close[i0:], buy=ps.buy[i0:], buy_hits=cut(ps.buy_hits),
        sell_hits=cut(ps.sell_hits), exit_ok=ps.exit_ok[i0:], eligible=ps.eligible[i0:], score=ps.score[i0:],
        score_pct=ps.score_pct[i0:], atr=ps.atr[i0:], can_buy=ps.can_buy[i0:], can_sell=ps.can_sell[i0:],
        regime_ok=ps.regime_ok[i0:], signal_day=np.r_[True, ps.signal_day[i0 + 1:]],   # 起点当天建仓
        industry=ps.industry, extra=cut(ps.extra))
    if ps.fill is not ps.close:
        sub.fill = ps.fill[i0:]          # 视图，不再经 __post_init__ 复制
    return sub
