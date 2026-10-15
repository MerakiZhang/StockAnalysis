"""换仓规则（纯函数，选股与回测策略共用，保证同口径）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RebalanceResult:
    holdings: list[str]                 # 新股池（按分数降序）
    buy: list[str]                      # 调入
    sell_exit: list[str]                # 因持有规则调出（分数为 NaN）
    sell_rank: list[str]                # 因排名末位调出
    kept: list[str] = field(default_factory=list)   # 留存


def rank_codes(score: pd.Series, codes=None) -> list[str]:
    """按分数降序、代码升序（同分时结果确定）排序；NaN 排最后。"""
    s = score if codes is None else score.reindex(list(codes))
    df = pd.DataFrame({"s": s.to_numpy(dtype=float), "c": list(s.index)})
    df = df.sort_values(["s", "c"], ascending=[False, True], na_position="last", kind="mergesort")
    return df["c"].tolist()


def rebalance(holdings: dict[str, int], score: pd.Series, entry: pd.Series,
              topk: int, n_drop: int, hold_thresh: int = 1) -> RebalanceResult:
    """TopkDropout 换仓规则（RULES.md 第 1 节第 4 步）。

    holdings: 上期股池 {代码: 已持有交易日数}；score: 当日分数（已套持有规则，NaN=不通过）；
    entry: 当日准入（True=可新买入）。与 qlib TopkDropoutStrategy 的区别：
    ① 持有规则不通过的持仓直接调出，不占 n_drop；② 候选只从准入通过的股票中选；
    ③ 持有不足 hold_thresh 的持仓不计入调出，也不为其买入替补（qlib 原版会多买）。
    """
    score = score.astype(float)
    held = list(holdings)
    sell_exit = [c for c in held if not np.isfinite(score.get(c, np.nan))]
    kept = rank_codes(score, [c for c in held if c not in sell_exit])

    ok = entry.reindex(score.index, fill_value=False).astype(bool) & score.notna() & ~score.index.isin(held)
    n_cand = max(n_drop + topk - len(kept), 0)
    candidates = rank_codes(score[ok])[:n_cand]

    comb = rank_codes(score, kept + candidates)
    bottom = set(comb[-n_drop:]) if n_drop > 0 else set()
    sell_rank = [c for c in kept if c in bottom and holdings[c] >= hold_thresh]

    n_buy = max(len(sell_rank) + topk - len(kept), 0)
    buy = candidates[:n_buy]
    stay = [c for c in kept if c not in sell_rank]
    return RebalanceResult(holdings=rank_codes(score, stay + buy), buy=buy,
                           sell_exit=sell_exit, sell_rank=sell_rank, kept=stay)
