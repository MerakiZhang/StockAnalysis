"""选股：上期股池 + 当日信号 → rebalance() → Selection（纯逻辑，不读写文件；文件见 store.py）。

非信号日（weekly / monthly 池，见 schedule.py）沿用上期股池不换仓，与回测在非信号日不交易一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .rebalance import RebalanceResult, rank_codes, rebalance
from .signal import Signal


def holding_days(prev: pd.DataFrame | None, date: pd.Timestamp, calendar: pd.DatetimeIndex) -> dict[str, int]:
    """{代码: 已持有交易日数}。入池信号日 E、当前信号日 T → 持有 = E 到 T 之间的交易日数，
    与回测中 qlib Position.get_stock_count 的口径一致（T+1 成交、每日收盘计数）。"""
    if prev is None or prev.empty:
        return {}
    pos = {d: i for i, d in enumerate(calendar)}
    t = pos[date]
    return {c: t - pos.get(pd.Timestamp(e), t) for c, e in zip(prev["code"], prev["entry_date"])}


@dataclass
class Selection:
    date: pd.Timestamp
    result: RebalanceResult
    table: pd.DataFrame        # 新股池
    reference: list[str]       # 纯排名前 topk（准入通过）
    topk: int
    n_entry: int
    n_scored: int
    rebalanced: bool = True    # False = 非信号日，沿用上期股池


def select(pool, signal: Signal, date: pd.Timestamp, prev: pd.DataFrame | None,
           calendar: pd.DatetimeIndex, names: dict[str, str] | None = None,
           rebalance_today: bool = True) -> Selection:
    """rebalance_today=False（非信号日）且有上期股池时，原样沿用上期持仓。"""
    names = names or {}
    score, entry, topk = signal.on(date)
    holdings = holding_days(prev, date, calendar)
    carry = not rebalance_today and bool(holdings)
    if carry:
        kept = rank_codes(score, list(holdings))
        res = RebalanceResult(holdings=kept, buy=[], sell_exit=[], sell_rank=[], kept=kept)
    else:
        res = rebalance(holdings, score, entry, topk, pool.rebalance.n_drop, pool.rebalance.hold_thresh)

    ranked = rank_codes(score[score.notna()])
    rank = {c: i + 1 for i, c in enumerate(ranked)}
    prev_entry = {} if prev is None else dict(zip(prev["code"], prev["entry_date"]))
    rows = []
    for c in res.holdings:
        new = c in res.buy
        rows.append({"code": c, "name": names.get(c, ""), "score": float(score.get(c, float("nan"))), "rank": rank.get(c),
                     "entry_date": date if new else pd.Timestamp(prev_entry[c]),
                     "status": "新进" if new else "留存"})
    table = pd.DataFrame(rows, columns=["code", "name", "score", "rank", "entry_date", "status"])
    reference = rank_codes(score[entry & score.notna()])[:topk]
    return Selection(date=date, result=res, table=table, reference=reference, topk=topk,
                     n_entry=int((entry & score.notna()).sum()), n_scored=len(ranked), rebalanced=not carry)
