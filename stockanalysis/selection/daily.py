"""每日选股流程：确定信号日 → 打分与过滤 → 读上期股池 → rebalance()（非调仓日沿用上期）→ 写股池与简报。"""
from __future__ import annotations

import pandas as pd

from ..config import load_pool


def signal_date(date: str | None = None) -> pd.Timestamp:
    """不晚于 date 的最近交易日（默认数据最新交易日）。"""
    from ..data.market import latest_trade_date, trade_calendar

    if date is None:
        return latest_trade_date()
    cal = trade_calendar(end=date)
    if not len(cal):
        raise ValueError(f"{date} 之前无交易日数据")
    return cal[-1]


def run_selection(name: str, date: str | None = None, init: bool = False,
                  names: dict[str, str] | None = None, health: bool = True, sets: list[str] | None = None):
    from ..data.market import trade_calendar
    from ..data.sync import data_version
    from .schedule import is_signal_day
    from .selector import select
    from .signal import build_signal
    from .store import load_prev, save_pool

    pool = load_pool(name, sets=sets)
    names = names or {}
    t = signal_date(date)
    if pool.points is not None:   # 买卖点池：模拟器推演（selection/points_daily.py）
        from .points_daily import run_points_selection

        print(f"\n===== {name} 买卖点池 @ {t.date()}（status={pool.status}）=====", flush=True)
        return run_points_selection(pool, t, names, health=health)
    print(f"\n===== {name} 股池 @ {t.date()}（scorer={pool.scorer['name']}，status={pool.status}）=====", flush=True)
    if pool.status != "production":
        print(f"[注意] {name} 为 observe 池：仅观察跟踪，不建议据此交易", flush=True)
    signal = build_signal(pool, str(t.date()), str(t.date()))
    prev, prev_date = (None, None) if init else load_prev(name, t)
    sel = select(pool, signal, t, prev, trade_calendar(end=str(t.date())), names,
                 rebalance_today=prev is None or is_signal_day(t, pool.rebalance.freq))
    meta = {"data_version": data_version(), "prev_date": None if prev_date is None else prev_date.date(),
            "lag_days": (pd.Timestamp.now().normalize() - t).days}
    csv_path, md_path = save_pool(pool, sel, names, meta)

    r = sel.result
    print(f"池容 {sel.topk}（准入 {sel.n_entry} / 有分 {sel.n_scored}），上期 {meta['prev_date'] or '无'}"
          + ("" if sel.rebalanced else f"；非 {pool.rebalance.freq} 信号日，沿用上期股池"))
    print(f"新进 {r.buy} / 调出(持有规则) {r.sell_exit} / 调出(排名) {r.sell_rank} / 留存 {len(r.kept)}")
    print(sel.table.to_string(index=False))
    print(f"股池: {csv_path}\n简报: {md_path}")
    return sel
