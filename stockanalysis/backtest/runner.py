"""一个池的回测流程：确定区间（默认只到研发区）→ build_signal → run_backtest。"""
from __future__ import annotations

import pandas as pd

from ..runtime import log_rss


def backtest_window(pool, start: str | None = None, end: str | None = None,
                    show_holdout: bool = False) -> tuple[str, str]:
    """回测区间。默认截止 in_sample_end（只看研发区）；show_holdout=True 才延伸到保留集。"""
    from ..data.market import latest_trade_date

    s = start or pool.backtest.get("start", "2018-01-01")
    e = end or pool.backtest.get("end") or str(latest_trade_date().date())
    ins = pool.backtest.get("in_sample_end")
    if ins and not show_holdout and pd.Timestamp(e) > pd.Timestamp(ins):
        if end:
            print(f"[保留集保护] 结束日 {end} 晚于研发区截止 {ins}，已截断；查看保留集需加 --show-holdout")
        e = ins
    return str(s), str(e)


def backtest_pool(pool, start: str, end: str, ctx: dict | None = None) -> dict:
    from ..selection.signal import build_signal
    from .engine import run_backtest

    print(f"\n===== {pool.name}：scorer={pool.scorer['name']} 回测 {start} ~ {end} =====", flush=True)
    signal = build_signal(pool, start, end, ctx=ctx)
    log_rss(f"{pool.name} 信号完成")
    print(f"池容分布 {signal.topk.value_counts().sort_index().to_dict()}", flush=True)
    res = run_backtest(signal, pool, start, end)
    log_rss(f"{pool.name} 回测完成")
    if not res.get("error"):
        print(f"超额年化 {res['excess_ann']:.2%} 超额回撤 {res['mdd']:.2%} IR {res['ir']:.2f} "
              f"日均换手 {res['turnover']:.2%} 年化成本 {res['cost']:.2%}", flush=True)
    return res
