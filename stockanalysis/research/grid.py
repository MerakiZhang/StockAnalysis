"""参数敏感性网格：换仓规则（n_drop × hold_thresh × freq）× 分数 EMA，默认只在研发区回测。

每池只加载一次行情上下文，score_ema 变化时重算信号，其余参数只重跑回测。
"""
from __future__ import annotations

import itertools

import pandas as pd

from ..config import load_pool
from ..runtime import log_rss


def run_grid(name: str, n_drops: list[int], holds: list[int], emas: list[int], freqs: list[str],
             sets: list[str] | None = None, show_holdout: bool = False) -> pd.DataFrame:
    from ..backtest.engine import run_backtest
    from ..backtest.runner import backtest_window
    from ..models.scorers import make_scorer
    from ..selection.signal import DEFAULT_WARMUP_DAYS, build_ctx, build_signal

    sets = list(sets or [])
    base = load_pool(name, sets=sets)
    start, end = backtest_window(base, None, None, show_holdout)
    scorer = make_scorer(base)
    ctx = build_ctx(base, start, end, max(DEFAULT_WARMUP_DAYS, int(getattr(scorer, "warmup_days", 0))))
    log_rss(f"{name} 上下文")

    rows: list[dict] = []
    for ema in emas:
        ema_sets = sets + [f"signal.score_ema={ema or 'null'}"]
        signal = build_signal(load_pool(name, sets=ema_sets), start, end, ctx=ctx, scorer=scorer)
        for n_drop, hold, freq in itertools.product(n_drops, holds, freqs):
            pool = load_pool(name, sets=ema_sets + [f"rebalance.n_drop={n_drop}", f"rebalance.hold_thresh={hold}",
                                                    f"rebalance.freq={freq}"])
            print(f"[{name}] ema={ema} n_drop={n_drop} hold={hold} {freq} ...", flush=True)
            res = run_backtest(signal, pool, start, end)
            if res.get("error"):
                print(f"  ERROR: {res['error']}", flush=True)
                continue
            row = {"score_ema": ema, "n_drop": n_drop, "hold_thresh": hold, "freq": freq,
                   **{k: res[k] for k in ("ann", "excess_ann", "mdd", "ir", "turnover", "cost")}}
            rows.append(row)
            print(f"  超额 {row['excess_ann']:.2%} 回撤 {row['mdd']:.2%} IR {row['ir']:.2f} "
                  f"换手 {row['turnover']:.2%} 成本 {row['cost']:.2%}", flush=True)
    return pd.DataFrame(rows)
