"""同口径验证：选股逐日回放 vs 回测持仓。

从空仓出发，对连续 N 个交易日 T（非调仓日按 schedule.py 沿用上期，与回测不交易一致）：
    选股：selector.select()（与 select 命令同一函数）→ 计划在 T+1 持有的股票
    回测：同一 Signal 跑 qlib 回测 → T+1 收盘后的实际持仓
逐日比对。停牌 / 涨停买不进 / 跌停卖不出 属于成交约束（预期内），其余记为"逻辑差异"（应为 0）。
出现差异后以回测持仓为准继续回放，以便检查后续每一天。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_pool


def trade_block_reason(ctx: dict, code: str, day: pd.Timestamp, side: str) -> str | None:
    """day 当日 code 不可成交的原因（None = 可成交）。"""
    vol = ctx["volume"].get(code)
    if vol is None or day not in vol.index or not np.isfinite(vol.loc[day]) or vol.loc[day] == 0:
        return "停牌"
    close = ctx["close"][code]
    i = close.index.get_loc(day)
    ret = close.iloc[i] / close.iloc[i - 1] - 1 if i > 0 else 0.0
    if side == "buy" and ret >= 0.095:
        return "涨停买不进"
    if side == "sell" and ret <= -0.095:
        return "跌停卖不出"
    return None


def check_pool(name: str, start: str, days: int, scorer: str | None = None,
               sets: list[str] | None = None) -> tuple[list[str], int]:
    from ..backtest.engine import holdings_by_date, run_backtest
    from ..data.market import trade_calendar
    from ..selection.schedule import signal_mask
    from ..selection.selector import select
    from ..selection.signal import build_ctx, build_signal

    pool = load_pool(name, override={"scorer": {"name": scorer}} if scorer else None, sets=sets)
    cal = trade_calendar(start)
    sig_days, end = cal[:days], cal[days]
    s0, e0 = str(sig_days[0].date()), str(end.date())
    ctx = build_ctx(pool, s0, e0)
    signal = build_signal(pool, s0, e0, ctx=ctx)
    bt = holdings_by_date(run_backtest(signal, pool, s0, e0)["positions"])
    full_cal = trade_calendar(end=e0)

    on_signal = dict(zip(signal.score.index, signal_mask(signal.score.index, pool.rebalance.freq)))
    lines = [f"## {name}（scorer={pool.scorer['name']}，调仓 {pool.rebalance.freq}，"
             f"信号日 {sig_days[0].date()} ~ {sig_days[-1].date()}）", "",
             "| 信号日 | 选股 | 回测(T+1收盘) | 差异 |", "|--------|------|--------------|------|"]
    prev: pd.DataFrame | None = None
    n_logic = 0
    for t, d in zip(sig_days, cal[1:days + 1]):
        sel = select(pool, signal, t, prev, full_cal, rebalance_today=bool(on_signal.get(t, True)))
        planned, actual = set(sel.table["code"]), set(bt.get(d, []))
        notes = []
        for c in sorted(planned - actual):
            r = trade_block_reason(ctx, c, d, "buy")
            notes.append(f"{c} 未买入（{r or '逻辑差异'}）")
            n_logic += r is None
        for c in sorted(actual - planned):
            r = trade_block_reason(ctx, c, d, "sell")
            notes.append(f"{c} 未卖出（{r or '逻辑差异'}）")
            n_logic += r is None
        lines.append(f"| {t.date()} | {' '.join(sorted(planned))} | {' '.join(sorted(actual))} | "
                     f"{'；'.join(notes) or '一致'} |")
        # 以回测实际持仓为准继续回放（入池日沿用：选股表 > 上期表 > 当日）
        entry = {**({} if prev is None else dict(zip(prev["code"], prev["entry_date"]))),
                 **dict(zip(sel.table["code"], sel.table["entry_date"]))}
        prev = pd.DataFrame({"code": sorted(actual), "entry_date": [entry.get(c, t) for c in sorted(actual)]})
    lines += ["", f"逻辑差异 {n_logic} 处。", ""]
    print("\n".join(lines), flush=True)
    return lines, n_logic
