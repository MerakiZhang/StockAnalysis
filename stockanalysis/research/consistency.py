"""同口径验证：选股逐日回放 vs 回测持仓。

从空仓出发，对连续 N 个交易日 T（非调仓日按 schedule.py 沿用上期，与回测不交易一致）：
    选股：selector.select()（与 select 命令同一函数）→ 计划在 T+1 持有的股票
    回测：同一 Signal 跑 qlib 回测 → T+1 收盘后的实际持仓
逐日比对。停牌 / 涨停买不进 / 跌停卖不出 属于成交约束（预期内），其余记为"逻辑差异"（应为 0）。
出现差异后以回测持仓为准继续回放，以便检查后续每一天。

买卖点池（strategy = signal_points）的选股与回测是同一个 simulate()，按构造一致；这里改为检查"不用未来数据"：
数据截止到 T 的模拟（= 实盘当日选股）与数据延续到更晚的模拟，在 T 日的持仓与次日计划必须相同。
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
    if pool.points is not None:
        return check_points_pool(pool, start, days)
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


def _head(ps, n: int):
    """PointSignal 的前 n 个交易日（面板按行截断，不重新计算）。"""
    from dataclasses import fields

    from ..selection.points import PointSignal

    def cut(x):
        if isinstance(x, np.ndarray):
            return x[:n]
        if isinstance(x, dict):
            return {k: cut(v) for k, v in x.items()}
        return x

    kw = {f.name: cut(getattr(ps, f.name)) for f in fields(PointSignal) if f.init}
    kw["dates"] = ps.dates[:n]
    return PointSignal(**kw)


def _state(sim) -> tuple[set, set, set]:
    return set(sim.holdings[-1]), {c for c, _ in sim.pending_buy}, {c for c, _ in sim.pending_sell}

def check_points_pool(pool, start: str, days: int, cuts: int = 3) -> tuple[list[str], int]:
    """前缀一致性：[start, start+days] 的完整模拟 vs 截止到若干个中间日 T 的模拟（实盘口径），比较 T 日持仓与次日计划。"""
    from ..data.market import trade_calendar
    from ..selection.points import WARMUP_DAYS, build_point_signal
    from ..selection.positions import simulate
    from ..selection.signal import build_ctx

    cal = trade_calendar(start)
    s0, e0 = str(cal[0].date()), str(cal[days].date())
    full_ps = build_point_signal(pool, s0, e0, build_ctx(pool, s0, e0, WARMUP_DAYS), with_extra=False)
    picks = sorted({int(round(k * (days - 1) / max(cuts - 1, 1))) for k in range(cuts)})
    lines = [f"## {pool.name}（买卖点池：前缀一致性，{s0} ~ {e0}，检查 freq={pool.points.freq}）", "",
             "| T | 持仓数 | 次日计划买 / 卖 | 差异 |", "|---|--------|----------------|------|"]
    n_logic = 0
    for i in picks:
        t = str(full_ps.dates[i].date())
        ps = build_point_signal(pool, s0, t, build_ctx(pool, s0, t, WARMUP_DAYS), with_extra=False)
        a = _state(simulate(ps, pool.points))                      # 数据截止到 T（实盘口径）
        b = _state(simulate(_head(full_ps, i + 1), pool.points))   # 数据延续到 e0 的面板截到 T
        diff = [f"{n}: 截止 T 独有 {sorted(x - y)} / 完整独有 {sorted(y - x)}"
                for n, x, y in zip(("持仓", "计划买", "计划卖"), a, b) if x != y]
        n_logic += len(diff)
        lines.append(f"| {t} | {len(a[0])} | {len(a[1])} / {len(a[2])} | {'；'.join(diff) or '一致'} |")
    lines += ["", f"逻辑差异 {n_logic} 处（截止到 T 的结果依赖了 T 之后的数据即为差异）。", ""]
    print("\n".join(lines), flush=True)
    return lines, n_logic
