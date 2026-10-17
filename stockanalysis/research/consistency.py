"""防未来检查（前缀一致性）。

每日选股与回测是同一个 simulate()，按构造一致；这里检查"不用未来数据"：
数据截止到 T 的模拟（= 实盘当日选股）与数据延续到更晚、再把面板截到 T 的模拟，在 T 日的持仓与次日计划必须相同。
"""
from __future__ import annotations

import numpy as np

from ..config import load_pool


def check_pool(name: str, start: str, days: int, scorer: str | None = None,
               sets: list[str] | None = None) -> tuple[list[str], int]:
    pool = load_pool(name, override={"scorer": {"name": scorer}} if scorer else None, sets=sets)
    return check_points_pool(pool, start, days)


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
