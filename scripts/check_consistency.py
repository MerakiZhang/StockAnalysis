"""同口径验证：选股逐日回放 vs 回测持仓。

从空仓出发，对 [start, start + N 个交易日] 每个信号日 T：
    选股：select()（与 run_selection 同一函数）→ 计划在 T+1 持有的股票
    回测：同一 Signal 跑 qlib 回测 → T+1 收盘后的实际持仓
逐日比对。差异按原因归类：停牌 / 涨停买不进 / 跌停卖不出 属于成交约束（预期内），
其余记为"逻辑差异"（应为 0）。出现差异后以回测持仓为准继续回放，以便检查后续每一天。

用法:
    python -X utf8 scripts/check_consistency.py --pool all --start 2024-01-02 --days 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import REPORT_DIR, list_pools, load_pool


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


def check_pool(name: str, start: str, days: int, scorer: str | None) -> tuple[list[str], int]:
    from qlib.data import D

    from src.backtest.engine import holdings_by_date, run_backtest
    from src.data.market import init_qlib
    from src.selection.pool import select
    from src.selection.signal import build_ctx, build_signal

    init_qlib()
    pool = load_pool(name, override={"scorer": {"name": scorer}} if scorer else None)
    cal = pd.DatetimeIndex(D.calendar(start_time=start))
    sig_days, end = cal[:days], cal[days]
    ctx = build_ctx(pool, str(sig_days[0].date()), str(end.date()))
    signal = build_signal(pool, str(sig_days[0].date()), str(end.date()), ctx=ctx)
    res = run_backtest(signal, pool, str(sig_days[0].date()), str(end.date()))
    bt = holdings_by_date(res["positions"])
    full_cal = pd.DatetimeIndex(D.calendar(end_time=str(end.date())))

    lines = [f"## {name}（scorer={pool.scorer['name']}，信号日 {sig_days[0].date()} ~ {sig_days[-1].date()}）", "",
             "| 信号日 | 选股 | 回测(T+1收盘) | 差异 |", "|--------|------|--------------|------|"]
    prev: pd.DataFrame | None = None
    n_logic = 0
    for t, d in zip(sig_days, cal[1:days + 1]):
        sel = select(pool, signal, t, prev, full_cal)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="all")
    ap.add_argument("--start", default="2024-01-02")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--scorer", default=None, help="覆盖打分器（如 momentum）")
    args = ap.parse_args()

    names = list_pools() if args.pool == "all" else args.pool.split(",")
    lines = ["# 选股 / 回测同口径检查", ""]
    total = 0
    for name in names:
        ls, n = check_pool(name, args.start, args.days, args.scorer)
        lines += ls
        total += n
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "consistency_check.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告: {out}；逻辑差异合计 {total}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
