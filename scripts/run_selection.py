"""生成当期股池（按池 yaml 的打分器 / 过滤规则 / 换仓规则，与回测同一实现）。

用法：
    python -X utf8 scripts/run_selection.py --pool hs300
    python -X utf8 scripts/run_selection.py --pool zz1000 --date 2026-10-09
    python -X utf8 scripts/run_selection.py --pool all

流程：数据版本检查 → 信号日 T（默认数据最新交易日）打分与过滤 →
读 T 之前最近一期股池 → rebalance() → data/pool/{pool}/{T}.csv 与简报。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import list_pools, load_global, load_pool
from src.data.sync import add_sync_args, data_version, preflight


def signal_date(date: str | None) -> pd.Timestamp:
    """不晚于 date 的最近交易日（默认数据最新交易日）。"""
    from qlib.data import D

    from src.data.market import init_qlib, latest_trade_date

    init_qlib()
    if date is None:
        return latest_trade_date()
    cal = D.calendar(end_time=date)
    if not len(cal):
        raise ValueError(f"{date} 之前无交易日数据")
    return pd.Timestamp(cal[-1])


def run_pool(name: str, date: str | None, init: bool, names: dict[str, str]) -> None:
    from qlib.data import D

    from src.selection.pool import load_prev, save_pool, select
    from src.selection.signal import build_signal

    pool = load_pool(name)
    t = signal_date(date)
    print(f"\n===== {name} 股池 @ {t.date()}（scorer={pool.scorer['name']}，status={pool.status}）=====", flush=True)
    if pool.status != "production":
        print(f"[注意] {name} 为 observe 池：仅观察跟踪，不建议据此交易", flush=True)
    signal = build_signal(pool, str(t.date()), str(t.date()))
    prev, prev_date = (None, None) if init else load_prev(name, t)
    calendar = pd.DatetimeIndex(D.calendar(end_time=str(t.date())))
    sel = select(pool, signal, t, prev, calendar, names)
    meta = {"data_version": data_version(), "prev_date": None if prev_date is None else prev_date.date(),
            "lag_days": (pd.Timestamp.now().normalize() - t).days}
    csv_path, md_path = save_pool(pool, sel, names, meta)

    r = sel.result
    print(f"池容 {sel.topk}（准入 {sel.n_entry} / 有分 {sel.n_scored}），上期 {meta['prev_date'] or '无'}")
    print(f"新进 {r.buy} / 调出(持有规则) {r.sell_exit} / 调出(排名) {r.sell_rank} / 留存 {len(r.kept)}")
    print(sel.table.to_string(index=False))
    print(f"股池: {csv_path}\n简报: {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list_pools() + ["all"])
    ap.add_argument("--date", default=None, help="信号日，默认数据最新交易日")
    ap.add_argument("--init", action="store_true", help="忽略上期股池，按纯排名重新建仓")
    add_sync_args(ap)
    args = ap.parse_args()
    preflight(args, load_global().get("data"))  # 必须早于 init_qlib

    from src.selection.pool import name_map

    names = name_map()
    for name in (list_pools() if args.pool == "all" else [args.pool]):
        run_pool(name, args.date, args.init, names)


if __name__ == "__main__":
    main()
