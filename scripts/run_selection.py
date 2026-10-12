"""生成当期股池。

用法：
    python scripts/run_selection.py --pool stable [--date 2026-10-09]
    python scripts/run_selection.py --pool short  [--date 2026-10-09]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.market import latest_trade_date
from src.selection import pool as pool_mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["stable", "short"], required=True)
    ap.add_argument("--date", default=None, help="默认最近交易日")
    args = ap.parse_args()

    cfg = load_config()
    date = pd.Timestamp(args.date) if args.date else latest_trade_date()
    print(f"生成 {args.pool} 池 @ {date.date()}", flush=True)

    names = pool_mod._name_map()
    if args.pool == "stable":
        score = pool_mod.stable_scores(date, cfg)
    else:
        score = pool_mod.short_scores(date, cfg)
    print(f"打分覆盖 {len(score)} 只，取前 {cfg[args.pool]['topk']} 只", flush=True)

    cur = pool_mod.build_pool(score, cfg[args.pool]["topk"], names)
    prev = pool_mod.latest_pool(args.pool)
    diff = pool_mod.diff_pool(prev, cur)
    md = pool_mod.save_pool(cur, args.pool, date, diff, names)

    print(f"新进 {len(diff['new'])} / 剔除 {len(diff['out'])} / 留存 {len(diff['kept'])}")
    print(f"简报: {md}")
    print(cur.to_string(index=False))


if __name__ == "__main__":
    main()
