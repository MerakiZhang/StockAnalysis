"""参数敏感性网格：n_drop × hold_thresh × score_ema × rebalance。

默认只回测**研发区**（backtest.start ~ in_sample_end）：调参只看这段，保留集留给最终一次性评估。
每池只加载一次行情上下文，score_ema 变化时重算信号，其余参数只重跑回测。

用法:
    python -X utf8 scripts/run_sensitivity.py --pool zz1000 --n-drop 1,2 --hold-thresh 1,3,5
    python -X utf8 scripts/run_sensitivity.py --pool hs300 --score-ema 0,5 --rebalance daily,weekly
    python -X utf8 scripts/run_sensitivity.py --pool zz500 --full      # 含保留集（仅最终评估时用）
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from run_backtest import backtest_window  # noqa: E402
from src.config import REPORT_DIR, list_pools, load_global, load_pool  # noqa: E402
from src.data.sync import add_sync_args, data_version, preflight  # noqa: E402


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def run_grid(name: str, args) -> pd.DataFrame:
    from src.backtest.engine import run_backtest
    from src.eval import log_rss
    from src.models.scorers import make_scorer
    from src.selection.signal import DEFAULT_WARMUP_DAYS, build_ctx, build_signal

    base = load_pool(name)
    start, end = backtest_window(base, None, None if args.full else base.backtest["in_sample_end"])
    scorer = make_scorer(base)
    ctx = build_ctx(base, start, end, max(DEFAULT_WARMUP_DAYS, int(getattr(scorer, "warmup_days", 0))))
    log_rss(f"{name} 上下文")

    rows: list[dict] = []
    for ema in ints(args.score_ema):
        pool_ema = load_pool(name, override={"score_ema": ema or None})
        signal = build_signal(pool_ema, start, end, ctx=ctx, scorer=scorer)
        for n_drop, hold, freq in itertools.product(ints(args.n_drop), ints(args.hold_thresh),
                                                     args.rebalance.split(",")):
            strategy = {**base.strategy, "n_drop": n_drop, "hold_thresh": hold}
            pool = load_pool(name, override={"score_ema": ema or None, "strategy": strategy, "rebalance": freq})
            print(f"[{name}] ema={ema} n_drop={n_drop} hold={hold} {freq} ...", flush=True)
            res = run_backtest(signal, pool, start, end)
            if res.get("error"):
                print(f"  ERROR: {res['error']}", flush=True)
                continue
            row = {"score_ema": ema, "n_drop": n_drop, "hold_thresh": hold, "rebalance": freq,
                   **{k: res[k] for k in ("ann", "excess_ann", "mdd", "ir", "turnover", "cost")}}
            rows.append(row)
            print(f"  超额 {row['excess_ann']:.2%} 回撤 {row['mdd']:.2%} IR {row['ir']:.2f} "
                  f"换手 {row['turnover']:.2%} 成本 {row['cost']:.2%}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="all", help="池名，逗号分隔或 all；可选: " + ",".join(list_pools()))
    ap.add_argument("--n-drop", default="1,2")
    ap.add_argument("--hold-thresh", default="1,3,5")
    ap.add_argument("--score-ema", default="0", help="分数 EMA 跨度，0 = 不平滑")
    ap.add_argument("--rebalance", default="daily", help="daily,weekly,monthly")
    ap.add_argument("--full", action="store_true", help="含保留集（默认只跑研发区）")
    ap.add_argument("--tag", default=None)
    add_sync_args(ap)
    args = ap.parse_args()
    preflight(args, load_global().get("data"))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = list_pools() if args.pool == "all" else [p.strip() for p in args.pool.split(",")]
    for name in names:
        df = run_grid(name, args)
        out = REPORT_DIR / f"sensitivity_{name}{'_' + args.tag if args.tag else ''}.md"
        span = "研发区+保留集" if args.full else "研发区"
        lines = [f"# 参数敏感性：{name}（{span}）", "",
                 f"生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 数据版本 {data_version()}", ""]
        lines += [df.to_markdown(index=False, floatfmt=".4f") if not df.empty else "全部失败", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"[{name}] 网格完成: {out}", flush=True)


if __name__ == "__main__":
    main()
