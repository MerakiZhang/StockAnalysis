"""参数敏感性网格：topk × n_drop 对回测指标的影响。

每池只生成一次真实打分（复用 run_backtest 的 real_scores_*，耗时主要在这里），
逐网格单元改 topk/n_drop 重跑 qlib 回测（每单元回测约 10~20 秒）。

用法:
    python -X utf8 scripts/run_sensitivity.py --pools stable --topk 20,30,50 --n-drop 3,5,8
    python -X utf8 scripts/run_sensitivity.py --pools short --topk 5,8,12 --n-drop 2,3,5
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from run_backtest import real_scores_short, real_scores_stable, segment_metrics  # noqa: E402
from src.backtest.engine import run_backtest, scores_on_rebalance_dates, wf_segments  # noqa: E402
from src.config import REPORT_DIR, load_config  # noqa: E402
from src.data.market import init_qlib, latest_trade_date  # noqa: E402
from src.eval import log_rss  # noqa: E402


def run_grid(pool: str, score_daily: pd.DataFrame, data_start: str, cfg: dict,
             topks: list[int], n_drops: list[int], end: str) -> pd.DataFrame:
    pcfg = cfg[pool]
    freq = "monthly" if pool == "stable" else "weekly"
    bt_start = pcfg["backtest"]["start"]
    in_sample_end = pcfg["backtest"]["in_sample_end"]
    scores = scores_on_rebalance_dates(score_daily, freq).loc[data_start:]
    segs = wf_segments(bt_start, in_sample_end, end)

    rows: list[dict] = []
    for topk in topks:
        for n_drop in n_drops:
            if n_drop >= topk:
                continue
            c = copy.deepcopy(cfg)
            c[pool]["topk"] = topk
            c[pool]["n_drop"] = n_drop
            print(f"[{pool}] topk={topk} n_drop={n_drop} 回测中...", flush=True)
            res = run_backtest(scores, pool, bt_start, end, c)
            if res.get("error"):
                print(f"  ERROR: {res['error']}", flush=True)
                continue
            row: dict = {
                "topk": topk,
                "n_drop": n_drop,
                "excess_ann": res["excess_annualized_return"],
                "mdd": res["max_drawdown"],
                "ir": res["information_ratio"],
                "turnover": res["turnover_mean"],
            }
            for tag, s, e in segs:
                m = segment_metrics(res["report"], s, e)
                if m:
                    row[f"{tag}超额"] = m["excess_ann"]
                    row[f"{tag}IR"] = m["ir"]
            rows.append(row)
            log_rss(f"{pool} topk={topk} n_drop={n_drop}")
            print(f"  超额年化 {row['excess_ann']:.2%} 回撤 {row['mdd']:.2%} "
                  f"IR {row['ir']:.2f} 换手 {row['turnover']:.2%}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="stable", help="逗号分隔: stable,short")
    ap.add_argument("--topk", default="20,30,50")
    ap.add_argument("--n-drop", default="3,5,8")
    args = ap.parse_args()
    pools = [p.strip() for p in args.pools.split(",")]
    topks = [int(x) for x in args.topk.split(",")]
    n_drops = [int(x) for x in args.n_drop.split(",")]

    init_qlib()
    cfg = load_config()
    end = str(latest_trade_date().date())
    log_rss("init")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "sensitivity_report.md"
    lines = ["# 参数敏感性报告", "", f"生成时间: {pd.Timestamp.now()}", f"数据截至: {end}", ""]
    for pool in pools:
        gen = real_scores_stable if pool == "stable" else real_scores_short
        print(f"\n===== {pool} 打分生成 =====", flush=True)
        score_daily, data_start = gen(cfg, end)
        log_rss(f"{pool} 打分完成")
        df = run_grid(pool, score_daily, data_start, cfg, topks, n_drops, end)
        del score_daily
        if df.empty:
            lines += [f"## {pool}", "", "全部失败", ""]
            continue
        lines += [f"## {pool}", "", df.to_markdown(index=False, floatfmt=".4f"), ""]
        out.write_text("\n".join(lines), encoding="utf-8")  # 每池完成即落盘
        print(f"[{pool}] 网格完成，已写入 {out}", flush=True)


if __name__ == "__main__":
    main()
