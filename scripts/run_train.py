"""滚动训练（walk-forward）：qlib LGBModel / LinearModel + Alpha158。

用法:
    python -X utf8 scripts/run_train.py --pool hs300
    python -X utf8 scripts/run_train.py --pool all --kinds lgbm,linear

产物见 src/models/rolling.py；训练报告 data/reports/train_{pool}.md（逐窗口 / 逐年 RankIC、特征重要性）。
训练完成后用 run_backtest.py --pool X --scorer lgbm 回测。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "4")  # 特征计算并行；内存紧张时设为 2 或 1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import REPORT_DIR, list_pools, load_global, load_pool
from src.data.sync import add_sync_args, data_version, preflight


def report(pool, manifests: dict, show_holdout: bool = False) -> list[str]:
    """默认只列研发区（测试期起点 ≤ in_sample_end）的窗口与 IC；保留集须 --show-holdout。"""
    ins = pd.Timestamp(pool.backtest.get("in_sample_end", "2022-12-31"))
    span = "研发区+保留集" if show_holdout else f"研发区（≤ {ins.date()}）"
    lines = [f"# 训练报告：{pool.name}（{span}）", "",
             f"生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 数据版本 {data_version()} · "
             f"universe {pool.universe} · 标签 {pool.model.get('label_horizon', 5)} 日", ""]
    for kind, m in manifests.items():
        ic = m["ic"] if show_holdout else m["ic_dev"]
        lines += [f"## {kind}", "", f"- 训练签名 {m['signature']} · 参数 {m['params']}",
                  f"- IC {ic['ic']:.4f} / RankIC {ic['rank_ic']:.4f} / RankICIR {ic.get('rank_ic_ir', float('nan')):.2f}",
                  "", "| 窗口 | train | valid | test | RankIC | RankICIR |", "|---|---|---|---|---|---|"]
        for w in m["windows"]:
            if not show_holdout and pd.Timestamp(w["test"][0]) > ins:
                continue
            lines.append(f"| {w['window']} | {w['train'][0]}~{w['train'][1]} | {w['valid'][0]}~{w['valid'][1]} | "
                         f"{w['test'][0]}~{w['test'][1]} | {w['rank_ic']:.4f} | {w['rank_ic_ir']:.2f} |")
        if "feature_importance_top30" in m:  # 特征重要性不含业绩信息，照常列出
            top = list(m["feature_importance_top30"].items())[:15]
            lines += ["", "最近窗口特征重要性 Top15：" + "、".join(f"{k}({v:.0f})" for k, v in top)]
        lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="all", help="池名，逗号分隔或 all；可选: " + ",".join(list_pools()))
    ap.add_argument("--kinds", default="lgbm,linear")
    ap.add_argument("--first-test", default=None, help="首个测试期起点（默认池 backtest.start）")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="点路径覆盖池配置，如 --set model.lgbm.learning_rate=0.05（模型按参数签名分目录保存）")
    ap.add_argument("--show-holdout", action="store_true", help="报告中列出保留集窗口与 IC")
    add_sync_args(ap)
    args = ap.parse_args()
    preflight(args, load_global().get("data"))

    from src.eval import log_rss
    from src.models.rolling import train_rolling

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = list_pools() if args.pool == "all" else [p.strip() for p in args.pool.split(",")]
    for name in names:
        pool = load_pool(name, sets=args.set)
        manifests = train_rolling(pool, tuple(args.kinds.split(",")), args.first_test,
                                  log=lambda s: print(s, flush=True))
        log_rss(f"{name} 训练完成")
        out = REPORT_DIR / f"train_{name}.md"
        out.write_text("\n".join(report(pool, manifests, args.show_holdout)), encoding="utf-8")
        print(f"报告: {out}", flush=True)


if __name__ == "__main__":
    main()
