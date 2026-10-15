"""回测入口（池无关：打分器 / 过滤规则 / 换仓规则 / 成本全部取自池 yaml）。

用法:
    python -X utf8 scripts/run_backtest.py --pool hs300
    python -X utf8 scripts/run_backtest.py --pool all --scorer momentum          # 冒烟
    python -X utf8 scripts/run_backtest.py --pool zz1000 --scorer random --start 2020-01-01 --end 2021-06-30 --tag smoke

报告：data/reports/backtest_{pool}[_{tag}].md，头部记录数据版本与配置 hash。
默认只回测**研发区**（backtest.start ~ in_sample_end，调参/选模型只看这段）；
保留集（in_sample_end 之后）须加 --show-holdout，留给参数冻结后的一次性最终评估。

内存：默认 QLIB_KERNELS=2（32G 机器实测安全）；紧张时 QLIB_KERNELS=1。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")  # 32G 内存机器默认双进程，可用环境变量覆盖

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import REPORT_DIR, list_pools, load_global, load_pool
from src.data.sync import add_sync_args, data_version, preflight

METRIC_HEADER = ["| 区间 | 年化 | 超额年化 | 超额回撤 | IR | 日胜率 | 日均换手 | 年化成本 |",
                 "|------|------|----------|----------|-----|--------|---------|---------|"]


def backtest_window(pool, start: str | None = None, end: str | None = None,
                    show_holdout: bool = False) -> tuple[str, str]:
    """回测区间。默认截止 in_sample_end（只看研发区）；show_holdout=True 才延伸到保留集。"""
    from src.data.market import latest_trade_date

    s = start or pool.backtest.get("start", "2018-01-01")
    e = end or pool.backtest.get("end") or str(latest_trade_date().date())
    ins = pool.backtest.get("in_sample_end")
    if ins and not show_holdout and pd.Timestamp(e) > pd.Timestamp(ins):
        if end:
            print(f"[保留集保护] --end {end} 晚于研发区截止 {ins}，已截断；查看保留集需加 --show-holdout")
        e = ins
    return str(s), str(e)


def metric_row(tag: str, m: dict) -> str:
    return (f"| {tag} | {m['ann']:.2%} | {m['excess_ann']:.2%} | {m['mdd']:.2%} | {m['ir']:.2f} "
            f"| {m['win_rate']:.2%} | {m['turnover']:.2%} | {m['cost']:.2%} |")


def report_lines(pool, res: dict, start: str, end: str, title: str | None = None) -> list[str]:
    from src.backtest.engine import segment_metrics, segments

    lines = [f"## {title or pool.name}", ""]
    if res.get("error"):
        return lines + [f"ERROR: {res['error']}", ""]
    lines += [f"- 状态 {pool.status} · 基准 {pool.benchmark} · 打分器 {pool.scorer} · 策略 {pool.strategy} · 调仓 {pool.rebalance}"
              f" · 资金 {pool.account:,.0f} · 配置 {pool.config_hash}", ""]
    lines += METRIC_HEADER
    for tag, s, e in segments(start, pool.backtest.get("in_sample_end", end), end):
        m = segment_metrics(res["report"], s, e)
        if m:
            lines.append(metric_row(tag, m))
    return lines + [""]


def run_pool(pool, start: str, end: str) -> dict:
    from src.backtest.engine import run_backtest
    from src.eval import log_rss
    from src.selection.signal import build_signal

    print(f"\n===== {pool.name}：scorer={pool.scorer['name']} 回测 {start} ~ {end} =====", flush=True)
    signal = build_signal(pool, start, end)
    log_rss(f"{pool.name} 信号完成")
    print(f"池容分布 {signal.topk.value_counts().sort_index().to_dict()}", flush=True)
    res = run_backtest(signal, pool, start, end)
    log_rss(f"{pool.name} 回测完成")
    if not res.get("error"):
        print(f"超额年化 {res['excess_ann']:.2%} 超额回撤 {res['mdd']:.2%} IR {res['ir']:.2f} "
              f"日均换手 {res['turnover']:.2%} 年化成本 {res['cost']:.2%}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="all", help="池名，逗号分隔或 all；可选: " + ",".join(list_pools()))
    ap.add_argument("--scorer", default=None, help="覆盖池 yaml 的打分器名（如 momentum / random）")
    ap.add_argument("--start", default=None, help="覆盖回测起始日")
    ap.add_argument("--end", default=None, help="覆盖回测结束日（默认数据最新交易日）")
    ap.add_argument("--tag", default=None, help="报告文件名后缀（A/B 实验用，避免覆盖基线报告）")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="点路径覆盖池配置，可重复，如 --set model.lgbm.learning_rate=0.05")
    ap.add_argument("--show-holdout", action="store_true",
                    help="回测延伸到保留集（in_sample_end 之后）。仅用于参数冻结后的最终评估")
    add_sync_args(ap)
    args = ap.parse_args()
    preflight(args, load_global().get("data"))  # 必须早于 init_qlib

    names = list_pools() if args.pool == "all" else [p.strip() for p in args.pool.split(",")]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        override = {"scorer": {"name": args.scorer}} if args.scorer else None
        pool = load_pool(name, override=override, sets=args.set)
        start, end = backtest_window(pool, args.start, args.end, args.show_holdout)
        res = run_pool(pool, start, end)
        lines = [f"# 回测报告：{name}", "", f"生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M} · "
                 f"数据版本 {data_version()} · 数据截至 {end}", ""]
        lines += report_lines(pool, res, start, end)
        out = REPORT_DIR / f"backtest_{name}{'_' + args.tag if args.tag else ''}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"报告: {out}")


if __name__ == "__main__":
    main()
