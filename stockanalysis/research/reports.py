"""Markdown 报告：统一的表头（生成时间 / 数据版本）、回测分段表、买卖点池交易统计、落盘。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REPORT_DIR

METRIC_HEADER = ["| 区间 | 年化 | 超额年化 | 超额回撤 | IR | 日胜率 | 日均换手 | 年化成本 |",
                 "|------|------|----------|----------|-----|--------|---------|---------|"]


def header(title: str, extra: str = "") -> list[str]:
    from ..data.sync import data_version

    line = f"生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 数据版本 {data_version()}"
    return [f"# {title}", "", line + (f" · {extra}" if extra else ""), ""]


def write_report(name: str, lines: list[str], tag: str | None = None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{name}{'_' + tag if tag else ''}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告: {out}", flush=True)
    return out


def metric_row(tag: str, m: dict) -> str:
    return (f"| {tag} | {m['ann']:.2%} | {m['excess_ann']:.2%} | {m['mdd']:.2%} | {m['ir']:.2f} "
            f"| {m['win_rate']:.2%} | {m['turnover']:.2%} | {m['cost']:.2%} |")


def backtest_section(pool, res: dict, start: str, end: str, title: str | None = None) -> list[str]:
    from ..backtest.metrics import segment_metrics, segments

    lines = [f"## {title or pool.name}", ""]
    if res.get("error"):
        return lines + [f"ERROR: {res['error']}", ""]
    lines += [f"- 状态 {pool.status} · 基准 {pool.benchmark} · 优先分 {pool.scorer} "
              f"· 资金 {pool.account:,.0f} · 配置 {pool.config_hash}", ""]
    lines += METRIC_HEADER
    for tag, s, e in segments(start, pool.backtest.get("in_sample_end", end), end):
        m = segment_metrics(res["report"], s, e)
        if m:
            lines.append(metric_row(tag, m))
    return lines + [""]


def points_section(pool, res: dict) -> list[str]:
    """买卖点池补充：交易统计、卖出原因分布、规则摘要。"""
    p = pool.points
    rules = {k: v for k, v in vars(p).items() if k not in ("colmap",) and v not in (None, {}, [], False)}
    lines = ["### 交易统计（扣往返成本）", "",
             f"- 完整交易 {res.get('trades', 0)} 笔 · 胜率 {res.get('win_rate', float('nan')):.1%} · "
             f"平均每笔 {res.get('avg_ret', float('nan')):.2%} · 盈亏比 {res.get('payoff', float('nan')):.2f} · "
             f"平均持有 {res.get('avg_days', float('nan')):.1f} 个交易日",
             f"- 平均持仓 {res.get('avg_pos', float('nan')):.1f} 只（上限 {p.max_positions}）· "
             f"平均仓位 {res.get('exposure', float('nan')):.1%}",
             "- 卖出原因：" + "、".join(f"{k} {v}" for k, v in res.get("sell_reasons", {}).items()), "",
             "### 规则", "", "```", *(f"{k}: {v}" for k, v in rules.items()), "```", ""]
    return lines
