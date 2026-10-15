"""股池文件读写：data/pool/{pool}/{YYYY-MM-DD}.csv 与 {YYYY-MM-DD}_brief.md。

日期为**信号日**（数据最新交易日 T），股池供 T 之后首个交易日执行。
CSV 列：code, name, score, rank, entry_date（入池信号日，用于计算持有天数）, status。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import POOL_OUTPUT_DIR
from .selector import Selection


def pool_dir(name: str) -> Path:
    return POOL_OUTPUT_DIR / name


def load_prev(name: str, before: pd.Timestamp) -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    """信号日 before 之前（不含）最近一期股池。同日重跑时以前一期为基础覆盖本期。"""
    files = [f for f in sorted(pool_dir(name).glob("????-??-??.csv")) if pd.Timestamp(f.stem) < before]
    if not files:
        return None, None
    df = pd.read_csv(files[-1], dtype={"code": str}, parse_dates=["entry_date"])
    return df, pd.Timestamp(files[-1].stem)


def save_pool(pool, sel: Selection, names: dict[str, str], meta: dict) -> tuple[Path, Path]:
    """写股池 CSV + 简报 md。meta：data_version / config_hash / scorer / lag_days / prev_date。"""
    d = pool_dir(pool.name)
    d.mkdir(parents=True, exist_ok=True)
    tag = sel.date.strftime("%Y-%m-%d")
    csv_path, md_path = d / f"{tag}.csv", d / f"{tag}_brief.md"
    out = sel.table.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    res = sel.result

    def fmt(codes: list[str]) -> str:
        return "、".join(f"{names.get(c, '')}({c})" for c in codes) or "无"

    shown = out.assign(score=out["score"].round(4))
    status_line = ("> **状态：production**（可作实盘参考）" if pool.status == "production" else
                   "> **状态：observe — 仅观察跟踪，不建议据此交易**（保留集评估未通过，见 RULES.md）")
    lines = [
        f"# {pool.name} 股池 {tag}",
        "",
        status_line,
        "",
        f"> 基于 {tag} 收盘数据，供其后首个交易日参考（数据滞后 {meta.get('lag_days', '?')} 个自然日）。",
        f"> 数据版本 {meta.get('data_version')} · 配置 {pool.config_hash} · 打分器 {pool.scorer['name']}"
        f" · 上期 {meta.get('prev_date') or '无（首次建仓）'}",
        "",
        f"- 池容 topk: {sel.topk}（准入通过 {sel.n_entry} 只 / 当日有分 {sel.n_scored} 只）",
        f"- 换仓规则: {pool.rebalance.freq} n_drop={pool.rebalance.n_drop} hold_thresh={pool.rebalance.hold_thresh}",
        f"- 新进: {fmt(res.buy)}",
        f"- 调出（持有规则不通过）: {fmt(res.sell_exit)}",
        f"- 调出（排名末位）: {fmt(res.sell_rank)}",
        f"- 留存: {len(res.kept)} 只",
        "",
        "调入/调出若因停牌、涨跌停无法成交，则顺延至下一交易日。股池等权。",
        "",
        "## 本期股池",
        "",
        shown.to_markdown(index=False),
        "",
        f"## 参考：准入通过的纯排名前 {sel.topk}",
        "",
        fmt(sel.reference),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
