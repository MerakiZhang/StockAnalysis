"""选股：上期股池 + 当日信号 → rebalance() → 股池 CSV 与简报。

股池文件按池分目录：data/pool/{pool}/{YYYY-MM-DD}.csv 与 {YYYY-MM-DD}_brief.md，
日期为**信号日**（数据最新交易日 T），股池供 T 之后首个交易日执行。
CSV 列：code, name, score, rank, entry_date（入池信号日，用于计算持有天数）, status。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT
from ..data import fundamental as fd
from .rebalance import RebalanceResult, rank_codes, rebalance
from .signal import Signal

POOL_DIR = PROJECT_ROOT / "data" / "pool"


def pool_dir(name: str) -> Path:
    return POOL_DIR / name


def name_map() -> dict[str, str]:
    """qlib 代码 -> 股票名称（yjbb 历史名称 + 当日快照，快照优先）。"""
    names: dict[str, str] = {}
    try:
        yj = fd.load_yjbb_all().sort_values("period")
        names.update(zip(yj["qlib_code"], yj["name"]))
    except FileNotFoundError:
        pass
    try:
        spot = fd.fetch_spot()
        names.update({fd.normalize_code(c): n for c, n in zip(spot["code"], spot["name"])})
    except Exception:  # noqa: BLE001 - 快照失败时只缺名称，不影响选股
        pass
    return names


def load_prev(name: str, before: pd.Timestamp) -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    """信号日 before 之前（不含）最近一期股池。同日重跑时以前一期为基础覆盖本期。"""
    files = sorted(pool_dir(name).glob("????-??-??.csv"))
    files = [f for f in files if pd.Timestamp(f.stem) < before]
    if not files:
        return None, None
    df = pd.read_csv(files[-1], dtype={"code": str}, parse_dates=["entry_date"])
    return df, pd.Timestamp(files[-1].stem)


def holding_days(prev: pd.DataFrame | None, date: pd.Timestamp, calendar: pd.DatetimeIndex) -> dict[str, int]:
    """{代码: 已持有交易日数}。入池信号日 E、当前信号日 T → 持有 = E 到 T 之间的交易日数，
    与回测中 qlib Position.get_stock_count 的口径一致（T+1 成交、每日收盘计数）。"""
    if prev is None or prev.empty:
        return {}
    pos = {d: i for i, d in enumerate(calendar)}
    t = pos[date]
    return {c: t - pos.get(pd.Timestamp(e), t) for c, e in zip(prev["code"], prev["entry_date"])}


@dataclass
class Selection:
    date: pd.Timestamp
    result: RebalanceResult
    table: pd.DataFrame        # 新股池
    reference: list[str]       # 纯排名前 topk（准入通过）
    topk: int
    n_entry: int
    n_scored: int


def select(pool, signal: Signal, date: pd.Timestamp, prev: pd.DataFrame | None,
           calendar: pd.DatetimeIndex, names: dict[str, str] | None = None) -> Selection:
    names = names or {}
    score, entry, topk = signal.on(date)
    holdings = holding_days(prev, date, calendar)
    res = rebalance(holdings, score, entry, topk, int(pool.strategy_params.get("n_drop", 1)),
                    int(pool.strategy_params.get("hold_thresh", 1)))

    ranked = rank_codes(score[score.notna()])
    rank = {c: i + 1 for i, c in enumerate(ranked)}
    prev_entry = {} if prev is None else dict(zip(prev["code"], prev["entry_date"]))
    rows = []
    for c in res.holdings:
        new = c in res.buy
        rows.append({"code": c, "name": names.get(c, ""), "score": float(score[c]), "rank": rank.get(c),
                     "entry_date": date if new else pd.Timestamp(prev_entry[c]),
                     "status": "新进" if new else "留存"})
    table = pd.DataFrame(rows, columns=["code", "name", "score", "rank", "entry_date", "status"])
    reference = rank_codes(score[entry & score.notna()])[:topk]
    return Selection(date=date, result=res, table=table, reference=reference, topk=topk,
                     n_entry=int((entry & score.notna()).sum()), n_scored=len(ranked))


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
    lines = [
        f"# {pool.name} 股池 {tag}",
        "",
        f"> 基于 {tag} 收盘数据，供其后首个交易日参考（数据滞后 {meta.get('lag_days', '?')} 个自然日）。",
        f"> 数据版本 {meta.get('data_version')} · 配置 {pool.config_hash} · 打分器 {pool.scorer['name']}"
        f" · 上期 {meta.get('prev_date') or '无（首次建仓）'}",
        "",
        f"- 池容 topk: {sel.topk}（准入通过 {sel.n_entry} 只 / 当日有分 {sel.n_scored} 只）",
        f"- 换仓规则: n_drop={pool.strategy_params.get('n_drop')} hold_thresh={pool.strategy_params.get('hold_thresh')}",
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
