"""买卖点池参数网格 + 研发区滚动验证（walk-forward）。

方案 = 若干点路径覆盖（与 --set 同语法）；--vary 给出每个键的候选值，取笛卡尔积。
同一行情上下文只加载一次；买卖条件 / 择时 / 优先分 / 过滤不变的方案复用同一 PointSignal（只重跑模拟器）。

滚动验证（只在研发区内）：
    F1：2018~2020 选最优方案 → 看它在 2021 的表现
    F2：2018~2021 选最优方案 → 看它在 2022 的表现
选优目标默认为扣成本后的绝对年化收益（"两个池都要赚钱"）；也可选 excess（超额年化）或 calmar（年化 / 最大回撤）。

多起点（--starts K）：10 只股票的组合对建仓日很敏感（长线池 5 年只有几十笔交易）。每个方案从 K 个错开的起点
（每隔 STEP 个交易日）各模拟一次，逐日取已开始的各起点收益的平均，合成一条"组合的组合"净值再算指标；
另报告各起点单独年化的最差 / 最好值，看结论是否依赖运气。
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from ..config import load_pool
from ..paths import REPORT_DIR

FOLDS = (("F1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
         ("F2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"))
SIGNAL_KEYS = ("buy", "buy_any", "avoid", "sell", "regime", "freq", "deal_price")   # 改变 PointSignal 的 points 子项（deal_price 决定 fill）
STEP = 42   # 多起点间隔（交易日，约 2 个月）
PARTIAL_DIR = REPORT_DIR / "logs"


def expand(vary: list[str]) -> list[list[str]]:
    """["points.stop_loss=0.05,0.08", "points.max_hold=5,10"] -> 笛卡尔积的 --set 列表。

    候选值含逗号（如 yaml 字典）时改用 | 分隔：points.buy={a: {}, b: {}}|{a: {}}
    """
    axes = []
    for item in vary:
        key, _, vals = item.partition("=")
        sep = "|" if "|" in vals else ","
        axes.append([f"{key.strip()}={v.strip()}" for v in vals.split(sep)])
    return [list(c) for c in itertools.product(*axes)] if axes else [[]]


def _abs_mdd(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min()) if len(nav) else np.nan


def _seg(res: dict, s: str, e: str) -> dict:
    from ..backtest.metrics import segment_metrics

    m = segment_metrics(res["report"], s, e)
    nav = res["report"]["nav"].loc[s:e]
    return {**m, "abs_mdd": _abs_mdd(nav)}


def _objective(m: dict, target: str) -> float:
    if not m:
        return -np.inf
    if target == "excess":
        return m["excess_ann"]
    if target == "calmar":
        return m["ann"] / abs(m["abs_mdd"]) if m.get("abs_mdd") else -np.inf
    return m["ann"]


def _signal_key(pool) -> str:
    p = pool.points
    blob = {k: getattr(p, k) for k in SIGNAL_KEYS}
    return json.dumps({"points": blob, "scorer": pool.scorer, "filters": pool.filters}, sort_keys=True, default=str)


def ensemble(reports: list[pd.DataFrame]) -> pd.DataFrame:
    """多个起点的 report 合成：逐日对已开始的起点取平均（等资金分给 K 个错开建仓的子组合）。"""
    idx = reports[0].index
    cols = ("return", "cost", "turnover", "n_pos")
    out = pd.DataFrame({c: pd.concat([r[c].reindex(idx) for r in reports], axis=1).mean(axis=1) for c in cols})
    out["bench"] = reports[0]["bench"].reindex(idx)
    out["cash"] = pd.concat([r["cash"].reindex(idx) for r in reports], axis=1).mean(axis=1)
    out["nav"] = (1 + out["return"] - out["cost"]).cumprod()
    return out


def run_grid(name: str, vary: list[str], base_sets: list[str] | None = None, target: str = "ann",
             starts: int = 1, log=print) -> list[str]:
    from ..backtest.points_engine import run_points_backtest
    from ..selection.points import WARMUP_DAYS, build_point_signal, slice_from
    from ..selection.signal import build_ctx

    base_sets = list(base_sets or [])
    variants = expand(vary)
    first = load_pool(name, sets=base_sets)
    start, end = first.backtest.get("start", "2018-01-01"), first.backtest["in_sample_end"]
    log(f"[{name}] {len(variants)} 个方案，研发区 {start} ~ {end}，加载行情 ...")
    ctx = build_ctx(first, start, end, WARMUP_DAYS)
    # 按信号键分组依次跑，内存里只留当前一组的 PointSignal（每个约数百 MB）
    pools = [load_pool(name, sets=base_sets + sets) for sets in variants]
    keys = [_signal_key(p) for p in pools]
    order = sorted(range(len(variants)), key=lambda i: (keys.index(keys[i]), i))
    partial = PARTIAL_DIR / f"points_grid_{name}.partial.csv"
    partial.parent.mkdir(parents=True, exist_ok=True)
    cur_key, ps = None, None
    rows = []
    for k, i in enumerate(order, 1):
        sets, pool = variants[i], pools[i]
        if keys[i] != cur_key:
            ps = None
            ps, cur_key = build_point_signal(pool, start, end, ctx, with_extra=False), keys[i]
        runs = [run_points_backtest(pool, str(ps.dates[j * STEP].date()), end, ps=slice_from(ps, j * STEP) if j else ps)
                for j in range(starts) if j * STEP < len(ps.dates) - 250]
        res = {**runs[0], "report": ensemble([r["report"] for r in runs]) if len(runs) > 1 else runs[0]["report"]}
        for kk in ("trades", "win_rate", "payoff", "avg_days", "avg_pos"):
            vals = [r.get(kk) for r in runs if r.get(kk) is not None]
            res[kk] = float(np.nanmean(vals)) if vals else np.nan
        row = {"方案": "; ".join(s.replace("points.", "") for s in sets) or "基准"}
        full = _seg(res, start, end)
        row.update({"年化": full.get("ann"), "超额年化": full.get("excess_ann"), "IR": full.get("ir"),
                    "最大回撤": full.get("abs_mdd"), "超额回撤": full.get("mdd"), "交易笔数": res.get("trades", 0),
                    "胜率": res.get("win_rate"), "盈亏比": res.get("payoff"), "平均持有": res.get("avg_days"),
                    "平均持仓": res.get("avg_pos"), "年化成本": full.get("cost")})
        row["最大单票权重"] = float(np.nanmax([r["report"]["max_w"].max() for r in runs]))
        if len(runs) > 1:
            each = [r.get("ann") for r in runs]
            row.update({"起点最差年化": float(np.nanmin(each)), "起点最好年化": float(np.nanmax(each))})
        for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            row[f"{y}年化"] = _seg(res, f"{y}-01-01", f"{y}-12-31").get("ann")
        for fold, ts, te, vs, ve in FOLDS:
            row[f"{fold}选优"] = _objective(_seg(res, ts, te), target)
            row[f"{fold}检验"] = _seg(res, vs, ve).get("ann")
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial, index=False, encoding="utf-8-sig")   # 中途崩溃也留下已跑完的方案
        log(f"  [{k}/{len(variants)}] {row['方案']}: 年化 {row['年化']:.2%} 超额 {row['超额年化']:.2%} "
            f"回撤 {row['最大回撤']:.2%} 交易 {row['交易笔数']} 笔")
    table = pd.DataFrame(rows)

    wf = []
    for fold, ts, te, vs, ve in FOLDS:
        best = table.loc[table[f"{fold}选优"].idxmax()]
        wf.append({"折": fold, "选优区间": f"{ts[:4]}~{te[:4]}", "选中方案": best["方案"],
                   "选优目标值": best[f"{fold}选优"], "检验区间": vs[:4], "检验年化": best[f"{fold}检验"],
                   "检验区间全部方案中位数": table[f"{fold}检验"].median()})
    pct_cols = [c for c in table.columns if c.endswith("年化") or c in ("最大回撤", "超额回撤", "胜率", "年化成本", "最大单票权重")
                or c.endswith("检验")]
    shown = table.sort_values("年化", ascending=False).copy()   # 先按数值排序，再格式化成百分比
    for c in pct_cols:
        shown[c] = shown[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    return [
        f"## {name}（研发区 {start} ~ {end}，选优目标 {target}）", "",
        f"- 基础覆盖：{'; '.join(base_sets) or '无'}"
        + (f"；多起点 {starts} 个（间隔 {STEP} 个交易日，指标为合成净值）" if starts > 1 else ""), "",
        "### 滚动验证", "", pd.DataFrame(wf).to_markdown(index=False, floatfmt=".3f"), "",
        "### 全部方案", "", shown.to_markdown(index=False, floatfmt=".2f"), "",
    ]
