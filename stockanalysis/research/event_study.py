"""条件事件研究：条件库（features/conditions.py）逐条、逐个阈值在研发区检验，连续因子做 RankIC，另看择时指标。

样本范围（base）：当日在市、通过全部烂股规则与准入规则（板块 / 上市满一年 / 无停牌 / 未涨停 / 成交额门槛）的股票，
即"真能买进"的股票。前向收益与标签同口径（T+1 收盘起算，features/labels.py）。

每个条件 × 阈值输出：
    日均触发    满足条件的日均只数
    h 日超额%   触发样本的前向收益 - 当日 base 均值，按样本平均（h = 5 / 20 / 60）
    5日胜率 / 5日正年数   短线看这两列（5 日超额 > 0 的样本比例、逐年平均为正的年数）
    20日胜率    超额 > 0 的样本比例；20日盈亏比 = 正超额均值 / 负超额均值的绝对值
    20日t       按交易日先取均值再算 t 值（重叠持有期会高估，只作参考，以逐年为准）
    正年数      20 日超额逐年平均为正的年数 / 年数
    20日绝对%   触发样本的前向绝对收益（"赚钱"目标看这一列）
    邻域同号    同一条件所有阈值的 20 日超额是否同号（阈值稍变结论就反转的条件不可靠）

这些都只是研究用的前向收益，不得进入选股信号。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..config import load_pool

HORIZONS = (5, 20, 60)


def _metrics(flag: np.ndarray, ex: dict[int, np.ndarray], fwd20: np.ndarray, years: np.ndarray) -> dict:
    row = {"日均触发": float(flag.sum(axis=1).mean())}
    for h, e in ex.items():
        f = flag & np.isfinite(e)
        row[f"{h}日超额%"] = float(e[f].mean() * 100) if f.any() else np.nan
    e = ex[20]
    f = flag & np.isfinite(e)
    if not f.any():
        return {**row, "5日胜率": np.nan, "5日正年数": "", "20日胜率": np.nan, "20日盈亏比": np.nan, "20日t": np.nan,
                "正年数": "", "20日绝对%": np.nan}
    vals = e[f]
    pos, neg = vals[vals > 0], vals[vals < 0]
    cnt = f.sum(axis=1)
    daily = np.where(cnt > 0, np.where(f, e, 0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
    d = daily[np.isfinite(daily)]
    t = d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan
    yearly = pd.Series(daily).groupby(years).mean().dropna()
    e5 = ex[5]
    f5 = flag & np.isfinite(e5)
    c5 = f5.sum(axis=1)
    daily5 = np.where(c5 > 0, np.where(f5, e5, 0).sum(axis=1) / np.maximum(c5, 1), np.nan)
    yearly5 = pd.Series(daily5).groupby(years).mean().dropna()
    row.update({"5日胜率": float((e5[f5] > 0).mean()) if f5.any() else np.nan,
                "5日正年数": f"{int((yearly5 > 0).sum())}/{len(yearly5)}"})
    row.update({"20日胜率": float((vals > 0).mean()),
                "20日盈亏比": float(pos.mean() / -neg.mean()) if len(pos) and len(neg) else np.nan,
                "20日t": float(t), "正年数": f"{int((yearly > 0).sum())}/{len(yearly)}",
                "20日绝对%": float(fwd20[flag & np.isfinite(fwd20)].mean() * 100)})
    return row


def _variants(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def study(name: str, show_holdout: bool = False, sets: list[str] | None = None,
          only: list[str] | None = None, log=print) -> list[str]:
    from ..data.market import latest_trade_date
    from ..features import regime
    from ..features.conditions import CONDITIONS, CONTINUOUS
    from ..features.labels import forward_returns
    from ..selection.signal import build_ctx, combine_masks
    from .metrics import rank_ic, summarize_ic

    pool = load_pool(name, sets=sets)
    start = pool.backtest.get("start", "2018-01-01")
    end = None if show_holdout else pool.backtest.get("in_sample_end")
    end = end or str(latest_trade_date().date())
    tail = str(min(pd.Timestamp(end) + pd.Timedelta(days=100), latest_trade_date()).date())
    log(f"[{name}] 加载 {pool.universe} {start} ~ {tail} ...")
    ctx = build_ctx(pool, start, tail, warmup_days=400)
    exit_ok, entry_ok = combine_masks(pool, ctx)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    base = (exit_ok & entry_ok & ctx["close"].notna()).loc[s:e]
    dates = base.index
    years = np.asarray(dates.year)
    b = base.to_numpy()
    log(f"  样本：日均 {b.sum(axis=1).mean():.0f} 只可买股票")

    fwd = {h: forward_returns(ctx["close"], h).loc[s:e].to_numpy(dtype=np.float64) for h in HORIZONS}
    ex = {}
    for h, f in fwd.items():
        f = np.where(b, f, np.nan)
        ex[h] = f - np.nanmean(f, axis=1, keepdims=True)
    fwd20 = np.where(b, fwd[20], np.nan)

    # ---------------- 条件
    rows = []
    names = only or CONDITIONS.names()
    for cname in names:
        meta = CONDITIONS.meta(cname)
        for params in _variants(meta["grid"]):
            try:
                flag = CONDITIONS.get(cname)(ctx, **params)
                flag = flag.reindex(index=ctx["close"].index, columns=ctx["close"].columns).loc[s:e]
                flag = flag.fillna(False).to_numpy(dtype=bool) & b
            except Exception as err:  # noqa: BLE001 - 单个条件失败（如数据未下载）不影响其余
                log(f"  {cname} {params} 失败: {str(err)[:120]}")
                continue
            m = _metrics(flag, ex, fwd20, years)
            rows.append({"条件": cname, "来源": meta["source"], "类型": meta["kind"],
                         "参数": ",".join(f"{k}={v}" for k, v in params.items()) or "-", **m})
            log(f"  {cname} {params}: 日均 {m['日均触发']:.1f} 只，20日超额 {m['20日超额%']:.2f}%")
    table = pd.DataFrame(rows)
    if len(table):
        sign = table.groupby("条件")["20日超额%"].agg(lambda x: "是" if (x > 0).all() or (x < 0).all() else "否")
        table["邻域同号"] = table["条件"].map(sign)
        table = table.sort_values("20日超额%", ascending=False)

    # ---------------- 连续因子 RankIC
    ic_rows = []
    fwd_df = {h: pd.DataFrame(np.where(b, fwd[h], np.nan), index=dates, columns=base.columns) for h in HORIZONS}
    for fname, fn in CONTINUOUS.items():
        try:
            x = fn(ctx).reindex(index=ctx["close"].index, columns=ctx["close"].columns).loc[s:e].where(base)
        except Exception as err:  # noqa: BLE001
            log(f"  因子 {fname} 失败: {str(err)[:120]}")
            continue
        row = {"因子": fname, "覆盖率": float(x.notna().sum().sum() / max(b.sum(), 1))}
        for h in HORIZONS:
            ic = rank_ic(x, fwd_df[h])
            sm = summarize_ic(ic)
            row[f"RankIC{h}d"], row[f"ICIR{h}d"] = sm["mean_ic"], sm["icir"]
            if h == 20:
                yearly = ic.groupby(ic.index.year).mean()
                row["20d正IC年数"] = f"{int((yearly > 0).sum())}/{len(yearly)}"
        ic_rows.append(row)
        log(f"  因子 {fname}: RankIC20d {row['RankIC20d']:.4f}")
    ic_table = pd.DataFrame(ic_rows).sort_values("RankIC20d", ascending=False) if ic_rows else pd.DataFrame()

    # ---------------- 择时：基准指数前向 20 日收益按市场状态分组
    rf = regime.features(ctx, pool.benchmark).loc[s:e]
    idx = regime.index_close(ctx, pool.benchmark)
    idx_fwd20 = (idx.shift(-21) / idx.shift(-1) - 1).loc[s:e]
    eq_fwd20 = pd.Series(np.nanmean(np.where(b, fwd[20], np.nan), axis=1), index=dates)  # base 等权
    reg_rows = []
    for col in ("breadth20", "breadth60", "market_temp", "limit_down_n", "index_ret20"):
        x = rf[col]
        if x.notna().sum() < 50:
            continue
        q = pd.qcut(x.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
        for lab, grp in idx_fwd20.groupby(q, observed=True):
            reg_rows.append({"指标": col, "分组": lab, "取值范围": f"{x[q == lab].min():.2f}~{x[q == lab].max():.2f}",
                             "天数": len(grp), "指数20日%": grp.mean() * 100,
                             "等权20日%": eq_fwd20[q == lab].mean() * 100})
    for flag_val in (1.0, 0.0):
        sel = rf["index_above_ma60"] == flag_val
        reg_rows.append({"指标": "index_above_ma60", "分组": "是" if flag_val else "否", "取值范围": "",
                         "天数": int(sel.sum()), "指数20日%": idx_fwd20[sel].mean() * 100,
                         "等权20日%": eq_fwd20[sel].mean() * 100})

    cov = ctx.get("_coverage", {})
    notes = [f"- 数据覆盖率：{'、'.join(f'{k} {v:.1%}' for k, v in sorted(cov.items()))}" if cov else "- 所有数据源均已加载"]
    return [
        f"## {name}（{pool.universe} ∩ boards={pool.filters['boards']['boards']}，"
        f"{'研发区+保留集' if show_holdout else '研发区'} {start} ~ {end}）", "",
        f"- 样本：日均 {b.sum(axis=1).mean():.0f} 只可买股票；base 20 日平均绝对收益 {np.nanmean(fwd20) * 100:.2f}%",
        *notes, "",
        "### 条件（按 20 日超额排序）", "",
        table.to_markdown(index=False, floatfmt=".2f") if len(table) else "（无）", "",
        "### 连续因子 RankIC", "",
        ic_table.to_markdown(index=False, floatfmt=".4f") if len(ic_table) else "（无）", "",
        "### 择时：按市场状态分组的后续 20 日收益", "",
        pd.DataFrame(reg_rows).to_markdown(index=False, floatfmt=".2f") if reg_rows else "（无）", "",
    ]
