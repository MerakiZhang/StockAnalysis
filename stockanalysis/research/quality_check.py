"""烂股剔除效果体检：每条规则每年命中多少只，命中组与保留组的后续收益差。

口径：当日在市（in_universe）∩ 板块权限（boards）内的股票。前向收益与标签一致（features/labels.py，
T+1 收盘起算），横截面减去当日全体均值后取平均（%/期）。另报告"大幅下跌概率"：前向 60 日收益 < -30% 的比例。
一条规则有效的标志：命中组的前向超额稳定为负、大跌概率明显更高；命中只数过多则说明阈值太严。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_pool

SKIP = ("in_universe", "boards", "exclude_limit", "min_amount")  # 不属于"烂股"判定
HORIZONS = (20, 60)
CRASH = -0.30


def check_pool(name: str, show_holdout: bool = False, sets: list[str] | None = None, log=print) -> list[str]:
    from ..data.market import latest_trade_date
    from ..features.labels import forward_returns
    from ..selection.filters import FILTERS, scope_of
    from ..selection.signal import build_ctx

    pool = load_pool(name, sets=sets)
    start = pool.backtest.get("start", "2018-01-01")
    end = None if show_holdout else pool.backtest.get("in_sample_end")
    end = end or str(latest_trade_date().date())
    tail = str(min(pd.Timestamp(end) + pd.Timedelta(days=100), latest_trade_date()).date())  # 前向收益
    log(f"[{name}] 加载 {pool.universe} {start} ~ {tail} ...")
    ctx = build_ctx(pool, start, tail, warmup_days=400)

    base = FILTERS.get("in_universe")(ctx) & FILTERS.get("boards")(ctx, **pool.filters["boards"])
    fwd = {h: forward_returns(ctx["close"], h) for h in HORIZONS}
    s, e = pd.Timestamp(start), pd.Timestamp(end)

    def stats(flag: pd.DataFrame) -> dict:
        flag = (flag & base).loc[s:e]
        b = base.loc[s:e]
        row = {"日均命中只数": float(flag.sum(axis=1).mean()), "命中占比": float(flag.sum().sum() / max(b.sum().sum(), 1))}
        for h, f in fwd.items():
            f = f.loc[s:e].where(b)
            ex = f.sub(f.mean(axis=1), axis=0)
            row[f"命中组{h}日超额%"] = float(ex.where(flag).stack().mean() * 100) if flag.any().any() else np.nan
            row[f"保留组{h}日超额%"] = float(ex.where(b & ~flag).stack().mean() * 100)
        f60 = fwd[60].loc[s:e]
        row["命中组60日大跌率"] = float((f60.where(flag) < CRASH).sum().sum() / max(f60.where(flag).notna().sum().sum(), 1))
        row["保留组60日大跌率"] = float((f60.where(b & ~flag) < CRASH).sum().sum()
                                     / max(f60.where(b & ~flag).notna().sum().sum(), 1))
        return row

    rows, yearly, any_flag = [], {}, pd.DataFrame(False, index=base.index, columns=base.columns)
    for fname, params in pool.filters.items():
        if fname in SKIP:
            continue
        log(f"  {fname} {params}")
        flag = ~FILTERS.get(fname)(ctx, **params).reindex_like(base).fillna(True).astype(bool)
        if scope_of(fname) == "exit":
            any_flag |= flag
        rows.append({"规则": fname, "范围": scope_of(fname), **stats(flag)})
        hit = (flag & base).loc[s:e]
        yearly[fname] = hit.sum(axis=1).groupby(hit.index.year).mean()
    rows.append({"规则": "exit 合计（任一命中）", "范围": "exit", **stats(any_flag)})
    hit = (any_flag & base).loc[s:e]
    yearly["exit 合计"] = hit.sum(axis=1).groupby(hit.index.year).mean()
    universe_n = base.loc[s:e].sum(axis=1).groupby(base.loc[s:e].index.year).mean()

    cov = ctx.get("_coverage", {})
    notes = [f"- 历史 ST 中有 baostock 数据的股票占比：{cov.get('st_baostock', 0):.1%}"
             "（其余按涨跌停幅度近似，只覆盖主板）"]
    for k, label in (("risk_notice", "公告"), ("release", "解禁")):
        if k in cov:
            notes.append(f"- {label}数据缺失：对应规则未生效（先运行 update-data --only events）")
    return [
        f"## {name}（{pool.universe} ∩ boards={pool.filters['boards']['boards']}，"
        f"{'研发区+保留集' if show_holdout else '研发区'} {start} ~ {end}）", "",
        *notes, "",
        pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), "",
        "### 每年日均命中只数", "",
        pd.DataFrame({"在市只数": universe_n, **yearly}).to_markdown(floatfmt=".0f"), "",
    ]
