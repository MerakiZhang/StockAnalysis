"""补充基本面特征体检：覆盖率、RankIC（5 日 / 20 日）、逐年 IC、特征间相关。

用途是发现坏特征（缺失多、方向异常、与已有特征高度重复）并留档，不用于手工定权重（打分由模型完成）。
口径：各池当日指数成分 ∩ boards；前向收益与模型标签一致（features/labels.py）。
"""
from __future__ import annotations

import pandas as pd

from ..config import load_pool


def check_pool(name: str, show_holdout: bool = False) -> list[str]:
    from ..data.market import latest_trade_date, load_wide
    from ..features.fundamental import FEATURES, START_YEAR, build_fundamental
    from ..features.labels import forward_returns
    from ..selection.filters import boards as boards_filter
    from ..selection.universe import membership_panel, universe_spans
    from .metrics import factor_correlation, rank_ic, summarize_ic

    pool = load_pool(name)
    start = pool.backtest.get("start", "2018-01-01")
    end = None if show_holdout else pool.backtest.get("in_sample_end")
    end = end or str(latest_trade_date().date())
    spans = universe_spans(pool.universe, start, end)
    codes = sorted(spans)
    print(f"[{name}] {len(codes)} 只，{start} ~ {end}", flush=True)
    tail = str(pd.Timestamp(end) + pd.Timedelta(days=40))  # 多取约 25 个交易日供前向收益使用
    wide = load_wide(codes, f"{START_YEAR}-01-01", tail, ("close", "factor"))
    feats = build_fundamental(codes, start, end, wide["close"], wide["factor"])
    cal = feats["ep_ttm"].index
    mask = membership_panel(spans, cal, codes)
    if "boards" in pool.filters:
        mask &= boards_filter({"close": mask}, **pool.filters["boards"])
    fwd = {h: forward_returns(wide["close"], h).reindex(index=cal, columns=codes) for h in (5, 20)}

    rows, yearly = [], {}
    for f in FEATURES:
        x = feats[f].where(mask)
        cov = x.notna().sum(axis=1) / mask.sum(axis=1)
        row = {"特征": f, "覆盖率": cov.mean()}
        for h in (5, 20):
            ic = rank_ic(x, fwd[h].where(mask))
            s = summarize_ic(ic)
            row.update({f"RankIC{h}d": s["mean_ic"], f"ICIR{h}d": s["icir"], f"正比例{h}d": s["positive_rate"]})
            if h == 5:
                yearly[f] = ic.groupby(ic.index.year).mean()
        row["最差年覆盖"] = cov.groupby(cov.index.year).mean().min()
        rows.append(row)
        print(f"  {f}: 覆盖 {row['覆盖率']:.1%} RankIC5d {row['RankIC5d']:.4f} ICIR {row['ICIR5d']:.2f}", flush=True)

    corr = factor_correlation({f: feats[f].where(mask).iloc[::5] for f in FEATURES})  # 每 5 日抽样，省时
    boards = pool.filters.get("boards", {}).get("boards", "全部")
    return [
        f"## {name}（{pool.universe} ∩ boards={boards}，{'研发区+保留集' if show_holdout else '研发区'} {start} ~ {end}）", "",
        pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"), "",
        "### 逐年 RankIC（5 日）", "", pd.DataFrame(yearly).to_markdown(floatfmt=".4f"), "",
        "### 特征间平均截面相关", "", corr.to_markdown(floatfmt=".2f"), "",
    ]
