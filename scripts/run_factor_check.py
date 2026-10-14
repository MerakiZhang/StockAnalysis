"""补充基本面特征体检：覆盖率、RankIC（5 日 / 20 日）、逐年 IC、特征间相关。

用途是发现坏特征（缺失多、方向异常、与已有特征高度重复）并留档，不用于手工定权重（打分由模型完成）。
口径：各池当日指数成分 ∩ boards；前向收益与模型标签一致（T+1 收盘起算）。
默认只看研发区（backtest.start ~ in_sample_end）。

用法:
    python -X utf8 scripts/run_factor_check.py --pool all
产出: data/reports/factor_report_{pool}.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import REPORT_DIR, list_pools, load_global, load_pool
from src.data.sync import add_sync_args, data_version, preflight


def check_pool(name: str, show_holdout: bool) -> list[str]:
    from qlib.data import D

    from src import eval as fe
    from src.factors.base import load_wide
    from src.factors.supplementary import FEATURES, START_YEAR, build_supplementary
    from src.selection.filters import boards as boards_filter
    from src.selection.universe import membership_panel, universe_spans

    pool = load_pool(name)
    start = pool.backtest.get("start", "2018-01-01")
    end = None if show_holdout else pool.backtest.get("in_sample_end")
    end = end or str(pd.Timestamp(D.calendar()[-1]).date())
    spans = universe_spans(pool.universe, start, end)
    codes = sorted(spans)
    print(f"[{name}] {len(codes)} 只，{start} ~ {end}", flush=True)
    # 行情多取 25 个交易日，供前向收益使用（仅研究用途）
    tail = str(pd.Timestamp(end) + pd.Timedelta(days=40))
    wide = load_wide(codes, f"{START_YEAR}-01-01", tail, ("close", "factor"))
    feats = build_supplementary(codes, start, end, wide["close"], wide["factor"])
    cal = feats["ep_ttm"].index
    mask = membership_panel(spans, cal, codes)
    if "boards" in pool.filters:
        mask &= boards_filter({"close": mask}, **pool.filters["boards"])
    close = wide["close"]
    fwd = {h: (close.shift(-(h + 1)) / close.shift(-1) - 1).reindex(index=cal, columns=codes) for h in (5, 20)}

    rows, yearly = [], {}
    for f in FEATURES:
        x = feats[f].where(mask)
        cov = x.notna().sum(axis=1) / mask.sum(axis=1)
        row = {"特征": f, "覆盖率": cov.mean()}
        for h in (5, 20):
            ic = fe.daily_rank_ic(x, fwd[h].where(mask))
            s = fe.summarize_ic(ic)
            row.update({f"RankIC{h}d": s["mean_ic"], f"ICIR{h}d": s["icir"], f"正比例{h}d": s["positive_rate"]})
            if h == 5:
                yearly[f] = ic.groupby(ic.index.year).mean()
        row["最差年覆盖"] = cov.groupby(cov.index.year).mean().min()
        rows.append(row)
        print(f"  {f}: 覆盖 {row['覆盖率']:.1%} RankIC5d {row['RankIC5d']:.4f} ICIR {row['ICIR5d']:.2f}", flush=True)

    table = pd.DataFrame(rows)
    corr = fe.factor_correlation({f: feats[f].where(mask).iloc[::5] for f in FEATURES})  # 每 5 日抽样，省时
    span = "研发区+保留集" if show_holdout else "研发区"
    return [
        f"## {name}（{pool.universe} ∩ boards={pool.filters.get('boards', {}).get('boards', '全部')}，{span} {start} ~ {end}）", "",
        table.to_markdown(index=False, floatfmt=".4f"), "",
        "### 逐年 RankIC（5 日）", "", pd.DataFrame(yearly).to_markdown(floatfmt=".4f"), "",
        "### 特征间平均截面相关", "", corr.to_markdown(floatfmt=".2f"), "",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="all", help="池名，逗号分隔或 all；可选: " + ",".join(list_pools()))
    ap.add_argument("--show-holdout", action="store_true", help="含保留集（默认只看研发区）")
    add_sync_args(ap)
    args = ap.parse_args()
    preflight(args, load_global().get("data"))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in (list_pools() if args.pool == "all" else args.pool.split(",")):
        lines = [f"# 补充基本面特征体检：{name}", "",
                 f"生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 数据版本 {data_version()}", ""]
        lines += check_pool(name, args.show_holdout)
        out = REPORT_DIR / f"factor_report_{name}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"报告: {out}", flush=True)


if __name__ == "__main__":
    main()
