"""模型 A/B 对比：同池同区间，各打分方案的 RankIC、TopK 未来超额与组合回测并列。

方案写作 "名称=点路径覆盖;点路径覆盖"，如 "lgbm_lr01=scorer.name=lgbm;model.lgbm.learning_rate=0.1"。
两类指标：组合回测贴近实盘但 5 只组合噪声大；TopK / RankIC 不受换仓路径影响，结论不一致时以后者为主。
"""
from __future__ import annotations

import pandas as pd

from ..config import load_pool

DEFAULT_VARIANTS = [
    "lgbm=scorer.name=lgbm",
    "lgbm_fund=scorer.name=lgbm;model.features=[alpha158,fundamental]",
    "linear=scorer.name=linear",
]


def parse_variant(text: str) -> tuple[str, list[str]]:
    name, _, sets = text.partition("=")
    return name.strip(), [s.strip() for s in sets.split(";") if s.strip()]


def compare_pool(name: str, variants: list[str], show_holdout: bool = False) -> list[str]:
    from ..backtest.engine import run_backtest
    from ..backtest.metrics import segment_metrics, segments
    from ..backtest.runner import backtest_window
    from ..features.labels import forward_returns
    from ..models.store import load_manifest
    from ..selection.signal import build_ctx, build_signal
    from .metrics import rank_ic, topk_excess

    rows, ctx = [], None
    for text in variants:
        vname, sets = parse_variant(text)
        pool = load_pool(name, sets=sets)
        start, end = backtest_window(pool, None, None, show_holdout)
        try:
            manifest, _ = load_manifest(pool, pool.scorer["name"])
        except FileNotFoundError as e:
            print(f"[{name}/{vname}] 跳过：{e}", flush=True)
            continue
        ctx = ctx or build_ctx(pool, start, end)  # 行情与过滤上下文各方案共用
        print(f"[{name}/{vname}] 回测 {start} ~ {end} ...", flush=True)
        signal = build_signal(pool, start, end, ctx=ctx)
        res = run_backtest(signal, pool, start, end)
        fwd = forward_returns(ctx["close"], int(pool.model.get("label_horizon", 5)))
        topk = topk_excess(signal.score, signal.entry, fwd)
        ric = rank_ic(signal.score, fwd)
        for tag, s, e in segments(start, pool.backtest.get("in_sample_end", end), end):
            if tag not in ("研发区", "保留集"):
                continue
            m = segment_metrics(res["report"], s, e)
            tk = {f"Top{k}超额%": v.loc[s:e].mean() * 100 for k, v in topk.items()}
            rows.append({"方案": vname, "区间": tag, "签名": manifest["signature"],
                         "RankIC": float(ric.loc[s:e].mean()), **tk, "超额年化": m["excess_ann"], "IR": m["ir"],
                         "超额回撤": m["mdd"], "日均换手": m["turnover"], "年化成本": m["cost"]})
        print(f"  超额 {res['excess_ann']:.2%} IR {res['ir']:.2f}", flush=True)
    table = pd.DataFrame(rows)
    return [f"## {name}（{'研发区+保留集' if show_holdout else '研发区'}）", "",
            "RankIC：滚动样本外预测在当日指数成分内的逐日秩相关均值；TopK：准入范围内前 K 只的未来 5 日超额（%/期）。", "",
            table.to_markdown(index=False, floatfmt=".4f") if not table.empty else "无可比方案", ""]
