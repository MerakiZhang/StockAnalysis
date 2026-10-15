"""LightGBM 固定树数选择：研发区各滚动窗口训练一次（最大树数），按前 k 棵树预测，比较 RankIC / TopK / 分数并列。

只用测试期 ≤ in_sample_end 的窗口。标签即未来 5 日收益（features/labels.py）。
"""
from __future__ import annotations

import pandas as pd

from ..config import load_pool


def run_rounds(name: str, rounds: list[int]) -> list[str]:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    from ..data.market import trade_calendar
    from ..features.alpha158 import build_handler
    from ..models.estimators import make_model, model_params
    from ..models.store import feature_sets
    from ..models.windows import pool_windows
    from ..selection.universe import board_of
    from .metrics import rank_ic, score_ties, topk_excess

    pool = load_pool(name, sets=["scorer.name=lgbm"])
    ins = pool.backtest["in_sample_end"]
    windows = pool_windows(pool, trade_calendar(), pool.backtest["start"], ins)
    handler = build_handler(pool.universe, windows[0].train[0], ins, int(pool.model.get("label_horizon", 5)),
                            feature_sets(pool))
    label = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0].unstack()
    allowed = set(pool.filters.get("boards", {}).get("boards", ["main", "cyb", "kcb", "bse"]))
    eligible = pd.DataFrame({c: board_of(c) in allowed for c in label.columns}, index=label.index)

    params = {**model_params(pool, "lgbm"), "early_stopping_rounds": 0, "num_boost_round": max(rounds)}
    preds: dict[int, list[pd.Series]] = {k: [] for k in rounds}
    for w in windows:
        ds = DatasetH(handler, segments={"train": w.train, "valid": w.valid, "test": w.test})
        model = make_model("lgbm", params)
        model.fit(ds, verbose_eval=1000)
        x = ds.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
        for k in rounds:
            preds[k].append(pd.Series(model.model.predict(x.values, num_iteration=k), index=x.index))
        print(f"[{name}] {w.test[0]} 完成", flush=True)

    rows = []
    for k in rounds:
        score = pd.concat(preds[k]).sort_index().unstack().reindex_like(label).where(eligible)
        ric = rank_ic(score, label)
        tk = topk_excess(score, eligible, label)
        ties = score_ties(score.dropna(how="all"))
        r = {"树数": k, "RankIC": ric.mean(), "RankICIR": ric.mean() / ric.std(),
             **{f"Top{n}超额%": v.mean() * 100 for n, v in tk.items()},
             "去重比例": ties["去重比例"].mean(), "最高分并列": ties["最高分并列"].mean()}
        rows.append(r)
        print(f"[{name}] {k} 棵: RankIC {r['RankIC']:.4f} Top5 {r['Top5超额%']:.3f} Top20 {r['Top20超额%']:.3f} "
              f"去重 {r['去重比例']:.2f}", flush=True)
    return [f"## {name}（研发区 {windows[0].test[0]} ~ {ins}，学习率 {params['learning_rate']}）", "",
            pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"), ""]
