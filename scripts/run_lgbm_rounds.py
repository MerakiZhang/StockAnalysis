"""LightGBM 固定树数选择：研发区滚动窗口各训练一次（最大树数），按前 k 棵树预测，比较 RankIC / TopK / 分数并列。

只用测试期 ≤ in_sample_end 的窗口（研发区）。TopK 超额 = 当日主板成分中预测前 K 只的标签均值 − 全体均值
（标签即未来 5 日收益，T+1 收盘起算）。

用法:
    python -X utf8 scripts/run_lgbm_rounds.py --pool hs300 --rounds 25,50,100,200,400
产出: data/reports/lgbm_rounds_{pool}.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import REPORT_DIR, list_pools, load_global, load_pool
from src.data.sync import add_sync_args, data_version, preflight


def evaluate(pred: pd.Series, label: pd.Series, board_ok: pd.Series, ks=(5, 10, 20)) -> dict:
    df = pd.DataFrame({"p": pred, "y": label, "ok": board_ok}).dropna()
    df = df[df["ok"]]
    g = df.groupby(level=0)
    ric = g.apply(lambda x: x["p"].rank().corr(x["y"].rank()))
    out = {"RankIC": ric.mean(), "RankICIR": ric.mean() / ric.std()}
    rank = g["p"].rank(ascending=False, method="first")
    base = g["y"].transform("mean")
    for k in ks:
        out[f"Top{k}超额%"] = (df["y"] - base)[rank <= k].groupby(level=0).mean().mean() * 100
    out["去重比例"] = g["p"].apply(lambda s: s.nunique() / len(s)).mean()
    out["最高分并列"] = g["p"].apply(lambda s: (s == s.max()).sum()).mean()
    return out


def run_pool(name: str, rounds: list[int]) -> list[str]:
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    from src.data.market import init_qlib
    from src.factors.alpha158 import build_handler
    from src.models.rolling import feature_sets, make_model, model_params, rolling_windows
    from src.selection.universe import board_of

    init_qlib()
    pool = load_pool(name, sets=["scorer.name=lgbm"])
    m = pool.model
    horizon = int(m.get("label_horizon", 5))
    ins = pool.backtest["in_sample_end"]
    cal = pd.DatetimeIndex(D.calendar())
    windows = [w for w in rolling_windows(cal, pool.backtest["start"], ins, int(m.get("train_years", 4)),
                                          int(m.get("valid_months", 6)), horizon + 1, m.get("retrain", "yearly"))]
    handler = build_handler(pool.universe, windows[0].train[0], ins, horizon, feature_sets(pool))
    label = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0]
    allowed = set(pool.filters.get("boards", {}).get("boards", ["main", "cyb", "kcb", "bse"]))
    board_ok = pd.Series([board_of(c) in allowed for c in label.index.get_level_values(1)], index=label.index)

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
        pred = pd.concat(preds[k]).sort_index()
        r = evaluate(pred, label, board_ok)
        rows.append({"树数": k, **r})
        print(f"[{name}] {k} 棵: RankIC {r['RankIC']:.4f} Top5 {r['Top5超额%']:.3f} Top20 {r['Top20超额%']:.3f} "
              f"去重 {r['去重比例']:.2f}", flush=True)
    return [f"## {name}（研发区 {windows[0].test[0]} ~ {ins}，学习率 {params['learning_rate']}）", "",
            pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"), ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="池名（中证1000 内存大，建议逐池单独运行）")
    ap.add_argument("--rounds", default="25,50,100,200,400")
    add_sync_args(ap)
    args = ap.parse_args()
    preflight(args, load_global().get("data"))
    rounds = sorted(int(x) for x in args.rounds.split(","))
    lines = [f"# LightGBM 树数选择：{args.pool}", "",
             f"生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 数据版本 {data_version()}", ""]
    lines += run_pool(args.pool, rounds)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"lgbm_rounds_{args.pool}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告: {out}", flush=True)


if __name__ == "__main__":
    main()
