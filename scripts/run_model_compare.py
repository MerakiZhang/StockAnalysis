"""模型 A/B 对比：同池同区间，各打分方案的 RankIC 与回测指标并列。

默认只看研发区（调参/选模型只能用这段）；参数冻结后加 --show-holdout 一次性评估保留集。

两类指标：
- 组合回测（含换仓规则与成本）：贴近实盘，但 5 只的组合路径依赖强、噪声大；
- TopK 未来 5 日超额（每日按分数取准入池前 K 只，相对准入池均值，不含成本）：不受换仓路径影响，
  更稳定地衡量"头部选股能力"。两者结论不一致时以 TopK 与 RankIC 为主。
方案用 "名称=点路径覆盖;点路径覆盖" 描述，模型须已用相同配置训练（run_train.py）。

用法:
    python -X utf8 scripts/run_model_compare.py --pool all
    python -X utf8 scripts/run_model_compare.py --pool zz1000 --variant "lgbm_lr02=scorer.name=lgbm;model.lgbm.learning_rate=0.2"
产出: data/reports/model_compare_{pool}.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from run_backtest import backtest_window  # noqa: E402
from src.config import REPORT_DIR, list_pools, load_global, load_pool  # noqa: E402
from src.data.sync import add_sync_args, data_version, preflight  # noqa: E402

DEFAULT_VARIANTS = [
    "lgbm=scorer.name=lgbm",
    "lgbm_fund=scorer.name=lgbm;model.features=[alpha158,fundamental]",
    "linear=scorer.name=linear",
]


def parse_variant(text: str) -> tuple[str, list[str]]:
    name, _, sets = text.partition("=")
    return name.strip(), [s.strip() for s in sets.split(";") if s.strip()]


def topk_excess(signal, ctx: dict, horizon: int, ks=(5, 10, 20)) -> dict[int, pd.Series]:
    """每日按分数取准入池前 K 只，其未来 horizon 日（T+1 收盘起算）收益相对准入池均值的超额。仅研究用途。"""
    close = ctx["close"]
    fwd = (close.shift(-(horizon + 1)) / close.shift(-1) - 1).reindex_like(signal.score)
    ok = signal.entry & fwd.notna()
    fwd = fwd.where(ok)
    excess = fwd.sub(fwd.mean(axis=1), axis=0)
    rank = signal.score.where(ok).rank(axis=1, ascending=False, method="first")
    return {k: excess.where(rank <= k).mean(axis=1) for k in ks}


def compare_pool(name: str, variants: list[str], show_holdout: bool) -> list[str]:
    from src.backtest.engine import run_backtest, segment_metrics, segments
    from src.models.rolling import load_manifest
    from src.selection.signal import build_ctx, build_signal

    rows, ctx = [], None
    for text in variants:
        vname, sets = parse_variant(text)
        pool = load_pool(name, sets=sets)
        start, end = backtest_window(pool, None, None, show_holdout)
        kind = pool.scorer["name"]
        try:
            manifest, _ = load_manifest(pool, kind)
        except FileNotFoundError as e:
            print(f"[{name}/{vname}] 跳过：{e}", flush=True)
            continue
        ic = manifest["ic"] if show_holdout else manifest["ic_dev"]
        ctx = ctx or build_ctx(pool, start, end)  # 行情与过滤上下文各方案共用
        print(f"[{name}/{vname}] 回测 {start} ~ {end} ...", flush=True)
        signal = build_signal(pool, start, end, ctx=ctx)
        res = run_backtest(signal, pool, start, end)
        topk = topk_excess(signal, ctx, pool.model.get("label_horizon", 5))
        for tag, s, e in segments(start, pool.backtest.get("in_sample_end", end), end):
            if tag not in ("研发区", "保留集"):
                continue
            m = segment_metrics(res["report"], s, e)
            tk = {f"Top{k}超额%": v.loc[s:e].mean() * 100 for k, v in topk.items()}
            rows.append({"方案": vname, "区间": tag, "签名": manifest["signature"],
                         "RankIC": ic["rank_ic"], **tk, "超额年化": m["excess_ann"], "IR": m["ir"],
                         "超额回撤": m["mdd"], "日均换手": m["turnover"], "年化成本": m["cost"]})
        print(f"  RankIC {ic['rank_ic']:.4f} 超额 {res['excess_ann']:.2%} IR {res['ir']:.2f}", flush=True)
    span = "研发区+保留集" if show_holdout else "研发区"
    table = pd.DataFrame(rows)
    return [f"## {name}（{span}）", "",
            table.to_markdown(index=False, floatfmt=".4f") if not table.empty else "无可比方案", ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="all", help="池名，逗号分隔或 all；可选: " + ",".join(list_pools()))
    ap.add_argument("--variant", action="append", default=None,
                    help='方案，可重复：名称=点路径覆盖;点路径覆盖（默认 lgbm / lgbm_fund / linear）')
    ap.add_argument("--show-holdout", action="store_true", help="含保留集（仅参数冻结后的最终评估）")
    ap.add_argument("--tag", default=None)
    add_sync_args(ap)
    args = ap.parse_args()
    preflight(args, load_global().get("data"))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name in (list_pools() if args.pool == "all" else args.pool.split(",")):
        lines = [f"# 模型对比：{name}", "", f"生成时间 {pd.Timestamp.now():%Y-%m-%d %H:%M} · 数据版本 {data_version()}", "",
                 "RankIC 为滚动样本外预测在对应区间的均值；回测按池 yaml 的过滤与换仓规则。", ""]
        lines += compare_pool(name, args.variant or DEFAULT_VARIANTS, args.show_holdout)
        out = REPORT_DIR / f"model_compare_{name}{'_' + args.tag if args.tag else ''}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"报告: {out}", flush=True)


if __name__ == "__main__":
    main()
