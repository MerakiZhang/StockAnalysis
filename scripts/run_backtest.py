"""回测入口。

用法:
    python -X utf8 scripts/run_backtest.py --score momentum   # 冒烟
    python -X utf8 scripts/run_backtest.py --score real       # 真实因子（双池）

real 模式：按 config/selector.yaml 的因子权重生成日频总分，
稳健池月度调仓回测（2018 起，基准沪深300），短线池周度调仓回测（2020 起，基准中证1000），
按 walk-forward 三段（~2020 / 2021~2022 / 2023~）+ 样本内外输出指标，
写入 data/reports/backtest_report.md。

A/B 实验参数（配合 --tag 避免覆盖基线报告）：
    --weights '{"neg_max_ret_20":1.0}'   覆盖因子权重（需配合 --pools）
    --boards main,cyb                    板块覆盖

内存：默认 QLIB_KERNELS=2（32G 机器实测安全）；紧张时 QLIB_KERNELS=1。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")  # 32G 内存机器默认双进程，可用环境变量覆盖

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest, scores_on_rebalance_dates, wf_segments
from src.config import REPORT_DIR, load_config, load_global
from src.data.sync import add_sync_args, preflight
from src.data.market import init_qlib, latest_trade_date, list_instruments
from src.eval import log_rss
from src.selection.universe import in_boards, is_stock, listing_dates

# 冒烟测试参数
SMOKE_DATA_START = "2019-06-01"
SMOKE_BT_START = "2020-01-01"
SMOKE_BT_END = "2021-06-30"


def make_synth_scores(kind: str, close: pd.DataFrame) -> pd.DataFrame:
    if kind == "momentum":
        return close.pct_change(20, fill_method=None)
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.random(close.shape), index=close.index, columns=close.columns)


def segment_metrics(report: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """从日频 report 切片算某区间指标。"""
    from qlib.contrib.evaluate import risk_analysis

    seg = report.loc[start:end]
    if seg.empty:
        return {}
    net = seg["return"] - seg["cost"]
    excess = net - seg["bench"]
    ana = risk_analysis(excess)["risk"]
    ana_net = risk_analysis(net)["risk"]
    return {
        "ann": float(ana_net["annualized_return"]),
        "excess_ann": float(ana["annualized_return"]),
        "mdd": float(ana["max_drawdown"]),
        "ir": float(ana["information_ratio"]),
        "turnover": float(seg["turnover"].mean()),
    }


def show(name: str, res: dict) -> list[str]:
    lines = [f"## {name}", ""]
    print(f"\n===== {name} =====")
    if res.get("error"):
        print("ERROR:", res["error"])
        return lines + [f"ERROR: {res['error']}", ""]
    print(f"年化收益(含成本):   {res['annualized_return']:>10.2%}")
    print(f"超额年化(含成本):   {res['excess_annualized_return']:>10.2%}")
    print(f"最大回撤(超额):     {res['max_drawdown']:>10.2%}")
    print(f"信息比率(超额):     {res['information_ratio']:>10.2f}")
    print(f"日胜率(跑赢基准):   {res['win_rate']:>10.2%}")
    print(f"日均单边换手:       {res['turnover_mean']:>10.2%}")
    lines += [
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 年化收益(含成本) | {res['annualized_return']:.2%} |",
        f"| 超额年化(含成本) | {res['excess_annualized_return']:.2%} |",
        f"| 最大回撤(超额) | {res['max_drawdown']:.2%} |",
        f"| 信息比率 | {res['information_ratio']:.2f} |",
        f"| 日胜率 | {res['win_rate']:.2%} |",
        f"| 日均单边换手 | {res['turnover_mean']:.2%} |",
    ]
    if "stop_loss_count" in res:
        print(f"止损触发次数:       {res['stop_loss_count']}")
        print(f"到期强平次数:       {res['max_hold_count']}")
        lines.append(f"| 止损/强平次数 | {res['stop_loss_count']} / {res['max_hold_count']} |")
    lines.append("")
    return lines


def run_pool(pool: str, freq: str, score_daily: pd.DataFrame,
             data_start: str, bt_start: str, bt_end: str, cfg: dict,
             in_sample_end: str) -> list[str]:
    scores = scores_on_rebalance_dates(score_daily, freq).loc[data_start:]
    n_sig = int(scores.dropna(how="all").shape[0])
    print(f"[{pool}] freq={freq} 信号日={n_sig} 回测 {bt_start} ~ {bt_end}", flush=True)
    res = run_backtest(scores, pool, bt_start, bt_end, cfg)
    log_rss(f"{pool} 回测完成")
    lines = show(f"{pool} 池全区间", res)
    if res.get("error"):
        return lines
    # walk-forward 三段 + 样本内/外
    lines.append("### 分段表现（walk-forward + 样本内外）")
    lines.append("")
    lines.append("| 区间 | 年化 | 超额年化 | 超额回撤 | IR | 日均换手 |")
    lines.append("|------|------|----------|----------|-----|---------|")
    for tag, s, e in wf_segments(bt_start, in_sample_end, bt_end):
        m = segment_metrics(res["report"], s, e)
        if m:
            lines.append(f"| {tag} | {m['ann']:.2%} | {m['excess_ann']:.2%} | {m['mdd']:.2%} | {m['ir']:.2f} | {m['turnover']:.2%} |")
            print(f"{tag}: 年化{m['ann']:.2%} 超额{m['excess_ann']:.2%} 回撤{m['mdd']:.2%} IR{m['ir']:.2f}")
    lines.append("")
    return lines


def real_scores_stable(cfg: dict, end: str) -> tuple[pd.DataFrame, str]:
    from src.factors.registry import market_ctx, weighted_factor_tables
    from src.selection.scoring import combine

    data_start = "2016-06-01"  # 基本面 ffill + 动量预热
    codes = [c for c in list_instruments("csi800", start="2018-01-01", end=end) if is_stock(c)]
    boards = cfg["stable"].get("boards")
    if boards:
        codes = [c for c in codes if in_boards(c, boards)]
    print(f"stable universe: {len(codes)} (boards={boards or '全板块'})", flush=True)
    ctx = market_ctx(codes, data_start, end)
    factors = weighted_factor_tables(cfg["stable"]["factor_weights"], codes, data_start, end, ctx)
    score = combine(factors, cfg["stable"]["factor_weights"])
    # 过滤：上市满180天 + 当日非停牌（volume 非 NaN）
    vol = ctx["volume"]
    listed_ok = pd.DataFrame(
        {c: (score.index - listing_dates().get(c, score.index[0])).days >= 180 for c in score.columns},
        index=score.index,
    )
    score = score.where(listed_ok & vol.reindex_like(score).notna())
    return score, data_start


def real_scores_short(cfg: dict, end: str) -> tuple[pd.DataFrame, str]:
    from src.factors.registry import market_ctx, weighted_factor_tables
    from src.selection.scoring import combine

    data_start = "2019-01-01"
    scfg = cfg["short"]
    codes = [c for c in list_instruments(scfg["universe"], start="2020-01-01", end=end) if is_stock(c)]
    boards = scfg.get("boards")
    if boards:
        codes = [c for c in codes if in_boards(c, boards)]
    print(f"short universe: {len(codes)} (boards={boards or '全板块'})", flush=True)
    ctx = market_ctx(codes, data_start, end)
    close, volume, amount = ctx["close"], ctx["volume"], ctx["amount"]
    score = combine(weighted_factor_tables(scfg["factor_weights"], codes, data_start, end, ctx),
                    scfg["factor_weights"])
    listed_ok = pd.DataFrame(
        {c: (score.index - listing_dates().get(c, score.index[0])).days >= scfg["min_list_days"] for c in score.columns},
        index=score.index,
    )
    liquid = amount.reindex_like(score) * 1000 >= scfg["min_amount"]
    mom5 = close / close.shift(5) - 1
    not_chase = ~(mom5.reindex_like(score) > scfg["chase_limit"])
    knife = scfg.get("falling_knife")
    not_knife = ~(mom5.reindex_like(score) < knife) if knife is not None else True
    score = score.where(listed_ok & liquid & not_chase & not_knife & volume.reindex_like(score).notna())
    return score, data_start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", choices=["momentum", "random", "real"], default="real")
    ap.add_argument("--pools", default="stable,short", help="逗号分隔，如 --pools short")
    ap.add_argument("--only-factor", default=None, help="短线池单因子回测：只用该因子，其余权重置零")
    ap.add_argument("--weights", default=None,
                    help='JSON，覆盖对应池的 factor_weights，如 \'{"neg_max_ret_20":1.0}\'（需配合 --pools）')
    ap.add_argument("--boards", default=None,
                    help="板块覆盖：逗号分隔 main,cyb,kcb,bse（覆盖 config 对应池的 boards）")
    ap.add_argument("--tag", default=None, help="报告文件名后缀（A/B 实验用，避免覆盖基线报告）")
    add_sync_args(ap)
    args = ap.parse_args()
    preflight(args, load_global().get("data"))  # 必须早于 init_qlib
    pools = set(args.pools.split(","))

    init_qlib()  # 默认 kernels=2（QLIB_KERNELS 环境变量可覆盖）
    cfg = load_config()
    if args.boards is not None:
        bl = [b.strip() for b in args.boards.split(",")]
        for p in pools & {"stable", "short"}:
            cfg[p]["boards"] = bl
    if args.weights:
        import json
        for p in pools & {"stable", "short"}:
            cfg[p]["factor_weights"].update(json.loads(args.weights))
    if args.only_factor:
        for k in cfg["short"]["factor_weights"]:
            cfg["short"]["factor_weights"][k] = 1.0 if k == args.only_factor else 0.0
        print(f"单因子模式: {args.only_factor}")
    end = str(latest_trade_date().date())
    all_lines = [f"# 回测报告（score={args.score}）", "", f"生成时间: {pd.Timestamp.now()}",
                 f"数据截至: {end}", ""]

    if args.score in ("momentum", "random"):
        codes = sorted(list_instruments("csi300", SMOKE_DATA_START, SMOKE_BT_END))
        from src.data.market import load_ohlcv
        close = load_ohlcv(codes, SMOKE_DATA_START, SMOKE_BT_END, ("close",))["close"].unstack(level="instrument").sort_index()
        score_daily = make_synth_scores(args.score, close)
        for pool, freq in (("stable", "monthly"), ("short", "weekly")):
            all_lines += run_pool(pool, freq, score_daily, SMOKE_DATA_START,
                                  SMOKE_BT_START, SMOKE_BT_END, cfg, "2020-12-31")
    else:
        if "stable" in pools:
            score_stable, ds = real_scores_stable(cfg, end)
            all_lines += run_pool("stable", "monthly", score_stable, ds,
                                  cfg["stable"]["backtest"]["start"], end, cfg,
                                  cfg["stable"]["backtest"]["in_sample_end"])
            del score_stable
        if "short" in pools:
            score_short, ds = real_scores_short(cfg, end)
            all_lines += run_pool("short", "weekly", score_short, ds,
                                  cfg["short"]["backtest"]["start"], end, cfg,
                                  cfg["short"]["backtest"]["in_sample_end"])

    suffix = f"_{args.tag}" if args.tag else ""
    out = REPORT_DIR / f"backtest_report{suffix}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
