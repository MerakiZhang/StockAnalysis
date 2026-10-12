"""单因子有效性检验：IC/RankIC、分层收益、因子相关性。

用法：
    python scripts/run_factor_check.py --pool stable [--start 2018-01-01] [--end 2025-12-31]
    python scripts/run_factor_check.py --pool short  [--start 2020-01-01]

产出：data/reports/factor_report.md（stable）/ factor_report_short.md（short）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QLIB_KERNELS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import eval as fe
from src.config import REPORT_DIR
from src.data.market import latest_trade_date, list_instruments
from src.factors import fundamental, technical
from src.factors.registry import build_factors, market_ctx
from src.selection.universe import is_stock, listing_dates


def build_stable_factors(codes: list[str], start: str, end: str,
                         ctx: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """稳健池研究因子集：注册表内因子 + 未入池的研究候选。"""
    close = ctx["close"]
    print("计算基本面因子 ...", flush=True)
    factors = build_factors(
        ["roe", "gross_margin", "revenue_yoy", "netprofit_yoy",
         "dividend_yield", "dividend_continuity"],
        codes, start, end, ctx,
    )
    try:
        factors["ocf_to_netprofit"] = fundamental.ocf_to_netprofit(codes, start, end)
    except Exception as e:  # noqa: BLE001 - xjll 缓存不全时跳过
        print(f"盈利质量因子跳过: {e}", flush=True)
    try:
        factors.update(build_factors(["neg_pe_ttm", "neg_pb", "neg_pb_pct"], codes, start, end, ctx))
        factors["neg_pe_pct"] = fundamental.neg_pe_ttm_percentile(codes, start, end)
    except Exception as e:  # noqa: BLE001 - 估值缓存不全时跳过
        print(f"估值因子跳过: {e}", flush=True)
    print("计算技术面因子 ...", flush=True)
    factors["risk_adj_momentum"] = technical.risk_adj_momentum(close)
    factors["mom_20"] = technical.mom_n(close, 20)
    factors["neg_vol_20"] = -technical.vol_n(close, 20)  # 低波动为优，取负统一方向
    factors["dist_high_250"] = technical.dist_high_n(close, 250)
    factors["neg_max_ret_20"] = -technical.max_ret_n(close, 20)
    factors["neg_downside_vol_20"] = -technical.downside_vol_n(close, 20)
    return factors


def check_stable(args: argparse.Namespace) -> None:
    end = args.end or str(latest_trade_date().date())
    codes = list(list_instruments(args.universe, start=args.start, end=end))
    print(f"universe={args.universe} codes={len(codes)} [{args.start} ~ {end}]", flush=True)

    print("加载行情 ...", flush=True)
    ctx = market_ctx(codes, args.start, end, ("close", "volume", "amount"))
    close = ctx["close"]
    factors = build_stable_factors(codes, args.start, end, ctx)
    fwd = fe.forward_returns(close, args.fwd_days)

    lines: list[str] = [
        "# 单因子检验报告",
        "",
        f"- 范围: {args.universe}（{len(codes)} 只），{args.start} ~ {end}",
        f"- 未来收益窗口: {args.fwd_days} 个交易日",
        "",
        "## IC 汇总",
        "",
        "| 因子 | mean IC | ICIR | IC>0 占比 | 近250日IC | 分层收益(1→5组) |",
        "|------|---------|------|-----------|-----------|------------------|",
    ]
    for name, f in factors.items():
        print(f"IC: {name}", flush=True)
        ic = fe.daily_rank_ic(f, fwd)
        s = fe.summarize_ic(ic)
        rolling = fe.rolling_rank_ic(f, fwd)
        s["recent_ic"] = float(rolling.iloc[-1]) if len(rolling) else float("nan")
        q = fe.quantile_returns(f, fwd)["mean_fwd_return"]
        qstr = " / ".join(f"{v * 100:.2f}%" for v in q.values)
        lines.append(
            f"| {name} | {s['mean_ic']:.4f} | {s['icir']:.3f} | {s['positive_rate']:.2%} | {s['recent_ic']:.4f} | {qstr} |"
        )

    lines += ["", "## 因子相关性（日均截面相关）", ""]
    print("因子相关性 ...", flush=True)
    corr = fe.factor_correlation(factors)
    lines.append(corr.round(2).to_markdown())

    lines += [
        "",
        "## 判读规则",
        "",
        "- |mean IC| < 0.02 的因子考虑剔除",
        "- IC 持续为负的因子（如动量类在 A 股可能反转为负）应取反后使用",
        "- 相关性 > 0.7 的因子对只保留 IC 较高者",
        "- 分层收益应单调递增（越大越好的方向下）",
    ]

    out = REPORT_DIR / "factor_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {out}", flush=True)


def check_short(args: argparse.Namespace) -> None:
    """短线因子 IC 检验（默认 5 日前向收益，全市场个股）。"""
    end = args.end or str(latest_trade_date().date())
    codes = [c for c in list_instruments("all", start=args.start, end=end) if is_stock(c)]
    print(f"universe: {len(codes)} 只 [{args.start} ~ {end}]", flush=True)

    ctx = market_ctx(codes, args.start, end)
    close, volume, amount, high = ctx["close"], ctx["volume"], ctx["amount"], ctx["high"]
    open_, low = ctx["open"], ctx["low"]

    factors = {
        "mom_5d": technical.mom_5d(close),
        "mom_10d": technical.mom_10d(close),
        "neg_mom_5d": -technical.mom_5d(close),      # 反转方向
        "neg_mom_10d": -technical.mom_10d(close),
        "volume_ratio": technical.volume_ratio(volume),
        "neg_volume_ratio": -technical.volume_ratio(volume),  # 缩量方向
        "breakout_20d": technical.breakout_20d(close, high),
        "turnover_surge": technical.turnover_surge(volume),
        "neg_turnover_surge": -technical.turnover_surge(volume),
        "limit_up_count": technical.limit_up_count(close),
        "neg_limit_up_count": -technical.limit_up_count(close),
        # 第二轮候选因子
        "dist_from_high_20d": technical.dist_from_high(close, 20),
        "neg_price_position_60d": -technical.price_position(close, 60),
        "lower_shadow": technical.lower_shadow(open_, close, low),
        "neg_gap": -technical.gap(open_, close),
        "neg_intraday_ret": -technical.intraday_ret(open_, close),
        "neg_range_5d": -technical.range_n(high, low, close, 5),
        "neg_volume_cv_20": -technical.volume_cv(volume, 20),
        "max_drawdown_5d": technical.max_drawdown_n(close, 5),
    }

    # 与选股一致的静态过滤：上市满180天 + 流动性（只用掩膜影响 IC 样本）
    listed_ok = pd.DataFrame(
        {c: (close.index - listing_dates().get(c, close.index[0])).days >= 180 for c in close.columns},
        index=close.index,
    )
    liquid = amount * 1000 >= 1e8
    mask = listed_ok & liquid

    fwd = fe.forward_returns(close, args.fwd_days)
    fwd = fwd.where(mask)  # 不满足过滤条件的样本不计入 IC

    lines = [
        "# 短线因子 IC 检验报告", "",
        f"- 范围: 全市场个股 {len(codes)} 只（上市>180天且日成交额>1亿），{args.start} ~ {end}",
        f"- 未来收益窗口: {args.fwd_days} 个交易日", "",
        "| 因子 | mean IC | ICIR | IC>0 占比 | 分层收益(1→5组) |",
        "|------|---------|------|-----------|------------------|",
    ]
    for name, f in factors.items():
        print(f"IC: {name}", flush=True)
        f_masked = f.where(mask)
        ic = fe.daily_rank_ic(f_masked, fwd)
        s = fe.summarize_ic(ic)
        q = fe.quantile_returns(f_masked, fwd)["mean_fwd_return"]
        qstr = " / ".join(f"{v * 100:.2f}%" for v in q.values)
        lines.append(f"| {name} | {s['mean_ic']:.4f} | {s['icir']:.3f} | {s['positive_rate']:.2%} | {qstr} |")

    out = REPORT_DIR / "factor_report_short.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告: {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["stable", "short"], default="stable",
                    help="stable=稳健池因子（csi800，fwd 20日）；short=短线因子（全市场，fwd 5日）")
    ap.add_argument("--start", default=None, help="默认 stable=2018-01-01 / short=2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--universe", default="csi800", help="仅 stable 池有效")
    ap.add_argument("--fwd-days", type=int, default=None, help="默认 stable=20 / short=5")
    args = ap.parse_args()

    if args.pool == "stable":
        args.start = args.start or "2018-01-01"
        args.fwd_days = args.fwd_days or 20
        check_stable(args)
    else:
        args.start = args.start or "2020-01-01"
        args.fwd_days = args.fwd_days or 5
        check_short(args)


if __name__ == "__main__":
    main()
