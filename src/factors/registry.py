"""因子注册表：config factor_weights 的键 -> 因子构建函数。

builder 签名：(codes, start, end, ctx) -> pd.DataFrame
    返回宽表（index=交易日, columns=qlib 代码），方向统一为"越大越好"。
ctx: market_ctx() 生成的上下文字典，含 close/volume/amount/open/high/low
    行情宽表（另有 "wide" 键为原始 {field: 宽表}），供需要行情的因子
    （如 dividend_yield 用 close 做分母）复用，避免重复加载。

新增因子 = 在 technical.py / fundamental.py 写函数 + 在 FACTORS 注册一行
+ 在 config/selector.yaml 配置权重（权重 0 的因子不会被计算）。
"""
from __future__ import annotations

import pandas as pd

from . import fundamental, technical
from .base import load_wide

CTX_FIELDS = ("open", "high", "low", "close", "volume", "amount")


def market_ctx(
    codes: list[str],
    start: str,
    end: str,
    fields: tuple[str, ...] = CTX_FIELDS,
) -> dict[str, pd.DataFrame]:
    """因子计算上下文：{field: 宽表} + "wide" 键（原始宽表字典）。"""
    wide = load_wide(codes, start, end, fields)
    ctx = dict(wide)
    ctx["wide"] = wide
    return ctx


FACTORS = {
    # ---------------- 技术面（A股反转/低波方向已由 IC 检验实证，取反后越大越好） ----------------
    "neg_mom_20": lambda c, s, e, ctx: -technical.mom_n(ctx["close"], 20),
    "neg_risk_adj_mom": lambda c, s, e, ctx: -technical.risk_adj_momentum(ctx["close"]),
    "neg_vol_20": lambda c, s, e, ctx: -technical.vol_n(ctx["close"], 20),
    "neg_max_ret_20": lambda c, s, e, ctx: -technical.max_ret_n(ctx["close"], 20),
    "neg_mom_5d": lambda c, s, e, ctx: -technical.mom_5d(ctx["close"]),
    "neg_mom_10d": lambda c, s, e, ctx: -technical.mom_10d(ctx["close"]),
    "neg_volume_ratio": lambda c, s, e, ctx: -technical.volume_ratio(ctx["volume"]),
    "neg_turnover_surge": lambda c, s, e, ctx: -technical.turnover_surge(ctx["volume"]),
    "neg_limit_up_count": lambda c, s, e, ctx: -technical.limit_up_count(ctx["close"]),
    # ---------------- 质量/成长 ----------------
    "roe": lambda c, s, e, ctx: fundamental.roe(c, s, e),
    "gross_margin": lambda c, s, e, ctx: fundamental.gross_margin(c, s, e),
    "revenue_yoy": lambda c, s, e, ctx: fundamental.revenue_yoy(c, s, e),
    "netprofit_yoy": lambda c, s, e, ctx: fundamental.netprofit_yoy(c, s, e),
    "ocf_to_netprofit": lambda c, s, e, ctx: fundamental.ocf_to_netprofit(c, s, e),
    # ---------------- 估值/股息 ----------------
    "neg_pe_ttm": lambda c, s, e, ctx: fundamental.neg_pe_ttm(c, s, e),
    "neg_pb": lambda c, s, e, ctx: fundamental.neg_pb(c, s, e),
    "neg_pb_pct": lambda c, s, e, ctx: fundamental.neg_pb_percentile(c, s, e),
    "dividend_yield": lambda c, s, e, ctx: fundamental.dividend_yield(c, s, e, ctx["close"]),
    "dividend_continuity": lambda c, s, e, ctx: fundamental.dividend_continuity(c, s, e),
}


def build_factors(
    names: list[str],
    codes: list[str],
    start: str,
    end: str,
    ctx: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """按名称从注册表构建因子宽表。ctx 缺省时自动加载行情。"""
    unknown = [n for n in names if n not in FACTORS]
    if unknown:
        raise KeyError(f"未注册的因子: {unknown}（请在 src/factors/registry.py 的 FACTORS 中注册）")
    if ctx is None:
        ctx = market_ctx(codes, start, end)
    return {n: FACTORS[n](codes, start, end, ctx) for n in names}


def weighted_factor_tables(
    weights: dict[str, float],
    codes: list[str],
    start: str,
    end: str,
    ctx: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """按 factor_weights 构建参与打分的因子（权重为 0 的因子不计算）。"""
    names = [k for k, w in weights.items() if float(w) != 0.0]
    return build_factors(names, codes, start, end, ctx)
