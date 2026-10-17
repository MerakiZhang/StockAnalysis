"""烂股识别：逐日布尔面板（index=交易日, columns=代码，True = 命中风险），全部按当时可见的信息判定。

    st                 历史 ST / *ST：baostock isST；无 baostock 数据的股票用涨跌停幅度推断（近似，见 st_by_price_limit）
    raw_close          不复权收盘价 = qlib 复权价 / 复权因子
    market_cap         总市值估算 = 不复权价 × 总股本（总股本 ≈ 累计净利润 / 累计 EPS，取最近可见报告期）
    consecutive_loss   最近两个已披露年报归母净利润均为负
    negative_equity    最近已披露报告期每股净资产 < 0
    small_revenue_loss 最近已披露年报营收 < 阈值且净利润为负（退市新规口径）
    risk_notice        近 N 天有立案调查 / 行政处罚 / 退市风险等公告（标题关键词，公告日次日起可见）
    release_ahead      未来 N 个交易日内限售解禁占流通市值比例合计 > 阈值（解禁日程事先公开）

注：业绩报表里的股票简称是**当前**名称（不随历史变化），不能用来判定历史 ST。
数据缺失时一律视为"未命中"（不因缺数据误杀），覆盖率在研究报告中单独列出。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..data import fundamental as fd
from ..data.universe import board_of

RISK_TITLE = re.compile("立案|行政处罚|处罚事先告知|退市风险警示|终止上市|暂停上市|重大违法|无法表示意见|否定意见|"
                        "被实施其他风险警示|实施风险警示")
RISK_TITLE_EXCLUDE = re.compile("撤销|解除|结案|进展|终止.*调查")  # 撤销风险警示 / 立案进展等不新增风险


def _cached(ctx: dict, key: str, fn):
    cache = ctx.setdefault("_cache", {})
    if key not in cache:
        cache[key] = fn()
    return cache[key]


def _frame(ctx: dict, value=False) -> pd.DataFrame:
    b = ctx["close"]
    return pd.DataFrame(value, index=b.index, columns=b.columns)


# ------------------------------------------------------------------ 价格 / 市值
def raw_close(ctx: dict) -> pd.DataFrame:
    return _cached(ctx, "raw_close", lambda: ctx["close"] / ctx["factor"])


def _yjbb(ctx: dict) -> pd.DataFrame:
    def load():
        yj = fd.load_yjbb_all(2012)
        yj = yj[yj["qlib_code"].isin(ctx["codes"])].copy()
        for c in ("eps", "bps", "netprofit", "revenue"):
            yj[c] = pd.to_numeric(yj[c], errors="coerce")
        return yj
    return _cached(ctx, "yjbb", load)


def _visible(ctx: dict, df: pd.DataFrame, field: str) -> pd.DataFrame:
    from .fundamental import _latest_visible

    return _latest_visible(df, field, pd.DatetimeIndex(ctx["dates"]), list(ctx["codes"]))


def market_cap(ctx: dict) -> pd.DataFrame:
    """总市值（元）估算。|EPS| < 0.02 时股本推算误差大，记为缺失。"""
    def calc():
        yj = _yjbb(ctx)
        yj = yj[(yj["eps"].abs() >= 0.02) & yj["netprofit"].notna()].copy()
        yj["shares"] = yj["netprofit"] / yj["eps"]
        shares = _visible(ctx, yj, "shares")
        return raw_close(ctx) * shares.reindex_like(ctx["close"])
    return _cached(ctx, "market_cap", calc)


# ------------------------------------------------------------------ ST
def st_by_price_limit(ctx: dict, window: int = 60) -> pd.DataFrame:
    """无 baostock 数据时的 ST 近似：主板股票近 window 日内触及过 ±5% 附近的涨跌停（收盘 = 最高 / 最低），
    且没有任何一天涨跌幅超过 5.5%。只适用于主板（创业板 / 科创板 ST 也是 20%，无法区分）。"""
    close, high, low = ctx["close"], ctx["high"], ctx["low"]
    ret = close.pct_change(fill_method=None)
    at_limit = ((ret.abs() - 0.05).abs() <= 0.003) & ((close >= high * 0.999) | (close <= low * 1.001))
    big = ret.abs() > 0.055
    hit = at_limit.astype(np.float32).rolling(window, min_periods=1).max() > 0
    clean = big.astype(np.float32).rolling(window, min_periods=1).max() == 0
    main = pd.Series({c: board_of(c) == "main" for c in close.columns})
    return (hit & clean) & main.reindex(close.columns).to_numpy()[None, :]


def st(ctx: dict) -> pd.DataFrame:
    def calc():
        from ..data.baostock_data import load_daily

        bs = load_daily("is_st", list(ctx["codes"]), ctx["dates"])
        has = bs.notna().any()
        out = st_by_price_limit(ctx).copy()
        cols = has[has].index
        if len(cols):
            out[cols] = bs[cols].ffill().fillna(0).astype(bool)
        ctx.setdefault("_coverage", {})["st_baostock"] = float(has.mean()) if len(has) else 0.0
        return out
    return _cached(ctx, "st", calc)


# ------------------------------------------------------------------ 财务
def _annual(ctx: dict) -> pd.DataFrame:
    yj = _yjbb(ctx)
    return yj[yj["period"].str.endswith("1231")].sort_values(["qlib_code", "period"])


def consecutive_loss(ctx: dict, years: int = 2) -> pd.DataFrame:
    def calc():
        an = _annual(ctx).copy()
        an["year"] = an["period"].str[:4].astype(int)
        an["loss"] = (an["netprofit"] < 0).astype(float)
        g = an.groupby("qlib_code")
        n_loss = an["loss"].copy()
        for k in range(1, years):
            prev = g["loss"].shift(k)
            consecutive = g["year"].shift(k) == an["year"] - k
            n_loss = n_loss + prev.where(consecutive, 0).fillna(0)
        an["flag"] = (n_loss >= years).astype(float)
        return _visible(ctx, an, "flag").fillna(0) > 0.5
    return _cached(ctx, f"consecutive_loss_{years}", calc)


def negative_equity(ctx: dict) -> pd.DataFrame:
    return _cached(ctx, "negative_equity", lambda: _visible(ctx, _yjbb(ctx), "bps") < 0)


def small_revenue_loss(ctx: dict, revenue: float = 3e8) -> pd.DataFrame:
    def calc():
        an = _annual(ctx).copy()
        an["flag"] = ((an["revenue"] < revenue) & (an["netprofit"] < 0)).astype(float)
        return _visible(ctx, an, "flag").fillna(0) > 0.5
    return _cached(ctx, f"small_revenue_loss_{revenue}", calc)


# ------------------------------------------------------------------ 公告 / 解禁
def risk_events(start: str | None = None) -> pd.DataFrame:
    """风险公告：(qlib_code, date, title)。"""
    from ..data.events import load_units

    nt = load_units("notice", start)
    hit = nt["title"].str.contains(RISK_TITLE, na=False) & ~nt["title"].str.contains(RISK_TITLE_EXCLUDE, na=False)
    return nt.loc[hit, ["qlib_code", "date", "title"]].dropna(subset=["date"])


def _event_recent(ctx: dict, events: pd.DataFrame, days: int, lag_days: int = 1) -> pd.DataFrame:
    """事件日 + lag 起，days 个自然日内为 True。"""
    out = _frame(ctx, False)
    dates = pd.DatetimeIndex(ctx["dates"])
    ev = events[events["qlib_code"].isin(out.columns)]
    if ev.empty:
        return out
    vals = out.to_numpy()
    col = {c: i for i, c in enumerate(out.columns)}
    for code, d in zip(ev["qlib_code"], pd.to_datetime(ev["date"])):
        s = d + pd.Timedelta(days=lag_days)
        lo, hi = dates.searchsorted(s), dates.searchsorted(s + pd.Timedelta(days=days))
        vals[lo:hi, col[code]] = True
    return pd.DataFrame(vals, index=out.index, columns=out.columns)


def risk_notice(ctx: dict, days: int = 365) -> pd.DataFrame:
    def calc():
        start = str((pd.Timestamp(ctx["dates"][0]) - pd.Timedelta(days=days)).date())
        try:
            ev = risk_events(start)
        except FileNotFoundError:
            ctx.setdefault("_coverage", {})["risk_notice"] = 0.0
            return _frame(ctx, False)
        return _event_recent(ctx, ev, days)
    return _cached(ctx, f"risk_notice_{days}", calc)


def release_ahead(ctx: dict, days: int = 20, ratio: float = 0.05) -> pd.DataFrame:
    """未来约 days 个交易日（按 days × 1.5 个自然日近似，不含当日）内解禁占流通市值比例之和 > ratio。

    用自然日窗口而非交易日历：实盘最后一天也要看到日历之外的未来解禁日程。
    """
    def calc():
        from ..data.events import load_units

        try:
            rl = load_units("release")
        except FileNotFoundError:
            ctx.setdefault("_coverage", {})["release"] = 0.0
            return _frame(ctx, False)
        out = _frame(ctx, 0.0).astype(np.float64)
        dates = pd.DatetimeIndex(ctx["dates"])
        rl = rl[rl["qlib_code"].isin(out.columns) & rl["date"].notna() & rl["float_ratio"].notna()]
        vals = out.to_numpy()
        col = {c: i for i, c in enumerate(out.columns)}
        span = pd.Timedelta(days=int(round(days * 1.5)))
        for code, d, r in zip(rl["qlib_code"], pd.to_datetime(rl["date"]), rl["float_ratio"]):
            lo, hi = dates.searchsorted(d - span), dates.searchsorted(d)   # lo <= t < d
            vals[lo:hi, col[code]] += float(r)
        return pd.DataFrame(vals > ratio, index=out.index, columns=out.columns)
    return _cached(ctx, f"release_ahead_{days}_{ratio}", calc)
