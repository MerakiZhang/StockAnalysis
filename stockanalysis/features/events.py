"""事件数据 → 逐日宽表（index=交易日, columns=代码），严格按"可见日"对齐（见 data/events.py 顶部）。

    margin_buy_ratio    融资买入额 / 当日成交额（T 日数据 T+1 可见 → 下移 1 个交易日）
    margin_bal_chg20    融资余额 20 日变化率（同上）
    lhb_jg_net          龙虎榜机构净买入额 / 当日市场成交额（T 日盘后公布，T 日可用；未上榜为 0）
    lhb_jg_net_recent5  近 5 日机构净买入占比之和
    gdhs_change         最近已披露的股东户数变化率（%，负值 = 筹码集中）
    reduce_recent90     近 90 天有大股东 / 股东减持公告（公告日次日起）
    increase_recent90   近 90 天有股东增持公告
    repurchase_recent90 近 90 天有回购方案（回购起始时间 = 方案公告日）
    forecast_score      最近 120 天内已公告的业绩预告：预增 / 扭亏 / 略增 = +1，续盈 = 0，预减 / 首亏 / 续亏 / 略减 = -1
    forecast_change     对应的净利润变动幅度（%）

数据缺失（未下载）时返回全 NaN / False，并在 ctx["_coverage"] 记录，研究报告据此提示。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POSITIVE_FORECAST = ("预增", "扭亏", "略增")
NEGATIVE_FORECAST = ("预减", "首亏", "续亏", "略减")


def _cached(ctx: dict, key: str, fn):
    cache = ctx.setdefault("_events", {})
    if key not in cache:
        cache[key] = fn()
    return cache[key]


def _empty(ctx: dict, value=np.nan) -> pd.DataFrame:
    b = ctx["close"]
    return pd.DataFrame(value, index=b.index, columns=b.columns)


def _missing(ctx: dict, name: str, value=np.nan) -> pd.DataFrame:
    ctx.setdefault("_coverage", {})[name] = 0.0
    return _empty(ctx, value)


def _load(dataset: str, start: str):
    from ..data.events import load_snapshot, load_units

    if dataset in ("ggcg", "repurchase"):
        return load_snapshot(dataset)
    return load_units(dataset, start)


def _start(ctx: dict, days: int = 400) -> str:
    return str((pd.Timestamp(ctx["dates"][0]) - pd.Timedelta(days=days)).date())


def _pivot_daily(ctx: dict, df: pd.DataFrame, value: str, agg: str = "sum") -> pd.DataFrame:
    dates = pd.DatetimeIndex(ctx["dates"])
    df = df[df["qlib_code"].isin(ctx["codes"])]
    w = df.pivot_table(index="date", columns="qlib_code", values=value, aggfunc=agg)
    return w.reindex(index=dates, columns=ctx["codes"])


def _recent_flag(ctx: dict, events: pd.DataFrame, days: int, lag_days: int = 1) -> pd.DataFrame:
    from .risk import _event_recent

    return _event_recent(ctx, events, days, lag_days)


# ------------------------------------------------------------------ 两融
def _margin(ctx: dict):
    def load():
        try:
            m = _load("margin", _start(ctx, 60))
        except FileNotFoundError:
            return None
        bal = _pivot_daily(ctx, m, "fin_balance", "last")
        buy = _pivot_daily(ctx, m, "fin_buy", "last")
        return bal.shift(1), buy.shift(1)   # T 日数据 T+1 可见
    return _cached(ctx, "margin", load)


def margin_buy_ratio(ctx: dict) -> pd.DataFrame:
    mg = _margin(ctx)
    if mg is None:
        return _missing(ctx, "margin")
    amount = (ctx["amount"] * 1000).shift(1)
    return mg[1] / amount.where(amount > 0)


def margin_bal_chg20(ctx: dict) -> pd.DataFrame:
    mg = _margin(ctx)
    if mg is None:
        return _missing(ctx, "margin")
    bal = mg[0]
    return bal / bal.shift(20) - 1


# ------------------------------------------------------------------ 龙虎榜
def lhb_jg_net(ctx: dict) -> pd.DataFrame:
    def calc():
        try:
            jg = _load("lhb_jg", _start(ctx, 40))
        except FileNotFoundError:
            return _missing(ctx, "lhb_jg")
        jg = jg.assign(ratio=jg["jg_net_buy"] / jg["market_amount"].where(jg["market_amount"] > 0))
        return _pivot_daily(ctx, jg, "ratio", "sum").fillna(0.0)
    return _cached(ctx, "lhb_jg_net", calc)


def lhb_jg_net_recent5(ctx: dict) -> pd.DataFrame:
    return _cached(ctx, "lhb_jg_net_recent5", lambda: lhb_jg_net(ctx).rolling(5, min_periods=1).sum())


# ------------------------------------------------------------------ 股东户数
def gdhs_change(ctx: dict) -> pd.DataFrame:
    def calc():
        from .fundamental import _latest_visible

        try:
            g = _load("gdhs", _start(ctx))
        except FileNotFoundError:
            return _missing(ctx, "gdhs")
        g = g.dropna(subset=["announce_date", "change_pct"]).copy()
        g["available_date"] = g["announce_date"] + pd.Timedelta(days=1)
        return _latest_visible(g, "change_pct", pd.DatetimeIndex(ctx["dates"]), list(ctx["codes"]))
    return _cached(ctx, "gdhs_change", calc)


# ------------------------------------------------------------------ 增减持 / 回购
def _ggcg(ctx: dict, direction: str, days: int) -> pd.DataFrame:
    try:
        g = _load("ggcg", _start(ctx))
    except FileNotFoundError:
        return _missing(ctx, "ggcg", False)
    ev = g[g["direction"] == direction].rename(columns={"announce_date": "date"})
    return _recent_flag(ctx, ev.dropna(subset=["date"]), days)


def reduce_recent90(ctx: dict) -> pd.DataFrame:
    return _cached(ctx, "reduce90", lambda: _ggcg(ctx, "减持", 90))


def increase_recent90(ctx: dict) -> pd.DataFrame:
    return _cached(ctx, "increase90", lambda: _ggcg(ctx, "增持", 90))


def repurchase_recent90(ctx: dict) -> pd.DataFrame:
    def calc():
        try:
            r = _load("repurchase", _start(ctx))
        except FileNotFoundError:
            return _missing(ctx, "repurchase", False)
        ev = r.rename(columns={"start": "date"}).dropna(subset=["date"])
        return _recent_flag(ctx, ev, 90)
    return _cached(ctx, "repurchase90", calc)


# ------------------------------------------------------------------ 业绩预告
def _forecast(ctx: dict):
    def calc():
        from ..data.fundamental import load_yjyg_all
        from .fundamental import _latest_visible

        try:
            fc = load_yjyg_all(pd.Timestamp(ctx["dates"][0]).year - 1)
        except FileNotFoundError:
            return None
        fc = fc[fc["indicator"].astype(str).str.contains("净利润")].copy()
        fc["score"] = np.where(fc["forecast_type"].isin(POSITIVE_FORECAST), 1.0,
                               np.where(fc["forecast_type"].isin(NEGATIVE_FORECAST), -1.0, 0.0))
        fc = fc.sort_values("announce_date").drop_duplicates(["qlib_code", "period"], keep="last")
        dates, codes = pd.DatetimeIndex(ctx["dates"]), list(ctx["codes"])
        score = _latest_visible(fc, "score", dates, codes)
        change = _latest_visible(fc, "change_pct", dates, codes)
        when = _latest_visible(fc.assign(t=fc["available_date"].astype("int64") / 1e9), "t", dates, codes)
        fresh = (dates.values.astype("int64")[:, None] / 1e9 - when.to_numpy()) <= 120 * 86400
        return score.where(fresh), change.where(fresh)
    return _cached(ctx, "forecast", calc)


def forecast_score(ctx: dict) -> pd.DataFrame:
    fc = _forecast(ctx)
    return _missing(ctx, "yjyg") if fc is None else fc[0]


def forecast_change(ctx: dict) -> pd.DataFrame:
    fc = _forecast(ctx)
    return _missing(ctx, "yjyg") if fc is None else fc[1]


EVENT_FEATURES = {
    "margin_buy_ratio": margin_buy_ratio, "margin_bal_chg20": margin_bal_chg20,
    "lhb_jg_net": lhb_jg_net, "lhb_jg_net_recent5": lhb_jg_net_recent5, "gdhs_change": gdhs_change,
    "reduce_recent90": reduce_recent90, "increase_recent90": increase_recent90,
    "repurchase_recent90": repurchase_recent90, "forecast_score": forecast_score, "forecast_change": forecast_change,
}


def get(ctx: dict, name: str) -> pd.DataFrame:
    return EVENT_FEATURES[name](ctx)
