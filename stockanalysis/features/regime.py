"""市场环境（择时）：逐日序列（index=交易日），全部只用当日及之前的数据。

    breadth20 / breadth60   当日在市股票中收盘价站上 MA20 / MA60 的比例
    limit_up_n / limit_down_n  当日涨停 / 跌停家数
    index_above_ma60        基准指数收盘 > MA60
    index_ret20             基准指数 20 日涨跌幅
    market_temp             长桥市场温度（0~100；口径由长桥定义，须经研发区检验）

state(ctx, rule) 把这些合成为"可开仓 / 不开仓"的布尔序列，规则在池 yaml 中声明，研发区检验后定稿。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cached(ctx: dict, key: str, fn):
    cache = ctx.setdefault("_regime", {})
    if key not in cache:
        cache[key] = fn()
    return cache[key]


def _listed(ctx: dict) -> pd.DataFrame:
    return ctx["membership"].reindex_like(ctx["close"]).fillna(False).astype(bool) & ctx["close"].notna()


def breadth(ctx: dict, n: int) -> pd.Series:
    def calc():
        c = ctx["close"]
        ma = c.rolling(n, min_periods=n).mean()
        ok = _listed(ctx) & ma.notna()
        return ((c > ma) & ok).sum(axis=1) / ok.sum(axis=1).replace(0, np.nan)
    return _cached(ctx, f"breadth{n}", calc)


def limit_counts(ctx: dict) -> tuple[pd.Series, pd.Series]:
    def calc():
        from ..selection.filters import limit_hits

        up, down = limit_hits(ctx["close"])
        ok = _listed(ctx)
        return (up & ok).sum(axis=1), (down & ok).sum(axis=1)
    return _cached(ctx, "limits", calc)


def index_close(ctx: dict, symbol: str) -> pd.Series:
    def calc():
        from ..data.market import load_wide

        start, end = str(pd.Timestamp(ctx["dates"][0]).date()), str(pd.Timestamp(ctx["dates"][-1]).date())
        w = load_wide([symbol], start, end, ("close",))["close"]
        return w[symbol].reindex(pd.DatetimeIndex(ctx["dates"])).astype(float)
    return _cached(ctx, f"index_{symbol}", calc)


def market_temp(ctx: dict) -> pd.Series:
    def calc():
        from ..data.longbridge_cli import market_temperature

        dates = pd.DatetimeIndex(ctx["dates"])
        try:
            mt = market_temperature(str(dates[0].date()), str(dates[-1].date()))
        except Exception as e:  # noqa: BLE001 - CLI 不可用时择时只用本地指标
            print(f"[warn] 市场温度不可用: {str(e)[:100]}")
            ctx.setdefault("_coverage", {})["market_temp"] = 0.0
            return pd.Series(np.nan, index=dates)
        return mt["temperature"].reindex(dates).ffill(limit=3)
    return _cached(ctx, "market_temp", calc)


def features(ctx: dict, benchmark: str) -> pd.DataFrame:
    """全部择时指标一张表。"""
    idx = index_close(ctx, benchmark)
    up, down = limit_counts(ctx)
    return pd.DataFrame({
        "breadth20": breadth(ctx, 20), "breadth60": breadth(ctx, 60),
        "limit_up_n": up, "limit_down_n": down,
        "index_above_ma60": (idx > idx.rolling(60, min_periods=60).mean()).astype(float),
        "index_ret20": idx / idx.shift(20) - 1,
        "market_temp": market_temp(ctx),
    })


def state(ctx: dict, benchmark: str, breadth_min: float | None = None, index_ma60: bool = False,
          temp_min: float | None = None, limit_down_max: int | None = None) -> pd.Series:
    """可开仓（True）的交易日：所有启用的条件同时满足；指标缺失时该条件视为满足。"""
    f = features(ctx, benchmark)
    ok = pd.Series(True, index=f.index)
    if breadth_min is not None:
        ok &= ~(f["breadth20"] < breadth_min)
    if index_ma60:
        ok &= ~(f["index_above_ma60"] == 0)
    if temp_min is not None:
        ok &= ~(f["market_temp"] < temp_min)
    if limit_down_max is not None:
        ok &= ~(f["limit_down_n"] > limit_down_max)
    return ok
