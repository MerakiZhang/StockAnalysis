"""技术面与资金面（量价近似）指标：宽表（index=交易日, columns=代码），按需计算并缓存在 ctx 里。

价格用 qlib 复权价（送转前后连续）；量能一律用成交额 amount（qlib volume 跨股票 / 跨时间口径不一致，不用于比较）。
TA-Lib 类指标（MACD / KDJ / RSI / BOLL / ATR / ADX / CCI / MFI / OBV）调用 akquant.talib，逐股计算：
停牌日（NaN）先剔除再算，算完放回原日期，避免 NaN 让 EMA 类指标从此全部失效。

    get(ctx, name) -> 宽表        名称见 INDICATORS（如 "ma20"、"macd_hist"、"cmf20"）
资金面近似（没有真实大单数据）：
    clv          收盘位置 ((C-L)-(H-C))/(H-L)，+1 收在最高、-1 收在最低
    cmf20        蔡金资金流 Σ(clv × 成交额) / Σ成交额
    mfi14        资金流量指数（典型价 × 成交额，分上涨 / 下跌日）
    obv_slope20  按涨跌累加成交额的 OBV，20 日回归斜率 / 20 日均额
    amt_ratio5   5 日均额 / 前 20 日均额（放量 > 1，缩量 < 1）
    close_vwap   收盘价 / 当日均价 - 1（尾盘抢筹 > 0）
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _cached(ctx: dict, key: str, fn):
    cache = ctx.setdefault("_tech", {})
    if key not in cache:
        cache[key] = fn()
    return cache[key]


def _per_column(fn, *wides: pd.DataFrame, n_out: int = 1):
    """逐股调用 talib 函数（输入为若干同形宽表），跳过 NaN 行；返回宽表或宽表元组。"""
    base = wides[0]
    outs = [np.full(base.shape, np.nan, dtype=np.float64) for _ in range(n_out)]
    arrs = [w.to_numpy(dtype=np.float64) for w in wides]
    for j in range(base.shape[1]):
        cols = [a[:, j] for a in arrs]
        ok = np.all([np.isfinite(c) for c in cols], axis=0)
        if ok.sum() < 30:
            continue
        res = fn(*[c[ok] for c in cols])
        res = res if isinstance(res, tuple) else (res,)
        for k in range(n_out):
            outs[k][ok, j] = np.asarray(res[k], dtype=np.float64)
    frames = tuple(pd.DataFrame(o, index=base.index, columns=base.columns) for o in outs)
    return frames if n_out > 1 else frames[0]


def _talib():
    import akquant.talib as ta

    return ta


def _rolling_slope(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 日线性回归斜率（对时间下标 0..n-1）。"""
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    w = x / (x ** 2).sum()
    vals = df.to_numpy(dtype=np.float64)
    out = np.full(vals.shape, np.nan)
    for i in range(n - 1, len(vals)):
        out[i] = w @ vals[i - n + 1:i + 1]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


# ------------------------------------------------------------------ 计算函数（name 解析见 get）
def _ma(ctx, n):
    return ctx["close"].rolling(n, min_periods=n).mean()


def _amt_ma(ctx, n):
    return (ctx["amount"] * 1000).rolling(n, min_periods=max(1, n // 2)).mean()


def _compute(ctx: dict, name: str) -> pd.DataFrame:
    c, h, l, a = ctx["close"], ctx["high"], ctx["low"], ctx["amount"] * 1000
    ta = _talib()
    m = re.fullmatch(r"(ma|ret|vol|high|low|amt_ma|rsi|atr|natr|cci|adx|mfi|cmf|obv_slope)(\d+)", name)
    if m:
        kind, n = m.group(1), int(m.group(2))
        if kind == "ma":
            return _ma(ctx, n)
        if kind == "ret":
            return c / c.shift(n) - 1
        if kind == "vol":
            return c.pct_change(fill_method=None).rolling(n, min_periods=n).std()
        if kind == "high":
            return h.rolling(n, min_periods=n).max()
        if kind == "low":
            return l.rolling(n, min_periods=n).min()
        if kind == "amt_ma":
            return _amt_ma(ctx, n)
        if kind == "rsi":
            return _per_column(lambda x: ta.RSI(x, n), c)
        if kind == "atr":
            return _per_column(lambda hh, ll, cc: ta.ATR(hh, ll, cc, n), h, l, c)
        if kind == "natr":
            return get(ctx, f"atr{n}") / c
        if kind == "cci":
            return _per_column(lambda hh, ll, cc: ta.CCI(hh, ll, cc, n), h, l, c)
        if kind == "adx":
            return _per_column(lambda hh, ll, cc: ta.ADX(hh, ll, cc, n), h, l, c)
        if kind == "mfi":
            return _per_column(lambda hh, ll, cc, aa: ta.MFI(hh, ll, cc, aa, n), h, l, c, a)
        if kind == "cmf":
            clv = get(ctx, "clv")
            return (clv * a).rolling(n, min_periods=n).sum() / a.rolling(n, min_periods=n).sum()
        if kind == "obv_slope":
            return _rolling_slope(get(ctx, "obv"), n) / _amt_ma(ctx, n)
    if name in ("macd_dif", "macd_dea", "macd_hist"):
        dif, dea, hist = _per_column(lambda x: ta.MACD(x, 12, 26, 9), c, n_out=3)
        ctx["_tech"].update(macd_dif=dif, macd_dea=dea, macd_hist=hist)
        return ctx["_tech"][name]
    if name in ("kdj_k", "kdj_d", "kdj_j"):
        k, d = _per_column(lambda hh, ll, cc: ta.STOCH(hh, ll, cc, 9, 3, 0, 3, 0), h, l, c, n_out=2)
        ctx["_tech"].update(kdj_k=k, kdj_d=d, kdj_j=3 * k - 2 * d)
        return ctx["_tech"][name]
    if name in ("boll_up", "boll_mid", "boll_low"):
        up, mid, lo = _per_column(lambda x: ta.BBANDS(x, 20, 2.0, 2.0), c, n_out=3)
        ctx["_tech"].update(boll_up=up, boll_mid=mid, boll_low=lo)
        return ctx["_tech"][name]
    if name == "clv":
        rng = (h - l).where(h > l)
        return (((c - l) - (h - c)) / rng).fillna(0.0).where(c.notna())
    if name == "obv":
        sign = np.sign(c.diff())
        return (sign * a).fillna(0).cumsum().where(c.notna())
    if name == "amt_ratio5":
        return _amt_ma(ctx, 5) / _amt_ma(ctx, 20).shift(5)
    if name == "amt_ratio1":
        return a / _amt_ma(ctx, 20).shift(1)
    if name == "close_vwap":
        return c / ctx["vwap"] - 1
    if name == "mom_120_20":
        return c.shift(20) / c.shift(120) - 1
    if name == "limit_up_n20":
        from ..selection.filters import limit_hits

        return limit_hits(c)[0].astype(np.float32).rolling(20, min_periods=1).sum()
    raise KeyError(f"未知指标 {name}")


def get(ctx: dict, name: str) -> pd.DataFrame:
    """按名称取指标宽表（首次计算后缓存）。"""
    cache = ctx.setdefault("_tech", {})
    if name not in cache:
        cache[name] = _compute(ctx, name)
    return cache[name]


# ------------------------------------------------------------------ 组合条件辅助
def cross_up(a: pd.DataFrame, b) -> pd.DataFrame:
    """a 今日上穿 b（昨日 a <= b，今日 a > b）。b 可为宽表或常数。"""
    b_prev = b.shift(1) if isinstance(b, pd.DataFrame) else b
    return (a > b) & (a.shift(1) <= b_prev)


def cross_down(a: pd.DataFrame, b) -> pd.DataFrame:
    b_prev = b.shift(1) if isinstance(b, pd.DataFrame) else b
    return (a < b) & (a.shift(1) >= b_prev)
