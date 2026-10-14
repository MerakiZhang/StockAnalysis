"""过滤规则注册表 FILTERS。

每条规则：`(ctx, **params) -> DataFrame[bool]`（index=交易日, columns=代码，True=通过），
回测用整张面板，选股取最后一行——同一实现两处共用。

scope（注册时声明）：
    exit  —— 持有规则：不通过则分数置 NaN，已持有的强制调出
    entry —— 准入规则：只限制新买入，已持有的不因此被卖出

ctx（signal.build_ctx 生成）：
    close/volume/amount/... 行情宽表（float32，已对齐交易日历）
    membership              当日是否为指数成分
    st_codes                当前 ST 集合（仅实盘最新一天有值，历史回测为空集）

新增规则 = 在本文件写一个 @FILTERS.register(name, scope=...) 函数 + 池 yaml 的 filters 段声明。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import Registry
from .universe import board_of, listing_dates

FILTERS = Registry("filter", skip=1)
SCOPES = ("entry", "exit")

CYB_20PCT_FROM = pd.Timestamp("2020-08-24")  # 创业板注册制，涨跌幅 10% → 20%


def scope_of(name: str) -> str:
    return FILTERS.meta(name)["scope"]


def _base(ctx: dict) -> pd.DataFrame:
    return ctx["close"]


def _const(ctx: dict, value: bool) -> pd.DataFrame:
    b = _base(ctx)
    return pd.DataFrame(value, index=b.index, columns=b.columns)


def _by_column(ctx: dict, flags: dict[str, bool] | pd.Series) -> pd.DataFrame:
    b = _base(ctx)
    row = pd.Series(flags).reindex(b.columns).fillna(False).astype(bool).to_numpy()
    return pd.DataFrame(np.broadcast_to(row, b.shape).copy(), index=b.index, columns=b.columns)


# ------------------------------------------------------------------ exit
@FILTERS.register("in_universe", scope="exit")
def in_universe(ctx: dict) -> pd.DataFrame:
    """当日为指数成分（强制规则）。"""
    b = _base(ctx)
    return ctx["membership"].reindex(index=b.index, columns=b.columns).fillna(False).astype(bool)


@FILTERS.register("exclude_st", scope="exit")
def exclude_st(ctx: dict) -> pd.DataFrame:
    """剔除 ST。只作用于最后一天：ST 只有当前快照，套用到历史会形成前视偏差。"""
    ok = _const(ctx, True)
    st = ctx.get("st_codes") or set()
    if st and len(ok):
        cols = ok.columns.isin(list(st))
        ok.iloc[-1, cols] = False
    return ok


# ------------------------------------------------------------------ entry
@FILTERS.register("boards", scope="entry")
def boards(ctx: dict, boards: list[str]) -> pd.DataFrame:
    """板块资格：main=主板 cyb=创业板 kcb=科创板 bse=北交所。"""
    allowed = set(boards)
    return _by_column(ctx, {c: board_of(c) in allowed for c in _base(ctx).columns})


@FILTERS.register("min_list_days", scope="entry")
def min_list_days(ctx: dict, days: int) -> pd.DataFrame:
    """上市满 N 个自然日（上市日取 qlib all.txt 起始日，为近似）。"""
    b = _base(ctx)
    ld = listing_dates()
    first = pd.Series({c: ld.get(c, pd.Timestamp("1990-01-01")) for c in b.columns})
    age = (b.index.values[:, None] - first.reindex(b.columns).values[None, :]).astype("timedelta64[D]").astype(int)
    return pd.DataFrame(age >= int(days), index=b.index, columns=b.columns)


@FILTERS.register("exclude_suspend", scope="entry")
def exclude_suspend(ctx: dict, days: int = 20) -> pd.DataFrame:
    """近 N 个交易日（含当日）有停牌（成交量 NaN 或 0）则不准入。"""
    vol = ctx["volume"]
    bad = (vol.isna() | vol.eq(0)).astype(np.float32)
    return ~(bad.rolling(int(days), min_periods=1).max() > 0)


@FILTERS.register("exclude_limit", scope="entry")
def exclude_limit(ctx: dict, tol: float = 0.005) -> pd.DataFrame:
    """当日涨停（收盘涨幅 ≥ 涨停幅度 - tol）不准入：次日开盘前无法确认能否买进。

    幅度：主板 10%、科创板 20%、创业板 2020-08-24 起 20%（之前 10%）、北交所 30%。
    """
    close = _base(ctx)
    ret = close.pct_change(fill_method=None)
    thr = pd.DataFrame(0.10, index=close.index, columns=close.columns, dtype=np.float32)
    for c in close.columns:
        b = board_of(c)
        if b == "kcb":
            thr[c] = 0.20
        elif b == "bse":
            thr[c] = 0.30
        elif b == "cyb":
            thr[c] = np.where(close.index >= CYB_20PCT_FROM, 0.20, 0.10)
    return ~(ret >= thr - tol).fillna(False)


@FILTERS.register("min_amount", scope="entry")
def min_amount(ctx: dict, amount: float, window: int = 20) -> pd.DataFrame:
    """近 window 日日均成交额 ≥ amount（元）。qlib amount 单位为千元。"""
    avg = (ctx["amount"].fillna(0) * 1000).rolling(int(window), min_periods=1).mean()
    return avg >= float(amount)


@FILTERS.register("chase_limit", scope="entry")
def chase_limit(ctx: dict, limit: float, days: int = 5) -> pd.DataFrame:
    """近 N 日累计涨幅 > limit 不准入（不追高）。"""
    close = _base(ctx)
    mom = close / close.shift(int(days)) - 1
    return ~(mom > float(limit)).fillna(False)


@FILTERS.register("falling_knife", scope="entry")
def falling_knife(ctx: dict, limit: float, days: int = 5) -> pd.DataFrame:
    """近 N 日累计跌幅 < limit（负数）不准入（不接飞刀）。"""
    close = _base(ctx)
    mom = close / close.shift(int(days)) - 1
    return ~(mom < float(limit)).fillna(False)


for _n in FILTERS.names():
    assert scope_of(_n) in SCOPES, _n
