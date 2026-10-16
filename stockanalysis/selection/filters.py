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
from ..data.universe import board_of, listing_dates

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


def limit_threshold(close: pd.DataFrame) -> pd.DataFrame:
    """逐日涨跌停幅度：主板 10%、科创板 20%、创业板 2020-08-24 起 20%（之前 10%）、北交所 30%。

    不区分 ST（5%）：历史 ST 判定见 features/risk.py。
    """
    boards = np.array([board_of(c) for c in close.columns])
    row = np.select([boards == "kcb", boards == "bse"], [0.20, 0.30], 0.10).astype(np.float32)
    thr = np.broadcast_to(row, close.shape).copy()
    after = np.asarray(close.index >= CYB_20PCT_FROM)
    thr[np.ix_(after, boards == "cyb")] = 0.20
    return pd.DataFrame(thr, index=close.index, columns=close.columns)


def limit_hits(close: pd.DataFrame, tol: float = 0.005) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(涨停, 跌停) 布尔面板：收盘涨跌幅达到幅度 - tol。"""
    ret = close.pct_change(fill_method=None)
    thr = limit_threshold(close)
    return (ret >= thr - tol).fillna(False), (ret <= -(thr - tol)).fillna(False)


@FILTERS.register("exclude_limit", scope="entry")
def exclude_limit(ctx: dict, tol: float = 0.005) -> pd.DataFrame:
    """当日涨停（收盘涨幅 ≥ 涨停幅度 - tol）不准入：次日开盘前无法确认能否买进。"""
    return ~limit_hits(_base(ctx), tol)[0]


@FILTERS.register("min_amount", scope="entry")
def min_amount(ctx: dict, amount: float, window: int = 20) -> pd.DataFrame:
    """近 window 日日均成交额 ≥ amount（元）。qlib amount 单位为千元。"""
    avg = (ctx["amount"].fillna(0) * 1000).rolling(int(window), min_periods=1).mean()
    return avg >= float(amount)


# ------------------------------------------------------------------ 烂股剔除（exit：已持有的也强制调出）
# 判定逻辑与数据口径见 features/risk.py；数据缺失视为通过（不因缺数据误杀）。
@FILTERS.register("not_st", scope="exit")
def not_st(ctx: dict) -> pd.DataFrame:
    """非 ST / *ST（历史时点：baostock isST，缺数据时按涨跌停幅度近似）。"""
    from ..features import risk

    return ~risk.st(ctx)


@FILTERS.register("min_price", scope="exit")
def min_price(ctx: dict, price: float = 2.0) -> pd.DataFrame:
    """不复权收盘价 ≥ price（面值退市风险）。"""
    from ..features import risk

    raw = risk.raw_close(ctx)
    return ~(raw < float(price))


@FILTERS.register("min_market_cap", scope="exit")
def min_market_cap(ctx: dict, cap: float = 3e9) -> pd.DataFrame:
    """总市值估算 ≥ cap（元）。"""
    from ..features import risk

    return ~(risk.market_cap(ctx) < float(cap))


@FILTERS.register("no_consecutive_loss", scope="exit")
def no_consecutive_loss(ctx: dict, years: int = 2) -> pd.DataFrame:
    """最近 years 个已披露年报不全为亏损。"""
    from ..features import risk

    return ~risk.consecutive_loss(ctx, int(years))


@FILTERS.register("positive_equity", scope="exit")
def positive_equity(ctx: dict) -> pd.DataFrame:
    """最近已披露每股净资产 ≥ 0。"""
    from ..features import risk

    return ~risk.negative_equity(ctx)


@FILTERS.register("revenue_floor", scope="exit")
def revenue_floor(ctx: dict, revenue: float = 3e8) -> pd.DataFrame:
    """非"营收 < revenue 且亏损"（退市新规口径）。"""
    from ..features import risk

    return ~risk.small_revenue_loss(ctx, float(revenue))


@FILTERS.register("no_risk_notice", scope="exit")
def no_risk_notice(ctx: dict, days: int = 365) -> pd.DataFrame:
    """近 days 天无立案调查 / 行政处罚 / 退市风险等公告。"""
    from ..features import risk

    return ~risk.risk_notice(ctx, int(days))


@FILTERS.register("no_release_ahead", scope="entry")
def no_release_ahead(ctx: dict, days: int = 20, ratio: float = 0.05) -> pd.DataFrame:
    """未来约 days 个交易日内限售解禁合计不超过流通市值的 ratio。"""
    from ..features import risk

    return ~risk.release_ahead(ctx, int(days), float(ratio))


for _n in FILTERS.names():
    assert scope_of(_n) in SCOPES, _n
