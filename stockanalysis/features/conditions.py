"""选股条件库 CONDITIONS：把长桥 / 通达信 / 同花顺的常用选股规则翻译成可按历史时点计算的布尔面板。

每个条件：`(ctx, **参数) -> DataFrame[bool]`（index=交易日, columns=代码，True = 满足），注册时声明：
    source  规则出处（长桥预设策略 / 长桥筛选器 / 通达信 / 同花顺 / 旧研究 / 本项目）
    kind    state（状态，可连续多日满足）或 trigger（触发，当日发生）
    grid    阈值邻域：事件研究会逐个取值检验，结果在邻域内平稳才算可靠

研究（research/event_study.py）逐条检验；买卖点规则（selection/points.py）按名字 + 参数引用同一实现。
MCP 平台规则在本地数据上的近似（数据所限）：
    负债率、净利率：无历史数据，长桥"高质量盈利 / 巴菲特"里的这两项分别省略 / 用毛利率代替；
    主力资金流：无历史大单数据，用 CMF / MFI / OBV / 收盘相对均价近似（features/technical.py）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import Registry
from . import events as ev
from . import technical as tech

CONDITIONS = Registry("condition", skip=1)


def register(name: str, source: str, kind: str = "state", grid: dict | None = None, desc: str = ""):
    return CONDITIONS.register(name, source=source, kind=kind, grid=grid or {}, desc=desc)


def _fund(ctx: dict, name: str) -> pd.DataFrame:
    from .fundamental import panel

    return panel(ctx)[name]


def _pe(ctx):
    ep = _fund(ctx, "ep_ttm")
    return (1.0 / ep).where(ep > 0)          # 亏损 PE 无意义 → NaN


def _pb(ctx):
    bp = _fund(ctx, "bp")
    return (1.0 / bp).where(bp > 0)


def _cap(ctx):
    from .risk import market_cap

    return market_cap(ctx)


def _t(ctx, name):
    return tech.get(ctx, name)


YI = 1e8  # 亿元


# ================================================================== 基本面（长桥预设策略）
@register("graham", "长桥·格雷厄姆选股", grid={"pe": [12, 15, 20], "pb": [1.0, 1.5, 2.0]},
          desc="PE ≤ pe、PB ≤ pb、市值 ≥ 100 亿、营收增长 ≥ 3%")
def graham(ctx, pe: float = 15, pb: float = 1.5, cap: float = 100, rev: float = 3):
    return (_pe(ctx) <= pe) & (_pb(ctx) <= pb) & (_cap(ctx) >= cap * YI) & (_fund(ctx, "revenue_yoy") >= rev)


@register("low_valuation", "长桥·低估值", grid={"pe": [10, 15, 20]},
          desc="PE ≤ pe、PB ≤ 1.5、市值 ≥ 100 亿")
def low_valuation(ctx, pe: float = 15, pb: float = 1.5, cap: float = 100):
    return (_pe(ctx) <= pe) & (_pb(ctx) <= pb) & (_cap(ctx) >= cap * YI)


@register("quality_profit", "长桥·高质量盈利（省略负债率 / 净利率）", grid={"roe": [8, 10, 12, 15]},
          desc="ROE ≥ roe、净利润增长 ≥ 5%、经营现金流 / 净利润 ≥ 0.5")
def quality_profit(ctx, roe: float = 10, np_yoy: float = 5, ocf: float = 0.5):
    return (_fund(ctx, "roe") >= roe) & (_fund(ctx, "netprofit_yoy") >= np_yoy) & (_fund(ctx, "ocf_to_netprofit") >= ocf)


@register("canslim", "长桥·CANSLIM", grid={"np_yoy": [30, 50, 80]},
          desc="营收增长 ≥ 15%、净利润增长 ≥ np_yoy、ROE ≥ 15")
def canslim(ctx, rev: float = 15, np_yoy: float = 50, roe: float = 15):
    return (_fund(ctx, "revenue_yoy") >= rev) & (_fund(ctx, "netprofit_yoy") >= np_yoy) & (_fund(ctx, "roe") >= roe)


@register("high_growth", "长桥·业绩上涨", grid={"growth": [30, 50, 80]},
          desc="营收与净利润增长均 ≥ growth")
def high_growth(ctx, growth: float = 50):
    return (_fund(ctx, "revenue_yoy") >= growth) & (_fund(ctx, "netprofit_yoy") >= growth)


@register("high_dividend", "长桥·高股息", grid={"dy": [0.03, 0.04, 0.05, 0.06]},
          desc="近一年现金分红 / 股价 ≥ dy")
def high_dividend(ctx, dy: float = 0.06):
    return _fund(ctx, "dividend_yield") >= dy


@register("dividend_growth", "长桥·高股息稳健成长", grid={"dy": [0.03, 0.04, 0.06]},
          desc="股息率 ≥ dy、营收增长 ≥ 15%")
def dividend_growth(ctx, dy: float = 0.06, rev: float = 15):
    return (_fund(ctx, "dividend_yield") >= dy) & (_fund(ctx, "revenue_yoy") >= rev)


@register("dividend_steady", "旧研究·股息连续性", grid={"times": [3, 4, 5]},
          desc="近 5 年现金分红次数 ≥ times")
def dividend_steady(ctx, times: int = 4):
    return _fund(ctx, "dividend_continuity") >= times


@register("buffett", "长桥·巴菲特（净利率用毛利率代替）", grid={"roe": [12, 15, 20]},
          desc="ROE ≥ roe、毛利率 ≥ 40%、市值 ≥ 10 亿")
def buffett(ctx, roe: float = 15, gm: float = 40, cap: float = 10):
    return (_fund(ctx, "roe") >= roe) & (_fund(ctx, "gross_margin") >= gm) & (_cap(ctx) >= cap * YI)


@register("small_growth", "长桥·小盘轻资产快速成长（省略负债率）", grid={"np_yoy": [15, 25, 40]},
          desc="市值 20~200 亿、净利润增长 ≥ np_yoy")
def small_growth(ctx, np_yoy: float = 25, cap_lo: float = 20, cap_hi: float = 200):
    cap = _cap(ctx)
    return (cap >= cap_lo * YI) & (cap <= cap_hi * YI) & (_fund(ctx, "netprofit_yoy") >= np_yoy)


def _industry_ep_rank(ctx) -> pd.DataFrame:
    """EP（盈利收益率，亏损为 NaN）在同行业内的分位（越高越便宜）；行业不足 5 只不排名。缓存在 ctx。"""
    cache = ctx.setdefault("_cond", {})
    if "industry_ep_rank" not in cache:
        from .fundamental import industry_map

        ep = _fund(ctx, "ep_ttm").where(_fund(ctx, "ep_ttm") > 0)
        ind = industry_map().reindex(ep.columns)
        out = pd.DataFrame(np.nan, index=ep.index, columns=ep.columns)
        for _, cols in ind.dropna().groupby(ind.dropna()).groups.items():
            cols = list(cols)
            if len(cols) >= 5:
                out[cols] = ep[cols].rank(axis=1, pct=True)
        cache["industry_ep_rank"] = out
    return cache["industry_ep_rank"]


@register("industry_cheap", "本项目·行业内估值", grid={"pct": [0.5, 0.7, 0.8]},
          desc="EP（盈利收益率）在同行业内的分位 ≥ pct（越高越便宜），亏损股不计")
def industry_cheap(ctx, pct: float = 0.7):
    return _industry_ep_rank(ctx) >= pct


@register("industry_expensive", "本项目·行业内估值变贵（卖点候选）", grid={"pct": [0.1, 0.2, 0.3]},
          desc="EP 在同行业内的分位 ≤ pct（越低越贵），或转为亏损")
def industry_expensive(ctx, pct: float = 0.2):
    return (_industry_ep_rank(ctx) <= pct) | (_fund(ctx, "ep_ttm") <= 0)


# ================================================================== 技术面（长桥筛选器 / 通达信）
@register("macd_golden_above0", "长桥筛选器·MACD 零轴上金叉", kind="trigger")
def macd_golden_above0(ctx):
    dif, dea = _t(ctx, "macd_dif"), _t(ctx, "macd_dea")
    return tech.cross_up(dif, dea) & (dif > 0)


@register("macd_up_zero", "长桥筛选器·MACD 上穿零轴", kind="trigger")
def macd_up_zero(ctx):
    return tech.cross_up(_t(ctx, "macd_dif"), 0.0)


@register("kdj_golden_low", "长桥筛选器·KDJ 低位金叉", kind="trigger", grid={"level": [20, 30, 40]})
def kdj_golden_low(ctx, level: float = 30):
    k, d = _t(ctx, "kdj_k"), _t(ctx, "kdj_d")
    return tech.cross_up(k, d) & (d <= level)


@register("rsi_golden_low", "长桥筛选器·RSI 低位金叉", kind="trigger", grid={"level": [30, 40, 50]})
def rsi_golden_low(ctx, level: float = 40):
    fast, slow = _t(ctx, "rsi6"), _t(ctx, "rsi12")
    return tech.cross_up(fast, slow) & (slow <= level)


@register("boll_break_up", "长桥筛选器·BOLL 升穿上轨", kind="trigger")
def boll_break_up(ctx):
    return tech.cross_up(ctx["close"], _t(ctx, "boll_up"))


@register("boll_rise_mid", "长桥筛选器·BOLL 升穿中轨", kind="trigger")
def boll_rise_mid(ctx):
    return tech.cross_up(ctx["close"], _t(ctx, "boll_mid"))


@register("boll_below_low", "长桥筛选器·BOLL 跌穿下轨", kind="trigger")
def boll_below_low(ctx):
    return tech.cross_down(ctx["close"], _t(ctx, "boll_low"))


@register("ma_bull", "通达信·均线多头排列")
def ma_bull(ctx):
    m5, m10, m20, m60 = (_t(ctx, f"ma{n}") for n in (5, 10, 20, 60))
    return (m5 > m10) & (m10 > m20) & (m20 > m60)


@register("above_ma", "本项目·站上均线", grid={"n": [20, 60, 120, 250]})
def above_ma(ctx, n: int = 120):
    return ctx["close"] > _t(ctx, f"ma{int(n)}")


@register("below_ma", "本项目·跌破均线（卖点候选）", grid={"n": [10, 20, 60, 250]})
def below_ma(ctx, n: int = 250):
    return ctx["close"] < _t(ctx, f"ma{int(n)}")


@register("breakout", "通达信·放量突破", kind="trigger", grid={"n": [20, 60], "amt": [1.5, 2.0]},
          desc="收盘创 n 日新高（不含当日）且成交额 ≥ amt 倍 20 日均额")
def breakout(ctx, n: int = 20, amt: float = 1.5):
    prior_high = ctx["high"].rolling(int(n), min_periods=int(n)).max().shift(1)
    return (ctx["close"] > prior_high) & (_t(ctx, "amt_ratio1") >= amt)


@register("pullback_shrink", "旧研究·强势缩量回踩", grid={"amt": [0.5, 0.7, 0.9]},
          desc="收盘 > MA60、近 5 日回调 3%~12%、5 日均额 < amt 倍前 20 日均额")
def pullback_shrink(ctx, amt: float = 0.7, lo: float = -0.12, hi: float = -0.03):
    r5 = _t(ctx, "ret5")
    return (ctx["close"] > _t(ctx, "ma60")) & (r5 >= lo) & (r5 <= hi) & (_t(ctx, "amt_ratio5") < amt)


@register("oversold", "旧研究·超跌反转", grid={"drop": [-0.10, -0.15, -0.20]},
          desc="近 10 日跌幅 ≤ drop")
def oversold(ctx, drop: float = -0.15):
    return _t(ctx, "ret10") <= drop


@register("falling_knife", "旧研究·飞刀（回避）", grid={"drop": [-0.15, -0.20]},
          desc="近 5 日跌幅 ≤ drop")
def falling_knife(ctx, drop: float = -0.15):
    return _t(ctx, "ret5") <= drop


@register("chase_high", "旧研究·追高（回避）", grid={"rise": [0.20, 0.30]},
          desc="近 5 日涨幅 ≥ rise")
def chase_high(ctx, rise: float = 0.20):
    return _t(ctx, "ret5") >= rise


@register("near_rise", "长桥·近期持续上升", desc="当日涨幅 ≤ 5%、近 20 日涨幅 ≥ 20%")
def near_rise(ctx):
    return (_t(ctx, "ret1") <= 0.05) & (_t(ctx, "ret20") >= 0.20)


@register("sharp_drop", "长桥·近期极速急跌", desc="当日跌幅 ≥ 5%、近 20 日跌幅 ≥ 20%")
def sharp_drop(ctx):
    return (_t(ctx, "ret1") <= -0.05) & (_t(ctx, "ret20") <= -0.20)


@register("limit_up", "通达信·涨停", kind="trigger")
def limit_up(ctx):
    from ..selection.filters import limit_hits

    return limit_hits(ctx["close"])[0]


@register("limit_up_streak", "通达信·连板", kind="trigger", grid={"n": [2, 3]})
def limit_up_streak(ctx, n: int = 2):
    from ..selection.filters import limit_hits

    up = limit_hits(ctx["close"])[0].astype(np.float32)
    return up.rolling(int(n), min_periods=int(n)).sum() >= n


@register("low_volatility", "旧研究·低波动", grid={"pct": [0.1, 0.2, 0.3]},
          desc="20 日波动率处于当日横截面最低 pct")
def low_volatility(ctx, pct: float = 0.2):
    return _t(ctx, "vol20").rank(axis=1, pct=True) <= pct


@register("strong_trend", "本项目·趋势稳定", grid={"adx": [20, 25, 30]},
          desc="ADX14 ≥ adx 且收盘 > MA60")
def strong_trend(ctx, adx: float = 25):
    return (_t(ctx, "adx14") >= adx) & (ctx["close"] > _t(ctx, "ma60"))


# ================================================================== 资金面（量价近似 / 两融 / 龙虎榜）
@register("money_inflow", "通达信·主力净流入（CMF 近似）", grid={"cmf": [0.05, 0.1, 0.2]})
def money_inflow(ctx, cmf: float = 0.1):
    return _t(ctx, "cmf20") >= cmf


@register("mfi_oversold", "通达信·资金超卖（MFI）", grid={"level": [10, 20, 30]})
def mfi_oversold(ctx, level: float = 20):
    return _t(ctx, "mfi14") <= level


@register("mfi_overbought", "通达信·资金超买（MFI，回避）", grid={"level": [80, 90]})
def mfi_overbought(ctx, level: float = 80):
    return _t(ctx, "mfi14") >= level


@register("obv_divergence", "本项目·量价背离（OBV 上升而价格下跌）")
def obv_divergence(ctx):
    return (_t(ctx, "obv_slope20") > 0) & (_t(ctx, "ret20") < 0)


@register("close_strong", "本项目·尾盘强势（收盘高于均价）", grid={"gap": [0.005, 0.01, 0.02]})
def close_strong(ctx, gap: float = 0.01):
    return _t(ctx, "close_vwap") >= gap


@register("volume_surge", "通达信·放量", grid={"amt": [2.0, 3.0]}, desc="当日成交额 ≥ amt 倍 20 日均额")
def volume_surge(ctx, amt: float = 2.0):
    return _t(ctx, "amt_ratio1") >= amt


@register("volume_shrink", "旧研究·缩量", grid={"amt": [0.5, 0.6, 0.7]}, desc="5 日均额 < amt 倍前 20 日均额")
def volume_shrink(ctx, amt: float = 0.6):
    return _t(ctx, "amt_ratio5") < amt


@register("margin_buy_surge", "通达信·融资买入放大", grid={"mult": [1.5, 2.0]},
          desc="融资买入占成交额 ≥ mult 倍其 20 日均值（T+1 可见）")
def margin_buy_surge(ctx, mult: float = 1.5):
    r = ev.margin_buy_ratio(ctx)
    return r >= mult * r.rolling(20, min_periods=10).mean().shift(1)


@register("margin_balance_up", "通达信·融资余额增加", grid={"chg": [0.05, 0.10, 0.20]})
def margin_balance_up(ctx, chg: float = 0.10):
    return ev.margin_bal_chg20(ctx) >= chg


@register("lhb_institution_buy", "同花顺·龙虎榜机构净买入", kind="trigger", grid={"ratio": [0.0, 0.02, 0.05]},
          desc="当日龙虎榜机构净买入额 > ratio × 市场成交额")
def lhb_institution_buy(ctx, ratio: float = 0.02):
    return ev.lhb_jg_net(ctx) > ratio


# ================================================================== 筹码 / 预期（事件）
@register("holders_concentrate", "同花顺·股东户数下降（筹码集中）", grid={"chg": [-5, -10, -20]})
def holders_concentrate(ctx, chg: float = -10):
    return ev.gdhs_change(ctx) <= chg


@register("forecast_positive", "同花顺·业绩预增 / 扭亏")
def forecast_positive(ctx):
    return ev.forecast_score(ctx) > 0


@register("forecast_negative", "同花顺·业绩预减 / 亏损（回避）")
def forecast_negative(ctx):
    return ev.forecast_score(ctx) < 0


@register("forecast_big", "同花顺·业绩大幅预增", grid={"chg": [50, 100]})
def forecast_big(ctx, chg: float = 50):
    return ev.forecast_change(ctx) >= chg


@register("repurchase", "同花顺·回购")
def repurchase(ctx):
    return ev.repurchase_recent90(ctx)


@register("holder_increase", "同花顺·股东增持")
def holder_increase(ctx):
    return ev.increase_recent90(ctx)


@register("holder_reduce", "同花顺·股东减持（回避）")
def holder_reduce(ctx):
    return ev.reduce_recent90(ctx)


# ================================================================== 连续因子（RankIC 体检用，越大越好的方向不预设）
CONTINUOUS = {
    # 技术
    "ret5": lambda c: _t(c, "ret5"), "ret20": lambda c: _t(c, "ret20"), "mom_120_20": lambda c: _t(c, "mom_120_20"),
    "vol20": lambda c: _t(c, "vol20"), "natr14": lambda c: _t(c, "natr14"), "rsi14": lambda c: _t(c, "rsi14"),
    "dist_ma20": lambda c: c["close"] / _t(c, "ma20") - 1, "dist_ma60": lambda c: c["close"] / _t(c, "ma60") - 1,
    "dist_high250": lambda c: c["close"] / _t(c, "high250") - 1, "adx14": lambda c: _t(c, "adx14"),
    "limit_up_n20": lambda c: _t(c, "limit_up_n20"),
    # 资金面近似
    "clv": lambda c: _t(c, "clv"), "cmf20": lambda c: _t(c, "cmf20"), "mfi14": lambda c: _t(c, "mfi14"),
    "obv_slope20": lambda c: _t(c, "obv_slope20"), "amt_ratio5": lambda c: _t(c, "amt_ratio5"),
    "close_vwap": lambda c: _t(c, "close_vwap"),
    # 两融 / 龙虎榜 / 筹码 / 预期
    "margin_buy_ratio": ev.margin_buy_ratio, "margin_bal_chg20": ev.margin_bal_chg20,
    "lhb_jg_net_recent5": ev.lhb_jg_net_recent5, "gdhs_change": ev.gdhs_change, "forecast_change": ev.forecast_change,
    # 基本面
    **{f: (lambda f: lambda c: _fund(c, f))(f) for f in ("ep_ttm", "bp", "dividend_yield", "dividend_continuity",
                                                          "roe", "gross_margin", "ocf_to_netprofit",
                                                          "revenue_yoy", "netprofit_yoy")},
    "log_cap": lambda c: np.log(_cap(c)),
}
